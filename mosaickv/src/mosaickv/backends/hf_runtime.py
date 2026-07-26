"""Unified eager Hugging Face runtime for FullKV and MosaicKV variants."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np

from mosaickv.adapters.huggingface import (
    AdapterCapabilities,
    CacheLayerSnapshot,
    CacheSnapshot,
    DecodeState,
    HuggingFaceMultimodalAdapter,
    ParityReport,
    PrefillOutput,
    PreparedInputs,
)
from mosaickv.baselines import (
    BaselineCompressionPlan,
    LookMCompressionPlan,
    PrefixKVCompressionPlan,
    PrefixKVOfflineProfile,
    VLCacheCompressionPlan,
    assert_vl_cache_calibration_disjoint,
    build_exact_baseline_plan,
    build_lookm_reimpl_plan,
    build_prefixkv_reimpl_plan,
    build_vl_cache_reimpl_plan,
    load_prefixkv_profile,
    lookm_runtime_payloads,
    prefixkv_runtime_payloads,
    vl_cache_runtime_payloads,
)
from mosaickv.cache_state import (
    FullKVState,
    KVBlockDescriptor,
    MosaicKVState,
    tensor_storage_bytes,
)
from mosaickv.config import RunConfig
from mosaickv.evaluation.forecast_diagnostics import (
    AttentionDiagnosticInput,
    ForecastQualityDiagnostics,
    evaluate_forecast_quality,
)
from mosaickv.evaluation.model import EvaluationRequest, GenerationMetrics, ModelGeneration
from mosaickv.evaluation.oracle_queries import (
    EvaluationOnlyOracleQueries,
    collect_evaluation_only_true_future_queries,
)
from mosaickv.forecasting import QueryForecast, forecast_with_rollout
from mosaickv.forecasting.huggingface import IsolatedQueryRollout
from mosaickv.generation import decode_quality_output, terminal_token_ids
from mosaickv.graph import SparseEvidenceGraph, build_evidence_graph
from mosaickv.measurements.timing import ModuleCudaTimer, SynchronizationAudit
from mosaickv.prototypes import (
    PrototypeRecord,
    ThreeTierCacheConstruction,
    TierConstructionMode,
    construct_three_tier_cache,
)
from mosaickv.repair import RepairCacheState, RepairEvent, repair_decode_step
from mosaickv.selection import (
    BlockUtilityTable,
    BudgetedObjective,
    SelectionBudget,
    SelectionResult,
    compute_block_utilities,
    lazy_greedy_select,
    select_all_exact,
)
from mosaickv.types import (
    BudgetUnit,
    JsonObject,
    JsonValue,
    MosaicKVMethod,
    OutputLengthPolicy,
)


class HuggingFaceRuntimeError(RuntimeError):
    """Raised when a requested HF runtime state cannot be represented safely."""


@dataclass(frozen=True, slots=True)
class RuntimePhaseTimings:
    """Single-run phase timings in seconds."""

    prefill: float
    image_video_encoder: float | None
    projector: float | None
    language_model_prefill: float | None
    forecast: float
    graph: float
    utility_and_selection: float
    tier_construction: float
    cache_packing: float
    decode: float
    per_token_decode: tuple[float, ...]
    repair_transfer: float
    repair_redecode: float
    total: float

    def __post_init__(self) -> None:
        scalar_values = (
            self.prefill,
            self.forecast,
            self.graph,
            self.utility_and_selection,
            self.tier_construction,
            self.cache_packing,
            self.decode,
            self.repair_transfer,
            self.repair_redecode,
            self.total,
        )
        optional_values = (
            self.image_video_encoder,
            self.projector,
            self.language_model_prefill,
        )
        if any(not math.isfinite(value) or value < 0 for value in scalar_values):
            raise ValueError("runtime timings must be finite and nonnegative")
        if any(
            value is not None and (not math.isfinite(value) or value < 0)
            for value in optional_values
        ):
            raise ValueError("optional runtime timings must be finite and nonnegative")
        if any(not math.isfinite(value) or value < 0 for value in self.per_token_decode):
            raise ValueError("per-token runtime timings must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class FidelityDiagnostic:
    """Evaluation-only comparison against an isolated FullKV decode."""

    token_agreement: float
    sequence_agreement: bool
    logit_kl: float
    attention_output_error: float


@dataclass(frozen=True, slots=True)
class MosaicKVCompressionPlan:
    """All inspectable algorithm outputs produced from one full prefill."""

    method: MosaicKVMethod
    effective_method: str
    full_state: FullKVState
    forecast: QueryForecast
    graph: SparseEvidenceGraph
    utilities: BlockUtilityTable
    selection: SelectionResult
    construction: ThreeTierCacheConstruction
    source_budget_value: int
    active_budget_value: int
    graph_seconds: float
    selection_seconds: float
    tier_seconds: float


@dataclass(frozen=True, slots=True)
class PackedRuntimeCache:
    """Uniform HF tensors plus masks for independently selected KV heads."""

    snapshot: CacheSnapshot
    validity_masks: tuple[Any, ...]
    prompt_capacity: int
    all_slots_valid: bool
    prototype_slots: tuple[tuple[int, int, int, int], ...]
    slot_records: tuple[JsonObject, ...]


@dataclass(frozen=True, slots=True)
class DensePhysicalCacheBudget:
    """Representable physical prefill budget for the dense HF cache layout."""

    source_bytes: int
    target_bytes: int
    prompt_capacity: int
    position_limits: Mapping[tuple[int, int], int]

    @property
    def bytes_per_position(self) -> int:
        """Physical bytes for one shared sequence slot across all layer/head tensors."""

        if self.prompt_capacity < 1 or self.target_bytes % self.prompt_capacity:
            raise HuggingFaceRuntimeError("dense target bytes are not position-aligned")
        return self.target_bytes // self.prompt_capacity

    def minimum_block_representable_fill(self, block_size: int) -> int:
        """Lower bound after unavoidable fixed-block rounding in every identity."""

        if block_size < 1:
            raise ValueError("block_size must be positive")
        return max(0, self.target_bytes - (block_size - 1) * self.bytes_per_position)


def _dense_physical_cache_budget(
    full_state: FullKVState,
    config: RunConfig,
) -> DensePhysicalCacheBudget | None:
    """Quantize a byte budget to one shared dense prompt capacity.

    The eager HF cache has one physical sequence dimension shared by all
    layers and KV heads.  A logical byte sum is therefore not a physical
    budget.  A shared per-identity position capacity makes the materialized
    tensor size predictable and auditable before selection, including for
    fixed blocks larger than one token.
    """

    if config.cache.budget_unit is not BudgetUnit.BYTES:
        return None
    if full_state.block_size != config.cache.block_size:
        raise HuggingFaceRuntimeError("runtime and FullKV block sizes disagree")
    lengths = {layer.sequence_length for layer in full_state.layers}
    if len(lengths) != 1:
        raise HuggingFaceRuntimeError(
            "dense HF physical-byte budgeting requires a uniform source sequence length"
        )
    source_length = next(iter(lengths))
    bytes_per_identity: dict[tuple[int, int], int] = {}
    for block in full_state.blocks:
        if block.byte_size % block.position_count:
            raise HuggingFaceRuntimeError("KV block bytes are not divisible by its positions")
        identity = (block.layer, block.kv_head)
        bytes_per_token = block.byte_size // block.position_count
        previous = bytes_per_identity.setdefault(identity, bytes_per_token)
        if previous != bytes_per_token:
            raise HuggingFaceRuntimeError("KV bytes per token vary within one layer/head identity")
    bytes_per_position = sum(bytes_per_identity.values())
    source_bytes = bytes_per_position * source_length
    if source_bytes != full_state.active_bytes:
        raise HuggingFaceRuntimeError(
            "dense HF byte accounting does not reconstruct the FullKV tensor storage"
        )
    requested = min(
        math.ceil(source_bytes * config.cache.retention_ratio),
        config.cache.budget_value,
    )
    prompt_capacity = requested // bytes_per_position
    if config.cache.retention_ratio == 1.0:
        prompt_capacity = source_length
    else:
        # A dense cache is fastest when every layer/head has exactly the same
        # number of real positions. Quantize the target to a block-subset sum
        # so the selector can fill every physical slot without padding masks.
        signatures: dict[tuple[int, int], tuple[tuple[int, bool], ...]] = {}
        for identity in bytes_per_identity:
            signatures[identity] = tuple(
                sorted(
                    (
                        block.position_count,
                        block.mandatory,
                    )
                    for block in full_state.blocks
                    if (block.layer, block.kv_head) == identity
                )
            )
        unique_signatures = set(signatures.values())
        if len(unique_signatures) != 1:
            raise HuggingFaceRuntimeError(
                "dense mask-free packing requires identical block structures per layer/head"
            )
        signature = next(iter(unique_signatures))
        mandatory_positions = sum(size for size, mandatory in signature if mandatory)
        if mandatory_positions > prompt_capacity:
            raise HuggingFaceRuntimeError(
                "physical byte budget cannot represent the mandatory block subset"
            )
        # Selection operates on fixed-size blocks, but modality boundaries can
        # leave smaller fragments. Treating a fragment-dependent sum as a
        # generally fillable dense capacity is unsafe: greedy selection can
        # consume a different fragment combination and become stranded just
        # below that capacity. Quantize the shared physical extent itself to
        # the fixed block lattice. Mandatory-only capacity is the exception
        # when the requested budget is smaller than one aligned extent.
        aligned_capacity = (prompt_capacity // full_state.block_size) * full_state.block_size
        prompt_capacity = max(mandatory_positions, aligned_capacity)
    if prompt_capacity < 1:
        raise HuggingFaceRuntimeError(
            "physical byte budget cannot represent one KV position per layer/head"
        )
    prompt_capacity = min(prompt_capacity, source_length)
    target_bytes = prompt_capacity * bytes_per_position
    limits = {identity: prompt_capacity for identity in bytes_per_identity}
    mandatory_counts: dict[tuple[int, int], int] = {identity: 0 for identity in bytes_per_identity}
    for block in full_state.blocks:
        if block.mandatory:
            mandatory_counts[(block.layer, block.kv_head)] += block.position_count
    violations = {
        identity: count for identity, count in mandatory_counts.items() if count > prompt_capacity
    }
    if violations:
        identity, count = min(violations.items())
        raise HuggingFaceRuntimeError(
            "physical byte budget is below mandatory per-head capacity: "
            f"layer={identity[0]}, kv_head={identity[1]}, "
            f"required={count}, capacity={prompt_capacity}"
        )
    return DensePhysicalCacheBudget(source_bytes, target_bytes, prompt_capacity, limits)


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - optional HF environment
        raise HuggingFaceRuntimeError("the unified HF runtime requires PyTorch") from error
    return torch


def _timed(exemplar: Any, action: Any) -> tuple[Any, float]:
    """Measure a phase with synchronized CUDA events or a host monotonic clock."""

    torch = _torch()
    device = getattr(exemplar, "device", None)
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = action()
        end.record()
        torch.cuda.synchronize(device)
        return result, float(start.elapsed_time(end)) / 1000.0
    started = time.perf_counter()
    result = action()
    return result, time.perf_counter() - started


def _cache_payload_bytes(
    adapter: HuggingFaceMultimodalAdapter,
    cache: Any,
) -> int:
    """Count K/V payload bytes without treating views as new allocations."""

    if isinstance(cache, CacheSnapshot):
        layers = tuple((layer.key, layer.value) for layer in cache.layers)
    else:
        layers, _source_kind = adapter._legacy_layers(cache)
    return sum(tensor_storage_bytes(key) + tensor_storage_bytes(value) for key, value in layers)


def _to_numpy(value: Any) -> np.ndarray[Any, Any]:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    result = np.asarray(value, dtype=np.float64)
    if not bool(np.all(np.isfinite(result))):
        raise HuggingFaceRuntimeError("runtime tensor contains NaN or infinity")
    return result


def _forecast_attention_by_node(
    full_state: FullKVState,
    prefill: PrefillOutput,
    rollout: IsolatedQueryRollout,
    config: RunConfig,
) -> dict[int, float]:
    """Pool actual eager prompt/draft attention into RoPE-aware block evidence."""

    by_head = _forecast_token_attention_by_head(full_state, prefill, rollout, config)
    return {
        node_id: float(
            by_head[(block.layer, block.kv_head)][list(block.physical_cache_indices)].sum()
        )
        for node_id, block in enumerate(full_state.blocks)
    }


def _forecast_token_attention_by_head(
    full_state: FullKVState,
    prefill: PrefillOutput,
    rollout: IsolatedQueryRollout,
    config: RunConfig,
) -> dict[tuple[int, int], np.ndarray[Any, Any]]:
    """Return the exact token-level evidence pooled by the online selector."""

    if not prefill.attention_weights:
        raise HuggingFaceRuntimeError("runtime prefill did not capture eager attention weights")
    prompt_layers = tuple(_to_numpy(value) for value in prefill.attention_weights)
    if len(prompt_layers) != len(full_state.layers):
        raise HuggingFaceRuntimeError("prefill attention layer count does not match the cache")
    draft_steps = tuple(
        tuple(_to_numpy(value) for value in step) for step in rollout.attention_steps
    )
    result: dict[tuple[int, int], np.ndarray[Any, Any]] = {}
    for layer_index, layer in enumerate(full_state.layers):
        prompt = prompt_layers[layer_index]
        if prompt.ndim != 4 or prompt.shape[0] != 1:
            raise HuggingFaceRuntimeError("eager prompt attention must have shape [1,H,Q,K]")
        query_heads = int(prompt.shape[1])
        if query_heads % layer.kv_heads:
            raise HuggingFaceRuntimeError("query heads are not divisible by cache KV heads")
        group = query_heads // layer.kv_heads
        sequence_length = layer.sequence_length
        for kv_head in range(layer.kv_heads):
            query_start = kv_head * group
            query_end = query_start + group
            samples: list[np.ndarray[Any, Any]] = []
            if config.forecasting.prompt_window:
                window = min(config.forecasting.prompt_window, int(prompt.shape[-2]))
                samples.append(
                    prompt[
                        0,
                        query_start:query_end,
                        -window:,
                        :sequence_length,
                    ].mean(axis=(0, 1))
                )
            for step in draft_steps:
                attention = step[layer_index]
                if attention.ndim != 4 or attention.shape[0] != 1 or attention.shape[-2] != 1:
                    raise HuggingFaceRuntimeError("eager draft attention must have shape [1,H,1,K]")
                samples.append(
                    attention[
                        0,
                        query_start:query_end,
                        0,
                        :sequence_length,
                    ].mean(axis=0)
                )
            if not samples:
                raise HuggingFaceRuntimeError("forecast attention has no prompt or draft samples")
            values = np.mean(np.stack(samples, axis=0), axis=0)
            if values.shape != (sequence_length,):
                raise HuggingFaceRuntimeError("forecast token attention length is invalid")
            result[(layer_index, kv_head)] = np.maximum(
                values,
                float(np.finfo(np.float64).tiny),
            )
    return result


def _oracle_token_attention_by_head(
    full_state: FullKVState,
    oracle: EvaluationOnlyOracleQueries,
) -> dict[tuple[int, int], np.ndarray[Any, Any]]:
    if not oracle.attention_steps:
        raise HuggingFaceRuntimeError("oracle diagnostics require captured FullKV attention")
    steps = tuple(tuple(_to_numpy(value) for value in step) for step in oracle.attention_steps)
    result: dict[tuple[int, int], np.ndarray[Any, Any]] = {}
    for layer_index, layer in enumerate(full_state.layers):
        exemplar = steps[0][layer_index]
        if exemplar.ndim != 4 or exemplar.shape[0] != 1 or exemplar.shape[-2] != 1:
            raise HuggingFaceRuntimeError("oracle decode attention must have shape [1,H,1,K]")
        query_heads = int(exemplar.shape[1])
        if query_heads % layer.kv_heads:
            raise HuggingFaceRuntimeError("oracle query heads are not divisible by KV heads")
        group = query_heads // layer.kv_heads
        for kv_head in range(layer.kv_heads):
            start = kv_head * group
            per_step = []
            for step in steps:
                attention = step[layer_index]
                per_step.append(
                    attention[
                        0,
                        start : start + group,
                        0,
                        : layer.sequence_length,
                    ].mean(axis=0)
                )
            result[(layer_index, kv_head)] = np.mean(np.stack(per_step, axis=0), axis=0)
    return result


def _forecast_quality_diagnostic(
    adapter: HuggingFaceMultimodalAdapter,
    prefill: PrefillOutput,
    full_state: FullKVState,
    forecast: QueryForecast,
    rollout: IsolatedQueryRollout,
    selection: SelectionResult,
    config: RunConfig,
) -> ForecastQualityDiagnostics:
    """Evaluation-only oracle audit; its outputs never enter online selection."""

    steps = config.diagnostics.forecast_oracle_steps
    if steps < 1:
        raise ValueError("forecast oracle diagnostics require a positive step count")
    oracle = collect_evaluation_only_true_future_queries(
        adapter,
        prefill,
        future_steps=steps,
        capture_attentions=True,
    )
    predicted = _forecast_token_attention_by_head(full_state, prefill, rollout, config)
    truth = _oracle_token_attention_by_head(full_state, oracle)
    selected = set(selection.selected_node_ids)
    inputs: dict[tuple[int, int], AttentionDiagnosticInput] = {}
    for identity in sorted(predicted):
        node_ids = tuple(
            node_id
            for node_id, block in enumerate(full_state.blocks)
            if (block.layer, block.kv_head) == identity
        )
        inputs[identity] = AttentionDiagnosticInput(
            predicted_attention=predicted[identity],
            true_attention=truth[identity],
            block_positions=tuple(
                full_state.blocks[node_id].physical_cache_indices for node_id in node_ids
            ),
            selected_block_count=sum(node_id in selected for node_id in node_ids),
        )
    return evaluate_forecast_quality(forecast, oracle, inputs)


def _prompt_attention_by_node(
    full_state: FullKVState,
    prefill: PrefillOutput,
    *,
    prompt_window: int,
) -> dict[int, float]:
    """Pool prompt-window attention mass without using draft or forecast queries."""

    if prompt_window < 1:
        raise HuggingFaceRuntimeError("prompt_attention_topk requires prompt_window >= 1")
    if not prefill.attention_weights:
        raise HuggingFaceRuntimeError("prompt_attention_topk did not capture eager attention")
    layers = tuple(_to_numpy(value) for value in prefill.attention_weights)
    if len(layers) != len(full_state.layers):
        raise HuggingFaceRuntimeError("prompt attention layer count does not match the cache")
    result: dict[int, float] = {}
    for node_id, block in enumerate(full_state.blocks):
        layer = full_state.layers[block.layer]
        attention = layers[block.layer]
        if attention.ndim != 4 or attention.shape[0] != 1:
            raise HuggingFaceRuntimeError("prompt attention must have shape [1,H,Q,K]")
        query_heads = int(attention.shape[1])
        if query_heads % layer.kv_heads:
            raise HuggingFaceRuntimeError("query heads are not divisible by cache KV heads")
        group_size = query_heads // layer.kv_heads
        query_start = block.kv_head * group_size
        query_end = query_start + group_size
        window = min(prompt_window, int(attention.shape[-2]))
        selected = attention[
            0,
            query_start:query_end,
            -window:,
            list(block.physical_cache_indices),
        ]
        score = float(selected.sum(axis=-1).mean())
        if not math.isfinite(score) or score < 0:
            raise HuggingFaceRuntimeError(
                "prompt attention block mass must be finite and nonnegative"
            )
        result[node_id] = score
    return result


def _budget_source_cost(full_state: FullKVState, unit: BudgetUnit) -> int:
    if unit is BudgetUnit.BLOCKS:
        return len(full_state.blocks)
    if unit is BudgetUnit.RETAINED_SLOTS:
        return sum(block.position_count for block in full_state.blocks)
    if unit is BudgetUnit.BYTES:
        return full_state.active_bytes
    raise ValueError(f"unsupported runtime budget unit: {unit}")


def _mandatory_cost(full_state: FullKVState, unit: BudgetUnit) -> int:
    budget = SelectionBudget(1, unit)
    return sum(budget.cost(block) for block in full_state.blocks if block.mandatory)


def _resolve_budget(full_state: FullKVState, config: RunConfig) -> tuple[int, int]:
    source = _budget_source_cost(full_state, config.cache.budget_unit)
    requested = math.ceil(source * config.cache.retention_ratio)
    target = min(requested, config.cache.budget_value)
    mandatory = _mandatory_cost(full_state, config.cache.budget_unit)
    if config.cache.retention_ratio == 1.0 and target != source:
        raise HuggingFaceRuntimeError(
            "retention 1.0 requires cache.budget_value to cover the complete FullKV cache"
        )
    if target < mandatory:
        raise HuggingFaceRuntimeError(
            f"retention budget {target} is below mandatory exact cost {mandatory}"
        )
    return source, target


def _selection(
    objective: BudgetedObjective,
    value: int,
    unit: BudgetUnit,
    *,
    retention_one: bool,
    position_limits: Mapping[tuple[int, int], int] | None = None,
    fill_position_limits: bool = False,
) -> SelectionResult:
    if retention_one:
        return select_all_exact(objective, SelectionBudget(value, unit))
    return lazy_greedy_select(
        objective,
        SelectionBudget(value, unit),
        position_limits=position_limits,
        fill_position_limits=fill_position_limits,
    )


def _exact_construction(
    full_state: FullKVState,
    graph: SparseEvidenceGraph,
    selection: SelectionResult,
    capabilities: AdapterCapabilities,
    config: RunConfig,
) -> ThreeTierCacheConstruction:
    return construct_three_tier_cache(
        full_state,
        graph,
        selection,
        capabilities,
        prototype_config=replace(config.prototypes, enabled=False),
        residual_config=replace(config.residual, enabled=False),
        retention_ratio=config.cache.retention_ratio,
    )


def _prototype_construction(
    full_state: FullKVState,
    graph: SparseEvidenceGraph,
    objective: BudgetedObjective,
    capabilities: AdapterCapabilities,
    config: RunConfig,
    target: int,
    position_limits: Mapping[tuple[int, int], int] | None,
) -> tuple[SelectionResult, ThreeTierCacheConstruction, float, float]:
    """Reserve active capacity for prototypes with bounded deterministic search."""

    unit = config.cache.budget_unit
    mandatory = _mandatory_cost(full_state, unit)
    selection_seconds = 0.0
    tier_seconds = 0.0
    if not capabilities.supports_prototype_merge or (
        config.method is MosaicKVMethod.MOSAICKV_FULL and not capabilities.supports_residual_repair
    ):
        started = time.perf_counter()
        selection = _selection(
            objective,
            target,
            unit,
            retention_one=False,
            position_limits=position_limits,
            fill_position_limits=position_limits is not None,
        )
        selection_seconds += time.perf_counter() - started
        started = time.perf_counter()
        construction = construct_three_tier_cache(
            full_state,
            graph,
            selection,
            capabilities,
            prototype_config=config.prototypes,
            residual_config=config.residual,
            retention_ratio=config.cache.retention_ratio,
        )
        tier_seconds += time.perf_counter() - started
        return selection, construction, selection_seconds, tier_seconds

    upper = max(mandatory, target - 1)
    span = upper - mandatory
    if span <= 128:
        candidates = tuple(range(upper, mandatory - 1, -1))
    elif config.method.is_prefixkv_reimplementation:
        candidates = tuple(
            sorted(
                {mandatory + math.floor(span * index / 127) for index in range(128)},
                reverse=True,
            )
        )
    last_selection: SelectionResult | None = None
    for exact_budget in candidates:
        started = time.perf_counter()
        candidate = _selection(
            objective,
            exact_budget,
            unit,
            retention_one=False,
            position_limits=position_limits,
        )
        candidate = replace(candidate, budget=SelectionBudget(target, unit))
        selection_seconds += time.perf_counter() - started
        last_selection = candidate
        started = time.perf_counter()
        construction = construct_three_tier_cache(
            full_state,
            graph,
            candidate,
            capabilities,
            prototype_config=config.prototypes,
            residual_config=(
                config.residual
                if config.method is MosaicKVMethod.MOSAICKV_FULL
                else replace(config.residual, enabled=False)
            ),
            retention_ratio=config.cache.retention_ratio,
        )
        tier_seconds += time.perf_counter() - started
        if construction.mode is TierConstructionMode.THREE_TIER:
            return candidate, construction, selection_seconds, tier_seconds
    if last_selection is None:  # pragma: no cover - target always gives one candidate
        raise HuggingFaceRuntimeError("prototype selection search produced no candidate")
    started = time.perf_counter()
    fallback = _selection(
        objective,
        target,
        unit,
        retention_one=False,
        position_limits=position_limits,
    )
    selection_seconds += time.perf_counter() - started
    started = time.perf_counter()
    construction = _exact_construction(full_state, graph, fallback, capabilities, config)
    tier_seconds += time.perf_counter() - started
    return fallback, construction, selection_seconds, tier_seconds


def build_compression_plan(
    full_state: FullKVState,
    forecast: QueryForecast,
    forecast_attention_by_node: Mapping[int, float],
    capabilities: AdapterCapabilities,
    config: RunConfig,
    *,
    position_limits: Mapping[tuple[int, int], int] | None = None,
) -> MosaicKVCompressionPlan:
    """Run graph, utility, selection, and tier construction for one prefill."""

    if config.method.is_full_cache:
        raise ValueError("FullKV does not run the MosaicKV compression planner")
    graph_started = time.perf_counter()
    graph = build_evidence_graph(full_state, config.graph)
    graph_seconds = time.perf_counter() - graph_started
    selection_started = time.perf_counter()
    utilities = compute_block_utilities(
        graph,
        config.utility,
        forecast_attention_by_node=forecast_attention_by_node,
        attention_provenance="eager_prompt_and_isolated_draft_attention_v1",
        rope_aware=True,
    )
    objective = BudgetedObjective(graph, utilities, config.selection)
    utility_seconds = time.perf_counter() - selection_started
    source_budget, target = _resolve_budget(full_state, config)
    if config.cache.retention_ratio == 1.0:
        selection = _selection(
            objective,
            source_budget,
            config.cache.budget_unit,
            retention_one=True,
        )
        if selection.selected_node_ids != tuple(range(len(full_state.blocks))):
            raise HuggingFaceRuntimeError("retention 1.0 did not select every source block")
        selection_seconds = time.perf_counter() - selection_started
        tier_started = time.perf_counter()
        construction = construct_three_tier_cache(
            full_state,
            graph,
            selection,
            capabilities,
            prototype_config=config.prototypes,
            residual_config=config.residual,
            retention_ratio=1.0,
        )
    elif config.method is MosaicKVMethod.MOSAICKV_EXACT:
        selection = _selection(
            objective,
            target,
            config.cache.budget_unit,
            retention_one=False,
            position_limits=position_limits,
            fill_position_limits=position_limits is not None,
        )
        selection_seconds = time.perf_counter() - selection_started
        tier_started = time.perf_counter()
        construction = _exact_construction(full_state, graph, selection, capabilities, config)
    else:
        selection, construction, selector_seconds, tier_seconds = _prototype_construction(
            full_state,
            graph,
            objective,
            capabilities,
            config,
            target,
            position_limits,
        )
        selection_seconds = utility_seconds + selector_seconds
        tier_started = None
    if tier_started is not None:
        tier_seconds = time.perf_counter() - tier_started
    if construction.mode is TierConstructionMode.THREE_TIER:
        effective_method = config.method.value
    elif construction.mode is TierConstructionMode.RETENTION_ONE:
        effective_method = f"{config.method.value}__retention_one_exact"
    elif config.method is MosaicKVMethod.MOSAICKV_EXACT:
        effective_method = config.method.value
    elif construction.mode is TierConstructionMode.EXACT_ONLY_UNSAFE:
        effective_method = f"{config.method.value}__mosaickv_exact_safety_fallback"
    else:
        effective_method = f"{config.method.value}__{construction.mode.value}"
    return MosaicKVCompressionPlan(
        config.method,
        effective_method,
        full_state,
        forecast,
        graph,
        utilities,
        selection,
        construction,
        source_budget,
        target,
        graph_seconds,
        selection_seconds,
        tier_seconds,
    )


def compare_runtime_retention_one(
    adapter: HuggingFaceMultimodalAdapter,
    prepared: PreparedInputs,
    config: RunConfig,
) -> ParityReport:
    """Compare packed retention-1 decoding with untouched FullKV from one prefill."""

    if config.method.is_full_cache or config.cache.retention_ratio != 1.0:
        raise ValueError("runtime parity requires a compressed method at retention ratio 1.0")
    if not (
        config.method.is_mosaickv
        or config.method.is_compressed_baseline
        or config.method.is_published_reimplementation
    ):
        raise ValueError("runtime parity method is unsupported")
    if config.generation.max_new_tokens < 2:
        raise ValueError("runtime parity requires at least two generated tokens")
    torch = _torch()
    prefill = adapter.prefill(
        prepared,
        capture_queries=config.method.is_mosaickv,
        capture_attentions=(
            config.method.is_mosaickv
            or config.method is MosaicKVMethod.PROMPT_ATTENTION_TOPK
            or config.method.is_published_reimplementation
        ),
        attention_query_window=(
            None
            if config.method.is_published_reimplementation
            else (
                max(1, config.forecasting.prompt_window)
                if config.method.is_mosaickv
                or config.method is MosaicKVMethod.PROMPT_ATTENTION_TOPK
                else None
            )
        ),
    )
    snapshot = adapter.extract_past_key_values(prefill.state.past_key_values)
    full_state = FullKVState.from_cache_snapshot(
        snapshot,
        modality_spans=prepared.modality_map,
        token_ids=prepared.model_inputs["input_ids"],
        block_size=config.cache.block_size,
        original_logical_sequence_length=prefill.state.logical_sequence_length,
        next_decode_position=prefill.state.next_decode_position,
        # The terminal prompt token is the only universally non-compressible
        # decode boundary.  A BOS token is useful but is not inherently
        # mandatory; marking both endpoints can make a valid 50% block budget
        # impossible for short multimodal prompts.
        mandatory_logical_positions=(
            tuple(range(prefill.state.logical_sequence_length))
            if config.method.is_published_reimplementation
            else (prefill.state.logical_sequence_length - 1,)
        ),
    )
    if config.method.is_mosaickv:
        forecast, rollout = forecast_with_rollout(
            adapter,
            prefill,
            config.forecasting,
            capture_attentions=True,
        )
        mosaic_plan = build_compression_plan(
            full_state,
            forecast,
            _forecast_attention_by_node(full_state, prefill, rollout, config),
            adapter.capabilities,
            config,
        )
        packed = pack_runtime_cache(
            adapter,
            full_state,
            mosaic_plan.construction.state,
            mosaic_plan.construction.prototypes,
            (),
        )
    elif config.method.is_compressed_baseline:
        prompt_attention = (
            _prompt_attention_by_node(
                full_state,
                prefill,
                prompt_window=config.forecasting.prompt_window,
            )
            if config.method is MosaicKVMethod.PROMPT_ATTENTION_TOPK
            else None
        )
        baseline_plan = build_exact_baseline_plan(
            full_state,
            config.method,
            config.cache,
            seed=config.execution.seed,
            prompt_attention_by_node=prompt_attention,
            value_chunk_size=config.graph.similarity_chunk_size,
        )
        packed = pack_runtime_cache(adapter, full_state, baseline_plan.state, (), ())
    elif config.method.is_lookm_reimplementation:
        lookm_plan = build_lookm_reimpl_plan(
            full_state,
            prefill.attention_weights,
            config.lookm,
            config.cache,
        )
        payloads, slot_records = lookm_runtime_payloads(lookm_plan)
        packed = pack_runtime_payloads(
            adapter,
            full_state,
            payloads,
            slot_records,
        )
    elif config.method.is_prefixkv_reimplementation:
        prefix_profile = (
            load_prefixkv_profile(
                config.prefixkv.profile_path,
                model_id=config.model.id,
                model_revision=config.model.revision,
                target_retention_ratio=config.cache.retention_ratio,
                start_size=config.prefixkv.start_size,
                protect_size=config.prefixkv.protect_size,
            )
            if config.prefixkv.profile_path is not None
            else None
        )
        prefix_plan = build_prefixkv_reimpl_plan(
            full_state,
            prefill.attention_weights,
            config.prefixkv,
            config.cache,
            model_id=config.model.id,
            model_revision=config.model.revision,
            profile=prefix_profile,
        )
        payloads, slot_records = prefixkv_runtime_payloads(prefix_plan)
        packed = pack_runtime_payloads(adapter, full_state, payloads, slot_records)
    else:
        vl_cache_plan = build_vl_cache_reimpl_plan(
            full_state,
            prefill.attention_weights,
            config.vl_cache,
            config.cache,
        )
        payloads, slot_records = vl_cache_runtime_payloads(vl_cache_plan)
        packed = pack_runtime_payloads(adapter, full_state, payloads, slot_records)
    reference_state = _clone_decode_state(adapter, prefill.state)
    candidate_state = _state_from_packed(adapter, prefill.state, packed)
    reference_token = prefill.next_token_id
    candidate_token = prefill.next_token_id.detach().clone()
    reference_tokens = [reference_token]
    candidate_tokens = [candidate_token]
    reference_logits = [prefill.logits]
    candidate_logits = [prefill.logits.detach().clone()]
    for _step in range(1, config.generation.max_new_tokens):
        reference = adapter.decode_one_token(
            reference_token,
            reference_state,
            capture_queries=False,
        )
        candidate = adapter.decode_one_token(
            candidate_token,
            candidate_state,
            capture_queries=False,
        )
        reference_state = reference.state
        candidate_state = candidate.state
        reference_token = reference.next_token_id
        candidate_token = candidate.next_token_id
        reference_tokens.append(reference_token)
        candidate_tokens.append(candidate_token)
        reference_logits.append(reference.logits)
        candidate_logits.append(candidate.logits)
    reference_ids = torch.cat(reference_tokens, dim=-1).detach().cpu().reshape(-1)
    candidate_ids = torch.cat(candidate_tokens, dim=-1).detach().cpu().reshape(-1)
    agreement = float((reference_ids == candidate_ids).float().mean().item())
    maximum = max(
        float(torch.max(torch.abs(reference.detach().float() - candidate.detach().float())).item())
        for reference, candidate in zip(reference_logits, candidate_logits, strict=True)
    )
    return ParityReport(
        comparison="unified_hf_runtime_retention_1_vs_fullkv",
        generated_tokens=int(reference_ids.numel()),
        token_agreement=agreement,
        maximum_logit_difference=maximum,
        reference_token_ids=tuple(int(value) for value in reference_ids.tolist()),
        candidate_token_ids=tuple(int(value) for value in candidate_ids.tolist()),
    )


def _axis(axis: int, rank: int) -> int:
    return axis if axis >= 0 else rank + axis


def _one_position(tensor: Any, sequence_axis: int, offset: int) -> Any:
    # A length-one basic slice is a view.  The previous index_select launched
    # one CUDA gather for every selected token before the subsequent pack
    # concatenated those tensors again (hundreds of thousands of launches for
    # LLaVA's 32 KV heads).
    index: list[Any] = [slice(None)] * int(tensor.ndim)
    index[_axis(sequence_axis, int(tensor.ndim))] = slice(offset, offset + 1)
    return tensor[tuple(index)]


def _tier_payloads(
    state: MosaicKVState,
    records: Sequence[PrototypeRecord],
    active_prototype_ids: Sequence[int],
) -> tuple[dict[tuple[int, int], list[tuple[int, str, int, Any, Any]]], tuple[JsonObject, ...]]:
    payloads: dict[tuple[int, int], list[tuple[int, str, int, Any, Any]]] = {}
    trace: list[JsonObject] = []
    for block_id, (block, key, value) in enumerate(
        zip(state.exact.blocks, state.exact.key_blocks, state.exact.value_blocks, strict=True)
    ):
        sequence_axis = _axis(-2, int(key.ndim))
        for offset, logical_position in enumerate(block.original_logical_positions):
            payloads.setdefault((block.layer, block.kv_head), []).append(
                (
                    logical_position,
                    "exact",
                    block_id,
                    _one_position(key, sequence_axis, offset),
                    _one_position(value, sequence_axis, offset),
                )
            )
    record_by_id = {record.prototype_id: record for record in records}
    for payload_offset, prototype_id in enumerate(active_prototype_ids):
        record = record_by_id[prototype_id]
        key = state.prototypes.prototype_keys[payload_offset]
        value = state.prototypes.prototype_values[payload_offset]
        payloads.setdefault((record.layer, record.kv_head), []).append(
            (record.anchor_logical_position, "prototype", prototype_id, key, value)
        )
    for identity in payloads:
        payloads[identity].sort(key=lambda item: (item[0], item[1] != "exact", item[2]))
        trace.extend(
            {
                "layer": identity[0],
                "kv_head": identity[1],
                "slot": slot,
                "logical_position": entry[0],
                "tier": entry[1],
                "source_id": entry[2],
            }
            for slot, entry in enumerate(payloads[identity])
        )
    return payloads, tuple(trace)


def pack_runtime_payloads(
    adapter: HuggingFaceMultimodalAdapter,
    full_state: FullKVState,
    payloads: Mapping[
        tuple[int, int],
        Sequence[tuple[int, str, int, Any, Any]],
    ],
    slot_records: tuple[JsonObject, ...],
    *,
    tail_snapshot: CacheSnapshot | None = None,
    old_prompt_capacity: int | None = None,
    physical_prompt_capacity: int | None = None,
) -> PackedRuntimeCache:
    """Pack algorithm-owned head payloads through the common HF cache interface."""

    torch = _torch()
    selected_capacity = max((len(items) for items in payloads.values()), default=0)
    if selected_capacity < 1:
        raise HuggingFaceRuntimeError("packed runtime cache has no active prompt slots")
    prompt_capacity = (
        selected_capacity if physical_prompt_capacity is None else physical_prompt_capacity
    )
    if prompt_capacity < selected_capacity:
        raise HuggingFaceRuntimeError(
            "selected cache exceeds the dense physical prompt capacity: "
            f"selected={selected_capacity}, capacity={prompt_capacity}"
        )
    if tail_snapshot is not None and old_prompt_capacity is None:
        raise ValueError("tail packing requires the previous prompt capacity")
    layers: list[CacheLayerSnapshot] = []
    validity_masks: list[Any] = []
    all_slots_valid = True
    prototype_slots: list[tuple[int, int, int, int]] = []
    finite_flags: list[Any] = []
    for layer_index, source_layer in enumerate(full_state.layers):
        head_keys: list[Any] = []
        head_values: list[Any] = []
        head_validity: list[Any] = []
        key_sequence = source_layer.key_sequence_dimension
        value_sequence = source_layer.value_sequence_dimension
        for kv_head in range(source_layer.kv_heads):
            entries = payloads.get((layer_index, kv_head), [])
            if not entries:
                raise HuggingFaceRuntimeError(
                    f"packed cache removed every slot for layer={layer_index}, head={kv_head}"
                )
            key = torch.cat(tuple(entry[3] for entry in entries), dim=key_sequence)
            value = torch.cat(tuple(entry[4] for entry in entries), dim=value_sequence)
            valid_count = len(entries)
            if valid_count < prompt_capacity:
                all_slots_valid = False
                key_shape = list(key.shape)
                key_shape[key_sequence] = prompt_capacity - valid_count
                value_shape = list(value.shape)
                value_shape[value_sequence] = prompt_capacity - valid_count
                key = torch.cat(
                    (key, torch.zeros(key_shape, dtype=key.dtype, device=key.device)),
                    dim=key_sequence,
                )
                value = torch.cat(
                    (value, torch.zeros(value_shape, dtype=value.dtype, device=value.device)),
                    dim=value_sequence,
                )
            if tail_snapshot is not None:
                tail_layer = tail_snapshot.layers[layer_index]
                tail_key_index: list[Any] = [slice(None)] * int(tail_layer.key.ndim)
                tail_value_index: list[Any] = [slice(None)] * int(tail_layer.value.ndim)
                tail_key_index[source_layer.key_head_dimension] = slice(kv_head, kv_head + 1)
                tail_value_index[source_layer.value_head_dimension] = slice(kv_head, kv_head + 1)
                tail_key_index[key_sequence] = slice(old_prompt_capacity, None)
                tail_value_index[value_sequence] = slice(old_prompt_capacity, None)
                key = torch.cat((key, tail_layer.key[tuple(tail_key_index)]), dim=key_sequence)
                value = torch.cat(
                    (value, tail_layer.value[tuple(tail_value_index)]), dim=value_sequence
                )
            tail_length = int(key.shape[key_sequence]) - prompt_capacity
            head_keys.append(key)
            head_values.append(value)
            head_validity.append(
                torch.tensor(
                    [True] * valid_count
                    + [False] * (prompt_capacity - valid_count)
                    + [True] * tail_length,
                    dtype=torch.bool,
                    device=key.device,
                )
            )
            for slot, entry in enumerate(entries):
                if entry[1] == "prototype":
                    prototype_slots.append((int(entry[2]), layer_index, kv_head, slot))
        packed_key = torch.cat(tuple(head_keys), dim=source_layer.key_head_dimension)
        packed_value = torch.cat(tuple(head_values), dim=source_layer.value_head_dimension)
        finite_flags.extend((torch.isfinite(packed_key).all(), torch.isfinite(packed_value).all()))
        layers.append(CacheLayerSnapshot(packed_key, packed_value, key_sequence))
        validity_masks.append(torch.stack(head_validity, dim=0))
    # Reduce all device-side validation in one host synchronization.  A sync
    # for K and V in every layer serialized dozens of tiny packing kernels.
    if not bool(torch.stack(finite_flags).all().item()):
        raise HuggingFaceRuntimeError("packed cache contains NaN or infinity")
    active_length = int(layers[0].key.shape[full_state.layers[0].key_sequence_dimension])
    if any(
        int(layer.key.shape[full_state.layers[index].key_sequence_dimension]) != active_length
        for index, layer in enumerate(layers)
    ):
        raise HuggingFaceRuntimeError("packed cache layers have inconsistent physical lengths")
    snapshot = CacheSnapshot(
        tuple(layers),
        full_state.source_class,
        full_state.source_kind,
        active_length,
        adapter.capabilities.cached_key_state,
    )
    return PackedRuntimeCache(
        snapshot,
        tuple(validity_masks),
        prompt_capacity,
        all_slots_valid,
        tuple(sorted(prototype_slots)),
        slot_records,
    )


def _gather_dense_heads(
    tensor: Any,
    *,
    head_axis: int,
    sequence_axis: int,
    positions: Sequence[Sequence[int]],
    prompt_capacity: int,
    tail_tensor: Any | None,
    old_prompt_capacity: int | None,
) -> tuple[Any, Any]:
    """Gather different position rows for every KV head in one CUDA operation."""

    torch = _torch()
    rank = int(tensor.ndim)
    normalized_head = _axis(head_axis, rank)
    normalized_sequence = _axis(sequence_axis, rank)
    moved = tensor.movedim((normalized_head, normalized_sequence), (0, 1))
    head_count = int(moved.shape[0])
    if len(positions) != head_count:
        raise HuggingFaceRuntimeError("dense gather position rows do not match KV heads")
    raw_counts = [len(row) for row in positions]
    if any(count < 1 for count in raw_counts):
        raise HuggingFaceRuntimeError("packed cache removed every slot from a KV head")
    if any(count > prompt_capacity for count in raw_counts):
        raise HuggingFaceRuntimeError("selected positions exceed the physical prompt capacity")
    counts = torch.tensor(raw_counts, dtype=torch.long, device=tensor.device)
    index_rows = [tuple(row) + (0,) * (prompt_capacity - len(row)) for row in positions]
    indices = torch.tensor(index_rows, dtype=torch.long, device=tensor.device)
    expanded_indices = indices.reshape(head_count, prompt_capacity, *([1] * (rank - 2))).expand(
        head_count, prompt_capacity, *moved.shape[2:]
    )
    packed = torch.gather(moved, 1, expanded_indices)
    validity = torch.arange(prompt_capacity, device=tensor.device).unsqueeze(0) < counts.unsqueeze(
        1
    )
    expanded_validity = validity.reshape(head_count, prompt_capacity, *([1] * (rank - 2)))
    packed = packed.masked_fill(~expanded_validity, 0)
    if tail_tensor is not None:
        if old_prompt_capacity is None:
            raise HuggingFaceRuntimeError("tail packing requires the old prompt capacity")
        tail = tail_tensor.movedim((normalized_head, normalized_sequence), (0, 1))
        packed = torch.cat((packed, tail[:, old_prompt_capacity:, ...]), dim=1)
        tail_length = int(tail.shape[1]) - old_prompt_capacity
        if tail_length < 0:
            raise HuggingFaceRuntimeError("old prompt capacity exceeds the tail cache")
        if tail_length:
            validity = torch.cat(
                (
                    validity,
                    torch.ones((head_count, tail_length), dtype=torch.bool, device=tensor.device),
                ),
                dim=1,
            )
    return packed.movedim((0, 1), (normalized_head, normalized_sequence)), validity


def _pack_exact_runtime_blocks(
    adapter: HuggingFaceMultimodalAdapter,
    full_state: FullKVState,
    blocks: Sequence[KVBlockDescriptor],
    *,
    tail_snapshot: CacheSnapshot | None,
    old_prompt_capacity: int | None,
    physical_prompt_capacity: int | None,
) -> PackedRuntimeCache:
    """Vectorized materialization for exact-only selectors and baselines."""

    torch = _torch()
    entries: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for block_id, block in enumerate(blocks):
        for logical, physical in zip(
            block.original_logical_positions,
            block.physical_cache_indices,
            strict=True,
        ):
            entries.setdefault((block.layer, block.kv_head), []).append(
                (logical, physical, block_id)
            )
    for values in entries.values():
        values.sort(key=lambda item: (item[0], item[2]))
    selected_capacity = max((len(values) for values in entries.values()), default=0)
    if selected_capacity < 1:
        raise HuggingFaceRuntimeError("packed runtime cache has no active prompt slots")
    prompt_capacity = (
        selected_capacity if physical_prompt_capacity is None else physical_prompt_capacity
    )
    if prompt_capacity < selected_capacity:
        raise HuggingFaceRuntimeError("selected cache exceeds its dense physical prompt capacity")

    layers: list[CacheLayerSnapshot] = []
    validity_masks: list[Any] = []
    finite_flags: list[Any] = []
    slot_records: list[JsonObject] = []
    all_slots_valid = True
    for layer_index, source_layer in enumerate(full_state.layers):
        position_rows: list[tuple[int, ...]] = []
        for kv_head in range(source_layer.kv_heads):
            row = entries.get((layer_index, kv_head), [])
            if not row:
                raise HuggingFaceRuntimeError(
                    f"packed cache removed every slot for layer={layer_index}, head={kv_head}"
                )
            position_rows.append(tuple(item[1] for item in row))
            all_slots_valid = all_slots_valid and len(row) == prompt_capacity
            slot_records.extend(
                {
                    "layer": layer_index,
                    "kv_head": kv_head,
                    "slot": slot,
                    "logical_position": item[0],
                    "tier": "exact",
                    "source_id": item[2],
                }
                for slot, item in enumerate(row)
            )
        tail_layer = None if tail_snapshot is None else tail_snapshot.layers[layer_index]
        packed_key, validity = _gather_dense_heads(
            source_layer.key,
            head_axis=source_layer.key_head_dimension,
            sequence_axis=source_layer.key_sequence_dimension,
            positions=position_rows,
            prompt_capacity=prompt_capacity,
            tail_tensor=None if tail_layer is None else tail_layer.key,
            old_prompt_capacity=old_prompt_capacity,
        )
        packed_value, _value_validity = _gather_dense_heads(
            source_layer.value,
            head_axis=source_layer.value_head_dimension,
            sequence_axis=source_layer.value_sequence_dimension,
            positions=position_rows,
            prompt_capacity=prompt_capacity,
            tail_tensor=None if tail_layer is None else tail_layer.value,
            old_prompt_capacity=old_prompt_capacity,
        )
        finite_flags.extend((torch.isfinite(packed_key).all(), torch.isfinite(packed_value).all()))
        layers.append(
            CacheLayerSnapshot(
                packed_key,
                packed_value,
                source_layer.key_sequence_dimension,
            )
        )
        validity_masks.append(validity)
    if not bool(torch.stack(finite_flags).all().item()):
        raise HuggingFaceRuntimeError("packed cache contains NaN or infinity")
    active_length = int(layers[0].key.shape[full_state.layers[0].key_sequence_dimension])
    if any(
        int(layer.key.shape[full_state.layers[index].key_sequence_dimension]) != active_length
        for index, layer in enumerate(layers)
    ):
        raise HuggingFaceRuntimeError("packed cache layers have inconsistent physical lengths")
    return PackedRuntimeCache(
        CacheSnapshot(
            tuple(layers),
            full_state.source_class,
            full_state.source_kind,
            active_length,
            adapter.capabilities.cached_key_state,
        ),
        tuple(validity_masks),
        prompt_capacity,
        all_slots_valid,
        (),
        tuple(slot_records),
    )


def pack_runtime_cache(
    adapter: HuggingFaceMultimodalAdapter,
    full_state: FullKVState,
    state: MosaicKVState,
    records: Sequence[PrototypeRecord],
    active_prototype_ids: Sequence[int],
    *,
    tail_snapshot: CacheSnapshot | None = None,
    old_prompt_capacity: int | None = None,
    physical_prompt_capacity: int | None = None,
) -> PackedRuntimeCache:
    """Pack MosaicKV/simple-baseline tiers through the common HF interface."""

    key_state = str(getattr(full_state.cached_key_state, "value", full_state.cached_key_state))
    if state.prototypes.prototype_keys and key_state == "pre_rope":
        raise HuggingFaceRuntimeError(
            "HF prototype materialization requires an adapter-specific safe RoPE transform"
        )
    if not state.prototypes.prototype_keys and not active_prototype_ids:
        return _pack_exact_runtime_blocks(
            adapter,
            full_state,
            state.exact.blocks,
            tail_snapshot=tail_snapshot,
            old_prompt_capacity=old_prompt_capacity,
            physical_prompt_capacity=physical_prompt_capacity,
        )
    payloads, slot_records = _tier_payloads(state, records, active_prototype_ids)
    return pack_runtime_payloads(
        adapter,
        full_state,
        payloads,
        slot_records,
        tail_snapshot=tail_snapshot,
        old_prompt_capacity=old_prompt_capacity,
        physical_prompt_capacity=physical_prompt_capacity,
    )


def _state_from_packed(
    adapter: HuggingFaceMultimodalAdapter,
    original: DecodeState,
    packed: PackedRuntimeCache,
) -> DecodeState:
    torch = _torch()
    device = packed.snapshot.layers[0].key.device
    model_state = dict(original.model_state)
    model_state.update(
        {
            "mosaickv_prompt_capacity": packed.prompt_capacity,
            "mosaickv_prototype_slots": packed.prototype_slots,
            "mosaickv_packed_all_valid": packed.all_slots_valid,
            "mosaickv_cache_all_valid": packed.all_slots_valid,
        }
    )
    if not packed.all_slots_valid:
        model_state["mosaickv_validity_masks"] = packed.validity_masks
    return DecodeState(
        past_key_values=adapter.inject_past_key_values(packed.snapshot),
        attention_mask=torch.ones(
            (1, packed.snapshot.active_sequence_length),
            dtype=original.attention_mask.dtype,
            device=device,
        ),
        active_cache_length=packed.snapshot.active_sequence_length,
        logical_sequence_length=original.logical_sequence_length,
        next_decode_position=original.next_decode_position,
        modality_map=original.modality_map,
        model_state=model_state,
    )


def maintain_prefixkv_decode_cache(
    adapter: HuggingFaceMultimodalAdapter,
    state: DecodeState,
    plan: PrefixKVCompressionPlan,
    *,
    step_index: int,
) -> tuple[DecodeState, tuple[JsonObject, ...]]:
    """Apply PrefixKV's at-most-one fixed-offset eviction per layer after a decode."""

    torch = _torch()
    raw_masks = state.model_state.get("mosaickv_validity_masks")
    if not isinstance(raw_masks, tuple) or len(raw_masks) != len(plan.layers):
        raise HuggingFaceRuntimeError("PrefixKV decode validity metadata is incomplete")
    snapshot = adapter.extract_past_key_values(state.past_key_values)
    keep_by_layer: list[tuple[int, ...]] = []
    events: list[JsonObject] = []
    for layer_plan, validity in zip(plan.layers, raw_masks, strict=True):
        if getattr(validity, "ndim", 0) != 2 or getattr(validity, "dtype", None) != torch.bool:
            raise HuggingFaceRuntimeError("PrefixKV validity masks must be rank-two bool tensors")
        if int(validity.shape[0]) != plan.full_state.layers[layer_plan.layer].kv_heads:
            raise HuggingFaceRuntimeError("PrefixKV validity KV-head count changed during decode")
        if not bool(torch.all(validity == validity[0:1])):
            raise HuggingFaceRuntimeError(
                "PrefixKV requires the same selected positions in every KV head"
            )
        valid_physical = tuple(
            int(value)
            for value in torch.nonzero(validity[0], as_tuple=False)
            .reshape(-1)
            .detach()
            .cpu()
            .tolist()
        )
        target = state.logical_sequence_length * layer_plan.retention_ratio
        # Official PrefixKV casts the difference to int32 and removes at most
        # one position per decode call.
        should_evict = math.trunc(len(valid_physical) - target) > 0
        removed: int | None = None
        if should_evict and len(valid_physical) > 1:
            offset = min(layer_plan.eviction_offset, len(valid_physical) - 1)
            removed = valid_physical[offset]
            valid_physical = tuple(
                position for index, position in enumerate(valid_physical) if index != offset
            )
        keep_by_layer.append(valid_physical)
        events.append(
            {
                "step": step_index,
                "layer": layer_plan.layer,
                "logical_sequence_length": state.logical_sequence_length,
                "target_active_positions": target,
                "eviction_offset": layer_plan.eviction_offset,
                "removed_physical_position": removed,
                "active_positions_after": len(valid_physical),
            }
        )
    if not any(event["removed_physical_position"] is not None for event in events):
        return state, tuple(events)

    capacity = max(len(positions) for positions in keep_by_layer)
    layers: list[CacheLayerSnapshot] = []
    masks: list[Any] = []
    for layer_index, (source, positions) in enumerate(
        zip(snapshot.layers, keep_by_layer, strict=True)
    ):
        storage = plan.full_state.layers[layer_index]
        key_index = torch.tensor(positions, dtype=torch.long, device=source.key.device)
        value_index = key_index.to(device=source.value.device)
        key = torch.index_select(source.key, storage.key_sequence_dimension, key_index)
        value = torch.index_select(source.value, storage.value_sequence_dimension, value_index)
        padding = capacity - len(positions)
        if padding:
            key_shape = list(key.shape)
            value_shape = list(value.shape)
            key_shape[storage.key_sequence_dimension] = padding
            value_shape[storage.value_sequence_dimension] = padding
            key = torch.cat(
                (key, torch.zeros(key_shape, dtype=key.dtype, device=key.device)),
                dim=storage.key_sequence_dimension,
            )
            value = torch.cat(
                (value, torch.zeros(value_shape, dtype=value.dtype, device=value.device)),
                dim=storage.value_sequence_dimension,
            )
        layers.append(CacheLayerSnapshot(key, value, storage.key_sequence_dimension))
        masks.append(
            torch.tensor(
                [True] * len(positions) + [False] * padding,
                dtype=torch.bool,
                device=key.device,
            )
            .unsqueeze(0)
            .expand(storage.kv_heads, -1)
            .clone()
        )
    compacted = CacheSnapshot(
        tuple(layers),
        snapshot.source_class,
        snapshot.source_kind,
        capacity,
        snapshot.cached_key_state,
    )
    model_state = dict(state.model_state)
    model_state["mosaickv_validity_masks"] = tuple(masks)
    model_state["mosaickv_prompt_capacity"] = capacity
    return (
        DecodeState(
            past_key_values=adapter.inject_past_key_values(compacted),
            attention_mask=torch.ones(
                (1, capacity),
                dtype=state.attention_mask.dtype,
                device=state.attention_mask.device,
            ),
            active_cache_length=capacity,
            logical_sequence_length=state.logical_sequence_length,
            next_decode_position=state.next_decode_position,
            modality_map=state.modality_map,
            model_state=model_state,
        ),
        tuple(events),
    )


def _clone_decode_state(adapter: HuggingFaceMultimodalAdapter, state: DecodeState) -> DecodeState:
    snapshot = adapter.extract_past_key_values(state.past_key_values)
    cloned_model_state: dict[str, Any] = {}
    for key, value in state.model_state.items():
        if isinstance(value, tuple):
            cloned_model_state[key] = tuple(
                item.detach().clone() if hasattr(item, "detach") else item for item in value
            )
        elif hasattr(value, "detach"):
            cloned_model_state[key] = value.detach().clone()
        else:
            cloned_model_state[key] = value
    return DecodeState(
        adapter.inject_past_key_values(snapshot),
        state.attention_mask.detach().clone(),
        state.active_cache_length,
        state.logical_sequence_length,
        state.next_decode_position,
        state.modality_map,
        cloned_model_state,
    )


def _attention_outputs(
    adapter: HuggingFaceMultimodalAdapter,
    attention_weights: Sequence[Any],
    state: DecodeState,
) -> tuple[Any, ...]:
    """Reconstruct pre-output-projection attention vectors from measured weights and V."""

    torch = _torch()
    live_cache = adapter.extract_past_key_values(state.past_key_values, clone=False)
    if isinstance(live_cache, CacheSnapshot):
        layers = tuple((layer.value, layer.sequence_dimension) for layer in live_cache.layers)
    else:
        legacy, _source_kind = adapter._legacy_layers(live_cache)
        layers = tuple((value, adapter._cache_sequence_dimension(value)) for _key, value in legacy)
    if len(attention_weights) != len(layers):
        raise HuggingFaceRuntimeError("attention/cache layer counts differ in fidelity diagnostic")
    outputs: list[Any] = []
    for layer_index, (attention, (value, sequence_dimension)) in enumerate(
        zip(attention_weights, layers, strict=True)
    ):
        if attention is None:
            raise HuggingFaceRuntimeError(
                f"layer {layer_index} did not expose eager attention weights"
            )
        probabilities = attention.detach().float()
        values = value.detach().float().movedim(sequence_dimension, -2)
        if probabilities.ndim != 4 or values.ndim != 4:
            raise HuggingFaceRuntimeError("fidelity diagnostic requires rank-four attention and V")
        query_heads = int(probabilities.shape[1])
        kv_heads = int(values.shape[-3])
        if kv_heads < 1 or query_heads % kv_heads:
            raise HuggingFaceRuntimeError("query/KV heads are incompatible in fidelity diagnostic")
        sequence = min(int(probabilities.shape[-1]), int(values.shape[-2]))
        if sequence < 1:
            raise HuggingFaceRuntimeError("fidelity diagnostic received an empty cache")
        expanded_values = values[..., -sequence:, :].repeat_interleave(
            query_heads // kv_heads,
            dim=1,
        )
        outputs.append(
            torch.matmul(probabilities[..., -1:, -sequence:], expanded_values).squeeze(-2)
        )
    return tuple(outputs)


def _attention_output_error(reference: Sequence[Any], candidate: Sequence[Any]) -> float:
    torch = _torch()
    if len(reference) != len(candidate) or not reference:
        raise HuggingFaceRuntimeError("attention-output diagnostic layer counts differ")
    numerator = torch.zeros((), dtype=torch.float64, device=reference[0].device)
    denominator = torch.zeros((), dtype=torch.float64, device=reference[0].device)
    for left, right in zip(reference, candidate, strict=True):
        if left.shape != right.shape:
            raise HuggingFaceRuntimeError("attention-output diagnostic shapes differ")
        delta = left.double() - right.double()
        numerator = numerator + torch.sum(delta * delta)
        denominator = denominator + torch.sum(left.double() * left.double())
    return float(torch.sqrt(numerator / denominator.clamp_min(1e-24)).item())


def _logit_kl(reference_logits: Any, candidate_logits: Any) -> float:
    torch = _torch()
    reference = reference_logits.detach().float()
    candidate = candidate_logits.detach().float()
    if reference.shape != candidate.shape:
        raise HuggingFaceRuntimeError("logit shapes differ in fidelity diagnostic")
    reference_log_probability = torch.log_softmax(reference, dim=-1)
    candidate_log_probability = torch.log_softmax(candidate, dim=-1)
    divergence = torch.sum(
        reference_log_probability.exp() * (reference_log_probability - candidate_log_probability),
        dim=-1,
    ).mean()
    return max(0.0, float(divergence.item()))


def _run_fidelity_diagnostic(
    adapter: HuggingFaceMultimodalAdapter,
    prefill: PrefillOutput,
    candidate_state: DecodeState,
    *,
    max_new_tokens: int,
) -> FidelityDiagnostic:
    """Decode isolated greedy FullKV/candidate trajectories without a second prefill."""

    torch = _torch()
    reference_state = _clone_decode_state(adapter, prefill.state)
    compressed_state = _clone_decode_state(adapter, candidate_state)
    reference_token = prefill.next_token_id.detach().clone()
    candidate_token = prefill.next_token_id.detach().clone()
    reference_tokens = [reference_token]
    candidate_tokens = [candidate_token]
    divergences: list[float] = []
    attention_errors: list[float] = []
    for _step in range(1, max_new_tokens):
        reference_output = adapter.decode_one_token(
            reference_token,
            reference_state,
            capture_queries=False,
            capture_attentions=True,
        )
        candidate_output = adapter.decode_one_token(
            candidate_token,
            compressed_state,
            capture_queries=False,
            capture_attentions=True,
        )
        divergences.append(_logit_kl(reference_output.logits, candidate_output.logits))
        attention_errors.append(
            _attention_output_error(
                _attention_outputs(
                    adapter,
                    reference_output.attention_weights,
                    reference_output.state,
                ),
                _attention_outputs(
                    adapter,
                    candidate_output.attention_weights,
                    candidate_output.state,
                ),
            )
        )
        reference_state = reference_output.state
        compressed_state = candidate_output.state
        reference_token = reference_output.next_token_id
        candidate_token = candidate_output.next_token_id
        reference_tokens.append(reference_token)
        candidate_tokens.append(candidate_token)
    reference_ids = torch.cat(reference_tokens, dim=-1).detach().reshape(-1)
    candidate_ids = torch.cat(candidate_tokens, dim=-1).detach().reshape(-1)
    agreement = float((reference_ids == candidate_ids).float().mean().item())
    return FidelityDiagnostic(
        token_agreement=agreement,
        sequence_agreement=bool(torch.equal(reference_ids, candidate_ids)),
        logit_kl=float(np.mean(divergences)) if divergences else 0.0,
        attention_output_error=float(np.mean(attention_errors)) if attention_errors else 0.0,
    )


def _prototype_attention_mass(
    attention_weights: Sequence[Any],
    prototype_slots: Sequence[tuple[int, int, int, int]],
    validity_masks: Sequence[Any],
) -> dict[int, float]:
    arrays = tuple(_to_numpy(value) for value in attention_weights)
    total_query_heads = sum(int(array.shape[1]) for array in arrays)
    if total_query_heads <= 0:
        raise HuggingFaceRuntimeError("decode attention exposes no query heads")
    masses: dict[int, float] = {}
    for prototype_id, layer, kv_head, slot in prototype_slots:
        attention = arrays[layer]
        query_heads = int(attention.shape[1])
        kv_heads = int(validity_masks[layer].shape[0])
        if query_heads % kv_heads:
            raise HuggingFaceRuntimeError("prototype attention has incompatible head groups")
        group = query_heads // kv_heads
        start = kv_head * group
        value = float(attention[0, start : start + group, :, slot].sum())
        masses[prototype_id] = value / (total_query_heads * int(attention.shape[-2]))
    total = sum(masses.values())
    if total > 1 + 1e-7:
        raise HuggingFaceRuntimeError("normalized prototype attention mass exceeds one")
    return masses


def _decode_text(processor: Any, token_ids: Any) -> str:
    decoder = getattr(processor, "batch_decode", None)
    if decoder is None:
        decoder = getattr(getattr(processor, "tokenizer", processor), "batch_decode", None)
    if decoder is None:
        raise HuggingFaceRuntimeError("processor/tokenizer does not provide batch_decode")
    answers = decoder(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    answer = str(answers[0]) if answers else ""
    if not answer.strip():
        fallback = decoder(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        answer = str(fallback[0]) if fallback else ""
    if not answer.strip():
        raise HuggingFaceRuntimeError("decoded output is empty")
    return answer


def _trace_catalog(
    records: Iterable[Mapping[str, object]],
    *,
    compact: bool,
) -> JsonValue:
    """Serialize a catalog or compute its identical compact digest in one pass."""

    if not compact:
        return [cast("JsonObject", dict(record)) for record in records]
    digest = hashlib.sha256()
    count = 0
    for record in records:
        payload = cast("JsonObject", dict(record))
        digest.update(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        )
        digest.update(b"\n")
        count += 1
    if count == 0:
        return []
    return {
        "storage": "omitted_non_diagnostic",
        "record_count": count,
        "sha256": digest.hexdigest(),
    }


def _forecast_trace(forecast: QueryForecast, *, compact_catalogs: bool = False) -> JsonObject:
    def head_records() -> Iterable[JsonObject]:
        for layer in forecast.layers:
            for head in layer:
                centroids = _to_numpy(head.normalized_centroids)
                weights = _to_numpy(head.forecast_weights)
                yield {
                    "layer": head.provenance.layer,
                    "kv_head": head.provenance.kv_head,
                    "prompt_sample_count": head.provenance.prompt_sample_count,
                    "draft_sample_count": head.provenance.draft_sample_count,
                    "centroid_shape": list(centroids.shape),
                    "centroid_norms": np.linalg.norm(centroids, axis=-1).tolist(),
                    "forecast_weights": weights.tolist(),
                }

    return cast(
        "JsonObject",
        {
            "provenance": asdict(forecast.provenance),
            "timing": asdict(forecast.timing),
            "heads": _trace_catalog(head_records(), compact=compact_catalogs),
        },
    )


def _utility_trace(plan: MosaicKVCompressionPlan) -> JsonObject:
    """Summarize commensurate utility signals without emitting another node catalog."""

    selected = set(plan.selection.selected_node_ids)

    def summary(values: Sequence[float]) -> JsonObject:
        array = np.asarray(values, dtype=np.float64)
        return {
            "minimum": float(array.min()),
            "mean": float(array.mean()),
            "maximum": float(array.max()),
        }

    blocks = plan.utilities.blocks
    local = tuple(block.local_utility for block in blocks)
    return {
        "normalization": plan.utilities.weights.normalization,
        "weights": cast("JsonObject", asdict(plan.utilities.weights)),
        "attention_probability": summary(
            tuple(block.forecast_attention_probability for block in blocks)
        ),
        "value_contribution": summary(tuple(block.value_contribution for block in blocks)),
        "uniqueness": summary(tuple(block.uniqueness for block in blocks)),
        "local_utility": summary(local),
        "negative_local_utility_fraction": sum(value < 0 for value in local) / len(local),
        "selected_mean_local_utility": float(
            np.mean([blocks[node_id].local_utility for node_id in sorted(selected)])
        ),
        "unselected_mean_local_utility": (
            float(
                np.mean([block.local_utility for block in blocks if block.node_id not in selected])
            )
            if len(selected) < len(blocks)
            else None
        ),
    }


def _plan_trace(
    plan: MosaicKVCompressionPlan,
    *,
    compact_catalogs: bool = False,
) -> JsonObject:
    selection = (
        {
            "node_id": decision.node_id,
            "selected": decision.selected,
            "selection_rank": decision.selection_rank,
            "marginal_gain": decision.marginal_gain,
            "reason": decision.reason.value,
        }
        for decision in plan.selection.decisions
    )
    prototypes = (
        {
            "prototype_id": record.prototype_id,
            "layer": record.layer,
            "kv_head": record.kv_head,
            "anchor_node_id": record.anchor_node_id,
            "assigned_node_ids": list(record.assigned_node_ids),
            "weights": [member.normalized_weight for member in record.members],
            "dispersion": record.diagnostics.dispersion,
            "position_span": record.diagnostics.position_span,
        }
        for record in plan.construction.prototypes
    )
    edges = (
        {
            "source": source,
            "target": target,
            "weight": weight,
            "type": edge_type.value,
        }
        for source, target, weight, edge_type in zip(
            plan.graph.row_indices,
            plan.graph.column_indices,
            plan.graph.weights,
            plan.graph.edge_types,
            strict=True,
        )
    )
    return cast(
        "JsonObject",
        {
            "requested_method": plan.method.value,
            "effective_method": plan.effective_method,
            "tier_mode": plan.construction.mode.value,
            "tier_reason": plan.construction.reason,
            "source_budget_value": plan.source_budget_value,
            "active_budget_value": plan.active_budget_value,
            "selected_blocks": _trace_catalog(selection, compact=compact_catalogs),
            "prototypes": _trace_catalog(prototypes, compact=compact_catalogs),
            "graph_edges": _trace_catalog(edges, compact=compact_catalogs),
            "graph_diagnostics": asdict(plan.graph.diagnostics),
            "forecast_statistics": _forecast_trace(
                plan.forecast,
                compact_catalogs=compact_catalogs,
            ),
            "utility_diagnostics": _utility_trace(plan),
        },
    )


def _baseline_plan_trace(
    plan: BaselineCompressionPlan,
    *,
    compact_catalogs: bool = False,
) -> JsonObject:
    selection = (
        {
            "node_id": decision.node_id,
            "selected": decision.selected,
            "selection_rank": decision.selection_rank,
            "score": decision.score,
            "cost": decision.cost,
            "reason": decision.reason.value,
            "stratum": [
                decision.stratum[0],
                decision.stratum[1],
                decision.stratum[2].value,
            ],
        }
        for decision in plan.selection.decisions
    )
    return cast(
        "JsonObject",
        {
            "requested_method": plan.method.value,
            "effective_method": (
                f"{plan.method.value}__retention_one_exact"
                if plan.state.is_retention_one
                else plan.method.value
            ),
            "tier_mode": "retention_one" if plan.state.is_retention_one else "exact_only_baseline",
            "tier_reason": "simple baselines retain exact source blocks only",
            "source_budget_value": plan.source_budget_value,
            "active_budget_value": plan.active_budget_value,
            "selected_source_bytes": plan.selection.active_bytes,
            "budget_spent": plan.selection.budget_spent,
            "budget_unit": plan.selection.budget.unit.value,
            "baseline_seed": plan.selection.seed,
            "selection_provenance": plan.selection.score_provenance,
            "selected_blocks": _trace_catalog(selection, compact=compact_catalogs),
            "prototypes": [],
            "graph_edges": [],
            "graph_diagnostics": {"status": "not_used_by_simple_baseline"},
            "forecast_statistics": {
                "status": "not_used_by_simple_baseline",
                "score_provenance": plan.selection.score_provenance,
            },
        },
    )


def _lookm_plan_trace(plan: LookMCompressionPlan) -> JsonObject:
    """Label local paper equations distinctly from the pinned official code."""

    payload = plan.trace()
    payload.update(
        {
            "requested_method": MosaicKVMethod.LOOKM_REIMPL.value,
            "effective_method": MosaicKVMethod.LOOKM_REIMPL.value,
            "tier_mode": "lookm_paper_merge",
            "tier_reason": (
                "local paper-faithful LOOK-M equations; this is not official LOOK-M code"
            ),
            "source_budget_value": plan.source_blocks,
            "active_budget_value": plan.active_slots,
            "selected_blocks": [
                {
                    "layer": head.layer,
                    "kv_head": head.kv_head,
                    "selected_physical_positions": list(head.selected_physical_positions),
                    "important_physical_positions": list(head.important_physical_positions),
                    "recent_physical_positions": list(head.recent_physical_positions),
                }
                for head in plan.heads
            ],
            "prototypes": [],
            "graph_edges": [],
            "graph_diagnostics": {"status": "not_used_by_lookm_reimpl"},
            "forecast_statistics": {
                "status": "not_used_by_lookm_reimpl",
                "score_provenance": "paper_cumulative_prompt_attention_with_text_prior",
            },
            "selection_provenance": ("Wan_et_al_2024_equations_4_to_8_local_reimplementation"),
        }
    )
    return payload


def _prefixkv_plan_trace(plan: PrefixKVCompressionPlan) -> JsonObject:
    """Normalize PrefixKV paper provenance into the shared trace vocabulary."""

    return {
        "requested_method": MosaicKVMethod.PREFIXKV_REIMPL.value,
        "effective_method": plan.implementation_label,
        "tier_mode": "prefixkv_layerwise_exact_selection",
        "tier_reason": "paper layer profile and prompt-attention prefix selection",
        "selected_blocks": [
            {
                "layer": layer.layer,
                "retained_positions": layer.retained_positions,
                "positions": list(layer.selected_physical_positions),
            }
            for layer in plan.layers
        ],
        "prototypes": [],
        "graph_edges": [],
        "graph_diagnostics": {"status": "not_used_by_prefixkv_reimpl"},
        "forecast_statistics": {"status": "not_used_by_prefixkv_reimpl"},
        "selection_provenance": "prefixkv_reimpl_paper_attention_and_layer_profile",
        "prefixkv": plan.trace(),
    }


def _vl_cache_plan_trace(plan: VLCacheCompressionPlan) -> JsonObject:
    """Normalize ICLR VL-Cache paper provenance into the shared trace vocabulary."""

    return {
        "requested_method": MosaicKVMethod.VL_CACHE_REIMPL.value,
        "effective_method": MosaicKVMethod.VL_CACHE_REIMPL.value,
        "tier_mode": "vl_cache_layer_adaptive_exact_selection",
        "tier_reason": "paper sparsity allocation and accumulated post-vision attention Top-K",
        "selected_blocks": [
            {
                "layer": head.layer,
                "kv_head": head.kv_head,
                "positions": list(head.selected_physical_positions),
            }
            for layer in plan.layers
            for head in layer.heads
        ],
        "prototypes": [],
        "graph_edges": [],
        "graph_diagnostics": {"status": "not_used_by_vl_cache_reimpl"},
        "forecast_statistics": {"status": "not_used_by_vl_cache_reimpl"},
        "selection_provenance": "vl_cache_reimpl_Tu_et_al_ICLR_2025",
        "vl_cache": plan.trace(),
    }


def _write_trace(path: Path, payload: JsonObject) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite runtime trace: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary = handle.name
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
        raise


_COMPACT_TRACE_LIST_KEYS = frozenset(
    {
        "graph_edges",
        "heads",
        "layers",
        "packed_slots",
        "prototypes",
        "selected_blocks",
    }
)


def _compact_trace_records(payload: JsonObject) -> None:
    """Replace large debug catalogs with deterministic audit summaries.

    Main-quality rows outside the fixed diagnostic subset retain hashes and
    counts rather than materializing petabytes of repeated graph/slot records.
    Diagnostic rows keep the complete debug payload.
    """

    for key, value in tuple(payload.items()):
        if key in _COMPACT_TRACE_LIST_KEYS and isinstance(value, list) and value:
            digest = hashlib.sha256()
            for record in value:
                digest.update(
                    json.dumps(
                        record,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    ).encode("utf-8")
                )
                digest.update(b"\n")
            payload[key] = {
                "storage": "omitted_non_diagnostic",
                "record_count": len(value),
                "sha256": digest.hexdigest(),
            }
        elif isinstance(value, dict) and key in {
            "forecast_statistics",
            "prefixkv",
            "vl_cache",
        }:
            _compact_trace_records(value)


class HuggingFaceMosaicKVModel:
    """Local-evaluation model integrating all MosaicKV stages around one HF adapter."""

    backend = "huggingface"

    def __init__(
        self,
        adapter: HuggingFaceMultimodalAdapter,
        config: RunConfig,
        *,
        trace_directory: str | Path,
    ) -> None:
        if config.execution.backend.value != "huggingface":
            raise ValueError("HuggingFaceMosaicKVModel requires execution.backend='huggingface'")
        if config.execution.attention_implementation != "eager" and not config.method.is_full_cache:
            raise ValueError(
                "compressed HF methods currently require eager attention; "
                "SDPA and FlashAttention-2 are correctness-gated for FullKV"
            )
        if config.generation.do_sample or config.generation.temperature != 0:
            raise ValueError("the unified HF runtime requires deterministic greedy decoding")
        if config.generation.output_length_policy is not OutputLengthPolicy.FIXED_MAX_NEW_TOKENS:
            raise ValueError(
                "the unified HF runtime currently supports only fixed_max_new_tokens output"
            )
        self.adapter = adapter
        self.config = config
        self.trace_directory = Path(trace_directory)
        self.model_id = config.model.id
        self.method = config.method.value
        self.retention_ratio = config.cache.retention_ratio
        self._terminal_token_ids = terminal_token_ids(adapter.model, adapter.processor)
        self.supports_video = adapter.capabilities.video
        self._prefixkv_profile: PrefixKVOfflineProfile | None = None
        if config.method.is_prefixkv_reimplementation and config.prefixkv.profile_path is not None:
            self._prefixkv_profile = load_prefixkv_profile(
                config.prefixkv.profile_path,
                model_id=config.model.id,
                model_revision=config.model.revision,
                target_retention_ratio=config.cache.retention_ratio,
                start_size=config.prefixkv.start_size,
                protect_size=config.prefixkv.protect_size,
            )
        if config.method.is_vl_cache_reimplementation:
            assert_vl_cache_calibration_disjoint(
                config.vl_cache.calibration_sample_ids,
                (),
            )

    def _trace_path(self, request: EvaluationRequest) -> Path:
        identity = hashlib.sha256(f"{request.run_id}\0{request.sample_id}".encode()).hexdigest()[
            :16
        ]
        base = self.trace_directory / request.run_id / f"{identity}.json"
        if not base.exists():
            return base
        # A process may be interrupted after its trace is committed but before
        # lmms-eval appends the scored JSONL row. Preserve that immutable trace
        # and allocate an auditable retry path instead of failing resume or
        # overwriting measured evidence.
        attempt = 1
        while True:
            candidate = base.with_name(f"{identity}-attempt-{attempt:03d}.json")
            if not candidate.exists():
                return candidate
            attempt += 1

    def generate(self, request: EvaluationRequest) -> ModelGeneration:
        """Run one explicit prefill/compress/decode request and always write a trace."""

        torch = _torch()
        trace_path = self._trace_path(request)
        decode_runtime_policy: JsonObject = {
            "cache_clone": "repair_provisional_steps_only",
            "attention_mask_growth": "amortized_device_buffer",
            "model_attention_mask": "omitted_when_all_physical_slots_are_valid",
            "cache_position": "amortized_logical_position_buffer",
            "finite_logit_validation": "single_deferred_sequence_reduction",
        }
        trace: JsonObject = {
            "schema_version": 1,
            "run_id": request.run_id,
            "sample_id": request.sample_id,
            "model": self.model_id,
            "backend": self.backend,
            "method": self.method,
            "retention_ratio": self.retention_ratio,
            "status": "started",
            "repair_events": [],
            "decode_runtime_policy": decode_runtime_policy,
        }
        started = time.perf_counter()
        trace_instrumentation_seconds = 0.0
        try:
            if self._prefixkv_profile is not None:
                self._prefixkv_profile.assert_evaluation_disjoint((request.sample_id,))
            if self.config.method.is_vl_cache_reimplementation:
                assert_vl_cache_calibration_disjoint(
                    self.config.vl_cache.calibration_sample_ids,
                    (request.sample_id,),
                )
            torch.manual_seed(self.config.execution.seed)
            if torch.cuda.is_available() and getattr(self.adapter.device, "type", "cpu") == "cuda":
                torch.cuda.manual_seed_all(self.config.execution.seed)
                torch.cuda.synchronize(self.adapter.device)
                torch.cuda.reset_peak_memory_stats(self.adapter.device)
            method = self.config.method
            self.adapter.set_attention_implementation(
                self.config.execution.attention_implementation
            )
            is_full_cache = method.is_full_cache
            retention_one_identity = not is_full_cache and self.config.cache.retention_ratio == 1.0
            fixed_diagnostic_sample = (
                self.config.diagnostics.enabled
                and request.sample_id in self.config.diagnostics.sample_ids
            )
            needs_prompt_attention = not retention_one_identity and (
                method.is_mosaickv
                or method is MosaicKVMethod.PROMPT_ATTENTION_TOPK
                or method.is_published_reimplementation
            )
            prepared = self.adapter.prepare_inputs(request.messages)

            def run_prefill() -> Any:
                return self.adapter.prefill(
                    prepared,
                    capture_queries=method.is_mosaickv and not retention_one_identity,
                    capture_attentions=needs_prompt_attention,
                    attention_query_window=(
                        None
                        if method.is_published_reimplementation
                        else (
                            max(1, self.config.forecasting.prompt_window)
                            if needs_prompt_attention
                            else None
                        )
                    ),
                )

            image_video_encoder_seconds: float | None = None
            projector_seconds: float | None = None
            language_model_prefill_seconds: float | None = None
            is_cuda = (
                torch.cuda.is_available() and getattr(self.adapter.device, "type", "cpu") == "cuda"
            )
            if is_cuda:
                profiling = self.adapter.get_profiling_modules()
                audit = SynchronizationAudit()
                vision_timer = ModuleCudaTimer(
                    profiling.vision_encoder,
                    torch,
                    self.adapter.device,
                    audit,
                )
                projector_timer = ModuleCudaTimer(
                    profiling.projector,
                    torch,
                    self.adapter.device,
                    audit,
                )
                language_timer = ModuleCudaTimer(
                    profiling.language_model,
                    torch,
                    self.adapter.device,
                    audit,
                )

                def instrumented_prefill() -> Any:
                    with ExitStack() as stack:
                        stack.enter_context(vision_timer)
                        stack.enter_context(projector_timer)
                        stack.enter_context(language_timer)
                        return run_prefill()

                prefill, prefill_seconds = _timed(
                    prepared.model_inputs["input_ids"],
                    instrumented_prefill,
                )
                image_video_encoder_seconds = vision_timer.total_seconds
                projector_seconds = projector_timer.total_seconds
                language_model_prefill_seconds = language_timer.total_seconds
                if (
                    profiling.vision_includes_projector
                    and image_video_encoder_seconds is not None
                    and projector_seconds is not None
                ):
                    image_video_encoder_seconds = max(
                        0.0,
                        image_video_encoder_seconds - projector_seconds,
                    )
            else:
                prefill, prefill_seconds = _timed(
                    prepared.model_inputs["input_ids"],
                    run_prefill,
                )
            if not bool(torch.isfinite(prefill.logits).all()):
                raise HuggingFaceRuntimeError("prefill logits contain NaN or infinity")
            source_prefill_kv_bytes = _cache_payload_bytes(
                self.adapter,
                prefill.state.past_key_values,
            )
            logical_prefill_active_kv_bytes = source_prefill_kv_bytes
            packed_prefill_active_kv_bytes = source_prefill_kv_bytes
            forecast_seconds = 0.0
            graph_seconds = 0.0
            selection_seconds = 0.0
            tier_seconds = 0.0
            packing_seconds = 0.0
            repair_state: RepairCacheState | None = None
            forecast_rollout: IsolatedQueryRollout | None = None
            dense_budget: DensePhysicalCacheBudget | None = None
            physical_prompt_capacity: int | None = None
            plan: MosaicKVCompressionPlan | None = None
            baseline_plan: BaselineCompressionPlan | None = None
            lookm_plan: LookMCompressionPlan | None = None
            prefixkv_plan: PrefixKVCompressionPlan | None = None
            vl_cache_plan: VLCacheCompressionPlan | None = None
            state = prefill.state
            if not is_full_cache and not retention_one_identity:
                snapshot = self.adapter.extract_past_key_values(prefill.state.past_key_values)
                input_ids = prepared.model_inputs["input_ids"]
                recent_count = (
                    int(prefill.state.logical_sequence_length * self.config.lookm.recent_ratio)
                    if method.is_lookm_reimplementation
                    else 1
                )
                if method.is_lookm_reimplementation:
                    mandatory_positions = tuple(
                        range(
                            prefill.state.logical_sequence_length - recent_count,
                            prefill.state.logical_sequence_length,
                        )
                    )
                elif method.is_prefixkv_reimplementation:
                    length = prefill.state.logical_sequence_length
                    mandatory_positions = tuple(
                        sorted(
                            set(range(min(self.config.prefixkv.start_size, length)))
                            | set(
                                range(
                                    max(0, length - self.config.prefixkv.protect_size),
                                    length,
                                )
                            )
                        )
                    )
                else:
                    mandatory_positions = (prefill.state.logical_sequence_length - 1,)
                full_state = FullKVState.from_cache_snapshot(
                    snapshot,
                    modality_spans=prepared.modality_map,
                    token_ids=input_ids,
                    block_size=self.config.cache.block_size,
                    original_logical_sequence_length=prefill.state.logical_sequence_length,
                    next_decode_position=prefill.state.next_decode_position,
                    mandatory_logical_positions=mandatory_positions,
                )
                dense_budget = _dense_physical_cache_budget(full_state, self.config)
                physical_prompt_capacity = (
                    None if dense_budget is None else dense_budget.prompt_capacity
                )
                planning_config = (
                    self.config
                    if dense_budget is None
                    else replace(
                        self.config,
                        cache=replace(
                            self.config.cache,
                            budget_value=dense_budget.target_bytes,
                        ),
                    )
                )
                if method.is_mosaickv:
                    forecast, rollout = forecast_with_rollout(
                        self.adapter,
                        prefill,
                        self.config.forecasting,
                        capture_attentions=True,
                    )
                    forecast_rollout = rollout
                    forecast_seconds = forecast.timing.total
                    attention = _forecast_attention_by_node(
                        full_state,
                        prefill,
                        rollout,
                        self.config,
                    )
                    plan = build_compression_plan(
                        full_state,
                        forecast,
                        attention,
                        self.adapter.capabilities,
                        planning_config,
                        position_limits=(
                            None if dense_budget is None else dense_budget.position_limits
                        ),
                    )
                    graph_seconds = plan.graph_seconds
                    selection_seconds = plan.selection_seconds
                    tier_seconds = plan.tier_seconds
                    packed_state = plan.construction.state
                    prototype_records = plan.construction.prototypes
                    logical_prefill_active_kv_bytes = (
                        plan.construction.state.statistics.active_kv_bytes
                    )
                    if (
                        method is MosaicKVMethod.MOSAICKV_EXACT
                        and dense_budget is not None
                        and not self.config.selection.stop_on_nonpositive_gain
                        and logical_prefill_active_kv_bytes
                        < dense_budget.minimum_block_representable_fill(
                            self.config.cache.block_size
                        )
                    ):
                        minimum_representable = dense_budget.minimum_block_representable_fill(
                            self.config.cache.block_size
                        )
                        raise HuggingFaceRuntimeError(
                            "budget-filling MosaicKV selection underfilled the dense cache: "
                            f"selected={logical_prefill_active_kv_bytes}, "
                            f"minimum_representable={minimum_representable}, "
                            f"target={dense_budget.target_bytes}"
                        )
                elif method.is_compressed_baseline:
                    prompt_started = time.perf_counter()
                    prompt_attention = (
                        _prompt_attention_by_node(
                            full_state,
                            prefill,
                            prompt_window=self.config.forecasting.prompt_window,
                        )
                        if method is MosaicKVMethod.PROMPT_ATTENTION_TOPK
                        else None
                    )
                    prompt_scoring_seconds = time.perf_counter() - prompt_started
                    baseline_plan = build_exact_baseline_plan(
                        full_state,
                        method,
                        planning_config.cache,
                        position_limits=(
                            None if dense_budget is None else dense_budget.position_limits
                        ),
                        seed=self.config.execution.seed,
                        prompt_attention_by_node=prompt_attention,
                        value_chunk_size=self.config.graph.similarity_chunk_size,
                    )
                    selection_seconds = prompt_scoring_seconds + baseline_plan.selection_seconds
                    tier_seconds = baseline_plan.tier_seconds
                    packed_state = baseline_plan.state
                    prototype_records = ()
                    logical_prefill_active_kv_bytes = baseline_plan.selection.active_bytes
                elif method.is_lookm_reimplementation:
                    lookm_plan, selection_seconds = _timed(
                        prefill.logits,
                        lambda: build_lookm_reimpl_plan(
                            full_state,
                            prefill.attention_weights,
                            self.config.lookm,
                            planning_config.cache,
                        ),
                    )
                    packed_state = None
                    prototype_records = ()
                    logical_prefill_active_kv_bytes = lookm_plan.active_bytes
                elif method.is_vl_cache_reimplementation:
                    vl_cache_plan, selection_seconds = _timed(
                        prefill.logits,
                        lambda: build_vl_cache_reimpl_plan(
                            full_state,
                            prefill.attention_weights,
                            self.config.vl_cache,
                            planning_config.cache,
                            physical_prompt_capacity=(
                                None if dense_budget is None else dense_budget.prompt_capacity
                            ),
                        ),
                    )
                    packed_state = None
                    prototype_records = ()
                    logical_prefill_active_kv_bytes = vl_cache_plan.retained_bytes
                elif method.is_prefixkv_reimplementation:
                    prefixkv_plan, selection_seconds = _timed(
                        prefill.logits,
                        lambda: build_prefixkv_reimpl_plan(
                            full_state,
                            prefill.attention_weights,
                            self.config.prefixkv,
                            planning_config.cache,
                            model_id=self.config.model.id,
                            model_revision=self.config.model.revision,
                            profile=self._prefixkv_profile,
                            physical_prompt_capacity=(
                                None if dense_budget is None else dense_budget.prompt_capacity
                            ),
                        ),
                    )
                    packed_state = None
                    prototype_records = ()
                    logical_prefill_active_kv_bytes = prefixkv_plan.retained_bytes
                else:  # pragma: no cover - strict method vocabulary guards this
                    raise HuggingFaceRuntimeError(f"unsupported runtime method: {method.value}")
                if lookm_plan is not None:
                    lookm_payloads, lookm_slots = lookm_runtime_payloads(lookm_plan)
                    packed, packing_seconds = _timed(
                        prefill.logits,
                        lambda: pack_runtime_payloads(
                            self.adapter,
                            full_state,
                            lookm_payloads,
                            lookm_slots,
                            physical_prompt_capacity=(
                                None if dense_budget is None else dense_budget.prompt_capacity
                            ),
                        ),
                    )
                elif prefixkv_plan is not None:
                    prefixkv_payloads, prefixkv_slots = prefixkv_runtime_payloads(prefixkv_plan)
                    packed, packing_seconds = _timed(
                        prefill.logits,
                        lambda: pack_runtime_payloads(
                            self.adapter,
                            full_state,
                            prefixkv_payloads,
                            prefixkv_slots,
                            physical_prompt_capacity=(
                                None if dense_budget is None else dense_budget.prompt_capacity
                            ),
                        ),
                    )
                elif vl_cache_plan is not None:
                    vl_cache_payloads, vl_cache_slots = vl_cache_runtime_payloads(vl_cache_plan)
                    packed, packing_seconds = _timed(
                        prefill.logits,
                        lambda: pack_runtime_payloads(
                            self.adapter,
                            full_state,
                            vl_cache_payloads,
                            vl_cache_slots,
                            physical_prompt_capacity=(
                                None if dense_budget is None else dense_budget.prompt_capacity
                            ),
                        ),
                    )
                else:
                    if packed_state is None:
                        raise HuggingFaceRuntimeError("runtime cache plan has no active state")
                    packed, packing_seconds = _timed(
                        prefill.logits,
                        lambda: pack_runtime_cache(
                            self.adapter,
                            full_state,
                            packed_state,
                            prototype_records,
                            tuple(record.prototype_id for record in prototype_records),
                            physical_prompt_capacity=physical_prompt_capacity,
                        ),
                    )
                state = _state_from_packed(self.adapter, prefill.state, packed)
                packed_prefill_active_kv_bytes = _cache_payload_bytes(
                    self.adapter,
                    packed.snapshot,
                )
                if (
                    dense_budget is not None
                    and packed_prefill_active_kv_bytes > dense_budget.target_bytes
                ):
                    raise HuggingFaceRuntimeError(
                        "materialized cache exceeded its hard physical byte budget: "
                        f"active={packed_prefill_active_kv_bytes}, "
                        f"budget={dense_budget.target_bytes}"
                    )
                if method is MosaicKVMethod.MOSAICKV_FULL and plan is not None:
                    repair_state = RepairCacheState.from_construction(
                        full_state,
                        plan.construction,
                    )
                trace_started = time.perf_counter()
                if plan is not None:
                    trace.update(
                        _plan_trace(
                            plan,
                            compact_catalogs=not fixed_diagnostic_sample,
                        )
                    )
                elif baseline_plan is not None:
                    trace.update(
                        _baseline_plan_trace(
                            baseline_plan,
                            compact_catalogs=not fixed_diagnostic_sample,
                        )
                    )
                elif lookm_plan is not None:
                    trace.update(_lookm_plan_trace(lookm_plan))
                elif prefixkv_plan is not None:
                    trace.update(_prefixkv_plan_trace(prefixkv_plan))
                elif vl_cache_plan is not None:
                    trace.update(_vl_cache_plan_trace(vl_cache_plan))
                trace["packed_slots"] = _trace_catalog(
                    packed.slot_records,
                    compact=not fixed_diagnostic_sample,
                )
                trace["packed_cache_all_slots_valid"] = packed.all_slots_valid
            elif retention_one_identity:
                trace_started = time.perf_counter()
                trace.update(
                    {
                        "requested_method": method.value,
                        "effective_method": f"{method.value}__retention_one_exact",
                        "tier_mode": "retention_one",
                        "tier_reason": (
                            "retention 1.0 reuses the untouched FullKV state; no "
                            "forecast, graph, selection, packing, prototype, residual, "
                            "quantization, offload, or repair operation was run"
                        ),
                        "selected_blocks": [],
                        "prototypes": [],
                        "graph_edges": [],
                        "graph_diagnostics": {"status": "not_used_by_retention_one_identity"},
                        "forecast_statistics": {"status": "not_used_by_retention_one_identity"},
                        "baseline_seed": self.config.execution.seed,
                        "selection_provenance": "retention_one_identity_no_selection",
                        "packed_slots": [],
                        "packed_cache_all_slots_valid": True,
                    }
                )
            else:
                trace_started = time.perf_counter()
                trace.update(
                    {
                        "requested_method": method.value,
                        "effective_method": method.value,
                        "tier_mode": "full_cache",
                        "tier_reason": "no cache transformation was run",
                        "selected_blocks": [],
                        "prototypes": [],
                        "graph_edges": [],
                        "graph_diagnostics": {"status": "not_used_by_full_kv"},
                        "forecast_statistics": {"status": "not_used_by_full_kv"},
                        "baseline_seed": self.config.execution.seed,
                        "selection_provenance": "full_cache_no_selection",
                        "packed_slots": [],
                        "packed_cache_all_slots_valid": True,
                    }
                )

            if not fixed_diagnostic_sample:
                _compact_trace_records(trace)
                trace["trace_detail"] = "compact_non_diagnostic"
            else:
                trace["trace_detail"] = "full_debug"
            trace_instrumentation_seconds += time.perf_counter() - trace_started

            fidelity: FidelityDiagnostic | None = None
            fidelity_seconds = 0.0
            forecast_diagnostic_seconds = 0.0
            if (
                fixed_diagnostic_sample
                and self.config.diagnostics.forecast_oracle_steps > 0
                and plan is not None
            ):
                if forecast_rollout is None:
                    raise HuggingFaceRuntimeError(
                        "MosaicKV oracle diagnostics are missing the online rollout"
                    )
                forecast_diagnostic, forecast_diagnostic_seconds = _timed(
                    prefill.logits,
                    lambda: _forecast_quality_diagnostic(
                        self.adapter,
                        prefill,
                        plan.full_state,
                        plan.forecast,
                        forecast_rollout,
                        plan.selection,
                        self.config,
                    ),
                )
                forecast_payload = cast("JsonObject", asdict(forecast_diagnostic))
                forecast_payload["overhead_seconds"] = forecast_diagnostic_seconds
                forecast_payload["online_selection_isolation"] = (
                    "evaluation_only_output_not_passed_to_compressor"
                )
                trace["forecast_quality_diagnostic"] = forecast_payload
            elif self.config.diagnostics.forecast_oracle_steps > 0:
                trace["forecast_quality_diagnostic"] = {
                    "status": (
                        "not_in_fixed_subset"
                        if not fixed_diagnostic_sample
                        else "not_applicable_to_non_mosaickv_method"
                    )
                }
            else:
                trace["forecast_quality_diagnostic"] = {"status": "disabled"}
            if (
                self.config.diagnostics.enabled
                and request.sample_id in self.config.diagnostics.sample_ids
            ):
                fidelity, fidelity_seconds = _timed(
                    prefill.logits,
                    lambda: _run_fidelity_diagnostic(
                        self.adapter,
                        prefill,
                        state,
                        max_new_tokens=self.config.generation.max_new_tokens,
                    ),
                )
                if (
                    torch.cuda.is_available()
                    and getattr(self.adapter.device, "type", "cpu") == "cuda"
                ):
                    torch.cuda.synchronize(self.adapter.device)
                    torch.cuda.reset_peak_memory_stats(self.adapter.device)
                trace["fidelity_diagnostic"] = {
                    "status": "completed",
                    "sample_ledger_sha": self.config.diagnostics.sample_ledger_sha,
                    "token_agreement": fidelity.token_agreement,
                    "sequence_agreement": fidelity.sequence_agreement,
                    "logit_kl": fidelity.logit_kl,
                    "attention_output_error": fidelity.attention_output_error,
                    "overhead_seconds": fidelity_seconds,
                    "trajectory": "independent_deterministic_greedy",
                    "attention_output_definition": "pre_output_projection_weighted_value",
                }
            elif self.config.diagnostics.enabled:
                trace["fidelity_diagnostic"] = {
                    "status": "not_in_fixed_subset",
                    "sample_ledger_sha": self.config.diagnostics.sample_ledger_sha,
                }
            else:
                trace["fidelity_diagnostic"] = {"status": "disabled"}

            preallocated_decode_cache = False
            if repair_state is None and prefixkv_plan is None:
                preallocated_decode_cache = self.adapter.preallocate_decode_cache(
                    state,
                    self.config.generation.max_new_tokens - 1,
                )
            decode_runtime_policy["kv_growth"] = (
                "preallocated_append_only_views"
                if preallocated_decode_cache
                else "backend_native_dynamic_growth"
            )
            decode_attention = (
                self.config.execution.decode_attention_implementation
                or self.config.execution.attention_implementation
            )
            if decode_attention != self.config.execution.attention_implementation:
                if repair_state is not None:
                    raise HuggingFaceRuntimeError(
                        "residual repair requires eager decode attention capture"
                    )
                if state.model_state.get("mosaickv_validity_masks") is not None:
                    raise HuggingFaceRuntimeError(
                        "hybrid decode attention is not yet admitted for padded per-head caches"
                    )
                self.adapter.set_attention_implementation(decode_attention)
            decode_runtime_policy["prefill_attention_implementation"] = (
                self.config.execution.attention_implementation
            )
            decode_runtime_policy["decode_attention_implementation"] = decode_attention

            tokens = [prefill.next_token_id]
            token = prefill.next_token_id
            decode_seconds = 0.0
            per_token_decode_seconds: list[float] = []
            decode_finite_flags: list[Any] = []
            repair_events: list[RepairEvent] = []
            prefixkv_decode_events: list[JsonObject] = []
            for step_index in range(1, self.config.generation.max_new_tokens):
                # Hugging Face cache objects normally grow in place.  A
                # provisional repair step therefore needs an isolated copy so
                # it can be discarded and re-decoded, but every non-repair
                # path can advance the live state directly.  Cloning the full
                # cache unconditionally here made ordinary token decoding
                # O(sequence_length) in host-side copy work per step.
                before = state if repair_state is None else _clone_decode_state(self.adapter, state)
                decode_token = token
                needs_attention = repair_state is not None
                provisional, step_seconds = _timed(
                    decode_token,
                    lambda current=decode_token, current_state=before, capture=needs_attention: (
                        self.adapter.decode_one_token(
                            current,
                            current_state,
                            capture_queries=False,
                            capture_attentions=capture,
                        )
                    ),
                )
                decode_seconds += step_seconds
                per_token_decode_seconds.append(step_seconds)
                # Queue device-side validation and reduce once after the
                # sequence.  Converting every scalar flag to bool here forced
                # a second CUDA synchronization on every token in addition to
                # the required timing boundary.
                decode_finite_flags.append(torch.isfinite(provisional.logits).all())
                if repair_state is None:
                    if prefixkv_plan is not None:

                        def maintain_prefix(
                            output: Any = provisional,
                            current_step: int = step_index,
                        ) -> tuple[DecodeState, tuple[JsonObject, ...]]:
                            return maintain_prefixkv_decode_cache(
                                self.adapter,
                                output.state,
                                prefixkv_plan,
                                step_index=current_step,
                            )

                        maintained, maintenance_seconds = _timed(
                            provisional.logits,
                            maintain_prefix,
                        )
                        state, maintenance_events = maintained
                        prefixkv_decode_events.extend(maintenance_events)
                        decode_seconds += maintenance_seconds
                        per_token_decode_seconds[-1] += maintenance_seconds
                    else:
                        state = provisional.state
                    token = provisional.next_token_id
                    tokens.append(token)
                    continue

                if plan is None:
                    raise HuggingFaceRuntimeError(
                        "repair state exists without a MosaicKV compression plan"
                    )

                prototype_slots = cast(
                    "tuple[tuple[int, int, int, int], ...]",
                    before.model_state.get("mosaickv_prototype_slots", ()),
                )
                raw_validity_masks = before.model_state.get("mosaickv_validity_masks")
                if raw_validity_masks is None:
                    if before.model_state.get("mosaickv_packed_all_valid") is not True:
                        raise HuggingFaceRuntimeError(
                            "repair cache is missing packed validity metadata"
                        )
                    # The dense fast path intentionally omits per-layer masks.
                    # Repair only needs their KV-head dimension to map grouped
                    # query heads back to prototype slots, so synthesize the
                    # all-valid shape without reintroducing a decode-time mask.
                    raw_validity_masks = tuple(
                        torch.ones(
                            (layer.kv_heads, before.active_cache_length),
                            dtype=torch.bool,
                            device=provisional.logits.device,
                        )
                        for layer in plan.full_state.layers
                    )
                validity_masks = cast("tuple[Any, ...]", raw_validity_masks)
                masses = _prototype_attention_mass(
                    provisional.attention_weights,
                    prototype_slots,
                    validity_masks,
                )
                repaired_output: list[Any] = []
                repair_base_state = state
                repair_token = token
                repair_sink = repaired_output

                def re_decode(
                    updated_repair: RepairCacheState,
                    base_state: DecodeState = repair_base_state,
                    decode_token: Any = repair_token,
                    sink: list[Any] = repair_sink,
                ) -> Any:
                    old_snapshot = self.adapter.extract_past_key_values(base_state.past_key_values)
                    old_capacity = int(base_state.model_state["mosaickv_prompt_capacity"])
                    repacked = pack_runtime_cache(
                        self.adapter,
                        updated_repair.full_state,
                        updated_repair.mosaic_state,
                        updated_repair.prototype_catalog,
                        updated_repair.active_prototype_ids,
                        tail_snapshot=old_snapshot,
                        old_prompt_capacity=old_capacity,
                    )
                    repacked_state = _state_from_packed(self.adapter, base_state, repacked)
                    output = self.adapter.decode_one_token(
                        decode_token,
                        repacked_state,
                        capture_queries=False,
                        capture_attentions=True,
                    )
                    sink.append(output)
                    return output.logits

                repair = repair_decode_step(
                    repair_state,
                    self.config.repair,
                    step_index=step_index,
                    provisional_logits=provisional.logits,
                    prototype_attention_mass=masses,
                    re_decode=re_decode,
                )
                repair_state = repair.state
                repair_events.append(repair.event)
                if repair.event.triggered:
                    if len(repaired_output) != 1:
                        raise HuggingFaceRuntimeError("triggered repair did not re-decode once")
                    final = repaired_output[0]
                    state = final.state
                    token = final.next_token_id
                    decode_finite_flags.append(torch.isfinite(final.logits).all())
                else:
                    state = provisional.state
                    token = provisional.next_token_id
                tokens.append(token)

            if decode_finite_flags:
                finite_logits, validation_seconds = _timed(
                    prefill.logits,
                    lambda: torch.stack(decode_finite_flags).all(),
                )
                trace_instrumentation_seconds += validation_seconds
                if not bool(finite_logits):
                    raise HuggingFaceRuntimeError("decode logits contain NaN or infinity")

            token_ids = torch.cat(tokens, dim=-1)
            timed_token_ids = tuple(
                int(value) for value in token_ids.detach().cpu().reshape(-1).tolist()
            )
            quality = decode_quality_output(
                self.adapter.processor,
                timed_token_ids,
                self._terminal_token_ids,
            )
            answer = quality.text
            live_cache = self.adapter.extract_past_key_values(state.past_key_values, clone=False)
            active_bytes = sum(
                tensor_storage_bytes(key) + tensor_storage_bytes(value)
                for key, value in self.adapter._legacy_layers(live_cache)[0]
            )
            residual_bytes = 0 if repair_state is None else repair_state.residual_storage.cpu_bytes
            total_seconds = max(
                0.0,
                time.perf_counter()
                - started
                - fidelity_seconds
                - forecast_diagnostic_seconds
                - trace_instrumentation_seconds,
            )
            repair_transfer = sum(event.transfer_time_ms for event in repair_events) / 1000.0
            repair_redecode = sum(event.re_decode_time_ms for event in repair_events) / 1000.0
            timings = RuntimePhaseTimings(
                prefill=prefill_seconds,
                image_video_encoder=image_video_encoder_seconds,
                projector=projector_seconds,
                language_model_prefill=language_model_prefill_seconds,
                forecast=forecast_seconds,
                graph=graph_seconds,
                utility_and_selection=selection_seconds,
                tier_construction=tier_seconds,
                cache_packing=packing_seconds,
                decode=decode_seconds,
                per_token_decode=tuple(per_token_decode_seconds),
                repair_transfer=repair_transfer,
                repair_redecode=repair_redecode,
                total=total_seconds,
            )
            timing_payload = cast("JsonObject", asdict(timings))
            timing_payload["trace_instrumentation"] = trace_instrumentation_seconds
            trace.update(
                {
                    "status": "completed",
                    "answer": answer,
                    "generated_token_ids": list(timed_token_ids),
                    "quality_decoding": {
                        **quality.to_record(),
                        "configured_terminal_token_ids": list(self._terminal_token_ids),
                    },
                    "repair_events": [event.to_record() for event in repair_events],
                    "prefixkv_decode_events": cast("list[JsonValue]", prefixkv_decode_events),
                    "timing_breakdown": timing_payload,
                    "source_prefill_kv_bytes": source_prefill_kv_bytes,
                    "logical_prefill_active_kv_bytes": logical_prefill_active_kv_bytes,
                    "packed_prefill_active_kv_bytes": packed_prefill_active_kv_bytes,
                    "packed_padding_kv_bytes": (
                        packed_prefill_active_kv_bytes - logical_prefill_active_kv_bytes
                    ),
                    "logical_cache_utilization": (
                        logical_prefill_active_kv_bytes / packed_prefill_active_kv_bytes
                    ),
                    "physical_budget_utilization": (
                        1.0
                        if dense_budget is None
                        else packed_prefill_active_kv_bytes / dense_budget.target_bytes
                    ),
                    "hard_active_kv_budget_bytes": (
                        None if dense_budget is None else dense_budget.target_bytes
                    ),
                    "active_kv_bytes": packed_prefill_active_kv_bytes,
                    "final_active_kv_bytes": active_bytes,
                    "residual_kv_bytes": residual_bytes,
                    "peak_gpu_memory_allocated": (
                        int(torch.cuda.max_memory_allocated(self.adapter.device))
                        if is_cuda
                        else None
                    ),
                    "peak_gpu_memory_reserved": (
                        int(torch.cuda.max_memory_reserved(self.adapter.device))
                        if is_cuda
                        else None
                    ),
                    "trace_path": str(trace_path.resolve()),
                }
            )
            _write_trace(trace_path, trace)
            compression_seconds = (
                forecast_seconds
                + graph_seconds
                + selection_seconds
                + tier_seconds
                + packing_seconds
            )
            peak_memory = (
                int(torch.cuda.max_memory_allocated(self.adapter.device))
                if torch.cuda.is_available()
                and getattr(self.adapter.device, "type", "cpu") == "cuda"
                else None
            )
            effective_method = str(trace.get("effective_method", self.method))
            return ModelGeneration(
                answer,
                GenerationMetrics(
                    ttft=prefill_seconds + compression_seconds,
                    prefill_time=prefill_seconds,
                    compression_time=compression_seconds,
                    decode_time=decode_seconds,
                    end_to_end_time=total_seconds,
                    generated_tokens=int(token_ids.numel()),
                    active_kv_bytes=packed_prefill_active_kv_bytes,
                    residual_kv_bytes=residual_bytes,
                    peak_gpu_memory=peak_memory,
                    repair_count=sum(event.triggered for event in repair_events),
                    repaired_bytes=sum(event.restored_bytes for event in repair_events),
                    token_agreement=(None if fidelity is None else fidelity.token_agreement),
                    logit_kl=None if fidelity is None else fidelity.logit_kl,
                    attention_output_error=(
                        None if fidelity is None else fidelity.attention_output_error
                    ),
                ),
                effective_method=effective_method,
            )
        except Exception as error:
            trace.update(
                {
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "end_to_end_time": time.perf_counter() - started,
                    "trace_path": str(trace_path.resolve()),
                }
            )
            if not trace_path.exists():
                _write_trace(trace_path, trace)
            raise


__all__ = [
    "HuggingFaceMosaicKVModel",
    "HuggingFaceRuntimeError",
    "MosaicKVCompressionPlan",
    "PackedRuntimeCache",
    "RuntimePhaseTimings",
    "build_compression_plan",
    "compare_runtime_retention_one",
    "maintain_prefixkv_decode_cache",
    "pack_runtime_cache",
    "pack_runtime_payloads",
]
