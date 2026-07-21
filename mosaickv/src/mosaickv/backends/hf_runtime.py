"""Unified eager Hugging Face runtime for FullKV and MosaicKV variants."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
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
from mosaickv.cache_state import FullKVState, MosaicKVState, tensor_storage_bytes
from mosaickv.config import RunConfig
from mosaickv.evaluation.model import EvaluationRequest, GenerationMetrics, ModelGeneration
from mosaickv.forecasting import QueryForecast, forecast_with_rollout
from mosaickv.forecasting.huggingface import IsolatedQueryRollout
from mosaickv.graph import SparseEvidenceGraph, build_evidence_graph
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
    forecast: float
    graph: float
    utility_and_selection: float
    tier_construction: float
    cache_packing: float
    decode: float
    repair_transfer: float
    repair_redecode: float
    total: float

    def __post_init__(self) -> None:
        if any(not math.isfinite(value) or value < 0 for value in asdict(self).values()):
            raise ValueError("runtime timings must be finite and nonnegative")


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
    prototype_slots: tuple[tuple[int, int, int, int], ...]
    slot_records: tuple[JsonObject, ...]


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

    if not prefill.attention_weights:
        raise HuggingFaceRuntimeError("runtime prefill did not capture eager attention weights")
    prompt_layers = tuple(_to_numpy(value) for value in prefill.attention_weights)
    if len(prompt_layers) != len(full_state.layers):
        raise HuggingFaceRuntimeError("prefill attention layer count does not match the cache")
    draft_steps = tuple(
        tuple(_to_numpy(value) for value in step) for step in rollout.attention_steps
    )
    result: dict[int, float] = {}
    for node_id, block in enumerate(full_state.blocks):
        layer = full_state.layers[block.layer]
        prompt = prompt_layers[block.layer]
        if prompt.ndim != 4 or prompt.shape[0] != 1:
            raise HuggingFaceRuntimeError("eager prompt attention must have shape [1,H,Q,K]")
        query_heads = int(prompt.shape[1])
        if query_heads % layer.kv_heads:
            raise HuggingFaceRuntimeError("query heads are not divisible by cache KV heads")
        group = query_heads // layer.kv_heads
        query_start = block.kv_head * group
        query_end = query_start + group
        samples: list[float] = []
        if config.forecasting.prompt_window:
            window = min(config.forecasting.prompt_window, int(prompt.shape[-2]))
            selected = prompt[
                0,
                query_start:query_end,
                -window:,
                list(block.physical_cache_indices),
            ]
            samples.append(float(selected.sum(axis=-1).mean()))
        for step in draft_steps:
            attention = step[block.layer]
            if attention.ndim != 4 or attention.shape[0] != 1 or attention.shape[-2] != 1:
                raise HuggingFaceRuntimeError("eager draft attention must have shape [1,H,1,K]")
            selected = attention[
                0,
                query_start:query_end,
                0,
                list(block.physical_cache_indices),
            ]
            samples.append(float(selected.sum(axis=-1).mean()))
        if not samples:
            raise HuggingFaceRuntimeError("forecast attention has no prompt or draft samples")
        result[node_id] = max(float(np.mean(samples)), float(np.finfo(np.float64).tiny))
    return result


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
) -> SelectionResult:
    if retention_one:
        return select_all_exact(objective, SelectionBudget(value, unit))
    return lazy_greedy_select(objective, SelectionBudget(value, unit))


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
        selection = _selection(objective, target, unit, retention_one=False)
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
        candidate = _selection(objective, exact_budget, unit, retention_one=False)
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
    fallback = _selection(objective, target, unit, retention_one=False)
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
    torch = _torch()
    index = torch.tensor((offset,), dtype=torch.long, device=tensor.device)
    return torch.index_select(tensor, sequence_axis, index)


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
) -> PackedRuntimeCache:
    """Pack algorithm-owned head payloads through the common HF cache interface."""

    torch = _torch()
    prompt_capacity = max((len(items) for items in payloads.values()), default=0)
    if prompt_capacity < 1:
        raise HuggingFaceRuntimeError("packed runtime cache has no active prompt slots")
    if tail_snapshot is not None and old_prompt_capacity is None:
        raise ValueError("tail packing requires the previous prompt capacity")
    layers: list[CacheLayerSnapshot] = []
    validity_masks: list[Any] = []
    prototype_slots: list[tuple[int, int, int, int]] = []
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
        if not bool(torch.isfinite(packed_key).all()) or not bool(
            torch.isfinite(packed_value).all()
        ):
            raise HuggingFaceRuntimeError("packed cache contains NaN or infinity")
        layers.append(CacheLayerSnapshot(packed_key, packed_value, key_sequence))
        validity_masks.append(torch.stack(head_validity, dim=0))
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
        tuple(sorted(prototype_slots)),
        slot_records,
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
) -> PackedRuntimeCache:
    """Pack MosaicKV/simple-baseline tiers through the common HF interface."""

    key_state = str(getattr(full_state.cached_key_state, "value", full_state.cached_key_state))
    if state.prototypes.prototype_keys and key_state == "pre_rope":
        raise HuggingFaceRuntimeError(
            "HF prototype materialization requires an adapter-specific safe RoPE transform"
        )
    payloads, slot_records = _tier_payloads(state, records, active_prototype_ids)
    return pack_runtime_payloads(
        adapter,
        full_state,
        payloads,
        slot_records,
        tail_snapshot=tail_snapshot,
        old_prompt_capacity=old_prompt_capacity,
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
            "mosaickv_validity_masks": packed.validity_masks,
            "mosaickv_prompt_capacity": packed.prompt_capacity,
            "mosaickv_prototype_slots": packed.prototype_slots,
        }
    )
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


def _forecast_trace(forecast: QueryForecast) -> JsonObject:
    heads: list[JsonObject] = []
    for layer in forecast.layers:
        for head in layer:
            centroids = _to_numpy(head.normalized_centroids)
            weights = _to_numpy(head.forecast_weights)
            heads.append(
                {
                    "layer": head.provenance.layer,
                    "kv_head": head.provenance.kv_head,
                    "prompt_sample_count": head.provenance.prompt_sample_count,
                    "draft_sample_count": head.provenance.draft_sample_count,
                    "centroid_shape": list(centroids.shape),
                    "centroid_norms": np.linalg.norm(centroids, axis=-1).tolist(),
                    "forecast_weights": weights.tolist(),
                }
            )
    return cast(
        "JsonObject",
        {
            "provenance": asdict(forecast.provenance),
            "timing": asdict(forecast.timing),
            "heads": heads,
        },
    )


def _plan_trace(plan: MosaicKVCompressionPlan) -> JsonObject:
    selection = [
        {
            "node_id": decision.node_id,
            "selected": decision.selected,
            "selection_rank": decision.selection_rank,
            "marginal_gain": decision.marginal_gain,
            "reason": decision.reason.value,
        }
        for decision in plan.selection.decisions
    ]
    prototypes = [
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
    ]
    edges = [
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
    ]
    return cast(
        "JsonObject",
        {
            "requested_method": plan.method.value,
            "effective_method": plan.effective_method,
            "tier_mode": plan.construction.mode.value,
            "tier_reason": plan.construction.reason,
            "source_budget_value": plan.source_budget_value,
            "active_budget_value": plan.active_budget_value,
            "selected_blocks": selection,
            "prototypes": prototypes,
            "graph_edges": edges,
            "graph_diagnostics": asdict(plan.graph.diagnostics),
            "forecast_statistics": _forecast_trace(plan.forecast),
        },
    )


def _baseline_plan_trace(plan: BaselineCompressionPlan) -> JsonObject:
    selection = [
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
    ]
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
            "selected_blocks": selection,
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
        if config.execution.attention_implementation != "eager":
            raise ValueError("the unified HF runtime is correctness-gated only for eager attention")
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
        return self.trace_directory / request.run_id / f"{identity}.json"

    def generate(self, request: EvaluationRequest) -> ModelGeneration:
        """Run one explicit prefill/compress/decode request and always write a trace."""

        torch = _torch()
        trace_path = self._trace_path(request)
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
        }
        started = time.perf_counter()
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
            is_full_cache = method.is_full_cache
            needs_prompt_attention = (
                method.is_mosaickv
                or method is MosaicKVMethod.PROMPT_ATTENTION_TOPK
                or method.is_published_reimplementation
            )
            prepared = self.adapter.prepare_inputs(request.messages)
            prefill, prefill_seconds = _timed(
                prepared.model_inputs["input_ids"],
                lambda: self.adapter.prefill(
                    prepared,
                    capture_queries=method.is_mosaickv,
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
                ),
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
            plan: MosaicKVCompressionPlan | None = None
            baseline_plan: BaselineCompressionPlan | None = None
            lookm_plan: LookMCompressionPlan | None = None
            prefixkv_plan: PrefixKVCompressionPlan | None = None
            vl_cache_plan: VLCacheCompressionPlan | None = None
            state = prefill.state
            if not is_full_cache:
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
                if method.is_mosaickv:
                    forecast, rollout = forecast_with_rollout(
                        self.adapter,
                        prefill,
                        self.config.forecasting,
                        capture_attentions=True,
                    )
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
                        self.config,
                    )
                    graph_seconds = plan.graph_seconds
                    selection_seconds = plan.selection_seconds
                    tier_seconds = plan.tier_seconds
                    packed_state = plan.construction.state
                    prototype_records = plan.construction.prototypes
                    logical_prefill_active_kv_bytes = (
                        plan.construction.state.statistics.active_kv_bytes
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
                        self.config.cache,
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
                            self.config.cache,
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
                            self.config.cache,
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
                            self.config.cache,
                            model_id=self.config.model.id,
                            model_revision=self.config.model.revision,
                            profile=self._prefixkv_profile,
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
                        ),
                    )
                state = _state_from_packed(self.adapter, prefill.state, packed)
                packed_prefill_active_kv_bytes = _cache_payload_bytes(
                    self.adapter,
                    packed.snapshot,
                )
                if method is MosaicKVMethod.MOSAICKV_FULL and plan is not None:
                    repair_state = RepairCacheState.from_construction(
                        full_state,
                        plan.construction,
                    )
                if plan is not None:
                    trace.update(_plan_trace(plan))
                elif baseline_plan is not None:
                    trace.update(_baseline_plan_trace(baseline_plan))
                elif lookm_plan is not None:
                    trace.update(_lookm_plan_trace(lookm_plan))
                elif prefixkv_plan is not None:
                    trace.update(_prefixkv_plan_trace(prefixkv_plan))
                elif vl_cache_plan is not None:
                    trace.update(_vl_cache_plan_trace(vl_cache_plan))
                trace["packed_slots"] = list(packed.slot_records)
            else:
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
                    }
                )

            tokens = [prefill.next_token_id]
            token = prefill.next_token_id
            decode_seconds = 0.0
            repair_events: list[RepairEvent] = []
            prefixkv_decode_events: list[JsonObject] = []
            for step_index in range(1, self.config.generation.max_new_tokens):
                before = _clone_decode_state(self.adapter, state)
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
                if not bool(torch.isfinite(provisional.logits).all()):
                    raise HuggingFaceRuntimeError("decode logits contain NaN or infinity")
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
                    else:
                        state = provisional.state
                    token = provisional.next_token_id
                    tokens.append(token)
                    continue

                prototype_slots = cast(
                    "tuple[tuple[int, int, int, int], ...]",
                    before.model_state.get("mosaickv_prototype_slots", ()),
                )
                validity_masks = cast(
                    "tuple[Any, ...]", before.model_state["mosaickv_validity_masks"]
                )
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
                else:
                    state = provisional.state
                    token = provisional.next_token_id
                tokens.append(token)

            token_ids = torch.cat(tokens, dim=-1)
            answer = _decode_text(self.adapter.processor, token_ids)
            live_cache = self.adapter.extract_past_key_values(state.past_key_values, clone=False)
            active_bytes = sum(
                tensor_storage_bytes(key) + tensor_storage_bytes(value)
                for key, value in self.adapter._legacy_layers(live_cache)[0]
            )
            residual_bytes = 0 if repair_state is None else repair_state.residual_storage.cpu_bytes
            total_seconds = time.perf_counter() - started
            repair_transfer = sum(event.transfer_time_ms for event in repair_events) / 1000.0
            repair_redecode = sum(event.re_decode_time_ms for event in repair_events) / 1000.0
            timings = RuntimePhaseTimings(
                prefill=prefill_seconds,
                forecast=forecast_seconds,
                graph=graph_seconds,
                utility_and_selection=selection_seconds,
                tier_construction=tier_seconds,
                cache_packing=packing_seconds,
                decode=decode_seconds,
                repair_transfer=repair_transfer,
                repair_redecode=repair_redecode,
                total=total_seconds,
            )
            trace.update(
                {
                    "status": "completed",
                    "answer": answer,
                    "generated_token_ids": [
                        int(value) for value in token_ids.detach().cpu().reshape(-1).tolist()
                    ],
                    "repair_events": [event.to_record() for event in repair_events],
                    "prefixkv_decode_events": cast("list[JsonValue]", prefixkv_decode_events),
                    "timing_breakdown": cast("JsonObject", asdict(timings)),
                    "source_prefill_kv_bytes": source_prefill_kv_bytes,
                    "logical_prefill_active_kv_bytes": logical_prefill_active_kv_bytes,
                    "packed_prefill_active_kv_bytes": packed_prefill_active_kv_bytes,
                    "packed_padding_kv_bytes": (
                        packed_prefill_active_kv_bytes - logical_prefill_active_kv_bytes
                    ),
                    "hard_active_kv_budget_bytes": (
                        min(
                            math.ceil(source_prefill_kv_bytes * self.config.cache.retention_ratio),
                            self.config.cache.budget_value,
                        )
                        if self.config.cache.budget_unit is BudgetUnit.BYTES
                        else None
                    ),
                    "active_kv_bytes": active_bytes,
                    "residual_kv_bytes": residual_bytes,
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
                    active_kv_bytes=active_bytes,
                    residual_kv_bytes=residual_bytes,
                    peak_gpu_memory=peak_memory,
                    repair_count=sum(event.triggered for event in repair_events),
                    repaired_bytes=sum(event.restored_bytes for event in repair_events),
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
