#!/usr/bin/env python3
"""Build the non-measured synthetic baseline byte-budget audit artifact.

The generated Parquet file is a structural validation artifact.  It contains
no model-quality, latency, or other measured paper result.
"""

from __future__ import annotations

import argparse
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mosaickv.adapters.huggingface import (
    AdapterCapabilities,
    CachedKeyState,
    QueryVectorState,
)
from mosaickv.backends import build_compression_plan
from mosaickv.baselines import (
    build_exact_baseline_plan,
    build_lookm_reimpl_plan,
    build_prefixkv_reimpl_plan,
    build_vl_cache_reimpl_plan,
)
from mosaickv.cache_state import FullKVState, Modality, ModalitySpan
from mosaickv.config import (
    CacheConfig,
    ForecastingConfig,
    GraphConfig,
    LookMConfig,
    PrefixKVConfig,
    PrototypeConfig,
    RepairConfig,
    ResidualConfig,
    RunConfig,
    SelectionConfig,
    UtilityConfig,
    VLCacheConfig,
    synthetic_smoke_config,
)
from mosaickv.forecasting import QueryForecast, build_query_forecast
from mosaickv.types import (
    BudgetUnit,
    ForecastMode,
    LookMMergeStrategy,
    MosaicKVMethod,
    PrefixKVProfileMode,
    RepairPolicy,
)

_METHODS = (
    MosaicKVMethod.FULL_KV,
    MosaicKVMethod.RANDOM_KV,
    MosaicKVMethod.PROMPT_ATTENTION_TOPK,
    MosaicKVMethod.LOOKM_REIMPL,
    MosaicKVMethod.PREFIXKV_REIMPL,
    MosaicKVMethod.VL_CACHE_REIMPL,
    MosaicKVMethod.MOSAICKV_EXACT,
    MosaicKVMethod.MOSAICKV_PROTO,
    MosaicKVMethod.MOSAICKV_FULL,
)
_LAYERS = 2
_KV_HEADS = 2
_SEQUENCE_LENGTH = 12
_HEAD_DIMENSION = 4
_DTYPE = np.dtype(np.float32)


@dataclass(frozen=True, slots=True)
class BudgetValidationRow:
    """One structural byte-accounting outcome, never a measured result."""

    schema_version: int
    evidence_type: str
    measurement_type: str
    workload_id: str
    comparison_group: str
    model_id: str
    dtype: str
    layers: int
    kv_heads_per_layer: int
    sequence_length: int
    head_dimension: int
    requested_method: str
    effective_method: str
    requested_retention_ratio: float
    source_prefill_kv_bytes: int
    primary_budget_bytes: int
    logical_prefill_active_kv_bytes: int
    packed_prefill_active_kv_bytes: int
    packed_padding_kv_bytes: int
    packed_budget_delta_bytes: int
    logical_budget_compliant: bool
    packed_budget_compliant: bool
    exact_primary_byte_match: bool
    requested_algorithm_exercised: bool
    eligible_for_byte_matched_comparison: bool
    configured_budget_unit: str
    configured_budget_value: int
    block_size: int
    mandatory_policy: str
    status: str
    note: str


def _base_tensors() -> tuple[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]], ...]:
    layers = []
    size = _KV_HEADS * _SEQUENCE_LENGTH * _HEAD_DIMENSION
    for layer in range(_LAYERS):
        start = 1 + layer * size
        key = np.arange(start, start + size, dtype=_DTYPE).reshape(
            1, _KV_HEADS, _SEQUENCE_LENGTH, _HEAD_DIMENSION
        )
        value = np.flip(key, axis=-1).copy() + np.float32(0.25)
        layers.append((key, value))
    return tuple(layers)


def _full_state(mandatory: tuple[int, ...]) -> FullKVState:
    return FullKVState.from_tensors(
        _base_tensors(),
        modality_spans=(
            ModalitySpan(0, 2, Modality.TEXT),
            ModalitySpan(2, 6, Modality.IMAGE, image_index=0),
            ModalitySpan(6, _SEQUENCE_LENGTH, Modality.TEXT),
        ),
        token_ids=np.arange(_SEQUENCE_LENGTH, dtype=np.int64),
        block_size=1,
        mandatory_logical_positions=mandatory,
        cached_key_state=CachedKeyState.POST_ROPE,
    )


def _attention() -> tuple[np.ndarray[Any, Any], ...]:
    result: list[np.ndarray[Any, Any]] = []
    for layer in range(_LAYERS):
        tensor = np.zeros((1, _KV_HEADS, _SEQUENCE_LENGTH, _SEQUENCE_LENGTH), dtype=np.float32)
        for head in range(_KV_HEADS):
            for query in range(_SEQUENCE_LENGTH):
                if layer == 0:
                    scores = np.arange(1, query + 2, dtype=np.float32)
                else:
                    scores = np.full(query + 1, 1e-5, dtype=np.float32)
                    scores[(query + head) % (query + 1)] = 1.0
                tensor[0, head, query, : query + 1] = scores / scores.sum()
        result.append(tensor)
    return tuple(result)


def _forecast() -> QueryForecast:
    queries = tuple(
        (
            np.arange(
                1 + layer * _KV_HEADS * _SEQUENCE_LENGTH * _HEAD_DIMENSION,
                1 + (layer + 1) * _KV_HEADS * _SEQUENCE_LENGTH * _HEAD_DIMENSION,
                dtype=np.float32,
            ).reshape(1, _KV_HEADS, _SEQUENCE_LENGTH, _HEAD_DIMENSION)
        )
        for layer in range(_LAYERS)
    )
    return build_query_forecast(
        queries,
        (),
        (_KV_HEADS,) * _LAYERS,
        ForecastingConfig(
            mode=ForecastMode.PROMPT_WINDOW,
            prompt_window=4,
            draft_steps=0,
            centroid_count=2,
        ),
        original_logical_sequence_length=_SEQUENCE_LENGTH,
        draft_cache_isolated=True,
    )


def _post_rope_capabilities() -> AdapterCapabilities:
    return AdapterCapabilities(
        model_family="synthetic_post_rope_audit",
        architectures=("SyntheticVLM",),
        attention_implementations=("eager",),
        image=True,
        multi_image=True,
        video=False,
        cache_classes=("tuple",),
        cache_sequence_dimension=-2,
        cached_key_state=CachedKeyState.POST_ROPE,
        query_vector_state=QueryVectorState.Q_PROJ_PRE_ROPE,
        supports_prototype_merge=False,
        supports_residual_repair=False,
    )


def _prompt_scores(full_state: FullKVState) -> dict[int, float]:
    return {
        node_id: float(
            (_SEQUENCE_LENGTH - block.physical_cache_indices[0])
            * (1 + block.layer)
            * (1 + block.kv_head)
        )
        for node_id, block in enumerate(full_state.blocks)
    }


def _packed_bytes(
    full_state: FullKVState,
    selected: dict[tuple[int, int], tuple[int, ...]],
) -> int:
    capacity = max(map(len, selected.values()))
    return sum(capacity * layer.byte_size // layer.sequence_length for layer in full_state.layers)


def _simple_positions(plan: Any) -> dict[tuple[int, int], tuple[int, ...]]:
    result: dict[tuple[int, int], list[int]] = {}
    for block in plan.selection.selected_blocks:
        result.setdefault((block.layer, block.kv_head), []).extend(block.physical_cache_indices)
    return {identity: tuple(sorted(values)) for identity, values in result.items()}


def _mosaic_positions(plan: Any) -> dict[tuple[int, int], tuple[int, ...]]:
    result: dict[tuple[int, int], list[int]] = {}
    for block in plan.construction.state.exact.blocks:
        result.setdefault((block.layer, block.kv_head), []).extend(block.physical_cache_indices)
    for record in plan.construction.prototypes:
        result.setdefault((record.layer, record.kv_head), []).append(record.anchor_logical_position)
    return {identity: tuple(sorted(values)) for identity, values in result.items()}


def _mosaic_config(
    method: MosaicKVMethod,
    ratio: float,
    byte_budget: int,
) -> RunConfig:
    base = synthetic_smoke_config()
    uses_prototypes = method in {MosaicKVMethod.MOSAICKV_PROTO, MosaicKVMethod.MOSAICKV_FULL}
    uses_repair = method is MosaicKVMethod.MOSAICKV_FULL
    return RunConfig(
        model=base.model,
        dataset=base.dataset,
        execution=base.execution,
        generation=base.generation,
        cache=CacheConfig(byte_budget, BudgetUnit.BYTES, ratio, 1),
        method=method,
        forecasting=ForecastingConfig(
            mode=ForecastMode.PROMPT_WINDOW,
            prompt_window=4,
            draft_steps=0,
            centroid_count=2,
        ),
        graph=GraphConfig(max_neighbors=4),
        utility=UtilityConfig(lambda_q=1.0, lambda_v=0.0, lambda_o=0.0),
        selection=SelectionConfig(
            lambda_g=-0.25,
            lambda_m=-0.25,
            stop_on_nonpositive_gain=False,
        ),
        prototypes=PrototypeConfig(enabled=uses_prototypes),
        residual=ResidualConfig(enabled=uses_repair, require_pinned_memory=False),
        repair=RepairConfig(
            enabled=uses_repair,
            policy=(RepairPolicy.ENTROPY if uses_repair else RepairPolicy.NONE),
            max_blocks_per_step=1 if uses_repair else 0,
        ),
    )


def _validate_method(method: MosaicKVMethod, ratio: float) -> BudgetValidationRow:
    source_state = _full_state((_SEQUENCE_LENGTH - 1,))
    source_bytes = source_state.active_bytes
    primary_budget = math.floor(source_bytes * ratio + 1e-12)
    slot_bytes = source_state.blocks[0].byte_size
    if primary_budget % slot_bytes:
        raise RuntimeError("synthetic audit budget is not representable in whole token/head slots")
    slot_budget = primary_budget // slot_bytes
    mandatory_policy = "terminal_prompt_token"
    configured_unit = BudgetUnit.BYTES
    configured_value = primary_budget
    status = "validated"
    note = ""

    if method is MosaicKVMethod.FULL_KV:
        logical_bytes = source_bytes
        packed_bytes = source_bytes
        effective = method.value
        exercised = ratio == 1.0
        if not exercised:
            status = "unsupported_configuration"
            note = "full_kv is defined only at retention 1.0"
    elif method in {MosaicKVMethod.RANDOM_KV, MosaicKVMethod.PROMPT_ATTENTION_TOPK}:
        cache = CacheConfig(primary_budget, BudgetUnit.BYTES, ratio, 1)
        plan = build_exact_baseline_plan(
            source_state,
            method,
            cache,
            seed=17,
            prompt_attention_by_node=(
                _prompt_scores(source_state)
                if method is MosaicKVMethod.PROMPT_ATTENTION_TOPK
                else None
            ),
        )
        logical_bytes = plan.selection.active_bytes
        packed_bytes = _packed_bytes(source_state, _simple_positions(plan))
        effective = (
            f"{method.value}__retention_one_exact" if plan.state.is_retention_one else method.value
        )
        exercised = ratio < 1.0
    elif method.is_lookm_reimplementation:
        recent_ratio = ratio / 2
        mandatory_count = math.floor(_SEQUENCE_LENGTH * recent_ratio + 1e-12)
        full_state = _full_state(tuple(range(_SEQUENCE_LENGTH - mandatory_count, _SEQUENCE_LENGTH)))
        plan = build_lookm_reimpl_plan(
            full_state,
            _attention(),
            LookMConfig(
                enabled=True,
                recent_ratio=recent_ratio,
                important_ratio=ratio - recent_ratio,
                merge_strategy=LookMMergeStrategy.PIVOTAL,
            ),
            CacheConfig(slot_budget, BudgetUnit.BLOCKS, ratio, 1),
        )
        selected = {
            (head.layer, head.kv_head): head.selected_physical_positions for head in plan.heads
        }
        logical_bytes = plan.active_bytes
        packed_bytes = _packed_bytes(full_state, selected)
        effective = method.value
        exercised = ratio < 1.0
        configured_unit = BudgetUnit.BLOCKS
        configured_value = slot_budget
        mandatory_policy = "lookm_recent_window"
    elif method.is_prefixkv_reimplementation:
        full_state = _full_state((0, _SEQUENCE_LENGTH - 1))
        plan = build_prefixkv_reimpl_plan(
            full_state,
            _attention(),
            PrefixKVConfig(
                enabled=True,
                profile_mode=PrefixKVProfileMode.FIXED_GLOBAL,
                start_size=1,
                protect_size=1,
            ),
            CacheConfig(primary_budget, BudgetUnit.BYTES, ratio, 1),
            model_id="llava-hf/llava-1.5-7b-hf",
            model_revision="synthetic-structural-audit",
        )
        selected = {
            (layer.layer, head): layer.selected_physical_positions
            for layer in plan.layers
            for head in range(full_state.layers[layer.layer].kv_heads)
        }
        logical_bytes = plan.retained_bytes
        packed_bytes = _packed_bytes(full_state, selected)
        effective = plan.implementation_label
        exercised = ratio < 1.0
        mandatory_policy = "prefixkv_start_and_protected_tail"
    elif method.is_vl_cache_reimplementation:
        full_state = source_state
        plan = build_vl_cache_reimpl_plan(
            full_state,
            _attention(),
            VLCacheConfig(enabled=True, recent_window_fraction=0.1),
            CacheConfig(slot_budget, BudgetUnit.BLOCKS, ratio, 1),
        )
        selected = {
            (head.layer, head.kv_head): head.selected_physical_positions
            for layer in plan.layers
            for head in layer.heads
        }
        logical_bytes = plan.retained_bytes
        packed_bytes = _packed_bytes(full_state, selected)
        effective = method.value
        exercised = ratio < 1.0
        configured_unit = BudgetUnit.BLOCKS
        configured_value = slot_budget
    else:
        full_state = source_state
        plan = build_compression_plan(
            full_state,
            _forecast(),
            _prompt_scores(full_state),
            _post_rope_capabilities(),
            _mosaic_config(method, ratio, primary_budget),
        )
        logical_bytes = plan.construction.state.statistics.active_kv_bytes
        packed_bytes = _packed_bytes(full_state, _mosaic_positions(plan))
        effective = plan.effective_method
        exercised = ratio < 1.0 and effective == method.value
        if ratio < 1.0 and not exercised:
            note = "post-RoPE adapter capability forced the documented exact-only safety fallback"

    logical_compliant = logical_bytes <= primary_budget
    packed_compliant = packed_bytes <= primary_budget
    exact_match = packed_bytes == primary_budget
    eligible = status == "validated" and exact_match and (ratio == 1.0 or exercised)
    if not note and logical_compliant and not packed_compliant:
        note = "uniform HF packing padding exceeded the logical selection budget"
    elif not note and packed_compliant and not exact_match:
        note = "realized physical storage is below, but not equal to, the primary byte budget"
    return BudgetValidationRow(
        schema_version=1,
        evidence_type="synthetic_structural_validation",
        measurement_type="validation_smoke_not_measured_result",
        workload_id="synthetic_post_rope_2l_2h_s12_d4_fp32_v1",
        comparison_group=f"retention_{ratio:.1f}",
        model_id="synthetic/post-rope-cache",
        dtype="float32",
        layers=_LAYERS,
        kv_heads_per_layer=_KV_HEADS,
        sequence_length=_SEQUENCE_LENGTH,
        head_dimension=_HEAD_DIMENSION,
        requested_method=method.value,
        effective_method=effective,
        requested_retention_ratio=ratio,
        source_prefill_kv_bytes=source_bytes,
        primary_budget_bytes=primary_budget,
        logical_prefill_active_kv_bytes=logical_bytes,
        packed_prefill_active_kv_bytes=packed_bytes,
        packed_padding_kv_bytes=packed_bytes - logical_bytes,
        packed_budget_delta_bytes=packed_bytes - primary_budget,
        logical_budget_compliant=logical_compliant,
        packed_budget_compliant=packed_compliant,
        exact_primary_byte_match=exact_match,
        requested_algorithm_exercised=exercised,
        eligible_for_byte_matched_comparison=eligible,
        configured_budget_unit=configured_unit.value,
        configured_budget_value=configured_value,
        block_size=1,
        mandatory_policy=mandatory_policy,
        status=status,
        note=note,
    )


def build_rows() -> tuple[BudgetValidationRow, ...]:
    """Return deterministic rows for a compressed and retention-one stratum."""

    return tuple(_validate_method(method, ratio) for ratio in (0.5, 1.0) for method in _METHODS)


def write_parquet(rows: tuple[BudgetValidationRow, ...], output: Path) -> Path:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "building the fairness artifact requires the evaluation environment with pyarrow"
        ) from error

    output.parent.mkdir(parents=True, exist_ok=True)
    records = [asdict(row) for row in rows]
    table = pa.Table.from_pylist(records)
    metadata = dict(table.schema.metadata or {})
    metadata.update(
        {
            b"mosaickv.artifact_type": b"baseline_budget_structural_validation",
            b"mosaickv.measurement_type": b"validation_smoke_not_measured_result",
            b"mosaickv.warning": (
                b"synthetic structural byte accounting; not a measured paper result"
            ),
            b"mosaickv.primary_budget": b"packed_prefill_active_kv_bytes",
        }
    )
    table = table.replace_schema_metadata(metadata)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".parquet",
            delete=False,
        ) as handle:
            temporary = handle.name
        pq.write_table(table, temporary, compression="zstd")
        os.replace(temporary, output)
        output.chmod(0o644)
    except BaseException:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
        raise
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "results"
        / "baseline_budget_validation.parquet",
    )
    args = parser.parse_args()
    destination = write_parquet(build_rows(), args.output.resolve())
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
