from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

import mosaickv.backends.hf_runtime as hf_runtime
from mosaickv.adapters.huggingface import (
    AdapterCapabilities,
    CachedKeyState,
    CacheLayerSnapshot,
    CacheSnapshot,
    DecodeState,
    QueryVectorState,
)
from mosaickv.backends import build_compression_plan
from mosaickv.backends.hf_runtime import (
    _attention_output_error,
    _attention_outputs,
    _compact_trace_records,
    _dense_physical_cache_budget,
    _gather_dense_heads,
    _logit_kl,
    _plan_has_live_residual_repair,
    _refined_budget_candidates,
    _trace_catalog,
)
from mosaickv.cache_state import FullKVState, Modality, ModalitySpan
from mosaickv.config import (
    CacheConfig,
    ForecastingConfig,
    ResidualConfig,
    RunConfig,
    SelectionConfig,
    UtilityConfig,
    synthetic_smoke_config,
)
from mosaickv.forecasting import QueryForecast, build_query_forecast
from mosaickv.types import BudgetUnit, ForecastMode, MosaicKVMethod


def _full() -> FullKVState:
    key = np.arange(8 * 4, dtype=np.float32).reshape(1, 1, 8, 4) + 1
    value = np.flip(key, axis=-1).copy()
    return FullKVState.from_tensors(
        ((key, value),),
        block_size=1,
        mandatory_logical_positions=(0, 7),
        cached_key_state=CachedKeyState.NOT_APPLICABLE,
    )


def _forecast() -> QueryForecast:
    queries = np.arange(1 * 2 * 8 * 4, dtype=np.float32).reshape(1, 2, 8, 4) + 1
    return build_query_forecast(
        (queries,),
        (),
        (1,),
        ForecastingConfig(
            mode=ForecastMode.PROMPT_WINDOW,
            prompt_window=4,
            draft_steps=0,
            centroid_count=2,
        ),
        original_logical_sequence_length=8,
        draft_cache_isolated=True,
    )


def _capabilities(*, safe: bool) -> AdapterCapabilities:
    return AdapterCapabilities(
        model_family="synthetic",
        architectures=("Synthetic",),
        attention_implementations=("eager",),
        image=True,
        multi_image=True,
        video=True,
        cache_classes=("tuple",),
        cache_sequence_dimension=-2,
        cached_key_state=(CachedKeyState.NOT_APPLICABLE if safe else CachedKeyState.POST_ROPE),
        query_vector_state=QueryVectorState.Q_PROJ_PRE_ROPE,
        supports_prototype_merge=safe,
        supports_residual_repair=safe,
    )


def _config(method: MosaicKVMethod, ratio: float) -> RunConfig:
    return replace(
        synthetic_smoke_config(),
        method=method,
        cache=CacheConfig(8, BudgetUnit.BLOCKS, ratio, 1),
        forecasting=ForecastingConfig(
            mode=ForecastMode.PROMPT_WINDOW,
            prompt_window=4,
            draft_steps=0,
            centroid_count=2,
        ),
        utility=UtilityConfig(lambda_q=1.0, lambda_v=0.0, lambda_o=0.0),
        selection=SelectionConfig(
            lambda_g=-0.25,
            lambda_m=-0.25,
            stop_on_nonpositive_gain=False,
        ),
        residual=ResidualConfig(require_pinned_memory=False),
    )


@pytest.mark.parametrize(
    "method",
    (
        MosaicKVMethod.MOSAICKV_EXACT,
        MosaicKVMethod.MOSAICKV_PROTO,
        MosaicKVMethod.MOSAICKV_FULL,
    ),
)
def test_tiny_synthetic_cache_plans_every_runtime_method(method: MosaicKVMethod) -> None:
    full = _full()
    plan = build_compression_plan(
        full,
        _forecast(),
        {node_id: float(8 - node_id) for node_id in range(8)},
        _capabilities(safe=True),
        _config(method, 0.5),
    )

    assert plan.method is method
    assert plan.construction.state.statistics.active_kv_bytes <= full.active_bytes
    assert plan.selection.budget_spent <= plan.active_budget_value
    assert {0, 7} <= set(plan.selection.selected_node_ids)


def test_no_forecast_control_accepts_uniform_attention_without_query_object() -> None:
    full = _full()
    plan = build_compression_plan(
        full,
        None,
        {node_id: 1.0 for node_id in range(8)},
        _capabilities(safe=False),
        _config(MosaicKVMethod.MOSAICKV_EXACT, 0.5),
        attention_provenance="declared_no_forecast_uniform_position_control_v1",
    )

    assert plan.forecast is None
    assert plan.utilities.attention_provenance.startswith("declared_no_forecast")


def test_post_rope_proto_and_full_are_explicit_exact_safety_fallbacks() -> None:
    full = _full()
    for method in (MosaicKVMethod.MOSAICKV_PROTO, MosaicKVMethod.MOSAICKV_FULL):
        plan = build_compression_plan(
            full,
            _forecast(),
            {node_id: float(8 - node_id) for node_id in range(8)},
            _capabilities(safe=False),
            _config(method, 0.5),
        )
        assert plan.effective_method.endswith("mosaickv_exact_safety_fallback")
        assert not plan.construction.prototypes
        assert not plan.construction.residual_storage.payloads
        assert not _plan_has_live_residual_repair(method, plan)


def test_retention_one_is_exact_and_active_bytes_are_monotonic() -> None:
    full = _full()
    active: list[int] = []
    for ratio in (0.25, 0.5, 0.75, 1.0):
        plan = build_compression_plan(
            full,
            _forecast(),
            {node_id: 1.0 for node_id in range(8)},
            _capabilities(safe=False),
            _config(MosaicKVMethod.MOSAICKV_EXACT, ratio),
        )
        active.append(plan.construction.state.statistics.active_kv_bytes)
        if ratio == 1.0:
            plan.construction.state.reconstruct_full_state(full)
            assert not plan.construction.prototypes
            assert not plan.construction.residual_storage.payloads
    assert active == sorted(active)


def test_dense_physical_byte_budget_quantizes_and_constrains_selection() -> None:
    full = _full()
    config = replace(
        _config(MosaicKVMethod.MOSAICKV_EXACT, 0.25),
        cache=CacheConfig(full.active_bytes // 4, BudgetUnit.BYTES, 0.25, 1),
    )
    dense = _dense_physical_cache_budget(full, config)

    assert dense is not None
    assert dense.prompt_capacity == 2
    assert dense.target_bytes == full.active_bytes // 4
    plan = build_compression_plan(
        full,
        _forecast(),
        {node_id: float(8 - node_id) for node_id in range(8)},
        _capabilities(safe=False),
        replace(config, cache=replace(config.cache, budget_value=dense.target_bytes)),
        position_limits=dense.position_limits,
    )
    assert plan.selection.selected_node_ids == (0, 7)
    assert plan.selection.active_bytes == dense.target_bytes


def test_prototype_search_handles_large_integer_byte_budget_lattice() -> None:
    key = np.arange(8 * 64, dtype=np.float32).reshape(1, 1, 8, 64) + 1
    full = FullKVState.from_tensors(
        ((key, np.flip(key, axis=-1).copy()),),
        block_size=1,
        mandatory_logical_positions=(0, 7),
        cached_key_state=CachedKeyState.NOT_APPLICABLE,
    )
    config = replace(
        _config(MosaicKVMethod.MOSAICKV_PROTO, 0.5),
        cache=CacheConfig(full.active_bytes, BudgetUnit.BYTES, 0.5, 1),
    )

    plan = build_compression_plan(
        full,
        _forecast(),
        {node_id: float(8 - node_id) for node_id in range(8)},
        _capabilities(safe=True),
        config,
    )

    assert plan.active_budget_value > 128
    assert plan.construction.active_device_bytes <= plan.active_budget_value


def test_full_method_materializes_prototypes_and_residuals_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    implementation = hf_runtime.construct_three_tier_cache

    def counted_construction(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return implementation(*args, **kwargs)

    monkeypatch.setattr(hf_runtime, "construct_three_tier_cache", counted_construction)
    plan = build_compression_plan(
        _full(),
        _forecast(),
        {node_id: float(8 - node_id) for node_id in range(8)},
        _capabilities(safe=True),
        _config(MosaicKVMethod.MOSAICKV_FULL, 0.5),
    )

    assert plan.construction.mode.value == "three_tier"
    assert plan.construction.prototypes
    assert plan.construction.residual_storage.payloads
    assert calls == 1


def test_prototype_budget_refinement_scans_only_transition_interval() -> None:
    assert _refined_budget_candidates(100, 60, (16, 16, 8)) == (92, 84, 76, 68)
    bounded = _refined_budget_candidates(10_000, 0, (1,), maximum_points=8)
    assert len(bounded) == 8
    assert bounded == tuple(sorted(set(bounded), reverse=True))


def test_dense_physical_byte_budget_supports_fixed_multitoken_blocks() -> None:
    key = np.arange(16 * 4, dtype=np.float32).reshape(1, 1, 16, 4) + 1
    full = FullKVState.from_tensors(
        ((key, np.flip(key, axis=-1).copy()),),
        block_size=4,
        mandatory_logical_positions=(15,),
        cached_key_state=CachedKeyState.NOT_APPLICABLE,
    )
    config = replace(
        _config(MosaicKVMethod.MOSAICKV_EXACT, 0.5),
        cache=CacheConfig(full.active_bytes, BudgetUnit.BYTES, 0.5, 4),
    )

    dense = _dense_physical_cache_budget(full, config)

    assert dense is not None
    assert dense.prompt_capacity == 8
    assert dense.target_bytes == full.active_bytes // 2
    assert dense.position_limits == {(0, 0): 8}
    assert dense.bytes_per_position == full.active_bytes // 16
    assert dense.minimum_block_representable_fill(4) == (
        dense.target_bytes - 3 * dense.bytes_per_position
    )


def test_dense_physical_byte_budget_aligns_capacity_after_mandatory_fragments() -> None:
    key = np.arange(65 * 4, dtype=np.float32).reshape(1, 1, 65, 4) + 1
    full = FullKVState.from_tensors(
        ((key, np.flip(key, axis=-1).copy()),),
        modality_spans=(
            ModalitySpan(0, 48, Modality.IMAGE),
            ModalitySpan(48, 49, Modality.TEXT),
            ModalitySpan(49, 65, Modality.IMAGE),
        ),
        block_size=16,
        mandatory_logical_positions=(48,),
        cached_key_state=CachedKeyState.NOT_APPLICABLE,
    )
    requested_capacity = 48
    bytes_per_position = full.active_bytes // 65
    config = replace(
        _config(MosaicKVMethod.MOSAICKV_EXACT, requested_capacity / 65),
        cache=CacheConfig(
            requested_capacity * bytes_per_position,
            BudgetUnit.BYTES,
            requested_capacity / 65,
            16,
        ),
    )

    dense = _dense_physical_cache_budget(full, config)

    assert dense is not None
    assert dense.prompt_capacity == 33
    assert dense.target_bytes == 33 * bytes_per_position
    assert dense.target_bytes <= config.cache.budget_value

    plan = build_compression_plan(
        full,
        _forecast(),
        {node_id: float(len(full.blocks) - node_id) for node_id in range(len(full.blocks))},
        _capabilities(safe=False),
        replace(config, cache=replace(config.cache, budget_value=dense.target_bytes)),
        position_limits=dense.position_limits,
    )
    selected_positions = sum(block.position_count for block in plan.selection.selected_blocks)
    assert selected_positions == dense.prompt_capacity
    assert plan.selection.active_bytes == dense.target_bytes


def test_dense_mask_free_selection_excludes_optional_partial_fragments() -> None:
    key = np.arange(64 * 4, dtype=np.float32).reshape(1, 1, 64, 4) + 1
    full = FullKVState.from_tensors(
        ((key, np.flip(key, axis=-1).copy()),),
        modality_spans=(
            ModalitySpan(0, 15, Modality.IMAGE),
            ModalitySpan(15, 48, Modality.TEXT),
            ModalitySpan(48, 64, Modality.IMAGE),
        ),
        block_size=16,
        mandatory_logical_positions=(63,),
        cached_key_state=CachedKeyState.NOT_APPLICABLE,
    )
    config = replace(
        _config(MosaicKVMethod.MOSAICKV_EXACT, 0.75),
        cache=CacheConfig(full.active_bytes, BudgetUnit.BYTES, 0.75, 16),
    )
    dense = _dense_physical_cache_budget(full, config)

    assert dense is not None
    assert dense.prompt_capacity == 48
    attention = {
        node_id: (100.0 if block.position_count < 16 else float(len(full.blocks) - node_id))
        for node_id, block in enumerate(full.blocks)
    }
    plan = build_compression_plan(
        full,
        _forecast(),
        attention,
        _capabilities(safe=False),
        replace(config, cache=replace(config.cache, budget_value=dense.target_bytes)),
        position_limits=dense.position_limits,
    )

    assert sum(block.position_count for block in plan.selection.selected_blocks) == 48
    assert all(
        block.mandatory or block.position_count == 16 for block in plan.selection.selected_blocks
    )


def test_fidelity_formulas_match_known_attention_and_logit_patterns() -> None:
    values = torch.tensor([[[[1.0, 0.0], [0.0, 2.0]]]])
    snapshot = CacheSnapshot(
        (CacheLayerSnapshot(values.clone(), values, -2),),
        tuple,
        "legacy_tuple",
        2,
        CachedKeyState.NOT_APPLICABLE,
    )

    class Adapter:
        @staticmethod
        def extract_past_key_values(value: object, *, clone: bool = True) -> object:
            del clone
            return value

    state = DecodeState(snapshot, torch.ones((1, 2)), 2, 2, 2, ())
    output = _attention_outputs(
        Adapter(),
        (torch.tensor([[[[0.25, 0.75]]]]),),
        state,
    )
    assert torch.allclose(output[0], torch.tensor([[[0.25, 1.5]]]))
    assert _attention_output_error(output, output) == pytest.approx(0.0)

    reference = torch.tensor([[3.0, 1.0, -2.0]])
    assert _logit_kl(reference, reference) == pytest.approx(0.0)
    assert _logit_kl(reference, torch.tensor([[1.0, 3.0, -2.0]])) > 0


@pytest.mark.parametrize("head_axis,sequence_axis", ((1, 2), (2, 1)))
def test_vectorized_dense_head_gather_preserves_layout_padding_and_tail(
    head_axis: int,
    sequence_axis: int,
) -> None:
    canonical = torch.arange(1 * 2 * 6 * 3, dtype=torch.float32).reshape(1, 2, 6, 3)
    tensor = canonical if head_axis == 1 else canonical.movedim((1, 2), (2, 1))
    packed, validity = _gather_dense_heads(
        tensor,
        head_axis=head_axis,
        sequence_axis=sequence_axis,
        positions=((4, 1), (0, 3, 2)),
        prompt_capacity=4,
        tail_tensor=tensor,
        old_prompt_capacity=5,
    )
    canonical_packed = packed if head_axis == 1 else packed.movedim((2, 1), (1, 2))
    expected = torch.zeros((1, 2, 5, 3))
    expected[:, 0, 0, :] = canonical[:, 0, 4, :]
    expected[:, 0, 1, :] = canonical[:, 0, 1, :]
    expected[:, 0, 4, :] = canonical[:, 0, 5, :]
    expected[:, 1, 0, :] = canonical[:, 1, 0, :]
    expected[:, 1, 1, :] = canonical[:, 1, 3, :]
    expected[:, 1, 2, :] = canonical[:, 1, 2, :]
    expected[:, 1, 4, :] = canonical[:, 1, 5, :]
    assert torch.equal(canonical_packed, expected)
    assert torch.equal(
        validity,
        torch.tensor([[True, True, False, False, True], [True, True, True, False, True]]),
    )


def test_fidelity_attention_accepts_installed_dynamic_cache_protocol() -> None:
    values = torch.tensor([[[[1.0, 0.0], [0.0, 2.0]]]])

    class DynamicCacheLike:
        def to_legacy_cache(self) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
            return ((values.clone(), values),)

    class Adapter:
        @staticmethod
        def extract_past_key_values(value: object, *, clone: bool = True) -> object:
            assert clone is False
            return value

        @staticmethod
        def _legacy_layers(
            cache: DynamicCacheLike,
        ) -> tuple[tuple[tuple[torch.Tensor, torch.Tensor], ...], str]:
            return cache.to_legacy_cache(), "cache_object"

        @staticmethod
        def _cache_sequence_dimension(_value: torch.Tensor) -> int:
            return -2

    state = DecodeState(DynamicCacheLike(), torch.ones((1, 2)), 2, 2, 2, ())
    output = _attention_outputs(
        Adapter(),
        (torch.tensor([[[[0.25, 0.75]]]]),),
        state,
    )

    assert torch.allclose(output[0], torch.tensor([[[0.25, 1.5]]]))


def test_compact_trace_preserves_counts_hashes_and_generated_tokens() -> None:
    payload = {
        "selected_blocks": [{"node_id": 2}, {"node_id": 7}],
        "graph_edges": [{"source": 2, "target": 7}],
        "packed_slots": [{"slot": 0}],
        "forecast_statistics": {"heads": [{"layer": 0, "kv_head": 0}]},
        "generated_token_ids": [4, 5, 6],
        "repair_events": [{"triggered": False}],
    }

    _compact_trace_records(payload)  # type: ignore[arg-type]

    for key, count in (
        ("selected_blocks", 2),
        ("graph_edges", 1),
        ("packed_slots", 1),
    ):
        summary = payload[key]
        assert isinstance(summary, dict)
        assert summary["storage"] == "omitted_non_diagnostic"
        assert summary["record_count"] == count
        assert len(summary["sha256"]) == 64
    forecast = payload["forecast_statistics"]
    assert isinstance(forecast, dict)
    assert forecast["heads"]["record_count"] == 1
    assert payload["generated_token_ids"] == [4, 5, 6]
    assert payload["repair_events"] == [{"triggered": False}]


def test_streaming_trace_catalog_matches_post_materialization_compaction() -> None:
    records = [
        {"node_id": 2, "weight": 0.25, "type": "fallback"},
        {"node_id": 7, "weight": 0.75, "type": "semantic"},
    ]
    payload = {"graph_edges": list(records)}
    _compact_trace_records(payload)  # type: ignore[arg-type]

    assert _trace_catalog(iter(records), compact=True) == payload["graph_edges"]
    assert _trace_catalog(iter(records), compact=False) == records
    assert _trace_catalog(iter(()), compact=True) == []
