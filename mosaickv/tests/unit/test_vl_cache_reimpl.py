from __future__ import annotations

import numpy as np
import pytest

from mosaickv.baselines import (
    VLCacheReimplementationError,
    VLCacheSensitivityPoint,
    analyze_vl_cache_sensitivity,
    assert_vl_cache_calibration_disjoint,
    build_vl_cache_reimpl_plan,
    paper_layer_retention_ratios,
    post_vision_attention_statistics,
    threshold_filter,
    vl_cache_runtime_payloads,
)
from mosaickv.cache_state import FullKVState, Modality, ModalitySpan
from mosaickv.config import CacheConfig, VLCacheConfig
from mosaickv.types import BudgetUnit


def _state(*, heads: int = 1) -> FullKVState:
    layers = []
    for layer in range(2):
        key = np.arange(heads * 6 * 2, dtype=np.float32).reshape(1, heads, 6, 2)
        key = key + 100 * layer
        layers.append((key, key + 1))
    return FullKVState.from_tensors(
        tuple(layers),
        modality_spans=(
            ModalitySpan(0, 1, Modality.TEXT),
            ModalitySpan(1, 3, Modality.IMAGE, image_index=0),
            ModalitySpan(3, 6, Modality.TEXT),
        ),
        block_size=1,
        mandatory_logical_positions=(5,),
    )


def _attention(*, query_heads: int = 1) -> tuple[np.ndarray, ...]:
    dense = np.zeros((1, query_heads, 6, 6), dtype=np.float32)
    sparse = np.zeros_like(dense)
    for query in range(6):
        dense[:, :, query, : query + 1] = 1.0 / (query + 1)
        sparse[:, :, query, query] = 1.0
    return dense, sparse


def _config(*, sparsity_threshold: float = 0.01) -> VLCacheConfig:
    return VLCacheConfig(enabled=True, sparsity_threshold=sparsity_threshold)


def test_threshold_filter_matches_paper_equation_one() -> None:
    attention = np.asarray([[0.80, 0.10, 0.079, 0.0]], dtype=np.float32)
    filtered = threshold_filter(attention, 0.1)

    np.testing.assert_allclose(filtered, [[0.80, 0.10, 0.0, 0.0]])


def test_sparsity_and_layer_budget_match_paper_formulas() -> None:
    statistics = post_vision_attention_statistics(
        _attention(),
        kv_heads_by_layer=(1, 1),
        post_vision_start=3,
        threshold=0.1,
    )

    assert statistics.layer_sparsity == pytest.approx((0.0, 0.8))
    assert statistics.layer_density == pytest.approx((1.0, 0.2))
    raw, clipped = paper_layer_retention_ratios(
        statistics.layer_sparsity,
        retention_ratio=0.5,
        minimum=0.01,
        maximum=1.0,
    )
    assert raw == pytest.approx((5 / 6, 1 / 6))
    assert clipped == pytest.approx(raw)


def test_denser_layer_receives_more_budget_and_hard_total_is_exact() -> None:
    state = _state(heads=2)
    cache = CacheConfig(12, BudgetUnit.BLOCKS, 0.5, 1)
    first = build_vl_cache_reimpl_plan(
        state,
        _attention(query_heads=4),
        _config(sparsity_threshold=0.1),
        cache,
    )
    second = build_vl_cache_reimpl_plan(
        state,
        _attention(query_heads=4),
        _config(sparsity_threshold=0.1),
        cache,
    )

    assert tuple(layer.retained_positions_per_head for layer in first.layers) == (5, 1)
    assert first.active_slots == first.target_slots == 12
    assert first.retained_bytes == state.active_bytes // 2
    assert first.layers == second.layers
    assert all(head.selected_physical_positions == (5,) for head in first.layers[1].heads)
    payloads, records = vl_cache_runtime_payloads(first)
    assert len(payloads) == 4
    assert len(records) == first.active_slots
    assert all(record["tier"] == "vl_cache_reimpl_exact" for record in records)


def test_retention_one_is_an_exact_no_transformation_path() -> None:
    state = _state(heads=2)
    plan = build_vl_cache_reimpl_plan(
        state,
        _attention(query_heads=4),
        _config(),
        CacheConfig(24, BudgetUnit.BLOCKS, 1.0, 1),
    )

    assert plan.is_retention_one
    assert plan.active_slots == plan.source_slots == 24
    assert plan.retained_bytes == state.active_bytes
    payloads, _ = vl_cache_runtime_payloads(plan)
    for layer in range(2):
        for head in range(2):
            entries = payloads[(layer, head)]
            packed_key = np.concatenate([entry[3] for entry in entries], axis=2)
            packed_value = np.concatenate([entry[4] for entry in entries], axis=2)
            np.testing.assert_array_equal(packed_key, state.layers[layer].key[:, head : head + 1])
            np.testing.assert_array_equal(
                packed_value, state.layers[layer].value[:, head : head + 1]
            )


def test_calibration_and_evaluation_sample_ids_cannot_overlap() -> None:
    with pytest.raises(VLCacheReimplementationError, match="overlap"):
        assert_vl_cache_calibration_disjoint(("cal-0", "shared"), ("shared", "eval-0"))

    with pytest.raises(VLCacheReimplementationError, match="overlap"):
        analyze_vl_cache_sensitivity(
            _state(),
            _attention(),
            _config(),
            CacheConfig(6, BudgetUnit.BLOCKS, 0.5, 1),
            calibration_sample_id="shared",
            calibration_sample_ids=("shared",),
            evaluation_sample_ids=("shared",),
        )


def test_sensitivity_grid_is_labeled_structural_and_reproducible() -> None:
    def run() -> tuple[VLCacheSensitivityPoint, ...]:
        return analyze_vl_cache_sensitivity(
            _state(),
            _attention(),
            _config(),
            CacheConfig(6, BudgetUnit.BLOCKS, 0.5, 1),
            calibration_sample_id="cal-0",
            calibration_sample_ids=("cal-0",),
            evaluation_sample_ids=("eval-0",),
            sparsity_thresholds=(0.01, 0.1),
            recent_window_fractions=(0.0, 0.1),
            max_post_vision_queries=(None,),
        )

    first = run()
    second = run()

    assert first == second
    assert len(first) == 4
    assert all(point.to_json_object()["method"] == "vl_cache_reimpl" for point in first)
    assert all(
        point.to_json_object()["measurement_type"]
        == "synthetic_or_calibration_structural_diagnostic"
        for point in first
    )


def test_no_media_is_rejected_instead_of_guessing_post_vision_boundary() -> None:
    key = np.zeros((1, 1, 6, 2), dtype=np.float32)
    state = FullKVState.from_tensors(
        ((key, key.copy()), (key.copy(), key.copy())),
        block_size=1,
        mandatory_logical_positions=(5,),
    )
    with pytest.raises(VLCacheReimplementationError, match="image/video span"):
        build_vl_cache_reimpl_plan(
            state,
            _attention(),
            _config(),
            CacheConfig(6, BudgetUnit.BLOCKS, 0.5, 1),
        )
