from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from mosaickv.adapters.huggingface import (
    CachedKeyState,
    CacheLayerSnapshot,
    CacheSnapshot,
    DecodeState,
)
from mosaickv.backends.hf_runtime import maintain_prefixkv_decode_cache
from mosaickv.baselines import (
    PrefixKVCalibrationObservation,
    PrefixKVOfflineProfile,
    PrefixKVReimplementationError,
    build_prefixkv_reimpl_plan,
    generate_prefixkv_profile,
    load_prefixkv_profile,
    prefixkv_attention_scores,
    prefixkv_runtime_payloads,
    search_prefixkv_layer_sizes,
)
from mosaickv.cache_state import FullKVState
from mosaickv.config import CacheConfig, PrefixKVConfig
from mosaickv.types import BudgetUnit, PrefixKVProfileMode


def _state(*, layers: int = 2, sequence: int = 6, heads: int = 1) -> FullKVState:
    payloads = []
    for layer in range(layers):
        key = np.arange(heads * sequence * 2, dtype=np.float32).reshape(1, heads, sequence, 2)
        key = key + 100 * layer
        value = key + 1
        payloads.append((key, value))
    return FullKVState.from_tensors(
        tuple(payloads),
        block_size=1,
        mandatory_logical_positions=(0, sequence - 1),
    )


def _attention() -> tuple[np.ndarray, ...]:
    # After protected positions 0 and 5, layer 0 favors position 1 and layer 1
    # favors position 4. The selection is shared by every KV head.
    return (
        np.asarray([[[[1.0, 9.0, 2.0, 3.0, 4.0, 1.0]]]], dtype=np.float32),
        np.asarray([[[[1.0, 2.0, 3.0, 4.0, 9.0, 1.0]]]], dtype=np.float32),
    )


def test_fixed_global_budget_sums_exactly_and_selection_is_deterministic() -> None:
    state = _state(heads=2)
    config = PrefixKVConfig(
        enabled=True,
        profile_mode=PrefixKVProfileMode.FIXED_GLOBAL,
        start_size=1,
        protect_size=1,
    )
    cache = CacheConfig(12, BudgetUnit.BLOCKS, 0.5, 1)
    first = build_prefixkv_reimpl_plan(
        state,
        _attention(),
        config,
        cache,
        model_id="llava-hf/llava-1.5-7b-hf",
        model_revision="a" * 40,
    )
    second = build_prefixkv_reimpl_plan(
        state,
        _attention(),
        config,
        cache,
        model_id="llava-hf/llava-1.5-7b-hf",
        model_revision="a" * 40,
    )

    assert first.layer_cache_sizes == (3, 3)
    assert first.active_slots == 12
    assert first.active_slots / first.source_slots == 0.5
    assert first.layers[0].selected_physical_positions == (0, 1, 5)
    assert first.layers[1].selected_physical_positions == (0, 4, 5)
    assert first.layers == second.layers
    assert first.retained_bytes == state.active_bytes // 2
    payloads, records = prefixkv_runtime_payloads(first)
    assert len(payloads) == 4
    assert len(records) == first.active_slots


def test_offline_profile_controls_adaptive_layer_sizes(tmp_path: Path) -> None:
    profile = PrefixKVOfflineProfile(
        model_id="llava-hf/llava-1.5-7b-hf",
        model_revision="a" * 40,
        target_retention_ratio=0.5,
        layer_forget_ratios=(0.2, 0.8),
        calibration_dataset_id="calibration",
        calibration_dataset_revision="b" * 40,
        calibration_split="train[:2]",
        calibration_sample_ids=("cal-0", "cal-1"),
        calibration_seed=7,
        start_size=1,
        protect_size=1,
    )
    path = profile.write(tmp_path / "profile.json")
    loaded = load_prefixkv_profile(
        path,
        model_id=profile.model_id,
        model_revision=profile.model_revision,
        target_retention_ratio=0.5,
        start_size=1,
        protect_size=1,
    )
    plan = build_prefixkv_reimpl_plan(
        _state(),
        _attention(),
        PrefixKVConfig(enabled=True, profile_path=str(path)),
        CacheConfig(6, BudgetUnit.BLOCKS, 0.5, 1),
        model_id=profile.model_id,
        model_revision=profile.model_revision,
        profile=loaded,
    )

    assert sum(plan.layer_cache_sizes) == 6
    assert plan.layer_cache_sizes == (4, 2)
    assert plan.profile is not None
    assert plan.profile.profile_sha256 == profile.profile_sha256
    plan.profile.assert_evaluation_disjoint(("eval-0",))
    with pytest.raises(PrefixKVReimplementationError, match="overlap"):
        plan.profile.assert_evaluation_disjoint(("cal-1",))


def test_profile_generation_rejects_calibration_evaluation_leakage() -> None:
    observations = (
        PrefixKVCalibrationObservation("cal-0", ((9.0, 1.0, 1.0, 1.0),) * 2),
        PrefixKVCalibrationObservation("cal-1", ((1.0, 1.0, 1.0, 9.0),) * 2),
    )
    with pytest.raises(PrefixKVReimplementationError, match="overlap"):
        generate_prefixkv_profile(
            observations,
            model_id="model",
            model_revision="a" * 40,
            dataset_id="dataset",
            dataset_revision="b" * 40,
            calibration_split="train",
            evaluation_sample_ids=("cal-1",),
            retention_ratio=0.75,
            seed=0,
        )
    profile = generate_prefixkv_profile(
        observations,
        model_id="model",
        model_revision="a" * 40,
        dataset_id="dataset",
        dataset_revision="b" * 40,
        calibration_split="train",
        evaluation_sample_ids=("eval-0",),
        retention_ratio=0.75,
        seed=0,
    )
    assert profile.calibration_sample_ids == ("cal-0", "cal-1")
    assert len(profile.layer_forget_ratios) == 2


def test_global_prefix_search_respects_protected_tokens_and_exact_sum() -> None:
    sizes = search_prefixkv_layer_sizes(
        (
            (10.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            (2.0, 2.0, 2.0, 2.0, 2.0, 2.0),
        ),
        retention_ratio=0.5,
        start_size=1,
        protect_size=1,
    )
    assert sum(sizes) == 6
    assert all(size >= 2 for size in sizes)


def test_attention_pooling_matches_official_dtype_operation_order() -> None:
    attention = np.asarray([[[[0.1000, 0.2000]], [[0.1001, 0.1999]]]], dtype=np.float16)
    expected = attention.mean(axis=1).astype(np.float32).sum(axis=-2, dtype=np.float32)[0]
    scores = prefixkv_attention_scores((attention,))
    assert np.array_equal(np.asarray(scores[0], dtype=np.float32), expected)


def test_official_raw_profile_is_loaded_as_unverifiable_calibration(tmp_path: Path) -> None:
    path = tmp_path / "official.json"
    path.write_text(json.dumps([0.25, 0.75]), encoding="utf-8")
    profile = load_prefixkv_profile(
        path,
        model_id="llava-hf/llava-1.5-7b-hf",
        model_revision="a" * 40,
        target_retention_ratio=0.5,
        start_size=1,
        protect_size=1,
    )
    assert profile.source_kind == "official_repository_config_calibration_ids_unavailable"
    assert not profile.calibration_sample_ids


def test_retention_one_is_exact_and_does_not_need_attention() -> None:
    state = _state()
    plan = build_prefixkv_reimpl_plan(
        state,
        (),
        PrefixKVConfig(enabled=True, profile_mode=PrefixKVProfileMode.FIXED_GLOBAL),
        CacheConfig(len(state.blocks), BudgetUnit.BLOCKS, 1.0, 1),
        model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        model_revision="a" * 40,
    )
    assert plan.layer_cache_sizes == (6, 6)
    assert plan.retained_bytes == state.active_bytes
    assert plan.implementation_label == "generalized_prefixkv_reimpl"


def test_decode_maintenance_prunes_at_most_one_fixed_offset_per_layer() -> None:
    torch = pytest.importorskip("torch")
    source = FullKVState.from_tensors(
        tuple(
            (
                torch.arange(12, dtype=torch.float32).reshape(1, 1, 6, 2) + layer * 100,
                torch.arange(12, dtype=torch.float32).reshape(1, 1, 6, 2) + layer * 1000,
            )
            for layer in range(2)
        ),
        block_size=1,
        mandatory_logical_positions=(0, 5),
    )
    plan = build_prefixkv_reimpl_plan(
        source,
        tuple(torch.from_numpy(value) for value in _attention()),
        PrefixKVConfig(enabled=True, profile_mode=PrefixKVProfileMode.FIXED_GLOBAL),
        CacheConfig(6, BudgetUnit.BLOCKS, 0.5, 1),
        model_id="llava-hf/llava-1.5-7b-hf",
        model_revision="a" * 40,
    )
    # Three selected prompt positions plus two generated positions. At logical
    # length eight, a 0.5 layer recipe is one slot over budget.
    live_layers = tuple(
        CacheLayerSnapshot(
            torch.arange(10, dtype=torch.float32).reshape(1, 1, 5, 2) + layer * 100,
            torch.arange(10, dtype=torch.float32).reshape(1, 1, 5, 2) + layer * 1000,
            2,
        )
        for layer in range(2)
    )
    cache = tuple((layer.key, layer.value) for layer in live_layers)

    class FakeAdapter:
        def extract_past_key_values(self, value: object) -> CacheSnapshot:
            pairs = cast("tuple[tuple[Any, Any], ...]", value)
            return CacheSnapshot(
                tuple(CacheLayerSnapshot(key.clone(), val.clone(), 2) for key, val in pairs),
                tuple,
                "tuple",
                5,
                CachedKeyState.NOT_APPLICABLE,
            )

        def inject_past_key_values(self, snapshot: CacheSnapshot) -> object:
            return tuple((layer.key, layer.value) for layer in snapshot.layers)

    state = DecodeState(
        past_key_values=cache,
        attention_mask=torch.ones((1, 5), dtype=torch.long),
        active_cache_length=5,
        logical_sequence_length=8,
        next_decode_position=8,
        modality_map=(),
        model_state={
            "mosaickv_validity_masks": tuple(
                torch.ones((1, 5), dtype=torch.bool) for _ in range(2)
            ),
            "mosaickv_prompt_capacity": 3,
        },
    )
    updated, events = maintain_prefixkv_decode_cache(
        FakeAdapter(),  # type: ignore[arg-type]
        state,
        plan,
        step_index=2,
    )
    assert updated.active_cache_length == 4
    assert all(event["removed_physical_position"] == 1 for event in events)
    assert all(event["active_positions_after"] == 4 for event in events)
