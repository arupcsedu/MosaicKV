"""Synthetic CPU fixtures; no pretrained-quality or measured-results claims."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from mosaickv.cache_state import FullKVState, Modality, ModalitySpan
from mosaickv.config import BoundedDecodeConfig, ConfigurationError, canonical_config, load_config
from mosaickv.selection.evidence_layer_allocation import (
    allocate_layer_slots,
    build_evidence_layer_plan,
    evidence_ranking,
    reconstruction_guard,
    reconstruction_mse,
    validate_reconstruction_trace,
)


def test_reconstruction_accounts_for_retained_softmax_renormalization():
    mass = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    values = torch.tensor([[[0.0], [2.0], [4.0], [6.0]]])
    error = reconstruction_mse(mass, values, ((0, 3),))
    assert error.item() == pytest.approx((4.8 - 4.0) ** 2)
    assert reconstruction_mse(mass * 10, values, ((0, 3),)).item() == pytest.approx(error.item())


@pytest.mark.parametrize("invalid", ["zero", "nan", "negative", "empty", "duplicate"])
def test_reconstruction_rejects_undefined_or_invalid_subsets(invalid):
    mass = torch.tensor([[0.0, 1.0]])
    values = torch.zeros(1, 2, 1)
    positions = ((0,),)
    if invalid == "nan":
        values[0, 1, 0] = float("nan")
    elif invalid == "negative":
        mass[0, 0] = -0.1
    elif invalid == "empty":
        positions = ((),)
    elif invalid == "duplicate":
        positions = ((1, 1),)
    with pytest.raises(ValueError):
        reconstruction_mse(mass, values, positions)


def test_guard_does_not_hide_regressed_head_behind_mean_improvement():
    before = ((1.0, 1.0), (0.0,))
    after = ((0.0, 1.1), (0.0,))
    trace = reconstruction_guard(before, after)
    assert not trace["accepted"]
    assert trace["rejected_layer_heads"] == [[0, 1]]
    assert trace["final_mse_by_layer_head"] == before
    assert trace["proposed_mse_by_layer_head"] == after
    assert trace["decision"] == "retain_same_byte_compressed_anchor"
    assert not trace["used_fullkv_fallback"]


def test_guard_accepts_improvement_and_fixed_fp32_tolerance_only():
    before = ((1.0, 0.0),)
    for after in [((0.5, 0.0),), ((1.0000005, 1e-8),)]:
        trace = reconstruction_guard(before, after)
        assert trace["accepted"]
        assert trace["final_mse_by_layer_head"] == after
    assert not reconstruction_guard(before, ((1.0, 1e-5),))["accepted"]
    with pytest.raises(ValueError, match="finite"):
        reconstruction_guard(before, ((float("nan"), 0.0),))


def test_guarded_plan_rolls_back_same_byte_anchor_and_preserves_original_values():
    generator = torch.Generator().manual_seed(7)
    layers = tuple(
        (torch.randn(1, 2, 8, 3, generator=generator), torch.randn(1, 2, 8, 3, generator=generator))
        for _ in range(2)
    )
    saved = tuple(tuple(t.clone() for t in pair) for pair in layers)
    state = FullKVState.from_tensors(layers, block_size=1, mandatory_logical_positions=(7,))
    attention = {(i, h): np.ones(8) / 8 for i in range(2) for h in range(2)}
    records = tuple(
        dict(layer=i, kv_head=h, logical_position=p, tier="exact")
        for i in range(2)
        for h in range(2)
        for p in (0, 1, 6, 7)
    )
    plan = build_evidence_layer_plan(
        state,
        attention,
        records,
        prompt_budget_bytes=state.active_bytes // 2,
        uniform_capacity=4,
        guard_reconstruction=True,
    )
    guard = plan.trace()["reconstruction_guard"]
    assert not guard["accepted"]
    assert guard["rejected_head_count"] > 0
    assert plan.positions == (((0, 1, 6, 7),) * 2,) * 2
    assert plan.capacities == (4, 4)
    assert plan.prompt_bytes == state.active_bytes // 2
    assert plan.trace()["padding_bytes"] == 0
    assert guard["anchor_positions_sha256"] == plan.trace()["selected_positions_sha256"]
    assert guard["proposed_positions_sha256"] != guard["anchor_positions_sha256"]
    assert guard["final_mse_by_layer_head"] == guard["anchor_mse_by_layer_head"]
    assert guard["host_wall_seconds_including_device_transfers"] >= 0
    import json

    serial = json.loads(json.dumps(plan.trace(), allow_nan=False))
    check = validate_reconstruction_trace(serial)
    assert check["final_mse_mean"] == check["anchor_mse_mean"]
    serial["reconstruction_guard"]["final_mse_by_layer_head"][0][0] += 0.1
    with pytest.raises(ValueError, match="inconsistent"):
        validate_reconstruction_trace(serial)
    for before, after in zip(saved, layers, strict=True):
        for a, b in zip(before, after, strict=True):
            torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_guarded_plan_accepts_a_same_byte_reconstruction_improvement():
    key = torch.arange(4, dtype=torch.float32).reshape(1, 1, 4, 1)
    state = FullKVState.from_tensors(
        ((key, torch.tensor([[[[0.0], [0.0], [1.0], [3.0]]]])),),
        block_size=1,
        mandatory_logical_positions=(0,),
    )
    records = tuple(dict(layer=0, kv_head=0, logical_position=p, tier="exact") for p in (0, 1))
    plan = build_evidence_layer_plan(
        state,
        {(0, 0): np.ones(4) / 4},
        records,
        prompt_budget_bytes=state.active_bytes // 2,
        uniform_capacity=2,
        guard_reconstruction=True,
    )
    guard = plan.trace()["reconstruction_guard"]
    assert guard["accepted"]
    assert guard["anchor_mse_by_layer_head"] == ((1.0,),)
    assert guard["final_mse_by_layer_head"] == ((0.25,),)
    assert plan.positions == (((0, 3),),)
    assert plan.prompt_bytes == state.active_bytes // 2


def test_guard_rejects_anchor_with_wrong_physical_capacity_or_missing_mandatory():
    key = torch.zeros(1, 1, 4, 1)
    state = FullKVState.from_tensors(((key, key),), block_size=1, mandatory_logical_positions=(3,))
    records = tuple(dict(layer=0, kv_head=0, logical_position=p, tier="exact") for p in (0, 1))
    with pytest.raises(ValueError, match="anchor must satisfy"):
        build_evidence_layer_plan(
            state,
            {(0, 0): np.ones(4) / 4},
            records,
            prompt_budget_bytes=state.active_bytes // 2,
            uniform_capacity=2,
            guard_reconstruction=True,
        )


def test_guarded_configuration_is_versioned_and_round_trips_without_changing_v2():
    from pathlib import Path

    from mosaickv.config import RunConfig, config_sha256

    cfg = load_config(
        Path(__file__).resolve().parents[2] / "configs/development/llava_bounded_history_v1.yaml"
    )
    v2 = replace(cfg, bounded_decode=BoundedDecodeConfig(32, 4, 0.2, "output_impact_v1"))
    guarded = replace(
        v2, bounded_decode=replace(v2.bounded_decode, layer_allocation="output_impact_guarded_v1")
    )
    assert config_sha256(v2) != config_sha256(guarded)
    assert config_sha256(RunConfig.from_mapping(canonical_config(guarded))) == config_sha256(
        guarded
    )
    assert "__guarded_output_impact_layers" in guarded.bounded_decode.effective_method(cfg.method)


def test_layer_allocator_spends_exact_bytes_and_responds_to_absolute_impact():
    counts = allocate_layer_slots(((1.0,) * 8, (10.0,) * 8), (2, 2), (6, 6), 8)
    assert counts == (2, 6)
    assert sum(counts) == 8
    assert allocate_layer_slots(((10.0,) * 8, (1.0,) * 8), (2, 2), (6, 6), 8) == (6, 2)
    assert allocate_layer_slots(((1.0,) * 8,) * 2, (2, 2), (6, 6), 8) == (6, 2)
    with pytest.raises(ValueError, match="exact global"):
        allocate_layer_slots(((1.0,) * 8,) * 2, (2, 2), (6, 6), 13)


def test_ranking_protects_mandatory_and_reserves_prompt_and_visual_evidence():
    score = np.array([0.0, 9, 8, 7, 6, 5, 4, 3, 2, 0.1])
    rank = evidence_ranking(score, (0, 9), (1, 2), ((3, 4), (5, 6)), 8)
    assert rank[:4] == (0, 9, 1, 3)
    assert sorted(rank) == list(range(10))
    assert evidence_ranking(score, (0, 9), (1, 2), ((3, 4),), 8) == rank


def test_plan_uses_no_labels_keeps_source_unchanged_and_retains_mandatory():
    key = torch.arange(32, dtype=torch.float32).reshape(1, 2, 8, 2)
    layers = ((key.clone(), key.clone()), (key.clone(), key.clone() * 10))
    saved = [(k.clone(), v.clone()) for k, v in layers]
    state = FullKVState.from_tensors(
        layers,
        block_size=1,
        mandatory_logical_positions=(0, 7),
        modality_spans=(
            ModalitySpan(0, 2, Modality.TEXT),
            ModalitySpan(2, 6, Modality.IMAGE, image_index=0),
            ModalitySpan(6, 8, Modality.TEXT),
        ),
    )
    attention = {(layer, head): np.ones(8) / 8 for layer in range(2) for head in range(2)}
    records = tuple(
        dict(layer=layer, kv_head=head, logical_position=p, tier="exact")
        for layer in range(2)
        for head in range(2)
        for p in (0, 1, 6, 7)
    )
    plan = build_evidence_layer_plan(
        state, attention, records, prompt_budget_bytes=state.active_bytes // 2, uniform_capacity=4
    )
    again = build_evidence_layer_plan(
        state, attention, records, prompt_budget_bytes=state.active_bytes // 2, uniform_capacity=4
    )
    assert plan == again
    assert plan.capacities == (2, 6)
    assert plan.prompt_bytes == state.active_bytes // 2
    for heads in plan.positions:
        assert len({len(row) for row in heads}) == 1
        for row in heads:
            assert row == tuple(sorted(set(row)))
            assert {0, 7} <= set(row)
    for original, after in zip(saved, layers, strict=True):
        for a, b in zip(original, after, strict=True):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert plan.trace()["benchmark_labels_accessed"] is False


def test_fraction_is_common_byte_driven_and_legacy_identity_is_preserved():
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "configs/development/llava_bounded_history_v1.yaml"
    cfg = load_config(path)
    assert canonical_config(cfg)["bounded_decode"] == {"generated_capacity": 32, "sink_tokens": 4}
    policy = BoundedDecodeConfig(32, 4, 0.2, "output_impact_v1")
    assert policy.resolved_capacity(40 * 2**20, 2**19) == 16
    assert policy.resolved_capacity(80 * 2**20, 2**19) == 32
    assert policy.resolved_capacity(32 * 2**20, 2**19) == 12
    assert "output_impact_layers" in policy.effective_method(cfg.method)
    assert (
        canonical_config(replace(cfg, bounded_decode=policy))["bounded_decode"]["history_fraction"]
        == 0.2
    )
    for fraction in [True, float("nan"), 0.0, 1.0, "0.2"]:
        with pytest.raises(ConfigurationError, match="history_fraction"):
            BoundedDecodeConfig(history_fraction=fraction)
