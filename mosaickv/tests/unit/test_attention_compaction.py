"""Algebraic CPU fixtures only; never measured VLM quality or memory results."""

from dataclasses import replace

import pytest
import torch

from mosaickv.prototypes.attention_compaction import (
    AttentionCompactionPolicy,
    compact_head,
    fit_batched_mass_bias,
    fit_mass_bias,
    plan_compaction_capacity,
)


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def duplicate_fixture(dtype=torch.float64):
    # Three identical original keys, one representative. One unmodified
    # appended history token proves why output-only reconstruction is not enough.
    keys = torch.zeros(3, 2, dtype=dtype)
    values = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=dtype)
    p = torch.full((1, 4), 0.25, dtype=torch.float64)
    args = dict(
        keys=keys[:1],
        values=values[:1],
        selected_attention=p[:, :1],
        full_outputs=p[:, :3] @ values.double(),
        history_mass=p[:, 3],
        history_numerator=torch.zeros(1, 2, dtype=torch.float64),
        protected=torch.tensor([False]),
        logical_positions=(0,),
        source_positions=3,
        policy=AttentionCompactionPolicy(True, mass_ridge=0, value_ridge=0),
        evidence_kind="draft_rollout",
    )
    return keys, values, args


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_duplicate_keys_preserve_mass_output_and_new_appended_history(dtype):
    source_k, source_v, args = duplicate_fixture(dtype)
    result = compact_head(**args)
    assert result.construction_accepted and result.requires_model_validation
    assert not result.diagnostics["benchmark_quality_validated"]
    assert result.diagnostics["exact_tier"] is False
    assert torch.equal(result.keys, source_k[:1])
    torch.testing.assert_close(
        result.log_bias.double().exp(),
        torch.tensor([3.0], dtype=torch.float64),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        result.values.double(), torch.full((1, 2), 2.0, dtype=torch.float64), atol=1e-5, rtol=1e-5
    )
    for history_count in (0, 1, 2, 5):
        expected = source_v.double().sum(0) / (3 + history_count)
        predicted = (result.log_bias.double().exp()[:, None] * result.values.double()).sum(0) / (
            result.log_bias.double().exp().sum() + history_count
        )
        torch.testing.assert_close(predicted, expected, atol=1e-5, rtol=1e-5)
    # Fitting a value alone to the one-history-token query gives V=3;
    # adding two history tokens then yields 1 rather than the correct 1.2.
    assert abs(3 / 3 - 6 / 5) > 0.1
    assert result.storage_bytes == sum(
        t.untyped_storage().nbytes() for t in (result.keys, result.values, result.log_bias)
    )
    assert result.keys.untyped_storage().data_ptr() != source_k.untyped_storage().data_ptr()
    assert (
        result.values.untyped_storage().nbytes() == source_v[:1].numel() * source_v.element_size()
    )
    assert torch.equal(source_v, torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=dtype))


def test_protected_values_keys_and_positions_remain_bitwise_exact():
    keys = torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    values = torch.tensor([[1.0, 2.0], [3.0, 0.0], [0.0, 4.0], [1.0, 1.0]], dtype=torch.float64)
    queries = torch.tensor([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    history_key = torch.tensor([[2.0, 0.0]], dtype=torch.float64)
    history_value = torch.tensor([[2.0, -1.0]], dtype=torch.float64)
    probabilities = (queries @ torch.cat((keys, history_key)).T).softmax(-1)
    selected = [0, 2, 3]
    target = probabilities @ torch.cat((values, history_value))
    protected = torch.tensor([False, False, True])
    result = compact_head(
        keys[selected],
        values[selected],
        probabilities[:, selected],
        target,
        probabilities[:, -1],
        probabilities[:, -1:] @ history_value,
        protected,
        (10, 30, 40),
        source_positions=4,
        policy=AttentionCompactionPolicy(True, mass_iterations=1024, mass_ridge=0, value_ridge=0),
        evidence_kind="hybrid",
    )
    assert result.construction_accepted
    assert torch.equal(result.keys, keys[selected]) and result.logical_positions == (10, 30, 40)
    assert torch.equal(result.values[protected], values[selected][protected])
    assert (result.log_bias[protected] == 0).all()
    assert result.diagnostics["attempted_reference_output_mse_after"] < 1e-8


def test_retention_one_and_disabled_mode_do_not_fit_or_transform_values():
    _, _, args = duplicate_fixture()
    disabled = compact_head(**dict(args, policy=AttentionCompactionPolicy()))
    assert not disabled.construction_accepted
    assert torch.equal(disabled.values, args["values"]) and (disabled.log_bias == 0).all()
    identity = compact_head(**dict(args, retention_one=True, source_positions=1))
    assert identity.diagnostics["reason"] == "retention_one_identity_no_fit"
    assert torch.equal(identity.keys, args["keys"]) and torch.equal(identity.values, args["values"])
    assert (identity.log_bias == 0).all()
    with pytest.raises(ValueError, match="every source"):
        compact_head(**dict(args, retention_one=True))


def test_failed_norm_guard_rolls_back_the_entire_bias_and_value_transaction():
    _, _, args = duplicate_fixture()
    rejected = compact_head(
        **dict(args, policy=replace(args["policy"], maximum_value_norm_ratio=1.01))
    )
    assert not rejected.construction_accepted
    assert torch.equal(rejected.values, args["values"]) and (rejected.log_bias == 0).all()
    assert (
        rejected.diagnostics["returned_reference_output_mse"]
        == rejected.diagnostics["reference_output_mse_before"]
    )


@pytest.mark.parametrize(
    "change",
    [
        dict(mass_iterations=0),
        dict(mass_iterations=True),
        dict(mass_ridge=float("nan")),
        dict(maximum_value_norm_ratio=0.5),
        dict(maximum_mass_multiplier=float("inf")),
        dict(svd_relative_tolerance=0),
        dict(enabled=1),
    ],
)
def test_invalid_policy_fails_clearly(change):
    with pytest.raises(ValueError):
        AttentionCompactionPolicy(**change)


def test_invalid_evidence_and_label_arguments_are_rejected():
    _, _, args = duplicate_fixture()
    for change in (
        dict(evidence_kind="oracle_future_queries"),
        dict(logical_positions=(-1,)),
        dict(full_outputs=args["full_outputs"] * float("nan")),
        dict(history_mass=torch.tensor([2.0], dtype=torch.float64)),
    ):
        with pytest.raises(ValueError):
            compact_head(**dict(args, **change))
    with pytest.raises(TypeError):
        compact_head(**dict(args, benchmark_answers=["not_an_input"]))


def test_capacity_charges_bias_and_mask_backing_storage_and_never_exceeds_cap():
    base = dict(
        layers=32,
        kv_heads=32,
        query_heads=32,
        key_dimension=128,
        value_dimension=128,
        cache_element_bytes=2,
        history_positions=16,
        mandatory_positions=2,
    )
    small = plan_compaction_capacity(cap_bytes=28 * 2**20, **base)
    assert small.prompt_positions == 39
    assert small.lifetime_kv_bytes == int(27.5 * 2**20)
    assert small.total_persistent_bytes <= 28 * 2**20
    assert (
        small.total_persistent_bytes
        == small.lifetime_kv_bytes + small.log_bias_bytes + small.additive_mask_bytes
    )
    assert small.total_persistent_bytes + small.unused_bytes == 28 * 2**20
    assert small.total_persistent_bytes + (32 * 32 * (512 + 4 + 4)) > 28 * 2**20
    assert [
        plan_compaction_capacity(cap_bytes=cap * 2**20, **base).prompt_positions
        for cap in (28, 32, 40)
    ] == sorted(
        plan_compaction_capacity(cap_bytes=cap * 2**20, **base).prompt_positions
        for cap in (28, 32, 40)
    )
    with pytest.raises(ValueError, match="cannot fit"):
        plan_compaction_capacity(cap_bytes=8 * 2**20, **base)


def test_mass_solver_determinism_and_disabled_gate():
    _, _, args = duplicate_fixture()
    a = fit_mass_bias(
        args["selected_attention"], args["history_mass"], args["protected"], args["policy"]
    )
    b = fit_mass_bias(
        args["selected_attention"], args["history_mass"], args["protected"], args["policy"]
    )
    assert torch.equal(a[0], b[0]) and a[1] == b[1]
    with pytest.raises(ValueError, match="disabled"):
        fit_mass_bias(
            args["selected_attention"],
            args["history_mass"],
            args["protected"],
            AttentionCompactionPolicy(),
        )


def test_batched_solver_matches_individual_heads_with_different_protected_masks():
    generator = torch.Generator().manual_seed(17)
    a = torch.rand((4, 7, 6), generator=generator, dtype=torch.float64) / 20
    h = torch.rand((4, 7), generator=generator, dtype=torch.float64) / 10
    protected = torch.tensor(
        [[True, False, False, False, False, True], [False] * 6, [True] * 6, [False] * 6]
    )
    a[3] = 0  # Zero-rank evidence must stay finite, not divide by zero.
    policy = AttentionCompactionPolicy(True)
    batched = fit_batched_mass_bias(a, h, protected, policy)
    for i in range(4):
        individual, _ = fit_mass_bias(a[i], h[i], protected[i], policy)
        torch.testing.assert_close(batched[i], individual, atol=1e-6, rtol=1e-6)
    assert (batched[protected] == 0).all()
    assert torch.isfinite(batched).all() and (batched[3] == 0).all()
    with pytest.raises(ValueError, match="batched"):
        fit_batched_mass_bias(a, h[:, :2], protected, policy)


def test_finite_inputs_with_overflowing_cache_cast_fail_closed_and_json_safe():
    import json

    _, _, args = duplicate_fixture(torch.float16)
    args["values"] = torch.full((1, 2), 60000, dtype=torch.float16)
    args["full_outputs"] = torch.full((1, 2), 120000, dtype=torch.float64)
    args["policy"] = replace(args["policy"], maximum_value_norm_ratio=8)
    result = compact_head(**args)
    assert not result.construction_accepted
    assert torch.equal(result.values, args["values"]) and (result.log_bias == 0).all()
    json.dumps(result.diagnostics, allow_nan=False)


def test_stored_bias_obeys_the_cap_after_fp32_rounding():
    a = torch.tensor([[1e-5]], dtype=torch.float64)
    h, protected = torch.zeros(1, dtype=torch.float64), torch.tensor([False])
    bias, diagnostics = fit_mass_bias(
        a, h, protected, AttentionCompactionPolicy(True, mass_ridge=0)
    )
    assert float(bias.double().exp().max()) <= 64
    assert diagnostics["maximum_multiplier"] <= 64


def test_model_inputs_and_gradients_are_not_mutated_or_trained():
    _, _, args = duplicate_fixture(torch.float64)
    args["values"] = args["values"].clone().requires_grad_(True)
    args["keys"] = args["keys"].clone().requires_grad_(True)
    original = args["values"].detach().clone()
    result = compact_head(**args)
    assert torch.equal(args["values"], original) and args["values"].grad is None
    assert args["keys"].grad is None
    assert all(not t.requires_grad for t in (result.keys, result.values, result.log_bias))
