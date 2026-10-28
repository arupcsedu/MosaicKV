"""Synthetic CPU optimization fixtures; no pretrained-quality claims."""

import pytest
import torch

from mosaickv.prototypes.attention_compaction import AttentionCompactionPolicy
from mosaickv.prototypes.constrained_compaction import (
    ConstrainedValuePolicy,
    attention_output,
    fit_component,
    solve_bounded_values,
)


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_saturated_one_dimensional_optimum_obeys_original_four_x_bound(dtype):
    values = torch.ones((1, 1, 1), dtype=dtype)
    solved, record = solve_bounded_values(
        torch.ones(1, 1, 1),
        torch.full((1, 1, 1), 9.0),
        values,
        torch.zeros((1, 1), dtype=torch.bool),
        AttentionCompactionPolicy(True, value_ridge=0),
        ConstrainedValuePolicy(),
    )
    assert 3.9 <= float(solved[0, 0, 0]) <= 4
    assert float(record["constrained_maximum_value_norm_ratio"].max()) <= 4
    assert record["constraints_enforced_during_optimization"]
    assert torch.equal(values, torch.ones_like(values))
    assert record["regularized_objective_after_cast"] < record["regularized_objective_initial"]


def test_rank_deficient_system_and_zero_values_cannot_create_unbounded_coefficients():
    x = torch.tensor([[[0.5, 0.5, 0.0], [0.5, 0.5 + 1e-12, 0.0]]], dtype=torch.float64)
    residual = torch.tensor([[[1.0], [-1.0]]], dtype=torch.float64)
    values = torch.tensor([[[1.0], [1.0], [0.0]]], dtype=torch.float64)
    protected = torch.tensor([[True, False, False]])
    solved, record = solve_bounded_values(
        x,
        residual,
        values,
        protected,
        AttentionCompactionPolicy(True, value_ridge=0),
        ConstrainedValuePolicy(),
    )
    assert torch.isfinite(solved).all() and float(solved.abs().max()) <= 4
    assert torch.equal(solved[protected], values[protected])
    assert solved[0, 2, 0] == 0 and float(record["constrained_maximum_value_norm_ratio"].max()) <= 4


@pytest.mark.parametrize("ratio", [1.0, 1.1, 4.0])
def test_stored_fp16_bounds_protection_no_input_mutation_and_no_gradients(ratio):
    rng = torch.Generator().manual_seed(52)
    values = torch.randn((3, 5, 7), generator=rng, dtype=torch.float16).requires_grad_()
    original = values.detach().clone()
    matrix = torch.randn((3, 4, 5), generator=rng, dtype=torch.float64)
    residual = torch.randn((3, 4, 7), generator=rng, dtype=torch.float64) * 100
    protected = torch.tensor([[True, False, False, False, True]]).expand(3, -1)
    core = AttentionCompactionPolicy(True, maximum_value_norm_ratio=ratio)
    solved, _ = solve_bounded_values(
        matrix, residual, values, protected, core, ConstrainedValuePolicy()
    )
    assert (solved.double().norm(dim=-1) <= ratio * original.double().norm(dim=-1)).all()
    assert torch.equal(solved[protected], original[protected])
    assert torch.equal(values, original) and values.grad is None and not solved.requires_grad


def fixture():
    a = torch.full((2, 4, 1), 0.25, dtype=torch.float64)
    values = torch.ones((2, 1, 2), dtype=torch.float16)
    target = torch.full((2, 4, 2), 1.5, dtype=torch.float64)
    h = torch.full((2, 4), 0.25, dtype=torch.float64)
    numerator = torch.zeros((2, 4, 2), dtype=torch.float64)
    protected = torch.zeros((2, 1), dtype=torch.bool)
    return a, values, target, h, numerator, protected


def test_all_components_have_distinct_semantics_and_exact_duplicate_key_joint_solution():
    core = AttentionCompactionPolicy(True, value_ridge=0, mass_ridge=0)
    result = {
        mode: fit_component(*fixture(), mode, core, ConstrainedValuePolicy())
        for mode in ("anchor", "mass_only", "value_only", "joint")
    }
    assert (result["anchor"].log_bias == 0).all() and not result["anchor"].admitted.any()
    assert torch.equal(result["mass_only"].values, fixture()[1])
    assert (result["value_only"].log_bias == 0).all()
    for mode in ("value_only", "joint"):
        assert result[mode].admitted.all()
        output = attention_output(
            fixture()[0], result[mode].values, result[mode].log_bias, fixture()[3], fixture()[4]
        )
        torch.testing.assert_close(output, fixture()[2], atol=1e-5, rtol=1e-5)
    assert float(result["joint"].values[0, 0, 0]) == pytest.approx(2.0, abs=1e-5)


def test_batch_solution_matches_separate_systems_and_is_deterministic():
    core = AttentionCompactionPolicy(True)
    args = fixture()
    batch = fit_component(*args, "joint", core, ConstrainedValuePolicy())
    again = fit_component(*args, "joint", core, ConstrainedValuePolicy())
    assert torch.equal(batch.values, again.values) and torch.equal(batch.log_bias, again.log_bias)
    for route in (0, 1):
        separate = fit_component(
            *(a[route : route + 1] for a in args), "joint", core, ConstrainedValuePolicy()
        )
        assert torch.equal(batch.values[route : route + 1], separate.values)
        assert torch.equal(batch.log_bias[route : route + 1], separate.log_bias)


def test_invalid_policy_evidence_and_annotations_are_rejected():
    for iterations in (0, True, 1025):
        with pytest.raises(ValueError, match="iterations"):
            ConstrainedValuePolicy(iterations)
    args = fixture()
    with pytest.raises(ValueError, match="enabled"):
        fit_component(*args, "joint", AttentionCompactionPolicy(), ConstrainedValuePolicy())
    with pytest.raises(ValueError, match="unknown"):
        fit_component(
            *args, "oracle_future", AttentionCompactionPolicy(True), ConstrainedValuePolicy()
        )
    with pytest.raises(ValueError, match="finite"):
        fit_component(
            args[0] * float("nan"),
            *args[1:],
            "joint",
            AttentionCompactionPolicy(True),
            ConstrainedValuePolicy(),
        )
    with pytest.raises(TypeError):
        fit_component(
            *args,
            "joint",
            AttentionCompactionPolicy(True),
            ConstrainedValuePolicy(),
            benchmark_answers=["forbidden"],
        )
    with pytest.raises(ValueError, match="mass biases"):
        fit_component(
            *args,
            "joint",
            AttentionCompactionPolicy(True),
            ConstrainedValuePolicy(),
            mass_bias=torch.full((2, 1), 10.0),
        )


def test_all_protected_entries_are_identity_and_cannot_be_fitted():
    a, v, y, h, n, p = fixture()
    fitted = fit_component(
        a, v, y, h, n, ~p, "joint", AttentionCompactionPolicy(True), ConstrainedValuePolicy()
    )
    assert torch.equal(fitted.values, v) and (fitted.log_bias == 0).all()
    assert not fitted.admitted.any()
