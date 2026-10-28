"""Synthetic CPU-only tests for conservative V3 and bounded submission scope."""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
import yaml

from mosaickv.prototypes.attention_compaction import AttentionCompactionPolicy
from mosaickv.prototypes.constrained_compaction import (
    ConstrainedValuePolicy,
    solve_bounded_values,
)

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def study(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "anchor_study_test", SCRIPTS / "study_anchor_compaction_v3.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize("delta_limit", [0.0, 0.01, 0.1, 0.25])
def test_actual_stored_change_and_final_value_obey_both_bounds(dtype, delta_limit):
    rng = torch.Generator().manual_seed(72)
    original = torch.randn(3, 5, 7, generator=rng, dtype=dtype)
    original[0, 2] = 0
    source = original.clone()
    matrix = torch.randn(3, 8, 5, generator=rng, dtype=torch.float64)
    residual = torch.randn(3, 8, 7, generator=rng, dtype=torch.float64) * 100
    protected = torch.tensor([[True, False, False, False, True]]).expand(3, -1)
    policy = ConstrainedValuePolicy(ridge_multiplier=10000.0, maximum_delta_norm_ratio=delta_limit)
    values, record = solve_bounded_values(
        matrix, residual, original, protected, AttentionCompactionPolicy(True), policy
    )
    norms = original.double().norm(dim=-1)
    assert ((values.double() - original.double()).norm(dim=-1) <= delta_limit * norms).all()
    assert (values.double().norm(dim=-1) <= 4 * norms).all()
    assert torch.equal(values[protected], original[protected])
    assert torch.equal(values[0, 2], original[0, 2])
    assert torch.equal(original, source) and not values.requires_grad
    assert record["relative_ridge"] == pytest.approx(0.01)
    if delta_limit == 0:
        assert torch.equal(values, original)


def test_stronger_ridge_reduces_correction_and_matches_closed_form():
    x = torch.ones(1, 1, 1, dtype=torch.float64)
    r = torch.full((1, 1, 1), 0.5, dtype=torch.float64)
    v = torch.ones(1, 1, 1, dtype=torch.float64)
    protected = torch.zeros(1, 1, dtype=torch.bool)
    core = AttentionCompactionPolicy(True, value_ridge=1e-6)
    weak, _ = solve_bounded_values(x, r, v, protected, core, ConstrainedValuePolicy())
    strong, record = solve_bounded_values(
        x, r, v, protected, core, ConstrainedValuePolicy(ridge_multiplier=1e6)
    )
    assert float(weak[0, 0, 0]) == pytest.approx(1 + 0.5 / (1 + 1e-6), abs=1e-8)
    assert float(strong[0, 0, 0]) == pytest.approx(1.25, abs=1e-8)
    assert (strong - v).norm() < (weak - v).norm()
    assert record["regularized_objective_after_cast"] <= record["regularized_objective_initial"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ridge_multiplier": 0.99},
        {"ridge_multiplier": True},
        {"ridge_multiplier": float("inf")},
        {"maximum_delta_norm_ratio": -0.1},
        {"maximum_delta_norm_ratio": 1.01},
        {"maximum_delta_norm_ratio": True},
        {"maximum_delta_norm_ratio": float("nan")},
    ],
)
def test_policy_cannot_weaken_regularization_or_accept_invalid_trust_regions(kwargs):
    with pytest.raises(ValueError):
        ConstrainedValuePolicy(**kwargs)


def test_constraint_intersection_fails_closed_when_delta_ball_not_inside_value_ball():
    with pytest.raises(ValueError, match="delta ball"):
        solve_bounded_values(
            torch.ones(1, 1, 1),
            torch.ones(1, 1, 1),
            torch.ones(1, 1, 1),
            torch.zeros(1, 1, dtype=torch.bool),
            AttentionCompactionPolicy(True, maximum_value_norm_ratio=1.0),
            ConstrainedValuePolicy(maximum_delta_norm_ratio=0.1),
        )


def fixture():
    rng = torch.Generator().manual_seed(8)
    a = torch.rand(2, 32, 4, generator=rng, dtype=torch.float64)
    a = a / a.sum(-1, keepdim=True) * 0.5
    v = torch.randn(2, 4, 3, generator=rng, dtype=torch.float16)
    y = torch.randn(2, 32, 3, generator=rng, dtype=torch.float64)
    h = torch.full((2, 32), 0.2, dtype=torch.float64)
    n = torch.zeros_like(y)
    protected = torch.tensor([[True, False, False, True]]).expand(2, -1)
    return a, v, y, h, n, protected


def test_holdout_changes_cannot_change_training_decisions_and_fields_remain_complete(study):
    arrays = fixture()
    fold = dict(train_prefixes=16, holdout_start=17, holdout_end=24)
    core = AttentionCompactionPolicy(True)
    policy = ConstrainedValuePolicy(ridge_multiplier=10000.0, maximum_delta_norm_ratio=0.1)
    first = study.evaluate_fold(arrays, fold, core, policy, chunk=2)
    changed = tuple(t.clone() for t in arrays)
    changed[2][:, 16:24] += 100
    second = study.evaluate_fold(changed, fold, core, policy, chunk=2)
    assert len(first) == 8
    for a, b in zip(first, second, strict=True):
        assert a["admitted"] == b["admitted"] and a["train_mse_after"] == b["train_mse_after"]
        assert a["maximum_value_delta_norm_ratio"] == b["maximum_value_delta_norm_ratio"]
        assert a["delta_constraint_valid"] and a["maximum_value_delta_norm_ratio"] <= 0.1
        assert a["protected_values_unchanged"] and a["protected_bias_zero"]
    assert any(
        a["holdout_mse_after"] != b["holdout_mse_after"] for a, b in zip(first, second, strict=True)
    )


def test_protocol_keeps_cases_fixed_and_cannot_launch_gpu_or_relax_guard(study):
    frozen = yaml.safe_load(study.PROTOCOL.read_text())
    study.validate_protocol(frozen)
    for key, value in (
        ("gpu_execution_authorized", True),
        ("calibration_authorized", True),
        ("fresh_cohort_authorized", True),
        ("unchanged_value_norm_limit", 5.0),
        ("future_guard_unchanged", {"horizon": 4}),
        ("cases", []),
    ):
        altered = copy.deepcopy(frozen)
        altered[key] = value
        with pytest.raises(ValueError, match="scope"):
            study.validate_protocol(altered)


def summaries(case, ratio=0.85):
    return [
        dict(
            case=case,
            component=mode,
            worker=worker,
            train_prefixes=t,
            constraints_valid=True,
            admitted_heads=100,
            holdout_mse_ratio=ratio,
        )
        for mode in ("mass_only", "value_only", "joint")
        for worker in (0, 1)
        for t in (8, 16, 24)
    ]


def test_positive_ablation_cannot_promote_negative_or_incomplete_fixed_candidate(study):
    screen = yaml.safe_load(study.PROTOCOL.read_text())["useful_screen"]
    rows = summaries("ridge_only") + summaries("delta_only") + summaries("anchor_preserving", 1.1)
    result = study.choose_candidate(rows, screen)
    assert not result["useful_bounded_fit"] and result["selected_case"] is None
    result = study.choose_candidate(summaries("anchor_preserving"), screen)
    assert result["useful_bounded_fit"] and result["selected_case"] == "anchor_preserving"
    assert not study.choose_candidate(summaries("anchor_preserving")[:-3], screen)[
        "useful_bounded_fit"
    ]


def test_corrected_scheduler_is_bound_and_preflight_never_changes_slurm(study, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("preflight must not call Slurm")

    monkeypatch.setattr(study.subprocess, "run", forbidden)
    result = study.scheduler_preflight()
    assert result["finite_duration_cases"]["30:00"] == 30
    assert result["sha256"] == study.digest(result["source"])
    assert result["no_slurm_actions"]


def test_cpu_submission_is_finite_and_has_failure_aware_analysis(study, monkeypatch, tmp_path):
    monkeypatch.setattr(study, "prepare", lambda root: (root / "plan.json").write_text("{}"))
    calls = []

    def check(args, **kwargs):
        calls.append(args)
        return str(3000 + len(calls))

    monkeypatch.setattr(study.subprocess, "check_output", check)
    study.submit(tmp_path)
    assert len(calls) == 2 and "--array=0-5%4" in calls[0]
    assert "--dependency=afterany:3001" in calls[1]
    assert all("gres" not in token for call in calls for token in call)
    record = json.loads((tmp_path / "submission.json").read_text())
    assert record["gpu_execution_authorized"] is False and not record["gpu_jobs_submitted"]
    assert not record["fresh_cohort_authorized"] and not record["calibration_authorized"]
    for name in ("anchor_compaction_cpu_v3", "anchor_compaction_analysis_v3"):
        text = (study.SOURCE / f"mosaickv/slurm/{name}.sbatch").read_text()
        assert 'CUDA_VISIBLE_DEVICES=""' in text and "--gres" not in text
        assert "cache_env.sh" in text and "squeue" not in text and "sleep " not in text
