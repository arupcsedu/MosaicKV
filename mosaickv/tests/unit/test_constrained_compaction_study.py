"""CPU study controls on synthetic fixtures, with no models or datasets."""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
import yaml

from mosaickv.prototypes.attention_compaction import AttentionCompactionPolicy
from mosaickv.prototypes.constrained_compaction import ConstrainedValuePolicy, attention_output

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


@pytest.fixture
def study(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "constrained_study_test", SCRIPTS / "study_constrained_compaction_v2.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def config(study):
    return yaml.safe_load(study.PROTOCOL.read_text())


def evidence():
    rng = torch.Generator().manual_seed(48)
    a = torch.rand(2, 32, 4, generator=rng, dtype=torch.float64)
    a = a / a.sum(-1, keepdim=True) * 0.5
    v = torch.randn(2, 4, 3, generator=rng, dtype=torch.float16)
    h = torch.full((2, 32), 0.2, dtype=torch.float64)
    n = torch.zeros(2, 32, 3, dtype=torch.float64)
    protected = torch.tensor([[True, False, False, True]]).expand(2, -1)
    target_values = v.clone()
    target_values[:, 1:3] *= 1.5
    target = attention_output(a, target_values, torch.zeros(2, 4), h, n)
    return a, v, target, h, n, protected


def test_protocol_cannot_authorize_new_cohorts_or_change_bound_holdouts_or_gpu_guard(study):
    frozen = config(study)
    study.validate_protocol(frozen)
    for key, value in (
        ("gpu_execution_authorized", True),
        ("fresh_cohort_authorized", True),
        ("calibration_authorized", True),
        ("paper_campaign_authorized", True),
        ("source_policy", "widen_the_norm_bound"),
        ("value_solver", {"iterations": 512}),
        ("effective_method", "mosaickv_exact"),
        ("prefix_folds", [{"train_prefixes": 32, "holdout_start": 1, "holdout_end": 32}]),
        ("GPU_followup_if_useful", {"horizon": 4}),
    ):
        altered = copy.deepcopy(frozen)
        altered[key] = value
        with pytest.raises(ValueError, match="scope"):
            study.validate_protocol(altered)


def test_holdout_outputs_are_never_fitting_inputs(study):
    arrays = evidence()
    original = tuple(t.clone() for t in arrays)
    fold = config(study)["prefix_folds"][1]
    core, policy = AttentionCompactionPolicy(True), ConstrainedValuePolicy()
    first, fits = study.evaluate_fold(arrays, fold, core, policy, chunk=2)
    changed = tuple(t.clone() for t in arrays)
    changed[2][:, 16:24] += 100
    second, other = study.evaluate_fold(changed, fold, core, policy, chunk=2)
    for mode in ("mass_only", "value_only", "joint"):
        for key in ("values", "log_bias", "admitted"):
            assert torch.equal(fits[mode][0][key], other[mode][0][key])
    assert all(
        a["train_mse_after"] == b["train_mse_after"] for a, b in zip(first, second, strict=True)
    )
    assert any(
        a["holdout_mse_after"] != b["holdout_mse_after"] for a, b in zip(first, second, strict=True)
    )
    assert all(torch.equal(a, b) for a, b in zip(arrays, original, strict=True))
    assert {r["component"] for r in first} == {"anchor", "mass_only", "value_only", "joint"}
    assert all(r["maximum_value_norm_ratio"] <= 4 for r in first)
    assert all(r["protected_values_unchanged"] and r["protected_bias_zero"] for r in first)


def test_complete_observations_and_partial_folds_do_not_fabricate_unexecuted_folds(study):
    arrays = evidence()
    records = []
    for fold in config(study)["prefix_folds"]:
        rows, _ = study.evaluate_fold(
            arrays, fold, AttentionCompactionPolicy(True), ConstrainedValuePolicy(), chunk=2
        )
        partial = study.summarize_fold(rows, 0, "synthetic", fold["train_prefixes"], 2)
        assert len(partial) == 4
        assert {r["train_prefixes"] for r in partial} == {fold["train_prefixes"]}
        records.extend(rows)
    summary = study.summarize(records, 0, "synthetic", expected_routes=2)
    assert len(summary) == 12
    for bad in (records[:-1], records + records[:1], records[:-1] + records[:1]):
        with pytest.raises(ValueError, match=r"incomplete|duplicate|extra"):
            study.summarize(bad, 0, "synthetic", expected_routes=2)
    invalid = copy.deepcopy(records)
    invalid[0]["holdout_mse_after"] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        study.summarize(invalid, 0, "synthetic", expected_routes=2)
    assert study.error_ratio(0, 0) == 1
    with pytest.raises(ValueError, match="zero-error"):
        study.error_ratio(1, 0)


def synthetic_summaries():
    return [
        dict(
            component=mode,
            worker=worker,
            train_prefixes=count,
            constraints_valid=True,
            admitted_heads=100,
            holdout_mse_ratio=ratio,
        )
        for mode, ratio in (("mass_only", 0.97), ("value_only", 0.85), ("joint", 0.9))
        for worker in (0, 1)
        for count in (8, 16, 24)
    ]


def test_usefulness_screen_requires_each_input_fold_and_constraints_not_just_average(study):
    screen = config(study)["useful_screen"]
    summaries = synthetic_summaries()
    assert study.choose_component(summaries, screen)["selected_component"] == "value_only"
    for key, value in (
        ("constraints_valid", False),
        ("admitted_heads", 0),
        ("holdout_mse_ratio", 1.001),
        ("holdout_mse_ratio", float("nan")),
    ):
        altered = copy.deepcopy(summaries)
        next(r for r in altered if r["component"] == "value_only")[key] = value
        assert study.choose_component(altered, screen)["selected_component"] == "joint"
    absent = [r for r in summaries if r["component"] != "joint"]
    absent.pop(next(i for i, r in enumerate(absent) if r["component"] == "value_only"))
    result = study.choose_component(absent, screen)
    assert not result["useful_bounded_fit"] and result["selected_component"] is None


def test_submission_is_two_cpu_workers_and_finite_afterany_analysis_without_gpu(
    study, monkeypatch, tmp_path
):
    monkeypatch.setattr(study, "prepare", lambda root: (root / "plan.json").write_text("{}"))
    calls = []

    def check_output(args, **kwargs):
        calls.append(args)
        return str(1000 + len(calls))

    monkeypatch.setattr(study.subprocess, "check_output", check_output)
    study.submit(tmp_path)
    assert len(calls) == 2
    assert "--array=0-1%2" in calls[0]
    assert "--dependency=afterany:1001" in calls[1]
    # Inspect scheduler options, not substrings of filesystem paths.
    assert all(not token.startswith(("--gres", "--gpus")) for call in calls for token in call)
    result = json.loads((tmp_path / "submission.json").read_text())
    assert not result["gpu_jobs_submitted"] and result["source_dirty_exploratory"]
    assert not result["calibration_authorized"] and not result["fresh_cohort_authorized"]


def test_slurm_scripts_hide_gpu_and_do_not_download_or_poll(study):
    for name in ("constrained_compaction_cpu_v2", "constrained_compaction_analysis_v2"):
        text = (study.SOURCE / f"mosaickv/slurm/{name}.sbatch").read_text()
        assert "--cpus-per-task=1" in text and "--gres" not in text
        assert 'CUDA_VISIBLE_DEVICES=""' in text
        assert "cache_env.sh" in text and "/scratch/djy8hg/env/mosaickv/bin/python" in text
        assert all(s not in text for s in ("pip install", "squeue", "sleep ", "nvidia-smi"))
