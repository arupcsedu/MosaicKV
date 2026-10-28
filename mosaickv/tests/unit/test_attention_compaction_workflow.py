"""Synthetic scope and afterany orchestration only; never submit real jobs."""

import copy
import json

import pytest
import yaml
from attention_compaction_hf_gate import BLOCKED, PROTOCOL, analyze, submit, validate_protocol
from bounded_history_pilot import write_once


def test_gate_rejects_scope_growth_threshold_changes_and_extra_probes():
    original = yaml.safe_load(PROTOCOL.read_text())
    validate_protocol(original)
    for key in BLOCKED:
        with pytest.raises(ValueError, match="scope"):
            validate_protocol(dict(original, **{key: True}))
    for key, value in (
        ("maximum_joint_candidates", 2),
        ("workers", 3),
        ("gpu_execution_authorized", False),
        ("effective_method", "mosaickv_exact"),
    ):
        with pytest.raises(ValueError, match="scope"):
            validate_protocol(dict(original, **{key: value}))
    altered = copy.deepcopy(original)
    altered["guard"]["absolute_tolerance"] = 0.01
    with pytest.raises(ValueError, match="thresholds"):
        validate_protocol(altered)


def test_manifest_uses_actual_cap_and_extended_accounting_not_the_parent_budget():
    from attention_compaction_hf_gate import manifest_control_config
    from bounded_history_pilot import SOURCE

    from mosaickv.config import load_config

    original = load_config(SOURCE / "mosaickv/configs/development/llava_bounded_history_v1.yaml")
    cap = original.cache.budget_value
    core = dict(capacity_planning=dict(cap_bytes=28 * 2**20, history_positions=16))
    resolved = manifest_control_config(original, core)
    assert resolved.cache.budget_value == 28 * 2**20
    assert resolved.cache.accounting_spec_sha != original.cache.accounting_spec_sha
    assert resolved.model == original.model and resolved.generation == original.generation
    assert original.cache.budget_value == cap


def test_missing_or_failed_workers_are_analyzed_without_promoting_anything(tmp_path, monkeypatch):
    import attention_compaction_hf_gate as module

    plan = dict(
        jobs=[dict(index=i, task=f"synthetic-{i}") for i in range(2)],
        source_sha="a" * 40,
        protocol=yaml.safe_load(PROTOCOL.read_text()),
    )
    monkeypatch.setattr(module, "validate", lambda *_: plan)
    analyze(tmp_path, "b" * 64)
    report = json.loads((tmp_path / "analysis/report.json").read_text())
    assert report["status"] == "incomplete" and len(report["errors"]) == 2
    assert not report["diagnostic_passed"] and not report["benchmark_quality_established"]
    assert report["action"] == "negative_diagnostic_no_automatic_successor"
    assert all(report[k] is False for k in BLOCKED)
    with pytest.raises(FileExistsError):
        analyze(tmp_path, "b" * 64)


def test_submission_has_two_one_a100_workers_afterany_analysis_and_finite_admission(
    tmp_path, monkeypatch
):
    import attention_compaction_hf_gate as module

    calls, runs = [], []
    root = tmp_path / "synthetic-run"

    def fake_prepare(path):
        plan = dict(source_sha="a" * 40)
        write_once(path / "plan.json", plan)
        return plan

    monkeypatch.setattr(module, "prepare", fake_prepare)
    monkeypatch.setattr(module, "validate", lambda *_: None)

    def submitted(args, **_kw):
        calls.append(args)
        return str(200 + len(calls)) + "\n"

    monkeypatch.setattr(module.subprocess, "check_output", submitted)
    monkeypatch.setattr(module.subprocess, "run", lambda args, **_kw: runs.append(args))
    submit(root)
    assert len(calls) == 2 and "--hold" in calls[0] and "--array=0-1%2" in calls[0]
    assert "--dependency=afterany:201" in calls[1]
    assert len(runs) == 1 and "--apply" in runs[0] and "--watch" not in runs[0]
    policy = json.loads((root / "scheduler/a100_schedule.json").read_text())
    assert policy["maximum_a100s"] == 6 and policy["six_a100s_explicitly_authorized"]
    assert policy["arrays"] == policy["adopt_existing_holds"] == ["201"]
    submission = json.loads((root / "submission.json").read_text())
    assert all(submission[k] is False for k in BLOCKED)
