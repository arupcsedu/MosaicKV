from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/evicastkv_review_closure_sequence.py"
SPEC = importlib.util.spec_from_file_location("closure_sequence_test", SCRIPT)
control = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(control)


@pytest.mark.parametrize("count", [1, 2])
def test_gpu_submission_resources_and_afterany_are_fixed(tmp_path, monkeypatch, count):
    monkeypatch.setattr(control.subprocess, "check_output", lambda *a, **k: "12345;cluster\n")
    assert control.submit(tmp_path, "shape", "shape-worker", [100], gpu_count=count) == "12345"
    request = control.read(tmp_path / "jobs/shape-request.json")
    args = request["command"]
    for expected in (
        "--reservation=bi_fox_dgx",
        "--gres=gpu:a100:1",
        "--cpus-per-task=12",
        "--time=02:00:00",
        f"--array=0-{count - 1}%{count}",
        "--dependency=afterany:100",
    ):
        assert expected in args
    with pytest.raises(FileExistsError):
        control.submit(tmp_path, "shape", "shape-worker", gpu_count=count)


def test_cpu_array_never_requests_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(control.subprocess, "check_output", lambda *a, **k: "12345")
    control.submit(tmp_path, "freshness", "freshness-worker", cpu_array=True)
    args = control.read(tmp_path / "jobs/freshness-request.json")["command"]
    assert "--array=0-7%4" in args and "--partition=bii" in args
    assert not any("--gres" in arg or "reservation" in arg for arg in args)


@pytest.mark.parametrize("count", [-1, 3, 6])
def test_unregistered_gpu_scope_rejected(tmp_path, count):
    with pytest.raises(ValueError):
        control.submit(tmp_path, "bad", "shape-worker", gpu_count=count)


def test_injection_or_array_dependency_rejected():
    for value in ("100;touch command", "100_4", "afterok:1", "1\n2"):
        with pytest.raises(ValueError):
            control.identity(value)


def test_dry_run_has_no_submissions(tmp_path, monkeypatch):
    previous, root = tmp_path / "previous", tmp_path / "new"
    previous.mkdir()
    (previous / "registration.json").write_text(
        json.dumps(dict(source_root="old", source_sha="abc", parent="parent"))
    )
    (previous / "FINAL_report.json").write_text(json.dumps(dict(source_sha="abc")))
    monkeypatch.setattr(control, "safe_root", lambda value: value)
    monkeypatch.setattr(control, "clean_source", lambda *a: "abc")
    monkeypatch.setattr(control, "sha", lambda p: "bound")
    monkeypatch.setattr(control, "gpu_snapshot", lambda: pytest.fail("dry run inspected scheduler"))
    monkeypatch.setattr(control, "submit", lambda *a, **k: pytest.fail("dry run submitted"))
    assert control.register(root, previous, execute=False)["submitted_jobs"] == 0
    assert control.read(root / "registration.json")["fresh_confirmation_authorized"] is False


def test_failed_cpu_suite_never_prepares_models(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "load", lambda r: dict(source_sha="abc"))

    def fail(*args):
        raise ValueError("one skipped CPU test")

    monkeypatch.setattr(control, "cpu_gate", fail)
    monkeypatch.setattr(control, "module", lambda *a: pytest.fail("CPU gate bypassed"))
    monkeypatch.setattr(control, "submit", lambda *a, **k: pytest.fail("CPU gate bypassed"))
    monkeypatch.setattr(control, "finish", lambda root: dict(status="blocked"))
    assert control.prepare(tmp_path)["status"] == "blocked"
    assert "skipped" in control.read(tmp_path / "errors/cpu.json")["error"]


@pytest.mark.parametrize(
    "gate",
    [
        dict(),
        dict(complete=False, actual_rows=14, failures=[]),
        dict(complete=True, actual_rows=13, failures=[]),
        dict(complete=True, actual_rows=14, failures=["bad clone"]),
    ],
)
def test_bridge_never_admitted_after_incomplete_or_invalid_fullkv_control(gate):
    assert not control.source_controls_allow_bridge(gate)


def test_lossy_quality_failure_is_preserved_not_filtered_from_diagnostic():
    gate = dict(
        complete=True, actual_rows=14, failures=[], all_source_variants_pass_registered_guard=False
    )
    assert control.source_controls_allow_bridge(gate)


def test_failed_shape_is_not_silently_relabeled_or_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "load", lambda root: {})
    calls = []
    monkeypatch.setattr(
        control, "analyze_lane", lambda root, lane: calls.append(lane) or dict(status="blocked")
    )
    monkeypatch.setattr(
        control, "launch_source", lambda root: calls.append("independent_bf16_scope")
    )
    control.run(tmp_path, "after-shape")
    assert calls == ["shape", "independent_bf16_scope"]


def test_aggregate_ceiling_blocks_model_admission(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "load", lambda root: {})
    monkeypatch.setattr(control, "gpu_snapshot", lambda: dict(active_gpus=7))
    monkeypatch.setattr(
        control,
        "module",
        lambda lane: SimpleNamespace(worker=lambda root: pytest.fail("ceiling bypassed")),
    )
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
    with pytest.raises(ValueError, match="six-GPU"):
        control.run(tmp_path, "shape-worker")
    assert control.read(tmp_path / "admission/shape-worker-0.json")["active_gpus"] == 7


def test_final_report_keeps_missing_gates_and_negative_scope(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "load", lambda root: dict(source_sha="abc"))
    (tmp_path / "registration.json").write_text("{}")
    report = control.finish(tmp_path)
    assert len(report["gates"]) == 5
    assert all(g["status"] == "missing_or_blocked" for g in report["gates"].values())
    assert report["scientific_completion"] is False
    assert report["superiority_established"] is False
    assert report["fresh_confirmation_authorized"] is False
