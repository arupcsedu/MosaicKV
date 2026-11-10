"""Synthetic finite-controller tests: no jobs, models or historical data."""

import json

import evicastkv_review_v8_diagnostics as control
import pytest


@pytest.mark.parametrize(
    "mode,memory,limit",
    [("rmsnorm-worker", "8G", "00:20:00"), ("protection-worker", "64G", "02:00:00")],
)
def test_single_gpu_bounds_and_afterany(tmp_path, monkeypatch, mode, memory, limit):
    monkeypatch.setattr(control.subprocess, "check_output", lambda *a, **kw: "12345;cluster\n")
    assert control.submit(tmp_path, mode, mode, [100], gpu=True) == "12345"
    command = control.read(tmp_path / f"jobs/{mode}-request.json")["command"]
    for flag in (
        "--reservation=bi_fox_dgx",
        "--gres=gpu:a100:1",
        "--cpus-per-task=12",
        f"--mem={memory}",
        f"--time={limit}",
        "--dependency=afterany:100",
    ):
        assert flag in command
    assert not any(flag.startswith("--array") for flag in command)
    with pytest.raises(FileExistsError):
        control.submit(tmp_path, mode, mode, gpu=True)


def test_cpu_suite_never_uses_a_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(control.subprocess, "check_output", lambda *a, **kw: "12345")
    control.submit(tmp_path, "cpu", "tests", tests=True)
    command = control.read(tmp_path / "jobs/cpu-request.json")["command"]
    assert "--partition=bii" in command
    assert not any("gres=" in flag or "reservation=" in flag for flag in command)
    assert command[-1] == "tests"
    assert "evicastkv_review_closure_v2.sbatch" in command[-4]


def test_gpu_tests_scope_rejected(tmp_path):
    with pytest.raises(ValueError):
        control.submit(tmp_path, "cpu", "tests", tests=True, gpu=True)


def test_failed_cpu_suite_never_prepares_or_submits(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "load", lambda root: dict(source_sha="abc"))

    def broken(*args):
        raise ValueError("one skipped test")

    monkeypatch.setattr(control, "cpu_gate", broken)
    monkeypatch.setattr(control, "module", lambda lane: pytest.fail("CPU gate bypassed"))
    monkeypatch.setattr(control, "submit", lambda *a, **kw: pytest.fail("CPU gate bypassed"))
    monkeypatch.setattr(control, "finish", lambda root: dict(status="blocked"))
    assert control.prepare(tmp_path)["status"] == "blocked"
    assert "skipped" in control.read(tmp_path / "errors/cpu.json")["error"]


def test_failed_operator_does_not_trigger_unregistered_model_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "load", lambda root: {})
    calls = []
    monkeypatch.setattr(
        control, "analyze", lambda root, lane: calls.append(lane) or dict(status="negative")
    )
    monkeypatch.setattr(control, "launch_protection", lambda root: calls.append("independent_bf16"))
    control.run(tmp_path, "after-rmsnorm")
    assert calls == ["rmsnorm", "independent_bf16"]


def test_unknown_mode_does_not_submit(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "load", lambda root: {})
    with pytest.raises(ValueError):
        control.run(tmp_path, "fresh-confirmation")


def test_worker_admission_respects_aggregate_six(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "load", lambda root: {})
    monkeypatch.setattr(control, "gpu_snapshot", lambda: dict(active_gpus=7))
    monkeypatch.setattr(control, "module", lambda lane: pytest.fail("admission bypassed"))
    with pytest.raises(ValueError, match="six-A100"):
        control.run(tmp_path, "rmsnorm-worker")


def test_existing_gpu_jobs_become_dependencies(tmp_path, monkeypatch):
    snapshot = dict(active=[dict(job="100_0")], pending=[dict(job="101_2")], active_gpus=1)
    monkeypatch.setattr(control, "gpu_snapshot", lambda: snapshot)
    calls = []
    monkeypatch.setattr(control, "submit", lambda *args, **kw: calls.append((args, kw)) or "300")
    assert control.submit_gpu(tmp_path, "rmsnorm", "rmsnorm-worker") == "300"
    assert calls[0][0][-1] == ["100", "101"]
    assert calls[0][1] == dict(gpu=True)


def test_missing_protection_is_reported_not_retried(tmp_path, monkeypatch):
    (tmp_path / "preparation.json").write_text(json.dumps(dict(protection=dict(status="blocked"))))
    monkeypatch.setattr(control, "submit_final", lambda root: "final")
    monkeypatch.setattr(control, "submit_gpu", lambda *a: pytest.fail("blocked lane admitted"))
    assert control.launch_protection(tmp_path) == "final"


def test_final_packet_preserves_missing_and_no_confirmation(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "load", lambda root: dict(source_sha="abc"))
    result = control.finish(tmp_path)
    assert result["status"] == "completed_with_blocked_controls"
    assert len(result["artifacts"]) == 3
    assert result["superiority_established"] is False
    assert result["fresh_confirmation_authorized"] is False
    assert result["no_automatic_full_model_shape_transport"] is True
    assert all(v["status"] == "missing_or_blocked" for v in result["artifacts"].values())


def test_after_protection_waits_for_freshness_analysis(tmp_path, monkeypatch):
    (tmp_path / "preparation.json").write_text(json.dumps(dict(freshness=dict(analysis="100"))))
    calls = []
    monkeypatch.setattr(control, "load", lambda root: {})
    monkeypatch.setattr(control, "analyze", lambda root, lane: calls.append(lane))
    monkeypatch.setattr(control, "submit", lambda *a, **kw: calls.append(a) or "200")
    assert control.run(tmp_path, "after-protection") == "200"
    assert calls[0] == "protection" and calls[1][-1] == ["100"]


def test_submission_error_keeps_request_and_does_not_retry(tmp_path, monkeypatch):
    calls = []

    def broken(command, **kwargs):
        calls.append(command)
        raise control.subprocess.CalledProcessError(1, command, stderr="scheduler unavailable")

    monkeypatch.setattr(control.subprocess, "check_output", broken)
    with pytest.raises(control.SubmissionError):
        control.submit(tmp_path, "cpu", "tests", tests=True)
    assert len(calls) == 1
    record = control.read(tmp_path / "jobs/cpu-submission-error.json")
    assert record["automatic_retry"] is False
    assert "scheduler unavailable" in record["stderr"]
    assert (tmp_path / "jobs/cpu-request.json").exists()


def test_failed_analyzer_submission_preserves_acknowledged_producer(tmp_path, monkeypatch):
    control.write_once(tmp_path / "jobs/producer-submission.json", dict(job="100"))
    monkeypatch.setattr(control, "load", lambda root: {})
    monkeypatch.setattr(control, "analyze", lambda *a: None)

    def broken(root):
        raise control.SubmissionError("analyzer not acknowledged")

    monkeypatch.setattr(control, "launch_protection", broken)
    result = control.run(tmp_path, "after-rmsnorm")
    assert result["status"] == "blocked_submission"
    assert result["automatic_retry"] is False
    assert next(iter(result["submitted"].values()))["job"] == "100"


def test_final_fallback_aggregate_only_after_definite_no_submission(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "load", lambda root: dict(source_sha="abc"))
    control.write_once(
        tmp_path / "preparation.json",
        dict(freshness=dict(workers="100", analysis_definitely_not_submitted=True)),
    )
    calls = []

    def aggregate(root):
        calls.append(root)
        control.write_once(root / "FINAL_report.json", dict(status="derived"))

    from types import SimpleNamespace

    monkeypatch.setattr(control, "module", lambda lane: SimpleNamespace(aggregate=aggregate))
    control.finish(tmp_path)
    assert calls == [tmp_path / "freshness-continuation"]


def test_uncertain_analyzer_is_not_blindly_duplicated(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "load", lambda root: dict(source_sha="abc"))
    control.write_once(
        tmp_path / "preparation.json",
        dict(freshness=dict(workers="100", analysis_submission_uncertain=True)),
    )
    monkeypatch.setattr(control, "module", lambda *a: pytest.fail("uncertain submission retried"))
    assert control.finish(tmp_path)["status"] == "completed_with_blocked_controls"


def test_failed_freshness_analyzer_still_waits_for_workers(tmp_path, monkeypatch):
    control.write_once(
        tmp_path / "preparation.json",
        dict(freshness=dict(workers="100", analysis=None, analysis_definitely_not_submitted=True)),
    )
    calls = []
    monkeypatch.setattr(control, "submit", lambda *args, **kwargs: calls.append(args) or "200")
    assert control.submit_final(tmp_path) == "200"
    assert calls[0][-1] == ["100"]
