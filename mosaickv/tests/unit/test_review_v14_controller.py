import sys
from types import SimpleNamespace

import evicastkv_review_v14 as control
import pytest

from mosaickv.evaluation.fox_campaign import write_once


@pytest.fixture
def ready(tmp_path, monkeypatch):
    # Synthetic scheduler tests must not inherit their real CPU test job.
    # Tests that exercise a current callback set their own explicit job ID.
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
    monkeypatch.setattr(control, "load", lambda root: {"source_sha": "test"})
    return tmp_path


def test_timing_array_is_two_one_gpu_workers(ready, monkeypatch):
    calls = []
    monkeypatch.setattr(control.subprocess, "check_output", lambda c, **kw: calls.append(c) or "12")
    control.submit(ready, "timing", "timing-worker", gpu=True, array=True, dependencies=["10"])
    command = calls[0]
    for flag in (
        "--array=0-1%2",
        "--gres=gpu:a100:1",
        "--cpus-per-task=12",
        "--reservation=bi_fox_dgx",
        "--time=01:00:00",
        "--dependency=afterany:10",
    ):
        assert flag in command


def test_fresh_acknowledged_dependency_does_not_query_lagging_database(ready, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "evicastkv_completion_v14",
        SimpleNamespace(
            resolve_dependencies=lambda ids: pytest.fail("fresh acknowledgement is sufficient")
        ),
    )
    calls = []
    monkeypatch.setattr(control.subprocess, "check_output", lambda c, **kw: calls.append(c) or "12")
    control.submit(ready, "prepare", "prepare", dependencies=["11"])
    assert "--dependency=afterany:11" in calls[0]
    assert "--partition=bii,standard" in calls[0]


def test_final_resolves_purged_terminal_dependencies(ready, monkeypatch):
    write_once(ready / "static-dependencies.json", {"jobs": ["11", "12"]})
    monkeypatch.setitem(
        sys.modules,
        "evicastkv_completion_v14",
        SimpleNamespace(
            resolve_dependencies=lambda ids: (["12"], {"original": ids, "terminal": ["11"]})
        ),
    )
    calls = []
    monkeypatch.setattr(control.subprocess, "check_output", lambda c, **kw: calls.append(c) or "13")
    control.submit_final(ready)
    assert "--dependency=afterany:12" in calls[0]
    receipt = control.read(ready / "jobs/final-request.json")
    assert receipt["dependency_resolution"]["aged_resolution"]["terminal"] == ["11"]


def test_final_keeps_fresh_orphaned_producer_and_current_callback(ready, monkeypatch):
    write_once(ready / "static-dependencies.json", {"jobs": []})
    write_once(
        ready / "jobs/providers-submission.json",
        {"job": "14", "acknowledged_at_unix_seconds": 100.0},
    )
    monkeypatch.setenv("SLURM_JOB_ID", "15")
    monkeypatch.setattr(control.time, "time", lambda: 110.0)
    monkeypatch.setitem(
        sys.modules,
        "evicastkv_completion_v14",
        SimpleNamespace(
            resolve_dependencies=lambda ids: pytest.fail("fresh live IDs bypass accounting")
        ),
    )
    calls = []
    monkeypatch.setattr(control.subprocess, "check_output", lambda c, **kw: calls.append(c) or "16")
    control.submit_final(ready)
    assert "--dependency=afterany:14:15" in calls[0]


def test_no_blind_submission_retry(ready, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("lost acknowledgement")

    monkeypatch.setattr(control.subprocess, "check_output", fail)
    with pytest.raises(control.shared.SubmissionError):
        control.submit(ready, "fetch", "fetch-worker")
    with pytest.raises(FileExistsError):
        control.submit(ready, "fetch", "fetch-worker")


def test_gpu_ceiling_before_loading_any_model(ready, monkeypatch):
    monkeypatch.setattr(control.shared, "gpu_snapshot", lambda: {"active_gpus": 7})
    with pytest.raises(ValueError, match="six-A100"):
        control.dispatch(ready, "timing-worker")


def test_provider_array_dispatch_is_cpu_only(ready, monkeypatch):
    seen = []
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "4")
    monkeypatch.setattr(
        control,
        "module",
        lambda lane: SimpleNamespace(worker=lambda root, index: seen.append((root.name, index))),
    )
    control.dispatch(ready, "providers-worker")
    assert seen == [("providers", 4)]


def test_failed_fetch_does_not_start_provider_reads(ready, monkeypatch):
    monkeypatch.setattr(control, "module", lambda lane: SimpleNamespace(analyze=lambda root: None))
    monkeypatch.setattr(control, "submit_final", lambda root: {"final": "registered"})
    assert control.after(ready, "fetch") == {"final": "registered"}
    assert not (ready / "providers").exists()


def test_provider_read_waits_for_completed_fetch(ready, monkeypatch):
    write_once(
        ready / "fetch/GATE_historical_provider_fetch.json", {"status": "pinned_download_complete"}
    )
    calls = []
    monkeypatch.setattr(
        control,
        "module",
        lambda lane: SimpleNamespace(
            analyze=lambda root: None, prepare=lambda root, **kw: calls.append(("prepared", lane))
        ),
    )
    monkeypatch.setattr(
        control, "submit", lambda root, label, mode, **kw: calls.append((label, mode, kw)) or "13"
    )
    result = control.after(ready, "fetch")
    assert result["providers"] == "13"
    assert calls[0] == ("prepared", "providers")
    assert calls[1][2]["array"] is True
    assert calls[2][2]["dependencies"] == ["13"]


def test_partial_reports_cannot_establish_paper_claim(ready):
    write_once(ready / "timing/GATE_lossless_streaming_timing.json", {"status": "passed"})
    result = control.finish(ready)
    assert result["independent_quality_N"] == 0
    assert result["nearest_baseline_superiority_established"] is False
    assert result["fresh_confirmation_established"] is False
    assert result["six_component_benefits_established"] is False
    assert result["gates"]["providers"]["status"] == "missing_gate"


def test_cpu_failure_stops_model_and_fetch(ready, monkeypatch):
    def fail(*args):
        raise ValueError("suite incomplete")

    monkeypatch.setattr(control.shared, "cpu_gate", fail)
    monkeypatch.setattr(control, "submit", lambda *a, **kw: pytest.fail("must not submit"))
    result = control.prepare(ready)
    assert result["errors"]
