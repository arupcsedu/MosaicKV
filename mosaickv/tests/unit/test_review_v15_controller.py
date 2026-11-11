import sys
from types import SimpleNamespace

import evicastkv_review_v15 as control
import pytest

from mosaickv.evaluation.fox_campaign import write_once


@pytest.fixture
def ready(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
    monkeypatch.setattr(
        control,
        "load",
        lambda root: dict(source_sha="test", maximum_derived_passes=3, derived_parent="parent"),
    )
    return tmp_path


def test_gpu_array_is_two_one_gpu_20_minute_workers(ready, monkeypatch):
    calls = []
    monkeypatch.setattr(
        control.subprocess, "check_output", lambda cmd, **kw: calls.append(cmd) or "12"
    )
    control.launch_lane(ready, "ownership")
    for item in (
        "--gres=gpu:a100:1",
        "--array=0-1%2",
        "--reservation=bi_fox_dgx",
        "--cpus-per-task=12",
        "--time=00:20:00",
    ):
        assert item in calls[0]
    assert "--partition=bii,standard" in calls[1]
    assert "--dependency=afterany:12" in calls[1]


def test_scheduler_errors_are_not_blindly_retried(ready, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("missing acknowledgement")

    monkeypatch.setattr(control.subprocess, "check_output", fail)
    with pytest.raises(control.shared.SubmissionError):
        control.submit(ready, "providers", "providers-worker", array="0-1%2")
    with pytest.raises(FileExistsError):
        control.submit(ready, "providers", "providers-worker", array="0-1%2")


def test_ceiling_checked_before_any_gpu_model_load(ready, monkeypatch):
    monkeypatch.setattr(control.shared, "gpu_snapshot", lambda: dict(active_gpus=7))
    monkeypatch.setattr(control, "module", lambda lane: pytest.fail("no model load permitted"))
    with pytest.raises(ValueError, match="six-A100"):
        control.dispatch(ready, "ownership-worker")


def test_no_final_until_every_lane_has_terminal_receipt(ready, monkeypatch):
    monkeypatch.setattr(control, "submit", lambda *a, **kw: pytest.fail("still waiting"))
    write_once(ready / "preparation-complete.json", {})
    for lane in control.LANES[:-1]:
        control.terminal(ready, lane, error=ValueError("preserved failure"))
    assert control.barrier(ready)["status"] == "waiting_for_terminal_callbacks_no_polling"
    assert not (ready / "final-submission-claim.json").exists()


def test_terminal_report_waits_for_fresh_orphaned_producer_and_current(ready, monkeypatch):
    write_once(ready / "preparation-complete.json", {})
    for lane in control.LANES:
        control.terminal(ready, lane, error=ValueError("known failure"))
    write_once(
        ready / "jobs/providers-submission.json", dict(job="14", acknowledged_at_unix_seconds=100)
    )
    monkeypatch.setenv("SLURM_JOB_ID", "15")
    monkeypatch.setattr(control.time, "time", lambda: 110)
    monkeypatch.setitem(
        sys.modules,
        "evicastkv_completion_v14",
        SimpleNamespace(
            resolve_dependencies=lambda ids: pytest.fail("fresh IDs need no accounting lookup")
        ),
    )
    seen = []
    monkeypatch.setattr(control, "submit", lambda *a, **kw: seen.append(kw) or "16")
    assert control.barrier(ready) == dict(final="16")
    assert seen == [dict(dependencies=["14", "15"])]
    assert control.barrier(ready)["status"] == "final_already_claimed_no_duplicate"


def test_final_reconciles_aged_terminal_jobs(ready, monkeypatch):
    write_once(ready / "preparation-complete.json", {})
    for lane in control.LANES:
        control.terminal(ready, lane)
    write_once(
        ready / "jobs/ownership-submission.json", dict(job="12", acknowledged_at_unix_seconds=1)
    )
    monkeypatch.setattr(control.time, "time", lambda: 500)
    monkeypatch.setitem(
        sys.modules,
        "evicastkv_completion_v14",
        SimpleNamespace(resolve_dependencies=lambda ids: ([], dict(terminal=ids))),
    )
    seen = []
    monkeypatch.setattr(control, "submit", lambda *a, **kw: seen.append(kw) or "16")
    control.barrier(ready)
    assert seen == [dict(dependencies=[])]
    assert control.read(ready / "final-dependency-resolution.json")["aged_resolution"] == dict(
        terminal=["12"]
    )


@pytest.mark.parametrize(
    "index,unattempted,continue_expected", [(1, 670, True), (2, 0, False), (3, 1, False)]
)
def test_derived_continues_only_unattempted_and_at_most_three_passes(
    ready, monkeypatch, index, unattempted, continue_expected
):
    output = ready / f"derived-{index}"
    write_once(
        output / control.GATES["derived"], dict(unattempted_files=unattempted, blocked_files=10)
    )
    calls = []
    monkeypatch.setattr(
        control,
        "module",
        lambda lane: SimpleNamespace(
            analyze=lambda root: None, prepare=lambda root, **kw: calls.append((root, kw))
        ),
    )
    monkeypatch.setattr(control, "launch_lane", lambda *a, **kw: "17")
    monkeypatch.setattr(control, "barrier", lambda root: dict(terminal=True))
    result = control.after(ready, "derived", index)
    if continue_expected:
        assert result == dict(next_callback="17")
        assert calls[0][1]["previous"] == [output]
        assert not (ready / "terminal/derived.json").exists()
    else:
        assert result == dict(terminal=True)
        assert not calls
        assert control.read(ready / "terminal/derived.json")["gate_path"] == str(
            output / control.GATES["derived"]
        )


def test_missing_gate_is_terminal_failure_not_scientific_pass(ready, monkeypatch):
    monkeypatch.setattr(control, "module", lambda lane: SimpleNamespace(analyze=lambda root: None))
    monkeypatch.setattr(control, "barrier", lambda root: dict(waiting=True))
    control.after(ready, "ownership")
    item = control.read(ready / "terminal/ownership.json")
    assert "FileNotFoundError" in item["error"]
    assert item["scientific_pass_not_implied"] is True


def test_cpu_failure_stops_every_gpu_and_metadata_producer(ready, monkeypatch):
    def fail(*a, **kw):
        raise ValueError("incomplete CPU suite")

    monkeypatch.setattr(control.shared, "cpu_gate", fail)
    monkeypatch.setattr(
        control, "launch_lane", lambda *a, **kw: pytest.fail("no producer permitted")
    )
    monkeypatch.setattr(control, "barrier", lambda root: dict(report=True))
    assert control.prepare(ready) == dict(report=True)
    assert all((ready / f"terminal/{lane}.json").exists() for lane in control.LANES)


def test_report_never_fabricates_claims(ready):
    for lane in control.LANES:
        control.terminal(ready, lane, error=ValueError("scientific gate still fails"))
    report = control.finish(ready)
    assert report["independent_quality_N"] == 0
    assert report["fresh_confirmation_established"] is False
    assert report["nearest_baseline_superiority_established"] is False
    assert report["six_component_benefits_established"] is False
    assert report["paper_campaign_authorized"] is False


@pytest.mark.parametrize("lane", ["history", "completion", "legacy-kv"])
def test_saved_cpu_only_lanes_never_load_gpu_worker(ready, monkeypatch, lane):
    calls = []
    monkeypatch.setattr(
        control,
        "module",
        lambda selected: SimpleNamespace(
            run=lambda *a, **kw: calls.append((selected, a, kw)),
            register=lambda *a, **kw: calls.append((selected, a, kw)),
            worker=lambda *a, **kw: pytest.fail("not a GPU or data-read worker"),
        ),
    )
    monkeypatch.setattr(
        control.shared, "gpu_snapshot", lambda: pytest.fail("no GPU admission necessary")
    )
    control.dispatch(ready, lane + "-worker")
    assert calls[0][0] == lane
    assert calls[0][1] == (ready / lane,)


def test_saved_cpu_gate_is_terminal_without_rerunning_analyzer(ready, monkeypatch):
    path = ready / "history" / control.GATES["history"]
    write_once(path, dict(fresh_confirmation_authorized=False))
    monkeypatch.setattr(control, "module", lambda lane: pytest.fail("gate already derived"))
    monkeypatch.setattr(control, "barrier", lambda root: dict(waiting=True))
    assert control.after(ready, "history") == dict(waiting=True)
    assert control.read(ready / "terminal/history.json")["error"] is None
