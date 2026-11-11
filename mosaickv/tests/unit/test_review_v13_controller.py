from pathlib import Path
from types import SimpleNamespace

import evicastkv_review_v13 as control
import pytest

from mosaickv.evaluation.fox_campaign import write_once


@pytest.fixture
def ready(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "load", lambda root: {"source_sha": "test"})
    return tmp_path


def test_gpu_request_one_device_twelve_cpus_and_afterany(ready, monkeypatch):
    calls = []
    monkeypatch.setattr(control.subprocess, "check_output", lambda c, **kw: calls.append(c) or "12")
    control.submit(ready, "lookm", "lookm-worker", gpu=True, dependencies=["10"])
    cmd = calls[0]
    assert "--gres=gpu:a100:1" in cmd
    assert "--cpus-per-task=12" in cmd
    assert "--reservation=bi_fox_dgx" in cmd
    assert "--dependency=afterany:10" in cmd
    assert "--time=00:20:00" in cmd
    with pytest.raises(FileExistsError):
        control.submit(ready, "lookm", "lookm-worker", gpu=True)


def test_does_not_blind_retry_lost_acknowledgement(ready, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("unacknowledged")

    monkeypatch.setattr(control.subprocess, "check_output", fail)
    with pytest.raises(control.shared.SubmissionError):
        control.submit(ready, "streaming", "streaming-worker", gpu=True)
    assert control.read(ready / "jobs/streaming-submission-error.json")["automatic_retry"] is False
    with pytest.raises(FileExistsError):
        control.submit(ready, "streaming", "streaming-worker", gpu=True)


def test_provider_dispatch_is_cpu_and_array_indexed(ready, monkeypatch):
    seen = []
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "3")
    monkeypatch.setattr(
        control,
        "module",
        lambda lane: SimpleNamespace(worker=lambda root, index: seen.append((root.name, index))),
    )
    control.dispatch(ready, "providers-worker")
    assert seen == [("providers", 3)]


def test_gpu_ceiling_checked_before_model(ready, monkeypatch):
    monkeypatch.setattr(control.shared, "gpu_snapshot", lambda: {"active_gpus": 7})
    with pytest.raises(ValueError, match="six-A100"):
        control.dispatch(ready, "streaming-worker")


def test_callback_failure_recorded_without_losing_completeness(ready, monkeypatch):
    def fail(*args):
        raise ValueError("numerical failure")

    monkeypatch.setattr(control, "module", lambda lane: SimpleNamespace(analyze=fail))
    result = control.analyze(ready, "lookm")
    assert result["gate_exists"] is False
    gate = control.finish(ready)
    assert gate["errors"]
    assert gate["gates"]["lookm"]["status"] == "missing_gate"
    assert gate["superiority_established"] is False
    assert gate["paper_campaign_authorized"] is False


def test_cpu_failure_stops_all_producers(ready, monkeypatch):
    def fail(*args):
        raise ValueError("incomplete suite")

    monkeypatch.setattr(control.shared, "cpu_gate", fail)
    monkeypatch.setattr(control, "submit", lambda *a, **kw: pytest.fail("must not submit"))
    result = control.prepare(ready)
    assert result["errors"]
    assert result["independent_quality_N"] == 0


def test_gate_presence_is_not_a_superiority_claim(ready):
    for lane, (_, name) in control.LANES.items():
        write_once(ready / lane / name, {"status": "passed"})
    report = control.finish(ready)
    assert report["fresh_confirmation_established"] is False
    assert report["genuine_final_peak_byte_matching_established"] is False
    assert report["six_component_benefits_established"] is False
    assert report["strict_latency_superiority_established"] is False
    assert report["independent_quality_N"] == 0


def test_gpu_array_rejected(ready):
    with pytest.raises(ValueError, match="one GPU"):
        control.submit(ready, "bad", "streaming-worker", gpu=True, array=True)


def test_numerical_callback_has_registered_thread_allocation(ready, monkeypatch):
    calls = []
    monkeypatch.setattr(control.subprocess, "check_output", lambda c, **kw: calls.append(c) or "12")
    control.submit(ready, "after-streaming", "after-streaming", dependencies=["11"])
    assert "--cpus-per-task=12" in calls[0]
    assert "--partition=bii" in calls[0]
    assert "--dependency=afterany:11" in calls[0]
    assert not any("gres=" in item for item in calls[0])


@pytest.mark.parametrize(
    "key",
    [
        "all_original_prompt_KV_bitwise",
        "all_128_source_tokens_agree",
        "all_unchanged_32_token_guards_pass",
        "both_initial_caches_smaller_than_raw",
    ],
)
def test_registration_requires_each_actual_lossless_prerequisite(tmp_path, monkeypatch, key):
    parent = tmp_path / "parent"
    write_once(
        parent / "registration.json", dict(source_sha=control.PARENT_SHA, source_root=str(parent))
    )
    required = dict(
        all_original_prompt_KV_bitwise=True,
        all_128_source_tokens_agree=True,
        all_unchanged_32_token_guards_pass=True,
        both_initial_caches_smaller_than_raw=True,
    )
    required[key] = False
    write_once(
        parent / "lossless/GATE_lossless_source_control.json",
        dict(source_sha=control.PARENT_SHA, failures=[], **required),
    )
    monkeypatch.setattr(control, "PARENT", parent)
    monkeypatch.setattr(control, "clean_source", lambda *args: "current")
    monkeypatch.setattr(control, "submit", lambda *a, **kw: pytest.fail("must not submit"))
    # Scope check is intentionally retained: choose an actual scratch-root path
    # without constructing it; all reads are from the synthetic parent above.
    output = Path("/scratch/djy8hg/workdir") / f"test-v13-no-submit-{tmp_path.name}-{key}"
    with pytest.raises(ValueError, match="actual passed lossless"):
        control.register(output, execute=True)
    assert not output.exists()
