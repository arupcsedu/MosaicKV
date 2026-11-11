from types import SimpleNamespace

import evicastkv_review_v16 as control
import pytest

from mosaickv.evaluation.fox_campaign import write_once


@pytest.fixture
def ready(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "load", lambda root: dict(source_sha="test"))
    return tmp_path


def test_every_request_is_cpu_only_and_bounded(ready, monkeypatch):
    calls = []
    monkeypatch.setattr(
        control.subprocess, "check_output", lambda cmd, **kw: calls.append(cmd) or "10"
    )
    control.submit(ready, "cpu", "tests", tests=True)
    control.submit(ready, "prepare", "prepare", dependencies=["10"])
    for cmd in calls:
        assert "--nodes=1" in cmd and "--partition=bii,standard" in cmd
        assert "--time=00:20:00" in cmd
        assert not any(part.startswith(("--gres", "--gpus")) for part in cmd)
    assert "--dependency=afterany:10" in calls[1]


def test_unacknowledged_job_is_not_duplicated(ready, monkeypatch):
    def fail(*a, **kw):
        raise RuntimeError("no acknowledgement")

    monkeypatch.setattr(control.subprocess, "check_output", fail)
    with pytest.raises(control.shared.SubmissionError):
        control.submit(ready, "cpu", "tests")
    with pytest.raises(FileExistsError):
        control.submit(ready, "cpu", "tests")


def test_cpu_failure_blocks_scanning(ready, monkeypatch):
    def fail(*a, **kw):
        raise ValueError("CPU suite incomplete")

    monkeypatch.setattr(control.shared, "cpu_gate", fail)
    monkeypatch.setattr(control, "module", lambda name: pytest.fail("no producer permitted"))
    report = control.prepare(ready)
    assert report["evidence"]["history"]["status"] == "missing_gate_not_scientific_success"
    assert report["acknowledged_corrective_GPU_requests"] == 0


def test_storage_precedes_finite_history_callback(ready, monkeypatch):
    seen = []
    monkeypatch.setattr(control.shared, "cpu_gate", lambda *a: dict(status="passed"))

    def mod(name):
        if name == "storage_boundary_reconciliation":
            return SimpleNamespace(run=lambda *a, **kw: seen.append("storage"))
        return SimpleNamespace(
            register=lambda *a, **kw: seen.append(("history", kw)) or dict(worker="11")
        )

    monkeypatch.setattr(control, "module", mod)
    result = control.prepare(ready)
    assert result["history"] == dict(worker="11")
    assert seen[0] == "storage"
    hook = seen[1][1]["completion_argv"]
    assert hook[0] == "sbatch" and hook[-1] == "after-history"
    assert not any(part.startswith(("--dependency", "--gres", "--gpus")) for part in hook)


def test_only_corrective_ownership_requests_two_gpus(ready):
    cmd = control.command(ready, "ownership", "ownership-worker", gpu=True)
    for part in (
        "--array=0-1%2",
        "--gres=gpu:a100:1",
        "--cpus-per-task=12",
        "--reservation=bi_fox_dgx",
    ):
        assert part in cmd
    with pytest.raises(ValueError, match="Only the two corrective"):
        control.command(ready, "prepare", "prepare", gpu=True)


def test_final_barrier_waits_for_both_dynamic_lanes(ready, monkeypatch):
    write_once(ready / "preparation-complete.json", {})
    write_once(ready / "terminal/history.json", {})
    monkeypatch.setattr(control, "submit", lambda *a, **kw: pytest.fail("ownership not terminal"))
    assert control.barrier(ready)["status"] == "waiting_for_finite_callbacks_no_polling"


def test_final_barrier_includes_fresh_acknowledged_producer(ready, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    write_once(ready / "preparation-complete.json", {})
    for lane in ("history", "ownership"):
        write_once(ready / f"terminal/{lane}.json", {})
    write_once(
        ready / "jobs/ownership-submission.json", dict(job="12", acknowledged_at_unix_seconds=100)
    )
    monkeypatch.setattr(control.time, "time", lambda: 110)
    monkeypatch.setattr(control, "module", lambda name: pytest.fail("fresh ACK needs no sacct"))
    seen = []
    monkeypatch.setattr(control, "submit", lambda *a, **kw: seen.append(kw) or "13")
    assert control.barrier(ready) == dict(final="13")
    assert seen == [dict(dependencies=["12"])]
    assert control.barrier(ready)["status"] == "final_already_claimed"


def test_final_preserves_negatives_and_unobserved_boundaries(ready):
    write_once(ready / "history" / control.HISTORY_GATE, dict(frontier_exhausted=False))
    result = control.finish(ready)
    assert result["independent_quality_N"] == 0
    assert result["fresh_confirmation_established"] is False
    assert result["whole_request_KV_peak_certified"] is False
    assert result["six_component_benefits_established"] is False
    assert result["paper_campaign_authorized"] is False


def failed_archive_fixture(parent):
    write_once(parent / "protocol.json", dict(source_sha=control.FAILED_SHA))
    write_once(parent / control.OWNERSHIP_GATE, dict(status="blocked_lossless_ownership"))
    for index in (0, 1):
        out = parent / f"workers/{index}"
        write_once(
            out / "raw.json",
            dict(
                source_sha=control.FAILED_SHA,
                index=index,
                record={},
                counters={},
                diagnostics={
                    name: dict(registered_guard_pass=True)
                    for name in (
                        "registered_guard_32",
                        "cache_dependent_guard_32",
                        "prefix_64",
                        "prefix_128",
                    )
                },
            ),
        )
        write_once(
            out / "result.json",
            dict(
                source_sha=control.FAILED_SHA,
                status="failed",
                all_32_128_guards_pass=True,
                index=index,
                failures=[dict(archive_error="ValueError: bounded semantic expansion exceeded")],
                outputs={"raw.json": control.sha(out / "raw.json")},
            ),
        )


def test_archival_retry_is_scoped_to_exact_completed_requests(ready, monkeypatch):
    parent = ready / "failed"
    failed_archive_fixture(parent)
    monkeypatch.setattr(
        control, "module", lambda name: SimpleNamespace(verify_request=lambda *a: True)
    )
    result = control.retry_evidence(parent)
    assert result["source_sha"] == control.FAILED_SHA
    assert result["numerical_failure_retried"] is False
    assert len(result["bindings"]) == 6


@pytest.mark.parametrize(
    "field,value",
    [
        ("all_32_128_guards_pass", False),
        ("failures", [dict(error="numerical guard failed")]),
        ("source_sha", "f" * 40),
    ],
)
def test_archival_retry_rejects_any_other_failure(ready, monkeypatch, field, value):
    parent = ready / "failed"
    failed_archive_fixture(parent)
    path = parent / "workers/1/result.json"
    record = control.read(path)
    record[field] = value
    path.write_text(control.json.dumps(record))
    monkeypatch.setattr(
        control, "module", lambda name: SimpleNamespace(verify_request=lambda *a: True)
    )
    with pytest.raises(ValueError, match="observed archive-resource"):
        control.retry_evidence(parent)
