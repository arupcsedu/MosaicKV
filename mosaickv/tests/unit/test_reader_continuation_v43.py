"""Synthetic fail-fast orchestration; no real history, model or scheduler."""

import evicastkv_reader_continuation_v43 as driver
import pytest


def test_CPU_failure_prevents_prepare(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "clean_source", lambda source: "a" * 40)
    monkeypatch.setattr(
        driver.worker, "cpu_gate", lambda *a: (_ for _ in ()).throw(ValueError("CPU gate"))
    )
    monkeypatch.setattr(driver.worker.history, "read", lambda *a: pytest.fail("history opened"))
    with pytest.raises(ValueError, match="CPU gate"):
        driver.prepare(tmp_path)


def test_failed_canary_cannot_start_retry_waves(tmp_path, monkeypatch):
    monkeypatch.setattr(driver.worker, "load", lambda root: dict(source_sha="a" * 40))
    monkeypatch.setattr(driver.worker, "cpu_gate", lambda *a: {})
    monkeypatch.setattr(
        driver.worker.history,
        "read",
        lambda path: dict(source_sha="a" * 40, streaming_bulk_authorized=False),
    )
    monkeypatch.setattr(driver.worker, "run_tasks", lambda *a, **k: pytest.fail("retry started"))
    result = driver.bulk(tmp_path)
    assert result["status"] == "blocked_canary_failed"
    assert result["fresh_confirmation_authorized"] is False


@pytest.mark.parametrize(
    "failure",
    [
        "bounded streaming metadata deadline reached",
        "bounded projected metadata capacity exceeded",
        None,
    ],
)
def test_eight_file_wave_stops_on_resource_failure_and_hashes_successes(
    tmp_path, monkeypatch, failure
):
    plan = dict(
        source_sha="a" * 40,
        json_canary_indices=[0],
        retry_indices=list(range(9)),
        bulk_wall_seconds=5400,
    )
    calls = []
    monkeypatch.setattr(driver.worker, "load", lambda root: plan)
    monkeypatch.setattr(driver.worker, "cpu_gate", lambda *a: {})
    monkeypatch.setattr(
        driver.worker.history,
        "read",
        lambda path: dict(source_sha="a" * 40, streaming_bulk_authorized=True),
    )
    monkeypatch.setattr(driver, "qualified_pair", lambda *a: {})
    monkeypatch.setattr(
        driver.worker,
        "run_tasks",
        lambda root, plan, indices, phase, label, **k: calls.append((indices.copy(), phase)),
    )

    def record(root, plan, index, phase):
        if index == 1:
            return dict(status="blocked", error=failure) if failure is not None else None
        return dict(status="passed")

    monkeypatch.setattr(driver.worker, "checked_task", record)
    result = driver.bulk(tmp_path)
    assert calls[0] == (list(range(8)), "parse")
    assert calls[1] == ([0, 2, 3, 4, 5, 6, 7], "hash")
    assert len(calls) == 2
    assert result["completed_waves"] == 1
    assert result["status"] == "resource_failure_stopped_later_waves"
    assert result["worker_seconds"] == 210
    assert result["worker_maximum_bytes"] == 2 * 2**30


def test_ordinary_syntax_rejection_stays_negative_without_blocking_other_files(
    tmp_path, monkeypatch
):
    plan = dict(
        source_sha="a" * 40,
        json_canary_indices=[0],
        retry_indices=list(range(9)),
        bulk_wall_seconds=5400,
    )
    calls = []
    monkeypatch.setattr(driver.worker, "load", lambda root: plan)
    monkeypatch.setattr(driver.worker, "cpu_gate", lambda *a: {})
    monkeypatch.setattr(
        driver.worker.history,
        "read",
        lambda path: dict(source_sha="a" * 40, streaming_bulk_authorized=True),
    )
    monkeypatch.setattr(driver, "qualified_pair", lambda *a: {})
    monkeypatch.setattr(
        driver.worker,
        "run_tasks",
        lambda root, plan, indices, phase, label, **k: calls.append((indices.copy(), phase)),
    )
    monkeypatch.setattr(
        driver.worker,
        "checked_task",
        lambda root, plan, index, phase: (
            dict(status="blocked", error="invalid JSON scalar")
            if index == 1
            else dict(status="passed")
        ),
    )
    result = driver.bulk(tmp_path)
    assert ([8], "parse") in calls
    assert result["completed_waves"] == 2
    assert result["fresh_confirmation_authorized"] is False
