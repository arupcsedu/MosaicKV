"""Finite scheduler and claim-admission boundary tests."""

from types import SimpleNamespace

import evicastkv_review_v11 as control
import pytest


def test_failed_cpu_gate_prevents_preparation_or_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "load", lambda root: {"source_sha": "a" * 40})

    def fail(*args):
        raise ValueError("failed CPU suite")

    monkeypatch.setattr(control.shared, "cpu_gate", fail)
    monkeypatch.setattr(control, "module", lambda lane: pytest.fail("CPU gate bypass"))
    monkeypatch.setattr(control, "finish", lambda root: {"status": "blocked"})
    assert control.prepare(tmp_path)["status"] == "blocked"


def test_negative_causal_result_does_not_authorize_quality_campaign(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "load", lambda root: {})
    monkeypatch.setattr(
        control,
        "module",
        lambda lane: SimpleNamespace(
            analyze=lambda root: {"status": "negative", "quality_gate": False}
        ),
    )
    (tmp_path / "prepared-bytes.json").touch()
    calls = []
    monkeypatch.setattr(control, "submit_next", lambda root, pending: calls.append(pending))
    control.run(tmp_path, "after-causal")
    assert calls == [["bytes"]]
    # Only independent cache construction is advanced; no fresh/calibration mode.
    assert set(control.LANES) == {"causal", "bytes", "ownership"}


def test_queue_failure_records_safe_final_packet(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "load", lambda root: {})

    def fail():
        raise RuntimeError("scheduler unavailable")

    monkeypatch.setattr(control.shared, "gpu_snapshot", fail)
    monkeypatch.setattr(control.shared, "submit", lambda *a, **kw: pytest.fail("blind submit"))
    monkeypatch.setattr(control, "finish", lambda root: {"status": "blocked"})
    assert control.submit_next(tmp_path, ["causal"])["status"] == "blocked"
    assert (tmp_path / "errors/admission-causal.json").exists()


def test_worker_enforces_aggregate_ceiling(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "load", lambda root: {})
    monkeypatch.setattr(control.shared, "gpu_snapshot", lambda: {"active_gpus": 7})
    monkeypatch.setattr(control, "module", lambda lane: pytest.fail("ceiling bypassed"))
    with pytest.raises(ValueError, match="six-GPU"):
        control.run(tmp_path, "causal-worker")


def test_final_waits_for_known_frontier_producer_if_analysis_rejected(tmp_path, monkeypatch):
    control.write_once(
        tmp_path / "freshness-frontier-submission.json", {"analysis": None, "workers": "123"}
    )
    calls = []
    monkeypatch.setattr(
        control.shared,
        "submit",
        lambda *args, **kwargs: calls.append((args, kwargs)) or "456",
    )
    result = control.finish_or_defer(tmp_path)
    assert result["final_analysis"] == "456"
    assert calls[0][0][3] == ["123"]
