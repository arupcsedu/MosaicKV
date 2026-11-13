"""Synthetic runner gates: historical JSON proofs must never be replayed."""

from copy import deepcopy

import evicastkv_provenance_inventory_v48 as runner
import pytest


def parents():
    rows = [{"synthetic": i} for i in range(1168)]
    counts = dict(raw_error_count=40760, net_unresolved_error_count=135)
    parent = dict(source_sha=runner.PRODUCER, counts=counts)
    plan = dict(
        source_sha=runner.PRODUCER,
        rows=rows,
        rows_sha256=runner.digest(rows),
        retry_indices=list(range(1168)),
    )
    decision = dict(
        source_sha=runner.PRODUCER,
        producer_source_sha=runner.PRODUCER,
        newly_resolved_JSON_occurrences=1168,
        passed_indices=list(range(1168)),
        reconciled_history_sha256=runner.BOUND["analysis/reconciled_history.json"],
        error_counts=counts,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
        historical_media_exposure_resolved=False,
        GPU_jobs=0,
    )
    return {
        "analysis/decision.json": decision,
        "analysis/reconciled_history.json": parent,
        "registration.json": plan,
        "bulk.json": dict(status="registered_work_complete"),
    }


def test_complete_parent_retains_original_identity_checkpoint():
    data = parents()
    original = deepcopy(data)
    parent, plan = runner.check_parent(data)
    assert parent is data["analysis/reconciled_history.json"]
    assert plan is data["registration.json"]
    assert data == original


@pytest.mark.parametrize(
    ("name", "key", "value"),
    [
        ("bulk.json", "status", "incomplete"),
        ("analysis/decision.json", "fresh_confirmation_authorized", True),
        ("analysis/decision.json", "materialization_authorized", True),
        ("analysis/decision.json", "historical_media_exposure_resolved", True),
        ("analysis/decision.json", "GPU_jobs", 1),
        ("analysis/decision.json", "newly_resolved_JSON_occurrences", 1167),
        ("registration.json", "rows_sha256", "f" * 64),
        ("registration.json", "source_sha", "c" * 40),
        ("analysis/reconciled_history.json", "source_sha", "c" * 40),
    ],
)
def test_invalid_or_widened_parent_fails_closed(name, key, value):
    data = parents()
    data[name][key] = value
    with pytest.raises(ValueError, match="complete immutable V47"):
        runner.check_parent(data)


def test_a_missing_or_repeated_success_index_cannot_authorize_inventory():
    data = parents()
    data["analysis/decision.json"]["passed_indices"][0] = 1
    with pytest.raises(ValueError, match="complete immutable V47"):
        runner.check_parent(data)


def test_limits_and_exact_parent_bindings_are_not_increased():
    assert runner.SECONDS == 210
    assert runner.BYTE_BOUND == 2 * 2**30
    assert runner.resume.reader.common.history.MAX_STATE == 512 * 2**20
    assert runner.resume.MAX_ARTIFACT_BYTES == 64 * 2**30
    assert set(runner.BOUND) == {
        "analysis/decision.json",
        "analysis/reconciled_history.json",
        "registration.json",
        "bulk.json",
    }


def test_expired_inventory_cannot_be_reported_as_a_pass(monkeypatch):
    monkeypatch.setattr(runner.time, "monotonic", lambda: 211)
    with pytest.raises(TimeoutError, match="unchanged 210-second"):
        runner._elapsed(0)


def test_afterany_missing_inventory_keeps_all_work_blocked(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "clean_source", lambda *a: "a" * 40)
    monkeypatch.setattr(runner.resume.reader.common, "cpu_gate", lambda *a: None)
    monkeypatch.setattr(
        runner, "_read_parents", lambda: pytest.fail("repeated historical-source read")
    )
    result = runner.analyze(tmp_path)
    assert result["status"] == "inventory_failed_or_incomplete"
    assert result["GPU_jobs"] == 0
    assert result["fresh_confirmation_authorized"] is False
    assert result["materialization_authorized"] is False
