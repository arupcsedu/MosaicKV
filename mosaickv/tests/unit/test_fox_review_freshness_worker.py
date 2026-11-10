"""Finite wrapper tests only; no historical filesystem scans or submissions."""

import json

import evicastkv_review_freshness_worker as wrapper
import pytest


@pytest.fixture
def configured(monkeypatch, tmp_path):
    monkeypatch.setattr(wrapper, "clean_source", lambda path: "a" * 40)
    return tmp_path / "audit"


def gate(index=0, shards=8, *, terminal=False, errors=0):
    return dict(
        source_sha="a" * 40,
        shard_index=index,
        shard_count=shards,
        traversal_complete=terminal,
        errors=errors,
        fresh_confirmation_authorized=False,
    )


def test_cpu_retry_scan_budget_is_at_most_nine_minutes_per_worker():
    assert wrapper.MAX_INVOCATIONS == 3
    assert wrapper.SECONDS_PER_INVOCATION == 180
    assert wrapper.MAX_INVOCATIONS * wrapper.SECONDS_PER_INVOCATION <= 540


def fake_run(monkeypatch, outcomes):
    calls = []

    def run(root, **kwargs):
        calls.append(kwargs)
        value = outcomes[len(calls) - 1]
        if isinstance(value, Exception):
            raise value
        root.mkdir(parents=True, exist_ok=True)
        (root / f"checkpoint-{len(calls) - 1:06d}.json").write_text("{}")
        return value

    monkeypatch.setattr(wrapper.stream, "run", run)
    return calls


def test_three_invocation_bound_and_resume(configured, monkeypatch):
    calls = fake_run(monkeypatch, [gate(), gate(), gate()])
    result = wrapper.worker(configured, 0)
    assert len(calls) == 3
    assert [call["resume"] for call in calls] == [False, True, True]
    assert all(call["seconds"] == 180 for call in calls)
    assert result["status"] == "incomplete_invocation_budget_exhausted"
    assert not result["fresh_confirmation_authorized"]


def test_terminal_first_gate_stops_early(configured, monkeypatch):
    calls = fake_run(monkeypatch, [gate(terminal=True)])
    result = wrapper.worker(configured, 0)
    assert len(calls) == 1
    assert result["status"] == "completed_declared_scope_freshness_blocked"
    assert result["traversal_complete"]


def test_terminal_coverage_errors_do_not_retry_or_authorize(configured, monkeypatch):
    calls = fake_run(monkeypatch, [gate(terminal=True, errors=5)])
    result = wrapper.worker(configured, 0)
    assert len(calls) == 1
    assert result["status"] == "completed_with_coverage_errors_freshness_blocked"
    assert not result["complete_historical_id_inventory"]


def test_existing_partial_checkpoint_uses_resume(configured, monkeypatch):
    shard, _summary = wrapper._paths(configured, 0)
    shard.mkdir(parents=True)
    (shard / "checkpoint-000000.json").write_text("{}")
    calls = fake_run(monkeypatch, [gate(terminal=True)])
    wrapper.worker(configured, 0)
    assert calls[0]["resume"] is True


@pytest.mark.parametrize(
    "exc", [ValueError("changed source"), OSError("unreadable"), RuntimeError("unexpected")]
)
def test_exception_is_classified_recorded_and_not_retried(configured, monkeypatch, exc):
    calls = fake_run(monkeypatch, [exc])
    result = wrapper.worker(configured, 0)
    assert len(calls) == 1
    assert result["status"] == "blocked_nonretryable_error"
    assert result["nonretryable_error"]["retryable"] is False
    saved = json.loads((configured / "workers/worker-000.json").read_text())
    assert saved["nonretryable_error"]["message"] == str(exc)


def test_unclassified_gate_is_nonretryable(configured, monkeypatch):
    calls = fake_run(monkeypatch, [{"success": True}])
    result = wrapper.worker(configured, 0)
    assert len(calls) == 1
    assert result["status"] == "blocked_nonretryable_error"


def test_write_once_worker_summary(configured, monkeypatch):
    fake_run(monkeypatch, [gate(terminal=True)])
    wrapper.worker(configured, 0)
    with pytest.raises(ValueError, match="already exists"):
        wrapper.worker(configured, 0)


def test_terminal_checkpoint_recovery_avoids_invalid_resume(configured, monkeypatch):
    shard, _summary = wrapper._paths(configured, 0)
    shard.mkdir(parents=True)
    checkpoint = shard / "checkpoint-000000.json"
    checkpoint.write_text("{}")
    saved = gate(terminal=True)
    saved["checkpoint_sha256"] = wrapper.sha(checkpoint)
    (shard / "GATE_chunk-000000.json").write_text(json.dumps(saved))
    calls = fake_run(monkeypatch, [])
    result = wrapper.worker(configured, 0)
    assert calls == []
    assert result["recovered_terminal_checkpoint"]
    assert result["traversal_complete"]


def test_aggregate_records_missing_and_incomplete_shards_without_merge(configured, monkeypatch):
    fake_run(monkeypatch, [gate(shards=2), gate(shards=2), gate(shards=2)])
    wrapper.worker(configured, 0, shards=2)
    calls = []
    monkeypatch.setattr(wrapper.stream, "aggregate", lambda *args: calls.append(args))
    result = wrapper.aggregate(configured, shards=2)
    assert calls == []
    assert result["missing_shards"] == [1]
    assert result["incomplete_shards"] == [0]
    assert result["status"] == "blocked_missing_or_incomplete_shards"


def test_aggregate_all_registered_terminal_shards_once(configured, monkeypatch):
    fake_run(monkeypatch, [gate(0, 2, terminal=True), gate(1, 2, terminal=True)])
    wrapper.worker(configured, 0, shards=2)
    wrapper.worker(configured, 1, shards=2)
    calls = []

    def merge(root, shards):
        calls.append((root, shards))
        return dict(complete_declared_identity_field_scope=True)

    monkeypatch.setattr(wrapper.stream, "aggregate", merge)
    result = wrapper.aggregate(configured, shards=2)
    assert len(calls) == 1
    assert len(calls[0][1]) == 2
    assert result["complete_declared_identity_field_scope"]
    assert not result["complete_historical_id_inventory"]
    assert not result["fresh_confirmation_authorized"]
    with pytest.raises(ValueError, match="already exists"):
        wrapper.aggregate(configured, shards=2)


def test_merge_exception_is_preserved(configured, monkeypatch):
    fake_run(monkeypatch, [gate(0, 1, terminal=True)])
    wrapper.worker(configured, 0, shards=1)

    def fail(*args):
        raise ValueError("checkpoint chain changed")

    monkeypatch.setattr(wrapper.stream, "aggregate", fail)
    result = wrapper.aggregate(configured, shards=1)
    assert result["merge_error"]["type"] == "ValueError"
    assert not result["complete_declared_identity_field_scope"]


@pytest.mark.parametrize("index,shards", [(-1, 8), (8, 8), (0, 257), (0, 0)])
def test_invalid_shard_scope_refused(configured, index, shards):
    with pytest.raises(ValueError, match="shard"):
        wrapper.worker(configured, index, shards=shards)
