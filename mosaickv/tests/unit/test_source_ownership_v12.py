from __future__ import annotations

import gzip
import json
from types import SimpleNamespace

import evicastkv_source_ownership_v12 as subject
import pytest


def test_raw_trace_and_snapshot_hashes_written_before_reducer(tmp_path):
    root = tmp_path
    (root / "worker").mkdir()
    (root / "protocol.json").write_text(json.dumps({"source_sha": "a" * 40}))
    capture = dict(
        records=[],
        stack_table=[],
        clock_calibration=None,
        complete_request=True,
        allocator_history_complete=True,
    )
    events = [dict(action="snapshot", sequence=0)]
    snapshot = dict(segments=[], device_traces=[events])
    receipt = subject.archive_before_attribution(root, capture, [], events, snapshot)
    assert receipt["attribution_not_yet_run"]
    assert receipt["new_roles_assigned"] == 0
    assert receipt["final_allocator_snapshot_sha256"] == subject.digest(snapshot)
    path = root / "worker/allocator_trace_pre_attribution.json.gz"
    assert receipt["trace_sha256"] == subject.sha(path)
    with gzip.open(path, "rt") as stream:
        saved = json.load(stream)
    assert saved["actual_events"] == events
    assert saved["final_allocator_snapshot"] == snapshot
    assert not receipt["exact_whole_request_KV_peak_certified"]
    with pytest.raises(FileExistsError):
        subject.archive_before_attribution(root, capture, [], events, snapshot)


def test_no_completed_archive_when_size_limit_exceeded(tmp_path, monkeypatch):
    path = tmp_path / "trace.json.gz"
    monkeypatch.setattr(subject, "MAX_TRACE_JSON_BYTES", 2)
    with pytest.raises(ValueError, match="exceeds 2 GiB"):
        subject._gzip_once(path, {"data": "not a real measurement"})
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_policy_is_separately_versioned_and_restored():
    old = subject.original.POLICY
    with subject._policy():
        assert subject.original.POLICY["observer_variant"] == subject.VARIANT
        assert subject.original.POLICY["predictions"] == old["predictions"]
        assert subject.original.POLICY["protection_fractions"] == old["protection_fractions"]
    assert subject.original.POLICY is old


def test_wrapper_archives_before_indexed_reduction_and_restores_hooks(tmp_path, monkeypatch):
    from mosaickv.evaluation import request_ownership

    order = []
    original_class = request_ownership._RequestSession
    monkeypatch.setattr(
        request_ownership,
        "_RequestSession",
        lambda *a, **k: SimpleNamespace(snapshot={"actual": "snapshot"}),
    )
    monkeypatch.setattr(
        subject, "archive_before_attribution", lambda *args: order.append("archive")
    )
    monkeypatch.setattr(
        subject, "indexed_conservative_report", lambda *a, **k: order.append("reduce")
    )

    def worker(*args):
        request_ownership._RequestSession()
        subject.original.conservative_report({}, [], events=[])
        return {"bounded": True}

    monkeypatch.setattr(subject.original, "worker", worker)
    original_reducer = subject.original.conservative_report
    assert subject.worker(tmp_path) == {"bounded": True}
    assert order == ["archive", "reduce"]
    assert subject.original.conservative_report is original_reducer
    assert (
        request_ownership._RequestSession is not original_class
    )  # fixture patch remains, not wrapper


def test_missing_actual_snapshot_is_not_fabricated(tmp_path):
    with pytest.raises(ValueError, match="actual final allocator snapshot"):
        subject.archive_before_attribution(tmp_path, {}, [], [], None)


def retry_fixture(tmp_path, monkeypatch):
    (tmp_path / "worker").mkdir()
    protocol = dict(
        source_root=str(tmp_path),
        source_sha="1aab16c63eee2f063194e6db8c9b3b5af466c3a1",
        bindings={},
        wall_minutes=20,
    )
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    (tmp_path / "worker/manifest.json").write_text(
        json.dumps(dict(source_sha=protocol["source_sha"], slurm_job_id="22796689"))
    )
    (tmp_path / "worker/raw.json").write_text(
        json.dumps(
            dict(
                arm=subject.original.ARM,
                counters=dict(updates=127),
                record=dict(output_tokens=128, storage=dict(history_updates_per_layer=[127] * 32)),
            )
        )
    )
    (tmp_path / "worker/logits.pt").write_bytes(b"hash-bound completed logits fixture")
    monkeypatch.setattr(subject.original, "clean_source", lambda *args: protocol["source_sha"])
    return "JobIDRaw|State|ElapsedRaw|TimelimitRaw\n22796689|TIMEOUT|1211|20\n"


def test_one_retry_only_for_proven_timeout_and_missing_trace_finality(tmp_path, monkeypatch):
    accounting = retry_fixture(tmp_path, monkeypatch)
    decision = subject.classify_retry(
        tmp_path, expected_job_id="22796689", accounting_text=accounting
    )
    assert decision["rerun"] and decision["one_retry_only"]
    assert decision["raw_trace_status"] == "missing"
    assert decision["completed_model_request"]
    assert set(decision["completed_request_sha256"]) == {"raw.json", "logits.pt"}


@pytest.mark.parametrize("defect", ["missing_logits", "incomplete_tokens", "wrong_arm"])
def test_model_load_or_incomplete_request_timeout_is_not_observer_retry(
    tmp_path, monkeypatch, defect
):
    accounting = retry_fixture(tmp_path, monkeypatch)
    if defect == "missing_logits":
        (tmp_path / "worker/logits.pt").unlink()
    else:
        path = tmp_path / "worker/raw.json"
        value = json.loads(path.read_text())
        if defect == "incomplete_tokens":
            value["record"]["output_tokens"] = 4
        else:
            value["arm"] = "fullkv"
        path.write_text(json.dumps(value))
    assert not subject.classify_retry(
        tmp_path, expected_job_id="22796689", accounting_text=accounting
    )["rerun"]


def test_eof_truncated_archive_is_recorded_and_retryable(tmp_path, monkeypatch):
    accounting = retry_fixture(tmp_path, monkeypatch)
    target = tmp_path / "worker/allocator_trace.json.gz"
    target.write_bytes(gzip.compress(b'{"metadata":"no answers"}')[:-4])
    before = target.read_bytes()
    decision = subject.classify_retry(
        tmp_path, expected_job_id="22796689", accounting_text=accounting
    )
    assert decision["rerun"]
    assert decision["raw_trace_status"] == "EOF_truncated_gzip"
    assert target.read_bytes() == before


def test_complete_gzip_needs_CPU_recovery_not_GPU_retry(tmp_path, monkeypatch):
    accounting = retry_fixture(tmp_path, monkeypatch)
    (tmp_path / "worker/allocator_trace.json.gz").write_bytes(gzip.compress(b"{}"))
    decision = subject.classify_retry(
        tmp_path, expected_job_id="22796689", accounting_text=accounting
    )
    assert not decision["rerun"]
    assert decision["raw_trace_status"] == "complete_gzip_CPU_recovery_only"


@pytest.mark.parametrize("state", ["FAILED", "COMPLETED", "CANCELLED", "TIMEOUT+"])
def test_other_states_never_authorize_retry(tmp_path, monkeypatch, state):
    accounting = retry_fixture(tmp_path, monkeypatch).replace("|TIMEOUT|", f"|{state}|")
    assert not subject.classify_retry(
        tmp_path, expected_job_id="22796689", accounting_text=accounting
    )["rerun"]


def test_source_hash_error_is_not_a_timeout_retry_waiver(tmp_path, monkeypatch):
    accounting = retry_fixture(tmp_path, monkeypatch)
    protocol = json.loads((tmp_path / "protocol.json").read_text())
    protocol["bindings"] = {str(tmp_path / "protocol.json"): "0" * 64}
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    outcome = subject.classify_retry(
        tmp_path, expected_job_id="22796689", accounting_text=accounting
    )
    assert not outcome["rerun"] and "hash changed" in outcome["error"]


def test_CRC_corruption_is_not_treated_as_timeout_truncation(tmp_path, monkeypatch):
    accounting = retry_fixture(tmp_path, monkeypatch)
    encoded = bytearray(gzip.compress(b"{}"))
    encoded[-8] ^= 1
    (tmp_path / "worker/allocator_trace.json.gz").write_bytes(encoded)
    outcome = subject.classify_retry(
        tmp_path, expected_job_id="22796689", accounting_text=accounting
    )
    assert not outcome["rerun"]
    assert outcome["status"] == "blocked_unverified_retry_scope"


def test_fast_gzip_policy_roundtrip_preserves_every_field(tmp_path):
    data = dict(all_records=[dict(pointer=17, generation=2, role="unknown")], no_field_drop=True)
    target = tmp_path / "trace.json.gz"
    subject._gzip_once(target, data)
    with gzip.open(target, "rt") as stream:
        assert json.load(stream) == data
    assert subject.POLICY["archive_gzip_compresslevel"] == 1
