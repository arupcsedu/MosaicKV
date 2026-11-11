"""Synthetic finite recovery, without reading historical result bodies."""

import hashlib
import json
import time
from pathlib import Path

import evicastkv_pending_oversize_recovery as recovery
import pytest


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return recovery.sha(path)


def source_file(path, count):
    chunks = [
        json.dumps(
            dict(sample_id=f"{path.stem}-{i}", answer="PRIVATE_LABEL", logits=[1, 2, 3])
        ).encode()
        + b"\n"
        for i in range(count)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(chunks))
    rows, errors, offset = [], [], 0
    for chunk in chunks:
        error = dict(
            path=str(path),
            offset=offset,
            error="JSONL record exceeds identity-projection line bound",
        )
        rows.append(
            dict(
                path=str(path),
                offset=offset,
                original_error_sha256=recovery.digest(error),
                complete_source_binding_available=True,
                index=0,
            )
        )
        errors.append(error)
        offset += len(chunk)
    segments = [dict(start=0, end=offset, sha256=hashlib.sha256(b"".join(chunks)).hexdigest())]
    binding = dict(
        signature=recovery.grouped.lines.signature(path),
        segments=segments,
        segmented_content_root=recovery.digest(segments),
    )
    return rows, errors, binding


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    fork, parent, history = [tmp_path / name for name in ("fork", "grouped", "history")]
    inputs = [
        source_file(tmp_path / "bodies" / f"source-{i}.jsonl", count)
        for i, count in enumerate((1, 1, 2))
    ]
    rows = [row for r, _, _ in inputs for row in r]
    bindings = {r[0]["path"]: binding for r, _, binding in inputs}
    qsha = put(fork / "oversize_queue.json", rows)
    bsha = put(fork / "recovery_source_bindings.json", bindings)
    projected = [
        recovery.grouped.project_group(r, binding, [tmp_path], time.monotonic() + 60)
        for r, _, binding in inputs[:2]
    ]
    old_proof = dict(
        projected[0]["record_proofs"][0],
        status="projected_registered_metadata_only",
        whole_file_sha256_before=projected[0]["whole_file_sha256_before"],
        whole_file_sha256_after=projected[0]["whole_file_sha256_after"],
    )
    fork_workers = {}
    for i in range(8):
        path = fork / "workers" / f"shard-{i:03d}.json"
        fork_workers[str(path)] = put(
            path,
            dict(
                source_sha="a" * 40,
                recovery_observations=[old_proof] if i == 0 else [],
                recovered_metadata=projected[0]["records"] if i == 0 else [],
            ),
        )
    fsha = put(
        fork / "GATE_freshness_frontier_fork.json",
        dict(source_sha="a" * 40, missing_shards=[], worker_bindings=fork_workers),
    )
    groups = [dict(path=r[0]["path"], rows=r, index=0) for r, _, _ in inputs[1:]]
    group_workers = {}
    for i in range(8):
        path = parent / "workers" / f"shard-{i}.json"
        group_workers[str(path)] = put(
            path,
            dict(
                source_sha="b" * 40,
                committed_groups=[projected[1]] if i == 0 else [],
                pending_groups=[groups[1]] if i == 0 else [],
                blocked_groups=[],
            ),
        )
    inventory = {k: [] for k in recovery.grouped.IDENTITIES}
    inventory["sample_ids"] = ["fork-retained", "grouped-retained"]
    isha = put(parent / "combined_identity_inventory.json", dict(identities=inventory))
    put(
        parent / "GATE_grouped_offset_recovery.json",
        dict(
            source_sha="b" * 40,
            missing_shards=[],
            worker_bindings=group_workers,
            identity_inventory_sha256=isha,
            previously_recovered_offsets=1,
            newly_recovered_offsets=1,
            pending_offsets=2,
        ),
    )
    scanner_sha = recovery.sha(Path(recovery.grouped.lines.collect.__code__.co_filename))
    put(
        parent / "plan.json",
        dict(
            source_sha="b" * 40,
            parent=str(fork),
            source_bindings=bindings,
            bindings={
                str(fork / "oversize_queue.json"): qsha,
                str(fork / "recovery_source_bindings.json"): bsha,
                str(fork / "GATE_freshness_frontier_fork.json"): fsha,
                **fork_workers,
            },
            groups=groups,
            roots=[str(tmp_path)],
            unchanged_projector_sha256=scanner_sha,
            unchanged_collector_sha256=scanner_sha,
        ),
    )
    last = history / "segment-4"
    esha = put(last / "preserved_errors.json", [e for _, errors, _ in inputs for e in errors])
    inventory["sample_ids"] = ["final-history-retained"]
    hsha = put(last / "combined_identity_inventory.json", dict(identities=inventory))
    dsha = put(last / "decision.json", dict(source_sha="c" * 40, errors_sha256=esha))
    put(
        history / "GATE_history_closure_v16.json",
        dict(
            source_sha="c" * 40,
            last_segment=4,
            missing_shards=[],
            stop_reason="maximum_four_segments",
            fresh_confirmation_authorized=False,
            original_errors_preserved=True,
            roots_restarted=False,
            last_decision_sha256=dsha,
            identity_inventory_sha256=hsha,
            pending_frontier_items=99,
        ),
    )
    for name, value in dict(
        GROUPED=parent,
        HISTORY=history,
        GROUPED_SOURCE="b" * 40,
        FORK_SOURCE="a" * 40,
        HISTORY_SOURCE="c" * 40,
        OUTPUT_BASE=tmp_path,
        EXPECTED_GROUPS=1,
        EXPECTED_PENDING=2,
        EXPECTED_PREVIOUS=2,
        EXPECTED_ORIGINAL=4,
    ).items():
        monkeypatch.setattr(recovery, name, value)
    monkeypatch.setattr(recovery, "clean_source", lambda *args: "d" * 40)
    monkeypatch.setattr(recovery, "cpu_gate", lambda root, sha: dict(status="passed"))
    root = tmp_path / "recovery"
    recovery.prepare(root, cpu_gate_root=str(tmp_path))
    return dict(root=root, inputs=inputs, parent=parent, history=history, fork=fork)


def test_only_pending_group_is_reread_and_older_proofs_and_identities_survive(
    prepared, monkeypatch
):
    root = prepared["root"]
    calls, original = [], recovery.grouped.project_group

    def project(rows, *args):
        calls.append(rows[0]["path"])
        return original(rows, *args)

    monkeypatch.setattr(recovery.grouped, "project_group", project)
    for i in range(2):
        recovery.worker(root, i)
    gate = recovery.analyze(root)
    assert calls == [prepared["inputs"][2][0][0]["path"]]
    assert gate["newly_recovered_offsets"] == 2
    assert gate["total_recovered_offsets"] == 4
    assert gate["pending_frontier_items"] == 99
    assert gate["status"] == "pending_oversize_recovery_complete"
    assert not gate["fresh_confirmation_authorized"] and not gate["materialization_authorized"]
    assert gate["external_video_journals_unresolved"] and gate["original_errors_preserved"]
    inventory = recovery.read(root / "combined_identity_inventory.json")
    assert set(inventory["identities"]["sample_ids"]) == {
        "fork-retained",
        "grouped-retained",
        "final-history-retained",
        "source-2-0",
        "source-2-1",
    }
    overlay = recovery.read(root / "coverage_overlay.json")
    assert len(overlay["imported_proofs"]) == 2 and len(overlay["new_proofs"]) == 2
    assert "PRIVATE_LABEL" not in json.dumps(overlay)


def test_timeout_discards_incomplete_transaction_and_preserves_denominator(prepared, monkeypatch):
    def pause(*args):
        raise TimeoutError("PRIVATE_LABEL")

    monkeypatch.setattr(recovery.grouped, "project_group", pause)
    for i in range(2):
        recovery.worker(prepared["root"], i)
    gate = recovery.analyze(prepared["root"])
    assert gate["pending_offsets"] == 2 and gate["newly_recovered_offsets"] == 0
    assert gate["total_recovered_offsets"] == 2
    assert gate["status"] == "pending_oversize_recovery_incomplete"


def test_blocked_record_errors_never_export_payload(prepared, monkeypatch):
    def fail(*args):
        raise ValueError("PRIVATE_LABEL")

    monkeypatch.setattr(recovery.grouped, "project_group", fail)
    for i in range(2):
        recovery.worker(prepared["root"], i)
    gate = recovery.analyze(prepared["root"])
    assert gate["blocked_offsets"] == 2
    assert "PRIVATE_LABEL" not in (prepared["root"] / "workers/shard-0.json").read_text()


def test_missing_workers_stay_explicit(prepared):
    gate = recovery.analyze(prepared["root"])
    assert gate["missing_shards"] == [0, 1] and gate["missing_worker_offsets"] == 2
    assert gate["total_recovered_offsets"] == 2


@pytest.mark.parametrize("field", ["record_proofs", "records"])
def test_dropped_record_or_proof_fails_closed(prepared, field):
    root = prepared["root"]
    for i in range(2):
        recovery.worker(root, i)
    path = root / "workers/shard-0.json"
    result = recovery.read(path)
    result["committed_groups"][0][field].pop()
    put(path, result)
    with pytest.raises(ValueError, match="offset set differs"):
        recovery.analyze(root)


def test_changed_parent_gate_blocks_worker(prepared):
    put(prepared["history"] / "GATE_history_closure_v16.json", {})
    with pytest.raises(ValueError, match="immutable parent"):
        recovery.worker(prepared["root"], 0)


def test_changed_pending_source_never_commits(prepared):
    path = Path(prepared["inputs"][2][0][0]["path"])
    path.write_bytes(path.read_bytes() + b"\n")
    for i in range(2):
        recovery.worker(prepared["root"], i)
    gate = recovery.analyze(prepared["root"])
    assert gate["blocked_offsets"] == 2 and gate["newly_recovered_offsets"] == 0


def test_parent_partition_and_original_error_hashes_are_checked(prepared):
    gate_path = prepared["parent"] / "GATE_grouped_offset_recovery.json"
    gate = recovery.read(gate_path)
    gate["pending_offsets"] = 1
    put(gate_path, gate)
    with pytest.raises(ValueError, match="partition differs"):
        recovery.inspect_parents(prepared["parent"], prepared["history"])


def test_wrong_final_history_error_hash_denies_recovery(prepared):
    last = prepared["history"] / "segment-4"
    esha = put(last / "preserved_errors.json", [])
    decision = recovery.read(last / "decision.json")
    decision["errors_sha256"] = esha
    dsha = put(last / "decision.json", decision)
    gate_path = prepared["history"] / "GATE_history_closure_v16.json"
    gate = recovery.read(gate_path)
    gate["last_decision_sha256"] = dsha
    put(gate_path, gate)
    with pytest.raises(ValueError, match="original offset error"):
        recovery.inspect_parents(prepared["parent"], prepared["history"])


def test_index_and_write_once_worker_lock(prepared):
    with pytest.raises(ValueError, match="index"):
        recovery.worker(prepared["root"], True)
    recovery.worker(prepared["root"], 1)
    with pytest.raises(FileExistsError):
        recovery.worker(prepared["root"], 1)


def test_assignment_is_deterministic_and_finite_source_budget_enforced(prepared, monkeypatch):
    a = recovery.inspect_parents(prepared["parent"], prepared["history"])
    b = recovery.inspect_parents(prepared["parent"], prepared["history"])
    assert a["groups"] == b["groups"]
    assert a["worker_source_bytes"][0] == a["source_bytes"]
    monkeypatch.setattr(recovery, "MAX_TOTAL_SOURCE_BYTES", 1)
    with pytest.raises(ValueError, match="total source-byte budget"):
        recovery.inspect_parents(prepared["parent"], prepared["history"])
