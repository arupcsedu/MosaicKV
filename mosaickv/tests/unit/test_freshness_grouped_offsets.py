"""Synthetic immutable-source transactions, no historical audit execution."""

import hashlib
import json
import time

import evicastkv_freshness_grouped_offsets as grouped
import pytest


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return grouped.sha(path)


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "historical.jsonl"
    chunks = [
        json.dumps(
            dict(sample_id=f"consumed-{i}", answer="UNTOUCHED_LABEL", logits=list(range(10)))
        ).encode()
        + b"\n"
        for i in range(3)
    ]
    path.write_bytes(b"".join(chunks))
    rows, offset = [], 0
    for chunk in chunks:
        rows.append(dict(path=str(path), offset=offset, original_error_sha256="e" * 64))
        offset += len(chunk)
    segments = [dict(start=0, end=offset, sha256=hashlib.sha256(b"".join(chunks)).hexdigest())]
    binding = dict(
        signature=grouped.lines.signature(path),
        segments=segments,
        segmented_content_root=grouped.digest(segments),
    )
    return dict(path=path, rows=rows, binding=binding, roots=[tmp_path])


def project(source):
    return grouped.project_group(
        source["rows"], source["binding"], source["roots"], time.monotonic() + 60
    )


def test_same_parser_projection_equivalence_with_only_two_whole_file_hashes(source, monkeypatch):
    calls, original = [], grouped.lines.hash_source

    def hashing(*args):
        calls.append(args[0])
        return original(*args)

    monkeypatch.setattr(grouped.lines, "hash_source", hashing)
    outcome = project(source)
    assert len(calls) == 2
    for row, proof in zip(source["rows"], outcome["record_proofs"], strict=True):
        old = grouped.lines.project_record(
            row, source["binding"], source["roots"], time.monotonic() + 60
        )
        for key in ("line_sha256", "line_bytes", "projected_metadata_sha256"):
            assert proof[key] == old[key]
    assert [r["identities"]["sample_ids"] for r in outcome["records"]] == [
        [f"consumed-{i}"] for i in range(3)
    ]
    assert "UNTOUCHED_LABEL" not in json.dumps(outcome)


def test_failed_final_source_hash_returns_no_committed_metadata(source, monkeypatch):
    original = grouped.lines.hash_source
    calls = []

    def hashing(*args):
        calls.append(1)
        if len(calls) == 2:
            raise ValueError("synthetic final source mismatch")
        return original(*args)

    monkeypatch.setattr(grouped.lines, "hash_source", hashing)
    with pytest.raises(ValueError, match="final source mismatch"):
        project(source)
    assert len(calls) == 2


def test_interrupted_file_does_not_export_partially_projected_rows(source, monkeypatch):
    original, calls = grouped.lines.identity_projection, []

    def projector(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise grouped.lines.ProjectionPaused("synthetic bounded pause")
        return original(*args, **kwargs)

    monkeypatch.setattr(grouped.lines, "identity_projection", projector)
    original_sha = grouped.sha(source["path"])
    with pytest.raises(TimeoutError):
        project(source)
    assert grouped.sha(source["path"]) == original_sha


def test_duplicate_or_non_boundary_offsets_fail_closed(source):
    source["rows"].append(source["rows"][0])
    with pytest.raises(ValueError, match="unique sorted"):
        project(source)
    source["rows"] = [dict(source["rows"][0], offset=1)]
    with pytest.raises(ValueError, match="record boundary"):
        project(source)


@pytest.fixture
def prepared(tmp_path, source, monkeypatch):
    parent, root = tmp_path / "parent", tmp_path / "new"
    queue_path, bindings_path = (
        parent / "oversize_queue.json",
        parent / "recovery_source_bindings.json",
    )
    queue_sha = put(queue_path, source["rows"])
    source_sha = put(bindings_path, {str(source["path"]): source["binding"]})
    put(
        parent / "plan.json",
        dict(
            source_sha="a" * 40,
            bindings={},
            auxiliary_bindings={str(queue_path): queue_sha, str(bindings_path): source_sha},
            roots=list(map(str, source["roots"])),
        ),
    )
    identities = {key: [] for key in grouped.IDENTITIES}
    inventory_sha = put(parent / "combined_identity_inventory.json", dict(identities=identities))
    bound = {}
    for index in range(8):
        path = parent / "workers" / f"shard-{index:03d}.json"
        bound[str(path)] = put(
            path, dict(source_sha="a" * 40, recovery_observations=[], recovered_metadata=[])
        )
    put(
        parent / "GATE_freshness_frontier_fork.json",
        dict(
            source_sha="a" * 40,
            missing_shards=[],
            identity_inventory_sha256=inventory_sha,
            worker_bindings=bound,
        ),
    )
    monkeypatch.setattr(grouped, "clean_source", lambda *args: "b" * 40)
    monkeypatch.setattr(grouped, "cpu_gate", lambda *args: dict(status="passed"))
    grouped.prepare(root, cpu_gate_root=tmp_path, parent=parent)
    plan = grouped.load(root)
    return dict(root=root, parent=parent, index=plan["groups"][0]["index"], source=source)


def test_worker_and_aggregate_recover_offsets_without_altering_parent(prepared):
    before = grouped.sha(prepared["parent"] / "oversize_queue.json")
    for index in range(8):
        grouped.worker(prepared["root"], index)
    gate = grouped.analyze(prepared["root"])
    assert gate["newly_recovered_offsets"] == 3 and gate["pending_offsets"] == 0
    assert gate["status"] == "bounded_grouped_recovery_complete"
    assert gate["original_errors_preserved"] and gate["error_waivers"] == 0
    assert not gate["fresh_confirmation_authorized"]
    assert grouped.sha(prepared["parent"] / "oversize_queue.json") == before


def test_timeout_worker_retains_exact_group_and_offsets(prepared, monkeypatch):
    def pause(*args):
        raise TimeoutError("synthetic finite deadline")

    monkeypatch.setattr(grouped, "project_group", pause)
    result = grouped.worker(prepared["root"], prepared["index"])
    assert result["committed_groups"] == 0 and result["pending_groups"] == 1
    saved = grouped.read(prepared["root"] / "workers" / f"shard-{prepared['index']}.json")
    assert saved["pending_groups"][0]["rows"] == prepared["source"]["rows"]
    assert saved["committed_groups"] == []


def test_missing_worker_never_implies_complete_recovery(prepared):
    gate = grouped.analyze(prepared["root"])
    assert gate["missing_shards"] == list(range(8))
    assert gate["status"] == "bounded_grouped_recovery_incomplete"
    assert not gate["fresh_confirmation_authorized"]


def test_changed_parent_queue_blocks_worker(prepared):
    put(prepared["parent"] / "oversize_queue.json", [])
    with pytest.raises(ValueError, match="parent evidence changed"):
        grouped.worker(prepared["root"], prepared["index"])


def test_proof_backed_parent_recovery_is_not_reprocessed(prepared):
    parent, source = prepared["parent"], prepared["source"]
    old = grouped.lines.project_record(
        source["rows"][0], source["binding"], source["roots"], time.monotonic() + 60
    )
    proof = dict(source["rows"][0], **old)
    identities = grouped.empty_identities()
    identities["sample_ids"] = ["consumed-0"]
    path = parent / "workers/shard-000.json"
    worker_sha = put(
        path,
        dict(
            source_sha="a" * 40,
            recovery_observations=[proof],
            recovered_metadata=[dict(path=str(source["path"]), offset=0, identities=identities)],
        ),
    )
    gate_path = parent / "GATE_freshness_frontier_fork.json"
    gate = grouped.read(gate_path)
    gate["worker_bindings"][str(path)] = worker_sha
    put(gate_path, gate)
    root = prepared["root"].parent / "separate-successor"
    grouped.prepare(root, cpu_gate_root=root.parent, parent=parent)
    plan = grouped.load(root)
    assert plan["original_offsets"] == 3
    assert plan["previously_recovered_offsets"] == 1
    assert [r["offset"] for r in plan["groups"][0]["rows"]] == [
        r["offset"] for r in source["rows"][1:]
    ]


def test_missing_worker_offset_denominator_is_not_silently_dropped(prepared):
    report = grouped.analyze(prepared["root"])
    assert report["missing_worker_offsets"] == report["original_offsets"] == 3
