"""Synthetic chains only: no historical outputs or model execution."""

import copy
import json
import subprocess
import time
from pathlib import Path

import evicastkv_derived_artifact_proofs as audit
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return audit.sha(path)


def parquet(path, *, extra=False):
    fields, arrays = [], []
    for key, kinds in audit.SCHEMAS[path.name].items():
        kind = next(k for k in ("int64", "double", "bool", "large_string") if k in kinds)
        dtype = {
            "int64": pa.int64(),
            "double": pa.float64(),
            "bool": pa.bool_(),
            "large_string": pa.large_string(),
        }[kind]
        value = {"int64": 0, "double": 0.5, "bool": True, "large_string": "internal_enum"}[kind]
        fields.append(pa.field(key, dtype))
        arrays.append(pa.array([value], type=dtype))
    if extra:
        fields.append(pa.field("sample_id", pa.string()))
        arrays.append(pa.array(["PRIVATE_ID"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_arrays(arrays, schema=pa.schema(fields)), path)


@pytest.fixture
def chain(tmp_path, monkeypatch):
    history, key = tmp_path / "history", "003-method-40-complete-r0"
    owner = history / "run/output/attempt"
    paths = [owner / "mechanism-terms" / key / name for name in sorted(audit.NAMES)]
    for path in paths:
        parquet(path)
    source = "c" * 40
    activity, record, worker = (
        owner / "mechanisms" / f"{key}.json",
        owner / "records" / f"{key}.json",
        owner / "worker.json",
    )
    activity_sha = put(
        activity, dict(raw_terms={str(p): audit.sha(p) for p in paths}, prompt="PRIVATE_PROMPT")
    )
    record_sha = put(
        record,
        dict(
            source_sha=source,
            mechanism_audit_path=str(activity),
            mechanism_audit_sha256=activity_sha,
            answer="PRIVATE_ANSWER",
            references=["PRIVATE_REFERENCE"],
            sample_id="PRIVATE_SAMPLE_ID",
        ),
    )
    put(worker, dict(source_sha=source, records=[dict(path=str(record), sha256=record_sha)]))
    monkeypatch.setattr(
        audit,
        "producer_proof",
        lambda revision, deadline: dict(
            recorded_producer_commit=revision,
            reviewed_AST_sha256s=copy.deepcopy(audit.TEMPLATES),
            source_blob_sha256s={},
            historical_source_executed=False,
            reviewed_template_commit=audit.REFERENCE_PRODUCER,
        ),
    )
    return dict(
        history=history, paths=paths, audit=activity, record=record, worker=worker, source=source
    )


def row(path):
    return dict(
        path=str(path),
        signature=audit.scanner.signature(path),
        original_error_sha256s=["e" * 64],
        registration_error=None,
        index=0,
    )


def proof(chain, path=None):
    return audit.prove(
        row(path or chain["paths"][0]), [str(chain["history"])], time.monotonic() + 60
    )


def rebind(chain):
    value = audit.read(chain["audit"])
    value["raw_terms"] = {str(path): audit.sha(path) for path in chain["paths"]}
    activity_sha = put(chain["audit"], value)
    value = audit.read(chain["record"])
    value["mechanism_audit_sha256"] = activity_sha
    record_sha = put(chain["record"], value)
    value = audit.read(chain["worker"])
    value["records"][0]["sha256"] = record_sha
    put(chain["worker"], value)


def test_closed_pair_footer_only_no_benchmark_values(chain, monkeypatch):
    monkeypatch.setattr(pq, "read_table", lambda *a, **k: pytest.fail("row decode forbidden"))
    for path in chain["paths"]:
        result = proof(chain, path)
        assert result["resolved_as"] == "derived_no_request_or_media_identity"
        assert result["parquet_footer"]["rows_decoded"] == 0
        assert result["whole_file_sha256_before"] == result["whole_file_sha256_after"]
        assert not result["input_independence_or_global_freshness_claimed"]
        assert "PRIVATE" not in json.dumps(result)


@pytest.mark.parametrize("broken", ["worker", "record", "audit", "artifact"])
def test_missing_or_stale_binding_unresolved(chain, broken):
    path = chain["paths"][0] if broken == "artifact" else chain[broken]
    if broken == "worker":
        path.unlink()
    else:
        path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises((OSError, ValueError)):
        proof(chain)


def test_identity_column_rejected_despite_rebound_hashes(chain):
    path = chain["paths"][0]
    parquet(path, extra=True)
    rebind(chain)
    with pytest.raises(ValueError, match="unexpected closed diagnostic schema"):
        proof(chain, path)


def test_string_id_in_numeric_column_rejected(chain):
    path = next(p for p in chain["paths"] if p.name == "edges.parquet")
    pq.write_table(
        pa.table(dict(source=["identity"], target=[0], weight=[0.5], edge_type=["enum"])), path
    )
    rebind(chain)
    with pytest.raises(ValueError, match="unexpected closed diagnostic schema"):
        proof(chain, path)


def test_unavailable_producer_is_not_substituted(chain, monkeypatch):
    def fail(*args):
        raise ValueError("unavailable reviewed producer")

    monkeypatch.setattr(audit, "producer_proof", fail)
    with pytest.raises(ValueError, match="unavailable reviewed producer"):
        proof(chain)


def test_producer_mismatch_and_duplicate_record_rejected(chain):
    value = audit.read(chain["worker"])
    value["source_sha"] = "f" * 40
    put(chain["worker"], value)
    with pytest.raises(ValueError, match="producer commits differ"):
        proof(chain)
    value["source_sha"] = chain["source"]
    value["records"].append(value["records"][0])
    put(chain["worker"], value)
    with pytest.raises(ValueError, match="exactly one"):
        proof(chain)


def test_basename_without_layout_is_not_a_proof(chain):
    path = chain["history"] / "unrelated" / "edges.parquet"
    parquet(path)
    with pytest.raises(ValueError, match="exact output layout"):
        proof(chain, path)


def test_outside_symlink_rejected(chain, tmp_path):
    outside = tmp_path / "edges.parquet"
    parquet(outside)
    path = chain["paths"][0]
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        proof(chain, path)


def test_deadline_and_mutation_withhold_whole_proof(chain, monkeypatch):
    with pytest.raises(TimeoutError):
        audit.prove(row(chain["paths"][0]), [str(chain["history"])], 0)
    original = audit.schema_proof

    def mutate(path):
        result = original(path)
        path.write_bytes(path.read_bytes() + b"changed")
        return result

    monkeypatch.setattr(audit, "schema_proof", mutate)
    with pytest.raises(ValueError, match="signature changed"):
        proof(chain)


def test_git_show_AST_check_never_executes_producer(monkeypatch):
    raw = b"def writer():\n    return 1\n"
    templates = {"source.py::writer": audit.ast_digest(raw, "writer")}
    monkeypatch.setattr(audit, "TEMPLATES", templates)
    calls = []

    def git(command, **kwargs):
        calls.append(command)
        return b"commit\n" if command[1] == "cat-file" else raw

    monkeypatch.setattr(subprocess, "check_output", git)
    assert (
        audit.producer_proof("a" * 40, time.monotonic() + 10)["reviewed_AST_sha256s"] == templates
    )
    assert calls == [["git", "cat-file", "-t", "a" * 40], ["git", "show", "a" * 40 + ":source.py"]]
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda command, **k: b"commit" if command[1] == "cat-file" else raw.replace(b"1", b"2"),
    )
    with pytest.raises(ValueError, match="differs"):
        audit.producer_proof("a" * 40, time.monotonic() + 10)


def test_git_tree_or_tag_object_cannot_substitute_for_recorded_commit(monkeypatch):
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: b"tree\n")
    with pytest.raises(ValueError, match="not a Git commit"):
        audit.producer_proof("a" * 40, time.monotonic() + 10)


@pytest.fixture
def prepared(chain, tmp_path, monkeypatch):
    parent, root = tmp_path / "parent", tmp_path / "new-proofs"
    put(parent / "plan.json", dict(source_sha="a" * 40, roots=[str(chain["history"])]))
    bindings, originals = {}, []
    for index in range(8):
        errors = (
            [dict(path=str(chain["paths"][index]), error=audit.PARQUET_ERROR)] if index < 2 else []
        )
        if index == 2:
            errors = [dict(path=str(chain["history"] / "unrelated.jsonl"), error="malformed JSONL")]
        originals.extend(errors)
        state = dict(errors=errors)
        target = parent / "workers" / f"shard-{index:03d}.json"
        bindings[str(target)] = put(
            target, dict(source_sha="a" * 40, state=state, state_sha256=audit.digest(state))
        )
    put(
        parent / audit.GATES[1],
        dict(source_sha="a" * 40, missing_shards=[], worker_bindings=bindings),
    )
    monkeypatch.setattr(audit, "clean_source", lambda *a: "b" * 40)
    monkeypatch.setattr(audit, "cpu_gate", lambda *a: dict(status="passed"))
    audit.prepare(root, cpu_gate_root=tmp_path, parent=parent)
    return root, parent, originals


def test_complete_proofs_preserve_errors_without_freshness_permission(prepared):
    root, parent, originals = prepared
    before = audit.sha(parent / "workers/shard-000.json")
    for index in range(8):
        audit.worker(root, index)
    gate = audit.analyze(root)
    assert gate["proved_files"] == 2 and gate["candidate_files"] == 2
    assert gate["original_errors_preserved"] and gate["error_waivers"] == 0
    assert not gate["fresh_confirmation_authorized"]
    assert not gate["complete_historical_identity_coverage"]
    assert audit.read(root / "original_errors.json") == originals
    assert audit.sha(parent / "workers/shard-000.json") == before
    with pytest.raises(FileExistsError):
        audit.analyze(root)


def test_reload_normalizes_JSON_gate_path(prepared, monkeypatch):
    root, _parent, _originals = prepared
    seen = []

    def gate(path, revision):
        assert isinstance(path, Path)
        seen.append(path)
        return {"status": "passed"}

    monkeypatch.setattr(audit, "cpu_gate", gate)
    audit.load(root)
    assert seen == [Path(audit.read(root / "plan.json")["cpu_gate_root"])]


def test_missing_workers_and_timeout_incomplete(prepared, monkeypatch):
    root, _parent, _originals = prepared

    def timeout(*args):
        raise TimeoutError()

    monkeypatch.setattr(audit, "prove", timeout)
    for index in range(7):
        audit.worker(root, index)
    gate = audit.analyze(root)
    assert gate["missing_shards"] == [7]
    assert gate["status"] == "derived_proof_queue_incomplete"
    assert gate["proved_files"] == 0


def test_mutated_parent_blocks_worker(prepared):
    root, parent, _originals = prepared
    path = parent / "workers/shard-000.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash changed"):
        audit.worker(root, 0)


def test_dropped_committed_row_rejected(prepared):
    root, _parent, _originals = prepared
    for index in range(8):
        audit.worker(root, index)
    for path in (root / "workers").iterdir():
        value = audit.read(path)
        if value["committed"]:
            value["committed"] = []
            put(path, value)
            break
    with pytest.raises(ValueError, match="omitted"):
        audit.analyze(root)


def test_provenance_mutation_between_worker_and_analyzer_fails(prepared, chain):
    root, _parent, _originals = prepared
    for index in range(8):
        audit.worker(root, index)
    chain["audit"].write_bytes(chain["audit"].read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash changed"):
        audit.analyze(root)


def test_arbitrary_footer_metadata_cannot_hide_identity(chain):
    path = chain["paths"][0]
    table = pq.read_table(path).replace_schema_metadata({b"sample_id": b"PRIVATE_ID"})
    pq.write_table(table, path)
    rebind(chain)
    with pytest.raises(ValueError, match="unexpected diagnostic footer metadata"):
        proof(chain)


def test_finite_queue_slices_and_imported_proofs_do_not_repeat_work(
    prepared, tmp_path, monkeypatch
):
    _old, parent, _errors = prepared
    monkeypatch.setattr(audit, "MAX_FILES", 1)
    first, second = tmp_path / "pass-one", tmp_path / "pass-two"
    outcome = audit.prepare(first, cpu_gate_root=tmp_path, parent=parent)
    assert outcome["files"] == 1 and outcome["deferred_files"] == 1
    first_queue = audit.load(first)["queue"]
    for index in range(8):
        audit.worker(first, index)
    gate = audit.analyze(first)
    assert gate["unattempted_files"] == 1 and gate["proved_files"] == 1
    outcome = audit.prepare(second, cpu_gate_root=tmp_path, parent=parent, previous=(first,))
    assert outcome["files"] == 1 and outcome["imported_proofs"] == 1
    assert audit.load(second)["queue"][0]["path"] != first_queue[0]["path"]
    for index in range(8):
        audit.worker(second, index)
    final = audit.analyze(second)
    assert final["candidate_files"] == 2 and final["proved_files"] == 2
    assert final["status"] == "derived_proof_queue_complete" and final["unattempted_files"] == 0


def test_terminal_unsupported_attempt_is_preserved_not_retried(prepared, tmp_path, monkeypatch):
    _old, parent, _errors = prepared
    monkeypatch.setattr(audit, "MAX_FILES", 1)
    first, second = tmp_path / "first-terminal", tmp_path / "remaining"
    audit.prepare(first, cpu_gate_root=tmp_path, parent=parent)
    original = audit.prove

    def unsupported(*args):
        raise ValueError("unsupported unchanged writer")

    monkeypatch.setattr(audit, "prove", unsupported)
    for index in range(8):
        audit.worker(first, index)
    gate = audit.analyze(first)
    assert gate["blocked_files"] == 1 and gate["unattempted_files"] == 1
    failed = audit.load(first)["queue"][0]["path"]
    monkeypatch.setattr(audit, "prove", original)
    outcome = audit.prepare(second, cpu_gate_root=tmp_path, parent=parent, previous=(first,))
    assert outcome["imported_terminal_unresolved"] == 1
    assert audit.load(second)["queue"][0]["path"] != failed
    for index in range(8):
        audit.worker(second, index)
    final = audit.analyze(second)
    assert final["status"] == "derived_evidence_queue_exhausted_with_unresolved"
    assert final["unattempted_files"] == 0 and final["blocked_files"] == 1
    assert not final["complete_historical_identity_coverage"]


def test_timeout_pending_is_retryable_but_mutated_previous_overlay_is_not(
    prepared, tmp_path, monkeypatch
):
    _old, parent, _errors = prepared
    first = tmp_path / "deadline-pass"
    audit.prepare(first, cpu_gate_root=tmp_path, parent=parent)
    original = audit.prove

    def pause(*args):
        raise TimeoutError()

    monkeypatch.setattr(audit, "prove", pause)
    for index in range(8):
        audit.worker(first, index)
    assert audit.analyze(first)["unattempted_files"] == 2
    monkeypatch.setattr(audit, "prove", original)
    second = tmp_path / "retry-pending"
    assert (
        audit.prepare(second, cpu_gate_root=tmp_path, parent=parent, previous=(first,))["files"]
        == 2
    )
    overlay = first / "coverage_overlay.json"
    overlay.write_bytes(overlay.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash changed"):
        audit.prepare(
            tmp_path / "must-block", cpu_gate_root=tmp_path, parent=parent, previous=(first,)
        )
