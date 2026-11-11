"""Synthetic producer-bound recovery; no historical Parquet execution."""

import copy
import json
from pathlib import Path

import evicastkv_native_v4_provenance_recovery as recovery
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return recovery.sha(path)


def parquet(path, extra=False):
    fields, arrays = [], []
    kinds = dict(
        int64=pa.int64(), double=pa.float64(), bool=pa.bool_(), large_string=pa.large_string()
    )
    values = dict(int64=0, double=0.5, bool=True, large_string="internal_enum")
    for field, allowed in recovery.derived.SCHEMAS[path.name].items():
        kind = next(k for k in kinds if k in allowed)
        fields.append(pa.field(field, kinds[kind]))
        arrays.append(pa.array([values[kind]], type=kinds[kind]))
    if extra:
        fields.append(pa.field("sample_id", pa.string()))
        arrays.append(pa.array(["PRIVATE_ID"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_arrays(arrays, schema=pa.schema(fields)), path)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    native, owner, key = (
        tmp_path / "native",
        tmp_path / "native/outputs/000/attempt",
        "000-method-40-r0",
    )
    paths = [owner / "mechanism-terms" / key / n for n in sorted(recovery.derived.NAMES)]
    for path in paths:
        parquet(path)
    audit_path, record_path, worker_path = (
        owner / "mechanisms" / f"{key}.json",
        owner / "records" / f"{key}.json",
        owner / "worker.json",
    )

    def rebind():
        audit_sha = put(
            audit_path,
            dict(raw_terms={str(p): recovery.sha(p) for p in paths}, prompt="PRIVATE_PROMPT"),
        )
        record_sha = put(
            record_path,
            dict(
                source_sha=recovery.PRODUCER,
                mechanism_audit_path=str(audit_path),
                mechanism_audit_sha256=audit_sha,
                answer="PRIVATE_ANSWER",
                references=["PRIVATE_REF"],
            ),
        )
        put(
            worker_path,
            dict(
                source_sha=recovery.PRODUCER,
                records=[dict(path=str(record_path), sha256=record_sha)],
            ),
        )

    rebind()
    errors = [dict(path=str(p), error=recovery.derived.PARQUET_ERROR) for p in paths]
    old_error = dict(path=str(native / "old/edges.parquet"), error=recovery.derived.PARQUET_ERROR)
    original_path = tmp_path / "parent/errors.json"
    put(original_path, [*errors, old_error])
    imported = [dict(original_error_sha256s=[recovery.digest(old_error)], old_proof_retained=True)]
    evidence = dict(
        bindings={str(original_path): recovery.sha(original_path)},
        imported_proofs=imported,
        covered_error_hashes=[recovery.digest(old_error)],
        original_errors_path=str(original_path),
        original_error_count=3,
        pending_frontier_items=9002,
        recipe_proofs_already_in_yaml=0,
    )

    def historical():
        return copy.deepcopy(evidence), {recovery.digest(e): e for e in [*errors, old_error]}

    monkeypatch.setattr(recovery, "historical_evidence", historical)
    monkeypatch.setattr(recovery, "NATIVE_ROOT", native)
    monkeypatch.setattr(recovery, "OUTPUT_BASE", tmp_path)
    monkeypatch.setattr(recovery, "EXPECTED_FILES", 2)
    monkeypatch.setattr(recovery, "EXPECTED_OWNERS", 1)
    monkeypatch.setattr(recovery, "EXPECTED_PRIOR", 1)
    monkeypatch.setattr(recovery, "clean_source", lambda *a: "a" * 40)
    monkeypatch.setattr(recovery, "cpu_gate", lambda root, revision: dict(status="passed"))
    monkeypatch.setattr(
        recovery.derived,
        "producer_proof",
        lambda rev, deadline: dict(
            recorded_producer_commit=rev,
            reviewed_AST_sha256s=copy.deepcopy(recovery.derived.TEMPLATES),
            source_blob_sha256s={},
            historical_source_executed=False,
            reviewed_template_commit=recovery.derived.REFERENCE_PRODUCER,
        ),
    )
    root = tmp_path / "recovery"
    recovery.prepare(root, cpu_gate_root=str(tmp_path))
    return dict(
        root=root,
        paths=paths,
        audit=audit_path,
        record=record_path,
        worker=worker_path,
        rebind=rebind,
        evidence=evidence,
    )


def test_exact_strict_validator_no_parquet_rows_and_prior_proofs_retained(prepared, monkeypatch):
    monkeypatch.setattr(
        pq, "read_table", lambda *a, **k: pytest.fail("Parquet row decode forbidden")
    )
    for i in range(2):
        recovery.worker(prepared["root"], i)
    gate = recovery.analyze(prepared["root"])
    assert gate["newly_proved_files"] == 2 and gate["previously_proved_files"] == 1
    assert gate["net_unresolved_after"] == 0
    assert gate["pending_frontier_items"] == 9002
    assert gate["status"] == "native_v4_provenance_recovery_complete"
    assert not gate["fresh_confirmation_authorized"] and not gate["materialization_authorized"]
    assert gate["external_video_journals_unresolved"] and gate["original_errors_preserved"]
    overlay = recovery.read(prepared["root"] / "coverage_overlay.json")
    assert overlay["imported_proofs"] == prepared["evidence"]["imported_proofs"]
    assert all(p["parquet_footer"]["rows_decoded"] == 0 for p in overlay["new_proofs"])
    assert "PRIVATE" not in json.dumps(overlay)


def test_timeout_preserves_exact_pending_registration(prepared, monkeypatch):
    def pause(*args):
        raise TimeoutError("PRIVATE_VALUE")

    monkeypatch.setattr(recovery.derived, "prove", pause)
    for i in range(2):
        recovery.worker(prepared["root"], i)
    gate = recovery.analyze(prepared["root"])
    assert gate["pending_files"] == 2 and gate["newly_proved_files"] == 0
    assert gate["net_unresolved_after"] == 2


def test_missing_workers_never_imply_complete(prepared):
    gate = recovery.analyze(prepared["root"])
    assert gate["missing_shards"] == [0, 1] and gate["missing_files"] == 2
    assert gate["status"] == "native_v4_provenance_recovery_incomplete"


def test_worker_failure_errors_never_export_payload(prepared, monkeypatch):
    def fail(*args):
        raise ValueError("PRIVATE_VALUE")

    monkeypatch.setattr(recovery.derived, "prove", fail)
    for i in range(2):
        recovery.worker(prepared["root"], i)
    gate = recovery.analyze(prepared["root"])
    assert gate["blocked_files"] == 2
    assert "PRIVATE_VALUE" not in (prepared["root"] / "workers/shard-0.json").read_text()


@pytest.mark.parametrize("which", ["worker", "record", "audit"])
def test_registered_provenance_mutation_blocks_execution(prepared, which):
    path = prepared[which]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="evidence changed"):
        recovery.worker(prepared["root"], 0)


def test_changed_artifact_is_blocked_not_silently_rebound(prepared):
    prepared["paths"][0].write_bytes(prepared["paths"][0].read_bytes() + b" ")
    for i in range(2):
        recovery.worker(prepared["root"], i)
    gate = recovery.analyze(prepared["root"])
    assert gate["blocked_files"] == 1 and gate["newly_proved_files"] == 1


def test_identity_schema_remains_rejected_with_fresh_registration(prepared, tmp_path):
    parquet(prepared["paths"][0], extra=True)
    prepared["rebind"]()
    root = tmp_path / "new-schema-retry"
    recovery.prepare(root, cpu_gate_root=tmp_path)
    for i in range(2):
        recovery.worker(root, i)
    assert recovery.analyze(root)["blocked_files"] == 1


def test_other_recorded_producer_never_accepted(prepared, tmp_path):
    value = recovery.read(prepared["worker"])
    value["source_sha"] = "b" * 40
    put(prepared["worker"], value)
    with pytest.raises(ValueError, match="exactly reviewed"):
        recovery.prepare(tmp_path / "must-not-prepare", cpu_gate_root=tmp_path)


def test_invalid_candidate_count_and_total_bytes_fail_before_plan(prepared, tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, "EXPECTED_FILES", 3)
    with pytest.raises(ValueError, match="678-file"):
        recovery.prepare(tmp_path / "wrong-count", cpu_gate_root=tmp_path)
    monkeypatch.setattr(recovery, "EXPECTED_FILES", 2)
    monkeypatch.setattr(recovery, "MAX_SOURCE_BYTES", 1)
    with pytest.raises(ValueError, match="total-byte"):
        recovery.prepare(tmp_path / "wrong-bytes", cpu_gate_root=tmp_path)


def test_dropped_or_altered_proof_never_passes(prepared):
    root = prepared["root"]
    for i in range(2):
        recovery.worker(root, i)
    path = root / "workers/shard-0.json"
    value = recovery.read(path)
    value["committed"][0]["whole_file_sha256_after"] = "0" * 64
    put(path, value)
    with pytest.raises(ValueError, match="proof differs"):
        recovery.analyze(root)


def test_source_cpu_paths_and_single_execution_lock(prepared, monkeypatch):
    paths = []
    monkeypatch.setattr(recovery, "cpu_gate", lambda path, revision: paths.append(path))
    recovery.worker(prepared["root"], 0)
    assert isinstance(paths[0], Path)
    with pytest.raises(FileExistsError):
        recovery.worker(prepared["root"], 0)
    with pytest.raises(ValueError, match="index"):
        recovery.worker(prepared["root"], True)


def test_historical_source_mismatch_rejected_before_reading_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, "HISTORY", tmp_path)
    put(tmp_path / "GATE_history_closure_v16.json", dict(source_sha="bad"))
    with pytest.raises(ValueError, match="four-segment history"):
        recovery.historical_evidence()


def test_error_path_binding_survives_canonical_history_alias(tmp_path, monkeypatch):
    actual, alias = tmp_path / "actual", tmp_path / "alias"
    (actual / "segment-4").mkdir(parents=True)
    alias.symlink_to(actual, target_is_directory=True)
    monkeypatch.setattr(recovery, "HISTORY", alias)
    monkeypatch.setattr(recovery, "EXPECTED_PRIOR", 1)
    error = dict(path="/known/old.parquet", error="old_error")
    error_sha = recovery.digest(error)
    prior = dict(
        source_proof=dict(reviewed_AST_sha256s=recovery.derived.TEMPLATES),
        whole_file_sha256_before="f" * 64,
        whole_file_sha256_after="f" * 64,
        resolved_as="derived_no_request_or_media_identity",
        original_error_sha256s=[error_sha],
    )

    def bound(path, bindings, expected=None):
        path = Path(path).resolve()
        bindings[str(path)] = "a" * 64
        name = path.name
        if name == "GATE_history_closure_v16.json":
            return dict(
                source_sha=recovery.HISTORY_SOURCE,
                last_segment=4,
                stop_reason="maximum_four_segments",
                missing_shards=[],
                fresh_confirmation_authorized=False,
                original_errors_preserved=True,
                last_decision_sha256="a" * 64,
                pending_frontier_items=9,
            )
        if name == "decision.json":
            return dict(source_sha=recovery.HISTORY_SOURCE, errors_sha256="a" * 64)
        if name == "preserved_errors.json":
            return [error]
        if name == "GATE_derived_artifact_proofs.json":
            return dict(
                source_sha=recovery.PRIOR_SOURCE,
                coverage_overlay_sha256="a" * 64,
                proved_files=1,
                missing_shards=[],
                blocked_files=0,
                unattempted_files=0,
                pending_files=0,
            )
        if name == "coverage_overlay.json" and path.parent == recovery.PRIOR.resolve():
            return dict(proofs=[prior])
        if name == "GATE_freshness_yaml_events.json":
            return dict(
                source_sha=recovery.YAML_SOURCE,
                missing_shards=[],
                projected_files=0,
                worker_bindings={str(tmp_path / f"shard-{i}.json"): "a" * 64 for i in range(8)},
            )
        if name.startswith("shard-"):
            return dict(source_sha=recovery.YAML_SOURCE, committed=[])
        if name == "GATE_pending_oversize_recovery.json":
            return dict(coverage_overlay_sha256="a" * 64)
        if name == "coverage_overlay.json":
            return dict(imported_proofs=[], new_proofs=[])
        if name == "GATE_freshness_recipe_recovery.json":
            return dict(coverage_overlay_sha256="a" * 64)
        if name == "indexed_coverage_overlay.json":
            return []
        pytest.fail(f"unexpected fixture metadata {name}")

    monkeypatch.setattr(recovery, "bound_read", bound)
    plan, _errors = recovery.historical_evidence()
    assert plan["original_errors_path"] in plan["bindings"]
    assert Path(plan["original_errors_path"]).parent == actual / "segment-4"
