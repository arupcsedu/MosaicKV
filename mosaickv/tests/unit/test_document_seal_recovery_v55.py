"""Synthetic transaction reuse; never open historical artifacts or credentials."""

from copy import deepcopy
from pathlib import Path

import evicastkv_document_seal_recovery_v55 as runner
import pytest

from mosaickv.evaluation.fox_campaign import digest
from mosaickv.evaluation.history_occurrence_reconciliation import OCCURRENCE_FIELDS
from mosaickv.evaluation.parquet_binding_recovery_v49 import classify_binding


@pytest.mark.parametrize("which", [0, 1])
def test_original_or_current_blob_mismatch_fails_before_source_body(monkeypatch, which):
    calls = []
    outputs = [runner.DOCUMENT_BLOB_OID, runner.DOCUMENT_BLOB_OID]
    outputs[which] = "0" * 40
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: None)

    def git(arguments, budget, **kwargs):
        calls.append(arguments)
        assert arguments[:2] == ["rev-parse", "--verify"]
        return outputs[len(calls) - 1].encode()

    monkeypatch.setattr(runner.packets.seals, "git_output", git)
    with pytest.raises(ValueError, match="before body access"):
        runner.original_document("a" * 40, runner.packets.prior.Budget())
    assert len(calls) == which + 1


def test_size_mismatch_rejects_before_document_body(monkeypatch):
    outputs = iter([runner.DOCUMENT_BLOB_OID.encode()] * 2 + [b"2535"])
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(runner.packets.seals, "git_output", lambda *a, **k: next(outputs))
    with pytest.raises(ValueError, match="document size"):
        runner.original_document("a" * 40, runner.packets.prior.Budget())


def test_original_source_receipt_matches_full_document_bytes(monkeypatch):
    # This is a tracked public source document, not a historical result body.
    raw = (runner.SOURCE / runner.DOCUMENT_PATH).read_bytes()
    outputs = iter([runner.DOCUMENT_BLOB_OID.encode()] * 2 + [str(len(raw)).encode(), raw])
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(runner.packets.seals, "git_output", lambda *a, **k: next(outputs))
    claims, facts = runner.original_document("a" * 40, runner.packets.prior.Budget())
    assert len(claims) == 2
    assert facts["original_commit"] == runner.DOCUMENT_COMMIT
    assert all(c["role"] == "artifact_only" and c["recorded_source_sha"] is None for c in claims)


def synthetic_inputs():
    from test_parquet_binding_recovery_v49 import fixture

    rows, previous, claims, transactions = [], [], {}, {}
    raw = (runner.SOURCE / runner.DOCUMENT_PATH).read_bytes()
    document_claims = runner.artifacts_from_seal(
        raw,
        dict(
            original_commit=runner.DOCUMENT_COMMIT,
            repository_path=runner.DOCUMENT_PATH,
            original_blob_oid=runner.DOCUMENT_BLOB_OID,
            current_blob_oid=runner.DOCUMENT_BLOB_OID,
            metadata_path=runner.DOCUMENT_METADATA_PATH,
        ),
    )
    for path, expected in runner.DOCUMENT_ARTIFACTS.items():
        row, txn, _, _ = fixture()
        row["path"] = txn["path"] = path
        txn["before_sha256"] = txn["after_sha256"] = expected
        row["original_registration_sha256"] = digest(
            {k: row[k] for k in OCCURRENCE_FIELDS | {"path", "signature", "operation"}}
        )
        previous.append(classify_binding(row, txn, [], reviewed_writers={}))
        rows.append(row)
        transactions[path] = txn
        claims[path] = []
    for index in range(132):
        row = deepcopy(rows[0])
        path = f"/synthetic/unchanged-{index}.parquet"
        row["path"] = path
        rows.append(row)
        value = deepcopy(previous[0])
        value.update(path=path, schema="unchanged_qualified_or_partial_binding")
        previous.append(value)
        transactions[path] = {}
    return rows, previous, claims, transactions, document_claims


def coordinator_report(monkeypatch, tmp_path):
    rows, previous, claims, transactions, document_claims = synthetic_inputs()
    parent = dict(
        source_sha=runner.PARENT_SOURCE,
        rows=previous,
        reused_hash_transactions=134,
        newly_resolved_occurrences=0,
        original_occurrence_counts={"semantic_unresolved": 135},
    )
    original = dict(rows=previous, binding_claims=claims)
    inventory = dict(rows=rows)
    packets = {
        runner.PARENT / "recovered_bindings.json": parent,
        runner.packets.startup.PARENT / "recovered_bindings.json": original,
        runner.packets.prior.PARENT / "inventory.json": inventory,
    }
    monkeypatch.setattr(runner, "clean_source", lambda *a: "a" * 40)
    monkeypatch.setattr(runner.packets.prior.resume.reader.common, "cpu_gate", lambda *a: None)
    monkeypatch.setattr(runner.packets, "saved_json", lambda path, *a, **k: deepcopy(packets[path]))
    facts = dict(
        original_commit=runner.DOCUMENT_COMMIT,
        repository_path=runner.DOCUMENT_PATH,
        original_blob_oid=runner.DOCUMENT_BLOB_OID,
        current_blob_oid=runner.DOCUMENT_BLOB_OID,
        metadata_path=runner.DOCUMENT_METADATA_PATH,
    )
    monkeypatch.setattr(runner, "original_document", lambda *a: (document_claims, facts))
    monkeypatch.setattr(runner.packets.prior, "allowed", lambda path: Path(path))
    geometry = {row["path"]: row["signature"] for row in rows}
    monkeypatch.setattr(runner.packets.prior, "geometry", lambda path: geometry[str(path)])
    monkeypatch.setattr(
        runner.packets.startup, "reused_transaction", lambda row, *a: transactions[row["path"]]
    )
    monkeypatch.setattr(runner.packets.prior.resume, "require_artifact_capacity", lambda *a: None)
    monkeypatch.setattr(runner.packets.prior, "REVIEWED", {})
    written = {}
    monkeypatch.setattr(
        runner, "write_once", lambda path, value: written.update({path.name: value})
    )
    result = runner.recover(tmp_path)
    report = written["recovered_bindings.json"]
    assert result["document_links_recovered"] == 2
    assert report["rows"][2:] == previous[2:]
    assert all(row["status"] == "artifact_binding_only" for row in report["rows"][:2])
    assert all(row["original_producer_commit"] is None for row in report["rows"][:2])
    assert report["reused_hash_transactions"] == 134
    assert report["parquet_bodies_reopened"] == 0
    assert report["credential_bodies_read"] is False
    assert report["fresh_confirmation_authorized"] is False
    return report, parent


def test_coordinator_recovers_only_two_and_preserves_132_without_original_io(monkeypatch, tmp_path):
    coordinator_report(monkeypatch, tmp_path)


@pytest.mark.parametrize(
    "change",
    [
        None,
        "unrelated_row",
        "artifact_hash",
        "signature",
        "runtime_claim",
        "canonical",
        "quality",
        "negative_result",
        "row_freshness",
        "report_freshness",
        "credential_read",
        "proof_replay",
        "bool_counter",
        "source",
        "duplicate",
        "row_order",
        "scientific_dependencies",
        "clean_runtime",
        "semantic_promotion",
        "semantic_count",
        "GPU_count",
        "parent_digest",
        "document_source",
        "document_claim",
        "claim_digest",
        "owner_index",
        "historical_ordinal",
        "error_hash",
        "registration_hash",
        "original_claims",
        "exclusion_reasons",
        "bool_owner",
        "bool_unchanged_owner",
        "bool_claim_flag",
    ],
)
def test_analysis_rechecks_scientific_boundaries_and_all_preserved_rows(
    monkeypatch, tmp_path, change
):
    report, parent = coordinator_report(monkeypatch, tmp_path)
    target = report["rows"][0]
    if change == "unrelated_row":
        report["rows"][-1]["status"] = "made_up_success"
    elif change == "artifact_hash":
        target["artifact_sha256"] = "0" * 64
    elif change == "signature":
        target["signature"][2] += 1
    elif change == "runtime_claim":
        target["original_producer_commit"] = "a" * 40
    elif change == "canonical":
        target["canonical_eligible"] = True
    elif change == "quality":
        target["quality_gate_pass"] = True
    elif change == "negative_result":
        target["negative_result_preserved"] = False
    elif change == "row_freshness":
        target[sorted(runner.contract.FALSE_FIELDS)[0]] = True
    elif change == "report_freshness":
        report["fresh_confirmation_authorized"] = "false"
    elif change == "credential_read":
        report["credential_bodies_read"] = True
    elif change == "proof_replay":
        report["original_JSON_identity_proofs_replayed"] = 1
    elif change == "bool_counter":
        report["parquet_bodies_reopened"] = False
    elif change == "source":
        report["source_sha"] = "b" * 40
    elif change == "duplicate":
        report["rows"].append(deepcopy(report["rows"][-1]))
    elif change == "row_order":
        report["rows"][10:12] = list(reversed(report["rows"][10:12]))
    elif change == "scientific_dependencies":
        target["scientific_dependency_validation"] = True
    elif change == "clean_runtime":
        target["clean_runtime_execution_claimed"] = True
    elif change == "semantic_promotion":
        target["occurrence_resolved"] = True
    elif change == "semantic_count":
        target["newly_resolved_occurrences"] = 1
    elif change == "GPU_count":
        target["GPU_jobs"] = 1
    elif change == "parent_digest":
        target["parent_binding_sha256"] = "e" * 64
    elif change == "document_source":
        report["document_source"]["original_commit"] = "e" * 40
    elif change == "document_claim":
        report["supplemental_document_claims"][target["path"]]["role"] = "runtime_writer"
    elif change == "claim_digest":
        target["original_document_seal_claim_sha256"] = "e" * 64
    elif change == "owner_index":
        target["owner_index"] = 999
    elif change == "historical_ordinal":
        target["historical_error_ordinal"] = 999
    elif change == "error_hash":
        target["original_error_sha256"] = (
            "0" * 64 if target["original_error_sha256"] != "0" * 64 else "1" * 64
        )
    elif change == "registration_hash":
        target["original_registration_sha256"] = "e" * 64
    elif change == "original_claims":
        target["original_metadata_claim_sha256s"] = []
    elif change == "exclusion_reasons":
        target["insufficient_evidence_reasons"] = ["fabricated"]
    elif change == "bool_owner":
        target["owner_index"] = bool(target["owner_index"])
    elif change == "bool_unchanged_owner":
        report["rows"][-1]["owner_index"] = bool(report["rows"][-1]["owner_index"])
    elif change == "bool_claim_flag":
        report["supplemental_document_claims"][target["path"]]["source_evidence_verified"] = 0
    monkeypatch.setattr(Path, "exists", lambda *a: True)
    monkeypatch.setattr(
        runner.packets,
        "saved_json",
        lambda path, *a, **k: deepcopy(parent if path.parent == runner.PARENT else report),
    )
    monkeypatch.setattr(runner.packets, "saved_sha", lambda *a: "c" * 64)
    if change is None:
        result = runner.analyze(tmp_path)
        assert result["other_132_rows_verbatim"] is True
        assert result["document_links_recovered"] == 2
        assert result["fresh_confirmation_authorized"] is False
    else:
        with pytest.raises(ValueError):
            runner.analyze(tmp_path)
