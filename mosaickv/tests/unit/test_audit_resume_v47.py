"""Proof-only recovery and exact checkpoint integration, with synthetic inputs."""

from copy import deepcopy

import evicastkv_audit_resume_v47 as resume
import pytest
from test_residual_occurrence_proofs import fixture


def inputs():
    parent, seeds = fixture()
    for row in parent["raw_error_records"]:
        if row["historical_error_ordinal"] == 0:
            row.update(
                path="/synthetic/source.json",
                error="JSON exceeds bounded projection size; not silently skipped",
            )
            row["original_error_sha256"] = resume.digest({k: row[k] for k in ("path", "error")})
    rows = []
    for row in parent["unresolved_occurrences"]:
        rows.append(
            {k: row[k] for k in resume.reader.core.contract.OCCURRENCE_FIELDS | {"path"}}
            | dict(operation="json_stream", signature=[1, 2, 30, 4])
        )
    plan = dict(rows=rows, retry_indices=[0, 1], seed_error_counts=seeds, source_sha="a" * 40)
    proof = dict(
        whole_file_sha256_before="d" * 64,
        whole_file_sha256_after="d" * 64,
        task_artifacts={},
        identity_shards=[],
    )
    return parent, plan, proof


def test_recovery_preserves_all_errors_and_identities_without_source_parsing(tmp_path, monkeypatch):
    parent, plan, proof = inputs()
    original = deepcopy(parent)

    def pair(root, plan, index):
        if index == 0:
            return deepcopy(proof)
        raise ValueError("synthetic failed original item")

    delta = {kind: set() for kind in resume.reader.identities.KINDS}
    delta.update(
        sample_ids={"new-id"}, scopes={("", "", "")}, scoped_identities={("", "", "", "new-id")}
    )
    monkeypatch.setattr(resume.reader, "pair", pair)
    monkeypatch.setattr(resume.reader, "exclusion_delta", lambda *a: deepcopy(delta))
    monkeypatch.setattr(resume.reader.common, "checked_task", lambda *a: dict(status="blocked"))
    monkeypatch.setattr(
        resume.reader.bounded,
        "project_file",
        lambda *a, **k: pytest.fail("historical JSON was reparsed"),
    )
    output = tmp_path / "derived"
    result = resume.reconcile(
        tmp_path / "producer", plan, output, parent, dict(parent["source_bindings"]), "c" * 40
    )
    assert parent == original
    assert result["newly_resolved_JSON_occurrences"] == 1
    assert result["passed_indices"] == [0]
    assert result["error_counts"]["raw_error_count"] == 3
    assert result["error_counts"]["net_unresolved_error_count"] == 1
    assert result["historical_source_bodies_reparsed"] == 0
    restored = resume._parent(
        output / "reconciled_history.json", result["reconciled_history_sha256"]
    )
    assert restored["raw_error_records"] == parent["raw_error_records"]
    assert restored["identities"]["sample_ids"] == ["new-id", "old-id"]
    assert restored["identities"]["scoped_identities"] == [["", "", "", "new-id"]]
    assert restored[resume.archives.INLINE] == []
    assert resume.archives.validate_external_receipts(restored)["receipt_count"] == 1
    assert all(restored[name] is False for name in resume.reader.core.contract.FALSE_FIELDS)
    assert (output / "failures.json").is_file()


def test_binding_collision_rejects_promotion_before_parent_mutation(tmp_path, monkeypatch):
    parent, plan, proof = inputs()
    original = deepcopy(parent)
    proof["task_artifacts"] = {"/metadata/parent.json": "f" * 64}
    monkeypatch.setattr(resume.reader, "pair", lambda *a: deepcopy(proof))
    monkeypatch.setattr(
        resume.reader,
        "exclusion_delta",
        lambda *a: {k: set() for k in resume.reader.identities.KINDS},
    )
    monkeypatch.setattr(resume.reader.common, "checked_task", lambda *a: dict(status="blocked"))
    result = resume.reconcile(
        tmp_path / "producer",
        plan,
        tmp_path / "derived",
        parent,
        dict(parent["source_bindings"]),
        "c" * 40,
    )
    assert parent == original
    assert result["newly_resolved_JSON_occurrences"] == 0
    assert result["error_counts"]["net_unresolved_error_count"] == 2


def test_limits_are_not_increased():
    assert resume.reader.SECONDS == 210
    assert resume.reader.BYTE_BOUND == 2 * 2**30
    assert resume.reader.SLOTS == 8
    assert resume.reader.common.history.MAX_STATE == 512 * 2**20
    assert resume.archives.MAXIMUM_BYTES == 512 * 2**20
    assert resume.MAX_ARTIFACT_BYTES == 64 * 2**30


def test_output_budget_is_checked_without_following_source_aliases(tmp_path, monkeypatch):
    (tmp_path / "data").write_bytes(b"x" * 30)
    assert resume.artifact_bytes(tmp_path) == 30
    monkeypatch.setattr(resume, "MAX_ARTIFACT_BYTES", 100)
    resume.require_artifact_capacity(tmp_path, 70)
    with pytest.raises(ValueError, match="aggregate artifact ceiling"):
        resume.require_artifact_capacity(tmp_path, 71)
    (tmp_path / "alias").symlink_to(tmp_path / "data")
    with pytest.raises(ValueError, match="only regular immutable"):
        resume.artifact_bytes(tmp_path)


def test_bulk_does_not_read_sources_when_canary_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(resume, "_load", lambda *a: dict(source_sha="a" * 40))
    monkeypatch.setattr(
        resume.reader.common.history,
        "read",
        lambda *a: dict(source_sha="a" * 40, streaming_bulk_authorized=False),
    )
    monkeypatch.setattr(resume.reader, "wave", lambda *a: pytest.fail("source work started"))
    assert resume.bulk(tmp_path)["status"] == "blocked_canary_failed"


def test_bulk_defers_before_artifact_ceiling_instead_of_relaxing_it(tmp_path, monkeypatch):
    plan = dict(
        source_sha="a" * 40,
        json_canary_indices=[],
        retry_indices=list(range(8)),
        bulk_wall_seconds=5400,
    )
    monkeypatch.setattr(resume, "_load", lambda *a: plan)
    monkeypatch.setattr(
        resume.reader.common.history,
        "read",
        lambda *a: dict(source_sha="a" * 40, streaming_bulk_authorized=True),
    )
    monkeypatch.setattr(resume, "artifact_bytes", lambda *a: resume.MAX_ARTIFACT_BYTES)
    monkeypatch.setattr(resume.reader, "wave", lambda *a: pytest.fail("over-budget work started"))
    assert resume.bulk(tmp_path)["status"] == "registered_artifact_bound_deferred"


def test_incomplete_expected_recovery_cannot_publish_a_success_checkpoint(tmp_path, monkeypatch):
    parent, plan, _proof = inputs()

    def missing(*args):
        raise ValueError("synthetic missing pair")

    monkeypatch.setattr(resume.reader, "pair", missing)
    monkeypatch.setattr(resume.reader.common, "checked_task", lambda *a: None)
    output = tmp_path / "derived"
    with pytest.raises(ValueError, match="all expected completed"):
        resume.reconcile(
            tmp_path / "producer",
            plan,
            output,
            parent,
            dict(parent["source_bindings"]),
            "c" * 40,
            expected_passes=1,
        )
    assert not (output / "decision.json").exists()
    assert not (output / "reconciled_history.json").exists()


def test_tampered_checkpoint_receipt_blocks_parent_loading(tmp_path, monkeypatch):
    parent, plan, proof = inputs()
    monkeypatch.setattr(resume.reader, "pair", lambda root, p, i: deepcopy(proof))
    monkeypatch.setattr(
        resume.reader,
        "exclusion_delta",
        lambda *a: {k: set() for k in resume.reader.identities.KINDS},
    )
    output = tmp_path / "derived"
    result = resume.reconcile(
        tmp_path / "producer", plan, output, parent, dict(parent["source_bindings"]), "c" * 40
    )
    raw = resume.reader.common.history.read(output / "reconciled_history.json")
    reference = raw[resume.archives.REFERENCES][0]
    with open(reference["path"], "ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError):
        resume._parent(output / "reconciled_history.json", result["reconciled_history_sha256"])
