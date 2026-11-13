"""Synthetic manifest claims reuse the existing transaction classifier."""

from copy import deepcopy
from pathlib import Path

import pytest
from test_parquet_binding_recovery_v49 import fixture

from mosaickv.evaluation import manifest_bindings_v52 as manifest
from mosaickv.evaluation.fox_campaign import digest
from mosaickv.evaluation.parquet_binding_recovery_v49 import classify_binding


def arguments(index=0, *, old_claim=False):
    row, transaction, _, _ = fixture()
    path, spec = list(manifest.REGISTERED.items())[index]
    row["path"] = transaction["path"] = path
    registration = {
        name: row[name]
        for name in manifest.contract.OCCURRENCE_FIELDS | {"path", "signature", "operation"}
    }
    row["original_registration_sha256"] = digest(registration)
    transaction["before_sha256"] = transaction["after_sha256"] = spec["artifact_sha256"]
    claim = dict(
        spec, role="runtime_writer", writer_path=manifest.WRITER, source_evidence_verified=True
    )
    old_claims = (
        [
            dict(
                claim,
                role="input_source",
                writer_path=None,
                source_blob_sha256=None,
                source_evidence_verified=False,
            )
        ]
        if old_claim
        else []
    )
    old_registry = {}
    new_registry = {(spec["recorded_source_sha"], manifest.WRITER): spec["source_blob_sha256"]}
    previous = classify_binding(row, transaction, old_claims, reviewed_writers=old_registry)
    return dict(
        previous=previous,
        original_row=row,
        transaction=transaction,
        original_claims=old_claims,
        claim=claim,
        source_closure={
            manifest.WRITER: spec["source_blob_sha256"],
            manifest.GUARD: manifest.GUARD_SHA,
        },
        previous_reviewed_writers=old_registry,
        reviewed_writers=new_registry,
    )


@pytest.mark.parametrize("index", [0, 1, 2])
@pytest.mark.parametrize("old_claim", [False, True])
def test_registered_manifest_accepts_only_its_exact_writer_and_preserves_original(index, old_claim):
    values = arguments(index, old_claim=old_claim)
    original = deepcopy(values)
    result = manifest.augment_manifest_binding(**values)
    assert values == original
    assert result["schema"] == "historical_original_manifest_binding_v52"
    assert result["status"] == "runtime_producer_and_artifact_binding_verified"
    assert result["parent_binding_sha256"] == digest(values["previous"])
    assert result["artifact_sha256"] == values["previous"]["artifact_sha256"]
    assert result["original_producer_commit"] == values["claim"]["recorded_source_sha"]
    assert result["reused_hash_transaction"] is True
    assert result["newly_resolved_occurrences"] == result["GPU_jobs"] == 0
    assert result["occurrence_resolved"] is False
    assert all(result[name] is False for name in manifest.contract.FALSE_FIELDS)


def test_manifest_admission_has_no_filesystem_repository_or_scheduler_access(monkeypatch):
    values = arguments()

    def forbidden(*_args, **_kwargs):
        pytest.fail("pure manifest validator performed I/O")

    monkeypatch.setattr("builtins.open", forbidden)
    for method in ("open", "stat", "lstat", "resolve", "read_bytes", "read_text"):
        monkeypatch.setattr(Path, method, forbidden)
    assert manifest.augment_manifest_binding(**values)["reused_hash_transaction"] is True


@pytest.mark.parametrize(
    "field",
    [
        "metadata_path",
        "metadata_sha256",
        "artifact_sha256",
        "recorded_source_sha",
        "source_blob_sha256",
        "role",
        "writer_path",
        "source_evidence_verified",
    ],
)
def test_modified_or_mislabeled_original_claim_rejected(field):
    values = arguments()
    if field == "source_evidence_verified":
        values["claim"][field] = 1
    elif field == "role":
        values["claim"][field] = "input_source"
    elif field in {"metadata_path", "writer_path"}:
        values["claim"][field] = "/unexpected/path"
    else:
        values["claim"][field] = "f" * (40 if field == "recorded_source_sha" else 64)
    with pytest.raises(ValueError):
        manifest.augment_manifest_binding(**values)


@pytest.mark.parametrize(
    "change",
    [
        "unregistered_path",
        "fallback_label",
        "unknown_schema",
        "prior_producer",
        "changed_prior_hash",
        "missing_claims",
        "forged_claim",
        "forged_transaction",
        "prior_fresh",
        "extra_claim_field",
        "wrong_registry",
    ],
)
def test_unknown_target_or_forged_saved_transaction_cannot_be_promoted(change):
    values = arguments(old_claim=True)
    if change == "unregistered_path":
        values["original_row"]["path"] = "/scratch/djy8hg/runs/unregistered/summary.parquet"
    elif change == "fallback_label":
        values["previous"]["status"] = "fallback_artifact_binding_only"
    elif change == "unknown_schema":
        values["previous"]["schema"] = "unregistered"
    elif change == "prior_producer":
        values["previous"]["original_producer_commit"] = "f" * 40
    elif change == "changed_prior_hash":
        values["previous"]["artifact_sha256"] = "f" * 64
    elif change == "missing_claims":
        values["original_claims"] = []
    elif change == "forged_claim":
        values["original_claims"][0]["metadata_sha256"] = "f" * 64
    elif change == "forged_transaction":
        values["transaction"]["bytes_read"] -= 1
    elif change == "prior_fresh":
        values["previous"]["fresh_confirmation_authorized"] = True
    elif change == "extra_claim_field":
        values["claim"]["unregistered_override"] = True
    else:
        values["reviewed_writers"] = {}
    with pytest.raises(ValueError):
        manifest.augment_manifest_binding(**values)


@pytest.mark.parametrize(
    "field", ["path", "signature", "before_sha256", "after_sha256", "bytes_read", "seconds"]
)
def test_existing_transaction_contract_remains_authoritative(field):
    values = arguments()
    del values["transaction"][field]
    with pytest.raises(ValueError):
        manifest.augment_manifest_binding(**values)


@pytest.mark.parametrize("change", ["missing_guard", "wrong_guard", "wrong_writer", "extra_source"])
def test_complete_original_clean_source_guard_is_mandatory(change):
    values = arguments()
    if change == "missing_guard":
        del values["source_closure"][manifest.GUARD]
    elif change == "wrong_guard":
        values["source_closure"][manifest.GUARD] = "f" * 64
    elif change == "wrong_writer":
        values["source_closure"][manifest.WRITER] = "f" * 64
    else:
        values["source_closure"]["mosaickv/scripts/unreviewed.py"] = "f" * 64
    with pytest.raises(ValueError, match="guard closure"):
        manifest.augment_manifest_binding(**values)
