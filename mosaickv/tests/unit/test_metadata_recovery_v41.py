"""Synthetic occurrence receipts; no historical data or scheduler reads."""

from copy import deepcopy

import pytest
from test_residual_occurrence_proofs import fixture

from mosaickv.evaluation import metadata_recovery_v41 as reducer
from mosaickv.evaluation.fox_campaign import digest


def pair(row):
    registration = {k: row[k] for k in reducer.contract.OCCURRENCE_FIELDS | {"path"}}
    registration.update(operation="small", signature=[1, 2, 30, 4])
    receipt = dict(
        registration_sha256=digest(registration),
        operation="small",
        path=row["path"],
        signature=registration["signature"].copy(),
        proof_path="/metadata/proof.json",
        proof_sha256="f" * 64,
        whole_file_sha256_before="b" * 64,
        whole_file_sha256_after="b" * 64,
        source_hash_kind="file_bytes",
        old_unbound_historical_version_byte_identity_claimed=False,
        **dict.fromkeys(reducer.contract.FALSE_FIELDS, False),
    )
    return registration, receipt


def apply(parent, seeds, registrations, receipts):
    return reducer.apply_metadata_proofs(
        parent,
        registrations,
        receipts,
        source_sha="c" * 40,
        bindings={"/metadata/proof.json": "f" * 64},
        seed_error_counts=seeds,
    )


def test_receipt_resolves_only_its_occurrence_and_preserves_every_prior_record():
    parent, seeds = fixture()
    original = deepcopy(parent)
    reg, receipt = pair(parent["unresolved_occurrences"][0])
    result = apply(parent, seeds, [reg], [receipt])
    assert parent == original
    assert result["raw_error_records"] == parent["raw_error_records"]
    assert result["resolved_occurrences"][:1] == parent["resolved_occurrences"]
    assert result["unresolved_occurrences"] == [parent["unresolved_occurrences"][1]]
    assert result["counts"]["unresolved_original_occurrences"] == 0
    assert result["counts"]["unresolved_newly_observed_occurrences"] == 1
    assert all(result[name] is False for name in reducer.contract.FALSE_FIELDS)


@pytest.mark.parametrize(
    "change",
    [
        "hash",
        "signature",
        "boolean_signature",
        "operation",
        "path",
        "fresh",
        "extra",
    ],
)
def test_forged_or_scope_widening_receipt_rejected(change):
    parent, seeds = fixture()
    reg, receipt = pair(parent["unresolved_occurrences"][0])
    if change == "hash":
        receipt["whole_file_sha256_after"] = "e" * 64
    elif change == "signature":
        receipt["signature"][1] += 1
    elif change == "boolean_signature":
        receipt["signature"][0] = True
    elif change == "operation":
        receipt["operation"] = "parquet"
    elif change == "path":
        receipt["proof_path"] = "/metadata/unbound.json"
    elif change == "fresh":
        receipt["fresh_confirmation_authorized"] = True
    else:
        receipt["unreviewed_waiver"] = True
    with pytest.raises(ValueError):
        apply(parent, seeds, [reg], [receipt])


def test_duplicate_receipt_cannot_waive_identical_new_occurrence():
    parent, seeds = fixture()
    reg, receipt = pair(parent["unresolved_occurrences"][0])
    with pytest.raises(ValueError, match="unique unresolved"):
        apply(parent, seeds, [reg, reg], [receipt, receipt])


def test_new_identities_are_added_only_to_exclusions():
    parent, seeds = fixture()
    reg, receipt = pair(parent["unresolved_occurrences"][0])
    receipt["identities"] = {key: [] for key in reducer.contract.IDENTITY_FIELDS}
    receipt["identities"]["sample_ids"] = ["new-id"]
    result = apply(parent, seeds, [reg], [receipt])
    assert result["identities"]["sample_ids"] == ["new-id", "old-id"]
    assert result["identity_addition_counts"]["sample_ids"] == 1
