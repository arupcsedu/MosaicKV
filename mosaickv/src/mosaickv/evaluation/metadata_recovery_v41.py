"""Pure, occurrence-specific application of independently checked V41 proofs.

File parsing and producer validation are caller responsibilities. This reducer
preserves every raw occurrence and prior exclusion; it never authorizes a
cohort, even when all metadata errors have been resolved.
"""

from __future__ import annotations

import copy
from collections import Counter

from mosaickv.evaluation import history_occurrence_reconciliation as contract
from mosaickv.evaluation.fox_campaign import digest
from mosaickv.evaluation.residual_occurrence_proofs import _key, apply_proofs

OPERATIONS = {
    "json_stream": "JSON exceeds bounded projection size; not silently skipped",
    "parquet": "Parquet has no safely projectable identity metadata columns",
    "small": (
        "YAML requires separately reviewed lexical identity projection",
        "ValueError: invalid historical metadata scalar",
        "ValueError: truncated historical metadata JSON",
    ),
    "source_alias": "ValueError: symlink target outside allowed historical roots",
}
REGISTRATION_FIELDS = contract.OCCURRENCE_FIELDS | {"path", "operation", "signature"}
RECEIPT_FIELDS = set(contract.FALSE_FIELDS) | {
    "registration_sha256",
    "operation",
    "path",
    "signature",
    "proof_path",
    "proof_sha256",
    "whole_file_sha256_before",
    "whole_file_sha256_after",
    "source_hash_kind",
    "old_unbound_historical_version_byte_identity_claimed",
}


def apply_metadata_proofs(
    parent, registrations, receipts, *, source_sha, bindings, seed_error_counts
):
    # Reuse the complete old partition/hash/seed-boundary validation, without
    # pretending any new proof belongs to V40's two supported categories.
    result = apply_proofs(
        parent,
        [],
        [],
        source_bindings=bindings,
        analysis_source_sha=source_sha,
        seed_error_counts=seed_error_counts,
    )
    if len(registrations) != len(receipts):
        raise ValueError("one receipt per registered occurrence required")
    unresolved = {_key(row): row for row in parent["unresolved_occurrences"]}
    identities = contract._identities(parent["identities"])
    before = {name: len(values) for name, values in identities.items()}
    seen, additions = set(), []
    for registration, receipt in zip(registrations, receipts, strict=True):
        if set(registration) != REGISTRATION_FIELDS or set(receipt) not in (
            RECEIPT_FIELDS,
            RECEIPT_FIELDS | {"identities"},
        ):
            raise ValueError("exact V41 occurrence registration required")
        key = _key(registration)
        error = unresolved.get(key)
        if key in seen or error is None:
            raise ValueError("unique unresolved occurrence required")
        operation = registration["operation"]
        permitted = OPERATIONS.get(operation, ())
        permitted = (permitted,) if isinstance(permitted, str) else permitted
        if (
            error["error"] not in permitted
            or registration["path"] != error["path"]
            or registration["original_error_sha256"] != error["original_error_sha256"]
            or receipt.get("registration_sha256") != digest(registration)
            or receipt.get("operation") != operation
            or receipt.get("path") != registration["path"]
            or digest(receipt.get("signature")) != digest(registration["signature"])
            or receipt.get("old_unbound_historical_version_byte_identity_claimed") is not False
            or any(receipt.get(field) is not False for field in contract.FALSE_FIELDS)
        ):
            raise ValueError("proof differs from its exact registered occurrence/scope")
        signature = registration["signature"]
        contract._path(registration["path"], "registered metadata", credential_allowed=True)
        if (
            not isinstance(signature, list)
            or len(signature) != 4
            or any(type(v) is not int or v < 0 for v in signature)
        ):
            raise ValueError("strict source signature required")
        first = contract._hash(receipt.get("whole_file_sha256_before"), "first source hash")
        second = contract._hash(receipt.get("whole_file_sha256_after"), "second source hash")
        if first != second:
            raise ValueError("independent source hashes disagree")
        proof_path = contract._path(receipt.get("proof_path"), "semantic proof")
        proof_sha = contract._hash(receipt.get("proof_sha256"), "semantic proof")
        if result["source_bindings"].get(proof_path) != proof_sha:
            raise ValueError("semantic proof is not bound to this analysis")
        if receipt.get("source_hash_kind") != (
            "symlink_target_utf8" if operation == "source_alias" else "file_bytes"
        ):
            raise ValueError("source hash boundary differs")
        if "identities" in receipt:
            projected = contract._identities(receipt["identities"])
            for name, values in projected.items():
                identities[name].update(values)
        seen.add(key)
        additions.append(
            dict(
                owner_index=key[0],
                historical_error_ordinal=key[1],
                original_error_sha256=error["original_error_sha256"],
                resolution_kind="V41_occurrence_specific_" + operation,
                additive_proof_sha256=digest(receipt),
                registration_sha256=digest(registration),
            )
        )
    result["resolved_occurrences"].extend(additions)
    result["unresolved_occurrences"] = [
        row for row in result["unresolved_occurrences"] if _key(row) not in seen
    ]
    result["unresolved_error_categories"] = dict(
        Counter(row["error"] for row in result["unresolved_occurrences"])
    )
    for name, values in identities.items():
        result["identities"][name] = (
            [list(v) for v in sorted(values)] if name in contract.NESTED_WIDTHS else sorted(values)
        )
        result["identity_addition_counts"][name] += len(values) - before[name]
    old_left = sum(
        row["historical_error_ordinal"] < seed_error_counts[str(row["owner_index"])]
        for row in result["unresolved_occurrences"]
    )
    counts = result["counts"]
    counts.update(
        resolved_error_count=len(result["resolved_occurrences"]),
        net_unresolved_error_count=len(result["unresolved_occurrences"]),
        unresolved_original_occurrences=old_left,
        unresolved_newly_observed_occurrences=len(result["unresolved_occurrences"]) - old_left,
        newly_resolved_original_occurrences=(
            counts["original_seed_error_count"]
            - old_left
            - counts["inherited_resolved_original_occurrences"]
        ),
    )
    counts["resolved_newly_observed_occurrences"] = (
        counts["newly_observed_error_count"] - counts["unresolved_newly_observed_occurrences"]
    )
    result.setdefault("metadata_recovery_v41_receipts", []).extend(copy.deepcopy(receipts))
    result["additive_proof_sha256s"] = sorted(
        set(result.get("additive_proof_sha256s", [])) | {digest(r) for r in receipts}
    )
    result["identity_proof_sha256s"] = sorted(
        set(result.get("identity_proof_sha256s", []))
        | {digest(r) for r in receipts if "identities" in r}
    )
    for name in contract.FALSE_FIELDS:
        result[name] = False
    return result
