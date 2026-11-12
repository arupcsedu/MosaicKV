"""Pure application of caller-validated, occurrence-specific residual proofs.

This reducer never opens files or decides semantic producer fidelity. Its
caller validates the pinned YAML-event or closed-Parquet proof first. Identical
path/error hashes do not authorize any unregistered occurrence.
"""

from __future__ import annotations

import copy
from collections import Counter
from pathlib import PurePosixPath

from mosaickv.evaluation import history_occurrence_reconciliation as contract
from mosaickv.evaluation.fox_campaign import digest

REG_FIELDS = contract.OCCURRENCE_FIELDS | {"path", "kind", "signature", "file_sha256"}
RECEIPT_FIELDS = {
    "registration_sha256",
    "kind",
    "signature",
    "whole_file_sha256_before",
    "whole_file_sha256_after",
}


def _key(row):
    owner, ordinal = row.get("owner_index"), row.get("historical_error_ordinal")
    if type(owner) is not int or owner not in range(8) or type(ordinal) is not int or ordinal < 0:
        raise ValueError("strict original owner/ordinal occurrence key required")
    contract._hash(row.get("original_error_sha256"), "original occurrence")
    return owner, ordinal


def apply_proofs(
    parent_report,
    registrations,
    receipts,
    *,
    source_bindings,
    analysis_source_sha,
    seed_error_counts,
):
    """Return an additive report; every original input remains untouched."""
    if contract.SHA.fullmatch(str(analysis_source_sha)) is None:
        raise ValueError("immutable analysis source SHA required")
    seeds = contract._counts(seed_error_counts)
    if digest(seeds) != parent_report["input_metadata_sha256s"]["seed_error_counts"]:
        raise ValueError("owner seed boundaries differ from bound parent metadata")
    if any(parent_report.get(name) is not False for name in contract.FALSE_FIELDS):
        raise ValueError("residual proofs cannot inherit a freshness/authorization claim")
    if (
        not isinstance(registrations, list)
        or not isinstance(receipts, list)
        or len(registrations) != len(receipts)
    ):
        raise ValueError("complete registration/receipt lists required")
    raw = parent_report["raw_error_records"]
    raw_by_key = {_key(row): row for row in raw}
    unresolved = {_key(row): row for row in parent_report["unresolved_occurrences"]}
    resolved = {_key(row): row for row in parent_report["resolved_occurrences"]}
    if (
        len(raw_by_key) != len(raw)
        or len(unresolved) != len(parent_report["unresolved_occurrences"])
        or len(resolved) != len(parent_report["resolved_occurrences"])
        or resolved.keys() & unresolved.keys()
        or resolved.keys() | unresolved.keys() != raw_by_key.keys()
    ):
        raise ValueError("parent occurrence partition is incomplete or duplicated")
    for key, row in raw_by_key.items():
        error = {
            name: value for name, value in row.items() if name not in contract.OCCURRENCE_FIELDS
        }
        if digest(contract._raw_error(error)) != row["original_error_sha256"]:
            raise ValueError("original error hash differs from preserved raw evidence")
        if key in unresolved and unresolved[key] != row:
            raise ValueError("unresolved occurrence was rewritten")
        if (
            key in resolved
            and resolved[key]["original_error_sha256"] != row["original_error_sha256"]
        ):
            raise ValueError("resolved original hash differs from raw occurrence")
    for owner in range(8):
        ordinals = sorted(key[1] for key in raw_by_key if key[0] == owner)
        if ordinals != list(range(len(ordinals))) or seeds[str(owner)] > len(ordinals):
            raise ValueError("preserved owner occurrence sequence or seed boundary has a gap")
    if any(
        parent_report["counts"][name] != value
        for name, value in {
            "raw_error_count": len(raw),
            "resolved_error_count": len(resolved),
            "net_unresolved_error_count": len(unresolved),
            "original_seed_error_count": sum(seeds.values()),
        }.items()
    ):
        raise ValueError("parent occurrence denominators differ")
    bindings = copy.deepcopy(parent_report["source_bindings"])
    for path, expected in contract._bindings(source_bindings).items():
        if path in bindings and bindings[path] != expected:
            raise ValueError("immutable source binding collision")
        bindings[path] = expected
    identities = contract._identities(parent_report["identities"])
    old_identity_counts = {name: len(values) for name, values in identities.items()}
    additions, seen, grouped = [], set(), {}
    for registration, receipt in zip(registrations, receipts, strict=True):
        if (
            not isinstance(registration, dict)
            or set(registration) != REG_FIELDS
            or not isinstance(receipt, dict)
            or set(receipt) not in (RECEIPT_FIELDS, RECEIPT_FIELDS | {"identities"})
        ):
            raise ValueError("strict occurrence registration/receipt schema required")
        key = _key(registration)
        row = unresolved.get(key)
        if key in seen or row is None:
            raise ValueError("only a unique registered unresolved occurrence may be resolved")
        seen.add(key)
        path = contract._path(registration["path"], "residual source")
        expected = contract._hash(registration["file_sha256"], "registered source")
        signature = registration["signature"]
        if (
            registration["kind"] not in {"yaml_events", "closed_parquet"}
            or path != row["path"]
            or registration["original_error_sha256"] != row["original_error_sha256"]
            or not isinstance(signature, list)
            or len(signature) != 4
            or any(type(value) is not int or value < 0 for value in signature)
            or receipt["registration_sha256"] != digest(registration)
            or receipt["kind"] != registration["kind"]
            or digest(receipt["signature"]) != digest(signature)
            or receipt["whole_file_sha256_before"] != expected
            or receipt["whole_file_sha256_after"] != expected
        ):
            raise ValueError("occurrence/source/signature/transaction binding differs")
        if (
            registration["kind"] == "yaml_events"
            and (
                PurePosixPath(path).suffix not in {".yaml", ".yml"}
                or not row["error"].startswith("YAML requires")
            )
        ) or (
            registration["kind"] == "closed_parquet"
            and (
                PurePosixPath(path).suffix != ".parquet"
                or not row["error"].startswith("Parquet has no safely projectable")
            )
        ):
            raise ValueError("proof kind differs from the registered residual source category")
        geometry = digest(
            dict(signature=signature, file_sha256=expected, kind=registration["kind"])
        )
        if path in grouped and grouped[path] != geometry:
            raise ValueError("one source file has conflicting registration geometry")
        grouped[path] = geometry
        if path in bindings and bindings[path] != expected:
            raise ValueError("registered file contradicts a bound source hash")
        projected = contract._identities(receipt["identities"]) if "identities" in receipt else None
        if projected is not None:
            if registration["kind"] == "closed_parquet" and any(projected.values()):
                raise ValueError("closed no-identity producer cannot add identity claims")
            for name in contract.IDENTITY_FIELDS:
                identities[name].update(projected[name])
        additions.append(
            dict(
                owner_index=key[0],
                historical_error_ordinal=key[1],
                original_error_sha256=row["original_error_sha256"],
                resolution_kind="V40_occurrence_specific_" + registration["kind"],
                additive_proof_sha256=digest(receipt),
                registration_sha256=digest(registration),
            )
        )
    output = copy.deepcopy(parent_report)
    output.update(
        source_sha=analysis_source_sha,
        parent_source_sha=parent_report["source_sha"],
        parent_report_sha256=digest(parent_report),
        source_bindings=bindings,
    )
    output["resolved_occurrences"].extend(additions)
    output["unresolved_occurrences"] = [
        row for row in output["unresolved_occurrences"] if _key(row) not in seen
    ]
    output["unresolved_error_categories"] = dict(
        Counter(row["error"] for row in output["unresolved_occurrences"])
    )
    output["identities"] = {
        name: [list(value) for value in sorted(values)]
        if name in contract.NESTED_WIDTHS
        else sorted(values)
        for name, values in identities.items()
    }
    for name in identities:
        output["identity_addition_counts"][name] += (
            len(identities[name]) - old_identity_counts[name]
        )
    old_unresolved = sum(key[1] < seeds[str(key[0])] for key in unresolved.keys() - seen)
    original_count = sum(seeds.values())
    inherited = parent_report["counts"]["inherited_resolved_original_occurrences"]
    original_resolved = original_count - old_unresolved
    output["counts"].update(
        raw_error_count=len(raw),
        original_seed_error_count=original_count,
        newly_observed_error_count=len(raw) - original_count,
        inherited_resolved_original_occurrences=inherited,
        newly_resolved_original_occurrences=original_resolved - inherited,
        resolved_error_count=len(resolved) + len(additions),
        net_unresolved_error_count=len(unresolved) - len(seen),
        unresolved_original_occurrences=old_unresolved,
        unresolved_newly_observed_occurrences=len(unresolved) - len(seen) - old_unresolved,
    )
    # Correct the new-observation resolved denominator separately from residuals.
    output["counts"]["resolved_newly_observed_occurrences"] = (
        len(raw) - original_count - output["counts"]["unresolved_newly_observed_occurrences"]
    )
    output.setdefault("residual_occurrence_receipts", []).extend(
        dict(registration=copy.deepcopy(reg), receipt=copy.deepcopy(proof))
        for reg, proof in zip(registrations, receipts, strict=True)
    )
    output["additive_proof_sha256s"] = sorted(
        set(output.get("additive_proof_sha256s", [])) | {digest(proof) for proof in receipts}
    )
    output["identity_proof_sha256s"] = sorted(
        set(output.get("identity_proof_sha256s", []))
        | {digest(proof) for proof in receipts if "identities" in proof}
    )
    for field in contract.FALSE_FIELDS:
        output[field] = False
    return output
