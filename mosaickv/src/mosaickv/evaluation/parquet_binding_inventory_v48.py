"""Pure inventory of unresolved historical producer bindings, not recovery.

The caller supplies already hash-verified metadata. No path is opened, resolved,
or statted here, including protected credential fixtures. An auditor's source
SHA, a filename, a schema, and a current artifact SHA do not identify its original
producer. This inventory therefore resolves no occurrence and enables no work.
"""

from __future__ import annotations

from collections import Counter
from pathlib import PurePosixPath
from typing import Any

from mosaickv.evaluation import history_occurrence_reconciliation as contract
from mosaickv.evaluation import metadata_recovery_v41 as metadata
from mosaickv.evaluation.fox_campaign import digest
from mosaickv.evaluation.residual_occurrence_proofs import _key

PARQUET_ERROR = "Parquet has no safely projectable identity metadata columns"
REGISTRATION_FIELDS = contract.OCCURRENCE_FIELDS | {"path", "operation", "signature"}


def _protected(path: str) -> bool:
    value = PurePosixPath(path)
    return ".codex" in value.parts or value.name.casefold() in {"auth.json", "credentials.json"}


def _partition(parent: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    """Check the complete preserved partition without copying identity payloads."""
    if not isinstance(parent, dict) or any(
        parent.get(name) is not False for name in contract.FALSE_FIELDS
    ):
        raise ValueError("inventory cannot inherit positive scientific permissions")
    if (
        not isinstance(parent.get("source_sha"), str)
        or contract.SHA.fullmatch(parent["source_sha"]) is None
    ):
        raise ValueError("immutable parent auditor SHA required")
    groups: dict[str, dict[tuple[int, int], dict[str, Any]]] = {}
    for name in ("raw_error_records", "resolved_occurrences", "unresolved_occurrences"):
        values = parent.get(name)
        if not isinstance(values, list):
            raise ValueError("complete preserved occurrence lists required")
        rows = {}
        for row in values:
            if not isinstance(row, dict):
                raise ValueError("explicit occurrence dictionaries required")
            key = _key(row)
            if key in rows:
                raise ValueError("duplicate owner/ordinal occurrence")
            rows[key] = row
        groups[name] = rows
    raw = groups["raw_error_records"]
    resolved, unresolved = groups["resolved_occurrences"], groups["unresolved_occurrences"]
    if resolved.keys() & unresolved.keys() or resolved.keys() | unresolved.keys() != raw.keys():
        raise ValueError("preserved occurrence partition is incomplete")
    for owner in contract.OWNERS:
        ordinals = sorted(key[1] for key in raw if key[0] == owner)
        if ordinals != list(range(len(ordinals))):
            raise ValueError("preserved owner occurrence sequence has a gap")
    for key, row in raw.items():
        original = {
            name: value for name, value in row.items() if name not in contract.OCCURRENCE_FIELDS
        }
        if digest(contract._raw_error(original)) != row["original_error_sha256"]:
            raise ValueError("raw occurrence digest differs from preserved metadata")
        if key in unresolved and row != unresolved[key]:
            raise ValueError("unresolved occurrence differs from preserved raw evidence")
        if (
            key in resolved
            and resolved[key]["original_error_sha256"] != row["original_error_sha256"]
        ):
            raise ValueError("resolved occurrence digest differs from preserved metadata")
    counts = parent.get("counts")
    names = (
        "raw_error_count",
        "resolved_error_count",
        "net_unresolved_error_count",
        "original_seed_error_count",
        "newly_observed_error_count",
        "unresolved_original_occurrences",
        "unresolved_newly_observed_occurrences",
        "resolved_newly_observed_occurrences",
        "inherited_resolved_original_occurrences",
        "newly_resolved_original_occurrences",
    )
    if not isinstance(counts, dict) or any(
        type(counts.get(name)) is not int or counts[name] < 0 for name in names
    ):
        raise ValueError("strict complete nonnegative occurrence denominators required")
    if (
        counts["raw_error_count"] != len(raw)
        or counts["resolved_error_count"] != len(resolved)
        or counts["net_unresolved_error_count"] != len(unresolved)
        or len(raw) != len(resolved) + len(unresolved)
        or counts["original_seed_error_count"] + counts["newly_observed_error_count"] != len(raw)
        or counts["unresolved_original_occurrences"]
        + counts["unresolved_newly_observed_occurrences"]
        != len(unresolved)
        or counts["resolved_newly_observed_occurrences"]
        + counts["unresolved_newly_observed_occurrences"]
        != counts["newly_observed_error_count"]
        or counts["inherited_resolved_original_occurrences"]
        + counts["newly_resolved_original_occurrences"]
        + counts["unresolved_original_occurrences"]
        != counts["original_seed_error_count"]
    ):
        raise ValueError("preserved occurrence denominators do not conserve")
    return unresolved


def build_inventory(
    parent_report: dict[str, Any],
    registration_rows: list[dict[str, Any]],
    *,
    auditor_source_sha: str,
) -> dict[str, Any]:
    """Classify every remaining exact occurrence without deriving a proof.

    Registration may include completed historical rows. Only its exact entries
    matching currently unresolved occurrences are nominated. No producer claim
    is accepted; such a claim requires a separately reviewed provenance proof.
    """
    if (
        not isinstance(auditor_source_sha, str)
        or contract.SHA.fullmatch(auditor_source_sha) is None
    ):
        raise ValueError("immutable inventory auditor SHA required")
    unresolved = _partition(parent_report)
    if not isinstance(registration_rows, list):
        raise ValueError("explicit original registration list required")
    registered: dict[tuple[int, int], dict[str, Any]] = {}
    geometry: dict[str, list[int]] = {}
    for row in registration_rows:
        if not isinstance(row, dict) or set(row) != REGISTRATION_FIELDS:
            raise ValueError("exact original registration schema required")
        key = _key(row)
        if key in registered:
            raise ValueError("duplicate registered owner/ordinal occurrence")
        path = contract._path(row["path"], "registered residual", credential_allowed=True)
        signature = row["signature"]
        if (
            row["operation"] not in metadata.OPERATIONS
            or not isinstance(signature, list)
            or len(signature) != 4
            or any(type(value) is not int or value < 0 for value in signature)
        ):
            raise ValueError("registered operation or source geometry differs")
        if path in geometry and geometry[path] != signature:
            raise ValueError("conflicting signatures for one registered source")
        geometry[path] = signature
        registered[key] = row
    # Protected bindings remain metadata only; their presence never permits I/O.
    bindings = parent_report.get("source_bindings")
    if not isinstance(bindings, dict):
        raise ValueError("explicit parent source-binding dictionary required")
    for path, expected in bindings.items():
        contract._path(path, "preserved source binding", credential_allowed=True)
        contract._hash(expected, "preserved source binding")
    rows = []
    for key, occurrence in sorted(unresolved.items()):
        registration = registered.get(key)
        if (
            registration is None
            or registration["path"] != occurrence["path"]
            or registration["original_error_sha256"] != occurrence["original_error_sha256"]
        ):
            raise ValueError("unresolved occurrence has no exact original registration")
        path, error = occurrence["path"], occurrence["error"]
        protected = _protected(path)
        if protected:
            if registration["operation"] != "small" or error not in metadata.OPERATIONS["small"]:
                raise ValueError("unknown protected residual category")
            status = "protected_credential_not_read"
        else:
            if (
                registration["operation"] != "parquet"
                or error != PARQUET_ERROR
                or PurePosixPath(path).suffix != ".parquet"
            ):
                raise ValueError("unreviewed unresolved residual category")
            status = (
                "existing_artifact_hash_only" if path in bindings else "missing_artifact_binding"
            )
        rows.append(
            dict(
                **{name: occurrence[name] for name in contract.OCCURRENCE_FIELDS},
                path=path,
                signature=list(registration["signature"]),
                operation=registration["operation"],
                original_error=error,
                original_registration_sha256=digest(registration),
                status=status,
                existing_artifact_sha256=None if protected else bindings.get(path),
                original_producer_commit=None,
                original_producer_source_blob_sha256=None,
                original_producer_binding_status="missing_independent_producer_proof",
                source_body_read=False,
                occurrence_resolved=False,
            )
        )
    return dict(
        schema="historical_parquet_binding_inventory_v48",
        auditor_source_sha=auditor_source_sha,
        parent_auditor_source_sha=parent_report["source_sha"],
        rows=rows,
        counts=dict(
            parent_report["counts"],
            parquet_occurrences=sum(row["operation"] == "parquet" for row in rows),
            protected_occurrences=sum(
                row["status"] == "protected_credential_not_read" for row in rows
            ),
            existing_artifact_bindings=sum(
                row["status"] == "existing_artifact_hash_only" for row in rows
            ),
            missing_artifact_bindings=sum(
                row["status"] == "missing_artifact_binding" for row in rows
            ),
        ),
        status_counts=dict(sorted(Counter(row["status"] for row in rows).items())),
        newly_resolved_occurrences=0,
        source_bodies_read=0,
        GPU_jobs=0,
        metadata_only=True,
        producer_inferred_from_schema_filename_or_auditor=False,
        **dict.fromkeys(contract.FALSE_FIELDS, False),
    )
