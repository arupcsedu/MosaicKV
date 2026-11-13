"""Pure classification of externally verified historical artifact bindings.

Two current hashes establish stable artifact bytes, not their original writer.
Only an independently recorded runtime-writer claim, verified against the
reviewed immutable source blob, identifies a producer. No classification here
resolves a historical occurrence or grants scientific execution permissions.
The caller performs all I/O and must not manufacture historical claims from a
newly calculated hash or substitute the auditor's source for a producer.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any

from mosaickv.evaluation import history_occurrence_reconciliation as contract
from mosaickv.evaluation import parquet_binding_inventory_v48 as inventory
from mosaickv.evaluation.fox_campaign import digest
from mosaickv.evaluation.residual_occurrence_proofs import _key

MAXIMUM_INPUT_BYTES = 2 * 1024**3
MAXIMUM_SECONDS = 210
MAXIMUM_CLAIMS = 128
MAXIMUM_PATH_BYTES = 4096
TRANSACTION_FIELDS = {"path", "signature", "before_sha256", "after_sha256", "bytes_read", "seconds"}
CLAIM_FIELDS = {
    "metadata_path",
    "metadata_sha256",
    "artifact_sha256",
    "role",
    "recorded_source_sha",
    "source_blob_sha256",
    "writer_path",
    "source_evidence_verified",
}
ROLES = {"runtime_writer", "input_source", "committed_snapshot", "artifact_only"}
ROW_FIELDS = contract.OCCURRENCE_FIELDS | {
    "path",
    "signature",
    "operation",
    "original_error",
    "original_registration_sha256",
    "status",
    "existing_artifact_sha256",
    "original_producer_commit",
    "original_producer_source_blob_sha256",
    "original_producer_binding_status",
    "source_body_read",
    "occurrence_resolved",
}


def _path(value: Any, label: str, *, relative: bool = False) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value.encode("utf-8")) > MAXIMUM_PATH_BYTES
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
        or "\\" in value
    ):
        raise ValueError(f"bounded canonical path required: {label}")
    path = PurePosixPath(value)
    if (
        path.is_absolute() == relative
        or str(path) != value
        or ".." in path.parts
        or value == "."
        or ".codex" in path.parts
        or path.name.casefold() in {"auth.json", "credentials.json", ".netrc", ".env"}
    ):
        raise ValueError(f"noncredential canonical path required: {label}")
    return value


def _signature(value: Any) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(type(item) is not int or item < 0 for item in value)
    ):
        raise ValueError("strict four-integer artifact signature required")
    return value


def _commit(value: Any) -> str:
    if not isinstance(value, str) or contract.SHA.fullmatch(value) is None:
        raise ValueError("explicit immutable forty-character producer commit required")
    return value


def _reviewed_writers(value: Any) -> dict[tuple[str, str], str]:
    if not isinstance(value, Mapping) or len(value) > MAXIMUM_CLAIMS:
        raise ValueError("bounded reviewed-writer mapping required")
    result = {}
    for key, blob in value.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError("reviewed writer key must bind commit and repository path")
        source, path = key
        _commit(source)
        _path(path, "reviewed writer", relative=True)
        result[key] = contract._hash(blob, "reviewed writer source blob")
    return result


def classify_binding(
    row: dict[str, Any],
    transaction: dict[str, Any],
    claims: list[dict[str, Any]],
    *,
    reviewed_writers: Mapping[tuple[str, str], str],
) -> dict[str, Any]:
    """Classify one original inventory row without opening or changing anything.

    ``bytes_read`` counts both complete independent artifact-hash passes. The
    caller's ``seconds`` covers the complete bounded transaction. Claims are
    projected, hash-verified original metadata, never modern inferred records.
    Invalid/mismatched evidence raises; insufficient evidence remains unbound.
    """
    if not isinstance(row, dict) or set(row) != ROW_FIELDS:
        raise ValueError("exact V48 inventory row schema required")
    _key(row)
    path = _path(row["path"], "original artifact")
    signature = _signature(row["signature"])
    if (
        row["operation"] != "parquet"
        or PurePosixPath(path).suffix != ".parquet"
        or row["original_error"] != inventory.PARQUET_ERROR
        or row["occurrence_resolved"] is not False
        or row["source_body_read"] is not False
        or row["original_producer_commit"] is not None
        or row["original_producer_source_blob_sha256"] is not None
        or row["original_producer_binding_status"] != "missing_independent_producer_proof"
        or row["status"] not in {"existing_artifact_hash_only", "missing_artifact_binding"}
    ):
        raise ValueError("unresolved original Parquet inventory binding required")
    registered = {name: row[name] for name in contract.OCCURRENCE_FIELDS | {"path", "signature"}}
    registered["operation"] = row["operation"]
    if digest(registered) != contract._hash(
        row["original_registration_sha256"], "original registration"
    ):
        raise ValueError("original registration digest differs")
    if not isinstance(transaction, dict) or set(transaction) != TRANSACTION_FIELDS:
        raise ValueError("exact bounded two-hash transaction schema required")
    if (
        _path(transaction["path"], "transaction artifact") != path
        or _signature(transaction["signature"]) != signature
    ):
        raise ValueError("transaction artifact path or signature differs")
    actual = contract._hash(transaction["before_sha256"], "opening artifact hash")
    if contract._hash(transaction["after_sha256"], "closing artifact hash") != actual:
        raise ValueError("independent artifact hashes differ")
    seconds, bytes_read = transaction["seconds"], transaction["bytes_read"]
    if (
        type(bytes_read) is not int
        or bytes_read != 2 * signature[2]
        or bytes_read > MAXIMUM_INPUT_BYTES
        or type(seconds) not in (int, float)
        or not 0 <= seconds <= MAXIMUM_SECONDS
        or not math.isfinite(seconds)
    ):
        raise ValueError("complete two-pass artifact transaction exceeds unchanged limits")
    existing = row["existing_artifact_sha256"]
    if existing is not None and contract._hash(existing, "existing artifact binding") != actual:
        raise ValueError("existing artifact binding differs from transaction")
    if (existing is None) != (row["status"] == "missing_artifact_binding"):
        raise ValueError("inventory artifact-binding status is inconsistent")
    if not isinstance(claims, list) or len(claims) > MAXIMUM_CLAIMS:
        raise ValueError("bounded explicit original claim list required")
    writers = _reviewed_writers(reviewed_writers)
    runtime, snapshots, reasons, claim_digests = set(), set(), set(), set()
    for claim in claims:
        if not isinstance(claim, dict) or set(claim) != CLAIM_FIELDS:
            raise ValueError("exact original metadata claim schema required")
        _path(claim["metadata_path"], "original metadata")
        contract._hash(claim["metadata_sha256"], "original metadata")
        if contract._hash(claim["artifact_sha256"], "recorded artifact") != actual:
            raise ValueError("original artifact claim conflicts with transaction")
        role = claim["role"]
        if (
            not isinstance(role, str)
            or role not in ROLES
            or type(claim["source_evidence_verified"]) is not bool
        ):
            raise ValueError("explicit recognized claim role and boolean verification required")
        source, blob, writer = (
            claim["recorded_source_sha"],
            claim["source_blob_sha256"],
            claim["writer_path"],
        )
        if source is not None:
            _commit(source)
        if blob is not None:
            contract._hash(blob, "recorded source blob")
        if writer is not None:
            _path(writer, "recorded writer path", relative=True)
        claim_digests.add(digest(claim))
        verified = claim["source_evidence_verified"] and None not in (source, blob, writer)
        if role == "runtime_writer" and verified:
            if writers.get((source, writer)) == blob:
                runtime.add((source, writer, blob))
            else:
                reasons.add("runtime_writer_not_bound_to_reviewed_source_blob")
        elif role == "committed_snapshot" and verified:
            snapshots.add((source, writer, blob))
        elif role in {"runtime_writer", "committed_snapshot"}:
            reasons.add("recorded_source_evidence_unavailable_or_incomplete")
        elif role == "input_source":
            reasons.add("input_source_is_not_artifact_writer")
    if len(runtime) > 1:
        raise ValueError("conflicting original runtime producer claims")
    producer = next(iter(runtime), None)
    status = (
        "runtime_producer_and_artifact_binding_verified"
        if producer
        else "committed_snapshot_artifact_binding_verified"
        if snapshots
        else "artifact_binding_only"
        if claims or existing is not None
        else "unrecovered"
    )
    return dict(
        schema="historical_parquet_binding_recovery_v49",
        **{name: row[name] for name in contract.OCCURRENCE_FIELDS},
        path=path,
        signature=list(signature),
        artifact_sha256=actual,
        original_registration_sha256=row["original_registration_sha256"],
        status=status,
        original_producer_commit=producer[0] if producer else None,
        original_producer_writer_path=producer[1] if producer else None,
        original_producer_source_blob_sha256=producer[2] if producer else None,
        committed_snapshot_bindings=[
            dict(commit=source, path=writer, source_blob_sha256=blob)
            for source, writer, blob in sorted(snapshots)
        ],
        original_metadata_claim_sha256s=sorted(claim_digests),
        insufficient_evidence_reasons=sorted(reasons),
        occurrence_resolved=False,
        newly_resolved_occurrences=0,
        GPU_jobs=0,
        **dict.fromkeys(contract.FALSE_FIELDS, False),
    )
