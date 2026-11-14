"""Pure artifact-only bindings from one exact committed development document.

The I/O caller verifies the original/current Git objects before supplying their
small identity projection. The entire reviewed document, not a fuzzy Markdown
interpretation, is pinned here. Its dirty overlay's base commit and job IDs are
never treated as the output writer. No historical artifact or proof is opened.
"""

from __future__ import annotations

import copy
import hashlib
import re
from collections.abc import Mapping
from typing import Any

from mosaickv.evaluation import history_occurrence_reconciliation as contract
from mosaickv.evaluation.fox_campaign import digest
from mosaickv.evaluation.negative_seal_bindings_v53 import _canonical
from mosaickv.evaluation.parquet_binding_recovery_v49 import (
    CLAIM_FIELDS,
    MAXIMUM_CLAIMS,
    _signature,
    classify_binding,
)
from mosaickv.evaluation.source_archive_bindings_v51 import _bounded, _schema

DOCUMENT_COMMIT = "70c3437c12abc42c9836720ca4dcaa5c30ab5e57"
DOCUMENT_PATH = "mosaickv/docs/output_preserving_draft_v1_development_seal.md"
DOCUMENT_BLOB_OID = "cb07e5da2cf8897d48a8c6415e8a42c11297a8cf"
DOCUMENT_SHA256 = "5739aae2b02a56a50ca6c89f53b5b30d16a9a3ef806a38e9750092c67973c254"
DOCUMENT_SIZE = 2534
DOCUMENT_METADATA_PATH = "/sfs/weka/scratch/djy8hg/workdir/MosaicKV/" + DOCUMENT_PATH
DEVELOPMENT_CLASSIFICATION = "retrospective_development_only_not_paper_evidence"
RECORDED_RUN_ROOT = "/scratch/djy8hg/runs/mosaickv/output-preserving-draft-edb6edf-20261015-v1"
RUN_ROOT = "/sfs/weka" + RECORDED_RUN_ROOT
DOCUMENT_ARTIFACTS = {
    RUN_ROOT + "/analysis/aggregates.parquet": (
        "7d910324d263b4a13297919325dba87f55b050a78feeb0ed15f20a25c6be53b2"
    ),
    RUN_ROOT + "/analysis/cross_curve_tests.parquet": (
        "78a6c9ab2fef43ecbb062a43bc21bb6bf20c7f4db1364fb5df9cf1e4c10ea07f"
    ),
}
GIT_FIELDS = {
    "original_commit",
    "repository_path",
    "original_blob_oid",
    "current_blob_oid",
    "metadata_path",
}
_LABELS = {
    "Run root",
    "Slurm jobs",
    "Source base",
    "Source snapshot",
    "Plan SHA-256",
    "Input ledger SHA-256",
    "Analysis report SHA-256",
    "Aggregates SHA-256",
    "Cross-curve tests SHA-256",
    "Paired samples SHA-256",
}
_PARENT_FIELDS = (
    set(contract.FALSE_FIELDS)
    | contract.OCCURRENCE_FIELDS
    | {
        "schema",
        "path",
        "signature",
        "artifact_sha256",
        "original_registration_sha256",
        "status",
        "original_producer_commit",
        "original_producer_writer_path",
        "original_producer_source_blob_sha256",
        "committed_snapshot_bindings",
        "original_metadata_claim_sha256s",
        "insufficient_evidence_reasons",
        "occurrence_resolved",
        "newly_resolved_occurrences",
        "GPU_jobs",
    }
)


def _metadata_path(value: Any) -> str:
    path = _canonical(value, "original committed development document")
    if path != DOCUMENT_METADATA_PATH:
        raise ValueError("exact original main-repository document metadata path required")
    return path


def _document_fields(raw: bytes) -> dict[str, str]:
    """Closed label grammar; never map unrelated hashes to guessed filenames."""
    if not isinstance(raw, bytes) or not raw.endswith(b"\n") or b"\r" in raw:
        raise ValueError("complete LF-terminated document bytes required")
    try:
        text = raw.decode("ascii")
    except UnicodeError:
        raise ValueError("exact ASCII development document required") from None
    values = {}
    for line in text.splitlines():
        if not line.startswith("- "):
            continue
        match = re.fullmatch(r"- ([^:\n]+): (.+)", line)
        if match is None or match[1] not in _LABELS or match[1] in values:
            raise ValueError("unique exact reviewed development-seal labels required")
        values[match[1]] = match[2]
    if set(values) != _LABELS or values["Run root"] != f"`{RECORDED_RUN_ROOT}`":
        raise ValueError("exact reviewed document run root and complete labels required")
    for label in _LABELS:
        if label.endswith(" SHA-256") and re.fullmatch(r"`[0-9a-f]{64}`", values[label]) is None:
            raise ValueError("complete lowercase SHA256 document label required")
    for label, expected in zip(
        ("Aggregates SHA-256", "Cross-curve tests SHA-256"),
        DOCUMENT_ARTIFACTS.values(),
        strict=True,
    ):
        if values[label] != f"`{expected}`":
            raise ValueError("recorded document artifact hash differs from its reviewed mapping")
    return values


def _claim(metadata_path: Any, expected: str) -> dict[str, Any]:
    return dict(
        metadata_path=_metadata_path(metadata_path),
        metadata_sha256=DOCUMENT_SHA256,
        artifact_sha256=expected,
        role="artifact_only",
        recorded_source_sha=None,
        source_blob_sha256=None,
        writer_path=None,
        source_evidence_verified=False,
    )


def artifacts_from_seal(raw: bytes, git_facts: dict[str, Any]) -> list[dict[str, Any]]:
    """Return two V49 claims in the fixed ``DOCUMENT_ARTIFACTS`` path order."""
    _schema(git_facts, GIT_FIELDS, "original/current committed document identity")
    if any(
        git_facts[name] != expected
        for name, expected in (
            ("original_commit", DOCUMENT_COMMIT),
            ("repository_path", DOCUMENT_PATH),
            ("original_blob_oid", DOCUMENT_BLOB_OID),
            ("current_blob_oid", DOCUMENT_BLOB_OID),
        )
    ):
        raise ValueError("exact unchanged original/current document Git identities required")
    metadata = _metadata_path(git_facts["metadata_path"])
    if not isinstance(raw, bytes) or len(raw) != DOCUMENT_SIZE:
        raise ValueError("exact bounded 2534-byte original document required")
    if (
        hashlib.sha256(raw).hexdigest() != DOCUMENT_SHA256
        or hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() != DOCUMENT_BLOB_OID
    ):
        raise ValueError("complete original document bytes differ from the reviewed Git seal")
    _document_fields(raw)
    return [_claim(metadata, expected) for expected in DOCUMENT_ARTIFACTS.values()]


def expected_document_binding(
    previous: dict[str, Any],
    claim: dict[str, Any],
) -> dict[str, Any]:
    """Build the exact expected row from an independently pinned V54 parent.

    This is a view for full-row comparison, not a replacement for promotion's
    original classifier-input validation. No classifier or saved proof is
    replayed here. Every inherited field is copied without reinterpretation.
    """
    _schema(previous, _PARENT_FIELDS, "unchanged original V49 parent row")
    if (
        previous.get("schema") != "historical_parquet_binding_recovery_v49"
        or previous.get("status") != "unrecovered"
        or any(
            previous.get(name) is not None
            for name in (
                "original_producer_commit",
                "original_producer_writer_path",
                "original_producer_source_blob_sha256",
            )
        )
        or any(previous.get(name) is not False for name in contract.FALSE_FIELDS)
        or previous["occurrence_resolved"] is not False
        or type(previous["newly_resolved_occurrences"]) is not int
        or previous["newly_resolved_occurrences"] != 0
        or type(previous["GPU_jobs"]) is not int
        or previous["GPU_jobs"] != 0
        or previous["committed_snapshot_bindings"] != []
        or previous["insufficient_evidence_reasons"] != []
    ):
        raise ValueError("exact unchanged unrecovered V54 original V49 row required")
    _signature(previous["signature"])
    if (
        type(previous["owner_index"]) is not int
        or previous["owner_index"] not in contract.OWNERS
        or type(previous["historical_error_ordinal"]) is not int
        or previous["historical_error_ordinal"] < 0
    ):
        raise ValueError("exact original occurrence key required")
    for name in ("original_registration_sha256", "original_error_sha256", "artifact_sha256"):
        contract._hash(previous[name], "unchanged original parent " + name)
    _schema(claim, CLAIM_FIELDS, "original committed document artifact claim")
    path = previous["path"]
    expected = DOCUMENT_ARTIFACTS.get(path) if isinstance(path, str) else None
    if (
        expected is None
        or claim != _claim(claim["metadata_path"], expected)
        or (claim["source_evidence_verified"] is not False)
    ):
        raise ValueError("only the two exact artifact-only document bindings may advance")
    if previous["artifact_sha256"] != expected:
        raise ValueError("original artifact conflicts with the exact recorded document hash")
    old_claim_digests = previous["original_metadata_claim_sha256s"]
    if (
        not isinstance(old_claim_digests, list)
        or len(old_claim_digests) >= MAXIMUM_CLAIMS
        or any(not isinstance(value, str) for value in old_claim_digests)
    ):
        raise ValueError("bounded unchanged original claim digest list required")
    for value in old_claim_digests:
        contract._hash(value, "original metadata claim digest")
    claim_digest = digest(claim)
    if old_claim_digests != sorted(set(old_claim_digests)) or claim_digest in old_claim_digests:
        raise ValueError("unique canonical original/document claim digests required")
    result = copy.deepcopy(previous)
    result.update(
        schema="historical_document_seal_binding_v55",
        status="artifact_binding_only",
        original_metadata_claim_sha256s=sorted([*old_claim_digests, claim_digest]),
        parent_binding_sha256=digest(previous),
        original_document_seal_claim_sha256=claim_digest,
        original_document_commit=DOCUMENT_COMMIT,
        original_document_repository_path=DOCUMENT_PATH,
        original_document_git_blob_oid=DOCUMENT_BLOB_OID,
        original_document_sha256=DOCUMENT_SHA256,
        binding_extension_schema="original_committed_document_seal_v55",
        development_classification=DEVELOPMENT_CLASSIFICATION,
        historical_development_overlay_recorded_dirty=True,
        negative_result_preserved=True,
        quality_gate_pass=False,
        scientific_dependency_validation=False,
        reused_hash_transaction=True,
        canonical_eligible=False,
        clean_runtime_execution_claimed=False,
        occurrence_resolved=False,
        newly_resolved_occurrences=0,
        GPU_jobs=0,
        **dict.fromkeys(contract.FALSE_FIELDS, False),
    )
    _bounded(result)
    return result


def augment_document_binding(
    previous: dict[str, Any],
    original_row: dict[str, Any],
    transaction: dict[str, Any],
    original_claims: list[dict[str, Any]],
    claim: dict[str, Any],
    *,
    reviewed_writers: Mapping[tuple[str, str], str],
) -> dict[str, Any]:
    """Re-derive one unchanged V54 parent row, adding only its fixed byte claim."""
    result = expected_document_binding(previous, claim)
    old = classify_binding(
        original_row, transaction, original_claims, reviewed_writers=reviewed_writers
    )
    if digest(old) != digest(previous):
        raise ValueError("saved V54 parent differs from its unchanged original V49 inputs")
    new = classify_binding(
        original_row, transaction, [*original_claims, claim], reviewed_writers=reviewed_writers
    )
    expected_base = {name: result[name] for name in _PARENT_FIELDS}
    expected_base["schema"] = previous["schema"]
    if digest(new) != digest(expected_base):
        raise ValueError("document binding classifier changed an inherited original parent field")
    return result
