"""Pure artifact-only bindings from independently verified negative seals.

The caller verifies the original YAML bytes and pinned Git record before
projecting only the seal fields below, optionally including an explicit ledger
root. A seal's commit, protocol, run, or worker SHA is not runtime-writer
evidence. This module never opens artifacts, replays a hash transaction,
resolves occurrences, or authorizes experiments.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any

from mosaickv.evaluation import history_occurrence_reconciliation as contract
from mosaickv.evaluation.fox_campaign import digest
from mosaickv.evaluation.parquet_binding_recovery_v49 import (
    CLAIM_FIELDS,
    MAXIMUM_CLAIMS,
    _path,
    classify_binding,
)
from mosaickv.evaluation.source_archive_bindings_v51 import _bounded, _schema

SEAL_FIELDS = {
    "run_root",
    "artifacts",
    "classification",
    "immutable_result",
    "frozen_candidate_created",
}
CLASSIFICATION = "sealed_negative_label_blind_calibration_result"


def _canonical(value: Any, label: str, *, relative: bool = False) -> str:
    value = _path(value, label, relative=relative)
    if value.startswith("//"):
        raise ValueError(f"canonical single-root path required: {label}")
    if value == "/scratch/djy8hg" or value.startswith("/scratch/djy8hg/"):
        return "/sfs/weka" + value
    return value


def artifact_claim(
    seal_projection: dict[str, Any],
    metadata_path: str,
    metadata_sha256: str,
    target_path: str,
) -> dict[str, Any]:
    """Project one exact recorded Parquet hash without inferring its producer.

    Every artifact entry is checked, including non-target entries. Relative
    artifact paths use their explicitly named declared root, or the run root
    for legacy unqualified entries. Qualified absolute paths must remain in
    their named root; legacy absolute non-target scripts may be outside it.
    The exact target Parquet must still be inside the recorded run root. No
    filename aliases, basename matches, resolution, or path guessing occur.
    """
    optional_root = isinstance(seal_projection, dict) and "immutable_ledger_root" in seal_projection
    _schema(
        seal_projection,
        SEAL_FIELDS | ({"immutable_ledger_root"} if optional_root else set()),
        "negative seal projection",
    )
    if (
        seal_projection["classification"] != CLASSIFICATION
        or seal_projection["immutable_result"] is not True
        or seal_projection["frozen_candidate_created"] is not False
    ):
        raise ValueError("explicit immutable negative-result seal contract required")
    root = _canonical(seal_projection["run_root"], "recorded run root")
    roots = {"run_root": root}
    if optional_root:
        roots["immutable_ledger_root"] = _canonical(
            seal_projection["immutable_ledger_root"], "recorded immutable ledger root"
        )
    metadata = _canonical(metadata_path, "original negative seal")
    metadata_sha256 = contract._hash(metadata_sha256, "original negative seal")
    target = _canonical(target_path, "original target artifact")
    if PurePosixPath(target).suffix != ".parquet" or not PurePosixPath(target).is_relative_to(
        PurePosixPath(root)
    ):
        raise ValueError("exact target Parquet must be under its recorded run root")
    artifacts = seal_projection["artifacts"]
    if not isinstance(artifacts, dict) or not 0 < len(artifacts) <= MAXIMUM_CLAIMS:
        raise ValueError("bounded nonempty explicit seal artifact fieldmap required")
    recorded = {}
    for name, entry in artifacts.items():
        _path(name, "artifact field name", relative=True)
        qualified = isinstance(entry, dict) and "root" in entry
        _schema(
            entry,
            {"path", "sha256"} | ({"root"} if qualified else set()),
            "negative seal artifact entry",
        )
        entry_root = root
        if qualified:
            if not isinstance(entry["root"], str) or entry["root"] not in roots:
                raise ValueError("artifact must name an explicitly declared seal root")
            entry_root = roots[entry["root"]]
        path = entry["path"]
        if not isinstance(path, str):
            raise ValueError("explicit canonical artifact path required")
        if PurePosixPath(path).is_absolute():
            path = _canonical(path, "recorded artifact")
            if qualified and not PurePosixPath(path).is_relative_to(PurePosixPath(entry_root)):
                raise ValueError("qualified artifact path is outside its declared root")
        else:
            path = str(
                PurePosixPath(entry_root) / _canonical(path, "recorded artifact", relative=True)
            )
            _path(path, "recorded root-relative artifact")
        expected = contract._hash(entry["sha256"], "original sealed artifact")
        if path in recorded:
            raise ValueError("duplicate or conflicting canonical seal artifact path")
        recorded[path] = expected
    if target not in recorded:
        raise ValueError("target has no exact original negative-seal artifact record")
    return dict(
        metadata_path=metadata,
        metadata_sha256=metadata_sha256,
        artifact_sha256=recorded[target],
        role="artifact_only",
        recorded_source_sha=None,
        source_blob_sha256=None,
        writer_path=None,
        source_evidence_verified=False,
    )


def augment_seal_binding(
    previous: dict[str, Any],
    original_row: dict[str, Any],
    transaction: dict[str, Any],
    original_claims: list[dict[str, Any]],
    claims: list[dict[str, Any]],
    *,
    reviewed_writers: Mapping[tuple[str, str], str],
) -> dict[str, Any]:
    """Reuse a saved V49 transaction and add only artifact-only seal claims."""
    if not isinstance(previous, dict):
        raise ValueError("explicit saved original binding witness required")
    _bounded(previous)
    if (
        previous.get("schema") != "historical_parquet_binding_recovery_v49"
        or previous.get("status") not in {"unrecovered", "artifact_binding_only"}
        or any(
            previous.get(name) is not None
            for name in (
                "original_producer_commit",
                "original_producer_writer_path",
                "original_producer_source_blob_sha256",
            )
        )
        or any(previous.get(name) is not False for name in contract.FALSE_FIELDS)
    ):
        raise ValueError("only an unresolved original artifact-only binding may advance")
    if not isinstance(claims, list) or not 0 < len(claims) <= MAXIMUM_CLAIMS:
        raise ValueError("bounded nonempty original negative-seal claim list required")
    for claim in claims:
        if (
            not isinstance(claim, dict)
            or set(claim) != CLAIM_FIELDS
            or claim.get("role") != "artifact_only"
            or claim.get("source_evidence_verified") is not False
            or any(
                claim.get(name) is not None
                for name in ("recorded_source_sha", "source_blob_sha256", "writer_path")
            )
        ):
            raise ValueError("negative seals supply artifact-only claims, never runtime producers")
    old = classify_binding(
        original_row, transaction, original_claims, reviewed_writers=reviewed_writers
    )
    if digest(old) != digest(previous):
        raise ValueError("saved original witness differs from its bound classifier inputs")
    claim_digests = [digest(claim) for claim in claims]
    if len(set(claim_digests)) != len(claim_digests) or set(claim_digests) & set(
        old["original_metadata_claim_sha256s"]
    ):
        raise ValueError("duplicate original negative-seal claim")
    new = classify_binding(
        original_row,
        transaction,
        [*original_claims, *claims],
        reviewed_writers=reviewed_writers,
    )
    if new["status"] != "artifact_binding_only" or any(
        new[name] is not None
        for name in (
            "original_producer_commit",
            "original_producer_writer_path",
            "original_producer_source_blob_sha256",
        )
    ):
        raise ValueError("negative seal cannot establish an original runtime producer")
    result = copy.deepcopy(new)
    result.update(
        schema="historical_negative_seal_binding_v53",
        parent_binding_sha256=digest(previous),
        original_negative_seal_claim_sha256s=sorted(claim_digests),
        binding_extension_schema="original_negative_result_seal_v53",
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
