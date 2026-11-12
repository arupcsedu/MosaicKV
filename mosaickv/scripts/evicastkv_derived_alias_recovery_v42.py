"""Recover one already-derived V6 report across the known scratch alias pair.

Original artifacts, hashes, numerical classifications and historical paths are
preserved. Only the lexical /scratch/djy8hg versus /sfs/weka/scratch/djy8hg
spelling is normalized for comparison; this never expands filesystem scope.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path, PurePosixPath

import evicastkv_small_residuals_v41 as prior

ORIGINAL = prior.V6_ORIGINAL
DERIVED = prior.V6_DERIVED
ORIGINAL_SHA = prior.V6_SHA
PROVENANCE_SHA = prior.V6_PROVENANCE_SHA
REPORT_SHA = prior.V6_REPORT_SHA
PRODUCER = "799ca3c6c3eccadadb52688ebc3823cfbe66522f"
ALIAS_PREFIX = PurePosixPath("/scratch/djy8hg")
CANONICAL_PREFIX = PurePosixPath("/sfs/weka/scratch/djy8hg")
HASH = re.compile(r"[0-9a-f]{64}\Z")


def normalize(value):
    if not isinstance(value, str):
        raise ValueError("absolute recorded artifact path required")
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ValueError("normalized absolute recorded artifact path required")
    if path.is_relative_to(ALIAS_PREFIX):
        path = CANONICAL_PREFIX / path.relative_to(ALIAS_PREFIX)
    elif not path.is_relative_to(CANONICAL_PREFIX):
        raise ValueError("only the registered scratch prefix alias pair is permitted")
    return str(path)


def artifact_bindings(recorded, parent):
    """Compare known path spellings; never open or follow a manifest key."""
    if not isinstance(recorded, dict):
        raise ValueError("original artifact hash mapping required")
    result, aliases = {}, []
    directory = PurePosixPath(str(parent)) / "analysis"
    for original, expected in recorded.items():
        canonical = normalize(original)
        path = PurePosixPath(canonical)
        if (
            not path.is_relative_to(directory)
            or not isinstance(expected, str)
            or HASH.fullmatch(expected) is None
        ):
            raise ValueError("original artifact binding escapes its exact analysis scope")
        if canonical in result and result[canonical] != expected:
            raise ValueError("conflicting hashes for the same known path alias")
        result[canonical] = expected
        aliases.append(dict(recorded_path=original, canonical_path=canonical, sha256=expected))
    return result, aliases


def prove(row, roots, deadline):
    prior.check(deadline)
    original = Path(row["path"])
    if normalize(str(original)) != str(ORIGINAL):
        raise ValueError("only the exact registered truncated V6 report may be recovered")
    path = prior.canonical(original, roots)
    if str(path) != str(ORIGINAL):
        raise ValueError("known lexical alias does not resolve to its registered physical report")
    signature, bindings = prior.signature(path), {}
    if row.get("signature") is not None and row["signature"] != signature:
        raise ValueError("registered original report signature changed")
    prior.read(path, roots, deadline, bindings, ORIGINAL_SHA)
    provenance = prior.read_json(
        prior.read(
            DERIVED / "analysis_provenance.json",
            roots,
            deadline,
            bindings,
            PROVENANCE_SHA,
        )
    )
    report_raw = prior.read(DERIVED / "report.json", roots, deadline, bindings, REPORT_SHA)
    report, projected = prior.root_metadata(
        report_raw,
        {"status", "completed_rows", "expected_rows", "plan_sha256"},
        deadline=deadline,
    )
    parent = ORIGINAL.parent.parent
    recorded, aliases = artifact_bindings(provenance["original_analysis_artifacts"], parent)
    if (
        normalize(provenance["parent_root"]) != str(parent)
        or recorded.get(str(path)) != ORIGINAL_SHA
        or provenance["gpu_reruns"] != 0
        or provenance["original_analysis_preserved"] is not True
        or provenance["producer_artifacts_verified"] is not True
        or provenance["producer_git_sha"] != PRODUCER
        or report["status"] != "passed_execution"
        or report["completed_rows"] != report["expected_rows"]
        or report["completed_rows"] != 64
        or report["plan_sha256"] != provenance["parent_plan_sha256"]
    ):
        raise ValueError("existing V6 derived-recovery provenance differs")
    plan, _projected_plan = prior.root_metadata(
        prior.read(
            parent / "plan.json", roots, deadline, bindings, provenance["parent_plan_sha256"]
        ),
        {"source_sha"},
        deadline=deadline,
    )
    if plan["source_sha"] != PRODUCER:
        raise ValueError("original V6 GPU producer differs from the derived recovery")
    prior.read(
        DERIVED / "analysis-source.patch",
        roots,
        deadline,
        bindings,
        provenance["analyzer_patch_sha256"],
    )
    for name in ("samples.parquet", "summary.parquet"):
        expected = recorded.get(str(parent / "analysis" / name))
        if expected is None:
            raise ValueError("original saved-data hash is missing")
        prior.read(parent / "analysis" / name, roots, deadline, bindings, expected)
        prior.read(DERIVED / name, roots, deadline, bindings, expected)
    expected = row.get("current_file_sha256", row.get("sha256"))
    if expected is not None and expected != ORIGINAL_SHA:
        raise ValueError("registered original report hash differs")
    prior.read(path, roots, deadline, bindings, ORIGINAL_SHA)
    if prior.signature(path) != signature:
        raise ValueError("original report changed during the recovery proof")
    prior.check(deadline)
    return dict(
        proof_kind="existing_V6_derived_recovery_known_scratch_aliases_v42",
        path=str(original),
        canonical_path=str(path),
        signature=signature,
        current_file_sha256=ORIGINAL_SHA,
        full_file_sha256_before=ORIGINAL_SHA,
        full_file_sha256_after=ORIGINAL_SHA,
        file_hash_basis="two_complete_bounded_reads",
        identities=prior.collected(projected, path),
        bindings=bindings,
        source_bindings={
            str(Path(module).resolve()): hashlib.sha256(Path(module).read_bytes()).hexdigest()
            for module in (__file__, prior.__file__)
        },
        canonical_alias_normalization=dict(
            allowed_prefix_pair=[str(ALIAS_PREFIX), str(CANONICAL_PREFIX)],
            parent_recorded_path=provenance["parent_root"],
            parent_canonical_path=str(parent),
            original_artifact_bindings=aliases,
            original_manifest_rewritten=False,
            original_hashes_rewritten=False,
            nonregistered_alias_followed=False,
        ),
        original_error_sha256s=row.get("original_error_sha256s", []),
        historical_source_executed=False,
        historical_numeric_validity_claimed=False,
        credential_bodies_read=False,
        error_waived=False,
        old_unbound_historical_version_byte_identity_claimed=False,
        globally_fresh=False,
        historical_media_exposure_resolved=False,
        source_group_disjointness_verified=False,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
        paper_campaign_authorized=False,
    )
