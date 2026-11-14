#!/usr/bin/env python3
"""Recover two original document-seal links without replaying artifact proofs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import evicastkv_freeze_writer_recovery_v54 as packets

from mosaickv.evaluation import history_occurrence_reconciliation as contract
from mosaickv.evaluation.document_seal_bindings_v55 import (
    DOCUMENT_ARTIFACTS,
    DOCUMENT_BLOB_OID,
    DOCUMENT_COMMIT,
    DOCUMENT_METADATA_PATH,
    DOCUMENT_PATH,
    DOCUMENT_SHA256,
    DOCUMENT_SIZE,
    artifacts_from_seal,
    augment_document_binding,
    expected_document_binding,
)
from mosaickv.evaluation.fox_campaign import clean_source, digest, write_once

SOURCE = Path(__file__).resolve().parents[2]
PARENT = Path("/scratch/djy8hg/workdir/fox-review-v54/run-20261114-be8ce21-1")
PARENT_SHA = "d269af08516beabc45dc2cb0f7b6df8e0e5ba4fb3b5d18f95d8bb7a3c505b788"
PARENT_SOURCE = "be8ce216cdf7c57247f05c6c4bf1136d1be74f52"
COMPLETED_ROOT = Path("/scratch/djy8hg/workdir/fox-review-v55/run-20261114-36a5783-1")
COMPLETED_SOURCE = Path("/scratch/djy8hg/workdir/fox-review-v55-followup-code")
COMPLETED_SOURCE_SHA = "36a5783b39a3412d1fad49ca3833f1ccf8038a6d"
COMPLETED_REPORT_SHA = "896c548d51863545badc16965c3ed890779b69b23960d99dafc6388e988da277"
ANALYSIS_OUTPUT_BASE = Path("/scratch/djy8hg/workdir/fox-review-v55")


def original_document(revision, budget):
    """Check exact historical/current Git objects before reading source bytes."""
    subprocess.run(
        ["git", "--no-replace-objects", "merge-base", "--is-ancestor", DOCUMENT_COMMIT, revision],
        cwd=SOURCE,
        timeout=min(15, max(0.01, packets.prior.SECONDS - budget.check())),
        check=True,
    )
    address = f"{DOCUMENT_COMMIT}:{DOCUMENT_PATH}"
    identities = []
    for item in (address, f"{revision}:{DOCUMENT_PATH}"):
        value = (
            packets.seals.git_output(["rev-parse", "--verify", item], budget, maximum=128)
            .decode("ascii")
            .strip()
        )
        if value != DOCUMENT_BLOB_OID:
            raise ValueError("original/current document blob differs before body access")
        identities.append(value)
    size = int(packets.seals.git_output(["cat-file", "-s", address], budget, maximum=128))
    if size != DOCUMENT_SIZE:
        raise ValueError("exact bounded historical document size required")
    raw = packets.seals.git_output(["show", address], budget, maximum=size)
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != DOCUMENT_SHA256:
        raise ValueError("original historical document content differs")
    facts = dict(
        original_commit=DOCUMENT_COMMIT,
        repository_path=DOCUMENT_PATH,
        original_blob_oid=identities[0],
        current_blob_oid=identities[1],
        metadata_path=DOCUMENT_METADATA_PATH,
    )
    return artifacts_from_seal(raw, facts), facts


def recover(root):
    revision = clean_source(SOURCE)
    packets.prior.resume.reader.common.cpu_gate(root / "cpu", revision)
    budget = packets.prior.Budget()
    parent = packets.saved_json(
        PARENT / "recovered_bindings.json", PARENT_SHA, budget, output_root=PARENT
    )
    original = packets.saved_json(
        packets.startup.PARENT / "recovered_bindings.json", packets.startup.PARENT_SHA, budget
    )
    inventory = packets.saved_json(
        packets.prior.PARENT / "inventory.json",
        packets.prior.BOUND["inventory.json"],
        budget,
    )
    if (
        parent.get("source_sha") != PARENT_SOURCE
        or len(parent.get("rows", [])) != 134
        or parent.get("reused_hash_transactions") != 134
        or parent.get("newly_resolved_occurrences") != 0
    ):
        raise ValueError("exact completed V54 receipt required")
    claims, facts = original_document(revision, budget)
    if len(claims) != 2:
        raise ValueError("exactly two separately reviewed document targets required")
    claim_map = dict(zip(DOCUMENT_ARTIFACTS, claims, strict=True))
    old_rows = {row["path"]: row for row in original["rows"]}
    inventory_rows = {
        row["path"]: row for row in inventory["rows"] if row["operation"] == "parquet"
    }
    paths = {row["path"] for row in parent["rows"]}
    if (
        len(paths) != 134
        or paths != old_rows.keys()
        or paths != inventory_rows.keys()
        or not claim_map.keys() <= paths
    ):
        raise ValueError("all original saved transactions must remain present")
    output_rows = []
    for previous in parent["rows"]:
        path = previous["path"]
        canonical = packets.prior.allowed(path)
        if str(canonical) != path or packets.prior.geometry(canonical) != previous["signature"]:
            raise ValueError("historical artifact changed; do not repeat its completed proof")
        transaction = packets.startup.reused_transaction(
            inventory_rows[path], old_rows[path], original
        )
        if path in claim_map:
            previous = augment_document_binding(
                previous,
                inventory_rows[path],
                transaction,
                original["binding_claims"][path],
                claim_map[path],
                reviewed_writers=packets.prior.REVIEWED,
            )
        output_rows.append(previous)
    result = dict(
        source_sha=revision,
        parent_report_path=str(PARENT / "recovered_bindings.json"),
        parent_report_sha256=PARENT_SHA,
        document_source=facts,
        document_sha256=DOCUMENT_SHA256,
        rows=output_rows,
        supplemental_document_claims=claim_map,
        promoted_paths=sorted(claim_map),
        status_counts=dict(Counter(row["status"] for row in output_rows)),
        original_occurrence_counts=parent["original_occurrence_counts"],
        reused_hash_transactions=134,
        other_rows_verbatim=132,
        original_JSON_identity_proofs_replayed=0,
        parquet_bodies_reopened=0,
        historical_metadata_bodies_reopened=0,
        historical_logs_reopened=0,
        Slurm_records_queried=0,
        credential_bodies_read=False,
        historical_code_executed=False,
        combined_input_bytes=budget.input_bytes,
        wall_seconds=budget.check(),
        newly_resolved_occurrences=0,
        GPU_jobs=0,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
        paper_campaign_authorized=False,
        historical_media_exposure_resolved=False,
        scientific_quality_claim_established=False,
    )
    packets.prior.projected_bound(result)
    packets.prior.resume.require_artifact_capacity(root, packets.prior.LIVE_BYTES)
    write_once(root / "recovered_bindings.json", result)
    return dict(
        status_counts=result["status_counts"],
        document_links_recovered=2,
        GPU_jobs=0,
        fresh_confirmation_authorized=False,
    )


def analyze(root):
    revision = clean_source(SOURCE)
    packets.prior.resume.reader.common.cpu_gate(root / "cpu", revision)
    return _analyze_report(root, revision, revision)


def _analyze_report(root, execution_revision, analysis_revision, *, expected_report_sha=None):
    report_path = root / "recovered_bindings.json"
    if not report_path.exists():
        if expected_report_sha is not None:
            raise ValueError("the registered completed recovery report is required")
        return dict(
            source_sha=execution_revision,
            analysis_source_sha=analysis_revision,
            status="document_link_validation_failed_or_incomplete",
            GPU_jobs=0,
            fresh_confirmation_authorized=False,
            paper_campaign_authorized=False,
        )
    budget = packets.prior.Budget()
    report = packets.saved_json(report_path, expected_report_sha, budget, output_root=root)
    parent = packets.saved_json(
        PARENT / "recovered_bindings.json", PARENT_SHA, budget, output_root=PARENT
    )
    if (
        not isinstance(report.get("rows"), list)
        or len(report["rows"]) != 134
        or [row["path"] for row in report["rows"]] != [row["path"] for row in parent["rows"]]
    ):
        raise ValueError("exactly 134 original rows in their preserved order required")
    old = {row["path"]: row for row in parent["rows"]}
    new = {row["path"]: row for row in report["rows"]}
    if len(old) != 134 or len(new) != 134 or old.keys() != new.keys():
        raise ValueError("all saved historical bindings must remain present")
    changes = sorted(path for path in old if digest(old[path]) != digest(new[path]))
    if changes != sorted(DOCUMENT_ARTIFACTS) or report["promoted_paths"] != changes:
        raise ValueError("only the two reviewed original document bindings may change")
    if any(
        new[path]["artifact_sha256"] != old[path]["artifact_sha256"]
        or new[path]["signature"] != old[path]["signature"]
        for path in old
    ):
        raise ValueError("original artifact hashes and signatures must remain unchanged")
    if any(
        new[path]["status"] != "artifact_binding_only"
        or new[path].get("schema") != "historical_document_seal_binding_v55"
        or new[path].get("canonical_eligible") is not False
        or new[path].get("negative_result_preserved") is not True
        or new[path].get("quality_gate_pass") is not False
        or new[path].get("scientific_dependency_validation") is not False
        or new[path].get("clean_runtime_execution_claimed") is not False
        or new[path].get("occurrence_resolved") is not False
        or new[path].get("reused_hash_transaction") is not True
        or new[path].get("parent_binding_sha256") != digest(old[path])
        or any(
            type(new[path].get(key)) is not int or new[path][key] != 0
            for key in ("GPU_jobs", "newly_resolved_occurrences")
        )
        or any(new[path].get(key) is not False for key in contract.FALSE_FIELDS)
        or any(
            new[path][key] is not None
            for key in (
                "original_producer_commit",
                "original_producer_writer_path",
                "original_producer_source_blob_sha256",
            )
        )
        for path in changes
    ):
        raise ValueError("a document seal cannot supply an original runtime producer")
    expected_facts = dict(
        original_commit=DOCUMENT_COMMIT,
        repository_path=DOCUMENT_PATH,
        original_blob_oid=DOCUMENT_BLOB_OID,
        current_blob_oid=DOCUMENT_BLOB_OID,
        metadata_path=DOCUMENT_METADATA_PATH,
    )
    if (
        report.get("document_source") != expected_facts
        or report.get("document_sha256") != DOCUMENT_SHA256
    ):
        raise ValueError("exact immutable document provenance required")
    if set(report.get("supplemental_document_claims", {})) != set(DOCUMENT_ARTIFACTS):
        raise ValueError("exactly two supplemental artifact-only document claims required")
    for path in changes:
        claim = dict(
            metadata_path=DOCUMENT_METADATA_PATH,
            metadata_sha256=DOCUMENT_SHA256,
            artifact_sha256=DOCUMENT_ARTIFACTS[path],
            role="artifact_only",
            recorded_source_sha=None,
            source_blob_sha256=None,
            writer_path=None,
            source_evidence_verified=False,
        )
        if (
            digest(report["supplemental_document_claims"][path]) != digest(claim)
            or digest(new[path]) != digest(expected_document_binding(old[path], claim))
            or new[path].get("original_document_seal_claim_sha256") != digest(claim)
            or new[path].get("original_document_commit") != DOCUMENT_COMMIT
            or new[path].get("original_document_repository_path") != DOCUMENT_PATH
            or new[path].get("original_document_git_blob_oid") != DOCUMENT_BLOB_OID
            or new[path].get("original_document_sha256") != DOCUMENT_SHA256
            or new[path].get("binding_extension_schema") != "original_committed_document_seal_v55"
            or new[path].get("historical_development_overlay_recorded_dirty") is not True
            or new[path].get("development_classification")
            != "retrospective_development_only_not_paper_evidence"
        ):
            raise ValueError("complete original document claim and development scope required")
    if (
        report["source_sha"] != execution_revision
        or type(report.get("other_rows_verbatim")) is not int
        or report["other_rows_verbatim"] != 132
        or type(report["reused_hash_transactions"]) is not int
        or report["reused_hash_transactions"] != 134
        or any(
            type(report.get(key)) is not int or report[key] != 0
            for key in (
                "parquet_bodies_reopened",
                "original_JSON_identity_proofs_replayed",
                "historical_metadata_bodies_reopened",
                "historical_logs_reopened",
                "Slurm_records_queried",
                "newly_resolved_occurrences",
                "GPU_jobs",
            )
        )
        or any(
            report.get(key) is not False
            for key in (
                "credential_bodies_read",
                "historical_code_executed",
                "fresh_confirmation_authorized",
                "materialization_authorized",
                "paper_campaign_authorized",
                "historical_media_exposure_resolved",
                "scientific_quality_claim_established",
            )
        )
        or digest(report["original_occurrence_counts"])
        != digest(parent["original_occurrence_counts"])
        or report["status_counts"] != dict(Counter(row["status"] for row in new.values()))
    ):
        raise ValueError("unchanged proof reuse and semantic boundaries required")
    return dict(
        source_sha=execution_revision,
        analysis_source_sha=analysis_revision,
        report_path=str(report_path),
        report_sha256=packets.saved_sha(report_path, budget),
        status_counts=report["status_counts"],
        document_links_recovered=2,
        all_134_hashes_and_signatures_unchanged=True,
        other_132_rows_verbatim=True,
        newly_resolved_occurrences=0,
        reused_hash_transactions=134,
        GPU_jobs=0,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
        paper_campaign_authorized=False,
    )


def analyze_completed(root):
    """Derive only a new decision from the one registered completed execution."""
    root = root.resolve()
    if not root.is_relative_to(ANALYSIS_OUTPUT_BASE.resolve()) or root.is_relative_to(
        COMPLETED_ROOT.resolve()
    ):
        raise ValueError("new derived output outside the completed run is required")
    if (root / "decision.json").exists():
        raise FileExistsError("preserve the existing derived decision")
    analysis_revision = clean_source(SOURCE)
    clean_source(COMPLETED_SOURCE, COMPLETED_SOURCE_SHA)
    packets.prior.resume.reader.common.cpu_gate(COMPLETED_ROOT / "cpu", COMPLETED_SOURCE_SHA)
    result = _analyze_report(
        COMPLETED_ROOT,
        COMPLETED_SOURCE_SHA,
        analysis_revision,
        expected_report_sha=COMPLETED_REPORT_SHA,
    )
    result.update(
        classification="CPU_derived_decision_from_completed_recovery_only",
        reused_CPU_source_sha=COMPLETED_SOURCE_SHA,
        CPU_suite_rerun=False,
        recovery_reexecuted=False,
        original_failure_preserved=True,
    )
    write_once(root / "decision.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cpu", "recover", "analyze", "analyze-completed"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to(Path("/scratch/djy8hg/workdir").resolve()):
        raise ValueError("scratch-only immutable output required")
    try:
        if args.mode == "cpu":
            result = packets.prior.resume.cpu(root)
        elif args.mode == "analyze-completed":
            result = analyze_completed(root)
        else:
            result = globals()[args.mode](root)
        if args.mode == "analyze":
            write_once(root / "decision.json", result)
    except Exception as error:
        write_once(
            root / f"failure-{args.mode}.json",
            dict(
                error_type=type(error).__name__,
                error="original document-seal validation failed closed",
                GPU_jobs=0,
                fresh_confirmation_authorized=False,
            ),
        )
        raise
    print(json.dumps(result, sort_keys=True))
