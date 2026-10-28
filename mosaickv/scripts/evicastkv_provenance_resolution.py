#!/usr/bin/env python3
"""Hash-bound root classification and observed media/source-locator audit.

Unknown episode/document aliases stay unknown. This script cannot choose a
confirmation cohort or turn absence of an observed overlap into freshness.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evicastkv_media_audit import historical_catalogs

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once
from mosaickv.evaluation.fox_milebench_scope import PROPOSED_TASKS, safe_relative

SOURCE = Path(__file__).resolve().parents[2]
BASE = Path("/scratch/djy8hg/runs")


def classify_root(root, original_ids):
    if root.name == "coco-cublas-retry-edb6edf-20261014":
        seal_path = root / "seal.json"
        seal = json.loads(seal_path.read_text())
        plans = [p for p in seal["bound_files"] if Path(p).name == "plan.json"]
        ledgers = [p for p in seal["bound_files"] if Path(p).name == "ledger.json"]
        if len(plans) != 1 or len(ledgers) != 1:
            raise ValueError("retry wrapper must bind one parent plan and input ledger")
        for path in (*plans, *ledgers):
            if sha(Path(path)) != seal["bound_files"][path]:
                raise ValueError("historical wrapper input hash mismatch")
        plan = json.loads(Path(plans[0]).read_text())
        if plan.get("dataset_id") != "lmms-lab/COCO-Caption":
            raise ValueError("wrapper is not the recorded COCO campaign")
        ledger = json.loads(Path(ledgers[0]).read_text())
        return dict(
            root=str(root),
            classification="launcher_for_existing_bound_COCO_campaign",
            purpose_verified=True,
            evidence={
                str(seal_path): sha(seal_path),
                **{p: sha(Path(p)) for p in (*plans, *ledgers)},
            },
            excluded_ids=[s["sample_id"] for s in ledger["samples"]],
            classification_basis="original seal and immutable parent input hashes",
        )
    if root.name == "milebench-actionprediction-extract-20261018":
        # A count-only scheduler log cannot prove that no inference occurred.
        # Quarantine the entire official pool rather than make that inference.
        return dict(
            root=str(root),
            classification="unknown_activity_entire_ActionPrediction_pool_quarantined",
            purpose_verified=False,
            evidence={str(p): sha(p) for p in root.iterdir() if p.is_file()},
            excluded_ids=sorted(s for s in original_ids if s.startswith("ActionPrediction-")),
            classification_basis="conservative full-pool exclusion; no log-based innocence claim",
        )
    if root.name == "fox-review-20261028-milebench-media-content-v1":
        report_path, media_path = root / "report.json", root / "media_identities.json"
        report, media = (json.loads(p.read_text()) for p in (report_path, media_path))
        if (
            report.get("observed_media_audit_completed") is not True
            or report.get("no_images_extracted_or_downloaded") is not True
            or report.get("confirmation_authorized") is not False
            or report["media_identity_content_sha256"] != digest(media["identities"])
        ):
            raise ValueError("media-audit root is not supported by its bound outputs")
        return dict(
            root=str(root),
            classification="input_media_audit_not_prediction_evidence",
            purpose_verified=True,
            evidence={str(p): sha(p) for p in (report_path, media_path)},
            excluded_ids=[],
            classification_basis=(
                "content-bound input audit; inspected IDs are not inferred model runs"
            ),
        )
    return dict(
        root=str(root),
        classification="unresolved",
        purpose_verified=False,
        excluded_ids=[],
        evidence={},
    )


def media_names(row):
    task = row["task"]
    return [
        *(f"{task}/images/{safe_relative(p)}" for p in row["source_image_paths"]),
        f"{task}/combined_1_images/{safe_relative(row['combined_1_images'])}",
    ]


def source_locators(row):
    """Only observed path prefixes; never infer an episode ID from numeric files."""
    return sorted({safe_relative(p).split("/")[0] for p in row["source_image_paths"] if "/" in p})


def compare_media(proposed, historical, identities):
    old_hashes = {
        identities[name][k]
        for r in historical
        for name in media_names(r)
        if name in identities
        for k in ("encoded_sha256", "pixel_sha256")
    }
    old_locators = {key for row in historical for key in source_locators(row)}
    audits = []
    for row in proposed:
        names = media_names(row)
        audits.append(
            dict(
                sample_id=row["sample_id"],
                task=row["task"],
                missing_assets=[name for name in names if name not in identities],
                observed_content_overlap=[
                    name
                    for name in names
                    if name in identities
                    and any(
                        identities[name][k] in old_hashes
                        for k in ("encoded_sha256", "pixel_sha256")
                    )
                ],
                observed_source_locator_overlap=sorted(set(source_locators(row)) & old_locators),
                observed_source_locators=source_locators(row),
                source_video_document_identity_complete=False,
                source_identity_limitation=(
                    "path prefixes lack certified global aliases; "
                    "flat files lack episode/document IDs"
                ),
                fresh_eligible=False,
            )
        )
    return audits


def audit(history_review, scope, media_root, iedit_root, output):
    revision = clean_source(SOURCE)
    if output.exists():
        raise FileExistsError("provenance resolution output must be new")
    review = json.loads(history_review.read_text())
    historical = historical_catalogs()  # Four pinned, column-projected input catalogs only.
    ids = {r["sample_id"] for r in historical}
    resolutions = [classify_root(Path(p), ids) for p in review["unresolved_roots"]]
    availability_path = scope / "availability.json"
    availability = json.loads(availability_path.read_text())
    proposed, bindings = (
        [],
        {str(history_review): sha(history_review), str(availability_path): sha(availability_path)},
    )
    for task, binding in availability["catalogs"].items():
        path = Path(binding["path"])
        if sha(path) != binding["sha256"]:
            raise ValueError("proposed ID/media catalog changed")
        bindings[str(path)] = sha(path)
        proposed.extend(dict(row, task=task) for row in json.loads(path.read_text())["rows"])
    media_path, extra_path = (
        media_root / "media_identities.json",
        iedit_root / "media_identities.json",
    )
    media = json.loads(media_path.read_text())
    extra = json.loads(extra_path.read_text())
    extra_protocol = json.loads((iedit_root / "protocol.json").read_text())
    if extra_protocol["historical_media_sha256"] != sha(media_path) or extra.get("errors"):
        raise ValueError("IEdit media evidence differs from its parent or failed")
    identities = dict(media["identities"])
    for name, identity in extra["identities"].items():
        if name in identities and identities[name] != identity:
            raise ValueError("conflicting media fingerprint")
        identities[name] = identity
    for path in (media_path, extra_path, iedit_root / "protocol.json"):
        bindings[str(path)] = sha(path)
    samples = compare_media(proposed, historical, identities)
    write_once(output / "historical_input_catalog.json", historical)
    write_once(output / "root_resolutions.json", resolutions)
    write_once(output / "observed_media_audit.json", samples)
    write_once(
        output / "report.json",
        dict(
            source_sha=revision,
            status="bounded_provenance_resolution_global_aliases_unverified",
            bindings=bindings,
            root_classifications_resolved_for_exclusion=all(
                r["classification"] != "unresolved" for r in resolutions
            ),
            roots_with_unverified_original_purpose=[
                r["root"] for r in resolutions if not r["purpose_verified"]
            ],
            unresolved_roots=[
                r["root"] for r in resolutions if r["classification"] == "unresolved"
            ],
            conservative_pool_exclusions=sorted(
                {sid for r in resolutions for sid in r["excluded_ids"]}
            ),
            tasks=[
                dict(
                    task=task,
                    total=sum(s["task"] == task for s in samples),
                    complete_media=sum(
                        s["task"] == task and not s["missing_assets"] for s in samples
                    ),
                    observed_content_overlap=sum(
                        s["task"] == task and bool(s["observed_content_overlap"]) for s in samples
                    ),
                    observed_source_locator_overlap=sum(
                        s["task"] == task and bool(s["observed_source_locator_overlap"])
                        for s in samples
                    ),
                    missing_global_source_identity=sum(s["task"] == task for s in samples),
                )
                for task in PROPOSED_TASKS
            ],
            references_or_answers_used=False,
            model_loaded=False,
            global_historical_media_coverage_complete=False,
            source_video_document_identity_complete=False,
            fresh_confirmation_authorized=False,
            COCO_boundary_B_failures_preserved=True,
            LOOKM_transformed_cache_fidelity_not_tested=True,
            blockers=[
                "global video/episode/document alias metadata is not certified",
                "known-used TextNeedle and other historical media are not covered "
                "by the four-task pixel audit",
            ],
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("history-review", "scope", "media-root", "iedit-root", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    a = p.parse_args()
    for path in vars(a).values():
        if not path.resolve().is_relative_to(Path("/scratch/djy8hg").resolve()):
            raise ValueError("scratch-only provenance audit")
    audit(a.history_review, a.scope, a.media_root, a.iedit_root, a.output)
