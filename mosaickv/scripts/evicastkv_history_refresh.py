#!/usr/bin/env python3
"""Bounded two-root historical ID review; never certifies unseen media as fresh."""

from __future__ import annotations

import argparse
from pathlib import Path

from evicastkv_evidence import exclusion_inventory

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once
from mosaickv.evaluation.milebench import MILEBENCH_DATASET_TO_GROUP

SOURCE = Path(__file__).resolve().parents[2]
EXTRA_SEEDS = (
    "protocol_lock.json",
    "analysis_plan.json",
    "prior-materialization-exclusions.json",
    "historical_ids.json",
    "controller-base.json",
    "launch.json",
    "submission.json",
)


def refresh(parent, output):
    revision = clean_source(SOURCE)
    if output.exists():
        raise FileExistsError("history refresh output must be new")
    reports = {}
    for name in ("mosaickv", "evicastkv"):
        path = output / f"{name}-inventory.json"
        report = exclusion_inventory(
            parent,
            path,
            max_seconds=90,
            run_root=Path("/scratch/djy8hg/runs") / name,
            identity_tasks=tuple(MILEBENCH_DATASET_TO_GROUP),
            extra_manifest_names=EXTRA_SEEDS,
        )
        reports[name] = dict(path=str(path), sha256=sha(path), report=report)
    ids = sorted({s for b in reports.values() for s in b["report"]["MileBench_consumed_ids"]})
    write_once(
        output / "review.json",
        dict(
            source_sha=revision,
            status="bounded_ID_review_not_global_media_certification",
            reports={k: {a: b[a] for a in ("path", "sha256")} for k, b in reports.items()},
            canonical_files=sum(b["report"]["canonical_files"] for b in reports.values()),
            remaining_manifest_queue=sum(
                b["report"]["remaining_manifest_queue"] for b in reports.values()
            ),
            errors=[e for b in reports.values() for e in b["report"]["errors"]],
            unresolved_roots=[
                r for b in reports.values() for r in b["report"]["unclassified_workload_roots"]
            ],
            MileBench_known_used_per_task={
                task: sum(s.startswith(task + "-") for s in ids)
                for task in MILEBENCH_DATASET_TO_GROUP
            },
            source_video_document_identity_complete=False,
            exhaustive_historical_media_inventory_complete=False,
            confirmation_authorized=False,
            labels_or_predictions_used_for_selection=False,
            no_models_loaded=True,
            no_catalog_selection_or_materialization=True,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    a = parser.parse_args()
    for p in (a.parent, a.output):
        if not p.resolve().is_relative_to(Path("/scratch/djy8hg").resolve()):
            raise ValueError("scratch-only historical review")
    refresh(a.parent, a.output)
