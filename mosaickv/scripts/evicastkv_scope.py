#!/usr/bin/env python3
"""Register a model-scoped continuation; never filter examples by task quality."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evicastkv_analyze import collect
from evicastkv_campaign import SOURCE, prepare, submit
from evicastkv_finalize import schedule as schedule_final
from evicastkv_packet import schedule_watch

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once


def supported_scope(parent):
    lock = json.loads((parent / "protocol_lock.json").read_text())
    clean_source(Path(lock["source_root"]), lock["source_sha"])
    gate_path = parent / "analysis/correctness/GATE_correctness.json"
    gate = json.loads(gate_path.read_text())
    if gate["status"] != "blocked" or (parent / "outputs/scoring").exists():
        raise ValueError("scope registration must precede task scoring of this failed parent")
    _spec, _workers, rows, _errors = collect(parent, "correctness")
    expected = set(lock["cohorts"]["coco"]["validation"]["sample_ids"])
    coco = [r for r in rows if r["workload"] == "coco"]
    if len(coco) != len(expected) * 4 or {r["sample_id"] for r in coco} != expected:
        raise ValueError("not a complete COCO correctness cohort")
    if any(r["status"] != "completed" for r in coco):
        raise ValueError("COCO itself has a failed numerical control; continuation prohibited")
    mile = [r for r in rows if r["workload"] == "milebench"]
    if not any(r["status"] != "completed" for r in mile):
        raise ValueError("no checkpoint-specific numerical failure establishes this scope")
    return dict(
        basis="model_scoped_numerical_failure_only",
        enabled_workloads=["coco"],
        originally_requested_workloads=["coco", "milebench"],
        parent=str(parent),
        parent_protocol_sha256=sha(parent / "protocol_lock.json"),
        parent_gate_path=str(gate_path),
        parent_gate_sha256=sha(gate_path),
        parent_overall_gate_status="blocked_not_reclassified",
        supported_coco_numerical_requests=len(coco),
        supported_coco_unique_inputs=len(expected),
        blocked_milebench_failed_numerical_requests=sum(r["status"] != "completed" for r in mile),
        blocked_scope="LLaVA-1.5/MileBench configuration, not selected difficult examples",
        task_scores_used=False,
        no_within_workload_sample_removal=True,
        numerical_thresholds_unchanged=True,
        precision_and_arithmetic_unchanged=True,
        common_gates_rerun_on_new_source=True,
        primary_family_kept_at_six=True,
        unexecuted_primary_contrasts=4,
        publication_scope="development-only COCO, not full two-workload confirmation",
    )


def run(parent, root):
    clean_source(SOURCE)
    registration = root / "scope_registration.json"
    write_once(registration, supported_scope(parent))
    prepare(root, workloads=("coco",), scope_registration=registration)
    inventory = parent / "history-inventory.json"
    if inventory.exists():
        write_once(root / "history-inventory.json", json.loads(inventory.read_text()))
    submit(root, "smoke")
    submission = json.loads((root / "jobs/smoke-submission.json").read_text())
    schedule_watch(root, "smoke", submission["analysis"])
    packet_submission = json.loads((root / "jobs/smoke-packet-submission.json").read_text())
    schedule_final(root, root, "smoke", packet_submission["job"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    run(args.parent, args.root)


if __name__ == "__main__":
    main()
