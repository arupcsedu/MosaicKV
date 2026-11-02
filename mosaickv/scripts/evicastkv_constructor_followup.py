#!/usr/bin/env python3
"""One finite conditional constructor check and same-consumed-input timing chain."""

import argparse
import os
from pathlib import Path
from unittest.mock import patch

import evicastkv_common_streaming as common
import evicastkv_construction_phase as phase

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = common.SOURCE
PARENT = Path(
    "/scratch/djy8hg/runs/evicastkv/host-metadata-once-request-timing-consumed-20261102-v1"
)
PARENT_PLAN_SHA = "b5b3e0d2d94855d39441a2668412a77e950cd39e5bd47e5958f9592977bf34eb"


def followup_needed(decision):
    if not decision.get("complete") or decision.get("failures"):
        raise ValueError("parent timing must complete without missing/failed rows")
    if type(decision.get("all_method_similar_time_gate")) is not bool:
        raise ValueError("explicit parent all-method timing decision required")
    return not decision["all_method_similar_time_gate"]


def start(root, source_sha):
    if clean_source(SOURCE) != source_sha or sha(PARENT / "plan.json") != PARENT_PLAN_SHA:
        raise ValueError("clean successor SHA and immutable parent required")
    if root.exists():
        raise FileExistsError("new follow-up root required")
    decision = common.previous.read(PARENT / "analysis/decision.json")
    if decision.get("plan_sha256") != PARENT_PLAN_SHA:
        raise ValueError("parent analysis binding differs")
    if not followup_needed(decision):
        write_once(
            root / "not_needed.json",
            dict(
                reason="parent all-method timing gate passed",
                source_sha=source_sha,
                parent_decision_sha256=sha(PARENT / "analysis/decision.json"),
                GPU_submitted=False,
            ),
        )
        return
    phase.PARENT = PARENT
    phase.prepare(root, verify_constructor_memory_once=True)
    expected = sha(root / "plan.json")
    common.previous.cpu(root, expected)
    original = common.control.parent.subprocess.check_output

    def scoped(command, **kwargs):
        command = [
            str(SOURCE / "mosaickv/slurm/evicastkv_constructor_followup.sbatch")
            if str(x).endswith("/evicastkv_funded_delimiter_diagnostic.sbatch")
            else x
            for x in command
        ]
        return original(command, **kwargs)

    with patch.object(common.control.parent.subprocess, "check_output", scoped):
        common.control.parent.submit(root, expected)


def analyze(root, expected):
    phase.analyze(root, expected)
    plan = common.previous.common.inputs(root, expected)
    decision = common.previous.read(root / "analysis/decision.json")
    if (
        not decision["complete"]
        or decision["failures"]
        or decision.get("constructor_memory_gate") is not True
    ):
        write_once(
            root / "timing_blocked.json",
            dict(
                reason="constructor identity/completeness/memory guard failed",
                GPU_timing_submitted=False,
                decision_sha256=sha(root / "analysis/decision.json"),
            ),
        )
        return
    target = root / "timing"
    parent = common.previous.read(PARENT / "plan.json")
    bound = dict(parent["bound_files"])
    for path in (
        root / "plan.json",
        root / "analysis/decision.json",
        root / "cpu.json",
        PARENT / "analysis/decision.json",
        SOURCE / "mosaickv/docs/transport_close_once_20261102.md",
    ):
        bound[str(path)] = sha(path)
    write_once(
        target / "plan.json",
        dict(
            parent,
            source_sha=plan["source_sha"],
            bound_files=bound,
            stage="consumed_aggregated_constructor_checks_complete_request_timing",
            verification_predicate_unchanged=True,
            constructor_memory_verified_in=str(root / "analysis/decision.json"),
            same_metadata_optimization_as_ancestor=True,
        ),
    )
    for name in ("cpu.json", "cpu.xml"):
        os.link(root / name, target / name)
    write_once(
        target / "cpu_reuse.json",
        dict(
            source_sha=plan["source_sha"],
            producer_root=str(root),
            cpu_json_sha256=sha(root / "cpu.json"),
            cpu_xml_sha256=sha(root / "cpu.xml"),
            same_complete_suite_same_immutable_source=True,
        ),
    )
    common.submit(target, sha(target / "plan.json"))
    write_once(
        root / "timing_promotion.json",
        dict(
            target=str(target),
            source_sha=plan["source_sha"],
            scope="same_two_consumed_inputs_only",
            fresh_evaluation_authorized=False,
            paper_campaign_authorized=False,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("start", "worker", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha")
    parser.add_argument("--source-sha")
    args = parser.parse_args()
    if args.mode == "start":
        start(args.root, args.source_sha)
    elif args.mode == "worker":
        phase.worker(args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    else:
        analyze(args.root, args.plan_sha)
