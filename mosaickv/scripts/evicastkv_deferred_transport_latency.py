#!/usr/bin/env python3
"""Same-parent deferred validation controls -> strict paired latency DAG."""

import argparse
import os
from pathlib import Path
from unittest.mock import patch

import evicastkv_direct_text_latency as direct
import evicastkv_strict_latency as strict

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE, driver = direct.SOURCE, direct.driver
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_deferred_transport_latency.sbatch"
PARENT = Path("/scratch/djy8hg/runs/evicastkv/direct-text-consumed-20261103-retry1")
PROFILE = Path("/scratch/djy8hg/runs/evicastkv/direct-construction-phases-consumed-20261103-retry1")
DOC = SOURCE / "mosaickv/docs/deferred_transport_latency_20261103.md"


def prepare(root, revision):
    if root.exists() or clean_source(SOURCE) != revision:
        raise ValueError("new immutable root and clean source required")
    old = driver.read(PARENT / "plan.json")
    decision = driver.read(PARENT / "paired-timing/analysis/decision.json")
    attribution = driver.read(PROFILE / "analysis/decision.json")
    if (
        not decision["complete"]
        or decision["all_input_latency_gate"]
        or not attribution["complete"]
    ):
        raise ValueError("complete preserved strict negative and phase attribution required")
    bound = dict(old["bound_files"])
    for path in (
        PARENT / "plan.json",
        PARENT / "paired-timing/analysis/decision.json",
        PROFILE / "analysis/decision.json",
        DOC,
    ):
        bound[str(path)] = sha(path)
    for i in (0, 1):
        path = PARENT / f"workers/{i}/study.json"
        bound[str(path)] = sha(path)
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=revision,
            bound_files=bound,
            stage="consumed_deferred_unchanged_transport_validation",
            direct_text_root=str(root),
            defer_transport_validation=True,
            immutable_parent_policies=True,
            automatic_quality_submission=False,
            paper_campaign_authorized=False,
            acceptance_ratio_upper_bound=1.0,
        ),
    )
    with patch.object(direct, "SCRIPT", SCRIPT):
        job = direct.submit(root, "cpu")
    write_once(root / "preparation_submission.json", dict(cpu=job))


def study(root, expected, index):
    frozen = driver.read(PARENT / f"workers/{index}/study.json")["records"]
    original = direct.execute
    primitive_controls_done = False

    def checked(*args, **kwargs):
        nonlocal primitive_controls_done
        if not primitive_controls_done:
            from mosaickv.evaluation.deferred_transport_checks import gpu_controls

            write_once(
                root / f"workers/{index}/validation_primitive_controls.json",
                dict(cases=gpu_controls(), no_task_evidence=True),
            )
            primitive_controls_done = True
        method, validate = args[5], kwargs.get("final_validator")

        def verify(packets):
            if direct.packet_signature(packets)["digest"] != frozen[method]["packet_sha256"]:
                raise RuntimeError("deferred validation changed frozen parent packet words")
            if validate is not None:
                validate(packets)

        measured = original(*args, **dict(kwargs, final_validator=verify))
        validation = measured["transport_validation"]
        if not validation["enabled"] or not validation["all_checks_before_decode"]:
            raise RuntimeError("validation transaction did not commit before decode")
        if method == "mosaickv_exact" and (
            validation["counts"] != dict(finite=64, protected=64, close=64)
            or validation["commit_flushes"] != 1
            or validation["validation_flag_peak_tensor_bytes"] > 1025
        ):
            raise RuntimeError("complete bounded source validation transaction required")
        return measured

    with patch.object(direct, "execute", checked):
        direct.study(root, expected, index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("prepare", "cpu", "study", "study_analysis", "timing", "timing_analysis")
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-sha")
    parser.add_argument("--plan-sha")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root, args.source_sha)
    elif args.action == "timing_analysis":
        strict.analyze(args.root, args.plan_sha, SOURCE, clean_source(SOURCE), "timing", "direct")
    else:
        with patch.object(direct, "SCRIPT", SCRIPT):
            if args.action == "study":
                study(args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))
            elif args.action == "timing":
                direct.timing(args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))
            else:
                getattr(direct, args.action)(args.root, args.plan_sha)


if __name__ == "__main__":
    main()
