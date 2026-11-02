#!/usr/bin/env python3
"""Finite conditional move of verified hardware tiling before text validation."""

import argparse
import os
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import evicastkv_constructor_followup as driver

ROW = Path("/scratch/djy8hg/runs/evicastkv/source-row-geometry-consumed-20261102-retry1")
ASYNC = Path("/scratch/djy8hg/runs/evicastkv/async-text-bitwise-consumed-20261102-v1/timing")


def control(root):
    return root.with_name(root.name + "-launch")


def schedule(root, revision, mode, dependency):
    folder = control(root)
    job = driver.common.control.parent.subprocess.check_output(
        [
            "sbatch",
            "--parsable",
            "--partition=bii",
            "--account=bii_dsc_community",
            "--cpus-per-task=2",
            "--mem=16G",
            "--time=00:25:00",
            f"--dependency=afterany:{dependency}",
            f"--job-name=mk-staged-{mode}",
            f"--output={folder}/{mode}-%j.out",
            f"--error={folder}/{mode}-%j.err",
            str(driver.SOURCE / "mosaickv/slurm/evicastkv_staged_geometry.sbatch"),
            str(driver.SOURCE),
            str(root),
            revision,
            mode,
        ],
        text=True,
    ).strip()
    driver.write_once(
        folder / (mode + "-submission.json"),
        dict(job_id=job, dependency=dependency, source_sha=revision),
    )


def defer(root, revision, mode):
    if driver.clean_source(driver.SOURCE) != revision:
        raise ValueError("clean immutable staging source required")
    if mode == "wait_start":
        path = ROW.with_name(ROW.name + "-launch") / "deferred.json"
        if not path.exists():
            driver.write_once(
                control(root) / "blocked.json",
                dict(reason="row-geometry chain was not admitted", GPU_submitted=False),
            )
            return
        schedule(root, revision, "wait_micro", driver.common.previous.read(path)["job_id"])
        return
    if mode == "wait_micro":
        if (ROW / "not_needed.json").exists():
            driver.write_once(
                control(root) / "not_needed.json",
                dict(reason="all timing gates already passed", GPU_submitted=False),
            )
            return
        path = ROW / "submission.json"
        if not path.exists():
            driver.write_once(
                control(root) / "blocked.json",
                dict(reason="row-geometry CPU gate did not submit workers", GPU_submitted=False),
            )
            return
        schedule(root, revision, "choose", driver.common.previous.read(path)["cpu_analysis"])
        return
    decision = driver.common.previous.read(ROW / "analysis/decision.json")
    if not decision.get("complete") or decision.get("failures"):
        driver.write_once(
            control(root) / "blocked.json",
            dict(reason="row geometry baseline/producer control failed", GPU_submitted=False),
        )
        return
    selected = decision.get("selected_geometry")
    parent = ROW / "timing" if selected is not None else ASYNC
    driver.write_once(
        control(root) / "choice.json",
        dict(
            parent=str(parent),
            geometry=selected or [16, 4],
            row_decision_sha256=driver.sha(ROW / "analysis/decision.json"),
        ),
    )
    dependency = driver.common.previous.read(parent / "submission.json")["cpu_analysis"]
    schedule(root, revision, "construct", dependency)


@contextmanager
def scoped(root):
    choice = driver.common.previous.read(control(root) / "choice.json")
    parent = Path(choice["parent"])
    with (
        patch.object(driver, "PARENT", parent),
        patch.object(driver, "PARENT_PLAN_SHA", driver.sha(parent / "plan.json")),
    ):
        yield choice


def construct(root, revision):
    prepare = driver.phase.prepare
    original = driver.common.control.parent.subprocess.check_output
    with scoped(root) as choice:

        def scoped_prepare(root, **kwargs):
            return prepare(
                root,
                **kwargs,
                construction_geometry=choice["geometry"],
                async_bitwise_text_checks=True,
                extra_bound_files=(
                    control(root) / "choice.json",
                    ROW / "analysis/decision.json",
                    driver.SOURCE / "mosaickv/docs/staged_source_geometry_20261102.md",
                ),
            )

        def scoped_submit(command, **kwargs):
            return original(
                [
                    str(driver.SOURCE / "mosaickv/slurm/evicastkv_staged_geometry.sbatch")
                    if str(x).endswith("/evicastkv_constructor_followup.sbatch")
                    else x
                    for x in command
                ],
                **kwargs,
            )

        with (
            patch.object(driver.phase, "prepare", scoped_prepare),
            patch.object(driver.common.control.parent.subprocess, "check_output", scoped_submit),
        ):
            driver.start(root, revision)


def finish(root, revision):
    if driver.clean_source(driver.SOURCE) != revision:
        raise ValueError("clean closeout source required")
    path = root / "timing/analysis/decision.json"
    if not path.exists():
        path = root / "analysis/decision.json"
    decision = driver.common.previous.read(path)
    driver.write_once(
        root / "engineering_closeout.json",
        dict(
            source_sha=revision,
            decision_path=str(path),
            decision_sha256=driver.sha(path),
            complete=decision.get("complete", False),
            failures=decision.get("failures", []),
            all_method_similar_time_gate=decision.get("all_method_similar_time_gate", False),
            independent_quality_N=0,
            quality_superiority_established=False,
            paper_superiority_established=False,
            next_action="independent quality validation only after engineering gates pass; "
            "otherwise inspect remaining measured costs without weakening controls",
            no_fresh_or_large_campaign_submitted=True,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("wait_start", "wait_micro", "choose", "construct", "worker", "analyze", "finish"),
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--source-sha")
    parser.add_argument("--plan-sha")
    args = parser.parse_args()
    if args.mode in ("wait_start", "wait_micro", "choose"):
        defer(args.root, args.source_sha, args.mode)
    elif args.mode == "construct":
        construct(args.root, args.source_sha)
    elif args.mode == "finish":
        finish(args.root, args.source_sha)
    elif args.mode == "worker":
        driver.phase.worker(args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    else:
        with scoped(args.root):
            driver.analyze(args.root, args.plan_sha)
        path = args.root / "timing/submission.json"
        revision = driver.clean_source(driver.SOURCE)
        if path.exists():
            schedule(
                args.root, revision, "finish", driver.common.previous.read(path)["cpu_analysis"]
            )
        else:
            finish(args.root, revision)
