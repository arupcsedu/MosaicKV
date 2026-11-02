#!/usr/bin/env python3
"""Finite consumed-input async-text check, memory gate, then request timing."""

import argparse
import os
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import evicastkv_constructor_followup as driver

PROFILE = Path("/scratch/djy8hg/runs/evicastkv/text-construction-subphases-consumed-20261102-v1")
PARENT = Path(
    "/scratch/djy8hg/runs/evicastkv/aggregate-transport-checks-consumed-20261102-v1/timing"
)
PARENT_PLAN_SHA = "e796b658190a3e3452897ec72d4f91609adae1bf95d14fb559023f697f33a238"


@contextmanager
def driver_scope():
    with (
        patch.object(driver, "PARENT", PARENT),
        patch.object(driver, "PARENT_PLAN_SHA", PARENT_PLAN_SHA),
    ):
        yield


def profile_passes(decision):
    groups = [s for s in decision.get("summaries", []) if s.get("method") == "mosaickv_exact"]
    return (
        decision.get("complete") is True
        and not decision.get("failures")
        and {s["input_index"] for s in groups} == {0, 1}
        and len(groups) == 2
        and all(
            s["phases"].get("text_source.verification_equal", {}).get("inclusive_wall_seconds", 0)
            >= 0.1
            for s in groups
        )
    )


def start(root, source_sha):
    if not profile_passes(driver.common.previous.read(PROFILE / "analysis/decision.json")):
        raise ValueError("complete dominant bitwise-verification phase evidence required")
    prepare = driver.phase.prepare
    original = driver.common.control.parent.subprocess.check_output

    def scoped_prepare(root, **kwargs):
        return prepare(
            root,
            **kwargs,
            async_bitwise_text_checks=True,
            extra_bound_files=(
                PROFILE / "plan.json",
                PROFILE / "analysis/decision.json",
                driver.SOURCE / "mosaickv/docs/async_text_checks_20261102.md",
            ),
        )

    def scoped_submit(command, **kwargs):
        return original(
            [
                str(driver.SOURCE / "mosaickv/slurm/evicastkv_text_verification.sbatch")
                if str(x).endswith("/evicastkv_constructor_followup.sbatch")
                else x
                for x in command
            ],
            **kwargs,
        )

    with (
        driver_scope(),
        patch.object(driver.phase, "prepare", scoped_prepare),
        patch.object(driver.common.control.parent.subprocess, "check_output", scoped_submit),
    ):
        driver.start(root, source_sha)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("start", "worker", "analyze"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--source-sha")
    parser.add_argument("--plan-sha")
    args = parser.parse_args()
    if args.mode == "start":
        start(args.root, args.source_sha)
    elif args.mode == "worker":
        driver.phase.worker(args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    else:
        with driver_scope():
            driver.analyze(args.root, args.plan_sha)
