#!/usr/bin/env python3
"""Bounded register-pressure correction; strictly conditional on phase evidence."""

import argparse
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import evicastkv_row_geometry as row

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = row.SOURCE
PROFILE = Path("/scratch/djy8hg/runs/evicastkv/decode-phase-attribution-consumed-20261102-v1")
ROW = Path("/scratch/djy8hg/runs/evicastkv/source-row-geometry-consumed-20261102-retry1")
WARPS = {"r16_h4": None, "r32_h4": 16, "r64_h4": 16, "r32_h2": 8}


def schedule(root, revision, dependency, mode, step):
    folder = root.with_name(root.name + "-launch")
    folder.mkdir(exist_ok=True)
    job = subprocess.check_output(
        [
            "sbatch",
            "--parsable",
            "--partition=bii",
            "--account=bii_dsc_community",
            "--cpus-per-task=2",
            "--mem=16G",
            "--time=00:25:00",
            f"--dependency=afterany:{dependency}",
            f"--job-name=mk-source-warps-{mode}",
            f"--output={folder}/{mode}-{step}-%j.out",
            f"--error={folder}/{mode}-{step}-%j.err",
            str(SOURCE / "mosaickv/slurm/evicastkv_source_warp_followup.sbatch"),
            str(SOURCE),
            str(root),
            revision,
            mode,
            str(step),
        ],
        text=True,
    ).strip()
    write_once(
        folder / f"{mode}-{step}.json", dict(job_id=job, dependency=dependency, source_sha=revision)
    )


def profile_dependency():
    path = PROFILE / "submission.json"
    if path.exists():
        return str(row.common.previous.read(path)["cpu_analysis"])
    folder = PROFILE.with_name(PROFILE.name + "-launch")
    records = sorted(folder.glob("watch-*.json"), key=lambda p: int(p.stem.split("-")[-1]))
    if not records:
        raise ValueError("published phase dependency required")
    return str(row.common.previous.read(records[-1])["job_id"])


def profile_justifies_tuning(decision):
    groups = [s for s in decision.get("summaries", []) if s.get("method") == "mosaickv_exact"]
    phase = "mosaickv.evaluation.shared_source_kernel.project"
    return (
        decision.get("complete") is True
        and not decision.get("failures")
        and len(groups) == 2
        and {g["input_index"] for g in groups} == {0, 1}
        and all(
            g["phases"].get(phase, {}).get("mean_inclusive_cuda_stream_ms", 0) >= 250
            for g in groups
        )
    )


def watch(root, revision, step):
    if clean_source(SOURCE) != revision or not 0 <= step <= 10:
        raise ValueError("clean bounded source-warp experiment required")
    folder = root.with_name(root.name + "-launch")
    if (PROFILE / "not_needed.json").exists():
        write_once(
            root / "not_needed.json",
            dict(reason="upstream timing gate passed", GPU_submitted=False),
        )
        return
    path = PROFILE / "analysis/decision.json"
    if not path.exists():
        dependency = profile_dependency()
        seen = {row.common.previous.read(p)["dependency"] for p in folder.glob("watch-*.json")}
        if dependency in seen:
            write_once(
                folder / "blocked.json",
                dict(reason="phase producer incomplete", GPU_submitted=False),
            )
            return
        schedule(root, revision, dependency, "watch", step + 1)
        return
    decision = row.common.previous.read(path)
    if not profile_justifies_tuning(decision):
        write_once(
            folder / "blocked.json",
            dict(reason="no safe dominant source-projection evidence", GPU_submitted=False),
        )
        return
    original = row.common.control.parent.subprocess.check_output

    def scoped(command, **kwargs):
        return original(
            [
                str(SOURCE / "mosaickv/slurm/evicastkv_source_warp_followup.sbatch")
                if str(x).endswith("/evicastkv_row_geometry.sbatch")
                else x
                for x in command
            ],
            **kwargs,
        )

    parent = Path(row.common.previous.read(PROFILE / "plan.json")["parent_timing"])
    with (
        patch.object(row, "PARENT", parent),
        patch.object(row.common.control.parent.subprocess, "check_output", scoped),
    ):
        row.start(
            root,
            revision,
            launch_warps=WARPS,
            extra_bound_files=(
                path,
                PROFILE / "plan.json",
                ROW / "analysis/decision.json",
                SOURCE / "mosaickv/docs/source_warp_followup_20261102.md",
            ),
        )


def analyze(root, expected):
    plan = row.common.previous.common.inputs(root, expected)
    with patch.object(row, "PARENT", Path(plan["parent_request_timing"])):
        row.analyze(root, expected)
    path = root / "timing/submission.json"
    if path.exists():
        schedule(
            root, plan["source_sha"], row.common.previous.read(path)["cpu_analysis"], "finish", 0
        )
    else:
        finish(root, plan["source_sha"])


def finish(root, revision):
    if clean_source(SOURCE) != revision:
        raise ValueError("immutable closeout source required")
    path = root / "timing/analysis/decision.json"
    if not path.exists():
        path = root / "analysis/decision.json"
    decision = row.common.previous.read(path)
    write_once(
        root / "engineering_closeout.json",
        dict(
            source_sha=revision,
            decision_path=str(path),
            decision_sha256=sha(path),
            complete=decision.get("complete", False),
            failures=decision.get("failures", []),
            all_method_similar_time_gate=decision.get("all_method_similar_time_gate", False),
            independent_quality_N=0,
            paper_superiority_established=False,
            no_fresh_or_large_campaign_submitted=True,
            stopping_rule="No further automatic GPU allocation after this bounded hardware study",
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("watch", "worker", "analyze", "finish"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-sha")
    parser.add_argument("--plan-sha")
    parser.add_argument("--step", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "watch":
        watch(args.root, args.source_sha, args.step)
    elif args.mode == "worker":
        row.worker(args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    elif args.mode == "analyze":
        analyze(args.root, args.plan_sha)
    else:
        finish(args.root, args.source_sha)
