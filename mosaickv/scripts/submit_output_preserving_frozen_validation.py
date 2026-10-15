#!/usr/bin/env python3
"""Submit the frozen validation under aggregate four-A100 admission control."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from prepare_output_preserving_sweep import source_digest
from run_prefixkv_decode_preflight import digest, write_once


def _submit(command: list[str]) -> str:
    return subprocess.run(
        command, check=True, capture_output=True, text=True
    ).stdout.strip().split(";", 1)[0]


def submit(args: argparse.Namespace) -> None:
    source, root = args.source.resolve(), args.root.resolve()
    plan_path = root / "plan.json"
    if digest(plan_path) != args.plan_sha256:
        raise ValueError("plan hash mismatch")
    plan = json.loads(plan_path.read_text())
    if (
        plan.get("source_root") != str(source)
        or plan.get("source_worktree_dirty") is not False
        or plan.get("source_content_sha256") != source_digest(source)
        or len(plan.get("jobs", ())) != 4
        or plan.get("max_global_a100s") != 4
        or plan.get("paper_campaign_submission_authorized") is not False
    ):
        raise ValueError("plan is not the clean four-arm frozen validation")
    if subprocess.run(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip():
        raise ValueError("submission requires a clean immutable source worktree")
    slurm = root / "slurm"
    slurm.mkdir(parents=True, exist_ok=False)
    array_id = _submit(
        [
            "sbatch",
            "--parsable",
            "--hold",
            "--array=0-3%4",
            f"--output={slurm}/%A_%a.out",
            f"--error={slurm}/%A_%a.err",
            str(source / "mosaickv/slurm/output_preserving_sweep.sbatch"),
            str(source),
            str(root),
            args.plan_sha256,
        ]
    )
    policy_path = root / "a100_policy.json"
    state_path = root / "a100_state.json"
    write_once(
        policy_path,
        {
            "timezone": "America/New_York",
            "maximum_worker_minutes": 120,
            "maximum_a100s": 4,
            "window": "all_times",
            "arrays": [int(array_id)],
            "adopt_existing_holds": [int(array_id)],
            "blocked_arrays": {},
            "policy_scope": "aggregate_across_owned_A100_jobs",
            "never_cancel_or_suspend_running_jobs": True,
            "source_git_sha": plan["source_base_git_sha"],
        },
    )
    controller_id = _submit(
        [
            "sbatch",
            "--parsable",
            "--job-name=mk-op-frozen-controller",
            "--nodes=1",
            "--ntasks=1",
            "--cpus-per-task=1",
            "--mem=256M",
            "--time=03:00:00",
            "--partition=bii",
            f"--output={slurm}/%j-controller.out",
            f"--error={slurm}/%j-controller.err",
            "--wrap",
            (
                f"/scratch/djy8hg/env/mosaickv/bin/python "
                f"{source}/mosaickv/scripts/control_a100_schedule.py "
                f"--policy {policy_path} --state {state_path} --apply --watch"
            ),
        ]
    )
    analysis_id = _submit(
        [
            "sbatch",
            "--parsable",
            f"--dependency=afterany:{array_id}",
            f"--output={slurm}/%j-analysis.out",
            f"--error={slurm}/%j-analysis.err",
            str(source / "mosaickv/slurm/output_preserving_frozen_validation_analysis.sbatch"),
            str(source),
            str(root),
            args.plan_sha256,
        ]
    )
    write_once(
        root / "submission.json",
        {
            "plan_sha256": args.plan_sha256,
            "source_git_sha": plan["source_base_git_sha"],
            "array_job_id": array_id,
            "controller_job_id": controller_id,
            "analysis_job_id": analysis_id,
            "array_indices": [0, 1, 2, 3],
            "array_throttle": 4,
            "paper_campaign_submitted": False,
        },
    )
    print((root / "submission.json").read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    submit(parser.parse_args())
