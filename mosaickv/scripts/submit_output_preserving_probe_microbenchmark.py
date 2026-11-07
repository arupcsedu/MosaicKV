#!/usr/bin/env python3
"""Submit the one-worker V6 surrogate probe timing diagnostic."""

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
        plan.get("protocol_version")
        not in {
            "output_preserving_probe_microbenchmark_v6",
            "output_preserving_probe_microbenchmark_v7",
        }
        or plan.get("timing_diagnostic_only") is not True
        or plan.get("quality_evidence") is not False
        or plan.get("source_root") != str(source)
        or plan.get("source_content_sha256") != source_digest(source)
        or len(plan.get("jobs", ())) != 1
        or plan.get("max_global_a100s") != 4
    ):
        raise ValueError("plan is not the registered probe diagnostic")
    observed_sha = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if observed_sha != plan["source_base_git_sha"]:
        raise ValueError("source base SHA changed")
    for record in plan["source_overlay_files"]:
        path = source / record["path"]
        if not path.is_file() or digest(path) != record["sha256"]:
            raise ValueError(f"source overlay changed: {record['path']}")

    slurm = root / "slurm"
    slurm.mkdir(parents=True, exist_ok=False)
    array_id = _submit(
        [
            "sbatch",
            "--parsable",
            "--hold",
            "--array=0",
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
            "source_content_sha256": plan["source_content_sha256"],
            "classification": plan["classification"],
        },
    )
    controller_id = _submit(
        [
            "sbatch",
            "--parsable",
            "--job-name=mk-op-probe-controller",
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
            "--job-name=mk-op-probe-analysis",
            "--nodes=1",
            "--ntasks=1",
            "--cpus-per-task=2",
            "--mem=8G",
            "--time=00:15:00",
            "--partition=bii",
            f"--output={slurm}/%j-analysis.out",
            f"--error={slurm}/%j-analysis.err",
            "--wrap",
            (
                f"source {source}/mosaickv/scripts/cache_env.sh && "
                f"export PYTHONPATH={source}/mosaickv/scripts && "
                f"/scratch/djy8hg/env/mosaickv/bin/python "
                f"{source}/mosaickv/scripts/"
                f"analyze_output_preserving_probe_microbenchmark.py "
                f"--root {root} --plan-sha256 {args.plan_sha256}"
            ),
        ]
    )
    submission = {
        "plan_sha256": args.plan_sha256,
        "array_job_id": array_id,
        "controller_job_id": controller_id,
        "analysis_job_id": analysis_id,
        "array_indices": [0],
        "array_throttle": 1,
        "canary_submitted": False,
        "calibration_submitted": False,
        "paper_campaign_submitted": False,
    }
    write_once(root / "submission.json", submission)
    print(json.dumps(submission, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    submit(parser.parse_args())
