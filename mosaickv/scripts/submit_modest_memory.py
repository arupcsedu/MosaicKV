#!/usr/bin/env python3
"""Verify every calibration worker, snapshot tooling, then submit held jobs.

Default is read-only verification. --submit authorizes ONLY the 128-image
calibration and dependent analysis. No confirmation/paper campaign is submitted.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

from prepare_modest_memory import clean_source, digest, write_once


def command(*arguments, **kwargs):
    return subprocess.check_output(arguments, text=True, **kwargs).strip()


def submit(args):
    root = args.root.resolve()
    plan = json.loads((root / "plan.json").read_text())
    plan_sha = digest(root / "plan.json")
    source = Path(plan["source_root"])
    clean_source(source, plan["source_sha"])
    if (
        plan.get("classification") != "prospective_modest_memory_calibration_not_paper_confirmation"
        or plan["max_global_a100s"] != 4
        or len(plan["jobs"]) != 104
        or plan["caps_mib"] != [28, 32, 35, 40]
    ):
        raise ValueError("not the registered 104-worker calibration")
    spec = importlib.util.spec_from_file_location(
        "modest_worker", source / "mosaickv/scripts/run_coco_shared_bytes.py"
    )
    assert spec and spec.loader
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    for i in range(104):
        worker.verify_plan(root, plan_sha, i)
    print(
        json.dumps(
            dict(
                verified_workers=104,
                source_sha=plan["source_sha"],
                plan_sha256=plan_sha,
                max_concurrent_a100s=4,
                confirmation_authorized=False,
            )
        ),
        flush=True,
    )
    if not args.submit:
        return
    if (root / "submission.json").exists() or (root / "tooling").exists():
        raise FileExistsError(
            "submission snapshot is write-once; inspect prior submission before retry"
        )
    repository = Path(__file__).resolve().parents[2]
    files = [
        Path(__file__),
        Path(__file__).with_name("prepare_modest_memory.py"),
        Path(__file__).with_name("analyze_modest_memory.py"),
        repository / "mosaickv/src/mosaickv/evaluation/cider_resampling.py",
        repository / "mosaickv/slurm/modest_memory_calibration.sbatch",
        repository / "mosaickv/slurm/modest_memory_analysis.sbatch",
    ]
    tooling = root / "tooling"
    tooling.mkdir()
    for path in files:
        shutil.copy2(path, tooling / path.name)
    write_once(tooling / "source_hashes.json", {path.name: digest(path) for path in files})
    logs = root / "logs"
    logs.mkdir()
    controller_source = Path("/sfs/weka/scratch/djy8hg/worktrees/mosaickv-scheduler-cap4-e8d2ee1")
    subprocess.run(
        ["bash", "mosaickv/scripts/assert_clean_worktree.sh"], cwd=controller_source, check=True
    )
    scheduler = root / "scheduler"
    scheduler.mkdir()
    for name in ("control_a100_schedule.py", "cache_env.sh"):
        shutil.copy2(controller_source / "mosaickv/scripts" / name, scheduler / name)
    result = dict(
        plan_sha256=plan_sha,
        inference_source_sha=plan["source_sha"],
        max_global_a100s=4,
        user_ceiling=6,
        confirmation_submitted=False,
    )
    try:
        gpu = command(
            "sbatch",
            "--parsable",
            "--hold",
            "--array=0-103%4",
            "--account=bii_dsc_community",
            f"--output={logs}/gpu-%A_%a.out",
            f"--error={logs}/gpu-%A_%a.err",
            str(tooling / "modest_memory_calibration.sbatch"),
            str(source),
            str(root),
            plan_sha,
        ).split(";")[0]
        result["gpu_array"] = gpu
        write_once(
            root / "gpu-submission.json",
            dict(job_id=gpu, state="held_pending_shared_controller", plan_sha256=plan_sha),
        )
        analysis = command(
            "sbatch",
            "--parsable",
            f"--dependency=afterany:{gpu}",
            "--account=bii_dsc_community",
            f"--output={logs}/analysis-%j.out",
            f"--error={logs}/analysis-%j.err",
            str(tooling / "modest_memory_analysis.sbatch"),
            str(source),
            str(root),
            plan_sha,
        ).split(";")[0]
        result["analysis_job"] = analysis
        policy = dict(
            arrays=[gpu],
            adopt_existing_holds=[gpu],
            maximum_a100s=4,
            maximum_worker_minutes=120,
            timezone="America/New_York",
            window="all_times",
        )
        write_once(scheduler / "a100_schedule.json", policy)
        environment = dict(os.environ, MOSAICKV_POLICY_ROOT=str(scheduler))
        controller = command(
            "sbatch",
            "--parsable",
            "--time=12:00:00",
            "--account=bii_dsc_community",
            f"--output={logs}/controller-%j.out",
            f"--error={logs}/controller-%j.err",
            str(controller_source / "mosaickv/slurm/a100_schedule_controller.sbatch"),
            env=environment,
        ).split(";")[0]
        result["controller_job"] = controller
        tick = command(
            "/scratch/djy8hg/env/mosaickv/bin/python",
            str(scheduler / "control_a100_schedule.py"),
            "--policy",
            str(scheduler / "a100_schedule.json"),
            "--state",
            str(scheduler / "state.json"),
            "--apply",
        )
        result["initial_admission"] = json.loads(tick)
        result["status"] = "submitted"
    except Exception as error:
        result.update(status="submission_incomplete", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_once(root / "submission.json", result)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    submit(parser.parse_args())
