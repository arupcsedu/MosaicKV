#!/usr/bin/env python3
"""Submit only the registered small repair/development chain, never a full campaign."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from bounded_history_pilot import (
    SOURCE,
    clean_source,
    digest,
    inputs,
    prepare,
    prepare_retry,
    write_once,
)

PARENT = Path("/scratch/djy8hg/runs/mosaickv/bounded-history-baselines-v1-20261023")
PARENT_SHA = "2dbfaa77622dd4ce367bea99959b0426ba1bcbbda62a1a1cbc8a931f1e1d043a"
PREFIX = Path("/scratch/djy8hg/runs/mosaickv/prefixkv/full-campaign-e4de390-20261019-v6")
PREFIX_SHA = "2500e5773cfb98c6d5aee8eda69359dd9fed77d5fc117dbec6af8654a73dc704"


def submit(root: Path) -> dict:
    sha = clean_source()
    root = root.resolve()
    if root.exists() or Path("/scratch/djy8hg").resolve() not in root.parents:
        raise ValueError("submission requires a new scratch output root")
    retry, grid = root / "vl-cache-retry", root / "memory-grid"
    prepare_retry(retry, PARENT, PARENT_SHA)
    prepare(grid, SOURCE / "mosaickv/configs/experiments/bounded_history_memory_v2.yaml")
    retry_sha, grid_sha = digest(retry / "plan.json"), digest(grid / "plan.json")
    retry_plan, grid_plan = inputs(retry, retry_sha), inputs(grid, grid_sha)
    for directory, plan_sha, plan in ((retry, retry_sha, retry_plan), (grid, grid_sha, grid_plan)):
        for job in plan["jobs"]:
            inputs(directory, plan_sha, job["index"])
    if digest(PREFIX / "plan.json") != PREFIX_SHA:
        raise ValueError("PrefixKV completed campaign plan hash mismatch")
    references = [job["index"] for job in grid_plan["jobs"] if job["method"] == "full_kv"]
    compressed = [job["index"] for job in grid_plan["jobs"] if job["method"] != "full_kv"]
    if len(references) != 2 or len(compressed) != 36:
        raise ValueError("unexpected small-development scope; refusing campaign expansion")
    logs = root / "logs"
    logs.mkdir()
    jobs: dict[str, str] = {}

    def sbatch(name: str, script: str, args: list[str], options: list[str]) -> str:
        result = (
            subprocess.run(
                [
                    "sbatch",
                    "--parsable",
                    "--account=bii_dsc_community",
                    f"--output={logs}/{name}-%A_%a.out",
                    f"--error={logs}/{name}-%A_%a.err",
                    *options,
                    str(SOURCE / script),
                    *args,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            .stdout.strip()
            .split(";", 1)[0]
        )
        jobs[name] = result
        write_once(root / "submission-events" / f"{name}.json", dict(job_id=result))
        return result

    gpu_script = "mosaickv/slurm/bounded_history_pilot.sbatch"
    try:
        vl = sbatch(
            "vl-retry",
            gpu_script,
            [str(SOURCE), str(retry), retry_sha],
            ["--hold", "--array=4,9,15,20%6", "--kill-on-invalid-dep=yes"],
        )
        ref = sbatch(
            "references",
            gpu_script,
            [str(SOURCE), str(grid), grid_sha],
            [
                "--hold",
                "--array=" + ",".join(map(str, references)) + "%6",
                "--kill-on-invalid-dep=yes",
            ],
        )
        gpu = sbatch(
            "memory-grid",
            gpu_script,
            [str(SOURCE), str(grid), grid_sha],
            [
                "--hold",
                "--array=" + ",".join(map(str, compressed)) + "%6",
                f"--dependency=afterok:{vl}:{ref}",
                "--kill-on-invalid-dep=yes",
            ],
        )
        for name, directory, plan_sha, dependency in (
            ("vl-analysis", retry, retry_sha, vl),
            ("memory-analysis", grid, grid_sha, gpu),
        ):
            sbatch(
                name,
                "mosaickv/slurm/bounded_history_analysis.sbatch",
                [str(SOURCE), str(directory), plan_sha],
                [f"--dependency=afterany:{dependency}"],
            )
        sbatch(
            "prefixkv-reanalysis",
            "mosaickv/slurm/prefixkv_reanalysis.sbatch",
            [
                str(SOURCE),
                str(PREFIX),
                PREFIX_SHA,
                str(PREFIX / "authorization/full-campaign.json"),
                str(root / "prefixkv-analysis"),
            ],
            [],
        )
        snapshot = root / "scheduler"
        snapshot.mkdir()
        for filename in ("control_a100_schedule.py", "cache_env.sh"):
            shutil.copy2(SOURCE / "mosaickv/scripts" / filename, snapshot / filename)
        policy = dict(
            timezone="America/New_York",
            maximum_worker_minutes=120,
            maximum_a100s=6,
            six_a100s_explicitly_authorized=True,
            window="all_times",
            arrays=[vl, ref, gpu],
            adopt_existing_holds=[vl, ref, gpu],
            release_after_arrays_finish={gpu: [vl, ref]},
            authorization="user_2026_10_23_six_A100_memory_efficiency_validation",
            source_sha=sha,
        )
        write_once(snapshot / "a100_schedule.json", policy)
        write_once(
            snapshot / "snapshot.json",
            dict(
                source_sha=sha, files={p.name: digest(p) for p in snapshot.iterdir() if p.is_file()}
            ),
        )
        os.environ["MOSAICKV_POLICY_ROOT"] = str(snapshot)
        sbatch(
            "controller",
            "mosaickv/slurm/a100_schedule_controller.sbatch",
            [],
            ["--time=04:00:00", "--export=ALL,MOSAICKV_POLICY_ROOT=" + str(snapshot)],
        )
    except Exception:
        # Never release partial submissions without a working aggregate controller.
        write_once(
            root / "submission-failed.json",
            dict(
                source_sha=sha,
                jobs=jobs,
                action="registered GPU arrays remain held; inspect before retry",
            ),
        )
        raise
    result = dict(
        source_sha=sha,
        jobs=jobs,
        root=str(root),
        retry_plan_sha=retry_sha,
        grid_plan_sha=grid_sha,
        reused_workers=18,
        new_GPU_workers=42,
        maximum_concurrent_a100s=6,
        cpus_per_worker=12,
        worker_time_limit="02:00:00",
        paper_campaign_submitted=False,
    )
    write_once(root / "submission.json", result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    submit(parser.parse_args().root)
