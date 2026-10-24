#!/usr/bin/env python3
"""Submit ONLY registered small V3/V4 development cohorts; no old retries."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from bounded_history_pilot import SOURCE, clean_source, digest, inputs, prepare, write_once


def validate_scope(plan):
    protocol = plan["protocol"]
    version = protocol["experiment_id"]
    is_v4 = version == "bounded_history_memory_v4"
    methods = (
        ["prompt_attention_topk", "mosaickv_exact", "mosaickv_direct_error_v4"]
        if is_v4
        else [
            "prompt_attention_topk",
            "mosaickv_exact",
            "mosaickv_output_impact_v2",
            "mosaickv_output_impact_guarded_v3",
        ]
    )
    if (
        version not in {"bounded_history_memory_v3", "bounded_history_memory_v4"}
        or protocol["tasks"] != ["llava_description", "mmvet_prefixkv"]
        or protocol["caps_mib"] != [32, 40, 80]
        or protocol["samples_per_task"] != 8
        or protocol["methods"] != methods
        or protocol["max_global_a100s"] != 6
        or protocol["paper_campaign_authorized"]
        or protocol["heldout_claim"]
        or protocol["promotion"]["automatic_paper_campaign"]
        or protocol["promotion"]["powered_validation_authorized"]
    ):
        raise ValueError("submission limited to the registered small development validation")
    checks = plan["development_disjointness"]
    if len(checks) != 2 or any(
        row["selected_count"] != 8
        or row["excluded_count"] != (14 if is_v4 else 6)
        or row["overlap_count"]
        or row["selection_uses_answers_or_scores"]
        for row in checks
    ):
        raise ValueError("missing complete development disjointness evidence")
    references = [job["index"] for job in plan["jobs"] if job["method"] == "full_kv"]
    compressed = [job["index"] for job in plan["jobs"] if job["method"] != "full_kv"]
    if len(references) != 2 or len(compressed) != (18 if is_v4 else 24):
        raise ValueError("unexpected GPU work: no retries or large campaigns permitted")
    if is_v4:
        feasibility = plan.get("cpu_feasibility", {})
        if (
            feasibility.get("passed") is not True
            or feasibility.get("source_sha") != plan["source_sha"]
            or feasibility.get("classification")
            != "synthetic_CPU_feasibility_not_pretrained_quality_evidence"
            or plan["bound_files"].get(feasibility.get("path")) != feasibility.get("sha256")
        ):
            raise ValueError("missing source-bound CPU feasibility; GPU submission blocked")
    return references, compressed


def submit(root: Path, version: int = 3) -> dict:
    if version not in {3, 4}:
        raise ValueError("only the small V3/V4 protocols may be submitted")
    sha = clean_source()
    root = root.resolve()
    if root.exists() or Path("/scratch/djy8hg").resolve() not in root.parents:
        raise ValueError("submission requires a new scratch output root")
    prepare(root, SOURCE / f"mosaickv/configs/experiments/bounded_history_memory_v{version}.yaml")
    plan_sha = digest(root / "plan.json")
    plan = inputs(root, plan_sha)
    references, compressed = validate_scope(plan)
    for job in plan["jobs"]:
        inputs(root, plan_sha, job["index"])
    logs = root / "logs"
    logs.mkdir()
    jobs = {}

    def sbatch(name, script, args, options):
        job_id = (
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
        jobs[name] = job_id
        write_once(root / "submission-events" / f"{name}.json", dict(job_id=job_id))
        return job_id

    args = [str(SOURCE), str(root), plan_sha]
    gpu_script = "mosaickv/slurm/bounded_history_pilot.sbatch"
    try:
        ref = sbatch(
            "references",
            gpu_script,
            args,
            [
                "--hold",
                "--array=" + ",".join(map(str, references)) + "%6",
                "--kill-on-invalid-dep=yes",
            ],
        )
        gpu = sbatch(
            "direct-error-grid" if version == 4 else "guarded-grid",
            gpu_script,
            args,
            [
                "--hold",
                "--array=" + ",".join(map(str, compressed)) + "%6",
                f"--dependency=afterok:{ref}",
                "--kill-on-invalid-dep=yes",
            ],
        )
        sbatch(
            "analysis",
            "mosaickv/slurm/bounded_history_analysis.sbatch",
            args,
            [f"--dependency=afterany:{gpu}"],
        )
        snapshot = root / "scheduler"
        snapshot.mkdir()
        for filename in ("control_a100_schedule.py", "cache_env.sh"):
            shutil.copy2(SOURCE / "mosaickv/scripts" / filename, snapshot / filename)
        write_once(
            snapshot / "a100_schedule.json",
            dict(
                timezone="America/New_York",
                maximum_worker_minutes=120,
                maximum_a100s=6,
                six_a100s_explicitly_authorized=True,
                window="all_times",
                arrays=[ref, gpu],
                adopt_existing_holds=[ref, gpu],
                release_after_arrays_finish={gpu: [ref]},
                source_sha=sha,
                authorization="user_six_A100_small_disjoint_development_only_no_paper_campaign",
            ),
        )
        write_once(
            snapshot / "snapshot.json",
            dict(
                source_sha=sha, files={p.name: digest(p) for p in snapshot.iterdir() if p.is_file()}
            ),
        )
        sbatch(
            "controller",
            "mosaickv/slurm/a100_schedule_controller.sbatch",
            [],
            ["--time=04:00:00", "--export=ALL,MOSAICKV_POLICY_ROOT=" + str(snapshot)],
        )
    except Exception:
        write_once(
            root / "submission-failed.json",
            dict(
                source_sha=sha,
                jobs=jobs,
                action="partial GPU submissions remain held; inspect before resubmitting",
            ),
        )
        raise
    result = dict(
        source_sha=sha,
        root=str(root),
        plan_sha256=plan_sha,
        jobs=jobs,
        new_gpu_workers=len(plan["jobs"]),
        expected_sample_rows=len(plan["jobs"]) * plan["protocol"]["samples_per_task"],
        reused_gpu_results=0,
        maximum_concurrent_a100s=6,
        cpus_per_worker=12,
        worker_time_limit="02:00:00",
        paper_campaign_submitted=False,
        powered_validation_submitted=False,
    )
    write_once(root / "submission.json", result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--version", type=int, choices=[3, 4], default=3)
    args = parser.parse_args()
    submit(args.root, args.version)
