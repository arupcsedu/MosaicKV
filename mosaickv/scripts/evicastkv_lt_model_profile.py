#!/usr/bin/env python3
"""Run actual-model attribution only after a negative registered latency gate."""

import argparse
import os
import subprocess
import traceback
from pathlib import Path

import evicastkv_lt_latency as common

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE, driver = common.SOURCE, common.driver
PARENT = Path("/scratch/djy8hg/runs/evicastkv/lt-reader-consumed-20261104-v1")
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_lt_model_profile.sbatch"
ARMS = (
    ("mosaickv_exact", "graph"),
    ("mosaickv_exact", "graph-lt"),
    ("prefixkv_reimpl", "graph-original"),
)


def submit(root, action, dependency=None):
    plan = driver.checked(root, sha(root / "plan.json"))
    args = ["sbatch", "--parsable", "--account=bii_dsc_community", "--nodes=1", "--ntasks=1"]
    if action == "profile":
        driver.cpu_proof(root, plan)
        args += [
            "--partition=bii-gpu",
            "--reservation=bi_fox_dgx",
            "--gres=gpu:a100:1",
            "--cpus-per-task=12",
            "--mem=48G",
            "--array=0-1%2",
            "--time=00:20:00",
        ]
    else:
        args += ["--partition=bii", "--cpus-per-task=2", "--mem=16G", "--time=00:25:00"]
    if dependency:
        args.append("--dependency=afterany:" + dependency)
    (root / "slurm").mkdir(exist_ok=True)
    args += [
        f"--job-name=mk-ltprof-{action}",
        f"--output={root}/slurm/{action}-%A_%a.out",
        f"--error={root}/slurm/{action}-%A_%a.err",
        str(SCRIPT),
        str(SOURCE),
        str(root),
        sha(root / "plan.json"),
        action,
    ]
    job = subprocess.check_output(args, text=True).strip()
    if action == "profile":
        from control_a100_schedule import tick

        policy = dict(
            timezone="America/New_York",
            maximum_worker_minutes=120,
            maximum_a100s=6,
            six_a100s_explicitly_authorized=True,
            arrays=[job],
        )
        write_once(root / "scheduler/policy.json", policy)
        write_once(
            root / "scheduler/registration.json",
            tick(policy, {}, root / "scheduler/state.json", False),
        )
    return job


def prepare(root, revision):
    if root.exists() or clean_source(SOURCE) != revision:
        raise ValueError("clean immutable source and new output root required")
    old = driver.read(PARENT / "plan.json")
    gate = driver.read(PARENT / "analysis/decision.json")
    if not gate["complete"] or not all(
        r["smaller_than_ALL_baseline_controls"] for r in gate["storage_controls"]
    ):
        raise ValueError("complete positive control/storage gate required")
    bound = dict(old["bound_files"])
    paths = [
        PARENT / "plan.json",
        PARENT / "analysis/decision.json",
        SOURCE / "mosaickv/docs/lt_model_profile_20261104.md",
    ]
    paths += [PARENT / f"workers/{i}/study.json" for i in (0, 1)]
    bound.update({str(p): sha(p) for p in paths})
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=revision,
            bound_files=bound,
            stage="consumed_native_actual_model_attribution",
            independent_quality_N=0,
            acceptance_not_established_by_profile=True,
            automatic_quality_submission=False,
            automatic_timing_retry=False,
            source_runtime_unchanged=True,
        ),
    )
    parent_analysis = driver.read(PARENT / "paired-timing/timing_submission.json")["analysis"]
    cpu = submit(root, "cpu")
    release = submit(root, "release", f"{cpu}:{parent_analysis}")
    write_once(
        root / "cpu_submission.json",
        dict(cpu=cpu, release=release, afterany_parent_analysis=parent_analysis),
    )


def release(root, expected):
    plan = driver.checked(root, expected)
    driver.cpu_proof(root, plan)
    path = PARENT / "paired-timing/analysis/decision.json"
    decision = driver.read(path)
    if not decision["complete"] or decision["four_percent_latency_gate"]:
        write_once(
            root / "release.json",
            dict(
                profile_released=False,
                parent_complete=decision["complete"],
                parent_latency_pass=decision["four_percent_latency_gate"],
                parent_decision=str(path),
                parent_sha256=sha(path),
                no_GPU_submitted=True,
            ),
        )
        return
    write_once(
        root / "release.json",
        dict(
            profile_released=True,
            parent_decision=str(path),
            parent_sha256=sha(path),
            no_overlapping_parent_GPU_timing=True,
        ),
    )
    job = submit(root, "profile")
    analysis = submit(root, "analysis", job)
    write_once(root / "profile_submission.json", dict(worker=job, analysis=analysis))


def worker(root, expected, index):
    out = root / f"workers/{index}"
    out.mkdir(parents=True, exist_ok=False)
    result = dict(
        status="failed", failures=[], source_sha=clean_source(SOURCE), plan_sha256=expected
    )
    try:
        activation = driver.read(root / "release.json")
        if (
            not activation["profile_released"]
            or sha(Path(activation["parent_decision"])) != activation["parent_sha256"]
        ):
            raise ValueError("reviewable unchanged negative parent gate required")
        from mosaickv.evaluation.lt_model_profile import collect

        plan, adapter, refs, messages, profile, _ = driver.setup(
            root,
            expected,
            index,
            manifest_overrides=dict(
                variant="consumed_native_actual_model_attribution",
                methods=["mosaickv_exact", "prefixkv_reimpl"],
                actual_timing_protocol=dict(
                    output_predictions=128, profiled_steps=[1, 64, 126], not_latency_evidence=True
                ),
            ),
        )
        records = driver.read(PARENT / f"workers/{index}/study.json")["records"]
        observations = []
        for method, arm in ARMS:
            folder = out / f"{method}-{arm}"
            folder.mkdir()

            def call(method=method, arm=arm):
                return common.execute(
                    adapter, refs, messages, profile, plan, method, records[method][arm]
                )

            call()  # Full excluded warmup, still parent-bitwise checked.
            report = collect(adapter, call, folder)
            report.update(method=method, arm=arm)
            write_once(folder / "profile.json", report)
            observations.append(report)
        write_once(
            out / "profile.json",
            dict(
                records=observations,
                independent_quality_N=0,
                no_quality_or_request_latency_claim=True,
            ),
        )
        result["status"] = "completed"
    except Exception as error:
        result["failures"].append(f"{type(error).__name__}: {error}")
        result["traceback"] = traceback.format_exc()
    driver.finish_worker(out, result)


def analyze(root, expected):
    plan = driver.checked(root, expected)
    rows, failures = driver.load_producers(root, plan, "profile.json")
    write_once(
        root / "analysis/decision.json",
        dict(
            complete=not failures,
            failures=failures,
            rows=rows,
            source_sha=plan["source_sha"],
            plan_sha256=expected,
            independent_quality_N=0,
            no_reader_promoted=True,
            no_automatic_quality_or_timing_submission=True,
            instrumented_phases_NOT_latency_acceptance=True,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "cpu", "release", "profile", "analysis"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-sha")
    parser.add_argument("--plan-sha")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root, args.source_sha)
    elif args.action == "cpu":
        driver.pair.common.previous.cpu(args.root, args.plan_sha)
    elif args.action == "release":
        release(args.root, args.plan_sha)
    elif args.action == "profile":
        worker(args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    else:
        analyze(args.root, args.plan_sha)


if __name__ == "__main__":
    main()
