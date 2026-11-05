#!/usr/bin/env python3
"""Finite CPU -> two synthetic A100 operator workers -> write-once analysis."""

import argparse
import importlib.metadata
import os
import subprocess
import traceback
from dataclasses import asdict
from pathlib import Path

import evicastkv_prompt_graph_latency as previous

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE, driver = previous.SOURCE, previous.driver
PARENT = Path("/scratch/djy8hg/runs/evicastkv/prompt-graph-owned-consumed-20261104-v1")
PROFILE = Path("/scratch/djy8hg/runs/evicastkv/prompt-graph-profile-consumed-20261104-v1")
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_row4_operator.sbatch"


def submit(root, action, dependency=None):
    plan = driver.checked(root, sha(root / "plan.json"))
    gpu = action == "operator"
    if gpu:
        driver.cpu_proof(root, plan)
    args = ["sbatch", "--parsable", "--account=bii_dsc_community", "--nodes=1", "--ntasks=1"]
    args += (
        [
            "--partition=bii-gpu",
            "--reservation=bi_fox_dgx",
            "--gres=gpu:a100:1",
            "--cpus-per-task=12",
            "--mem=24G",
            "--array=0-1%2",
            "--time=00:15:00",
        ]
        if gpu
        else ["--partition=bii", "--cpus-per-task=2", "--mem=16G", "--time=00:25:00"]
    )
    if dependency:
        args.append(f"--dependency=afterany:{dependency}")
    (root / "slurm").mkdir(exist_ok=True)
    args += [
        f"--job-name=mk-row4-{action}",
        f"--output={root}/slurm/{action}-%A_%a.out",
        f"--error={root}/slurm/{action}-%A_%a.err",
        str(SCRIPT),
        str(SOURCE),
        str(root),
        sha(root / "plan.json"),
        action,
    ]
    job = subprocess.check_output(args, text=True).strip()
    if gpu:
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


def stage(root, action):
    job = submit(root, action)
    analysis = submit(root, action + "_analysis", job)
    write_once(root / f"{action}_submission.json", dict(worker=job, analysis=analysis))


def prepare(root, revision):
    if root.exists() or clean_source(SOURCE) != revision:
        raise ValueError("new root and clean immutable source required")
    old = driver.read(PARENT / "plan.json")
    bound = dict(old["bound_files"])
    for p in (
        PARENT / "plan.json",
        PARENT / "analysis/decision.json",
        PROFILE / "diagnostic.py",
        PROFILE / "decision.json",
        SOURCE / "mosaickv/docs/row_major_source_operator_20261104.md",
    ):
        bound[str(p)] = sha(p)
    if driver.read(PROFILE / "decision.json")["status"] != "completed":
        raise ValueError("complete measured source/graph attribution required")
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=revision,
            bound_files=bound,
            stage="synthetic_lossless_rowmajor_code_operator_only",
            operator_upper_ratio=0.8,
            no_checkpoint_loaded=True,
            automatic_model_submission=False,
            automatic_quality_submission=False,
            paper_campaign_authorized=False,
        ),
    )
    stage(root, "cpu")


def worker(root, expected, index):
    out = root / f"workers/{index}"
    out.mkdir(parents=True, exist_ok=False)
    result = dict(
        status="failed", failures=[], source_sha=clean_source(SOURCE), plan_sha256=expected
    )
    try:
        import torch

        from mosaickv.evaluation.row_major_source_operator import run
        from mosaickv.measurements.telemetry import capture_gpu_environment

        plan = driver.checked(root, expected)
        driver.cpu_proof(root, plan)
        if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name():
            raise ValueError("one actual A100 required")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ValueError("deterministic CUBLAS environment required before Python starts")
        torch.set_num_threads(12)
        torch.manual_seed(plan["seed"])
        torch.use_deterministic_algorithms(True)
        versions = {}
        for p in ("transformers", "vllm", "sglang"):
            try:
                versions[p] = importlib.metadata.version(p)
            except importlib.metadata.PackageNotFoundError:
                versions[p] = "not_installed"
        write_once(
            out / "manifest.json",
            dict(
                source_sha=plan["source_sha"],
                config_sha=expected,
                git_dirty=False,
                model_id="not_used_synthetic_operator",
                model_revision="not_used",
                dataset="generated_code_transport_fixture",
                dataset_revision=plan["source_sha"],
                torch=torch.__version__,
                cuda=torch.version.cuda,
                installed_versions=versions,
                frameworks_not_executed=["transformers", "vllm", "sglang"],
                gpu_type=torch.cuda.get_device_name(),
                gpu_count=1,
                backend="triton",
                attention_implementation="not_used",
                seed=plan["seed"],
                measurement_type="synthetic_operator_diagnostic_measured",
                environment=asdict(capture_gpu_environment(torch)),
                slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                slurm_array_task_id=os.environ.get("SLURM_ARRAY_TASK_ID"),
                independent_quality_N=0,
            ),
        )
        write_once(out / "operator.json", run(index))
        result["status"] = "completed"
    except Exception as error:
        result["failures"].append(f"{type(error).__name__}: {error}")
        result["traceback"] = traceback.format_exc()
    driver.finish_worker(out, result)


def analyze(root, expected, action):
    plan = driver.checked(root, expected)
    failures, rows = [], []
    if action == "cpu_analysis":
        try:
            driver.cpu_proof(root, plan)
        except Exception as error:
            failures.append(str(error))
    else:
        rows, failures = driver.load_producers(root, plan, "operator.json")
    passed = not failures and len(rows) == 2 and all(x["summary"]["operator_gate"] for x in rows)
    write_once(
        root / ("cpu-analysis" if action == "cpu_analysis" else "analysis") / "decision.json",
        dict(
            complete=not failures,
            failures=failures,
            rows=rows,
            operator_gate=passed,
            source_sha=plan["source_sha"],
            plan_sha256=expected,
            independent_quality_N=0,
            model_latency_or_quality_established=False,
            no_reader_promoted=True,
            no_automatic_model_or_quality_submission=True,
        ),
    )
    if action == "cpu_analysis" and not failures:
        stage(root, "operator")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("prepare", "cpu", "cpu_analysis", "operator", "operator_analysis")
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-sha")
    parser.add_argument("--plan-sha")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root, args.source_sha)
    elif args.action == "cpu":
        driver.pair.common.previous.cpu(args.root, args.plan_sha)
    elif args.action == "operator":
        worker(args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    else:
        analyze(args.root, args.plan_sha, args.action)


if __name__ == "__main__":
    main()
