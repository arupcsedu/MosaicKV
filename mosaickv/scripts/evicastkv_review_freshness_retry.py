#!/usr/bin/env python3
"""Single finite CPU-only linear-scanner retry; never authorizes confirmation."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

from evicastkv_review_followups import cpu_gate

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
OLD = Path("/scratch/djy8hg/audits/fox-review-v7-closure-20261110")


def read(path):
    return json.loads(Path(path).read_text())


def submit(root, label, mode, dependencies=(), *, tests=False, array=False):
    command = [
        "sbatch",
        "--parsable",
        "--account=bii_dsc_community",
        "--nodes=1",
        "--ntasks=1",
        "--partition=bii",
        "--cpus-per-task=4",
        "--mem=16G",
        "--time=00:15:00",
        f"--job-name=fox7scan-{label}",
        f"--output={root}/slurm/{label}-%A_%a.out",
        f"--error={root}/slurm/{label}-%A_%a.err",
    ]
    if dependencies:
        if any(re.fullmatch(r"\d+", str(value)) is None for value in dependencies):
            raise ValueError("concrete afterany scheduler IDs required")
        command += ["--dependency=afterany:" + ":".join(map(str, dependencies))]
    if array:
        command += ["--array=0-7%4"]
    name = "evicastkv_review_closure_v2" if tests else "evicastkv_review_freshness_retry"
    command += [
        str(SOURCE / f"mosaickv/slurm/{name}.sbatch"),
        str(SOURCE),
        str(root / "full-cpu") if tests else str(root),
        "tests" if tests else mode,
    ]
    (root / "slurm").mkdir(parents=True, exist_ok=True)
    write_once(root / f"jobs/{label}-request.json", dict(command=command, CPU_only=True))
    value = subprocess.check_output(command, text=True).strip().split(";", 1)[0]
    if re.fullmatch(r"\d+", value) is None:
        raise ValueError("scheduler returned no concrete job ID")
    write_once(root / f"jobs/{label}-submission.json", dict(job=value, command=command))
    return value


def register(root, *, execute=False):
    root = Path(root)
    root.resolve().relative_to(Path("/scratch/djy8hg/audits").resolve())
    if root.exists():
        raise FileExistsError("new write-once CPU retry root required")
    revision = clean_source(SOURCE)
    previous = read(OLD / "registration.json")
    clean_source(Path(previous["source_root"]), previous["source_sha"])
    paths = [
        OLD / "registration.json",
        Path(__file__),
        SOURCE / "mosaickv/scripts/evicastkv_review_freshness_stream.py",
        SOURCE / "mosaickv/scripts/evicastkv_review_freshness_worker.py",
    ]
    old_gate = OLD / "freshness/GATE_freshness_workers.json"
    if old_gate.exists():
        paths.append(old_gate)
    write_once(
        root / "registration.json",
        dict(
            source_sha=revision,
            execute=execute,
            bindings={str(path): sha(path) for path in paths},
            old_source=previous["source_root"],
            old_source_sha=previous["source_sha"],
            old_array="22763193",
            old_array_canceled_by_parent=True,
            old_gate_ready=old_gate.exists(),
            shard_count=8,
            concurrency=4,
            maximum_GPU_workers=0,
            fresh_confirmation_authorized=False,
            cohort_selection_authorized=False,
        ),
    )
    if not execute:
        return dict(status="dry_run", submitted_jobs=0)
    tests = submit(root, "cpu", "tests", tests=True)
    return dict(cpu=tests, launch=submit(root, "launch", "launch", [tests]))


def load(root):
    lock = read(root / "registration.json")
    clean_source(SOURCE, lock["source_sha"])
    if lock["execute"] is not True:
        raise ValueError("dry-run retry cannot execute")
    for path, expected in lock["bindings"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"immutable scanner retry binding changed: {path}")
    return lock


def run(root, mode):
    root = Path(root)
    lock = load(root)
    if mode == "launch":
        try:
            write_once(root / "GATE_cpu_suite.json", cpu_gate(root, lock["source_sha"]))
        except Exception as error:
            value = dict(
                status="blocked_cpu", error=str(error), fresh_confirmation_authorized=False
            )
            write_once(root / "FINAL_report.json", value)
            return value
        workers = submit(root, "workers", "worker", array=True)
        return dict(workers=workers, analysis=submit(root, "analysis", "analyze", [workers]))
    import evicastkv_review_freshness_worker as scanner

    if mode == "worker":
        return scanner.worker(root / "freshness", int(os.environ["SLURM_ARRAY_TASK_ID"]))
    if mode != "analyze":
        raise ValueError("unregistered CPU retry mode")
    try:
        gate = scanner.aggregate(root / "freshness")
        result = dict(
            status="completed_CPU_inventory_not_confirmation",
            scanner_gate=gate,
            source_sha=lock["source_sha"],
            old_array_preserved="22763193",
            GPU_jobs=0,
            cohort_selected=False,
            fresh_confirmation_authorized=False,
        )
    except Exception as error:
        result = dict(
            status="blocked_inventory",
            error=f"{type(error).__name__}: {error}",
            fresh_confirmation_authorized=False,
        )
    write_once(root / "FINAL_report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("register", "launch", "worker", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    result = (
        register(args.root, execute=args.execute)
        if args.mode == "register"
        else run(args.root, args.mode)
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
