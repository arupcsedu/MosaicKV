#!/usr/bin/env python3
"""Finite CPU-only recovery of already registered oversize metadata artifacts.

No GPU command, new scan, cohort construction or scientific promotion exists in
this controller. Every submitted request is write-once and never blindly retried.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import time
from pathlib import Path

import evicastkv_review_v8_diagnostics as shared

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
PARENT_SOURCE = "84900a6a402e97c7d58499a722c159cfc86e46c7"
HISTORY = Path("/scratch/djy8hg/workdir/fox-review-v16/run-20261111-2/history")
PROOF_RECOVERY = Path("/scratch/djy8hg/workdir/fox-review-v15/run-20261111-1/history")
RECOVERY_MODULE = "evicastkv_pending_oversize_recovery"
RECOVERY_GATE = "GATE_pending_oversize_recovery.json"
WORKERS = 2
MODES = ("register", "prepare", "oversize-worker", "after-oversize", "finish")
NO_CLAIMS = dict(
    GPU_execution_authorized=False,
    GPU_requests=0,
    new_scan_authorized=False,
    new_cohort_authorized=False,
    materialization_authorized=False,
    fresh_confirmation_established=False,
    independent_quality_N=0,
    exact_KV_peak_certified=False,
    byte_matched_superiority_established=False,
    strongest_baseline_superiority_established=False,
    six_component_benefits_established=False,
    paper_campaign_authorized=False,
)


def read(path):
    path = Path(path)
    if path.stat().st_size > 8 * 2**20:
        raise ValueError("bounded controller metadata exceeded eight MiB")
    return json.loads(path.read_text())


def module():
    return importlib.import_module(RECOVERY_MODULE)


def parent_bindings():
    paths = (
        HISTORY / "GATE_history_closure_v16.json",
        HISTORY / "registration.json",
        PROOF_RECOVERY / "GATE_history_segment_recovery.json",
        PROOF_RECOVERY / "plan.json",
    )
    history, proof = read(paths[0]), read(paths[2])
    if (
        history["source_sha"] != PARENT_SOURCE
        or history["original_errors_preserved"] is not True
        or proof["original_errors_preserved"] is not True
        or any(gate["fresh_confirmation_authorized"] for gate in (history, proof))
        or any(gate["materialization_authorized"] for gate in (history, proof))
        or any(gate["error_waivers"] != 0 for gate in (history, proof))
    ):
        raise ValueError("unchanged negative historical/proof recovery evidence required")
    return {str(path): sha(path) for path in paths}


def load(root):
    plan = read(Path(root) / "registration.json")
    clean_source(SOURCE, plan["source_sha"])
    if (
        plan["execute"] is not True
        or plan["CPU_only"] is not True
        or plan["parent_source_sha"] != PARENT_SOURCE
        or plan["registered_workers"] != WORKERS
        or any(plan[key] != value for key, value in NO_CLAIMS.items())
    ):
        raise ValueError("only the finite CPU-only V17 scope is executable")
    if any(sha(Path(name)) != expected for name, expected in plan["bindings"].items()):
        raise ValueError("immutable V17 source or parent evidence changed")
    return plan


def command(root, label, mode, *, tests=False, workers=None):
    if (tests and mode != "tests") or (not tests and mode not in MODES[1:]):
        raise ValueError("unregistered CPU controller mode")
    if workers is not None and (mode != "oversize-worker" or workers != WORKERS or tests):
        raise ValueError("only the registered two-worker CPU array is allowed")
    if mode == "oversize-worker" and workers != WORKERS:
        raise ValueError("oversize execution requires the registered CPU array")
    cores = 12 if tests else 1 if workers is not None else 4
    minutes = 20 if tests else 5 if workers is not None else 15
    return [
        "sbatch",
        "--parsable",
        "--account=bii_dsc_community",
        "--nodes=1",
        "--ntasks=1",
        "--partition=bii,standard",
        f"--job-name=fox17-{label}",
        f"--cpus-per-task={cores}",
        "--mem=16G",
        f"--time=00:{minutes:02d}:00",
        f"--output={root}/slurm/{label}-%A_%a.out",
        f"--error={root}/slurm/{label}-%A_%a.err",
        *([f"--array=0-{workers - 1}%{workers}"] if workers is not None else []),
        str(SOURCE / "mosaickv/slurm/evicastkv_review_v17.sbatch"),
        str(SOURCE),
        str(root),
        "tests" if tests else mode,
    ]


def submit(root, label, mode, *, dependencies=(), tests=False, workers=None):
    root = Path(root)
    request = root / f"jobs/{label}-request.json"
    if request.exists():
        raise FileExistsError("scheduler request already attempted; no automatic retry")
    cmd = command(root, label, mode, tests=tests, workers=workers)
    if dependencies:
        identities = list(dict.fromkeys(shared.identity(str(d)) for d in dependencies))
        cmd.insert(2, "--dependency=afterany:" + ":".join(identities))
    (root / "slurm").mkdir(parents=True, exist_ok=True)
    write_once(request, dict(command=cmd, CPU_only=True, automatic_retry=False))
    try:
        raw = subprocess.check_output(cmd, text=True, stderr=subprocess.PIPE, timeout=30)
        job = shared.identity(raw.strip().split(";", 1)[0])
    except Exception as error:
        write_once(
            root / f"jobs/{label}-submission-error.json",
            dict(
                error=f"{type(error).__name__}: {error}",
                acknowledgement_uncertain=True,
                automatic_retry=False,
            ),
        )
        raise shared.SubmissionError("CPU job acknowledgement missing; no blind retry") from error
    write_once(
        root / f"jobs/{label}-submission.json",
        dict(command=cmd, job=job, acknowledged_at_unix_seconds=time.time()),
    )
    return job


def register(root, *, execute=False):
    root = Path(root).resolve()
    allowed = Path("/scratch/djy8hg/workdir").resolve()
    if root.exists() or root == allowed or not root.is_relative_to(allowed):
        raise ValueError("new immutable scratch V17 output root required")
    revision = clean_source(SOURCE)
    paths = (
        Path(__file__),
        SOURCE / "mosaickv/slurm/evicastkv_review_v17.sbatch",
        SOURCE / f"mosaickv/scripts/{RECOVERY_MODULE}.py",
        SOURCE / "mosaickv/scripts/evicastkv_review_closure_v2.py",
        SOURCE / "mosaickv/env/common/requirements.lock",
    )
    write_once(
        root / "registration.json",
        dict(
            source_sha=revision,
            source_root=str(SOURCE),
            parent_source_sha=PARENT_SOURCE,
            execute=execute,
            CPU_only=True,
            registered_workers=WORKERS,
            bindings={**parent_bindings(), **{str(p): sha(p) for p in paths}},
            scope="existing pending oversize metadata only; no historical-root restart",
            typed_refinement_is_separately_scheduled=True,
            **NO_CLAIMS,
        ),
    )
    if not execute:
        return dict(status="disabled", jobs=0, **NO_CLAIMS)
    try:
        cpu = submit(root, "cpu", "tests", tests=True)
        prep = submit(root, "prepare", "prepare", dependencies=[cpu])
        return dict(cpu=cpu, prepare=prep, GPU_requests=0)
    except Exception as error:
        shared.failure(root, "registration-submission", error)
        return finish(root)


def current():
    return [shared.identity(os.environ["SLURM_JOB_ID"])] if os.environ.get("SLURM_JOB_ID") else []


def prepare(root):
    root = Path(root)
    plan = load(root)
    try:
        write_once(root / "GATE_cpu_suite.json", shared.cpu_gate(root, plan["source_sha"]))
        prepared = module().prepare(root / "oversize", cpu_gate_root=root)
        if type(prepared.get("workers")) is not int or prepared["workers"] != WORKERS:
            raise ValueError("pending-artifact preparation changed the registered worker count")
        write_once(root / "oversize-prepared.json", prepared)
        producer = submit(root, "oversize", "oversize-worker", workers=WORKERS)
        analyzer = submit(
            root, "after-oversize", "after-oversize", dependencies=[producer, *current()]
        )
        return dict(producer=producer, analyzer=analyzer, workers=WORKERS, GPU_requests=0)
    except Exception as error:
        shared.failure(root, "prepare", error)
        return finish(root)


def worker(root, index):
    root = Path(root)
    plan = load(root)
    shared.cpu_gate(root, plan["source_sha"])
    if type(index) is not int or not 0 <= index < WORKERS:
        raise ValueError("one registered CPU-array index required")
    if read(root / "oversize-prepared.json")["workers"] != WORKERS:
        raise ValueError("complete immutable oversize preparation required")
    return module().worker(root / "oversize", index)


def after(root):
    root = Path(root)
    load(root)
    try:
        module().analyze(root / "oversize")
        gate = root / "oversize" / RECOVERY_GATE
        read(gate)
        write_once(root / "oversize-analysis-receipt.json", dict(path=str(gate), sha256=sha(gate)))
    except Exception as error:
        shared.failure(root, "after-oversize", error)
    return finish(root)


def finish(root):
    root = Path(root)
    plan = load(root)
    gate_path = root / "oversize" / RECOVERY_GATE
    gate = (
        dict(path=str(gate_path), sha256=sha(gate_path), gate=read(gate_path))
        if gate_path.is_file()
        else dict(status="missing_recovery_gate_not_scientific_success", path=str(gate_path))
    )
    request_names = {
        p.stem.removesuffix("-request") for p in (root / "jobs").glob("*-request.json")
    }
    acknowledged = {
        p.stem.removesuffix("-submission") for p in (root / "jobs").glob("*-submission.json")
    }
    uncertain = sorted(request_names - acknowledged)
    report = dict(
        status="blocked_unacknowledged_CPU_submission"
        if uncertain
        else "finite_CPU_recovery_not_paper_confirmation",
        source_sha=plan["source_sha"],
        parent_source_sha=PARENT_SOURCE,
        recovery_evidence=gate,
        errors={p.name: read(p) for p in sorted((root / "errors").glob("*.json"))},
        unacknowledged_requests=uncertain,
        original_negative_results_preserved=True,
        typed_KV_refinement_not_run_by_controller=True,
        source_corrective_readiness_not_empirical_superiority=True,
        **NO_CLAIMS,
    )
    write_once(root / "FINAL_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.mode == "register":
        result = register(args.root, execute=args.execute)
    elif args.mode == "oversize-worker":
        index = os.environ.get("SLURM_ARRAY_TASK_ID", "")
        if not index.isdigit():
            raise ValueError("explicit CPU-array task identity required")
        result = worker(args.root, int(index))
    else:
        result = {"prepare": prepare, "after-oversize": after, "finish": finish}[args.mode](
            args.root
        )
    print(json.dumps(result, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
