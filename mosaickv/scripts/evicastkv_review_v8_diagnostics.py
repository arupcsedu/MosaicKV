#!/usr/bin/env python3
"""Finite consumed-input follow-up; no new cohort or model-output correction."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import traceback
from pathlib import Path

from evicastkv_review_followups import cpu_gate, gpu_snapshot

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
V7 = Path("/scratch/djy8hg/audits/fox-review-v7-closure-20261110")
FRESHNESS = Path("/scratch/djy8hg/audits/fox-review-v7-freshness-linear-20261110")
MODULES = {
    "rmsnorm": "evicastkv_boundary_rmsnorm_replay",
    "protection": "evicastkv_source_protection_diagnostic",
    "freshness": "evicastkv_review_freshness_continue",
}
GATES = {
    "rmsnorm": "rmsnorm/GATE_native_rmsnorm_replay.json",
    "protection": "protection/analysis/GATE_source_protection.json",
}


class SubmissionError(RuntimeError):
    """A submission failed or its scheduler acknowledgement was lost."""


def read(path):
    return json.loads(Path(path).read_text())


def identity(value):
    text = str(value).strip().split(";", 1)[0]
    if not text.isascii() or not text.isdecimal():
        raise ValueError("a concrete Slurm job ID is required")
    return text


def submit(
    root, label, mode, dependencies=(), *, gpu=False, tests=False, launcher=None, job_prefix="fox8"
):
    if gpu and tests:
        raise ValueError("the CPU test suite must not request a GPU")
    command = [
        "sbatch",
        "--parsable",
        "--account=bii_dsc_community",
        "--nodes=1",
        "--ntasks=1",
        f"--job-name={job_prefix}-{label}",
        f"--output={root}/slurm/{label}-%j.out",
        f"--error={root}/slurm/{label}-%j.err",
    ]
    if gpu:
        command += [
            "--partition=bii-gpu",
            "--reservation=bi_fox_dgx",
            "--gres=gpu:a100:1",
            "--cpus-per-task=12",
        ]
        command += (
            ["--mem=8G", "--time=00:20:00"]
            if mode == "rmsnorm-worker"
            else ["--mem=64G", "--time=00:20:00"]
            if mode in ("source-length-worker", "causal-worker", "ownership-worker")
            else ["--mem=64G", "--time=00:30:00"]
            if mode == "bytes-worker"
            else ["--mem=64G", "--time=02:00:00"]
        )
    else:
        command += ["--partition=bii", "--cpus-per-task=12", "--mem=16G", "--time=00:30:00"]
    if dependencies:
        command += ["--dependency=afterany:" + ":".join(identity(d) for d in dependencies)]
    script = (
        "evicastkv_review_closure_v2" if tests else launcher or "evicastkv_review_v8_diagnostics"
    )
    command += [
        str(SOURCE / f"mosaickv/slurm/{script}.sbatch"),
        str(SOURCE),
        str(root / "full-cpu") if tests else str(root),
        "tests" if tests else mode,
    ]
    (root / "slurm").mkdir(parents=True, exist_ok=True)
    write_once(root / f"jobs/{label}-request.json", dict(command=command, single_attempt=True))
    try:
        job = identity(subprocess.check_output(command, text=True, stderr=subprocess.PIPE))
    except Exception as error:
        # A lost scheduler acknowledgement must not cause a blind duplicate
        # submission. Keep the complete request and any concrete earlier IDs.
        write_once(
            root / f"jobs/{label}-submission-error.json",
            dict(
                error=f"{type(error).__name__}: {error}",
                stderr=str(getattr(error, "stderr", "")),
                scheduler_outcome="failed_or_unacknowledged",
                automatic_retry=False,
            ),
        )
        raise SubmissionError(f"{label}: failed or unacknowledged scheduler request") from error
    write_once(root / f"jobs/{label}-submission.json", dict(job=job, command=command))
    return job


def register(root, *, execute=False):
    root = Path(root)
    root.resolve().relative_to(Path("/scratch/djy8hg/audits").resolve())
    if root.exists():
        raise FileExistsError("a new write-once diagnostic root is required")
    revision = clean_source(SOURCE)
    prior = read(V7 / "registration.json")
    clean_source(Path(prior["source_root"]), prior["source_sha"])
    paths = [
        V7 / "registration.json",
        V7 / "shape/GATE_shape_transport.json",
        V7 / "common-bridge/analysis/GATE_common_bridge.json",
        V7 / "ownership/analysis/GATE_ownership_closure.json",
        FRESHNESS / "FINAL_report.json",
        SOURCE / "mosaickv/scripts/evicastkv_review_v8_diagnostics.py",
    ]
    paths += [SOURCE / f"mosaickv/scripts/{name}.py" for name in MODULES.values()]
    lock = dict(
        source_sha=revision,
        source_root=str(SOURCE),
        execute=execute,
        bindings={str(p): sha(p) for p in paths},
        maximum_a100s=6,
        six_a100s_explicitly_authorized=True,
        sequential_gpu_workers=1,
        no_new_cohort=True,
        no_new_protection_policy=True,
        production_runtime_changed=False,
        all_registered_model_guards_unchanged=True,
        fresh_confirmation_authorized=False,
        configuration_freeze_authorized=False,
        consumed_source_diagnosis_not_algorithm_tuning=True,
    )
    write_once(root / "registration.json", lock)
    if not execute:
        return dict(status="dry_run", submitted_jobs=0)
    try:
        cpu = submit(root, "cpu", "tests", tests=True)
        return dict(cpu=cpu, preparation=submit(root, "prepare", "prepare", [cpu]))
    except Exception as error:
        return submission_block(root, "registration", error)


def load(root):
    lock = read(root / "registration.json")
    clean_source(SOURCE, lock["source_sha"])
    if lock["execute"] is not True:
        raise ValueError("a dry registration cannot execute")
    for path, expected in lock["bindings"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"immutable diagnostic binding changed: {path}")
    return lock


def module(lane):
    return importlib.import_module(MODULES[lane])


def failure(root, lane, error):
    record = dict(
        status="blocked", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc()
    )
    write_once(root / f"errors/{lane}.json", record)
    return record


def submission_block(root, lane, error):
    failure(root, f"submission-{lane}", error)
    record = dict(
        status="blocked_submission",
        automatic_retry=False,
        # Keep successfully acknowledged producers visible for recovery.
        submitted={str(p): read(p) for p in sorted((root / "jobs").glob("*-submission.json"))},
        unresolved_requests=[
            str(p) for p in sorted((root / "jobs").glob("*-submission-error.json"))
        ],
        fresh_confirmation_authorized=False,
        superiority_established=False,
    )
    write_once(root / f"decisions/{lane}-submission-blocked.json", record)
    return record


def prepare(root):
    lock = load(root)
    try:
        write_once(root / "GATE_cpu_suite.json", cpu_gate(root, lock["source_sha"]))
    except Exception as error:
        failure(root, "cpu", error)
        return finish(root)
    prepared = {}
    for lane, parent in (("rmsnorm", V7 / "shape"), ("protection", V7 / "source-prefix")):
        try:
            prepared[lane] = dict(
                status="prepared", result=module(lane).prepare(root / lane, parent)
            )
        except Exception as error:
            prepared[lane] = failure(root, f"prepare-{lane}", error)
    try:
        prepared["freshness"] = module("freshness").register(
            root / "freshness-continuation",
            parent=FRESHNESS,
            execute=True,
            cpu_gate_root=root,
        )
    except Exception as error:
        prepared["freshness"] = failure(root, "freshness-registration", error)
    write_once(root / "preparation.json", prepared)
    if prepared["rmsnorm"]["status"] == "prepared":
        job = submit_gpu(root, "rmsnorm", "rmsnorm-worker")
        return dict(rmsnorm=job, analysis=submit(root, "rmsnorm-analysis", "after-rmsnorm", [job]))
    return launch_protection(root)


def submit_gpu(root, label, mode):
    snapshot = gpu_snapshot()
    dependencies = sorted(
        {item["job"].split("_")[0] for item in snapshot["active"] + snapshot["pending"]}
    )
    write_once(
        root / f"admission/{label}.json",
        dict(
            snapshot=snapshot,
            wait_for_existing_own_gpu_jobs=dependencies,
            maximum_a100s=6,
            gpu_workers=1,
            no_other_users_jobs_changed=True,
        ),
    )
    return submit(root, label, mode, dependencies, gpu=True)


def analyze(root, lane):
    try:
        module(lane).analyze(root / lane)
        return read(root / GATES[lane])
    except Exception as error:
        return failure(root, f"analyze-{lane}", error)


def launch_protection(root):
    # Independent BF16/source scope: a Mistral operator result cannot qualify
    # or invalidate its already-passed native BF16 FullKV transport controls.
    prepared = read(root / "preparation.json")
    if prepared["protection"]["status"] == "prepared":
        job = submit_gpu(root, "protection", "protection-worker")
        return dict(
            protection=job, analysis=submit(root, "protection-analysis", "after-protection", [job])
        )
    return submit_final(root)


def submit_final(root):
    fresh = read(root / "preparation.json")["freshness"]
    # A failed analyzer acknowledgement is never retried. Wait for the known
    # workers instead, and perform only the already-registered CPU aggregation.
    dependency = fresh.get("analysis") or fresh.get("workers")
    dependencies = [dependency] if dependency else []
    return submit(root, "final", "finish", dependencies)


def finish(root):
    lock = load(root)
    fresh_root = root / "freshness-continuation"
    prep = root / "preparation.json"
    if prep.exists():
        fresh = read(prep).get("freshness", {})
        if (
            fresh.get("workers")
            and fresh.get("analysis_definitely_not_submitted") is True
            and not (fresh_root / "FINAL_report.json").exists()
        ):
            try:
                module("freshness").aggregate(fresh_root)
            except Exception as error:
                failure(root, "fallback-freshness-aggregate", error)
    paths = [root / path for path in GATES.values()]
    paths += [root / "freshness-continuation/FINAL_report.json"]
    errors = {str(p): read(p) for p in sorted((root / "errors").glob("*.json"))}
    missing = [str(p) for p in paths if not p.exists()]
    result = dict(
        source_sha=lock["source_sha"],
        status="completed_with_blocked_controls"
        if errors or missing
        else "completed_bounded_diagnostics_not_confirmation",
        artifacts={
            str(p): dict(sha256=sha(p), value=read(p))
            if p.exists()
            else dict(status="missing_or_blocked")
            for p in paths
        },
        errors=errors,
        missing_artifacts=missing,
        original_negative_controls_preserved=True,
        new_protection_policies=0,
        fresh_confirmation_authorized=False,
        configuration_freeze_authorized=False,
        superiority_established=False,
        no_automatic_full_model_shape_transport=True,
    )
    write_once(root / "FINAL_report.json", result)
    return result


def _run(root, mode):
    load(root)
    if mode == "prepare":
        return prepare(root)
    if mode in ("rmsnorm-worker", "protection-worker"):
        if gpu_snapshot()["active_gpus"] > 6:
            raise ValueError("aggregate six-A100 ceiling exceeded; no model work admitted")
        return (
            module("rmsnorm").worker(root / "rmsnorm")
            if mode == "rmsnorm-worker"
            else module("protection").worker(root / "protection", index=0)
        )
    if mode == "after-rmsnorm":
        analyze(root, "rmsnorm")
        return launch_protection(root)
    if mode == "after-protection":
        analyze(root, "protection")
        return submit_final(root)
    if mode == "finish":
        return finish(root)
    raise ValueError("unregistered diagnostic stage")


def run(root, mode):
    try:
        return _run(root, mode)
    except SubmissionError as error:
        return submission_block(root, mode, error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "register",
            "prepare",
            "rmsnorm-worker",
            "after-rmsnorm",
            "protection-worker",
            "after-protection",
            "finish",
        ),
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    result = (
        register(args.root, execute=args.execute)
        if args.mode == "register"
        else run(args.root, args.mode)
    )
    print(json.dumps(result, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
