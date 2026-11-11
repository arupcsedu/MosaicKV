#!/usr/bin/env python3
"""Finite history/storage audits and one corrective ownership-archive retry."""

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
RECOVERY = Path("/scratch/djy8hg/workdir/fox-review-v15/run-20261111-1/history")
FAILED_OWNERSHIP = RECOVERY.parent / "ownership"
FAILED_SHA = "279c4f6336bd852d2ccd8408d519e1ddfa2a53b1"
HISTORY_GATE = "GATE_history_closure_v16.json"
STORAGE_GATE = "GATE_storage_boundary_reconciliation.json"
OWNERSHIP_GATE = "GATE_streaming_lossless_ownership.json"


def read(path):
    return json.loads(Path(path).read_text())


def module(name):
    return importlib.import_module("evicastkv_" + name)


def retry_evidence(parent=FAILED_OWNERSHIP):
    """Exactly two completed requests lost only their bounded metadata archive."""
    ownership = module("streaming_lossless_ownership")
    parent = Path(parent)
    plan = read(parent / "protocol.json")
    if plan["source_sha"] != FAILED_SHA:
        raise ValueError("Exact immutable V15 ownership failure required")
    paths = [parent / "protocol.json", parent / OWNERSHIP_GATE]
    for index in (0, 1):
        out = parent / f"workers/{index}"
        result, raw = read(out / "result.json"), read(out / "raw.json")
        if (
            result["source_sha"] != FAILED_SHA
            or result["status"] != "failed"
            or result.get("all_32_128_guards_pass") is not True
            or result["failures"]
            != [dict(archive_error="ValueError: bounded semantic expansion exceeded")]
            or result["index"] != index
            or raw["source_sha"] != FAILED_SHA
            or raw["index"] != index
        ):
            raise ValueError("Only the observed archive-resource failure permits this retry")
        ownership.verify_request(raw["record"], raw["counters"])
        for name in ("registered_guard_32", "cache_dependent_guard_32", "prefix_64", "prefix_128"):
            if raw["diagnostics"][name]["registered_guard_pass"] is not True:
                raise ValueError("Numerical failure cannot be relabeled as an archival retry")
        paths.append(out / "result.json")
        for name, expected in result["outputs"].items():
            if Path(name).name != name or sha(out / name) != expected:
                raise ValueError("Failed producer output hash changed")
            paths.append(out / name)
    return dict(
        parent=str(parent),
        source_sha=FAILED_SHA,
        numerical_failure_retried=False,
        bindings={str(path): sha(path) for path in paths},
    )


def load(root):
    plan = read(root / "registration.json")
    clean_source(SOURCE, plan["source_sha"])
    if (
        not plan["execute"]
        or not plan["metadata_lanes_CPU_only"]
        or not plan["bounded_corrective_ownership_GPU_only"]
        or plan["fresh_confirmation_authorized"]
        or plan["maximum_a100s"] != 6
    ):
        raise ValueError("Only finite metadata and corrective ownership scope is executable")
    if any(sha(Path(path)) != expected for path, expected in plan["bindings"].items()):
        raise ValueError("Immutable V16 evidence changed")
    return plan


def command(root, label, mode, *, tests=False, gpu=False):
    if gpu and (tests or mode != "ownership-worker"):
        raise ValueError("Only the two corrective ownership workers may request GPUs")
    launcher = "evicastkv_review_closure_v2" if tests else "evicastkv_review_v16"
    return [
        "sbatch",
        "--parsable",
        "--account=bii_dsc_community",
        "--nodes=1",
        "--ntasks=1",
        "--partition=bii-gpu" if gpu else "--partition=bii,standard",
        f"--job-name=fox16-{label}",
        "--cpus-per-task=12" if tests or gpu or mode == "after-ownership" else "--cpus-per-task=4",
        "--mem=64G" if gpu else "--mem=32G" if mode == "after-ownership" else "--mem=16G",
        "--time=00:20:00",
        f"--output={root}/slurm/{label}-%j.out",
        f"--error={root}/slurm/{label}-%j.err",
        *(["--reservation=bi_fox_dgx", "--gres=gpu:a100:1", "--array=0-1%2"] if gpu else []),
        str(SOURCE / f"mosaickv/slurm/{launcher}.sbatch"),
        str(SOURCE),
        str(root / "full-cpu") if tests else str(root),
        "tests" if tests else mode,
    ]


def submit(root, label, mode, *, dependencies=(), tests=False, gpu=False):
    request = root / f"jobs/{label}-request.json"
    if request.exists():
        raise FileExistsError("Do not blindly repeat a scheduler request")
    cmd = command(root, label, mode, tests=tests, gpu=gpu)
    if dependencies:
        cmd.insert(2, "--dependency=afterany:" + ":".join(shared.identity(d) for d in dependencies))
    (root / "slurm").mkdir(parents=True, exist_ok=True)
    write_once(request, dict(command=cmd, automatic_retry=False))
    try:
        job = shared.identity(
            subprocess.check_output(cmd, text=True, stderr=subprocess.PIPE, timeout=30)
        )
    except Exception as error:
        write_once(
            root / f"jobs/{label}-submission-error.json",
            dict(error=f"{type(error).__name__}: {error}", automatic_retry=False),
        )
        raise shared.SubmissionError("V16 scheduler acknowledgement missing") from error
    write_once(
        root / f"jobs/{label}-submission.json",
        dict(command=cmd, job=job, acknowledged_at_unix_seconds=time.time()),
    )
    return job


def register(root, *, execute=False):
    root = Path(root).resolve()
    allowed = Path("/scratch/djy8hg/workdir").resolve()
    if root.exists() or root == allowed or not root.is_relative_to(allowed):
        raise ValueError("New immutable scratch metadata root required")
    revision = clean_source(SOURCE)
    retry = retry_evidence()
    paths = [
        Path(__file__),
        SOURCE / "mosaickv/env/common/requirements.lock",
        SOURCE / "mosaickv/scripts/evicastkv_storage_boundary_reconciliation.py",
        SOURCE / "mosaickv/scripts/evicastkv_history_closure_v16.py",
        SOURCE / "mosaickv/scripts/evicastkv_streaming_lossless_ownership.py",
        SOURCE / "mosaickv/scripts/evicastkv_source_trace_codec.py",
        RECOVERY / "GATE_history_segment_recovery.json",
        RECOVERY / "plan.json",
    ]
    write_once(
        root / "registration.json",
        dict(
            source_sha=revision,
            source_root=str(SOURCE),
            execute=execute,
            metadata_lanes_CPU_only=True,
            bounded_corrective_ownership_GPU_only=True,
            maximum_a100s=6,
            six_a100s_explicitly_authorized=True,
            bindings={**{str(path): sha(path) for path in paths}, **retry["bindings"]},
            ownership_retry_evidence=retry,
            maximum_history_segments=4,
            original_errors_preserved=True,
            fresh_confirmation_authorized=False,
            materialization_authorized=False,
            GPU_execution_authorized=True,
            paper_campaign_authorized=False,
        ),
    )
    if not execute:
        return dict(status="disabled", jobs=0)
    cpu = submit(root, "cpu", "tests", tests=True)
    return dict(cpu=cpu, prepare=submit(root, "prepare", "prepare", dependencies=[cpu]))


def prepare(root):
    plan = load(root)
    try:
        write_once(root / "GATE_cpu_suite.json", shared.cpu_gate(root, plan["source_sha"]))
    except Exception as error:
        shared.failure(root, "cpu", error)
        return finish(root)
    try:
        ownership = module("streaming_lossless_ownership")
        ownership.prepare(root / "ownership", cpu_gate_root=root)
        if (
            read(root / "ownership/protocol.json")["jobs"]
            != read(FAILED_OWNERSHIP / "protocol.json")["jobs"]
        ):
            raise ValueError(
                "Corrective retry must preserve both exact consumed inputs/configurations"
            )
        snapshot = shared.gpu_snapshot()
        if snapshot["active_gpus"] + 2 > 6 or snapshot["pending"]:
            raise ValueError("Two corrective workers require room under the aggregate six-A100 cap")
        write_once(
            root / "GPU-admission.json",
            dict(snapshot=snapshot, maximum_a100s=6, corrective_workers=2),
        )
        producer = submit(root, "ownership", "ownership-worker", gpu=True)
        submit(root, "after-ownership", "after-ownership", dependencies=[producer, *current()])
    except Exception as error:
        shared.failure(root, "ownership", error)
        write_once(root / "terminal/ownership.json", dict(error=f"{type(error).__name__}: {error}"))
    try:
        module("storage_boundary_reconciliation").run(root / "storage", cpu_gate_root=root)
    except Exception as error:
        shared.failure(root, "storage", error)
    try:
        result = module("history_closure_v16").register(
            root / "history",
            cpu_gate_root=root,
            parent_recovery=RECOVERY,
            execute=True,
            completion_argv=command(root, "after-history", "after-history"),
        )
        write_once(root / "history-registration.json", result)
        write_once(root / "preparation-complete.json", dict(history_registered=True))
        return dict(history=result, final_barrier=barrier(root))
    except Exception as error:
        shared.failure(root, "history-registration", error)
        write_once(root / "terminal/history.json", dict(error=f"{type(error).__name__}: {error}"))
        write_once(root / "preparation-complete.json", dict(history_registered=False))
        return barrier(root)


def current():
    return [shared.identity(os.environ["SLURM_JOB_ID"])] if os.environ.get("SLURM_JOB_ID") else []


def barrier(root):
    if not (root / "preparation-complete.json").is_file() or not all(
        (root / f"terminal/{lane}.json").is_file() for lane in ("ownership", "history")
    ):
        return dict(status="waiting_for_finite_callbacks_no_polling")
    try:
        write_once(root / "final-claim.json", dict(callback=current()))
    except FileExistsError:
        return dict(status="final_already_claimed")
    candidates, fresh = set(current()), set(current())
    now = time.time()
    for path in sorted((root / "jobs").glob("*-submission.json")):
        if path.name in {"cpu-submission.json", "prepare-submission.json"}:
            continue
        saved = read(path)
        candidates.add(saved["job"])
        age = now - saved.get("acknowledged_at_unix_seconds", float("-inf"))
        if 0 <= age < 240:
            fresh.add(saved["job"])
    aged = sorted(candidates - fresh)
    retained, receipt = ([], None)
    if aged:
        retained, receipt = module("completion_v14").resolve_dependencies(aged)
    write_once(
        root / "final-dependency-resolution.json",
        dict(aged=receipt, fresh=sorted(fresh), candidates=sorted(candidates)),
    )
    return dict(final=submit(root, "final", "finish", dependencies=sorted(set(retained) | fresh)))


def after(root, lane):
    load(root)
    gate = root / lane / (OWNERSHIP_GATE if lane == "ownership" else HISTORY_GATE)
    try:
        if lane == "ownership":
            module("streaming_lossless_ownership").analyze(root / lane)
        read(gate)
        write_once(root / f"terminal/{lane}.json", dict(gate=str(gate), sha256=sha(gate)))
    except Exception as error:
        shared.failure(root, "after-" + lane, error)
        write_once(root / f"terminal/{lane}.json", dict(error=f"{type(error).__name__}: {error}"))
    return barrier(root)


def finish(root):
    plan = load(root)
    evidence = {}
    for lane, name in (
        ("storage", STORAGE_GATE),
        ("history", HISTORY_GATE),
        ("ownership", OWNERSHIP_GATE),
    ):
        path = root / lane / name
        if path.is_file():
            evidence[lane] = dict(path=str(path), sha256=sha(path), gate=read(path))
        else:
            evidence[lane] = dict(status="missing_gate_not_scientific_success", path=str(path))
    result = dict(
        status="finite_corrective_followup_not_paper_confirmation",
        source_sha=plan["source_sha"],
        evidence=evidence,
        errors={path.name: read(path) for path in sorted((root / "errors").glob("*.json"))},
        maximum_corrective_GPU_requests=2,
        acknowledged_corrective_GPU_requests=2
        if (root / "jobs/ownership-submission.json").is_file()
        else 0,
        independent_quality_N=0,
        fresh_confirmation_established=False,
        byte_matched_superiority_established=False,
        whole_request_KV_peak_certified=False,
        six_component_benefits_established=False,
        paper_campaign_authorized=False,
    )
    write_once(root / "FINAL_report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "register",
            "prepare",
            "finish",
            "ownership-worker",
            "after-ownership",
            "after-history",
        ),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.mode == "register":
        result = register(args.root, execute=args.execute)
    elif args.mode == "prepare":
        result = prepare(args.root)
    elif args.mode == "finish":
        result = finish(args.root)
    elif args.mode.startswith("after-"):
        result = after(args.root, args.mode.removeprefix("after-"))
    else:
        load(args.root)
        if shared.gpu_snapshot()["active_gpus"] > 6:
            raise ValueError("Aggregate six-A100 ceiling exceeded")
        result = module("streaming_lossless_ownership").worker(
            args.root / "ownership", int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
        )
    print(json.dumps(result, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
