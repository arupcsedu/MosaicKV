#!/usr/bin/env python3
"""Finite clean-source gate callbacks; no polling or new research cohorts."""

from __future__ import annotations

import argparse
import json
import subprocess
import traceback
from pathlib import Path

import evicastkv_coco_boundary_v28 as numerical
import evicastkv_review_cpu_continuation_v30 as provenance
import evicastkv_whole_peak_adjudicate_v27 as whole
from evicastkv_review_followups import cpu_gate

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
SCRATCH = Path("/scratch/djy8hg/workdir")
DISPATCH_JOB = "22832447"
POLICY = dict(
    id="V32_finite_whole_COCO_provenance_callbacks_v1",
    actual_producer_sha=whole.PRODUCER,
    actual_parent_plan_sha256=whole.PARENT_PLAN_SHA,
    actual_parent_dispatch_job=DISPATCH_JOB,
    CPU_suite_timeout_seconds=840,
    fresh_confirmation_authorized=False,
    new_identity_selection_authorized=False,
    materialization_authorized=False,
    paper_campaign_authorized=False,
    direct_GPU_jobs=0,
    automatic_retry=False,
    continuous_polling=False,
)


def read(path):
    return json.loads(Path(path).read_text())


def acknowledgement(path, producer, mode, *, job=None):
    receipt = read(path)
    actual = receipt.get("job")
    command = receipt.get("command")
    if (
        receipt.get("source_sha") != producer
        or not isinstance(actual, str)
        or not actual.isdigit()
        or (job is not None and actual != job)
        or not isinstance(command, list)
        or len(command) < 3
        or command[-2:] != [mode, producer]
    ):
        raise ValueError("actual acknowledged producer/mode/job differs")
    return actual


def original_gate_ready():
    path = whole.PARENT / f"analysis/{numerical.BRIDGE_GATE_NAME}"
    if not path.exists():
        return False
    plan = read(whole.PARENT / "plan.json")
    if sha(whole.PARENT / "plan.json") != whole.PARENT_PLAN_SHA:
        raise ValueError("actual original plan binding changed")
    whole.validate_parent(plan, read(path))
    for name, expected in plan["bindings"].items():
        if sha(Path(name)) != expected:
            raise ValueError("original input binding changed")
    for index in range(4):
        out = whole.PARENT / f"workers/{index}"
        record = read(out / "result.json")
        if (
            record.get("source_sha") != whole.PRODUCER
            or record.get("plan_sha256") != whole.PARENT_PLAN_SHA
            or record.get("failures") != []
        ):
            raise ValueError("original worker incomplete or source differs")
        for name, expected in record["outputs"].items():
            path = (out / name).resolve(strict=True)
            if not path.is_relative_to(out.resolve()) or sha(path) != expected:
                raise ValueError("actual original GPU output changed")
    return True


def active_job(job):
    """One event-time query, not a monitoring loop; terminal/tombstoned is false."""
    if not isinstance(job, str) or not job.isdigit():
        raise ValueError("numeric acknowledged job required")
    result = subprocess.run(
        ["scontrol", "show", "job", job, "-o"],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if result.returncode:
        if "Invalid job id" in result.stderr or "Invalid job id" in result.stdout:
            return False
        raise ValueError(f"scheduler state unavailable: {result.stderr.strip()}")
    return any(
        f"JobState={state}" in result.stdout
        for state in ("PENDING", "RUNNING", "COMPLETING", "CONFIGURING", "SUSPENDED")
    )


def submit(root, revision, mode, *, dependencies=()):
    root = Path(root)
    heavy = mode == "after-whole"
    if any(not isinstance(job, str) or not job.isdigit() for job in dependencies):
        raise ValueError("only acknowledged numeric dependencies allowed")
    command = [
        "sbatch",
        "--parsable",
        "--account=bii_dsc_community",
        "--nodes=1",
        "--ntasks=1",
        "--partition=interactive",
        f"--job-name=fox32-{mode}",
        f"--cpus-per-task={12 if heavy else 4}",
        f"--mem={'24G' if heavy else '8G'}",
        f"--time=00:{20 if heavy else 15 if mode == 'tests' else 5}:00",
        f"--output={root}/slurm/{mode}-%j.out",
        f"--error={root}/slurm/{mode}-%j.err",
    ]
    if dependencies:
        command.append("--dependency=afterany:" + ":".join(dependencies))
    command += [
        str(SOURCE / "mosaickv/slurm/fox_review_stages_v32.sbatch"),
        str(SOURCE),
        str(root),
        mode,
        revision,
    ]
    write_once(root / f"jobs/{mode}-request.json", dict(source_sha=revision, command=command))
    job = subprocess.check_output(command, text=True, timeout=30).strip().split(";")[0]
    if not job.isdigit():
        raise ValueError("uncertain submission acknowledgement; no retry")
    write_once(
        root / f"jobs/{mode}-submission.json", dict(source_sha=revision, command=command, job=job)
    )
    return job


def load(root):
    plan = read(Path(root) / "plan.json")
    clean_source(SOURCE, plan["source_sha"])
    if plan["policy_sha256"] != digest(POLICY) or digest(plan["policy"]) != digest(POLICY):
        raise ValueError("immutable finite-stage policy changed")
    for name, expected in plan["bindings"].items():
        if sha(Path(name)) != expected:
            raise ValueError("immutable stage prerequisite changed")
    return plan


def launch(root):
    revision = clean_source(SOURCE)
    numerical.execution_base_proof(revision)
    plan_path = whole.PARENT / "plan.json"
    dispatch = whole.PARENT / "jobs/dispatch-submission.json"
    if sha(plan_path) != whole.PARENT_PLAN_SHA or read(plan_path)["source_sha"] != whole.PRODUCER:
        raise ValueError("actual predecessor plan changed")
    acknowledgement(dispatch, whole.PRODUCER, "dispatch", job=DISPATCH_JOB)
    root = Path(root).resolve()
    if root == SCRATCH.resolve() or not root.is_relative_to(SCRATCH.resolve()) or root.exists():
        raise ValueError("new immutable scratch stage root required")
    root.mkdir(parents=True, exist_ok=False)
    (root / "jobs").mkdir()
    (root / "slurm").mkdir()
    bindings = {str(path): sha(path) for path in (plan_path, dispatch)}
    dispatch_result = whole.PARENT / "jobs/dispatch-result.json"
    if dispatch_result.exists():
        actual = read(dispatch_result)
        if actual.get("blockers") != [] or any(
            not isinstance(actual.get(field), str) or not actual[field].isdigit()
            for field in ("workers", "analysis")
        ):
            raise ValueError("actual predecessor dispatch has recorded blockers")
        for label, mode, field in (
            ("gpu", "worker", "workers"),
            ("analysis", "analyze", "analysis"),
        ):
            path = whole.PARENT / f"jobs/{label}-submission.json"
            acknowledgement(path, whole.PRODUCER, mode, job=actual.get(field))
            bindings[str(path)] = sha(path)
        bindings[str(dispatch_result)] = sha(dispatch_result)
    write_once(
        root / "plan.json",
        dict(
            source_sha=revision,
            policy=POLICY,
            policy_sha256=digest(POLICY),
            bindings=bindings,
        ),
    )
    try:
        ready = original_gate_ready()
        wait_dispatch = not ready and active_job(DISPATCH_JOB)
        tests = submit(root, revision, "tests")
        dependencies = [tests]
        if wait_dispatch:
            dependencies.append(DISPATCH_JOB)
        callback = submit(root, revision, "after-dispatch", dependencies=tuple(dependencies))
    except Exception as exc:
        write_once(
            root / "decisions/launch.json",
            dict(
                status="blocked_stage",
                stage="launch",
                failures=[
                    dict(
                        error_type=type(exc).__name__,
                        error=str(exc),
                        traceback=traceback.format_exc(),
                    )
                ],
                fresh_confirmation_authorized=False,
                materialization_authorized=False,
                paper_campaign_authorized=False,
            ),
        )
        raise
    write_once(
        root / "launch.json",
        dict(
            source_sha=revision,
            cpu=tests,
            after_dispatch=callback,
            already_completed_verified_whole_gate=ready,
            original_dispatch_job=DISPATCH_JOB,
            GPU_jobs_directly_submitted=0,
            continuous_polling=False,
        ),
    )
    return dict(cpu=tests, after_dispatch=callback)


def after_dispatch(root, plan):
    cpu_gate(root, plan["source_sha"])
    path = whole.PARENT / "jobs/dispatch-result.json"
    result = read(path)
    if result.get("blockers") != [] or any(
        not isinstance(result.get(key), str) or not result[key].isdigit()
        for key in ("workers", "analysis")
    ):
        raise ValueError(f"actual whole-reader dispatch failed: {result}")
    worker = acknowledgement(
        whole.PARENT / "jobs/gpu-submission.json", whole.PRODUCER, "worker", job=result["workers"]
    )
    analysis = acknowledgement(
        whole.PARENT / "jobs/analysis-submission.json",
        whole.PRODUCER,
        "analyze",
        job=result.get("analysis"),
    )
    ready = original_gate_ready()
    dependencies = () if ready or not active_job(analysis) else (analysis,)
    callback = submit(root, plan["source_sha"], "after-whole", dependencies=dependencies)
    return dict(
        workers=worker,
        actual_analysis=analysis,
        after_whole=callback,
        verified_completed_gate=ready,
        dispatch_result_sha256=sha(path),
    )


def after_whole(root, plan):
    cpu_gate(root, plan["source_sha"])
    if not original_gate_ready():
        raise ValueError("actual whole-request gate missing after producer")
    gate = whole.adjudicate(root / "whole-replay")
    if gate.get("failures") != [] or any(
        gate.get(key) is not True
        for key in (
            "complete",
            "source_fidelity_128_pass",
            "whole_request_KV_byte_match_pass",
            "whole_request_KV_peak_certified",
        )
    ):
        raise ValueError("independent whole-request replay did not pass")
    whole_path = root / "whole-replay/GATE_bounded_reader_v27.json"
    numerical.require_whole_gate(read(whole_path), plan["source_sha"])
    numerical.launch(root / "coco", whole_path)
    analysis = acknowledgement(
        root / "coco/jobs/analysis-submission.json", plan["source_sha"], "analyze"
    )
    gate_path = root / "coco/analysis/GATE_coco_boundary_v28.json"
    dependencies = (analysis,) if not gate_path.exists() and active_job(analysis) else ()
    callback = submit(root, plan["source_sha"], "after-coco", dependencies=dependencies)
    return dict(whole_gate_sha256=sha(whole_path), coco_analysis=analysis, after_coco=callback)


def after_coco(root, plan):
    cpu_gate(root, plan["source_sha"])
    whole_path = root / "whole-replay/GATE_bounded_reader_v27.json"
    coco_path = root / "coco/analysis/GATE_coco_boundary_v28.json"
    numerical.validate(root / "coco")
    bindings, _ = provenance.empirical_bindings(whole_path, coco_path, plan["source_sha"])
    result = provenance.launch(root / "provenance", whole_path, coco_path, cpu_gate_root=root)
    return dict(
        provenance_chain=result,
        empirical_binding_count=len(bindings),
        source_sha=plan["source_sha"],
        GPU_jobs_directly_submitted=0,
        fresh_confirmation_authorized=False,
        paper_campaign_authorized=False,
    )


def run(root, mode):
    root = Path(root)
    try:
        plan = load(root)
        write_once(root / f"locks/{mode}.json", dict(source_sha=plan["source_sha"], stage=mode))
        result = globals()[mode.replace("-", "_")](root, plan)
        decision = dict(status="completed_stage", stage=mode, result=result, failures=[])
    except Exception as exc:
        decision = dict(
            status="blocked_stage",
            stage=mode,
            failures=[
                dict(
                    error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc()
                )
            ],
        )
    decision.update(
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
        new_identity_selection_authorized=False,
        paper_campaign_authorized=False,
    )
    write_once(root / f"decisions/{mode}.json", decision)
    return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("launch", "after-dispatch", "after-whole", "after-coco"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = launch(args.root) if args.mode == "launch" else run(args.root, args.mode)
    print(json.dumps(result, sort_keys=True))
    if result.get("status") == "blocked_stage":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
