#!/usr/bin/env python3
"""Retry only uncompleted timing, reusing immutable successful model controls."""

import os
import subprocess
from contextvars import ContextVar
from pathlib import Path

import evicastkv_graph_lookahead as parent

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once
from mosaickv.evaluation.reservation_window import actual

SOURCE, controller, base = parent.SOURCE, parent.controller, parent.base
FAILED = Path("/scratch/djy8hg/runs/evicastkv/graph-lookahead-consumed-20261105-v1")
PARENT_SHA = "adef139e9b529ab8e4acdc2bfb4b7e838f275e29"
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_graph_lookahead_retry.sbatch"
ALLOWED = {
    "mosaickv/scripts/evicastkv_graph_lookahead_retry.py",
    "mosaickv/src/mosaickv/evaluation/reservation_window.py",
    "mosaickv/slurm/evicastkv_graph_lookahead_retry.sbatch",
    "mosaickv/docs/graph_lookahead_retry.md",
    "mosaickv/tests/unit/test_graph_lookahead_retry.py",
}
_root = ContextVar("evicastkv_lookahead_retry_root", default=None)


def compatible(revision, *, changed=None):
    if changed is None:
        changed = subprocess.check_output(
            ["git", "diff", "--name-only", PARENT_SHA, revision], cwd=SOURCE, text=True
        ).splitlines()
    if not set(changed) <= ALLOWED:
        raise ValueError("audit reuse forbids all runtime, model, and numerical-guard changes")


def reused(root, index, records):
    plan = controller.driver.read(root / "plan.json")
    for method, arm, candidate in parent.variants(records):
        entry = plan["reused_audits"][str(index)][method + "/" + arm]
        path = Path(entry["path"])
        if entry["producer_sha"] != PARENT_SHA or sha(path) != entry["sha256"]:
            raise ValueError("unchanged original-producer control hash required")
        audited = controller.driver.read(path)
        if (
            not audited["logits_bitwise"]
            or not audited["no_FullKV_fallback"]
            or any(audited[k] != candidate["audited"][k] for k in controller.driver.STORAGE)
            or not audited["prompt_graph_reader"]["lookahead"]["retired"]
            or audited["prompt_graph_reader"]["lookahead"]["hits"] != 3906
        ):
            raise ValueError("only complete positive original model/storage controls reusable")
        records[method][arm] = dict(candidate, audited=audited)
    return []  # Numerical controls were measured, not synthesized/re-executed.


def variants(records):
    return reused(_root.get(), int(os.environ["SLURM_ARRAY_TASK_ID"]), records)


def submit(root, action, *, dependency=None):
    if action != "screen":
        return parent.submit(root, action, dependency=dependency)
    plan = controller.driver.checked(root, sha(root / "plan.json"))
    controller.driver.cpu_proof(root, plan)
    admission = actual()
    args = [
        "sbatch",
        "--parsable",
        "--account=bii_dsc_community",
        "--partition=bii-gpu",
        "--reservation=bi_fox_dgx",
        "--nodes=1",
        "--ntasks=1",
        "--gres=gpu:a100:1",
        "--cpus-per-task=12",
        "--mem=48G",
        "--array=0-1%2",
        "--time=00:20:00",
        "--job-name=mk-lookahead-retry",
        f"--output={root}/slurm/screen-%A_%a.out",
        f"--error={root}/slurm/screen-%A_%a.err",
    ]
    if admission["begin"]:
        args.append("--begin=" + admission["begin"])
    if dependency:
        args.append("--dependency=afterany:" + dependency)
    args += [str(SCRIPT), str(SOURCE), str(root), sha(root / "plan.json"), action]
    job = subprocess.check_output(args, text=True).strip()
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
        root / "scheduler/registration.json", tick(policy, {}, root / "scheduler/state.json", False)
    )
    write_once(root / "scheduler/reservation_window.json", admission)
    return job


def prepare(root, revision):
    if root.exists() or clean_source(SOURCE) != revision:
        raise ValueError("new output root and clean immutable source required")
    compatible(revision)
    old = controller.driver.read(FAILED / "plan.json")
    failed = controller.driver.read(FAILED / "analysis/decision.json")
    if old["source_sha"] != PARENT_SHA or failed["complete"]:
        raise ValueError("exact incomplete reservation-boundary predecessor required")
    # Do not silently reclassify numerical/software failure as reservation loss.
    state = subprocess.check_output(
        ["sacct", "-j", "22498994", "-n", "-X", "-P", "-o", "JobID,State,End"], text=True
    )
    lines = [line.split("|") for line in state.splitlines() if line.strip()]
    if len(lines) != 2 or any(row[1:] != ["TIMEOUT", "2026-11-05T08:00:00"] for row in lines):
        raise ValueError("both prior workers must have the recorded reservation-end timeout")
    bound = dict(old["bound_files"])
    paths = [
        FAILED / "plan.json",
        FAILED / "analysis/decision.json",
        SOURCE / "mosaickv/docs/graph_lookahead_retry.md",
    ]
    reuse = {}
    for index in (0, 1):
        control = FAILED / f"gpu-controls-{index}.json"
        if controller.driver.read(control)["status"] != "completed":
            raise ValueError("positive actual GPU controls required")
        paths.append(control)
        reuse[str(index)] = {}
        for method, arm in parent.ARMS:
            if not arm.endswith("-lookahead"):
                continue
            path = FAILED / f"workers/{index}/audited-{method}-{arm}.json"
            paths.append(path)
            reuse[str(index)][method + "/" + arm] = dict(
                path=str(path),
                sha256=sha(path),
                producer_sha=PARENT_SHA,
                reused_measured_control=True,
                NOT_new_GPU_measurement=True,
            )
    bound.update({str(path): sha(path) for path in paths})
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=revision,
            bound_files=bound,
            stage="consumed_graph_lookahead_timing_only_retry",
            reused_audits=reuse,
            predecessor_sacct=state,
            all_runtime_files_identical_to_audited_parent=True,
            automatic_quality_submission=False,
            automatic_formal_submission=False,
            automatic_bounded_cohort_after_correctness=False,
            automatic_bounded_followup=False,
            automatic_full_request_submission=False,
        ),
    )
    base.common.stage(root, "cpu")


def worker(root, expected, index):
    try:
        actual(at_worker_start=True)
    except Exception as error:
        out = root / f"workers/{index}"
        out.mkdir(parents=True, exist_ok=False)
        controller.driver.finish_worker(
            out,
            dict(
                status="failed",
                failures=[f"reservation admission: {type(error).__name__}: {error}"],
                source_sha=clean_source(SOURCE),
                plan_sha256=expected,
                no_heavy_inference_started=True,
            ),
        )
        return
    token = _root.set(root)
    try:
        parent.worker(root, expected, index)
    finally:
        _root.reset(token)


def main():
    parent.SCRIPT = SCRIPT
    controller.MANIFEST_VARIANT = "consumed_graph_lookahead_timing_only_retry"
    controller.SCRIPT, controller.ARMS, controller.CANDIDATE = SCRIPT, parent.ARMS, parent.CANDIDATE
    controller.prepare, controller.variants = prepare, variants
    controller.execute, controller.worker, controller.summary = (
        parent.execute,
        worker,
        parent.summary,
    )
    base.submit = submit
    controller.main()


if __name__ == "__main__":
    main()
