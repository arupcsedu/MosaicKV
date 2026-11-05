#!/usr/bin/env python3
"""Numerical feasibility BEFORE any fused incoming-token runtime rewrite."""

import subprocess
from pathlib import Path

import evicastkv_lt_model_profile as controller

from mosaickv.evaluation import joint_row_feasibility as probe
from mosaickv.evaluation import lt_model_profile
from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once
from mosaickv.evaluation.reservation_window import actual

SOURCE, PARENT, driver = controller.SOURCE, controller.PARENT, controller.driver
WAIT = Path("/scratch/djy8hg/runs/evicastkv/static-transport-consumed-20261105-v1")
PROFILE = Path("/scratch/djy8hg/runs/evicastkv/optimized-runtime-profile-consumed-20261105-v1")
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_joint_row_feasibility.sbatch"
ORIGINAL_EXECUTE, ORIGINAL_SUBMIT = controller.common.execute, controller.submit
DOC = SOURCE / "mosaickv/docs/joint_row_feasibility.md"
STAGE = "consumed_joint_row_feasibility"
VARIANT = "consumed_joint_row_operator_feasibility"
EXTRA_PLAN = {}


def execute(adapter, refs, messages, profile, plan, method, record, **kwargs):
    state = probe.current()
    if state is not None:
        kwargs["final_validator"] = state.capture
    return ORIGINAL_EXECUTE(
        adapter,
        refs,
        messages,
        profile,
        plan,
        method,
        dict(record, graph_accounting="event_sum", reuse_patch_dispatch=True),
        **kwargs,
    )


def submit(root, action, dependency=None):
    if action != "profile":
        return ORIGINAL_SUBMIT(root, action, dependency)
    plan = driver.checked(root, sha(root / "plan.json"))
    driver.cpu_proof(root, plan)
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
        "--job-name=mk-joint-row",
        f"--output={root}/slurm/profile-%A_%a.out",
        f"--error={root}/slurm/profile-%A_%a.err",
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
        raise ValueError("new immutable root and clean source required")
    previous = driver.read(WAIT / "plan.json")
    submission = driver.read(WAIT / "cpu_submission.json")
    cpu_dependency = str(submission.get("analysis", submission.get("release", "")))
    if not cpu_dependency.isdecimal():
        raise ValueError("registered preceding CPU release job required")
    old = driver.read(PARENT / "plan.json")
    paths = [
        WAIT / "plan.json",
        WAIT / "cpu_submission.json",
        PROFILE / "analysis/decision.json",
        DOC,
    ]
    bound = dict(old["bound_files"])
    bound.update({str(path): sha(path) for path in paths})
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=revision,
            bound_files=bound,
            stage=STAGE,
            independent_quality_N=0,
            no_new_identities=True,
            automatic_quality_submission=False,
            automatic_full_request_submission=False,
            automatic_formal_submission=False,
            automatic_bounded_followup=False,
            automatic_bounded_cohort_after_correctness=False,
            frozen_operator_windows=list(probe.WINDOWS),
            only_layer=0,
            original_forward_operators_NEVER_replaced=True,
            any_changed_word_rejects_same_parent_integration=True,
            predecessor_source_sha=previous["source_sha"],
            predecessor_plan_sha256=sha(WAIT / "plan.json"),
            no_overlapping_GPU_screens=True,
            **EXTRA_PLAN,
        ),
    )
    cpu = submit(root, "cpu")
    release_job = submit(root, "release", f"{cpu}:{cpu_dependency}")
    write_once(root / "cpu_submission.json", dict(cpu=cpu, release=release_job))


def release(root, expected):
    plan = driver.checked(root, expected)
    try:
        driver.cpu_proof(root, plan)
        path = WAIT / "analysis/decision.json"
        if not path.exists():
            # One event-driven deferral, never a sleep/polling loop.
            if (root / "deferred_release.json").exists():
                raise ValueError("one-shot predecessor deferral exhausted")
            dependency = str(driver.read(WAIT / "screen_submission.json")["analysis"])
            if not dependency.isdecimal():
                raise ValueError("registered predecessor GPU analysis job required")
            job = submit(root, "release", dependency)
            write_once(root / "deferred_release.json", dict(release=job, afterany=dependency))
            return
        decision = driver.read(path)
        if (
            not decision["complete"]
            or decision["failures"]
            or decision["source_sha"] != plan["predecessor_source_sha"]
            or decision["plan_sha256"] != plan["predecessor_plan_sha256"]
        ):
            raise ValueError("complete source/plan-bound preceding diagnostic required")
    except Exception as error:
        write_once(
            root / "release.json",
            dict(
                profile_released=False,
                no_GPU_submitted=True,
                failures=[f"{type(error).__name__}: {error}"],
                source_sha=plan["source_sha"],
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
    gpu = None
    try:
        gpu = submit(root, "profile")
        analysis = submit(root, "analysis", gpu)
        write_once(root / "profile_submission.json", dict(worker=gpu, analysis=analysis))
    except Exception as error:
        write_once(
            root / "submission_failure.json",
            dict(
                submitted_GPU_job=gpu,
                source_sha=plan["source_sha"],
                error=f"{type(error).__name__}: {error}",
            ),
        )
        raise


def analyze(root, expected):
    plan = driver.checked(root, expected)
    rows, failures = driver.load_producers(root, plan, "profile.json")
    flags = [r["records"][0]["exact_feasibility_by_backend"] for r in rows]
    write_once(
        root / "analysis/decision.json",
        dict(
            complete=not failures,
            failures=failures,
            rows=rows,
            source_sha=plan["source_sha"],
            plan_sha256=expected,
            exact_feasible_both_inputs={
                name: not failures and len(flags) == 2 and all(flag[name] for flag in flags)
                for name in ("torch_joint", "Lt_joint")
            },
            original_model_path_unchanged=True,
            no_runtime_integrated=True,
            no_latency_or_quality_acceptance=True,
            independent_quality_N=0,
            no_automatic_followup=True,
        ),
    )


def main():
    controller.CPU_NODE = "udc-aj36-2c0"
    controller.MANIFEST_METHODS = ("mosaickv_exact",)
    controller.MANIFEST_VARIANT = VARIANT
    controller.MANIFEST_TIMING_PROTOCOL = dict(
        output_predictions=128,
        operator_windows=list(probe.WINDOWS),
        only_layer=0,
        actual_model_forward_unchanged=True,
        not_latency_evidence=True,
    )
    controller.SCRIPT, controller.ARMS = SCRIPT, (("mosaickv_exact", "graph"),)
    controller.prepare, controller.submit, controller.release = prepare, submit, release
    controller.analyze, controller.common.execute = analyze, execute
    lt_model_profile.collect = probe.collect
    controller.main()


if __name__ == "__main__":
    main()
