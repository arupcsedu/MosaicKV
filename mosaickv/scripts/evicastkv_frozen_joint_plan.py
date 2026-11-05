#!/usr/bin/env python3
"""Remove the measured repeated numerical-kernel search, not its checks."""

import traceback
from pathlib import Path

import evicastkv_joint_input_reuse as prior
import evicastkv_joint_key_latency as control
import evicastkv_joint_kv_transaction as correctness

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once
from mosaickv.evaluation.frozen_joint_plan import gpu_controls

SOURCE, parent, driver = control.SOURCE, control.parent, control.driver
WAIT, ROTARY = prior.WAIT, prior.ROTARY
INPUT = Path("/scratch/djy8hg/runs/evicastkv/joint-input-reuse-consumed-20261105-v1")
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_frozen_joint_plan.sbatch"
DOC = SOURCE / "mosaickv/docs/frozen_joint_plan.md"
ARMS = (
    ("mosaickv_exact", "joint-previous"),
    ("mosaickv_exact", "joint-frozen"),
    ("prefixkv_reimpl", "graph"),
    ("prefixkv_reimpl", "graph-original"),
)
COMMON = dict(segment=False, rotary=False, copy=False)
ORIGINAL_SUMMARY, ORIGINAL_WORKER = control.summary, control.worker


def copy_policy(decision):
    if not decision["complete"] or decision["failures"] or len(decision["rows"]) != 2:
        return False
    for row in decision["rows"]:
        if not row.get("own_initial_final_peak_identical"):
            return False
        means = {(x["method"], x["arm"]): x["total_wall_seconds"] for x in row["means"]}
        if means["mosaickv_exact", "joint-input"] > means["mosaickv_exact", "joint-previous"]:
            return False
    return True


def prepare(root, revision):
    if root.exists() or clean_source(SOURCE) != revision:
        raise ValueError("clean immutable source and new output root required")
    previous, gate = driver.read(WAIT / "plan.json"), driver.read(WAIT / "analysis/decision.json")
    if not gate["complete"] or gate["failures"]:
        raise ValueError("complete common epilogue controls required")
    policies = dict(
        segment=prior.rotary.segment_policy(gate),
        rotary=prior.common_rotary_policy(driver.read(ROTARY / "analysis/decision.json")),
        copy=copy_policy(driver.read(INPUT / "analysis/decision.json")),
    )
    old, bound = driver.read(parent.PARENT / "plan.json"), {}
    bound.update(old["bound_files"])
    paths = [DOC]
    for path in (WAIT, ROTARY, INPUT):
        paths.extend((path / "plan.json", path / "analysis/decision.json"))
    bound.update({str(path): sha(path) for path in paths})
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=revision,
            bound_files=bound,
            stage="consumed_frozen_joint_configuration",
            arms=ARMS,
            warmups=5,
            repetitions=3,
            frozen_runtime_policies=policies,
            no_CUDA_handle_tensor_or_request_cache=True,
            previously_verified_GEMM_configuration_unchanged=True,
            full_default_and_joint_word_controls_required=True,
            no_new_GPU_workspace=True,
            independent_quality_N=0,
            no_fresh_identities=True,
            whole_request_GPU_peak_certified=False,
            automatic_quality_submission=False,
            automatic_formal_submission=False,
            automatic_full_request_submission=False,
            automatic_bounded_followup=False,
            automatic_bounded_cohort_after_correctness=False,
            predecessor_source_sha=previous["source_sha"],
            predecessor_plan_sha256=sha(WAIT / "plan.json"),
        ),
    )
    cpu = parent.submit(root, "cpu")
    release = parent.submit(root, "release", cpu)
    write_once(root / "cpu_submission.json", dict(cpu=cpu, release=release))


def configurations(records):
    result = {}
    for method, arm in ARMS:
        active = method == "mosaickv_exact"
        result[method, arm] = dict(
            records[method]["graph-lt" if active else arm],
            graph_accounting="event_sum",
            reuse_patch_dispatch=True,
            reuse_graph_replay_plan=True,
            post_attention_segment=COMMON["segment"],
            exact_rotary_dispatch=COMMON["rotary"],
            reuse_source_input_copy=COMMON["copy"] if active else True,
            frozen_joint_plan=arm == "joint-frozen" or not active,
            **(dict(lt_fit_policy="direct", joint_incoming_projection="kv") if active else {}),
        )
    return result


def execute(adapter, refs, messages, profile, plan, method, arm, record, **kwargs):
    effective = "joint-input" if method == "mosaickv_exact" and COMMON["copy"] else "joint-previous"
    result = prior.execute(
        adapter,
        refs,
        messages,
        profile,
        plan,
        method,
        effective if method == "mosaickv_exact" else arm,
        record,
        **kwargs,
    )
    info = result["joint_incoming_projection"]["frozen_joint_plan"]
    if info != dict(
        requested=record["frozen_joint_plan"], actual_engines=2 if arm == "joint-frozen" else 0
    ):
        raise RuntimeError("exactly the two eligible private constructors must use the frozen plan")
    return result


def summary(rows, audited):
    old, new = (
        audited["mosaickv_exact", "joint-previous"],
        audited["mosaickv_exact", "joint-frozen"],
    )
    if any(old[k] != new[k] for k in driver.STORAGE):
        raise ValueError("CPU plan construction changed a real cache boundary")
    return dict(
        ORIGINAL_SUMMARY(rows, audited),
        own_initial_final_peak_identical=True,
        independent_quality_N=0,
        whole_request_GPU_peak_certified=False,
    )


def worker(root, expected, index):
    COMMON.update(driver.checked(root, expected)["frozen_runtime_policies"])
    try:
        controls = gpu_controls()
    except Exception as error:
        out = root / f"workers/{index}"
        out.mkdir(parents=True, exist_ok=False)
        driver.finish_worker(
            out,
            dict(
                status="failed",
                source_sha=clean_source(SOURCE),
                plan_sha256=expected,
                failures=[f"{type(error).__name__}: {error}"],
                traceback=traceback.format_exc(),
                phase="frozen_GEMM_control",
            ),
        )
        return
    write_once(root / f"frozen_operator_controls_{index}.json", controls)
    ORIGINAL_WORKER(root, expected, index)


def main():
    correctness.configure()
    parent.WAIT, parent.JOB_NAME = WAIT, "mk-frozen-joint"
    control.WAIT, control.SCRIPT = WAIT, SCRIPT
    control.CANDIDATE_ARM, control.POLICY = "joint-frozen", "kv"
    control.VARIANT, control.DOC = "consumed_frozen_joint_configuration", DOC
    control.ARMS = ARMS
    control.AUXILIARY_PROFILE_ARMS = frozenset(
        {("mosaickv_exact", "joint-frozen"), ("prefixkv_reimpl", "graph")}
    )
    control.prepare, control.configurations, control.execute, control.summary = (
        prepare,
        configurations,
        execute,
        summary,
    )
    control.worker, control.JOB_NAME = worker, "mk-frozen-joint"
    control.main()


if __name__ == "__main__":
    main()
