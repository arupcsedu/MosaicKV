#!/usr/bin/env python3
"""Finite, common BF16 RoPE dispatch screen with no arithmetic reassociation."""

import traceback
from pathlib import Path

import evicastkv_joint_key_latency as control
import evicastkv_joint_kv_transaction as correctness

from mosaickv.evaluation.exact_rotary_dispatch import gpu_controls
from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE, parent, driver = control.SOURCE, control.parent, control.driver
WAIT = Path("/scratch/djy8hg/runs/evicastkv/prepared-joint-replay-consumed-20261105-v1")
SEGMENT = Path("/scratch/djy8hg/runs/evicastkv/post-attention-segment-consumed-20261105-retry1")
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_exact_rotary_dispatch.sbatch"
DOC = SOURCE / "mosaickv/docs/exact_rotary_dispatch.md"
ARMS = (
    ("mosaickv_exact", "joint-kv-prepared"),
    ("mosaickv_exact", "joint-kv-rotary"),
    ("prefixkv_reimpl", "graph-prepared"),
    ("prefixkv_reimpl", "graph-rotary"),
    ("prefixkv_reimpl", "graph-original-prepared"),
    ("prefixkv_reimpl", "graph-original-rotary"),
)
USE_POST_SEGMENT = False
ORIGINAL_EXECUTE, ORIGINAL_SUMMARY, ORIGINAL_WORKER = (
    control.execute,
    control.summary,
    control.worker,
)


def segment_policy(decision):
    if not decision["complete"] or decision["failures"] or len(decision["rows"]) != 2:
        return False
    for row in decision["rows"]:
        if not row.get("exact_common_workspace_charge"):
            return False
        means = {(x["method"], x["arm"]): x["total_wall_seconds"] for x in row["means"]}
        for method, old, new in (
            ("mosaickv_exact", "joint-kv-prepared", "joint-kv-segment"),
            ("prefixkv_reimpl", "graph-prepared", "graph-segment"),
            ("prefixkv_reimpl", "graph-original-prepared", "graph-original-segment"),
        ):
            if means[method, new] > means[method, old]:
                return False
    return True


def prepare(root, revision):
    if root.exists() or clean_source(SOURCE) != revision:
        raise ValueError("new immutable output root and clean source required")
    previous, gate = driver.read(WAIT / "plan.json"), driver.read(WAIT / "analysis/decision.json")
    if not gate["complete"] or gate["failures"]:
        raise ValueError("complete prepared-replay controls required")
    segment = driver.read(SEGMENT / "analysis/decision.json")
    enabled = segment_policy(segment)
    old = driver.read(parent.PARENT / "plan.json")
    bound = dict(old["bound_files"])
    for path in (
        WAIT / "plan.json",
        WAIT / "analysis/decision.json",
        SEGMENT / "plan.json",
        SEGMENT / "analysis/decision.json",
        DOC,
    ):
        bound[str(path)] = sha(path)
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=revision,
            bound_files=bound,
            stage="consumed_exact_rotary_dispatch",
            arms=ARMS,
            warmups=5,
            repetitions=3,
            use_post_attention_segment=enabled,
            common_segment_policy_rule="all_three_own_controls_improve_on_both_consumed_inputs",
            common_rotary_optimization_offered_to_every_compared_baseline=True,
            no_fresh_identities=True,
            independent_quality_N=0,
            original_BF16_rounding_after_each_multiplication=True,
            FMA_disabled=True,
            weights_positions_and_selected_KV_unchanged=True,
            no_new_persistent_GPU_tensor_workspace=True,
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
        joint = method == "mosaickv_exact"
        base = (
            "graph-lt"
            if joint
            else ("graph-original" if arm.startswith("graph-original") else "graph")
        )
        result[method, arm] = dict(
            records[method][base],
            graph_accounting="event_sum",
            reuse_patch_dispatch=True,
            reuse_graph_replay_plan=True,
            post_attention_segment=USE_POST_SEGMENT,
            exact_rotary_dispatch=arm.endswith("-rotary"),
            **(dict(lt_fit_policy="direct", joint_incoming_projection="kv") if joint else {}),
        )
    return result


def execute(adapter, refs, messages, profile, plan, method, arm, record, **kwargs):
    result = ORIGINAL_EXECUTE(adapter, refs, messages, profile, plan, method, arm, record, **kwargs)
    info, enabled = result["exact_rotary_dispatch"], record["exact_rotary_dispatch"]
    if (
        info["enabled"] is not enabled
        or info["calls"] != (4064 if enabled else 0)
        or info["extra_persistent_GPU_bytes"]
        or not info["BF16_intermediate_rounding_preserved"]
        or not info["FMA_disabled"]
        or not info["common_to_every_method"]
    ):
        raise RuntimeError("exact common rotary engagement required")
    return result


def summary(rows, audited):
    for method, old, active in (
        ("mosaickv_exact", "joint-kv-prepared", "joint-kv-rotary"),
        ("prefixkv_reimpl", "graph-prepared", "graph-rotary"),
        ("prefixkv_reimpl", "graph-original-prepared", "graph-original-rotary"),
    ):
        if any(audited[method, old][k] != audited[method, active][k] for k in driver.STORAGE):
            raise ValueError("rotary launch coalescing changed a measured cache boundary")
    return dict(
        ORIGINAL_SUMMARY(rows, audited),
        identical_own_control_initial_final_peak=True,
        whole_request_GPU_peak_certified=False,
        independent_quality_N=0,
    )


def worker(root, expected, index):
    global USE_POST_SEGMENT
    USE_POST_SEGMENT = driver.checked(root, expected)["use_post_attention_segment"]
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
                phase="checkpoint_independent_rotary_control",
            ),
        )
        return
    write_once(root / f"rotary_operator_controls_{index}.json", controls)
    ORIGINAL_WORKER(root, expected, index)


def main():
    correctness.configure()
    parent.WAIT, parent.JOB_NAME = WAIT, "mk-exact-rope"
    control.WAIT, control.SCRIPT = WAIT, SCRIPT
    control.CANDIDATE_ARM, control.POLICY = "joint-kv-rotary", "kv"
    control.VARIANT, control.DOC = "consumed_exact_rotary_dispatch", DOC
    control.ARMS, control.AUXILIARY_PROFILE_ARMS = ARMS, frozenset()
    control.prepare, control.configurations, control.execute, control.summary = (
        prepare,
        configurations,
        execute,
        summary,
    )
    control.worker, control.JOB_NAME = worker, "mk-exact-rope"
    control.main()


if __name__ == "__main__":
    main()
