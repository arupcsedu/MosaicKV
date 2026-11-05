#!/usr/bin/env python3
"""Finite common-operator replay screen; same consumed inputs and guards."""

import traceback
from pathlib import Path

import evicastkv_joint_key_latency as control
import evicastkv_joint_kv_transaction as correctness

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once
from mosaickv.evaluation.post_attention_segment import gpu_controls

SOURCE, parent, driver = control.SOURCE, control.parent, control.driver
WAIT = Path("/scratch/djy8hg/runs/evicastkv/prepared-joint-replay-consumed-20261105-v1")
DOC = SOURCE / "mosaickv/docs/post_attention_segment.md"
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_post_attention_segment.sbatch"
ARMS = (
    ("mosaickv_exact", "joint-kv-prepared"),
    ("mosaickv_exact", "joint-kv-segment"),
    ("prefixkv_reimpl", "graph-prepared"),
    ("prefixkv_reimpl", "graph-segment"),
    ("prefixkv_reimpl", "graph-original-prepared"),
    ("prefixkv_reimpl", "graph-original-segment"),
)
ORIGINAL_EXECUTE, ORIGINAL_SUMMARY, ORIGINAL_WORKER = (
    control.execute,
    control.summary,
    control.worker,
)


def prepare(root, revision):
    if root.exists() or clean_source(SOURCE) != revision:
        raise ValueError("new immutable output root and clean source required")
    preceding = driver.read(WAIT / "plan.json")
    dependency = str(driver.read(WAIT / "profile_submission.json")["analysis"])
    if not dependency.isdecimal():
        raise ValueError("registered predecessor completion-analysis dependency required")
    old = driver.read(parent.PARENT / "plan.json")
    bound = dict(old["bound_files"])
    paths = (WAIT / "plan.json", WAIT / "profile_submission.json", DOC)
    bound.update({str(path): sha(path) for path in paths})
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=revision,
            bound_files=bound,
            stage="consumed_common_post_attention_segment",
            arms=ARMS,
            warmups=5,
            repetitions=3,
            common_model_optimization_offered_to_every_compared_baseline=True,
            independent_quality_N=0,
            no_fresh_identities=True,
            unchanged_original_model_operators=True,
            weights_positions_selected_KV_unchanged=True,
            shared_activation_workspace_explicitly_charged=True,
            backend_internal_workspace_NOT_certified=True,
            whole_request_GPU_peak_certified=False,
            automatic_quality_submission=False,
            automatic_formal_submission=False,
            automatic_full_request_submission=False,
            automatic_bounded_followup=False,
            automatic_bounded_cohort_after_correctness=False,
            predecessor_source_sha=preceding["source_sha"],
            predecessor_plan_sha256=sha(WAIT / "plan.json"),
        ),
    )
    cpu = parent.submit(root, "cpu")
    # CPU feasibility can overlap, but GPU release must await a complete,
    # source/plan-bound predecessor. parent.release verifies its decision.
    release = parent.submit(root, "release", f"{cpu}:{dependency}")
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
            post_attention_segment=arm.endswith("-segment"),
            **(dict(lt_fit_policy="direct", joint_incoming_projection="kv") if joint else {}),
        )
    return result


def execute(adapter, refs, messages, profile, plan, method, arm, record, **kwargs):
    result = ORIGINAL_EXECUTE(adapter, refs, messages, profile, plan, method, arm, record, **kwargs)
    state = result["post_attention_segment"]
    active = record["post_attention_segment"]
    if state["enabled"] is not active:
        raise RuntimeError("common segment flag did not engage")
    if active and (
        state["calls"] != 4064
        or state["captures"] != 32
        or state["replays"] != 4064
        or state["explicit_workspace_bytes"] != 117764
        or state["additional_graph_tensor_bytes"]
        or not state["retired"]
        or not state["common_to_every_method"]
    ):
        raise RuntimeError("exact common segment count, explicit storage and retirement required")
    return result


def summary(rows, audited):
    for method, original, active in (
        ("mosaickv_exact", "joint-kv-prepared", "joint-kv-segment"),
        ("prefixkv_reimpl", "graph-prepared", "graph-segment"),
        ("prefixkv_reimpl", "graph-original-prepared", "graph-original-segment"),
    ):
        before, after = audited[method, original], audited[method, active]
        if any(before[k] != after[k] for k in driver.STORAGE[:2]):
            raise ValueError("epilogue changed initial/final compressed cache storage")
        extra = after["post_attention_segment"]["explicit_workspace_bytes"]
        if after[driver.STORAGE[-1]] != before[driver.STORAGE[-1]] + extra:
            raise ValueError("all common activation workspace must be included in the decode peak")
    return dict(
        ORIGINAL_SUMMARY(rows, audited),
        exact_common_workspace_charge=True,
        whole_request_GPU_peak_certified=False,
        independent_quality_N=0,
    )


def worker(root, expected, index):
    # Cheap, checkpoint-independent CUDA test precedes all model requests.
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
                phase="checkpoint_independent_operator_control",
            ),
        )
        return
    ORIGINAL_WORKER(root, expected, index)
    write_once(root / f"workers/{index}/segment_operator_controls.json", controls)


def main():
    correctness.configure()
    parent.WAIT, parent.JOB_NAME = WAIT, "mk-post-segment"
    control.WAIT, control.SCRIPT = WAIT, SCRIPT
    control.CANDIDATE_ARM, control.POLICY = "joint-kv-segment", "kv"
    control.VARIANT, control.DOC = "consumed_common_post_attention_segment", DOC
    control.ARMS, control.AUXILIARY_PROFILE_ARMS = ARMS, frozenset()
    control.prepare, control.configurations, control.execute, control.summary = (
        prepare,
        configurations,
        execute,
        summary,
    )
    control.worker, control.JOB_NAME = worker, "mk-post-segment"
    control.main()


if __name__ == "__main__":
    main()
