#!/usr/bin/env python3
"""Bounded once-only source copy, atop individually verified common paths."""

import traceback
from pathlib import Path

import evicastkv_exact_rotary_dispatch as rotary
import evicastkv_joint_key_latency as control
import evicastkv_joint_kv_transaction as correctness

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once
from mosaickv.evaluation.joint_input_copy import gpu_controls

SOURCE, parent, driver = control.SOURCE, control.parent, control.driver
WAIT = Path("/scratch/djy8hg/runs/evicastkv/post-attention-segment-consumed-20261105-retry1")
ROTARY = Path("/scratch/djy8hg/runs/evicastkv/exact-rotary-consumed-20261105-retry1")
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_joint_input_reuse.sbatch"
DOC = SOURCE / "mosaickv/docs/joint_input_reuse.md"
ARMS = (
    ("mosaickv_exact", "joint-previous"),
    ("mosaickv_exact", "joint-input"),
    ("prefixkv_reimpl", "graph"),
    ("prefixkv_reimpl", "graph-original"),
)
COMMON_SEGMENT = COMMON_ROTARY = False
ORIGINAL_EXECUTE, ORIGINAL_SUMMARY, ORIGINAL_WORKER = (
    control.execute,
    control.summary,
    control.worker,
)


def common_rotary_policy(decision):
    if not decision["complete"] or decision["failures"] or len(decision["rows"]) != 2:
        return False
    for row in decision["rows"]:
        if not row.get("identical_own_control_initial_final_peak"):
            return False
        means = {(x["method"], x["arm"]): x["total_wall_seconds"] for x in row["means"]}
        for method, old, new in (
            ("mosaickv_exact", "joint-kv-prepared", "joint-kv-rotary"),
            ("prefixkv_reimpl", "graph-prepared", "graph-rotary"),
            ("prefixkv_reimpl", "graph-original-prepared", "graph-original-rotary"),
        ):
            if means[method, new] > means[method, old]:
                return False
    return True


def prepare(root, revision):
    if root.exists() or clean_source(SOURCE) != revision:
        raise ValueError("new immutable output root and clean source required")
    previous, gate = driver.read(WAIT / "plan.json"), driver.read(WAIT / "analysis/decision.json")
    if not gate["complete"] or gate["failures"]:
        raise ValueError("complete common epilogue numerical/storage controls required")
    common_segment = rotary.segment_policy(gate)
    common_rotary = common_rotary_policy(driver.read(ROTARY / "analysis/decision.json"))
    old, paths = (
        driver.read(parent.PARENT / "plan.json"),
        (
            WAIT / "plan.json",
            WAIT / "analysis/decision.json",
            ROTARY / "plan.json",
            ROTARY / "analysis/decision.json",
            DOC,
        ),
    )
    bound = dict(old["bound_files"])
    bound.update({str(path): sha(path) for path in paths})
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=revision,
            bound_files=bound,
            stage="consumed_joint_normalized_input_reuse",
            arms=ARMS,
            warmups=5,
            repetitions=3,
            common_segment=common_segment,
            common_rotary=common_rotary,
            common_runtime_policy_rule="all_own_controls_improve_on_both_consumed_inputs_or_off_for_all",
            unchanged_original_selected_KV_history_and_positions=True,
            no_extra_GPU_tensor_workspace=True,
            independent_quality_N=0,
            no_fresh_identities=True,
            baseline_ineligible_for_duplicate_input_copy_is_truthful_noop=True,
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
        base = "graph-lt" if active else arm
        result[method, arm] = dict(
            records[method][base],
            graph_accounting="event_sum",
            reuse_patch_dispatch=True,
            reuse_graph_replay_plan=True,
            post_attention_segment=COMMON_SEGMENT,
            exact_rotary_dispatch=COMMON_ROTARY,
            reuse_source_input_copy=arm == "joint-input" or not active,
            **(dict(lt_fit_policy="direct", joint_incoming_projection="kv") if active else {}),
        )
    return result


def execute(adapter, refs, messages, profile, plan, method, arm, record, **kwargs):
    result = ORIGINAL_EXECUTE(adapter, refs, messages, profile, plan, method, arm, record, **kwargs)
    info = result["joint_incoming_projection"]["source_input_copy"]
    if info["enabled"] is not record["reuse_source_input_copy"]:
        raise RuntimeError("declared once-only source-copy policy did not engage")
    if arm == "joint-input":
        if info["reports"] != [
            dict(
                prepared=4064,
                consumed=4064,
                pending=False,
                compiled_dispatch_hits=4063,
                compiled_dispatch_misses=1,
                extra_GPU_tensor_bytes=0,
                original_two_destinations_and_source_words=True,
            )
        ]:
            raise RuntimeError("every source input must fan out and be consumed exactly once")
    elif info["reports"]:
        raise RuntimeError("ineligible baseline/original must remain an honest unchanged no-op")
    return result


def summary(rows, audited):
    old, new = audited["mosaickv_exact", "joint-previous"], audited["mosaickv_exact", "joint-input"]
    if any(old[k] != new[k] for k in driver.STORAGE):
        raise ValueError("once-only input copy changed a real cache boundary")
    return dict(
        ORIGINAL_SUMMARY(rows, audited),
        own_initial_final_peak_identical=True,
        independent_quality_N=0,
        whole_request_GPU_peak_certified=False,
    )


def worker(root, expected, index):
    global COMMON_SEGMENT, COMMON_ROTARY
    plan = driver.checked(root, expected)
    COMMON_SEGMENT, COMMON_ROTARY = plan["common_segment"], plan["common_rotary"]
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
                phase="lossless_copy_control",
            ),
        )
        return
    write_once(root / f"source_copy_controls_{index}.json", controls)
    ORIGINAL_WORKER(root, expected, index)


def main():
    correctness.configure()
    parent.WAIT, parent.JOB_NAME = WAIT, "mk-input-reuse"
    control.WAIT, control.SCRIPT = WAIT, SCRIPT
    control.CANDIDATE_ARM, control.POLICY = "joint-input", "kv"
    control.VARIANT, control.DOC = "consumed_joint_normalized_input_reuse", DOC
    control.ARMS, control.AUXILIARY_PROFILE_ARMS = ARMS, frozenset()
    control.prepare, control.configurations, control.execute, control.summary = (
        prepare,
        configurations,
        execute,
        summary,
    )
    control.worker, control.JOB_NAME = worker, "mk-input-reuse"
    control.main()


if __name__ == "__main__":
    main()
