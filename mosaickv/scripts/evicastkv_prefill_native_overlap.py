#!/usr/bin/env python3
"""Bounded construction scheduling, not another selection/quality campaign."""

from pathlib import Path

import evicastkv_frozen_joint_plan as prior
import evicastkv_joint_key_latency as control
import evicastkv_joint_kv_transaction as correctness

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE, parent, driver = control.SOURCE, control.parent, control.driver
WAIT = prior.WAIT
FROZEN = Path("/scratch/djy8hg/runs/evicastkv/frozen-joint-consumed-20261105-v1")
DOC = SOURCE / "mosaickv/docs/prefill_native_overlap.md"
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_prefill_native_overlap.sbatch"
ARMS = (
    ("mosaickv_exact", "joint-previous"),
    ("mosaickv_exact", "joint-overlap"),
    ("prefixkv_reimpl", "graph"),
    ("prefixkv_reimpl", "graph-original"),
)
POLICIES = dict(segment=False, rotary=False, copy=False, frozen=False)
ORIGINAL_WORKER, ORIGINAL_SUMMARY = control.worker, control.summary


def frozen_policy(decision):
    if not decision["complete"] or decision["failures"] or len(decision["rows"]) != 2:
        raise ValueError("complete frozen-plan numerical and storage controls required")
    for row in decision["rows"]:
        if not row["own_initial_final_peak_identical"]:
            raise ValueError("own-control storage changed")
        means = {(x["method"], x["arm"]): x["total_wall_seconds"] for x in row["means"]}
        if means["mosaickv_exact", "joint-frozen"] > means["mosaickv_exact", "joint-previous"]:
            return False
    return True


def prepare(root, revision):
    if root.exists() or clean_source(SOURCE) != revision:
        raise ValueError("new output root and clean immutable source required")
    before = driver.read(FROZEN / "plan.json")
    policies = dict(
        before["frozen_runtime_policies"],
        frozen=frozen_policy(driver.read(FROZEN / "analysis/decision.json")),
    )
    old = driver.read(parent.PARENT / "plan.json")
    bound = dict(old["bound_files"])
    bound.update(
        {
            str(path): sha(path)
            for path in (
                DOC,
                FROZEN / "plan.json",
                FROZEN / "analysis/decision.json",
                WAIT / "analysis/decision.json",
            )
        }
    )
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=revision,
            bound_files=bound,
            stage="consumed_prefill_native_overlap",
            arms=ARMS,
            warmups=5,
            repetitions=3,
            frozen_runtime_policies=policies,
            predecessor_source_sha=driver.read(WAIT / "plan.json")["source_sha"],
            predecessor_plan_sha256=sha(WAIT / "plan.json"),
            one_thread_stream_handle_per_request=True,
            maximum_pending_layers=2,
            original_all_word_and_stride_checks_unchanged=True,
            final_original_source_identity_required=True,
            independent_quality_N=0,
            no_fresh_identities=True,
            whole_request_GPU_peak_certified=False,
            automatic_quality_submission=False,
            automatic_formal_submission=False,
            automatic_full_request_submission=False,
            automatic_bounded_followup=False,
            automatic_bounded_cohort_after_correctness=False,
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
            post_attention_segment=POLICIES["segment"],
            exact_rotary_dispatch=POLICIES["rotary"],
            reuse_source_input_copy=POLICIES["copy"] if active else True,
            frozen_joint_plan=POLICIES["frozen"] if active else True,
            overlap_native_fit=arm == "joint-overlap" or not active,
            **(dict(lt_fit_policy="direct", joint_incoming_projection="kv") if active else {}),
        )
    return result


def execute(adapter, refs, messages, profile, plan, method, arm, record, **kwargs):
    effective = "joint-frozen" if method == "mosaickv_exact" and POLICIES["frozen"] else arm
    if method == "mosaickv_exact" and not POLICIES["frozen"]:
        effective = "joint-previous"
    value = prior.execute(
        adapter, refs, messages, profile, plan, method, effective, record, **kwargs
    )
    info = value["overlapping_native_construction"]
    active = arm == "joint-overlap"
    if info["requested"] is not record["overlap_native_fit"] or info["eligible"] is not active:
        raise RuntimeError("overlap must engage only for the declared eligible Source arm")
    if active:
        report = info["report"]
        if (
            not report["completed"]
            or not report["thread_retired"]
            or report["submitted"] != 32
            or report["consumed"] != 32
            or report["engine_count"] != 1
            or report["maximum_pending_layers"] > 2
            or (
                kwargs.get("audit", False)
                and (report["explicit_temporary_peak"] <= 0 or report["audited_operations"] <= 0)
            )
        ):
            raise RuntimeError("all 32 unchanged fits, audit and private retirement required")
    elif info["report"] is not None:
        raise RuntimeError("ineligible/original must remain an unchanged no-op")
    return value


def summary(rows, audited):
    old, new = (
        audited["mosaickv_exact", "joint-previous"],
        audited["mosaickv_exact", "joint-overlap"],
    )
    if any(old[k] != new[k] for k in driver.STORAGE):
        raise ValueError("constructor scheduling changed decode-owned physical storage")
    return dict(
        ORIGINAL_SUMMARY(rows, audited),
        own_initial_final_peak_identical=True,
        independent_quality_N=0,
        whole_request_GPU_peak_certified=False,
    )


def worker(root, expected, index):
    POLICIES.update(driver.checked(root, expected)["frozen_runtime_policies"])
    prior.COMMON.update({k: v for k, v in POLICIES.items() if k != "frozen"})
    ORIGINAL_WORKER(root, expected, index)


def main():
    correctness.configure()
    parent.WAIT, parent.JOB_NAME = WAIT, "mk-prefill-fit"
    control.WAIT, control.SCRIPT = WAIT, SCRIPT
    control.CANDIDATE_ARM, control.POLICY = "joint-overlap", "kv"
    control.VARIANT, control.DOC = "consumed_prefill_native_overlap", DOC
    control.ARMS, control.AUXILIARY_PROFILE_ARMS = ARMS, frozenset()
    control.prepare, control.configurations, control.execute, control.summary = (
        prepare,
        configurations,
        execute,
        summary,
    )
    control.worker, control.JOB_NAME = worker, "mk-prefill-fit"
    control.main()


if __name__ == "__main__":
    main()
