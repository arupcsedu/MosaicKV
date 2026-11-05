#!/usr/bin/env python3
"""Finite all-reader input normalization screen; no new quality identities."""

import traceback
from pathlib import Path

import evicastkv_joint_key_latency as control
import evicastkv_joint_kv_transaction as correctness
import evicastkv_native_history_reader as prior

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE, parent, driver = control.SOURCE, control.parent, control.driver
WAIT = prior.WAIT
PREVIOUS = Path("/scratch/djy8hg/runs/evicastkv/native-history-reader-consumed-20261105-v1")
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_input_norm_segment.sbatch"
DOC = SOURCE / "mosaickv/docs/input_norm_segment.md"
ARMS = (
    ("mosaickv_exact", "joint-previous"),
    ("mosaickv_exact", "joint-norm"),
    ("prefixkv_reimpl", "graph"),
    ("prefixkv_reimpl", "graph-norm"),
    ("prefixkv_reimpl", "graph-original"),
    ("prefixkv_reimpl", "graph-original-norm"),
)
POLICIES = dict(segment=False, rotary=False, copy=False, frozen=False, overlap=False)
HISTORY = False
ORIGINAL_WORKER, ORIGINAL_SUMMARY = control.worker, control.summary


def history_policy(decision):
    if not decision["complete"] or decision["failures"] or len(decision["rows"]) != 2:
        raise ValueError("complete history numerical and ownership controls required")
    for row in decision["rows"]:
        means = {(m["method"], m["arm"]): m["total_wall_seconds"] for m in row["means"]}
        for method, old, new in (
            ("mosaickv_exact", "joint-previous", "joint-history"),
            ("prefixkv_reimpl", "graph", "graph-history"),
            ("prefixkv_reimpl", "graph-original", "graph-original-history"),
        ):
            if means[method, new] > means[method, old]:
                return False
    return True


def prepare(root, revision):
    if root.exists() or clean_source(SOURCE) != revision:
        raise ValueError("new immutable output root and clean source required")
    previous = driver.read(PREVIOUS / "plan.json")
    decision = driver.read(PREVIOUS / "analysis/decision.json")
    history = history_policy(decision)
    old = driver.read(parent.PARENT / "plan.json")
    bound = dict(old["bound_files"])
    bound.update(
        {
            str(path): sha(path)
            for path in (
                DOC,
                PREVIOUS / "plan.json",
                PREVIOUS / "analysis/decision.json",
                WAIT / "plan.json",
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
            stage="consumed_common_input_normalization_replay",
            arms=ARMS,
            warmups=5,
            repetitions=3,
            frozen_runtime_policies=previous["frozen_runtime_policies"],
            common_native_history=history,
            predecessor_source_sha=driver.read(WAIT / "plan.json")["source_sha"],
            predecessor_plan_sha256=sha(WAIT / "plan.json"),
            original_norm_operators_and_rounding=True,
            common_extra_owned_workspace_bytes=49156,
            common_lazy_unused_prefill_audit_fix=True,
            unchanged_all_128_logit_vectors_required=True,
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
        base = "graph-lt" if active else arm.removesuffix("-norm")
        result[method, arm] = dict(
            records[method][base],
            graph_accounting="event_sum",
            reuse_patch_dispatch=True,
            reuse_graph_replay_plan=True,
            post_attention_segment=POLICIES["segment"],
            exact_rotary_dispatch=POLICIES["rotary"],
            reuse_source_input_copy=POLICIES["copy"] if active else True,
            frozen_joint_plan=POLICIES["frozen"] if active else True,
            overlap_native_fit=POLICIES["overlap"] if active else True,
            native_history_reader=HISTORY,
            input_norm_segment=arm.endswith("-norm"),
            **(dict(lt_fit_policy="direct", joint_incoming_projection="kv") if active else {}),
        )
    return result


def execute(adapter, refs, messages, profile, plan, method, arm, record, **kwargs):
    effective = (
        ("joint-history" if HISTORY else "joint-previous")
        if method == "mosaickv_exact"
        else (arm.removesuffix("-norm") + ("-history" if HISTORY else ""))
    )
    value = prior.execute(
        adapter, refs, messages, profile, plan, method, effective, record, **kwargs
    )
    report = value["common_input_norm"]
    if report["enabled"] is not record["input_norm_segment"]:
        raise RuntimeError("common input norm did not engage as configured")
    if report["enabled"] and (
        (report["calls"], report["original_cold_calls"], report["captures"], report["replays"])
        != (4064, 32, 32, 4032)
        or report["explicit_workspace_bytes"] != 49156
        or report["additional_graph_tensor_bytes"]
        or not report["retired"]
        or not report["common_to_every_method"]
    ):
        raise RuntimeError("complete exact norm counts, ownership and retirement required")
    return value


def summary(rows, audited):
    for method, old, new in (
        ("mosaickv_exact", "joint-previous", "joint-norm"),
        ("prefixkv_reimpl", "graph", "graph-norm"),
        ("prefixkv_reimpl", "graph-original", "graph-original-norm"),
    ):
        before, after = audited[method, old], audited[method, new]
        if any(before[k] != after[k] for k in driver.STORAGE[:2]):
            raise ValueError("input norm changed initial/final encoded cache storage")
        if after[driver.STORAGE[-1]] != before[driver.STORAGE[-1]] + 49156:
            raise ValueError("all norm tensor workspace must be charged in decode peak")
    return dict(
        ORIGINAL_SUMMARY(rows, audited),
        exact_common_norm_workspace_charge=True,
        whole_request_GPU_peak_certified=False,
        independent_quality_N=0,
    )


def worker(root, expected, index):
    global HISTORY
    plan = driver.checked(root, expected)
    POLICIES.update(plan["frozen_runtime_policies"])
    HISTORY = plan["common_native_history"]
    prior.POLICIES.update(POLICIES)
    prior.prior.POLICIES.update({k: v for k, v in POLICIES.items() if k != "overlap"})
    prior.prior.prior.COMMON.update({k: POLICIES[k] for k in ("segment", "rotary", "copy")})
    try:
        from mosaickv.evaluation.input_norm_segment import gpu_controls

        report = gpu_controls()
        write_once(root / f"norm-controls/{index}.json", report)
    except Exception as error:
        out = root / f"workers/{index}"
        out.mkdir(parents=True, exist_ok=False)
        driver.finish_worker(
            out,
            dict(
                status="failed",
                failures=[str(error)],
                traceback=traceback.format_exc(),
                source_sha=plan["source_sha"],
                plan_sha256=expected,
            ),
        )
        return
    ORIGINAL_WORKER(root, expected, index)


def main():
    correctness.configure()
    parent.WAIT, parent.JOB_NAME = WAIT, "mk-input-norm"
    control.WAIT, control.SCRIPT = WAIT, SCRIPT
    control.CANDIDATE_ARM, control.POLICY = "joint-norm", "kv"
    control.VARIANT, control.DOC = "consumed_common_input_normalization_replay", DOC
    control.ARMS, control.AUXILIARY_PROFILE_ARMS = ARMS, frozenset()
    control.prepare, control.configurations, control.execute, control.summary = (
        prepare,
        configurations,
        execute,
        summary,
    )
    control.worker, control.JOB_NAME = worker, "mk-input-norm"
    control.main()


if __name__ == "__main__":
    main()
