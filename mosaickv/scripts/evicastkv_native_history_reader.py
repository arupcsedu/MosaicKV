#!/usr/bin/env python3
"""Finite all-reader native-history runtime screen after operator feasibility."""

from pathlib import Path

import evicastkv_joint_key_latency as control
import evicastkv_joint_kv_transaction as correctness
import evicastkv_prefill_native_overlap as prior

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE, parent, driver = control.SOURCE, control.parent, control.driver
WAIT = prior.WAIT
PREVIOUS = Path("/scratch/djy8hg/runs/evicastkv/prefill-native-overlap-consumed-20261105-retry1")
OPERATOR = Path("/scratch/djy8hg/runs/evicastkv/native-history-feasibility-20261105-llama1")
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_native_history_reader.sbatch"
DOC = SOURCE / "mosaickv/docs/native_history_reader.md"
ARMS = (
    ("mosaickv_exact", "joint-previous"),
    ("mosaickv_exact", "joint-history"),
    ("prefixkv_reimpl", "graph"),
    ("prefixkv_reimpl", "graph-history"),
    ("prefixkv_reimpl", "graph-original"),
    ("prefixkv_reimpl", "graph-original-history"),
)
POLICIES = dict(segment=False, rotary=False, copy=False, frozen=False, overlap=False)
ORIGINAL_WORKER, ORIGINAL_SUMMARY = control.worker, control.summary


def overlap_policy(decision):
    if not decision["complete"] or decision["failures"] or len(decision["rows"]) != 2:
        raise ValueError("complete overlap numerical/ownership controls required before combining")
    for row in decision["rows"]:
        if not row["own_initial_final_peak_identical"]:
            raise ValueError("previous source storage mismatch")
        means = {(m["method"], m["arm"]): m["total_wall_seconds"] for m in row["means"]}
        if means["mosaickv_exact", "joint-overlap"] > means["mosaickv_exact", "joint-previous"]:
            return False
    return True


def prepare(root, revision):
    if root.exists() or clean_source(SOURCE) != revision:
        raise ValueError("new immutable output root and clean source required")
    operator = driver.read(OPERATOR / "decision.json")
    if (
        not operator["complete"]
        or not operator["all_epochs_operator_faster"]
        or [r["rows"] for r in operator["records"]] != list(range(16, 113, 16))
        or any(
            r["native_added_repair_bytes"] > 32768
            or not r["original_words_and_tail_sentinels_preserved"]
            for r in operator["records"]
        )
    ):
        raise ValueError("all seven exact bounded native operator epochs must pass")
    previous = driver.read(PREVIOUS / "plan.json")
    policies = dict(
        previous["frozen_runtime_policies"],
        overlap=overlap_policy(driver.read(PREVIOUS / "analysis/decision.json")),
    )
    old = driver.read(parent.PARENT / "plan.json")
    bound = dict(old["bound_files"])
    bound.update(
        {
            str(path): sha(path)
            for path in (
                DOC,
                PREVIOUS / "plan.json",
                PREVIOUS / "analysis/decision.json",
                OPERATOR / "manifest.json",
                OPERATOR / "config.json",
                OPERATOR / "decision.json",
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
            stage="consumed_common_native_history",
            arms=ARMS,
            warmups=5,
            repetitions=3,
            frozen_runtime_policies=policies,
            predecessor_source_sha=driver.read(WAIT / "plan.json")["source_sha"],
            predecessor_plan_sha256=sha(WAIT / "plan.json"),
            maximum_native_history_repair_bytes_per_layer=32768,
            original_prompt_bytes_unchanged=True,
            all_history_words_unchanged=True,
            final_bytes_may_change_only_by_measured_numerical_repairs=True,
            owned_native_workspace_maximum_bytes=2**21,
            common_native_history_offered_to_all_readers=True,
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
        base = "graph-lt" if active else arm.removesuffix("-history")
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
            native_history_reader=arm.endswith("-history"),
            **(dict(lt_fit_policy="direct", joint_incoming_projection="kv") if active else {}),
        )
    return result


def execute(adapter, refs, messages, profile, plan, method, arm, record, **kwargs):
    effective = (
        ("joint-overlap" if POLICIES["overlap"] else "joint-previous")
        if method == "mosaickv_exact"
        else arm.removesuffix("-history")
    )
    value = prior.execute(
        adapter, refs, messages, profile, plan, method, effective, record, **kwargs
    )
    info = value["common_native_history"]
    if info["enabled"] is not record["native_history_reader"]:
        raise RuntimeError("native history configuration did not actually engage")
    if info["enabled"]:
        report = info["report"]
        if (
            not report["retired"]
            or report["built_chunks"] != 7 * 32
            or report["consolidated_layers"] != 6 * 32
            or report["projections"] <= 7104
            or report["maximum_per_layer_repair_bytes"] > 32768
            or report["additional_owned_workspace_bytes"]
            != (0 if method == "mosaickv_exact" else 2**21)
            or report["reused_existing_prompt_workspace_bytes"]
            != (2**21 if method == "mosaickv_exact" else 0)
        ):
            raise RuntimeError(
                "complete real native history and exact charged/retired owners required"
            )
    return value


def summary(rows, audited):
    comparisons = []
    for method, old, new in (
        ("mosaickv_exact", "joint-previous", "joint-history"),
        ("prefixkv_reimpl", "graph", "graph-history"),
        ("prefixkv_reimpl", "graph-original", "graph-original-history"),
    ):
        before, after = audited[method, old], audited[method, new]
        if before[driver.STORAGE[0]] != after[driver.STORAGE[0]]:
            raise ValueError("native history changed the original prompt storage")
        comparisons.append(
            dict(
                method=method,
                original_arm=old,
                native_arm=new,
                original_storage={k: before[k] for k in driver.STORAGE},
                native_storage={k: after[k] for k in driver.STORAGE},
            )
        )
    return dict(
        ORIGINAL_SUMMARY(rows, audited),
        own_initial_prompt_bytes_identical=True,
        storage_comparisons=comparisons,
        final_history_repairs_measured_not_assumed_identical=True,
        independent_quality_N=0,
        whole_request_GPU_peak_certified=False,
    )


def worker(root, expected, index):
    POLICIES.update(driver.checked(root, expected)["frozen_runtime_policies"])
    prior.POLICIES.update({k: v for k, v in POLICIES.items() if k != "overlap"})
    prior.prior.COMMON.update({k: POLICIES[k] for k in ("segment", "rotary", "copy")})
    ORIGINAL_WORKER(root, expected, index)


def main():
    correctness.configure()
    parent.WAIT, parent.JOB_NAME = WAIT, "mk-native-history"
    control.WAIT, control.SCRIPT = WAIT, SCRIPT
    control.CANDIDATE_ARM, control.POLICY = "joint-history", "kv"
    control.VARIANT, control.DOC = "consumed_common_native_history", DOC
    control.ARMS, control.AUXILIARY_PROFILE_ARMS = ARMS, frozenset()
    control.prepare, control.configurations, control.execute, control.summary = (
        prepare,
        configurations,
        execute,
        summary,
    )
    control.worker, control.JOB_NAME = worker, "mk-native-history"
    control.main()


if __name__ == "__main__":
    main()
