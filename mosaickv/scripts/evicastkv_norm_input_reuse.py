#!/usr/bin/env python3
"""Same-consumed-input, common epilogue-to-normalization buffer reuse."""

from pathlib import Path

import evicastkv_input_norm_segment as prior
import evicastkv_joint_key_latency as control
import evicastkv_joint_kv_transaction as correctness

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE, parent, driver = control.SOURCE, control.parent, control.driver
WAIT = prior.WAIT
PREVIOUS = Path("/scratch/djy8hg/runs/evicastkv/input-norm-consumed-20261105-v1")
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_norm_input_reuse.sbatch"
DOC = SOURCE / "mosaickv/docs/norm_input_reuse.md"
ARMS = (
    ("mosaickv_exact", "joint-previous"),
    ("mosaickv_exact", "joint-reuse"),
    ("prefixkv_reimpl", "graph"),
    ("prefixkv_reimpl", "graph-reuse"),
    ("prefixkv_reimpl", "graph-original"),
    ("prefixkv_reimpl", "graph-original-reuse"),
)
NORM = False
ORIGINAL_SUMMARY = control.summary


def norm_policy(decision):
    if not decision["complete"] or decision["failures"] or len(decision["rows"]) != 2:
        raise ValueError("complete predecessor norm controls required")
    for row in decision["rows"]:
        means = {(m["method"], m["arm"]): m["total_wall_seconds"] for m in row["means"]}
        for method, old, new in (
            ("mosaickv_exact", "joint-previous", "joint-norm"),
            ("prefixkv_reimpl", "graph", "graph-norm"),
            ("prefixkv_reimpl", "graph-original", "graph-original-norm"),
        ):
            if means[method, new] > means[method, old]:
                return False
    return True


def prepare(root, revision):
    if root.exists() or clean_source(SOURCE) != revision:
        raise ValueError("new immutable root and clean source required")
    previous = driver.read(PREVIOUS / "plan.json")
    norm = norm_policy(driver.read(PREVIOUS / "analysis/decision.json"))
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
            stage="consumed_common_norm_input_reuse",
            arms=ARMS,
            warmups=5,
            repetitions=3,
            frozen_runtime_policies=previous["frozen_runtime_policies"],
            common_native_history=previous["common_native_history"],
            common_original_norm_replay=norm,
            predecessor_source_sha=driver.read(WAIT / "plan.json")["source_sha"],
            predecessor_plan_sha256=sha(WAIT / "plan.json"),
            later_layer_original_epilogue_output_must_alias=True,
            expected_skipped_norm_input_copies=3906,
            expected_embedding_input_copies=126,
            original_unused_input_buffer_still_counted=True,
            no_arithmetic_or_KV_changes=True,
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
        active, reuse = method == "mosaickv_exact", arm.endswith("-reuse")
        base = "graph-lt" if active else arm.removesuffix("-reuse")
        result[method, arm] = dict(
            records[method][base],
            graph_accounting="event_sum",
            reuse_patch_dispatch=True,
            reuse_graph_replay_plan=True,
            post_attention_segment=prior.POLICIES["segment"],
            exact_rotary_dispatch=prior.POLICIES["rotary"],
            reuse_source_input_copy=prior.POLICIES["copy"] if active else True,
            frozen_joint_plan=prior.POLICIES["frozen"] if active else True,
            overlap_native_fit=prior.POLICIES["overlap"] if active else True,
            native_history_reader=prior.HISTORY,
            input_norm_segment=NORM or reuse,
            reuse_norm_input=reuse,
            **(dict(lt_fit_policy="direct", joint_incoming_projection="kv") if active else {}),
        )
    return result


def execute(adapter, refs, messages, profile, plan, method, arm, record, **kwargs):
    norm = record["input_norm_segment"]
    effective = (
        ("joint-norm" if norm else "joint-previous")
        if method == "mosaickv_exact"
        else (arm.removesuffix("-reuse") + ("-norm" if norm else ""))
    )
    value = prior.execute(
        adapter, refs, messages, profile, plan, method, effective, record, **kwargs
    )
    report = value["common_input_norm"]
    if norm and (
        report["input_reuse_requested"] is not record["reuse_norm_input"]
        or (report["skipped_later_layer_input_copies"], report["first_layer_embedding_copies"])
        != ((3906, 126) if record["reuse_norm_input"] else (0, 0))
    ):
        raise RuntimeError("actual complete common input-reuse trajectory required")
    return value


def summary(rows, audited):
    for method, old, new in (
        ("mosaickv_exact", "joint-previous", "joint-reuse"),
        ("prefixkv_reimpl", "graph", "graph-reuse"),
        ("prefixkv_reimpl", "graph-original", "graph-original-reuse"),
    ):
        before, after = audited[method, old], audited[method, new]
        if any(before[k] != after[k] for k in driver.STORAGE[:2]):
            raise ValueError("input reuse changed initial/final storage")
        if after[driver.STORAGE[-1]] != before[driver.STORAGE[-1]] + (0 if NORM else 49156):
            raise ValueError("exact common norm workspace delta required")
    return dict(
        ORIGINAL_SUMMARY(rows, audited),
        exact_common_norm_workspace_charge=True,
        no_storage_removed_for_unused_buffer=True,
        independent_quality_N=0,
        whole_request_GPU_peak_certified=False,
    )


def worker(root, expected, index):
    global NORM
    NORM = driver.checked(root, expected)["common_original_norm_replay"]
    prior.worker(root, expected, index)


def main():
    correctness.configure()
    parent.WAIT, parent.JOB_NAME = WAIT, "mk-norm-input"
    control.WAIT, control.SCRIPT = WAIT, SCRIPT
    control.CANDIDATE_ARM, control.POLICY = "joint-reuse", "kv"
    control.VARIANT, control.DOC = "consumed_common_norm_input_reuse", DOC
    control.ARMS, control.AUXILIARY_PROFILE_ARMS = ARMS, frozenset()
    control.prepare, control.configurations, control.execute, control.summary = (
        prepare,
        configurations,
        execute,
        summary,
    )
    control.worker, control.JOB_NAME = worker, "mk-norm-input"
    control.main()


if __name__ == "__main__":
    main()
