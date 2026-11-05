#!/usr/bin/env python3
"""Remove measured duplicate replay preparation, offered to every reader."""

from pathlib import Path

import evicastkv_joint_key_latency as control
import evicastkv_joint_kv_transaction as correctness

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once
from mosaickv.evaluation.joint_graph_scope import reconcile

SOURCE, parent, driver = control.SOURCE, control.parent, control.driver
WAIT = Path("/scratch/djy8hg/runs/evicastkv/joint-kv-latency-consumed-20261105-v1")
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_prepared_joint_replay.sbatch"
DOC = SOURCE / "mosaickv/docs/prepared_joint_replay.md"
ARMS = (
    ("mosaickv_exact", "graph"),
    ("mosaickv_exact", "joint-kv-legacy"),
    ("mosaickv_exact", "joint-kv-prepared"),
    ("prefixkv_reimpl", "graph"),
    ("prefixkv_reimpl", "graph-prepared"),
    ("prefixkv_reimpl", "graph-original"),
    ("prefixkv_reimpl", "graph-original-prepared"),
)
ORIGINAL_EXECUTE, ORIGINAL_SUMMARY = control.execute, control.summary


def scope_review():
    corrections = []
    for index in (0, 1):
        for path in sorted((WAIT / f"workers/{index}").glob("audited-*.json")):
            raw = driver.read(path)
            derived = reconcile(raw)
            if derived["graph_scope_metadata_correction"]["correction_required"]:
                corrections.append(
                    dict(
                        raw_artifact=str(path),
                        raw_sha256=sha(path),
                        original_source_sha=driver.read(WAIT / "plan.json")["source_sha"],
                        correction=derived["graph_scope_metadata_correction"],
                        corrected_graph_scope={
                            k: derived["prompt_graph_reader"][k]
                            for k in (
                                "only_immutable_prompt_captured",
                                "captured_incoming_components",
                                "reused_incoming_components",
                                "attention_softmax_and_model_layers_captured",
                            )
                        },
                    )
                )
    if len(corrections) != 2:
        raise ValueError("exactly the two recorded joint-KV scope labels require correction")
    return dict(
        corrections=corrections,
        all_original_raw_files_preserved=True,
        no_timing_bytes_tokens_or_logits_changed=True,
        provenance_correction_not_a_new_GPU_measurement=True,
    )


def prepare(root, revision):
    if root.exists() or clean_source(SOURCE) != revision:
        raise ValueError("new root and clean immutable source required")
    gate, preceding = driver.read(WAIT / "analysis/decision.json"), driver.read(WAIT / "plan.json")
    if not gate["complete"] or gate["failures"] or gate["four_percent_POINT_only"]:
        raise ValueError("complete negative four-percent predecessor screen required")
    old = driver.read(parent.PARENT / "plan.json")
    review = root / "predecessor_scope_review.json"
    write_once(review, scope_review())
    bound = dict(old["bound_files"])
    paths = (
        WAIT / "plan.json",
        WAIT / "analysis/decision.json",
        DOC,
        review,
        SOURCE / "mosaickv/src/mosaickv/evaluation/joint_graph_scope.py",
    )
    bound.update({str(path): sha(path) for path in paths})
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=revision,
            bound_files=bound,
            stage="consumed_prepared_joint_graph_replay",
            arms=ARMS,
            warmups=5,
            repetitions=3,
            optimization_offered_to_all_compared_baselines=True,
            independent_quality_N=0,
            no_fresh_identities=True,
            no_algorithm_math_or_cache_transformation=True,
            mandatory_KV_positions_and_precision_unchanged=True,
            no_extra_GPU_tensor_workspace=True,
            automatic_quality_submission=False,
            automatic_formal_submission=False,
            automatic_full_request_submission=False,
            automatic_bounded_followup=False,
            automatic_bounded_cohort_after_correctness=False,
            predecessor_source_sha=preceding["source_sha"],
            predecessor_plan_sha256=sha(WAIT / "plan.json"),
            no_byte_matched_algorithm_claim=True,
            whole_request_GPU_peak_certified=False,
        ),
    )
    cpu = parent.submit(root, "cpu")
    release = parent.submit(root, "release", cpu)
    write_once(root / "cpu_submission.json", dict(cpu=cpu, release=release))


def configurations(records):
    result = {}
    for method, arm in ARMS:
        active = arm.startswith("joint-kv")
        base = "graph-lt" if active else arm.removesuffix("-prepared")
        result[method, arm] = dict(
            records[method][base],
            graph_accounting="event_sum",
            reuse_patch_dispatch=True,
            reuse_graph_replay_plan=arm.endswith("-prepared"),
            **(dict(lt_fit_policy="direct", joint_incoming_projection="kv") if active else {}),
        )
    return result


def execute(adapter, refs, messages, profile, plan, method, arm, record, **kwargs):
    value = ORIGINAL_EXECUTE(adapter, refs, messages, profile, plan, method, arm, record, **kwargs)
    graph = value["prompt_graph_reader"]
    prepared = record.get("reuse_graph_replay_plan", False)
    info = graph["prepared_replay_plan"]
    active = record.get("joint_incoming_projection", "none") != "none"
    if (
        info["enabled"] is not prepared
        or info["hits"] != (8064 if prepared else 0)
        or info["joint_key_consume_bypasses"] != (4032 if prepared and active else 0)
        or info["additional_GPU_tensor_bytes"]
        or not info["CPU_only_metadata"]
        or graph["only_immutable_prompt_captured"] is not (not active)
    ):
        raise RuntimeError(
            "exact prepared replay engagement and honest dynamic graph scope required"
        )
    return value


def summary(rows, audited):
    pairs = (
        ("mosaickv_exact", "joint-kv-legacy", "joint-kv-prepared"),
        ("prefixkv_reimpl", "graph", "graph-prepared"),
        ("prefixkv_reimpl", "graph-original", "graph-original-prepared"),
    )
    if any(
        audited[m, old][k] != audited[m, new][k] for m, old, new in pairs for k in driver.STORAGE
    ):
        raise ValueError("prepared replay changed an actual initial/final/peak storage boundary")
    return dict(
        ORIGINAL_SUMMARY(rows, audited),
        own_control_three_storage_boundaries_identical=True,
        original_numerical_guards_unchanged=True,
        predecessor_scope_correction_is_metadata_only=True,
    )


def main():
    correctness.configure()
    parent.WAIT, parent.JOB_NAME = WAIT, "mk-prepared-replay"
    control.WAIT, control.SCRIPT = WAIT, SCRIPT
    control.CANDIDATE_ARM, control.POLICY = "joint-kv-prepared", "kv"
    control.VARIANT, control.DOC = "consumed_prepared_joint_graph_replay", DOC
    control.GATE_FIELD = "complete"
    control.ARMS, control.AUXILIARY_PROFILE_ARMS = ARMS, frozenset()
    control.prepare, control.configurations = prepare, configurations
    control.execute, control.summary = execute, summary
    control.JOB_NAME = "mk-prepared-replay"
    control.main()


if __name__ == "__main__":
    main()
