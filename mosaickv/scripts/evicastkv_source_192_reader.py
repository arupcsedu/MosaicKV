#!/usr/bin/env python3
"""Bit-exact corrected source-192 full-request screen, same consumed inputs."""

from pathlib import Path

import evicastkv_joint_key_latency as control
import evicastkv_joint_kv_transaction as correctness
import evicastkv_norm_input_reuse as prior

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE, parent, driver = control.SOURCE, control.parent, control.driver
WAIT = prior.WAIT
PREVIOUS = Path("/scratch/djy8hg/runs/evicastkv/norm-input-reuse-consumed-20261105-v1")
OPERATOR = Path("/scratch/djy8hg/runs/evicastkv/source-192-corrected-operator-20261105-v1")
NEGATIVE = Path("/scratch/djy8hg/runs/evicastkv/source-192-operator-20261105-v1")
SERIAL = Path("/scratch/djy8hg/runs/evicastkv/native-history-host-consumed-20261105-v1")
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_source_192_reader.sbatch"
DOC = SOURCE / "mosaickv/docs/source_192_reader.md"
ARMS = (
    ("mosaickv_exact", "joint-previous"),
    ("mosaickv_exact", "joint-192"),
    ("prefixkv_reimpl", "graph"),
    ("prefixkv_reimpl", "graph-original"),
)
REUSE = False
ORIGINAL_WORKER, ORIGINAL_SUMMARY = control.worker, control.summary


def reuse_policy(decision):
    if not decision["complete"] or decision["failures"] or len(decision["rows"]) != 2:
        raise ValueError("complete exact input-reuse predecessor required")
    for row in decision["rows"]:
        means = {(m["method"], m["arm"]): m["total_wall_seconds"] for m in row["means"]}
        for method, old, new in (
            ("mosaickv_exact", "joint-previous", "joint-reuse"),
            ("prefixkv_reimpl", "graph", "graph-reuse"),
            ("prefixkv_reimpl", "graph-original", "graph-original-reuse"),
        ):
            if means[method, new] > means[method, old]:
                return False
    return True


def check_operator(result):
    if (
        not result["complete"]
        or not result.get("numerically_compatible")
        or not result.get("all_operator_medians_faster")
        or [r["source_length"] for r in result["records"]] != [600, 604]
        or any(
            not r["bitwise_after_actual_counted_correction"]
            or not r["lossless_corrections_explicitly_registered"]
            or r["actual_added_numerical_correction_bytes"] > 32768
            or any(r["different_incoming_words"])
            for r in result["records"]
        )
    ):
        raise ValueError("both exact, counted and faster corrected operator controls required")


def prepare(root, revision):
    if root.exists() or clean_source(SOURCE) != revision:
        raise ValueError("new immutable output root and clean source required")
    check_operator(driver.read(OPERATOR / "decision.json"))
    old = driver.read(PREVIOUS / "plan.json")
    reuse = reuse_policy(driver.read(PREVIOUS / "analysis/decision.json"))
    serial = str(driver.read(SERIAL / "profile_submission.json")["analysis"])
    if not serial.isdecimal() or old["common_native_history"]:
        raise ValueError("finite profiling dependency and original history required")
    bound = dict(old["bound_files"])
    paths = [
        DOC,
        PREVIOUS / "plan.json",
        PREVIOUS / "analysis/decision.json",
        OPERATOR / "config.json",
        OPERATOR / "manifest.json",
        OPERATOR / "decision.json",
        NEGATIVE / "config.json",
        NEGATIVE / "decision.json",
        SERIAL / "plan.json",
        SERIAL / "profile_submission.json",
    ]
    bound.update({str(path): sha(path) for path in paths})
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=revision,
            bound_files=bound,
            stage="consumed_corrected_source_192_reader",
            arms=ARMS,
            warmups=5,
            repetitions=3,
            selected_common_input_reuse=reuse,
            norm_replay_used_in_192_arm=old["common_original_norm_replay"] or reuse,
            extra_decode_owned_source_workspace_bytes=2**20,
            unchanged_per_layer_patch_cap_bytes=32768,
            unchanged_request_patch_cap_bytes=2**20,
            initial_final_bytes_may_change_only_by_actual_native_corrections=True,
            original_KV_and_all_128_logit_words_required=True,
            operator_zero_correction_failure_preserved=True,
            common_source_capacity_offered_to_all_eligible_readers=True,
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
    release = parent.submit(root, "release", f"{cpu}:{serial}")
    write_once(
        root / "cpu_submission.json", dict(cpu=cpu, release=release, serialized_after=serial)
    )


def configurations(records):
    policies = prior.prior.POLICIES
    result = {}
    for method, arm in ARMS:
        active = method == "mosaickv_exact"
        capacity = 192 if arm == "joint-192" or not active else 128
        base = "graph-lt" if active else arm
        result[method, arm] = dict(
            records[method][base],
            graph_accounting="event_sum",
            reuse_patch_dispatch=True,
            reuse_graph_replay_plan=True,
            post_attention_segment=policies["segment"],
            exact_rotary_dispatch=policies["rotary"],
            reuse_source_input_copy=policies["copy"] if active else True,
            frozen_joint_plan=policies["frozen"] if active else True,
            overlap_native_fit=policies["overlap"] if active else True,
            native_history_reader=False,
            input_norm_segment=prior.NORM or REUSE,
            reuse_norm_input=REUSE,
            source_tile_rows=capacity,
            **(dict(lt_fit_policy="direct", joint_incoming_projection="kv") if active else {}),
        )
    return result


def execute(adapter, refs, messages, profile, plan, method, arm, record, **kwargs):
    effective = (
        ("joint-reuse" if REUSE else "joint-previous")
        if method == "mosaickv_exact"
        else (arm + ("-reuse" if REUSE else ""))
    )
    value = prior.execute(
        adapter, refs, messages, profile, plan, method, effective, record, **kwargs
    )
    info, graph = value["joint_incoming_projection"], value["prompt_graph_reader"]
    if info["source_tile_rows"] != record["source_tile_rows"]:
        raise RuntimeError("declared source capacity did not propagate")
    if method == "mosaickv_exact":
        capacity = record["source_tile_rows"]
        if (
            graph["lt_workspace_peak_bytes"] != 2 * capacity * 4096 * 2 + 8192
            or graph["lt_matmul_dispatches"] != (32512 if capacity == 192 else 40640)
            or value["lt_source_conversion"]["added_patch_bytes"] > 2**20
            or not value["lt_source_conversion"]["parent_KV_bitwise"]
        ):
            raise RuntimeError("actual bounded geometry, words, corrections and workspace required")
    elif info["eligible_layers"] or info["transactions"]:
        raise RuntimeError("ineligible baseline must remain an explicit source-reader no-op")
    return value


def summary(rows, audited):
    old, new = (audited["mosaickv_exact", arm] for arm in ("joint-previous", "joint-192"))
    delta = new[driver.STORAGE[0]] - old[driver.STORAGE[0]]
    if (
        new[driver.STORAGE[1]] - old[driver.STORAGE[1]] != delta
        or new[driver.STORAGE[2]] - old[driver.STORAGE[2]] != delta + 2**20
        or delta
        != (
            new["lt_source_conversion"]["added_patch_bytes"]
            - old["lt_source_conversion"]["added_patch_bytes"]
        )
    ):
        raise ValueError(
            "all actual numerical corrections and extra source workspace must be charged"
        )
    return dict(
        ORIGINAL_SUMMARY(rows, audited),
        actual_native_correction_byte_delta=delta,
        additional_explicit_workspace_bytes=2**20,
        physical_storage={
            f"{m}:{a}": {k: v[k] for k in driver.STORAGE} for (m, a), v in audited.items()
        },
        independent_quality_N=0,
        whole_request_GPU_peak_certified=False,
    )


def worker(root, expected, index):
    global REUSE
    plan = driver.checked(root, expected)
    REUSE = plan["selected_common_input_reuse"]
    prior.NORM = plan["common_original_norm_replay"]
    prior.prior.POLICIES.update(plan["frozen_runtime_policies"])
    prior.prior.HISTORY = False
    prior.prior.prior.POLICIES.update(prior.prior.POLICIES)
    prior.prior.prior.prior.POLICIES.update(
        {k: v for k, v in prior.prior.POLICIES.items() if k != "overlap"}
    )
    prior.prior.prior.prior.prior.COMMON.update(
        {k: prior.prior.POLICIES[k] for k in ("segment", "rotary", "copy")}
    )
    ORIGINAL_WORKER(root, expected, index)


def main():
    correctness.configure()
    parent.WAIT, parent.JOB_NAME = WAIT, "mk-source-192"
    control.WAIT, control.SCRIPT = WAIT, SCRIPT
    control.CANDIDATE_ARM, control.POLICY = "joint-192", "kv"
    control.VARIANT, control.DOC = "consumed_corrected_source_192_reader", DOC
    control.ARMS, control.AUXILIARY_PROFILE_ARMS = ARMS, frozenset()
    control.prepare, control.configurations, control.execute, control.summary = (
        prepare,
        configurations,
        execute,
        summary,
    )
    control.worker, control.JOB_NAME = worker, "mk-source-192"
    control.main()


if __name__ == "__main__":
    main()
