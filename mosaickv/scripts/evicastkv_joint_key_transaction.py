#!/usr/bin/env python3
"""Correctness/isolation first; no timing promotion or fresh examples."""

import traceback
from pathlib import Path

import evicastkv_direct_text_latency as direct
import evicastkv_joint_row_feasibility as parent

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once
from mosaickv.evaluation.reservation_window import actual

SOURCE, driver = parent.SOURCE, parent.driver
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_joint_key_transaction.sbatch"
WAIT = Path("/scratch/djy8hg/runs/evicastkv/joint-all-layer-compatibility-consumed-20261105-v1")
ORIGINAL_PREPARE = parent.prepare
SEQUENCE = ("graph-lt", "joint-keys", "graph-lt")


def prepare(root, revision):
    gate = driver.read(WAIT / "analysis/decision.json")
    if not gate["complete"] or gate["failures"] or not gate["all_layer_exact_both_inputs"]:
        raise ValueError("complete positive all-layer frozen-kernel gate required")
    ORIGINAL_PREPARE(root, revision)


def validate(measured, active, *, require_audited=True):
    if (
        not measured["logits_bitwise"]
        or not measured["no_FullKV_fallback"]
        or (require_audited and not measured["peak_measurement_admissible"])
    ):
        raise ValueError("audited complete same-parent trajectory and actual tensor peak required")
    graph, info = measured["prompt_graph_reader"], measured["joint_incoming_projection"]
    if (
        graph["captures"] != 64
        or graph["replays"] != 8128
        or graph["graph_private_live_bytes"]
        or graph["active_workspace_bytes"]
        or not graph["retired"]
    ):
        raise ValueError(
            "exactly the original graph count and complete workspace retirement required"
        )
    if active:
        if (
            info["policy"] != "keys"
            or info["eligible_layers"] != 32
            or info["transactions"]
            != [
                dict(
                    key_calls=4064,
                    folded_keys=4032,
                    original_cold_keys=32,
                    pending=False,
                    all_value_projections_original=True,
                )
            ]
            or graph["lt_workspace_peak_bytes"] != 2**21 + 8192
        ):
            raise ValueError("all 4032 warm key projections must actually share the prompt graph")
    elif info["policy"] != "none" or info["transactions"] or info["eligible_layers"]:
        raise ValueError("original control was silently modified")
    if not info["value_projection_unchanged"] or not info["no_placeholder_values"]:
        raise ValueError("no deferred or placeholder values are allowed in the key-only gate")


def worker(root, expected, index):
    out = root / f"workers/{index}"
    out.mkdir(parents=True, exist_ok=False)
    result = dict(
        status="failed", failures=[], source_sha=clean_source(SOURCE), plan_sha256=expected
    )
    try:
        actual(at_worker_start=True)
        from mosaickv.evaluation.joint_incoming_projection import gpu_decoder_controls

        write_once(out / "decoder_controls.json", gpu_decoder_controls())
        activation = driver.read(root / "release.json")
        if (
            not activation["profile_released"]
            or sha(Path(activation["parent_decision"])) != activation["parent_sha256"]
        ):
            raise ValueError("unchanged all-layer gate required")
        plan, adapter, refs, messages, profile, _ = driver.setup(
            root,
            expected,
            index,
            manifest_overrides=dict(
                variant="consumed_joint_key_transaction_A_B_A",
                methods=["mosaickv_exact"],
                actual_timing_protocol=dict(
                    instrumented_A_B_A=True,
                    output_predictions=128,
                    sequence=SEQUENCE,
                    NOT_latency_acceptance=True,
                ),
            ),
        )
        records = driver.read(parent.PARENT / f"workers/{index}/study.json")["records"]
        observations = []
        for order, arm in enumerate(SEQUENCE):
            record = dict(
                records["mosaickv_exact"]["graph-lt"],
                graph_accounting="event_sum",
                lt_fit_policy="direct",
                reuse_patch_dispatch=True,
                joint_incoming_projection="keys" if arm == "joint-keys" else "none",
            )
            measured = direct.execute(
                adapter, refs, messages, profile, plan, "mosaickv_exact", record, audit=True
            )
            validate(measured, arm == "joint-keys")
            row = dict(order=order, arm=arm, measurement=measured)
            write_once(out / f"audited-{order}-{arm}.json", row)
            observations.append(row)
        original = [x["measurement"] for x in observations if x["arm"] == "graph-lt"]
        fields = ("initial_encoded_bytes", "final_encoded_bytes", "decode_owned_peak_bytes")
        if any(original[0][k] != original[1][k] for k in fields):
            raise ValueError("A-B-A original storage boundaries changed")
        write_once(
            out / "profile.json",
            dict(
                records=observations,
                A_B_A_isolation=True,
                no_full_request_timing_claim=True,
                independent_quality_N=0,
            ),
        )
        result["status"] = "completed"
    except Exception as error:
        result["failures"].append(f"{type(error).__name__}: {error}")
        result["traceback"] = traceback.format_exc()
    driver.finish_worker(out, result)


def analyze(root, expected):
    plan = driver.checked(root, expected)
    rows, failures = driver.load_producers(root, plan, "profile.json")
    write_once(
        root / "analysis/decision.json",
        dict(
            complete=not failures,
            failures=failures,
            rows=rows,
            source_sha=plan["source_sha"],
            plan_sha256=expected,
            joint_key_128_vector_parity_and_isolation=not failures and len(rows) == 2,
            no_timing_or_quality_acceptance=True,
            independent_quality_N=0,
            whole_request_GPU_peak_certified=False,
            automatic_larger_or_quality_submission=False,
        ),
    )


def main():
    parent.WAIT, parent.SCRIPT = WAIT, SCRIPT
    parent.ONLY_LAYER, parent.JOB_NAME = None, "mk-joint-keys"
    parent.STAGE = parent.VARIANT = "consumed_joint_key_transaction_A_B_A"
    parent.DOC = SOURCE / "mosaickv/docs/joint_key_transaction.md"
    parent.EXTRA_PLAN = dict(
        sequence=SEQUENCE,
        only_keys_folded=True,
        extra_incoming_buffer_bytes=8192,
        no_placeholder_or_deferred_values=True,
        correctness_before_any_latency_benchmark=True,
    )
    parent.EXTRA_BOUND_FILES = (WAIT / "analysis/decision.json",)
    parent.prepare, parent.analyze = prepare, analyze
    parent.controller.worker = worker
    parent.main()


if __name__ == "__main__":
    main()
