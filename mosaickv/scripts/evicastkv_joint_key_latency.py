#!/usr/bin/env python3
"""Finite paired key-fold timing screen after successful A-B-A correctness."""

import json
import math
import random
import statistics
import subprocess
import traceback
from pathlib import Path

import evicastkv_direct_text_latency as direct
import evicastkv_joint_key_transaction as correctness
import evicastkv_joint_row_feasibility as parent
import evicastkv_lt_latency as native

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once
from mosaickv.evaluation.reservation_window import actual

SOURCE, driver = parent.SOURCE, parent.driver
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_joint_key_latency.sbatch"
WAIT = Path("/scratch/djy8hg/runs/evicastkv/joint-key-transaction-consumed-20261105-v1")
ORIGINAL_PREPARE = parent.prepare
ARMS = (
    ("mosaickv_exact", "graph"),
    ("mosaickv_exact", "graph-lt"),
    ("mosaickv_exact", "joint-keys"),
    ("prefixkv_reimpl", "graph"),
    ("prefixkv_reimpl", "graph-original"),
)
CANDIDATE_ARM, POLICY = "joint-keys", "keys"
VARIANT = "consumed_joint_key_paired_screen"
JOB_NAME = "mk-joint-K-time"
DOC = SOURCE / "mosaickv/docs/joint_key_latency.md"
GATE_FIELD = "joint_key_128_vector_parity_and_isolation"
AUXILIARY_PROFILE_ARMS = frozenset()
APPROVED_CONTROLLER_PATHS = {
    "mosaickv/scripts/evicastkv_joint_key_latency.py",
    "mosaickv/scripts/evicastkv_joint_key_transaction.py",
    "mosaickv/slurm/evicastkv_joint_key_latency.sbatch",
    "mosaickv/docs/joint_key_latency.md",
    "mosaickv/tests/unit/test_joint_key_latency.py",
}


def prepare(root, revision):
    gate = driver.read(WAIT / "analysis/decision.json")
    if not gate["complete"] or gate["failures"] or not gate[GATE_FIELD]:
        raise ValueError("complete positive joint-key A-B-A gate required")
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", gate["source_sha"], revision], cwd=SOURCE, text=True
    ).splitlines()
    if not set(changed) <= APPROVED_CONTROLLER_PATHS:
        raise ValueError("this timing-only stage forbids runtime changes from passed A-B-A")
    ORIGINAL_PREPARE(root, revision)


def configurations(records):
    return {
        (m, a): dict(
            records[m]["graph-lt" if a == CANDIDATE_ARM else a],
            graph_accounting="event_sum",
            reuse_patch_dispatch=True,
            lt_fit_policy="direct",
            joint_incoming_projection=POLICY,
        )
        if a == CANDIDATE_ARM
        else dict(
            records[m][a],
            graph_accounting="event_sum",
            reuse_patch_dispatch=True,
            **({"lt_fit_policy": "direct"} if a == "graph-lt" else {}),
        )
        for m, a in ARMS
    }


def execute(adapter, refs, messages, profile, plan, method, arm, record, **kwargs):
    value = direct.execute(adapter, refs, messages, profile, plan, method, record, **kwargs)
    if record.get("joint_incoming_projection", "none") != "none":
        # Keep every engagement check; an unaudited timing is not peak evidence.
        correctness.validate(value, True, require_audited=kwargs.get("audit", False))
        if not kwargs.get("audit", False) and value["peak_measurement_admissible"]:
            raise RuntimeError("unaudited timing may not claim an admissible peak")
        native.arena.validate_engagement(value, plan, method, record)
    else:
        native.validate(value, plan, method, record)
    return value


def summary(rows, audited):
    keys = [(r["method"], r["arm"], r["trial"]) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != {
        (m, a, t) for m, a in ARMS for t in range(-5, 3)
    }:
        raise ValueError("complete unique five-warmup/three-trial paired screen required")
    for row in rows:
        measured = row["measurement"]
        if (
            row["warmup"] is not (row["trial"] < 0)
            or measured["logits_bitwise"] is not True
            or measured["no_FullKV_fallback"] is not True
            or any(
                not math.isfinite(measured[k]) or measured[k] < 0
                for k in (
                    "total_wall_seconds",
                    "preprocessing_wall_seconds",
                    "prefill_construct_retire_wall_seconds",
                    "decode_wall_seconds",
                )
            )
            or measured["total_wall_seconds"] <= 0
        ):
            raise ValueError("finite complete parent-bitwise phase measurements required")
    means = {}
    phases = []
    for m, a in ARMS:
        selected = [
            r["measurement"] for r in rows if (r["method"], r["arm"]) == (m, a) and not r["warmup"]
        ]
        means[m, a] = statistics.mean(r["total_wall_seconds"] for r in selected)
        phases.append(
            dict(
                method=m,
                arm=a,
                **{
                    k: statistics.mean(r[k] for r in selected)
                    for k in (
                        "total_wall_seconds",
                        "preprocessing_wall_seconds",
                        "prefill_construct_retire_wall_seconds",
                        "decode_wall_seconds",
                    )
                },
            )
        )
    source = means["mosaickv_exact", CANDIDATE_ARM]
    ratios = {a: source / means[m, a] for m, a in ARMS if m == "prefixkv_reimpl"}
    candidate = audited["mosaickv_exact", CANDIDATE_ARM]
    storage = all(
        candidate[k] < audited[m, a][k]
        for m, a in ARMS
        if m == "prefixkv_reimpl"
        for k in driver.STORAGE
    )
    return dict(
        means=phases,
        candidate_ratios_vs_all_optimized_Prefix=ratios,
        smaller_at_all_three_decode_owned_boundaries=storage,
        four_percent_POINT_only=storage and max(ratios.values()) <= 0.96,
        formal_latency_acceptance=False,
        independent_quality_N=0,
        no_quality_or_larger_campaign_authorized=True,
        whole_request_GPU_peak_certified=False,
    )


def worker(root, expected, index):
    out = root / f"workers/{index}"
    out.mkdir(parents=True, exist_ok=False)
    result = dict(
        status="failed", failures=[], source_sha=clean_source(SOURCE), plan_sha256=expected
    )
    try:
        actual(at_worker_start=True)
        activation = driver.read(root / "release.json")
        if (
            not activation["profile_released"]
            or sha(Path(activation["parent_decision"])) != activation["parent_sha256"]
        ):
            raise ValueError("unchanged successful key-fold correctness required")
        plan, adapter, refs, messages, profile, _ = driver.setup(
            root,
            expected,
            index,
            manifest_overrides=dict(
                variant=VARIANT,
                methods=["mosaickv_exact", "prefixkv_reimpl"],
                actual_timing_protocol=dict(
                    warmups=5,
                    repetitions=3,
                    output_predictions=128,
                    randomized_paired_order=True,
                    complete_prefill_construction_capture_decode_retirement_included=True,
                    NOT_formal_latency_acceptance=True,
                    auxiliary_requests_per_arm=int(bool(AUXILIARY_PROFILE_ARMS)),
                    separate_auxiliary_profile_arms=sorted(AUXILIARY_PROFILE_ARMS),
                ),
            ),
        )
        records = driver.read(parent.PARENT / f"workers/{index}/study.json")["records"]
        configured, audited = configurations(records), {}
        for m, a in ARMS:
            record = configured[m, a]

            def verify(packets, wanted=record["packet_sha256"]):
                if direct.packet_signature(packets)["digest"] != wanted:
                    raise RuntimeError(
                        "original preconversion source/selected KV packet words changed"
                    )

            measured = execute(
                adapter,
                refs,
                messages,
                profile,
                plan,
                m,
                a,
                record,
                audit=True,
                final_validator=verify,
            )
            audited[m, a] = measured
            write_once(out / f"audited-{m}-{a}.json", measured)
        if AUXILIARY_PROFILE_ARMS:
            from mosaickv.evaluation.optimized_runtime_profile import collect

            for m, a in ARMS:

                def call(m=m, a=a):
                    return execute(adapter, refs, messages, profile, plan, m, a, configured[m, a])

                if (m, a) in AUXILIARY_PROFILE_ARMS:
                    folder = out / f"auxiliary-profile-{m}-{a}"
                    folder.mkdir()
                    write_once(folder / "profile.json", collect(adapter, call, folder))
                else:
                    call()  # Identical auxiliary count, never a timed repetition.
        rows, rng = [], random.Random(plan["seed"])
        for trial in range(-5, 3):
            order = list(ARMS)
            rng.shuffle(order)
            for place, (m, a) in enumerate(order):
                value = execute(adapter, refs, messages, profile, plan, m, a, configured[m, a])
                if any(value[k] != audited[m, a][k] for k in driver.STORAGE[:2]):
                    raise ValueError("timed initial/final bytes differ from audited request")
                row = dict(
                    method=m,
                    arm=a,
                    trial=trial,
                    warmup=trial < 0,
                    order_index=place,
                    measurement=value,
                    source_sha=plan["source_sha"],
                    plan_sha256=expected,
                )
                rows.append(row)
                with (out / "trials.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
        write_once(out / "profile.json", summary(rows, audited))
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
            four_percent_POINT_only=not failures
            and len(rows) == 2
            and all(r["four_percent_POINT_only"] for r in rows),
            formal_latency_acceptance=False,
            independent_quality_N=0,
            no_automatic_larger_or_quality_submission=True,
        ),
    )


def main():
    parent.WAIT, parent.SCRIPT = WAIT, SCRIPT
    parent.ONLY_LAYER, parent.JOB_NAME = None, JOB_NAME
    parent.STAGE = parent.VARIANT = VARIANT
    parent.DOC = DOC
    parent.EXTRA_PLAN = dict(
        warmups=5,
        repetitions=3,
        arms=ARMS,
        no_runtime_change_from_passed_key_transaction=True,
        acceptance_ratio_upper_bound=0.96,
        formal_gate_NOT_decided_by_screen=True,
        auxiliary_requests_per_arm=int(bool(AUXILIARY_PROFILE_ARMS)),
        auxiliary_profiles_not_latency_acceptance=True,
    )
    parent.EXTRA_BOUND_FILES = (WAIT / "analysis/decision.json",)
    parent.prepare, parent.analyze = prepare, analyze
    parent.controller.worker = worker
    parent.main()


if __name__ == "__main__":
    main()
