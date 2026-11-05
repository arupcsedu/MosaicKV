#!/usr/bin/env python3
"""Consumed-input exact row-construction experiment; no automatic larger run."""

import argparse
import json
import os
import random
import traceback
from pathlib import Path

import evicastkv_lt_direct_constructor as base

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE, driver = base.SOURCE, base.driver
PARENT = base.PARENT
NEGATIVE = Path("/scratch/djy8hg/runs/evicastkv/lt-direct-constructor-consumed-20261105-v1")
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_selective_text_constructor.sbatch"
CANDIDATE = ("mosaickv_exact", "graph-selective-text")
ARMS = (
    ("mosaickv_exact", "graph"),
    CANDIDATE,
    ("prefixkv_reimpl", "graph"),
    ("prefixkv_reimpl", "graph-original"),
)


def variants(records):
    return [
        (
            CANDIDATE[0],
            CANDIDATE[1],
            dict(records["mosaickv_exact"]["graph"], selective_text_rows=True),
        )
    ]


def prepare(root, revision):
    if root.exists() or clean_source(SOURCE) != revision:
        raise ValueError("new immutable root and clean source required")
    decision = driver.read(NEGATIVE / "analysis/decision.json")
    if not decision["complete"] or decision["screen_pass"]:
        raise ValueError("preserved complete failed direct-constructor screen required")
    old = driver.read(PARENT / "plan.json")
    bound = dict(old["bound_files"])
    paths = [
        PARENT / "plan.json",
        PARENT / "analysis/decision.json",
        NEGATIVE / "analysis/decision.json",
        SOURCE / "mosaickv/docs/selective_text_constructor_20261105.md",
    ]
    paths += [PARENT / f"workers/{i}/study.json" for i in (0, 1)]
    bound.update({str(p): sha(p) for p in paths})
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=revision,
            bound_files=bound,
            stage="exact_selective_text_constructor_screen",
            warmups=5,
            repetitions=3,
            screen_NOT_acceptance=True,
            no_fresh_identities=True,
            independent_quality_N=0,
            automatic_quality_submission=False,
            automatic_formal_submission=False,
            original_selected_KV_and_GPU_storage_unchanged=True,
            CPU_index_metadata_reported=True,
        ),
    )
    base.common.stage(root, "cpu")


def summary(rows):
    # Reuse the strict completeness/numerical/timing validator. No inferential
    # gate is decided from three repetitions; the <=0.96 formal gate is intact.
    old = base.screen_summary(rows, arms=ARMS, candidate=CANDIDATE)
    means = {(r["method"], r["arm"]): r["mean"] for r in old["summaries"]}
    candidate = means[CANDIDATE]
    ratios = [candidate / means["prefixkv_reimpl", arm] for arm in ("graph", "graph-original")]
    return dict(
        summaries=old["summaries"],
        paired_ratios=old["all_paired_ratios"],
        mean_ratios_against_PrefixKV=ratios,
        construction_progress_vs_original=bool(candidate < means["mosaickv_exact", "graph"]),
        four_percent_POINT_only=bool(max(ratios) <= 0.96),
        formal_latency_acceptance=False,
        independent_quality_N=0,
        full_twenty_repeat_adjusted_ratio_gate_still_required=True,
    )


def execute(adapter, refs, messages, profile, plan, method, record, **kwargs):
    value = base.ORIGINAL_EXECUTE(adapter, refs, messages, profile, plan, method, record, **kwargs)
    if record.get("selective_text_rows", False):
        proof = value["selective_text_construction"]
        if (
            not proof["enabled"]
            or proof["source_layers"] < 1
            or not 0 < proof["projected_rows"] < proof["original_full_rows"]
            or proof["cpu_metadata_hits"] != proof["source_layers"]
            or proof["cpu_metadata_misses"] != 0
        ):
            raise RuntimeError(
                "actual reduced aligned projection and constructor-owned indices required"
            )
    return value


def worker(root, expected, index):
    out = root / f"workers/{index}"
    out.mkdir(parents=True, exist_ok=False)
    result = dict(
        status="failed", failures=[], source_sha=clean_source(SOURCE), plan_sha256=expected
    )
    try:
        plan, adapter, refs, messages, profile, _ = driver.setup(
            root,
            expected,
            index,
            manifest_overrides=dict(
                variant="consumed_exact_selective_text_constructor",
                methods=["mosaickv_exact", "prefixkv_reimpl"],
                actual_timing_protocol=dict(
                    output_predictions=128, warmups=5, repetitions=3, not_acceptance=True
                ),
            ),
        )
        records = driver.read(PARENT / f"workers/{index}/study.json")["records"]
        for method, arm, candidate in variants(records):

            def verify(packets, expected=candidate["packet_sha256"]):
                actual = base.common.previous.direct.packet_signature(packets)["digest"]
                if actual != expected:
                    raise RuntimeError("selective construction changed parent cache tensor words")

            audited = execute(
                adapter,
                refs,
                messages,
                profile,
                plan,
                method,
                candidate,
                audit=True,
                final_validator=verify,
            )
            if any(audited[k] != candidate["audited"][k] for k in driver.STORAGE):
                raise RuntimeError(
                    "selective construction changed initial/final/decode-owned peak GPU storage"
                )
            records[method][arm] = dict(candidate, audited=audited)
            write_once(out / f"audited-{method}-{arm}.json", audited)
        rows = []
        rng = random.Random(plan["seed"])
        for trial in range(-5, 3):
            placement = list(ARMS)
            rng.shuffle(placement)
            for method, arm in placement:
                record = records[method][arm]
                value = execute(adapter, refs, messages, profile, plan, method, record)
                if any(value[k] != record["audited"][k] for k in driver.STORAGE[:2]):
                    raise RuntimeError("persistent bytes differ from audited same-input control")
                row = dict(
                    method=method,
                    arm=arm,
                    trial=trial,
                    warmup=trial < 0,
                    measurement=value,
                    source_sha=plan["source_sha"],
                    plan_sha256=expected,
                )
                rows.append(row)
                with (out / "trials.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
        write_once(out / "screen.json", summary(rows))
        result["status"] = "completed"
    except Exception as error:
        result["failures"].append(f"{type(error).__name__}: {error}")
        result["traceback"] = traceback.format_exc()
    driver.finish_worker(out, result)


def analyze(root, expected, action):
    plan = driver.checked(root, expected)
    failures, records = [], []
    if action == "cpu_analysis":
        try:
            driver.cpu_proof(root, plan)
        except Exception as error:
            failures.append(str(error))
    else:
        records, failures = driver.load_producers(root, plan, "screen.json")
    write_once(
        root / ("cpu-analysis" if action == "cpu_analysis" else "analysis") / "decision.json",
        dict(
            complete=not failures,
            failures=failures,
            rows=records,
            source_sha=plan["source_sha"],
            plan_sha256=expected,
            independent_quality_N=0,
            formal_latency_gate=False,
            no_reader_promoted=True,
            no_automatic_formal_or_quality_submission=True,
        ),
    )
    if action == "cpu_analysis" and not failures:
        base.common.stage(root, "screen")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("prepare", "cpu", "cpu_analysis", "screen", "screen_analysis")
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-sha")
    parser.add_argument("--plan-sha")
    args = parser.parse_args()
    base.SCRIPT = base.common.SCRIPT = SCRIPT
    base.common.submit = base.submit
    if args.action == "prepare":
        prepare(args.root, args.source_sha)
    elif args.action == "cpu":
        driver.pair.common.previous.cpu(args.root, args.plan_sha)
    elif args.action == "screen":
        worker(args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    else:
        analyze(args.root, args.plan_sha, args.action)


if __name__ == "__main__":
    main()
