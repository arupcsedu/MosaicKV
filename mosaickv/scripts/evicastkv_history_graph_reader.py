#!/usr/bin/env python3
"""Bounded private history-graph correctness followed by a separate point screen."""

import argparse
import json
import sys
import traceback
from pathlib import Path

import evicastkv_history_phase_fastpath as prior

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

control, SOURCE, parent, driver = prior.control, prior.SOURCE, prior.parent, prior.driver
WAIT = prior.WAIT
PREVIOUS = Path("/scratch/djy8hg/runs/evicastkv/history-phase-fastpath-consumed-20261105-v1")
DEBUG_ROOT = Path("/scratch/djy8hg/runs/evicastkv/history-graph-correctness-consumed-20261105-v1")
OPERATOR = Path("/scratch/djy8hg/runs/evicastkv/full-frozen192-operator-20261105-v1")
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_history_graph_reader.sbatch"
DOC = SOURCE / "mosaickv/docs/native_history_graph.md"
ARMS = (
    ("mosaickv_exact", "joint-192"),
    ("mosaickv_exact", "joint-history-graph"),
    ("prefixkv_reimpl", "graph"),
    ("prefixkv_reimpl", "graph-history-graph"),
    ("prefixkv_reimpl", "graph-original"),
    ("prefixkv_reimpl", "graph-original-history-graph"),
)
DEBUG_ARMS = (ARMS[0], ARMS[1], ARMS[3], ARMS[5])
PREPARE_SCREEN = False
DEBUG_VARIANT = "consumed_immutable_history_graph_correctness"
ORIGINAL_WORKER, ORIGINAL_SUMMARY = prior.prior.ORIGINAL_WORKER, control.summary


def require_debug(value):
    if (
        not value.get("complete")
        or value.get("failures")
        or len(value.get("rows", [])) != 2
        or not all(row.get("debug_controls_passed") is True for row in value["rows"])
    ):
        raise ValueError("both complete exact history-graph debug controls required")


def prepare(root, revision):
    if root.exists() or clean_source(SOURCE) != revision:
        raise ValueError("new immutable root and clean source required")
    decision = driver.read(PREVIOUS / "analysis/decision.json")
    if not decision["complete"] or decision["failures"]:
        raise ValueError("complete prior history numerical and latency result required")
    old = driver.read(PREVIOUS / "plan.json")
    bound = dict(old["bound_files"])
    paths = [DOC, PREVIOUS / "plan.json", PREVIOUS / "analysis/decision.json"]
    if PREPARE_SCREEN:
        require_debug(driver.read(DEBUG_ROOT / "analysis/decision.json"))
        paths += [DEBUG_ROOT / "plan.json", DEBUG_ROOT / "analysis/decision.json"]
    bound.update({str(path): sha(path) for path in paths})
    serial = str(driver.read(OPERATOR / "manifest.json")["job_id"])
    if not serial.isdecimal():
        raise ValueError("finite preceding operator dependency required")
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=revision,
            bound_files=bound,
            stage="consumed_immutable_history_graph",
            correctness_only=not PREPARE_SCREEN,
            arms=ARMS if PREPARE_SCREEN else DEBUG_ARMS,
            warmups=5 if PREPARE_SCREEN else 0,
            repetitions=3 if PREPARE_SCREEN else 1,
            history_reader_graph_only=True,
            extra_GPU_tensor_bytes_allowed=0,
            expected_history_captures=448,
            expected_history_replays=7104,
            maximum_live_history_graphs=64,
            original_thresholds_and_parent_words=True,
            independent_quality_N=0,
            no_automatic_fresh_or_quality_submission=True,
        ),
    )
    cpu = parent.submit(root, "cpu")
    release = parent.submit(root, "release", f"{cpu}:{serial}")
    write_once(
        root / "cpu_submission.json", dict(cpu=cpu, release=release, serialized_after=serial)
    )


def configurations(records):
    base, result = prior.configurations(records), {}
    for method, arm in ARMS:
        active = arm.endswith("-history-graph")
        previous = (
            ("joint-fast-history" if active else "joint-192")
            if method == "mosaickv_exact"
            else (arm.removesuffix("-history-graph") + ("-fast-history" if active else ""))
        )
        result[method, arm] = dict(base[method, previous], native_history_graph=active)
    return result


def execute(adapter, refs, messages, profile, plan, method, arm, record, **kwargs):
    active = record["native_history_graph"]
    previous = (
        ("joint-fast-history" if active else "joint-192")
        if method == "mosaickv_exact"
        else (arm.removesuffix("-history-graph") + ("-fast-history" if active else ""))
    )
    value = prior.execute(
        adapter, refs, messages, profile, plan, method, previous, record, **kwargs
    )
    info = value["common_native_history"]
    report = info["report"]["immutable_history_graph"] if info["enabled"] else dict(enabled=False)
    if report["enabled"] is not active:
        raise RuntimeError("history graph did not engage as configured")
    if active and (
        report["captures"] != 448
        or report["retired_graphs"] != 448
        or report["replays"] != 7104
        or report["maximum_live_graphs"] > 64
        or report["live_graphs"]
        or not report["retired"]
        or report["warm_calls"] != 448
        or report["additional_GPU_tensor_bytes"]
    ):
        raise RuntimeError("complete bounded real history replay and exact retirement required")
    return value


def summary(rows, audited):
    return dict(
        ORIGINAL_SUMMARY(rows, audited),
        physical_storage={
            f"{m}:{a}": {key: v[key] for key in driver.STORAGE} for (m, a), v in audited.items()
        },
        independent_quality_N=0,
        whole_request_GPU_peak_certified=False,
        no_fresh_or_larger_campaign_authorized=True,
    )


def debug_worker(root, expected, index):
    out = root / f"workers/{index}"
    out.mkdir(parents=True, exist_ok=False)
    result = dict(
        status="failed", failures=[], source_sha=clean_source(SOURCE), plan_sha256=expected
    )
    try:
        control.actual(at_worker_start=True)
        plan, adapter, refs, messages, profile, _ = driver.setup(
            root,
            expected,
            index,
            manifest_overrides=dict(
                variant=DEBUG_VARIANT,
                methods=["mosaickv_exact", "prefixkv_reimpl"],
                actual_timing_protocol=dict(
                    audited_requests_per_arm=1,
                    nonaudited_requests_per_arm=1,
                    output_predictions=128,
                    instrumented_NOT_latency_acceptance=True,
                ),
            ),
        )
        records = configurations(
            driver.read(parent.PARENT / f"workers/{index}/study.json")["records"]
        )
        rows = []
        for method, arm in plan["arms"]:
            record = records[method, arm]

            def verify(packets, wanted=record["packet_sha256"]):
                if control.direct.packet_signature(packets)["digest"] != wanted:
                    raise RuntimeError("original preconversion source/selected words changed")

            audited = execute(
                adapter,
                refs,
                messages,
                profile,
                plan,
                method,
                arm,
                record,
                audit=True,
                final_validator=verify,
            )
            measured = execute(adapter, refs, messages, profile, plan, method, arm, record)
            if any(audited[key] != measured[key] for key in driver.STORAGE[:2]):
                raise RuntimeError("debug storage changed when audit was disabled")
            row = dict(method=method, arm=arm, audited=audited, uninstrumented=measured)
            write_once(out / f"audited-{method}-{arm}.json", row)
            rows.append(row)
        write_once(
            out / "profile.json",
            dict(
                debug_controls_passed=True,
                records=rows,
                formal_latency_acceptance=False,
                independent_quality_N=0,
            ),
        )
        result["status"] = "completed"
    except Exception as error:
        result["failures"].append(f"{type(error).__name__}: {error}")
        result["traceback"] = traceback.format_exc()
    driver.finish_worker(out, result)


def worker(root, expected, index):
    plan = driver.checked(root, expected)
    prior.prior.ORIGINAL_WORKER = debug_worker if plan["correctness_only"] else ORIGINAL_WORKER
    prior.prior.worker(root, expected, index)


def analyze(root, expected):
    plan = driver.checked(root, expected)
    rows, failures = driver.load_producers(root, plan, "profile.json")
    complete = not failures and len(rows) == 2
    value = dict(
        complete=complete,
        failures=failures,
        rows=rows,
        source_sha=plan["source_sha"],
        plan_sha256=expected,
        formal_latency_acceptance=False,
        independent_quality_N=0,
        no_automatic_fresh_or_quality_submission=True,
    )
    if not plan["correctness_only"]:
        value["four_percent_POINT_only"] = complete and all(
            r["four_percent_POINT_only"] for r in rows
        )
    write_once(root / "analysis/decision.json", value)
    print(json.dumps(dict(complete=complete, failures=failures)))


def main():
    global PREPARE_SCREEN
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--screen", action="store_true")
    flags, rest = parser.parse_known_args()
    PREPARE_SCREEN, sys.argv = flags.screen, [sys.argv[0], *rest]
    prior.prior.correctness.configure()
    parent.WAIT, parent.JOB_NAME = WAIT, "mk-history-graph"
    control.WAIT, control.SCRIPT = WAIT, SCRIPT
    control.CANDIDATE_ARM, control.POLICY = "joint-history-graph", "kv"
    control.VARIANT, control.DOC = "consumed_immutable_history_graph", DOC
    control.ARMS, control.AUXILIARY_PROFILE_ARMS = ARMS, frozenset()
    control.prepare, control.configurations, control.execute, control.summary = (
        prepare,
        configurations,
        execute,
        summary,
    )
    control.worker, control.analyze, control.JOB_NAME = worker, analyze, "mk-history-graph"
    control.main()


if __name__ == "__main__":
    main()
