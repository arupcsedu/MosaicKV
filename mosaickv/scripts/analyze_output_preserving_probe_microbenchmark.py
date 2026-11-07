#!/usr/bin/env python3
"""Aggregate the one-identity V6 surrogate probe timing diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_prefixkv_decode_preflight import digest, write_once

REQUIRED_STAGES = frozenset({"surrogate_generation", "short_two_candidate_probe"})


def validate_probe_trace(
    stability: object,
    *,
    expected_strategy: str = "layer_capacity_vectorized_surrogate_beam",
) -> tuple[dict[str, object], ...]:
    """Fail closed unless the registered surrogate two-then-one schedule ran."""

    if not isinstance(stability, dict):
        raise ValueError("probe worker omitted stability trace")
    timings = stability.get("probe_phase_timings")
    if not isinstance(timings, list) or not timings:
        raise ValueError("probe worker omitted phase timings")
    stages = {item.get("stage") for item in timings if isinstance(item, dict)}
    if not stages >= REQUIRED_STAGES:
        raise ValueError("probe worker omitted a required surrogate phase")
    extended = "extended_one_candidate_probe" in stages
    expected_batches = 2 if extended else 1
    if (
        stability.get("candidate_strategy") != expected_strategy
        or stability.get("surrogate_candidate_count") != 8
        or stability.get("model_probed_candidate_count") != 2
        or stability.get("probe_batch_count") != expected_batches
        or len(stability.get("evaluated_candidates", ())) != 2
        or stability.get("benchmark_labels_accessed") is not False
        or stability.get("fullkv_prefill_reused") is not True
        or stability.get("second_prefill_performed") is not False
        or stability.get("probe_states_discarded_before_measured_decode") is not True
    ):
        raise ValueError("probe worker did not execute the registered surrogate schedule")
    short = [item for item in timings if item.get("stage") == "short_two_candidate_probe"]
    extended_rows = [
        item for item in timings if item.get("stage") == "extended_one_candidate_probe"
    ]
    if (
        len(short) != 1
        or short[0].get("realized_batch_size") != 2
        or short[0].get("horizon") != 2
        or len(extended_rows) > 1
        or (
            extended_rows
            and (
                extended_rows[0].get("realized_batch_size") != 1
                or extended_rows[0].get("horizon") != 8
            )
        )
    ):
        raise ValueError("probe worker recorded an invalid surrogate probe shape")
    return tuple(timings)


def analyze(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    plan_path = root / "plan.json"
    if digest(plan_path) != args.plan_sha256:
        raise ValueError("plan hash mismatch")
    plan = json.loads(plan_path.read_text())
    report: dict[str, object] = {
        "status": "failed",
        "classification": plan["classification"],
        "timing_diagnostic_only": True,
        "quality_evidence": False,
        "plan_sha256": args.plan_sha256,
        "rows": [],
        "missing_or_failed": [],
    }
    rows = report["rows"]
    missing = report["missing_or_failed"]
    assert isinstance(rows, list) and isinstance(missing, list)
    for job in plan["jobs"]:
        output = Path(job["output"])
        workers = sorted(output.glob("attempt-*/worker.json"))
        passed = [
            path
            for path in workers
            if json.loads(path.read_text()).get("status") == "passed"
        ]
        if len(passed) != 1:
            missing.append(
                {
                    "index": job["index"],
                    "batch_size": job["batch_size"],
                    "passed_workers": len(passed),
                }
            )
            continue
        worker = json.loads(passed[0].read_text())
        traces = worker.get("traces", {})
        if len(traces) != 1:
            raise ValueError("probe worker must expose exactly one completed trace")
        trace = json.loads(Path(next(iter(traces.values()))).read_text())
        stability = trace.get("short_horizon_logit_stability", {})
        timings = validate_probe_trace(
            stability, expected_strategy=str(job["candidate_strategy"])
        )
        aggregate = {
            field: sum(float(item.get(field, 0.0)) for item in timings)
            for field in (
                "candidate_construction_seconds",
                "cache_packing_seconds",
                "batch_state_construction_seconds",
                "layerwise_decode_seconds",
                "metric_reduction_seconds",
                "state_cleanup_seconds",
                "total_seconds",
            )
        }
        rows.append(
            {
                "index": job["index"],
                "candidate_strategy": job["candidate_strategy"],
                "probe_batch_count": stability.get("probe_batch_count"),
                "candidate_count": len(stability.get("evaluated_candidates", ())),
                "selected_candidate_id": stability.get("selected_candidate_id"),
                "used_fail_closed_anchor": stability.get(
                    "used_fail_closed_scalar_anchor"
                ),
                "phase_totals": aggregate,
                "phase_records": timings,
                "worker_wall_seconds": worker.get("worker_wall_seconds"),
            }
        )
    complete = not missing and len(rows) == 1
    wall_seconds = float(rows[0]["worker_wall_seconds"]) if complete else float("inf")
    report["maximum_worker_wall_seconds"] = plan["maximum_worker_wall_seconds"]
    report["runtime_gate_passed"] = (
        complete and wall_seconds <= float(plan["maximum_worker_wall_seconds"])
    )
    surrogate_seconds = (
        sum(
            float(item.get("total_seconds", 0.0))
            for item in rows[0]["phase_records"]
            if item.get("stage") == "surrogate_generation"
        )
        if complete
        else float("inf")
    )
    report["maximum_surrogate_generation_seconds"] = plan[
        "maximum_surrogate_generation_seconds"
    ]
    report["surrogate_generation_seconds"] = surrogate_seconds
    report["surrogate_generation_gate_passed"] = surrogate_seconds <= float(
        plan["maximum_surrogate_generation_seconds"]
    )
    report["status"] = "passed" if complete else "incomplete"
    report["paper_claim_authorized"] = False
    report["canary_submission_authorized"] = bool(
        report["runtime_gate_passed"]
        and report["surrogate_generation_gate_passed"]
    )
    write_once(root / "analysis/report.json", report)
    print(json.dumps(report, indent=2))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    analyze(parser.parse_args())
