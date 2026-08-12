#!/usr/bin/env python3
"""Analyze the isolated Choice-C batched-compute latency A/B experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
from analyze_choice_c_latency_factorial import (
    TIMING_FIELDS,
    TRACE_PARITY_FIELDS,
    _bootstrap_mean_interval,
    _result_payload,
    _telemetry_is_clean,
)

RUN_PATTERN = re.compile(
    r"^choice-c-compute-(reference|optimized)-round([01])-rank([01])-[0-9a-f]{10}$"
)
SELECTION_IDENTITY_FIELDS = (
    "node_id",
    "selected",
    "selection_rank",
    "cost",
    "reason",
)
SELECTION_SCORE_FIELDS = ("marginal_gain", "marginal_gain_per_cost")


def _semantic_trace_payload(record: dict[str, Any]) -> str:
    """Hash semantic trace fields while handling selection evidence separately."""

    payload: list[Any] = []
    for field in TRACE_PARITY_FIELDS:
        if field == "selected_blocks":
            continue
        value = record.get(field)
        if field == "forecast_statistics" and isinstance(value, dict):
            value = {key: item for key, item in value.items() if key != "timing"}
        payload.append(value)
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def _selection_records(record: dict[str, Any]) -> list[dict[str, Any]]:
    evidence = record.get("selection_decision_evidence")
    if not isinstance(evidence, dict) or evidence.get("schema_version") != 1:
        raise RuntimeError("trace is missing selection decision evidence schema version 1")
    records = evidence.get("records")
    if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
        raise RuntimeError("selection decision evidence records must be an object array")
    if evidence.get("record_count") != len(records):
        raise RuntimeError("selection decision evidence count does not match its records")
    for node_id, item in enumerate(records):
        if item.get("node_id") != node_id:
            raise RuntimeError("selection decision evidence must cover ordered node IDs")
        for field in (*SELECTION_IDENTITY_FIELDS, *SELECTION_SCORE_FIELDS):
            if field not in item:
                raise RuntimeError(f"selection decision evidence is missing {field}")
        for field in SELECTION_SCORE_FIELDS:
            value = item[field]
            if not isinstance(value, int | float) or not math.isfinite(float(value)):
                raise RuntimeError(f"selection decision {field} must be finite")
    return records


def _selection_parity(
    records_by_sample: dict[str, list[list[dict[str, Any]]]],
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
    expected_replicas: int = 4,
) -> dict[str, Any]:
    """Compare decision identity exactly and floating scores within tolerance."""

    identity_parity = True
    score_parity = True
    comparison_count = 0
    non_bitwise_score_count = 0
    maximum_absolute_difference = {field: 0.0 for field in SELECTION_SCORE_FIELDS}
    maximum_relative_difference = {field: 0.0 for field in SELECTION_SCORE_FIELDS}
    worst: dict[str, Any] | None = None
    for sample_id, replicas in sorted(records_by_sample.items()):
        if len(replicas) != expected_replicas:
            raise RuntimeError(
                f"{sample_id} expected {expected_replicas} selection evidence replicas, "
                f"found {len(replicas)}"
            )
        reference = replicas[0]
        reference_identity = [
            tuple(item[field] for field in SELECTION_IDENTITY_FIELDS) for item in reference
        ]
        for replica in replicas[1:]:
            if len(replica) != len(reference):
                identity_parity = False
                score_parity = False
                continue
            identity = [
                tuple(item[field] for field in SELECTION_IDENTITY_FIELDS) for item in replica
            ]
            identity_parity = identity_parity and identity == reference_identity
            for node_id, (expected, observed) in enumerate(zip(reference, replica, strict=True)):
                for field in SELECTION_SCORE_FIELDS:
                    left = float(expected[field])
                    right = float(observed[field])
                    difference = abs(left - right)
                    scale = max(abs(left), abs(right), absolute_tolerance)
                    relative = difference / scale
                    comparison_count += 1
                    if difference:
                        non_bitwise_score_count += 1
                    if difference > maximum_absolute_difference[field]:
                        maximum_absolute_difference[field] = difference
                        maximum_relative_difference[field] = relative
                        worst = {
                            "sample_id": sample_id,
                            "node_id": node_id,
                            "field": field,
                            "reference": left,
                            "observed": right,
                            "absolute_difference": difference,
                            "relative_difference": relative,
                        }
                    score_parity = score_parity and math.isclose(
                        left,
                        right,
                        rel_tol=relative_tolerance,
                        abs_tol=absolute_tolerance,
                    )
    return {
        "identity_parity": identity_parity,
        "score_parity": score_parity,
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "score_comparison_count": comparison_count,
        "non_bitwise_score_count": non_bitwise_score_count,
        "maximum_absolute_difference": maximum_absolute_difference,
        "maximum_relative_difference_at_maximum_absolute": maximum_relative_difference,
        "worst_observation": worst,
    }


def _paired_effect(trials: pd.DataFrame, metric: str) -> dict[str, Any]:
    keys = ["round", "sample_id"]
    reference = trials.loc[trials["compute_mode"] == "reference", [*keys, metric]].rename(
        columns={metric: "reference"}
    )
    optimized = trials.loc[trials["compute_mode"] == "optimized", [*keys, metric]].rename(
        columns={metric: "optimized"}
    )
    paired = reference.merge(optimized, on=keys, how="inner", validate="one_to_one")
    if len(paired) != 8:
        raise RuntimeError(f"{metric} expected 8 paired samples, found {len(paired)}")
    reductions = (paired["reference"] - paired["optimized"]).to_numpy(dtype=float)
    low, high = _bootstrap_mean_interval(reductions)
    reference_mean = float(paired["reference"].mean())
    reduction = float(reductions.mean())
    round_effects: dict[str, float] = {}
    for round_index, rows in paired.groupby("round", sort=True):
        round_reference = float(rows["reference"].mean())
        round_optimized = float(rows["optimized"].mean())
        round_effects[str(round_index)] = (round_reference - round_optimized) / round_reference
    return {
        "pair_count": len(paired),
        "reference_mean_seconds": reference_mean,
        "optimized_mean_seconds": float(paired["optimized"].mean()),
        "mean_reduction_seconds": reduction,
        "relative_reduction": reduction / reference_mean,
        "bootstrap_95_reduction_seconds": [low, high],
        "relative_reduction_by_round": round_effects,
    }


def _binding_is_clean(root: Path) -> tuple[bool, int]:
    records = [
        json.loads(path.read_text())
        for path in sorted((root / "gpu_binding").glob("job-*/*/rank-*.json"))
    ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        key = (str(record.get("job_id")), str(record.get("step_id")))
        grouped.setdefault(key, []).append(record)
    clean = bool(
        len(records) == 4
        and len(grouped) == 2
        and all(
            len(items) == 2
            and {int(item["rank"]) for item in items} == {0, 1}
            and len({str(item["visible_gpu_uuid"]) for item in items}) == 2
            for items in grouped.values()
        )
    )
    return clean, len(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--required-speedup", default=0.03, type=float)
    parser.add_argument("--selection-score-atol", default=1e-12, type=float)
    parser.add_argument("--selection-score-rtol", default=1e-12, type=float)
    args = parser.parse_args()
    if not 0 < args.required_speedup < 1:
        raise ValueError("required-speedup must lie in (0, 1)")
    if args.selection_score_atol < 0 or args.selection_score_rtol < 0:
        raise ValueError("selection score tolerances must be nonnegative")

    root = args.output_root.resolve()
    runs: list[tuple[Path, str, int, int]] = []
    for path in sorted(root.glob("choice-c-compute-*")):
        if not path.is_dir() or "warmup" in path.name:
            continue
        match = RUN_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        mode, round_text, rank_text = match.groups()
        runs.append((path, mode, int(round_text), int(rank_text)))
    counts = {mode: sum(item[1] == mode for item in runs) for mode in ("reference", "optimized")}
    if counts != {"reference": 2, "optimized": 2}:
        raise RuntimeError(f"compute A/B coverage is incomplete: {counts}")

    result_payloads: dict[str, set[tuple[Any, ...]]] = {}
    semantic_trace_payloads: dict[str, set[str]] = {}
    raw_selection_catalogs: dict[str, set[str]] = {}
    selection_records: dict[str, list[list[dict[str, Any]]]] = {}
    trial_rows: list[dict[str, Any]] = []
    diagnostics_disabled = True
    for run, mode, round_index, rank in runs:
        frame = pd.read_parquet(run / "aggregate.parquet")
        required_timing = (
            "compression_time",
            "decode_time",
            "end_to_end_time",
            "prefill_time",
            "ttft",
        )
        if (
            len(frame) != 4
            or set(frame["status"]) != {"completed"}
            or frame[list(required_timing)].isna().any(axis=None)
            or frame[list(required_timing)].isin([math.inf, -math.inf]).any(axis=None)
        ):
            raise RuntimeError(f"run has incomplete or nonfinite timing rows: {run.name}")
        manifest = json.loads((run / "manifest.json").read_text())
        diagnostics_disabled = bool(
            diagnostics_disabled and not manifest["resolved_config"]["diagnostics"]["enabled"]
        )
        traces: dict[str, dict[str, Any]] = {}
        for path in sorted((root / "traces" / run.name).glob("*.json")):
            record = json.loads(path.read_text())
            sample_id = str(record["sample_id"])
            if sample_id in traces:
                raise RuntimeError(f"duplicate trace for {run.name}/{sample_id}")
            traces[sample_id] = record
            semantic_trace_payloads.setdefault(sample_id, set()).add(
                _semantic_trace_payload(record)
            )
            raw_selection_catalogs.setdefault(sample_id, set()).add(
                json.dumps(record.get("selected_blocks"), sort_keys=True, separators=(",", ":"))
            )
            selection_records.setdefault(sample_id, []).append(_selection_records(record))
            if record.get("fidelity_diagnostic") != {"status": "disabled"}:
                diagnostics_disabled = False
            if record.get("forecast_quality_diagnostic") != {"status": "disabled"}:
                diagnostics_disabled = False
        if set(traces) != set(frame["sample_id"]):
            raise RuntimeError(f"trace/result sample mismatch: {run.name}")
        for _index, row in frame.iterrows():
            sample_id = str(row["sample_id"])
            result_payloads.setdefault(sample_id, set()).add(_result_payload(row))
            timing = traces[sample_id]["timing_breakdown"]
            payload: dict[str, Any] = {
                "compute_mode": mode,
                "round": round_index,
                "rank": rank,
                "sample_id": sample_id,
                "compression": float(row["compression_time"]),
                "end_to_end": float(row["end_to_end_time"]),
                "decode_result": float(row["decode_time"]),
            }
            payload.update({field: float(timing[field]) for field in TIMING_FIELDS})
            trial_rows.append(payload)

    row_parity = all(len(values) == 1 for values in result_payloads.values())
    semantic_trace_parity = all(len(values) == 1 for values in semantic_trace_payloads.values())
    raw_selection_catalog_bitwise_parity = all(
        len(values) == 1 for values in raw_selection_catalogs.values()
    )
    selection_evidence = _selection_parity(
        selection_records,
        absolute_tolerance=args.selection_score_atol,
        relative_tolerance=args.selection_score_rtol,
    )
    selection_identity_parity = bool(selection_evidence["identity_parity"])
    selection_score_parity = bool(selection_evidence["score_parity"])
    trace_parity = bool(
        semantic_trace_parity and selection_identity_parity and selection_score_parity
    )
    trials = pd.DataFrame(trial_rows).sort_values(
        ["compute_mode", "round", "sample_id"], ignore_index=True
    )
    trials.to_parquet(root / "compute_trials.parquet", index=False)
    summary = (
        trials.groupby("compute_mode", as_index=False)
        .agg(
            measurements=("sample_id", "size"),
            unique_samples=("sample_id", "nunique"),
            mean_end_to_end=("end_to_end", "mean"),
            mean_compression=("compression", "mean"),
            mean_forecast=("forecast", "mean"),
            mean_utility_selection=("utility_and_selection", "mean"),
            mean_decode=("decode", "mean"),
        )
        .sort_values("compute_mode", ignore_index=True)
    )
    summary.to_parquet(root / "compute_summary.parquet", index=False)

    telemetry_records = [
        json.loads(path.read_text()) for path in sorted((root / "telemetry").glob("*.json"))
    ]
    telemetry_clean = bool(
        len(telemetry_records) == 4 and all(_telemetry_is_clean(item) for item in telemetry_records)
    )
    binding_clean, binding_record_count = _binding_is_clean(root)
    fixed_clock_statuses = sorted(
        {str(item.get("fixed_clock_status", "missing")) for item in telemetry_records}
    )
    metrics = (
        "prefill",
        "forecast",
        "graph",
        "utility_and_selection",
        "tier_construction",
        "cache_packing",
        "decode",
        "compression",
        "end_to_end",
    )
    effects = {metric: _paired_effect(trials, metric) for metric in metrics}
    speedup = float(effects["end_to_end"]["relative_reduction"])
    passed = bool(
        row_parity
        and trace_parity
        and diagnostics_disabled
        and binding_clean
        and telemetry_clean
        and speedup >= args.required_speedup
    )
    gate: dict[str, Any] = {
        "status": "passed" if passed else "failed",
        "pass": passed,
        "classification": "exploratory_dirty_tree_system_latency_compute_ab",
        "design": "paired_per_head_reference_vs_layer_batched_centroids_and_edges",
        "run_count": len(runs),
        "measurement_count": len(trials),
        "unique_sample_count": int(trials["sample_id"].nunique()),
        "rounds": int(trials["round"].nunique()),
        "row_parity": row_parity,
        "trace_parity": trace_parity,
        "semantic_trace_parity": semantic_trace_parity,
        "selection_identity_parity": selection_identity_parity,
        "selection_score_parity": selection_score_parity,
        "selection_score_evidence": selection_evidence,
        "raw_selection_catalog_bitwise_parity": raw_selection_catalog_bitwise_parity,
        "evaluation_diagnostics_disabled": diagnostics_disabled,
        "gpu_binding_preflight_clean": binding_clean,
        "gpu_binding_record_count": binding_record_count,
        "gpu_telemetry_clean": telemetry_clean,
        "fixed_clock_statuses": fixed_clock_statuses,
        "required_end_to_end_speedup": args.required_speedup,
        "effects": effects,
    }
    (root / "compute_gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    report = [
        "# Choice C isolated batched-compute latency gate",
        "",
        f"- Status: **{gate['status']}**",
        f"- Runs: {len(runs)}",
        f"- Paired measurements: {len(trials)} over {trials['sample_id'].nunique()} samples",
        f"- Result-row parity: {row_parity}",
        f"- Combined semantic trace parity: {trace_parity}",
        f"- Token/graph/prototype/forecast parity: {semantic_trace_parity}",
        f"- Exact selection identity/rank/reason parity: {selection_identity_parity}",
        f"- Selection score parity within tolerance: {selection_score_parity}",
        (
            "- Selection score tolerance: "
            f"atol={args.selection_score_atol:.1e}, rtol={args.selection_score_rtol:.1e}"
        ),
        (
            "- Maximum marginal-gain absolute difference: "
            f"{selection_evidence['maximum_absolute_difference']['marginal_gain']:.3e}"
        ),
        (f"- Raw score-bit catalog parity (informational): {raw_selection_catalog_bitwise_parity}"),
        f"- Evaluation-only diagnostics disabled: {diagnostics_disabled}",
        f"- Distinct-GPU binding preflight: {binding_clean}",
        f"- GPU telemetry/background-process gate: {telemetry_clean}",
        f"- Fixed-clock statuses: {', '.join(fixed_clock_statuses)}",
        f"- Required end-to-end reduction: {100 * args.required_speedup:.2f}%",
        f"- Measured end-to-end reduction: {100 * speedup:.2f}%",
        "",
        "## Paired phase effects",
        "",
        (
            "| Prefill | Forecast | Graph | Utility/selection | Tier | Packing | "
            "Decode | Compression | E2E |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| "
        + " | ".join(
            f"{100 * float(effects[metric]['relative_reduction']):+.2f}%" for metric in metrics
        )
        + " |",
        "",
        "Positive percentages mean the batched compute path was faster.",
        "",
        "This is a dirty-tree exploratory systems gate, not a paper result.",
    ]
    (root / "compute_gate_report.md").write_text("\n".join(report) + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
