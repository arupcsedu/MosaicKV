#!/usr/bin/env python3
"""Analyze same-GPU alternating Choice-C latency measurements."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
from analyze_choice_c_latency_compute import (
    _selection_parity,
    _selection_records,
    _semantic_trace_payload,
)
from analyze_choice_c_latency_factorial import (
    TIMING_FIELDS,
    _bootstrap_mean_interval,
    _result_payload,
    _telemetry_is_clean,
)

RUN_PATTERN = re.compile(
    r"^choice-c-blocked-rank([01])-cycle([0-3])-order([01])-"
    r"(reference|optimized)-[0-9a-f]{10}$"
)
METRICS = (
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


def _paired_effect(trials: pd.DataFrame, metric: str) -> tuple[dict[str, Any], pd.DataFrame]:
    keys = ["rank", "cycle", "sample_id"]
    reference = trials.loc[
        trials["compute_mode"] == "reference", [*keys, "order_index", metric]
    ].rename(columns={"order_index": "reference_order", metric: "reference"})
    optimized = trials.loc[
        trials["compute_mode"] == "optimized", [*keys, "order_index", metric]
    ].rename(columns={"order_index": "optimized_order", metric: "optimized"})
    paired = reference.merge(optimized, on=keys, how="inner", validate="one_to_one")
    if len(paired) != 32:
        raise RuntimeError(f"{metric} expected 32 same-GPU pairs, found {len(paired)}")
    paired["reduction"] = paired["reference"] - paired["optimized"]
    blocks = (
        paired.groupby(["rank", "cycle"], as_index=False)
        .agg(
            reference=("reference", "mean"),
            optimized=("optimized", "mean"),
            reduction=("reduction", "mean"),
            optimized_order=("optimized_order", "first"),
        )
        .sort_values(["rank", "cycle"], ignore_index=True)
    )
    blocks["relative_reduction"] = blocks["reduction"] / blocks["reference"]
    low, high = _bootstrap_mean_interval(blocks["reduction"].to_numpy(dtype=float))
    reference_mean = float(paired["reference"].mean())
    reduction = float(paired["reduction"].mean())
    rank_effects: dict[str, float] = {}
    for rank, rows in paired.groupby("rank", sort=True):
        rank_reference = float(rows["reference"].mean())
        rank_effects[str(rank)] = (
            rank_reference - float(rows["optimized"].mean())
        ) / rank_reference
    return (
        {
            "pair_count": len(paired),
            "block_count": len(blocks),
            "reference_mean_seconds": reference_mean,
            "optimized_mean_seconds": float(paired["optimized"].mean()),
            "mean_reduction_seconds": reduction,
            "relative_reduction": reduction / reference_mean,
            "block_bootstrap_95_reduction_seconds": [low, high],
            "positive_block_count": int((blocks["reduction"] > 0).sum()),
            "relative_reduction_by_rank": rank_effects,
            "relative_reduction_by_block": {
                f"rank{int(row['rank'])}:cycle{int(row['cycle'])}": float(row["relative_reduction"])
                for _, row in blocks.iterrows()
            },
        },
        blocks,
    )


def _binding_is_clean(root: Path) -> tuple[bool, int]:
    records = [
        json.loads(path.read_text())
        for path in sorted((root / "gpu_binding").glob("job-*/*/rank-*.json"))
    ]
    clean = bool(
        len(records) == 2
        and {int(item["rank"]) for item in records} == {0, 1}
        and len({str(item["visible_gpu_uuid"]) for item in records}) == 2
        and len({str(item["step_id"]) for item in records}) == 1
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
    runs: list[tuple[Path, int, int, int, str]] = []
    for path in sorted(root.glob("choice-c-blocked-rank*")):
        if not path.is_dir() or "warmup" in path.name:
            continue
        match = RUN_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        rank, cycle, order_index, mode = match.groups()
        runs.append((path, int(rank), int(cycle), int(order_index), mode))
    if len(runs) != 16:
        raise RuntimeError(f"blocked A/B coverage expected 16 runs, found {len(runs)}")
    order_counts = {
        (mode, order): sum(item[4] == mode and item[3] == order for item in runs)
        for mode in ("reference", "optimized")
        for order in (0, 1)
    }
    order_balance = all(count == 4 for count in order_counts.values())

    result_payloads: dict[str, set[tuple[Any, ...]]] = {}
    semantic_trace_payloads: dict[str, set[str]] = {}
    raw_selection_catalogs: dict[str, set[str]] = {}
    selection_records: dict[str, list[list[dict[str, Any]]]] = {}
    trial_rows: list[dict[str, Any]] = []
    diagnostics_disabled = True
    for run, rank, cycle, order_index, mode in runs:
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
        for _, row in frame.iterrows():
            sample_id = str(row["sample_id"])
            result_payloads.setdefault(sample_id, set()).add(_result_payload(row))
            timing = traces[sample_id]["timing_breakdown"]
            payload: dict[str, Any] = {
                "compute_mode": mode,
                "rank": rank,
                "cycle": cycle,
                "order_index": order_index,
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
        expected_replicas=16,
    )
    selection_identity_parity = bool(selection_evidence["identity_parity"])
    selection_score_parity = bool(selection_evidence["score_parity"])
    trace_parity = bool(
        semantic_trace_parity and selection_identity_parity and selection_score_parity
    )

    trials = pd.DataFrame(trial_rows).sort_values(
        ["rank", "cycle", "order_index", "sample_id"], ignore_index=True
    )
    trials.to_parquet(root / "blocked_trials.parquet", index=False)
    effects: dict[str, Any] = {}
    block_frames: list[pd.DataFrame] = []
    for metric in METRICS:
        effect, blocks = _paired_effect(trials, metric)
        effects[metric] = effect
        block_frames.append(blocks.assign(metric=metric))
    all_blocks = pd.concat(block_frames, ignore_index=True)
    all_blocks.to_parquet(root / "blocked_blocks.parquet", index=False)
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
    summary.to_parquet(root / "blocked_summary.parquet", index=False)

    telemetry_records = [
        json.loads(path.read_text()) for path in sorted((root / "telemetry").glob("*.json"))
    ]
    telemetry_clean = bool(
        len(telemetry_records) == 2
        and all(_telemetry_is_clean(item) for item in telemetry_records)
        and all(item.get("model_load_count") == 1 for item in telemetry_records)
    )
    binding_clean, binding_record_count = _binding_is_clean(root)
    fixed_clock_statuses = sorted(
        {str(item.get("fixed_clock_status", "missing")) for item in telemetry_records}
    )
    end_to_end = effects["end_to_end"]
    speedup = float(end_to_end["relative_reduction"])
    block_ci_low = float(end_to_end["block_bootstrap_95_reduction_seconds"][0])
    both_gpu_replicas_positive = all(
        float(value) > 0 for value in end_to_end["relative_reduction_by_rank"].values()
    )
    passed = bool(
        row_parity
        and trace_parity
        and diagnostics_disabled
        and order_balance
        and binding_clean
        and telemetry_clean
        and speedup >= args.required_speedup
        and block_ci_low > 0
        and both_gpu_replicas_positive
    )
    gate: dict[str, Any] = {
        "status": "passed" if passed else "failed",
        "pass": passed,
        "classification": "exploratory_dirty_tree_same_gpu_blocked_latency_ab",
        "design": "two_gpu_within_gpu_alternating_ab_four_cycles",
        "run_count": len(runs),
        "measurement_count": len(trials),
        "pair_count": int(end_to_end["pair_count"]),
        "block_count": int(end_to_end["block_count"]),
        "unique_sample_count": int(trials["sample_id"].nunique()),
        "row_parity": row_parity,
        "trace_parity": trace_parity,
        "semantic_trace_parity": semantic_trace_parity,
        "selection_identity_parity": selection_identity_parity,
        "selection_score_parity": selection_score_parity,
        "selection_score_evidence": selection_evidence,
        "raw_selection_catalog_bitwise_parity": raw_selection_catalog_bitwise_parity,
        "evaluation_diagnostics_disabled": diagnostics_disabled,
        "order_balance": order_balance,
        "order_counts": {
            f"{mode}:order{order}": count for (mode, order), count in order_counts.items()
        },
        "gpu_binding_preflight_clean": binding_clean,
        "gpu_binding_record_count": binding_record_count,
        "gpu_telemetry_clean": telemetry_clean,
        "fixed_clock_statuses": fixed_clock_statuses,
        "required_end_to_end_speedup": args.required_speedup,
        "requires_positive_block_ci": True,
        "block_ci_lower_positive": block_ci_low > 0,
        "requires_both_gpu_replicas_positive": True,
        "both_gpu_replicas_positive": both_gpu_replicas_positive,
        "effects": effects,
    }
    (root / "blocked_gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    report = [
        "# Choice C same-GPU blocked latency gate",
        "",
        f"- Status: **{gate['status']}**",
        f"- Runs: {len(runs)}",
        f"- Same-GPU paired measurements: {end_to_end['pair_count']}",
        f"- Independent GPU-cycle blocks: {end_to_end['block_count']}",
        f"- Result-row parity: {row_parity}",
        f"- Combined semantic trace parity: {trace_parity}",
        f"- Exact selection identity/rank/reason parity: {selection_identity_parity}",
        f"- Selection score parity within tolerance: {selection_score_parity}",
        f"- Counterbalanced execution order: {order_balance}",
        f"- Distinct-GPU binding preflight: {binding_clean}",
        f"- GPU telemetry/background-process gate: {telemetry_clean}",
        f"- Both GPU replicas show positive E2E effects: {both_gpu_replicas_positive}",
        f"- Positive block-bootstrap lower bound: {block_ci_low > 0}",
        f"- Required end-to-end reduction: {100 * args.required_speedup:.2f}%",
        f"- Measured end-to-end reduction: {100 * speedup:.2f}%",
        (
            "- Block-bootstrap 95% reduction interval: "
            f"[{end_to_end['block_bootstrap_95_reduction_seconds'][0]:+.3f}, "
            f"{end_to_end['block_bootstrap_95_reduction_seconds'][1]:+.3f}] seconds"
        ),
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
            f"{100 * float(effects[metric]['relative_reduction']):+.2f}%" for metric in METRICS
        )
        + " |",
        "",
        "Positive percentages mean the optimized path was faster.",
        "",
        "This is a dirty-tree exploratory systems gate, not a paper result.",
    ]
    (root / "blocked_gate_report.md").write_text("\n".join(report) + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
