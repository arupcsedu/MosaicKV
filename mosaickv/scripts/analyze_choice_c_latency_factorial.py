#!/usr/bin/env python3
"""Analyze the isolated 2^3 Choice-C online-latency factorial experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

FACTORS = ("graph_optimized", "exact_optimized", "prototype_optimized")
FACTOR_IDS = tuple(
    f"g{graph}e{exact}p{prototype}"
    for graph in (0, 1)
    for exact in (0, 1)
    for prototype in (0, 1)
)
RUN_PATTERN = re.compile(
    r"^choice-c-online-(g[01]e[01]p[01])-round([01])-rank([0-7])-[0-9a-f]{10}$"
)
RESULT_PARITY_COLUMNS = (
    "status",
    "answer",
    "reference",
    "task_score",
    "generated_tokens",
    "active_kv_bytes",
    "residual_kv_bytes",
    "repair_count",
    "repaired_bytes",
)
TRACE_PARITY_FIELDS = (
    "requested_method",
    "effective_method",
    "tier_mode",
    "generated_token_ids",
    "selected_blocks",
    "prototypes",
    "graph_edges",
    "graph_diagnostics",
    "forecast_statistics",
    "source_prefill_kv_bytes",
    "logical_prefill_active_kv_bytes",
    "packed_prefill_active_kv_bytes",
    "residual_kv_bytes",
)
TIMING_FIELDS = (
    "prefill",
    "forecast",
    "graph",
    "utility_and_selection",
    "tier_construction",
    "cache_packing",
    "decode",
    "trace_instrumentation",
    "total",
)


def _factor_values(factor_id: str) -> dict[str, bool]:
    if factor_id not in FACTOR_IDS:
        raise ValueError(f"unknown factor ID: {factor_id}")
    return {
        "graph_optimized": factor_id[1] == "1",
        "exact_optimized": factor_id[3] == "1",
        "prototype_optimized": factor_id[5] == "1",
    }


def _trace_payload(record: dict[str, Any]) -> str:
    payload: list[Any] = []
    for field in TRACE_PARITY_FIELDS:
        value = record.get(field)
        if field == "forecast_statistics" and isinstance(value, dict):
            value = {key: item for key, item in value.items() if key != "timing"}
        payload.append(value)
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def _result_payload(row: pd.Series[Any]) -> tuple[Any, ...]:
    return tuple(row[column] for column in RESULT_PARITY_COLUMNS)


def _bootstrap_mean_interval(values: np.ndarray[Any, Any]) -> tuple[float, float]:
    if values.size < 2:
        value = float(values.mean())
        return value, value
    rng = np.random.default_rng(20270808)
    indices = rng.integers(0, values.size, size=(20_000, values.size))
    means = values[indices].mean(axis=1)
    low, high = np.percentile(means, (2.5, 97.5))
    return float(low), float(high)


def _paired_effect(
    trials: pd.DataFrame,
    factor: str,
    metric: str,
) -> dict[str, Any]:
    other_factors = [name for name in FACTORS if name != factor]
    keys = ["round", "sample_id", *other_factors]
    off = trials.loc[~trials[factor], [*keys, metric]].rename(columns={metric: "off"})
    on = trials.loc[trials[factor], [*keys, metric]].rename(columns={metric: "on"})
    paired = off.merge(on, on=keys, how="inner", validate="one_to_one")
    expected = 2 * 4 * 4
    if len(paired) != expected:
        raise RuntimeError(
            f"{factor}/{metric} expected {expected} factorial pairs, found {len(paired)}"
        )
    reductions = (paired["off"] - paired["on"]).to_numpy(dtype=float)
    low, high = _bootstrap_mean_interval(reductions)
    off_mean = float(paired["off"].mean())
    reduction = float(reductions.mean())
    return {
        "pair_count": len(paired),
        "off_mean_seconds": off_mean,
        "on_mean_seconds": float(paired["on"].mean()),
        "mean_reduction_seconds": reduction,
        "relative_reduction": reduction / off_mean,
        "bootstrap_95_reduction_seconds": [low, high],
    }


def _full_candidate_effect(trials: pd.DataFrame, metric: str) -> dict[str, Any]:
    keys = ["round", "sample_id"]
    reference = trials.loc[trials["factor_id"] == "g0e0p0", [*keys, metric]].rename(
        columns={metric: "reference"}
    )
    candidate = trials.loc[trials["factor_id"] == "g1e1p1", [*keys, metric]].rename(
        columns={metric: "candidate"}
    )
    paired = reference.merge(candidate, on=keys, how="inner", validate="one_to_one")
    if len(paired) != 8:
        raise RuntimeError(f"full candidate comparison expected 8 pairs, found {len(paired)}")
    reductions = (paired["reference"] - paired["candidate"]).to_numpy(dtype=float)
    low, high = _bootstrap_mean_interval(reductions)
    reference_mean = float(paired["reference"].mean())
    reduction = float(reductions.mean())
    rounds: dict[str, float] = {}
    for round_index, rows in paired.groupby("round", sort=True):
        ref = float(rows["reference"].mean())
        cand = float(rows["candidate"].mean())
        rounds[str(round_index)] = (ref - cand) / ref
    return {
        "pair_count": len(paired),
        "reference_mean_seconds": reference_mean,
        "candidate_mean_seconds": float(paired["candidate"].mean()),
        "mean_reduction_seconds": reduction,
        "relative_reduction": reduction / reference_mean,
        "bootstrap_95_reduction_seconds": [low, high],
        "relative_reduction_by_round": rounds,
    }


def _telemetry_is_clean(payload: dict[str, Any]) -> bool:
    if payload.get("status") != "completed":
        return False
    for stage in ("before_measurement", "after_measurement"):
        snapshot = payload.get(stage)
        if not isinstance(snapshot, dict) or snapshot.get("query_error") is not None:
            return False
        devices = snapshot.get("devices")
        if not isinstance(devices, list):
            return False
        visible = [
            item
            for item in devices
            if isinstance(item, dict) and item.get("visible_to_process")
        ]
        if len(visible) != 1:
            return False
        if snapshot.get("background_processes") or int(
            snapshot.get("visible_process_concurrency", -1)
        ) > 1:
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--required-speedup", type=float, default=0.03)
    args = parser.parse_args()
    if not 0 < args.required_speedup < 1:
        raise ValueError("required speedup must lie in (0, 1)")

    root = args.output_root.resolve()
    runs: list[tuple[Path, str, int, int]] = []
    for path in sorted(root.glob("choice-c-online-*")):
        if not path.is_dir() or "warmup" in path.name:
            continue
        match = RUN_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        factor_id, round_text, rank_text = match.groups()
        runs.append((path, factor_id, int(round_text), int(rank_text)))
    counts = {factor_id: sum(item[1] == factor_id for item in runs) for factor_id in FACTOR_IDS}
    if counts != {factor_id: 2 for factor_id in FACTOR_IDS}:
        raise RuntimeError(f"factorial run coverage is incomplete: {counts}")

    result_payloads: dict[str, set[tuple[Any, ...]]] = {}
    trace_payloads: dict[str, set[str]] = {}
    trial_rows: list[dict[str, Any]] = []
    diagnostics_disabled = True
    for run, factor_id, round_index, rank in runs:
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
            diagnostics_disabled
            and not manifest["resolved_config"]["diagnostics"]["enabled"]
        )
        traces: dict[str, dict[str, Any]] = {}
        for path in sorted((root / "traces" / run.name).glob("*.json")):
            record = json.loads(path.read_text())
            sample_id = str(record["sample_id"])
            if sample_id in traces:
                raise RuntimeError(f"duplicate trace for {run.name}/{sample_id}")
            traces[sample_id] = record
            trace_payloads.setdefault(sample_id, set()).add(_trace_payload(record))
            if record.get("fidelity_diagnostic") != {"status": "disabled"}:
                diagnostics_disabled = False
            if record.get("forecast_quality_diagnostic") != {"status": "disabled"}:
                diagnostics_disabled = False
        if set(traces) != set(frame["sample_id"]):
            raise RuntimeError(f"trace/result sample mismatch: {run.name}")
        factors = _factor_values(factor_id)
        for _index, row in frame.iterrows():
            sample_id = str(row["sample_id"])
            result_payloads.setdefault(sample_id, set()).add(_result_payload(row))
            timing = traces[sample_id]["timing_breakdown"]
            payload: dict[str, Any] = {
                "factor_id": factor_id,
                "round": round_index,
                "rank": rank,
                "sample_id": sample_id,
                **factors,
                "compression": float(row["compression_time"]),
                "end_to_end": float(row["end_to_end_time"]),
                "decode_result": float(row["decode_time"]),
            }
            payload.update({field: float(timing[field]) for field in TIMING_FIELDS})
            trial_rows.append(payload)

    row_parity = all(len(values) == 1 for values in result_payloads.values())
    trace_parity = all(len(values) == 1 for values in trace_payloads.values())
    trials = pd.DataFrame(trial_rows).sort_values(
        ["factor_id", "round", "sample_id"], ignore_index=True
    )
    trials.to_parquet(root / "factorial_trials.parquet", index=False)
    summary = (
        trials.groupby(["factor_id", *FACTORS], as_index=False)
        .agg(
            measurements=("sample_id", "size"),
            unique_samples=("sample_id", "nunique"),
            mean_end_to_end=("end_to_end", "mean"),
            mean_compression=("compression", "mean"),
            mean_graph=("graph", "mean"),
            mean_tier_construction=("tier_construction", "mean"),
            mean_decode=("decode", "mean"),
        )
        .sort_values("factor_id", ignore_index=True)
    )
    summary.to_parquet(root / "factorial_summary.parquet", index=False)

    telemetry_records = [
        json.loads(path.read_text()) for path in sorted((root / "telemetry").glob("*.json"))
    ]
    telemetry_clean = bool(
        len(telemetry_records) == 16
        and all(_telemetry_is_clean(item) for item in telemetry_records)
    )
    fixed_clock_statuses = sorted(
        {str(item.get("fixed_clock_status", "missing")) for item in telemetry_records}
    )
    cpu_affinity_sizes = sorted(
        {
            len(item.get("slurm", {}).get("cpu_affinity", []))
            for item in telemetry_records
            if isinstance(item.get("slurm"), dict)
        }
    )

    metrics = ("end_to_end", "compression", "graph", "tier_construction", "decode")
    main_effects = {
        factor: {metric: _paired_effect(trials, factor, metric) for metric in metrics}
        for factor in FACTORS
    }
    full_candidate = {metric: _full_candidate_effect(trials, metric) for metric in metrics}
    full_speedup = float(full_candidate["end_to_end"]["relative_reduction"])
    passed = bool(
        row_parity
        and trace_parity
        and diagnostics_disabled
        and telemetry_clean
        and full_speedup >= args.required_speedup
    )
    gate: dict[str, Any] = {
        "status": "passed" if passed else "failed",
        "pass": passed,
        "classification": "exploratory_dirty_tree_system_latency_factorial",
        "factorial_design": "2^3_graph_transfer_x_exact_gather_x_prototype_source_gather",
        "run_count": len(runs),
        "measurement_count": len(trials),
        "unique_sample_count": int(trials["sample_id"].nunique()),
        "rounds": int(trials["round"].nunique()),
        "row_parity": row_parity,
        "trace_parity": trace_parity,
        "evaluation_diagnostics_disabled": diagnostics_disabled,
        "gpu_telemetry_clean": telemetry_clean,
        "fixed_clock_statuses": fixed_clock_statuses,
        "cpu_affinity_sizes": cpu_affinity_sizes,
        "required_full_candidate_end_to_end_speedup": args.required_speedup,
        "main_effects": main_effects,
        "full_candidate_effects": full_candidate,
    }
    (root / "factorial_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n"
    )
    report = [
        "# Choice C isolated online-latency factorial gate",
        "",
        f"- Status: **{gate['status']}**",
        f"- Runs: {len(runs)} (8 factor cells x 2 counterbalanced rounds)",
        f"- Paired measurements: {len(trials)} over {trials['sample_id'].nunique()} samples",
        f"- Result-row parity: {row_parity}",
        f"- Token/selection/graph/prototype trace parity: {trace_parity}",
        f"- Evaluation-only diagnostics disabled: {diagnostics_disabled}",
        f"- GPU telemetry/background-process gate: {telemetry_clean}",
        f"- Fixed-clock statuses: {', '.join(fixed_clock_statuses)}",
        f"- CPU affinity sizes: {cpu_affinity_sizes}",
        f"- Required full-candidate reduction: {100 * args.required_speedup:.2f}%",
        f"- Measured full-candidate reduction: {100 * full_speedup:.2f}%",
        "",
        "## Factor main effects",
        "",
        "| Factor | End-to-end | Compression | Graph | Tier construction | Decode |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "graph_optimized": "Deferred graph host transfer",
        "exact_optimized": "Batched exact-tier gather",
        "prototype_optimized": "Batched prototype-source gather",
    }
    for factor in FACTORS:
        values = main_effects[factor]
        report.append(
            "| "
            + labels[factor]
            + " | "
            + " | ".join(
                f"{100 * float(values[metric]['relative_reduction']):+.2f}%"
                for metric in metrics
            )
            + " |"
        )
    report.extend(
        [
            "",
            "Positive percentages mean the optimized factor was faster after averaging "
            "over the other two factors, both rounds, and all paired samples.",
            "",
            "This is a dirty-tree exploratory systems gate, not a paper result.",
        ]
    )
    (root / "factorial_gate_report.md").write_text("\n".join(report) + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
