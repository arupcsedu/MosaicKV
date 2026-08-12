#!/usr/bin/env python3
"""Gate the Choice-C host-transfer optimization on parity and online latency."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

REFERENCE = "per_group_reference"
CANDIDATE = "deferred_layer"
PARITY_COLUMNS = (
    "status",
    "answer",
    "reference",
    "task_score",
    "generated_tokens",
    "active_kv_bytes",
    "residual_kv_bytes",
    "repair_count",
    "repaired_bytes",
    "token_agreement",
    "logit_kl",
    "attention_output_error",
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
    "fidelity_diagnostic",
    "source_prefill_kv_bytes",
    "logical_prefill_active_kv_bytes",
    "packed_prefill_active_kv_bytes",
    "residual_kv_bytes",
)


def _runs(root: Path, mode: str) -> tuple[Path, ...]:
    matches = sorted(path for path in root.glob(f"choice-c-latency-{mode}-*") if path.is_dir())
    if len(matches) != 2:
        raise RuntimeError(f"expected two counterbalanced {mode} runs, found {len(matches)}")
    return tuple(matches)


def _traces(root: Path, runs: tuple[Path, ...]) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for run in runs:
        for path in sorted((root / "traces" / run.name).glob("*.json")):
            record = json.loads(path.read_text())
            key = run.name, str(record["sample_id"])
            if key in result:
                raise RuntimeError(f"duplicate trace for {key}")
            result[key] = record
    return result


def _speedup(reference: float, candidate: float) -> float:
    if not math.isfinite(reference) or not math.isfinite(candidate) or reference <= 0:
        raise RuntimeError("latency values must be positive and finite")
    return (reference - candidate) / reference


def _parity_payload(record: dict[str, Any], field: str) -> Any:
    """Remove explicitly measured overhead fields from semantic trace parity."""

    value = record.get(field)
    if field == "forecast_statistics" and isinstance(value, dict):
        return {key: item for key, item in value.items() if key != "timing"}
    if field == "fidelity_diagnostic" and isinstance(value, dict):
        return {key: item for key, item in value.items() if key != "overhead_seconds"}
    return value


def _canonical_rows(frame: pd.DataFrame) -> dict[str, tuple[Any, ...]]:
    result: dict[str, tuple[Any, ...]] = {}
    for sample_id, rows in frame.groupby("sample_id", sort=True):
        payloads = {
            tuple(row[column] for column in PARITY_COLUMNS) for _index, row in rows.iterrows()
        }
        if len(payloads) != 1:
            raise RuntimeError(f"replicas disagree on result row for {sample_id}")
        result[str(sample_id)] = payloads.pop()
    return result


def _canonical_traces(
    traces: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, tuple[Any, ...]]:
    grouped: dict[str, set[str]] = {}
    decoded: dict[str, tuple[Any, ...]] = {}
    for (_run_id, sample_id), record in traces.items():
        payload = tuple(_parity_payload(record, field) for field in TRACE_PARITY_FIELDS)
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        grouped.setdefault(sample_id, set()).add(serialized)
        decoded[sample_id] = payload
    disagreements = sorted(sample_id for sample_id, values in grouped.items() if len(values) != 1)
    if disagreements:
        raise RuntimeError(f"replicas disagree on trace payloads: {disagreements}")
    return decoded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--required-speedup", type=float, default=0.03)
    args = parser.parse_args()
    if not 0 < args.required_speedup < 1:
        raise ValueError("required speedup must lie in (0, 1)")

    root = args.output_root.resolve()
    reference_runs = _runs(root, REFERENCE)
    candidate_runs = _runs(root, CANDIDATE)
    reference = pd.concat(
        [pd.read_parquet(run / "aggregate.parquet") for run in reference_runs],
        ignore_index=True,
    ).sort_values("sample_id")
    candidate = pd.concat(
        [pd.read_parquet(run / "aggregate.parquet") for run in candidate_runs],
        ignore_index=True,
    ).sort_values("sample_id")
    if set(reference["sample_id"]) != set(candidate["sample_id"]):
        raise RuntimeError("latency A/B runs contain different sample IDs")
    if len(reference) < 8 or len(candidate) != len(reference):
        raise RuntimeError("latency A/B gate requires two replicas of four paired samples")

    row_parity = _canonical_rows(reference) == _canonical_rows(candidate)
    reference_traces = _traces(root, reference_runs)
    candidate_traces = _traces(root, candidate_runs)
    trace_parity = _canonical_traces(reference_traces) == _canonical_traces(candidate_traces)

    reference_total = float(reference["end_to_end_time"].mean())
    candidate_total = float(candidate["end_to_end_time"].mean())
    reference_compression = float(reference["compression_time"].mean())
    candidate_compression = float(candidate["compression_time"].mean())
    reference_graph = sum(
        float(record["timing_breakdown"]["graph"]) for record in reference_traces.values()
    ) / len(reference_traces)
    candidate_graph = sum(
        float(record["timing_breakdown"]["graph"]) for record in candidate_traces.values()
    ) / len(candidate_traces)
    reference_tier = sum(
        float(record["timing_breakdown"]["tier_construction"])
        for record in reference_traces.values()
    ) / len(reference_traces)
    candidate_tier = sum(
        float(record["timing_breakdown"]["tier_construction"])
        for record in candidate_traces.values()
    ) / len(candidate_traces)
    total_speedup = _speedup(reference_total, candidate_total)
    compression_speedup = _speedup(reference_compression, candidate_compression)
    graph_speedup = _speedup(reference_graph, candidate_graph)
    tier_speedup = _speedup(reference_tier, candidate_tier)
    passed = bool(row_parity and trace_parity and total_speedup >= args.required_speedup)
    gate = {
        "status": "passed" if passed else "failed",
        "pass": passed,
        "classification": "exploratory_dirty_tree_system_latency_gate_not_paper_result",
        "sample_count": len(reference),
        "unique_sample_count": int(reference["sample_id"].nunique()),
        "replicas_per_mode": len(reference_runs),
        "reference_mode": REFERENCE,
        "candidate_mode": CANDIDATE,
        "row_parity": row_parity,
        "trace_parity": trace_parity,
        "required_end_to_end_speedup": args.required_speedup,
        "reference_mean_end_to_end_seconds": reference_total,
        "candidate_mean_end_to_end_seconds": candidate_total,
        "relative_end_to_end_speedup": total_speedup,
        "reference_mean_compression_seconds": reference_compression,
        "candidate_mean_compression_seconds": candidate_compression,
        "relative_compression_speedup": compression_speedup,
        "reference_mean_graph_seconds": reference_graph,
        "candidate_mean_graph_seconds": candidate_graph,
        "relative_graph_speedup": graph_speedup,
        "reference_mean_tier_construction_seconds": reference_tier,
        "candidate_mean_tier_construction_seconds": candidate_tier,
        "relative_tier_construction_speedup": tier_speedup,
    }
    (root / "latency_gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    report = [
        "# Choice C host-transfer latency gate",
        "",
        f"- Status: **{gate['status']}**",
        f"- Paired measurements: {len(reference)}",
        f"- Unique samples: {reference['sample_id'].nunique()}",
        f"- Counterbalanced replicas per mode: {len(reference_runs)}",
        f"- Result-row parity: {row_parity}",
        f"- Token/selection/graph/prototype trace parity: {trace_parity}",
        f"- Required end-to-end reduction: {100 * args.required_speedup:.2f}%",
        f"- Measured end-to-end reduction: {100 * total_speedup:.2f}%",
        f"- Measured compression reduction: {100 * compression_speedup:.2f}%",
        f"- Measured graph reduction: {100 * graph_speedup:.2f}%",
        f"- Measured tier-construction reduction: {100 * tier_speedup:.2f}%",
        "",
        "This is a dirty-tree exploratory systems gate, not a paper result.",
    ]
    (root / "latency_gate_report.md").write_text("\n".join(report) + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
