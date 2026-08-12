#!/usr/bin/env python3
"""Analyze the isolated reduced Choice-C online-latency comparison."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
from analyze_choice_c_latency_factorial import (
    RUN_PATTERN,
    TIMING_FIELDS,
    _bootstrap_mean_interval,
    _result_payload,
    _telemetry_is_clean,
    _trace_payload,
)

SELECTED_FACTORS = ("g0e0p0", "g1e1p0", "g1e1p1")
EXPECTED_COUNTS = {"g0e0p0": 4, "g1e1p0": 2, "g1e1p1": 2}
REFERENCE_RANK = {
    ("g1e1p0", 0): 0,
    ("g1e1p0", 1): 1,
    ("g1e1p1", 0): 3,
    ("g1e1p1", 1): 2,
}


def _candidate_effect(
    trials: pd.DataFrame,
    candidate_id: str,
    metric: str,
) -> dict[str, Any]:
    paired_frames: list[pd.DataFrame] = []
    for round_index in (0, 1):
        reference_rank = REFERENCE_RANK[(candidate_id, round_index)]
        keys = ["round", "sample_id"]
        reference = trials.loc[
            (trials["factor_id"] == "g0e0p0")
            & (trials["round"] == round_index)
            & (trials["rank"] == reference_rank),
            [*keys, metric],
        ].rename(columns={metric: "reference"})
        candidate = trials.loc[
            (trials["factor_id"] == candidate_id)
            & (trials["round"] == round_index),
            [*keys, metric],
        ].rename(columns={metric: "candidate"})
        paired_frames.append(
            reference.merge(candidate, on=keys, how="inner", validate="one_to_one")
        )
    paired = pd.concat(paired_frames, ignore_index=True)
    if len(paired) != 8:
        raise RuntimeError(
            f"{candidate_id}/{metric} expected 8 paired samples, found {len(paired)}"
        )
    reductions = (paired["reference"] - paired["candidate"]).to_numpy(dtype=float)
    low, high = _bootstrap_mean_interval(reductions)
    reference_mean = float(paired["reference"].mean())
    reduction = float(reductions.mean())
    round_effects: dict[str, float] = {}
    for round_index, rows in paired.groupby("round", sort=True):
        round_reference = float(rows["reference"].mean())
        round_candidate = float(rows["candidate"].mean())
        round_effects[str(round_index)] = (
            round_reference - round_candidate
        ) / round_reference
    return {
        "pair_count": len(paired),
        "reference_mean_seconds": reference_mean,
        "candidate_mean_seconds": float(paired["candidate"].mean()),
        "mean_reduction_seconds": reduction,
        "relative_reduction": reduction / reference_mean,
        "bootstrap_95_reduction_seconds": [low, high],
        "relative_reduction_by_round": round_effects,
    }


def _incremental_prototype_effect(
    trials: pd.DataFrame,
    metric: str,
) -> dict[str, Any]:
    keys = ["round", "sample_id"]
    exact = trials.loc[trials["factor_id"] == "g1e1p0", [*keys, metric]].rename(
        columns={metric: "exact"}
    )
    prototype = trials.loc[
        trials["factor_id"] == "g1e1p1", [*keys, metric]
    ].rename(columns={metric: "prototype"})
    paired = exact.merge(prototype, on=keys, how="inner", validate="one_to_one")
    if len(paired) != 8:
        raise RuntimeError(
            f"prototype incremental/{metric} expected 8 pairs, found {len(paired)}"
        )
    reductions = (paired["exact"] - paired["prototype"]).to_numpy(dtype=float)
    low, high = _bootstrap_mean_interval(reductions)
    exact_mean = float(paired["exact"].mean())
    reduction = float(reductions.mean())
    return {
        "pair_count": len(paired),
        "exact_mean_seconds": exact_mean,
        "prototype_mean_seconds": float(paired["prototype"].mean()),
        "mean_reduction_seconds": reduction,
        "relative_reduction": reduction / exact_mean,
        "bootstrap_95_reduction_seconds": [low, high],
    }


def _binding_gate(root: Path) -> tuple[bool, list[dict[str, Any]]]:
    records = [
        json.loads(path.read_text())
        for path in sorted((root / "gpu_binding").glob("job-*/*/rank-*.json"))
    ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        key = (str(record.get("job_id")), str(record.get("step_id")))
        grouped.setdefault(key, []).append(record)
    clean = bool(
        len(records) == 8
        and len(grouped) == 4
        and all(
            len(items) == 2
            and {int(item["rank"]) for item in items} == {0, 1}
            and len({str(item["visible_gpu_uuid"]) for item in items}) == 2
            for items in grouped.values()
        )
    )
    return clean, records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--required-speedup", default=0.03, type=float)
    args = parser.parse_args()
    if not 0 < args.required_speedup < 1:
        raise ValueError("required-speedup must lie in (0, 1)")

    root = args.output_root.resolve()
    runs: list[tuple[Path, str, int, int]] = []
    for path in sorted(root.glob("choice-c-online-*")):
        if not path.is_dir() or "warmup" in path.name:
            continue
        match = RUN_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        factor_id, round_text, rank_text = match.groups()
        if factor_id in SELECTED_FACTORS:
            runs.append((path, factor_id, int(round_text), int(rank_text)))
    counts = {
        factor_id: sum(item[1] == factor_id for item in runs)
        for factor_id in SELECTED_FACTORS
    }
    if counts != EXPECTED_COUNTS:
        raise RuntimeError(f"reduced run coverage is incomplete: {counts}")

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
        for _index, row in frame.iterrows():
            sample_id = str(row["sample_id"])
            result_payloads.setdefault(sample_id, set()).add(_result_payload(row))
            timing = traces[sample_id]["timing_breakdown"]
            payload: dict[str, Any] = {
                "factor_id": factor_id,
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
    trace_parity = all(len(values) == 1 for values in trace_payloads.values())
    trials = pd.DataFrame(trial_rows).sort_values(
        ["factor_id", "round", "rank", "sample_id"], ignore_index=True
    )
    trials.to_parquet(root / "reduced_trials.parquet", index=False)
    summary = (
        trials.groupby("factor_id", as_index=False)
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
    summary.to_parquet(root / "reduced_summary.parquet", index=False)

    telemetry_records = [
        json.loads(path.read_text())
        for path in sorted((root / "telemetry").glob("*.json"))
    ]
    telemetry_clean = bool(
        len(telemetry_records) == 8
        and all(_telemetry_is_clean(item) for item in telemetry_records)
    )
    binding_clean, binding_records = _binding_gate(root)
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
    candidate_effects = {
        candidate_id: {
            metric: _candidate_effect(trials, candidate_id, metric)
            for metric in metrics
        }
        for candidate_id in ("g1e1p0", "g1e1p1")
    }
    prototype_incremental = {
        metric: _incremental_prototype_effect(trials, metric) for metric in metrics
    }
    full_speedup = float(
        candidate_effects["g1e1p1"]["end_to_end"]["relative_reduction"]
    )
    passed = bool(
        row_parity
        and trace_parity
        and diagnostics_disabled
        and binding_clean
        and telemetry_clean
        and full_speedup >= args.required_speedup
    )
    gate: dict[str, Any] = {
        "status": "passed" if passed else "failed",
        "pass": passed,
        "classification": "exploratory_dirty_tree_system_latency_reduced_factorial",
        "design": "paired_reference_vs_graph_exact_and_graph_exact_prototype",
        "run_count": len(runs),
        "measurement_count": len(trials),
        "unique_sample_count": int(trials["sample_id"].nunique()),
        "rounds": int(trials["round"].nunique()),
        "row_parity": row_parity,
        "trace_parity": trace_parity,
        "evaluation_diagnostics_disabled": diagnostics_disabled,
        "gpu_binding_preflight_clean": binding_clean,
        "gpu_binding_record_count": len(binding_records),
        "gpu_telemetry_clean": telemetry_clean,
        "fixed_clock_statuses": fixed_clock_statuses,
        "cpu_affinity_sizes": cpu_affinity_sizes,
        "required_full_candidate_end_to_end_speedup": args.required_speedup,
        "candidate_effects": candidate_effects,
        "prototype_incremental_effects": prototype_incremental,
    }
    (root / "reduced_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n"
    )
    report = [
        "# Choice C isolated reduced online-latency gate",
        "",
        f"- Status: **{gate['status']}**",
        f"- Runs: {len(runs)}",
        f"- Paired measurements: {len(trials)} over {trials['sample_id'].nunique()} samples",
        f"- Result-row parity: {row_parity}",
        f"- Token/selection/graph/prototype trace parity: {trace_parity}",
        f"- Evaluation-only diagnostics disabled: {diagnostics_disabled}",
        f"- Distinct-GPU binding preflight: {binding_clean}",
        f"- GPU telemetry/background-process gate: {telemetry_clean}",
        f"- Fixed-clock statuses: {', '.join(fixed_clock_statuses)}",
        f"- CPU affinity sizes: {cpu_affinity_sizes}",
        f"- Required full-candidate reduction: {100 * args.required_speedup:.2f}%",
        f"- Measured full-candidate reduction: {100 * full_speedup:.2f}%",
        "",
        "## Candidate effects against paired references",
        "",
        "| Candidate | End-to-end | Compression | Graph | Tier construction | Decode |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "g1e1p0": "Deferred graph + batched exact gather",
        "g1e1p1": "Deferred graph + batched exact/prototype gather",
    }
    for candidate_id in ("g1e1p0", "g1e1p1"):
        values = candidate_effects[candidate_id]
        report.append(
            "| "
            + labels[candidate_id]
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
            "## Incremental prototype-source gather effect",
            "",
            "| End-to-end | Compression | Graph | Tier construction | Decode |",
            "|---:|---:|---:|---:|---:|",
            "| "
            + " | ".join(
                f"{100 * float(prototype_incremental[metric]['relative_reduction']):+.2f}%"
                for metric in metrics
            )
            + " |",
            "",
            "Positive percentages mean the optimized configuration was faster.",
            "",
            "This is a dirty-tree exploratory systems gate, not a paper result.",
        ]
    )
    (root / "reduced_gate_report.md").write_text("\n".join(report) + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
