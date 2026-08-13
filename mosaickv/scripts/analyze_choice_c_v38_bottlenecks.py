#!/usr/bin/env python3
"""Aggregate v38 compression phase, fidelity, and cache-invariant evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
from run_choice_c_v38_bottleneck_worker import RANK_MODES

PHASES = (
    "forecast",
    "graph",
    "utility_and_selection",
    "tier_construction",
    "cache_packing",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument(
        "--modes",
        nargs="*",
        choices=tuple(mode for modes in RANK_MODES.values() for mode in modes),
        help="analyze only these completed modes (default: every v38 mode)",
    )
    args = parser.parse_args()
    root = args.output_root.resolve()
    reference_root = args.reference_root.resolve()

    selected_modes = set(args.modes or ())
    mode_pairs = tuple(
        (rank, mode)
        for rank, modes in RANK_MODES.items()
        for mode in modes
        if not selected_modes or mode in selected_modes
    )
    rows: list[dict[str, Any]] = []
    for rank, mode in mode_pairs:
        aggregates = sorted(root.glob(f"v38-measured-{mode}-rank{rank}-*/aggregate.parquet"))
        if len(aggregates) != 1:
            raise RuntimeError(f"expected one aggregate for rank={rank}, mode={mode}")
        frame = pd.read_parquet(aggregates[0])
        if len(frame) != 2 or set(frame["status"]) != {"completed"}:
            raise RuntimeError(f"incomplete v38 result for {mode}")
        run_id = str(frame["run_id"].iloc[0])
        traces = sorted((root / "traces" / run_id).glob("*.json"))
        if len(traces) != 2:
            raise RuntimeError(f"trace/result mismatch for {run_id}")
        frame_by_sample = frame.set_index("sample_id")
        for trace_path in traces:
            trace = json.loads(trace_path.read_text())
            sample_id = str(trace["sample_id"])
            timing = trace.get("timing_breakdown")
            if not isinstance(timing, dict):
                raise RuntimeError(f"missing timing breakdown: {trace_path}")
            effective_method = str(trace.get("effective_method"))
            tier_mode = str(trace.get("tier_mode"))
            genuine_three_tier = (
                effective_method == "mosaickv_full" and tier_mode == "three_tier"
            )
            if trace.get("packed_cache_all_slots_valid") is not True:
                raise RuntimeError(f"{mode} produced padded active cache slots")
            if int(trace["active_kv_bytes"]) > int(trace["hard_active_kv_budget_bytes"]):
                raise RuntimeError(f"{mode} exceeded its active byte budget")
            row = frame_by_sample.loc[sample_id]
            token_payload = json.dumps(trace.get("generated_token_ids", []), separators=(",", ":"))
            item: dict[str, Any] = {
                "mode": mode,
                "sample_id": sample_id,
                "effective_method": effective_method,
                "tier_mode": tier_mode,
                "genuine_three_tier": genuine_three_tier,
                "fallback_reason": "" if genuine_three_tier else str(trace.get("tier_reason")),
                "answer": str(row["answer"]),
                "generated_tokens_sha256": hashlib.sha256(token_payload.encode()).hexdigest(),
                "task_score": float(row["task_score"]),
                "token_agreement": (
                    None if pd.isna(row["token_agreement"]) else float(row["token_agreement"])
                ),
                "active_kv_bytes": int(row["active_kv_bytes"]),
                "residual_kv_bytes": int(row["residual_kv_bytes"]),
                "compression_time": float(row["compression_time"]),
                "end_to_end_time": float(row["end_to_end_time"]),
                "graph_nodes": int(trace["graph_diagnostics"]["node_count"]),
                "graph_edges": int(trace["graph_diagnostics"]["edge_count"]),
            }
            for phase in PHASES:
                value = float(timing[phase])
                if not math.isfinite(value) or value < 0:
                    raise RuntimeError(f"invalid {phase} time for {mode}/{sample_id}")
                item[phase] = value
            rows.append(item)

    trials = pd.DataFrame(rows)
    reference_rows: list[dict[str, Any]] = []
    for trace_path in sorted(reference_root.glob("traces/v36-measured-mosaickv*/*.json")):
        trace = json.loads(trace_path.read_text())
        timing = trace["timing_breakdown"]
        token_payload = json.dumps(trace.get("generated_token_ids", []), separators=(",", ":"))
        reference_rows.append(
            {
                "sample_id": str(trace["sample_id"]),
                "answer": str(trace["answer"]),
                "generated_tokens_sha256": hashlib.sha256(token_payload.encode()).hexdigest(),
                "active_kv_bytes": int(trace["active_kv_bytes"]),
                "residual_kv_bytes": int(trace["residual_kv_bytes"]),
                **{phase: float(timing[phase]) for phase in PHASES},
            }
        )
    reference_frame = pd.DataFrame(reference_rows)
    for sample_id, group in reference_frame.groupby("sample_id"):
        for field in (
            "answer",
            "generated_tokens_sha256",
            "active_kv_bytes",
            "residual_kv_bytes",
        ):
            if group[field].nunique() != 1:
                raise RuntimeError(f"v36 reference disagrees for {sample_id}/{field}")
    reference_identity = reference_frame.drop_duplicates("sample_id").set_index("sample_id")
    trials["output_parity"] = tuple(
        row.answer == reference_identity.loc[row.sample_id, "answer"]
        and row.generated_tokens_sha256
        == reference_identity.loc[row.sample_id, "generated_tokens_sha256"]
        for row in trials.itertuples()
    )
    trials["storage_parity"] = tuple(
        row.active_kv_bytes == reference_identity.loc[row.sample_id, "active_kv_bytes"]
        and row.residual_kv_bytes == reference_identity.loc[row.sample_id, "residual_kv_bytes"]
        for row in trials.itertuples()
    )
    # Diagnostics were intentionally disabled for this systems-only run.
    # Preserve unavailable token agreement as a nullable field rather than a
    # floating NaN, and do not manufacture an aggregate value from it.
    trials["token_agreement"] = trials["token_agreement"].astype("Float64")
    reference = reference_frame.groupby("sample_id", as_index=False)[list(PHASES)].mean()
    reference_means = {phase: float(reference[phase].mean()) for phase in PHASES}
    summary = (
        trials.groupby("mode", sort=False)
        .agg(
            rows=("sample_id", "size"),
            genuine_three_tier_rows=("genuine_three_tier", "sum"),
            output_parity_rows=("output_parity", "sum"),
            storage_parity_rows=("storage_parity", "sum"),
            task_score=("task_score", "mean"),
            active_kv_bytes=("active_kv_bytes", "mean"),
            residual_kv_bytes=("residual_kv_bytes", "mean"),
            compression_time=("compression_time", "mean"),
            end_to_end_time=("end_to_end_time", "mean"),
            graph_nodes=("graph_nodes", "mean"),
            graph_edges=("graph_edges", "mean"),
            **{phase: (phase, "mean") for phase in PHASES},
        )
        .reset_index()
    )
    for phase in PHASES:
        summary[f"{phase}_speedup"] = 1.0 - summary[phase] / reference_means[phase]
    summary["compression_speedup"] = 1.0 - summary["compression_time"] / sum(
        reference_means.values()
    )
    summary["accepted"] = (
        (summary["genuine_three_tier_rows"] == summary["rows"])
        & (summary["output_parity_rows"] == summary["rows"])
        & (summary["storage_parity_rows"] == summary["rows"])
    )
    trials.to_parquet(root / "v38_trials.parquet", index=False)
    summary.to_parquet(root / "v38_summary.parquet", index=False)
    payload = {
        "classification": "exploratory_dirty_tree_retrospective_non_claim_validation",
        "reference_phase_means": reference_means,
        "rows": summary.to_dict(orient="records"),
    }
    (root / "v38_analysis.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (root / "v38_report.md").write_text(
        "# MosaicKV compression-bottleneck validation\n\n"
        "This is retrospective dirty-tree implementation evidence, not paper evidence.\n\n"
        "Candidates are accepted only when every sample used genuine three-tier "
        "`mosaickv_full`; fallback rows remain visible and are never averaged into a "
        "MosaicKV claim.\n\n"
        + summary.to_markdown(index=False)
        + "\n\n## Rejected fallback rows\n\n"
        + (
            "None.\n"
            if bool(trials[~trials["genuine_three_tier"]].empty)
            else trials.loc[
                ~trials["genuine_three_tier"],
                ["mode", "sample_id", "effective_method", "tier_mode", "fallback_reason"],
            ].to_markdown(index=False)
            + "\n"
        )
        + "\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
