#!/usr/bin/env python3
"""Read-only, label-free mechanism audit of completed development runs.

EviCastKV is the paper name for MosaicKV. Historical identifiers and producer
SHAs are retained, not rewritten. This command cannot submit jobs or tune a
candidate. It writes a new derived directory and never edits its inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import Counter
from pathlib import Path

import pandas as pd

SOURCE = Path(__file__).resolve().parents[2]
PREFIX = Path("/scratch/djy8hg/runs/mosaickv/prefix-native-intermediate-20261029-v2-retry1")
FOX = Path("/scratch/djy8hg/runs/evicastkv/fox-installed-guard-20261029-v2-retry1")
PHASES = (
    "cache_state_construction",
    "forecast",
    "planner_preparation",
    "graph",
    "utility_and_selection",
    "tier_construction",
    "cache_packing",
    "logit_stability",
    "progressive_eviction",
    "hierarchical_eviction",
    "repair_transfer",
    "repair_redecode",
)


def paper_name(identifier):
    """Display-only alias, never an execution/configuration identity rewrite."""
    return {
        "mosaickv_exact": "EviCastKV exact",
        "mosaickv_proto": "EviCastKV prototype variant",
        "mosaickv_full": "EviCastKV full variant",
    }.get(identifier, identifier)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Evidence:
    def __init__(self):
        self.hashes = {}

    def bind(self, path):
        path = Path(path).resolve(strict=True)
        value = sha(path)
        if str(path) in self.hashes and self.hashes[str(path)] != value:
            raise ValueError(f"input changed while auditing: {path}")
        self.hashes[str(path)] = value
        return path

    def read(self, path):
        return json.loads(self.bind(path).read_text())


def guard_summary(traces):
    """Only source/route/guard telemetry is used; no answers or references."""
    counts = Counter()
    routes = Counter()
    for trace in traces:
        activity = trace["experimental_evidence_policy"]
        guard = activity["installed_output_guard"]
        if activity["grouped_query_heads_averaged"]:
            raise ValueError("guarded study unexpectedly pooled GQA query heads")
        expected = {tuple(route) for route in trace["candidate_graph_routing"]["graph_head_routes"]}
        records = guard["routes"]
        if {(r["layer"], r["kv_head"]) for r in records} != expected:
            raise ValueError("graph and guard route identities disagree")
        counts["requests"] += 1
        counts["requests_with_changes"] += guard["changed_positions"] > 0
        counts["changed_positions"] += guard["changed_positions"]
        counts["whole_cache_fallbacks"] += guard["whole_cache_fallback"]
        for row in records:
            routes[f"{row['layer']}:{row['kv_head']}"] += 1
            counts["routes"] += 1
            counts["accepted_routes"] += row["proposal_accepted"]
            counts["rolled_back_routes"] += row["rolled_back"]
            counts["mass_guard_failures"] += not row["mass_guard_passed"]
            counts["output_guard_failures"] += not row["output_guard_passed"]
    return dict(counts=dict(counts), selected_routes=dict(routes))


def routing_audit(root, evidence):
    path = evidence.bind(root / "analysis/installed_guard_development/observations.parquet")
    # Column projection prevents benchmark answers/references/scores entering this audit.
    data = pd.read_parquet(
        path, columns=["variant", "budget_bytes", "trace_path", "trace_sha256", "sample_id"]
    )
    data = data[data.variant == "evidence-output-guarded-v2"]
    if data.empty or data.duplicated(["sample_id", "budget_bytes"]).any():
        raise ValueError("missing or duplicate guarded requests")
    result = {}
    for cap, group in data.groupby("budget_bytes"):
        traces = []
        for row in group.itertuples():
            path = evidence.bind(row.trace_path)
            if sha(path) != row.trace_sha256:
                raise ValueError("saved trace hash mismatch")
            traces.append(evidence.read(path))
        result[str(int(cap))] = guard_summary(traces)
    return dict(
        by_initial_cap_bytes=result,
        full_route_entropy_arrays_saved=False,
        empirical_entropy_winner_replay="unavailable_without_full_head_scores",
        distinction="synthetic mapping/ranking tests are not an empirical winner replay",
        benchmark_answers_references_and_scores_used=False,
    )


def scalar_phases(trace, compression_seconds):
    """Non-overlapping compression phases; do not add nested planner subphases."""
    values = {name: float(trace["timing_breakdown"][name]) for name in PHASES}
    if any(not math.isfinite(v) or v < 0 for v in values.values()):
        raise ValueError("nonfinite or negative phase")
    values["compression_time"] = float(compression_seconds)
    values["compression_phase_sum"] = sum(values[name] for name in PHASES)
    values["compression_accounting_difference"] = (
        values["compression_time"] - values["compression_phase_sum"]
    )
    # Instrumentation is excluded from inference, and is not a compression phase.
    values["excluded_trace_instrumentation"] = float(
        trace["timing_breakdown"]["trace_instrumentation"]
    )
    return values


def phase_audit(root, evidence):
    records = []
    for task_index, task in enumerate(("llava_description", "mmvet_prefixkv")):
        directory = root / f"timing/{task_index:02d}"
        for trial in sorted((directory / "trials").glob("r*.json")):
            row = evidence.read(trial)
            if row["repeat"] < 0:
                continue
            traces = list((directory / "traces" / trial.stem).rglob("*.json"))
            if len(traces) != 1:
                raise ValueError("timing trial must have exactly one generation trace")
            trace = evidence.read(traces[0])
            if trace["execution_phase"] != "generation":
                raise ValueError("teacher forcing is not generation timing")
            records.append(
                dict(
                    task=task,
                    method=row["method"],
                    paper_method=paper_name(row["method"]),
                    cap_mib=row["cap_mib"],
                    repeat=row["repeat"],
                    **scalar_phases(trace, row["compression_time"]),
                )
            )
    data = pd.DataFrame(records)
    if data.empty or data.duplicated(["task", "method", "cap_mib", "repeat"]).any():
        raise ValueError("empty or duplicate timing observations")
    sizes = data.groupby(["task", "method", "cap_mib"]).size()
    if len(sizes) != 8 or not (sizes == 20).all():
        raise ValueError("incomplete registered 20-repeat timing study")
    return data


def prefix_audit(root, evidence):
    plan = evidence.read(root / "plan.json")
    configs = {}
    layers = []
    for job in plan["jobs"]:
        if job["method"] != "prefixkv_reimpl":
            continue
        config_path = Path(job["config"])
        cfg = evidence.read(config_path)
        prefix = cfg["prefixkv"]
        evidence.bind(prefix["profile_path"])
        configs[str(config_path)] = dict(
            model=cfg["model"],
            profile_sha256=sha(prefix["profile_path"]),
            decode_policy=prefix["decode_policy"],
            profile_reference_retention_ratio=prefix["profile_reference_retention_ratio"],
            rescale_offline_profile_to_byte_budget=prefix["rescale_offline_profile_to_byte_budget"],
            allocation_policy=prefix["allocation_policy"],
            question_score_weight=prefix["question_score_weight"],
            max_new_tokens=cfg["generation"]["max_new_tokens"],
        )
        directory = root / f"outputs/{job['index']:03d}/traces"
        for path in sorted(directory.rglob("*.json")):
            trace = evidence.read(path)
            if trace["execution_phase"] != "generation":
                continue
            info = trace["prefixkv"]
            summary = info["layer_capacity_summary"]
            layers.append(
                dict(
                    task=job["task"],
                    cap_mib=job["cap_mib"],
                    sample_id=trace["sample_id"],
                    implementation_label=info["implementation_label"],
                    paper_native_algorithm=info["paper_native_algorithm"],
                    minimum_layer_positions=summary["minimum"],
                    maximum_layer_positions=summary["maximum"],
                    total_layer_positions=summary["total"],
                    layers_below_eviction_distance=summary["layers_below_eviction_distance"],
                    full_selected_position_lists_saved=isinstance(info["layers"], list),
                    actual_decode_evictions=sum(
                        r["removed_logical_position"] is not None
                        for r in trace["prefixkv_decode_events"]
                    ),
                )
            )
    frame = pd.DataFrame(layers)
    if len(frame) != 192 or frame.duplicated(["task", "cap_mib", "sample_id"]).any():
        raise ValueError("incomplete PrefixKV generation audit")
    return dict(
        configs=configs,
        classification="byte-rescaled append-only selection-only reimplementation",
        official_path_parity_for_this_regime="not_established_by_historical_native_parity",
        low_budget_rouge_cause="not_proven_from_saved_aggregate_and_light_trace_evidence",
        no_metric_pooling=True,
        no_baseline_algorithm_changed=True,
    ), frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix-root", type=Path, default=PREFIX)
    parser.add_argument("--fox-root", type=Path, default=FOX)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("write-once derived output required")
    for root in (args.prefix_root, args.fox_root):
        if args.output.resolve().is_relative_to(root.resolve()):
            raise ValueError("derived audit must be outside immutable input runs")
    evidence = Evidence()
    routing = routing_audit(args.fox_root, evidence)
    prefix, layer_rows = prefix_audit(args.prefix_root, evidence)
    phases = phase_audit(args.prefix_root, evidence)
    means = phases.groupby(["task", "method", "paper_method", "cap_mib"], as_index=False).mean(
        numeric_only=True
    )
    args.output.mkdir(parents=True)
    phases.to_parquet(args.output / "compression_trials.parquet", index=False)
    means.to_parquet(args.output / "compression_summary.parquet", index=False)
    layer_rows.to_parquet(args.output / "prefix_layer_capacity.parquet", index=False)
    payload = dict(
        schema_version=1,
        measurement_type="derived",
        result_category="unified_comparison",
        scope="CPU-only retrospective mechanism audit; not independent confirmation",
        system_name="EviCastKV",
        historical_system_name="MosaicKV",
        same_system_not_identical_variant_configurations=True,
        historical_identifiers_preserved=True,
        routing=routing,
        prefix=prefix,
        new_gpu_jobs=0,
        thresholds_or_budgets_changed=False,
        paper_campaign_authorized=False,
    )
    (args.output / "audit.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    report = [
        "# EviCastKV CPU mechanism audit",
        "",
        "EviCastKV is the paper name for MosaicKV, not another baseline. Historical "
        "method IDs and producer commits are unchanged. Guarded and frozen runs are "
        "different configurations of that same system.",
        "",
        "## Routing",
        "",
        "```json",
        json.dumps(routing, indent=2),
        "```",
        "",
        "Full per-route entropy arrays were not saved. The source and synthetic CPU "
        "tests can audit mapping/ranking, but cannot certify that the observed real "
        "head was the correct entropy winner. Constant selection is not by itself "
        "proof of an indexing defect.",
        "",
        "## PrefixKV boundary",
        "",
        "These are byte-rescaled append-only `prefixkv_selection_only_reimpl` rows, "
        "not official PrefixKV. The upstream native path uses different profile "
        "rounding and decode eviction. Historical native parity is not parity for "
        "this transferred-profile, FP16 HF operating regime. PPL and reference "
        "ROUGE-L remain separate. The low-budget ROUGE cause is not proven; do not "
        "claim an attention-sink, layer-starvation, or scoring bug without evidence.",
        "",
        "## Measured compression breakdown",
        "",
        "Means over 20 technical repetitions on one fixed input per task. Units: "
        "seconds. Planner subphases are nested and must not be added twice. Trace "
        "instrumentation is excluded from inference and listed separately.",
        "",
        "```text",
        means.drop(columns=["repeat"]).to_string(index=False),
        "```",
        "",
        "## Bounded next engineering actions",
        "",
        "1. Preserve negative development gates and consumed identities. No GPU retry "
        "or large campaign is authorized by this audit.",
        "2. Before changing the router, record all entropy scores and the selected "
        "score/rank in a separately versioned consumed-input diagnostic; keep the "
        "existing guard and caps fixed.",
        "3. Differential-test upstream full-query scoring/selection on CPU; a new "
        "model-level parity assertion still needs matching checkpoint, precision, "
        "profile, prompt, and decode policy. Do not optimize a baseline against labels.",
        "4. Optimize only the measured dominant construction boundary. Any future "
        "fast path must preserve exact tensor values, logical positions, mandatory "
        "tokens, physical bytes, and the measured selection, and apply equally to "
        "compatible methods. Do not promise quality gains from an execution rewrite.",
        "5. Independent quality validation remains blocked pending a meaningful "
        "candidate improvement and separately frozen disjoint protocol.",
    ]
    (args.output / "report.md").write_text("\n".join(report) + "\n")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=SOURCE, text=True)
    manifest = dict(
        source_sha=revision,
        git_dirty=bool(status),
        script_sha256=sha(__file__),
        input_sha256=evidence.hashes,
        artifact_sha256={p.name: sha(p) for p in args.output.iterdir()},
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(output=str(args.output), input_files=len(evidence.hashes))))


if __name__ == "__main__":
    main()
