#!/usr/bin/env python3
"""Read-only, label-blind audit of how much each saved policy changes Top-K.

This is a mechanism diagnostic, not a quality experiment or method selector.
It never reads predictions, references or individual scores into its summaries.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from evicastkv_analyze import collect, pairing

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]


def selection_delta(reference, candidate):
    """Count replacements, not twice-counted symmetric differences."""
    if len(set(reference)) != len(reference) or len(set(candidate)) != len(candidate):
        raise ValueError("duplicate selected source IDs")
    left, right = set(reference), set(candidate)
    if not left or len(left) != len(right):
        raise ValueError("selection cardinality differs; cannot call it an equal-capacity swap")
    return dict(
        selected_count=len(left),
        replacements=len(right - left),
        replacement_fraction=len(right - left) / len(left),
        jaccard=len(left & right) / len(left | right),
        changed=left != right,
    )


def summarize(records):
    groups = defaultdict(list)
    for record in records:
        groups[(record["workload"], record["variant"], record["budget_bytes"])].append(record)
    result = []
    for (workload, variant, cap), rows in sorted(groups.items()):
        entry = dict(
            workload=workload,
            variant=variant,
            budget_bytes=cap,
            inputs=len(rows),
            changed_inputs=sum(row["changed"] for row in rows),
        )
        for field in (
            "replacement_fraction",
            "replacements",
            "graph_nodes",
            "graph_edges",
            "routed_heads",
        ):
            values = [row[field] for row in rows]
            entry[field] = dict(
                mean=statistics.mean(values), minimum=min(values), maximum=max(values)
            )
        sources = sorted({source for row in rows for source in row["edge_sources"]})
        entry["edge_sources"] = {
            source: sum(row["edge_sources"].get(source, 0) for row in rows) for source in sources
        }
        removed = [
            row["cross_modal_edges_removed"]
            for row in rows
            if row["cross_modal_edges_removed"] is not None
        ]
        entry["cross_modal_edges_removed"] = sum(removed) if removed else None
        result.append(entry)
    return result


def build(root, output):
    source_sha = clean_source(SOURCE)
    if output.exists():
        raise FileExistsError("use a new immutable diagnostic output directory")
    lock_path = root / "protocol_lock.json"
    lock = json.loads(lock_path.read_text())
    clean_source(Path(lock["source_root"]), lock["source_sha"])
    gate_path = root / "analysis/mechanisms/GATE_mechanisms.json"
    if json.loads(gate_path.read_text())["status"] != "passed":
        raise ValueError("mechanism producer did not pass")
    spec, _, rows, errors = collect(root, "mechanisms")
    if errors:
        raise ValueError(f"incomplete mechanism producer: {len(errors)} errors")
    pairing(rows)  # Reject differences in actual initial/final/peak bytes or controls.
    expected_ids = {
        workload: set(cohort["development"]["sample_ids"])
        for workload, cohort in lock["cohorts"].items()
    }
    # Compare all registered treatments, never pick a winning fraction or seed.
    reference = {}
    activities = {}
    bindings = {str(lock_path): sha(lock_path), str(gate_path): sha(gate_path)}
    for row in rows:
        path = Path(row["mechanism_audit_path"])
        actual_sha = sha(path)
        if actual_sha != row["mechanism_audit_sha256"]:
            raise ValueError("mechanism activity hash changed")
        activity = json.loads(path.read_text())
        key = (row["workload"], row["sample_id"], row["budget_bytes"])
        activities[(*key, row["variant"])] = activity
        bindings[str(path)] = actual_sha
        if row["variant"] == "route-zero-control":
            reference[key] = activity["selected_source_ids"]
    records = []
    coverage = defaultdict(set)
    for row in rows:
        key = (row["workload"], row["sample_id"], row["budget_bytes"])
        if key not in reference:
            raise ValueError("missing same-input/same-capacity Prompt Top-K reference")
        activity = activities[(*key, row["variant"])]
        delta = selection_delta(reference[key], activity["selected_source_ids"])
        records.append(
            dict(
                workload=row["workload"],
                variant=row["variant"],
                budget_bytes=row["budget_bytes"],
                **delta,
                graph_nodes=activity.get("graph_nodes", 0),
                graph_edges=activity.get("graph_edges", 0),
                routed_heads=len(activity.get("graph_routes", ())),
                edge_sources=activity.get("edge_sources", {}),
                cross_modal_edges_removed=activity.get("cross_modal_edges_removed"),
            )
        )
        coverage[(row["workload"], row["variant"], row["budget_bytes"])].add(row["sample_id"])
    for (workload, _variant, _cap), ids in coverage.items():
        if ids != expected_ids[workload]:
            raise ValueError("incomplete per-treatment identity coverage")
    if sum(len(job["indices"]) * len(job["arms"]) for job in spec["jobs"]) != len(rows):
        raise ValueError("registered request count differs from completed rows")
    output.mkdir(parents=True)
    summary = summarize(records)
    write_once(output / "selection_reach.json", summary)
    write_once(
        output / "manifest.json",
        dict(
            source_sha=source_sha,
            producer_source_sha=lock["source_sha"],
            rows=len(rows),
            bindings=bindings,
            result_sha256=sha(output / "selection_reach.json"),
            scope="consumed_development_label_blind_selection_activity_only",
            benchmark_quality_evaluated=False,
            fresh_confirmation=False,
            algorithm_changed=False,
            gpu_inference_executed=False,
        ),
    )
    lines = [
        "# Selection reach diagnostic",
        "",
        "Saved consumed inputs only. No inference, benchmark scores, "
        "method selection, or confirmation.",
        "",
        "Replacement is |candidate minus Top-K| / |Top-K|, not symmetric difference / cache size.",
        "",
        "| Budget MiB | Variant | Inputs changed / N | Mean replacements (%) "
        "| Mean routed heads | Mean nodes |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for item in summary:
        lines.append(
            f"| {item['budget_bytes'] / 2**20:g} | {item['variant']} | "
            f"{item['changed_inputs']}/{item['inputs']} | "
            f"{100 * item['replacement_fraction']['mean']:.6f} | "
            f"{item['routed_heads']['mean']:.2f} | {item['graph_nodes']['mean']:.1f} |"
        )
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    build(args.root, args.output)
