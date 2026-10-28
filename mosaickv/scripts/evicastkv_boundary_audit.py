#!/usr/bin/env python3
"""CPU-only qualification of saved A/B diagnostics, never a new quality claim."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from evicastkv_analyze import collect
from evicastkv_campaign import SOURCE

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once


def qualify(pairs):
    """Every method inherits its input's independent FullKV chunking control."""
    seen, controls = set(), {}
    for row, boundary in pairs:
        identity = (row["workload"], row["sample_id"])
        key = (*identity, row["method"], row["budget_bytes"])
        if key in seen:
            raise ValueError("duplicate boundary pair")
        seen.add(key)
        if row["method"] == "full_kv":
            if identity in controls:
                raise ValueError("duplicate FullKV boundary control")
            controls[identity] = (
                boundary.get("status") == "completed"
                and boundary.get("fullkv_chunked_logit_control_passed") is True
            )
    if any((r["workload"], r["sample_id"]) not in controls for r, _ in pairs):
        raise ValueError("missing independent FullKV boundary control")
    records = []
    for row, boundary in pairs:
        control = controls[row["workload"], row["sample_id"]]
        records.append(
            dict(
                workload=row["workload"],
                sample_id=row["sample_id"],
                method=row["method"],
                budget_bytes=row["budget_bytes"],
                execution_status=boundary.get("status", "missing"),
                independent_fullkv_control_passed=control,
                numerically_qualified=control and boundary.get("status") == "completed",
                first_token_changed=boundary.get("first_token_changed"),
                answer_changed=boundary.get("A_B_answer_changed"),
                first_token_A=boundary.get("first_token_A"),
                first_token_B=boundary.get("first_token_id"),
                first_probability_A=boundary.get("first_token_A_probability"),
                first_probability_B=boundary.get("first_token_B_probability"),
                A_B_max_logit_difference=boundary.get("A_B_first_logit_max_difference"),
                output_prefix_B=json.dumps(boundary.get("token_ids", [])[:8]),
                prediction_A=row.get("answer"),
                prediction_B=boundary.get("answer"),
                capacity_stratum=boundary.get("capacity_stratum"),
                directly_byte_matched_to_A=boundary.get("directly_byte_matched_to_A"),
            )
        )
    return records, dict(
        status="controls_passed"
        if records and all(r["numerically_qualified"] for r in records)
        else "BLOCKED",
        planned_pairs=len(pairs),
        completed_pairs=sum(r["execution_status"] == "completed" for r in records),
        fullkv_controls=len(controls),
        passing_fullkv_controls=sum(controls.values()),
        qualified_pairs=sum(r["numerically_qualified"] for r in records),
        quality_inference_authorized=False,
        native_B_quality_not_established=True,
        A_primary_gate_not_rewritten=True,
        note=(
            "Execution completion is not numerical validation. B is a changed "
            "observation/capacity protocol, not a primary score or isolated causal effect. "
            "Failed chunked-FullKV controls preclude interpreting B as compression quality."
        ),
    )


def audit(root, boundary_export, expected_export_sha, output):
    revision = clean_source(SOURCE)
    if sha(boundary_export) != expected_export_sha:
        raise ValueError("boundary export hash changed")
    gate = root / "analysis/scoring/GATE_scoring.json"
    if json.loads(gate.read_text())["status"] != "passed":
        raise ValueError("complete primary scoring gate required")
    _, workers, _, errors = collect(root, "scoring")
    if errors:
        raise ValueError("incomplete primary rows; do not create a surviving intersection")
    exported = json.loads(boundary_export.read_text())["records"]
    snapshots = {item["path"]: item for item in exported}
    if len(snapshots) != len(exported):
        raise ValueError("duplicate exported boundary paths")
    pairs, used = [], set()
    bindings = {str(boundary_export): expected_export_sha, str(gate): sha(gate)}
    for worker in workers:
        for binding in worker["records"]:
            raw = Path(binding["path"])
            boundary_path = raw.parent.parent / "boundary" / raw.name
            item = snapshots.get(str(boundary_path))
            if item is None or sha(boundary_path) != item["sha256"]:
                raise ValueError("missing or changed B record")
            boundary = json.loads(boundary_path.read_text())
            if {k: v for k, v in item.items() if k not in {"path", "sha256"}} != boundary:
                raise ValueError("exported boundary contents changed")
            pairs.append((json.loads(raw.read_text()), boundary))
            used.add(str(boundary_path))
            bindings[str(raw)] = binding["sha256"]
            bindings[str(boundary_path)] = item["sha256"]
    if used != set(snapshots):
        raise ValueError("extra boundary records outside registered primary cohort")
    records, summary = qualify(pairs)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "first_token_sensitivity.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    write_once(
        output / "GATE_boundary_B.json",
        summary
        | dict(
            analysis_source_sha=revision,
            runtime_source_sha=json.loads((root / "protocol_lock.json").read_text())["source_sha"],
            protocol_sha256=sha(root / "protocol_lock.json"),
            source_bindings=bindings,
            GPU_requests=0,
        ),
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--boundary-export", type=Path, required=True)
    parser.add_argument("--boundary-export-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.root, args.boundary_export, args.boundary_export_sha, args.output)
