#!/usr/bin/env python3
"""Audit selection and prompt-mass telemetry without benchmark references."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, cast


def _reason_class(reason: str) -> str:
    if "prototype count exceeds maximum_prototypes_per_head" in reason:
        return "prototype_count_exceeds_configured_limit"
    if "prototype anchor-cover segment" in reason:
        return "anchor_cover_segment_incompatible"
    if not reason:
        return "missing"
    return reason.split(":", maxsplit=1)[0]


def _mean(values: list[float]) -> float | None:
    return None if not values else statistics.fmean(values)


def _trace_summary(paths: tuple[Path, ...]) -> dict[str, Any]:
    modes: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    prompt_status: Counter[str] = Counter()
    packed_padding: list[float] = []
    active_bytes: list[float] = []
    exact_mass: list[float] = []
    represented_mass: list[float] = []
    prototype_counts: list[int] = []
    selected_catalog_available = 0
    for path in paths:
        trace = cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))
        if trace.get("status") != "completed":
            raise RuntimeError(f"selection audit received failed trace: {path}")
        modes[str(trace.get("tier_mode", "missing"))] += 1
        reasons[_reason_class(str(trace.get("tier_reason", "")))] += 1
        prompt = cast("dict[str, Any]", trace.get("prompt_mass_audit") or {})
        prompt_status[str(prompt.get("status", "unavailable"))] += 1
        if prompt.get("exact_prompt_mass_fraction") is not None:
            exact_mass.append(float(prompt["exact_prompt_mass_fraction"]))
        if prompt.get("represented_prompt_mass_fraction") is not None:
            represented_mass.append(float(prompt["represented_prompt_mass_fraction"]))
        packed_padding.append(float(trace.get("packed_padding_kv_bytes", 0)))
        active_bytes.append(float(trace["active_kv_bytes"]))
        prototype_counts.append(
            sum(
                int(item["prototype_count"])
                for item in cast(
                    "list[dict[str, Any]]",
                    trace.get("prototype_diagnostics_by_kv_head") or [],
                )
            )
        )
        catalog = cast("dict[str, Any]", trace.get("selected_blocks") or {})
        if catalog.get("storage") != "omitted_non_diagnostic":
            selected_catalog_available += 1
    return {
        "sample_count": len(paths),
        "tier_modes": dict(sorted(modes.items())),
        "failure_reason_classes": dict(sorted(reasons.items())),
        "prompt_mass_status": dict(sorted(prompt_status.items())),
        "mean_active_kv_bytes": _mean(active_bytes),
        "mean_packed_padding_bytes": _mean(packed_padding),
        "mean_exact_prompt_mass_fraction": _mean(exact_mass),
        "mean_represented_prompt_mass_fraction": _mean(represented_mass),
        "mean_prototype_count": _mean([float(value) for value in prototype_counts]),
        "selected_block_catalog_available_samples": selected_catalog_available,
        "selected_block_membership_auditable": selected_catalog_available == len(paths),
    }


def analyze(output_root: Path) -> dict[str, Any]:
    calibration_path = output_root / "analysis/label_blind_calibration.json"
    calibration = cast(
        "dict[str, Any]", json.loads(calibration_path.read_text(encoding="utf-8"))
    )
    task_results = cast("dict[str, Any]", calibration["task_results"])
    candidate_ids = tuple(
        str(item["candidate_id"])
        for item in cast("list[dict[str, Any]]", calibration["candidate_ranking"])
    )
    task_audit: dict[str, Any] = {}
    for task in sorted(task_results):
        arms: dict[str, Any] = {}
        for arm in ("prompt_attention_topk", *candidate_ids):
            trace_root = output_root / "traces" / f"v74-{task}-trial-00-{arm}"
            paths = tuple(sorted(trace_root.glob("*.json")))
            expected = int(cast("dict[str, Any]", task_results[task])["sample_count"])
            if len(paths) != expected:
                raise RuntimeError(
                    f"{task}/{arm} expected {expected} traces, found {len(paths)}"
                )
            arms[arm] = _trace_summary(paths)
        task_audit[task] = arms
    result = {
        "schema_version": 1,
        "classification": "derived_label_blind_selection_prompt_mass_audit",
        "labels_accessed": False,
        "official_task_scores_accessed": False,
        "source_calibration": str(calibration_path.resolve()),
        "tasks": task_audit,
    }
    analysis_root = output_root / "analysis"
    json_path = analysis_root / "label_blind_selection_audit.json"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Label-blind selected-block and prompt-mass audit",
        "",
        "This derived audit used generated answers, cache telemetry, and traces only. "
        "It did not access benchmark references or official task scores.",
        "",
        "| Task | Arm | Tier mode | Mean active | Padding | Exact prompt mass | "
        "Represented prompt mass | Prototypes | Membership |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for task, arms_raw in task_audit.items():
        for arm, row_raw in cast("dict[str, dict[str, Any]]", arms_raw).items():
            mode = ", ".join(f"{key}:{value}" for key, value in row_raw["tier_modes"].items())
            exact = row_raw["mean_exact_prompt_mass_fraction"]
            represented = row_raw["mean_represented_prompt_mass_fraction"]
            lines.append(
                f"| {task} | {arm} | {mode} | "
                f"{float(row_raw['mean_active_kv_bytes']) / 1_000_000:.2f} MB | "
                f"{float(row_raw['mean_packed_padding_bytes']) / 1_000_000:.2f} MB | "
                f"{'n/a' if exact is None else f'{float(exact):.2%}'} | "
                f"{'n/a' if represented is None else f'{float(represented):.2%}'} | "
                f"{float(row_raw['mean_prototype_count']):.2f} | "
                f"{'available' if row_raw['selected_block_membership_auditable'] else 'omitted'} |"
            )
    lines.extend(
        [
            "",
            "The v74.5 compact traces predate aggregate prompt-mass telemetry and omit "
            "per-node selection catalogs. Those unavailable quantities are reported as "
            "`n/a`; they are not inferred. The next fresh calibration records prompt-mass "
            "fractions and selected exact/prototype block counts directly.",
        ]
    )
    (analysis_root / "label_blind_selection_audit.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    result = analyze(args.output_root.resolve())
    print(json.dumps({"tasks": sorted(result["tasks"]), "labels_accessed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
