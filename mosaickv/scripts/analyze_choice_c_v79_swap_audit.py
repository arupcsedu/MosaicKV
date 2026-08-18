#!/usr/bin/env python3
"""Derive a label-blind swap/output audit from immutable v79 traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, cast

ARMS = (
    "adaptive_swap2_text9975_gain150_b4_27_5",
    "adaptive_swap2_text995_gain125_b4_27_5",
)
TASKS = ("docvqa", "textvqa")
CONTROL = "v78_negative_control_b4_27_5"
FULL = "full_kv"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _traces(root: Path, task: str, arm: str) -> dict[str, dict[str, Any]]:
    directory = root / "traces" / f"v79-{task}-trial-00-{arm}"
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("*.json")):
        trace = cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))
        sample_id = str(trace["sample_id"])
        if sample_id in records:
            raise RuntimeError(f"duplicate trace for {task}/{arm}/{sample_id}")
        if "reference" in trace or "task_score" in trace:
            raise RuntimeError("swap audit input unexpectedly contains benchmark labels or scores")
        records[sample_id] = trace
    if not records:
        raise RuntimeError(f"no traces found for {task}/{arm}")
    return records


def _token_agreement(candidate: list[int], reference: list[int]) -> float:
    denominator = max(len(candidate), len(reference))
    if denominator == 0:
        return 1.0
    return sum(a == b for a, b in zip(candidate, reference, strict=False)) / denominator


def _correlation(first: list[float], second: list[float]) -> float | None:
    if len(first) < 2 or len(set(first)) < 2 or len(set(second)) < 2:
        return None
    first_mean = statistics.fmean(first)
    second_mean = statistics.fmean(second)
    numerator = math.fsum(
        (left - first_mean) * (right - second_mean)
        for left, right in zip(first, second, strict=True)
    )
    denominator = math.sqrt(
        math.fsum((value - first_mean) ** 2 for value in first)
        * math.fsum((value - second_mean) ** 2 for value in second)
    )
    return None if denominator == 0 else numerator / denominator


def _arm_audit(
    full: dict[str, dict[str, Any]],
    control: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if set(full) != set(control) or set(full) != set(candidate):
        raise RuntimeError("paired v79 trace arms do not cover identical samples")
    changed = improved = worsened = tied = 0
    swap_counts: list[float] = []
    changed_flags: list[float] = []
    per_head_available = True
    route_swap_records = 0
    for sample_id in sorted(full):
        full_tokens = [int(value) for value in full[sample_id]["generated_token_ids"]]
        control_tokens = [int(value) for value in control[sample_id]["generated_token_ids"]]
        candidate_tokens = [int(value) for value in candidate[sample_id]["generated_token_ids"]]
        capacity = cast(
            "dict[str, Any]",
            cast("dict[str, Any]", candidate[sample_id]["candidate_graph_routing"])[
                "external_stable_capacity"
            ],
        )
        swap_counts.append(float(capacity["adaptive_swap_count"]))
        route_records = capacity.get("routes_with_adaptive_swaps")
        if not isinstance(route_records, list):
            per_head_available = False
        else:
            route_swap_records += len(route_records)
        is_changed = candidate_tokens != control_tokens
        changed_flags.append(float(is_changed))
        if not is_changed:
            continue
        changed += 1
        control_fidelity = _token_agreement(control_tokens, full_tokens)
        candidate_fidelity = _token_agreement(candidate_tokens, full_tokens)
        if candidate_fidelity > control_fidelity:
            improved += 1
        elif candidate_fidelity < control_fidelity:
            worsened += 1
        else:
            tied += 1
    changed_swaps = [swaps for swaps, flag in zip(swap_counts, changed_flags, strict=True) if flag]
    unchanged_swaps = [
        swaps for swaps, flag in zip(swap_counts, changed_flags, strict=True) if not flag
    ]
    return {
        "sample_count": len(full),
        "sequence_changed_count": changed,
        "sequence_changed_fraction": changed / len(full),
        "token_fidelity_improved_count": improved,
        "token_fidelity_worsened_count": worsened,
        "token_fidelity_tied_count": tied,
        "mean_swap_count": statistics.fmean(swap_counts),
        "mean_swap_count_when_sequence_changed": (
            None if not changed_swaps else statistics.fmean(changed_swaps)
        ),
        "mean_swap_count_when_sequence_unchanged": (
            None if not unchanged_swaps else statistics.fmean(unchanged_swaps)
        ),
        "swap_count_sequence_change_correlation": _correlation(swap_counts, changed_flags),
        "per_head_swap_records_available": per_head_available,
        "per_head_swap_record_count": route_swap_records,
    }


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-markdown", required=True, type=Path)
    args = parser.parse_args()
    root = args.input_root.resolve()
    calibration = root / "analysis" / "label_blind_calibration.json"
    analysis = cast("dict[str, Any]", json.loads(calibration.read_text(encoding="utf-8")))
    if analysis.get("labels_accessed") is not False:
        raise RuntimeError("v79 calibration was not label-blind")

    tasks: dict[str, Any] = {}
    per_head_available = True
    for task in TASKS:
        full = _traces(root, task, FULL)
        control = _traces(root, task, CONTROL)
        arms: dict[str, Any] = {}
        for arm in ARMS:
            arms[arm] = _arm_audit(full, control, _traces(root, task, arm))
            per_head_available &= bool(arms[arm]["per_head_swap_records_available"])
        tasks[task] = arms

    result = {
        "schema_version": 1,
        "classification": "derived_label_blind_v79_swap_audit",
        "labels_accessed": False,
        "official_task_scores_accessed": False,
        "input_root": str(root),
        "input_calibration_sha256": _sha(calibration),
        "per_head_causal_attribution_available": per_head_available,
        "instrumentation_finding": (
            "v79 compact traces omitted routes_with_adaptive_swaps; exact per-head causal "
            "attribution is unavailable and must not be inferred from aggregate swap counts"
            if not per_head_available
            else "per-head swap records are available"
        ),
        "tasks": tasks,
        "design_decision": {
            "retire_ungated_broad_swaps": True,
            "require_exact_per_head_swap_pairs_in_future_traces": True,
            "risk_gate_signals": [
                "maximum_normalized_prompt_text_loss",
                "minimum_future_query_forecast_visual_gain_relative_to_text_loss",
            ],
        },
    }
    _write(args.output_json.resolve(), json.dumps(result, indent=2, sort_keys=True) + "\n")

    lines = [
        "# V79 label-blind adaptive-swap audit",
        "",
        "No benchmark references or official task scores were accessed.",
        "",
        "| Task | Arm | Changed | Improved | Worsened | Mean swaps | Swap/change correlation |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        for arm in ARMS:
            row = tasks[task][arm]
            correlation = row["swap_count_sequence_change_correlation"]
            lines.append(
                f"| {task} | {arm} | {row['sequence_changed_count']}/{row['sample_count']} | "
                f"{row['token_fidelity_improved_count']} | "
                f"{row['token_fidelity_worsened_count']} | "
                f"{row['mean_swap_count']:.2f} | "
                f"{'NA' if correlation is None else f'{correlation:.4f}'} |"
            )
    lines.extend(
        [
            "",
            "## Instrumentation conclusion",
            "",
            str(result["instrumentation_finding"]),
            "The next protocol therefore records exact accepted block pairs for every layer/head "
            "and rejects swaps unless both prompt-loss and future-query forecast-risk checks pass.",
        ]
    )
    _write(args.output_markdown.resolve(), "\n".join(lines) + "\n")
    print(json.dumps({"per_head_causal_attribution_available": per_head_available}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
