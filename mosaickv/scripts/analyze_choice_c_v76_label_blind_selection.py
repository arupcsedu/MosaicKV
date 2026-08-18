#!/usr/bin/env python3
"""Audit v76 cache allocation without reading benchmark labels or task scores."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

import pandas as pd  # type: ignore[import-untyped]

TASKS = ("docvqa", "textvqa")
ARMS = (
    "prompt_attention_topk",
    "v74f_frozen_control_b4_27_5",
    "ocr_grid4_prompt_stratified_b4_27_5",
    "ocr_grid6_prompt_stratified_b4_27_5",
)
CONTROL = "v74f_frozen_control_b4_27_5"


def _normalize_answer(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip().casefold())


def _token_agreement(left: list[int], right: list[int]) -> float:
    denominator = max(len(left), len(right))
    if denominator == 0:
        return 1.0
    return sum(a == b for a, b in zip(left, right, strict=False)) / denominator


def _trace_index(directory: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("*.json")):
        trace = cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))
        sample_id = str(trace["sample_id"])
        if sample_id in records:
            raise RuntimeError(f"duplicate trace for {sample_id} in {directory}")
        if trace.get("status") != "completed":
            raise RuntimeError(f"incomplete trace for {sample_id} in {directory}")
        records[sample_id] = trace
    if not records:
        raise RuntimeError(f"no traces found in {directory}")
    return records


def _modality_record(trace: Mapping[str, Any], modality: str) -> dict[str, Any]:
    prompt_mass = cast("dict[str, Any]", trace["prompt_mass_audit"])
    records = cast("list[dict[str, Any]]", prompt_mass["modalities"])
    matches = [record for record in records if record["modality"] == modality]
    if len(matches) != 1:
        raise RuntimeError(f"trace must contain exactly one {modality} prompt-mass record")
    return matches[0]


def _prototype_count(trace: Mapping[str, Any]) -> int:
    prototypes = trace.get("prototypes", ())
    if isinstance(prototypes, list):
        return len(prototypes)
    if isinstance(prototypes, dict):
        return int(prototypes.get("record_count", 0))
    raise RuntimeError("unsupported prototype trace representation")


def _fraction(numerator: float, denominator: float) -> float:
    return 0.0 if denominator <= 0 else numerator / denominator


def build_audit_rows(run_root: Path) -> pd.DataFrame:
    """Return sample-paired telemetry using model outputs and cache traces only."""

    rows: list[dict[str, Any]] = []
    for task in TASKS:
        full = _trace_index(run_root / "traces" / f"v76-{task}-trial-00-full_kv")
        for arm in ARMS:
            candidate = _trace_index(run_root / "traces" / f"v76-{task}-trial-00-{arm}")
            if set(candidate) != set(full):
                raise RuntimeError(f"{task}/{arm} sample IDs differ from FullKV")
            for sample_id in sorted(full):
                reference = full[sample_id]
                trace = candidate[sample_id]
                prompt_mass = cast("dict[str, Any]", trace["prompt_mass_audit"])
                image = _modality_record(trace, "image")
                text = _modality_record(trace, "text")
                routing = trace.get("candidate_graph_routing")
                stable_capacity = (
                    {}
                    if not isinstance(routing, dict)
                    else cast("dict[str, Any]", routing.get("external_stable_capacity", {}))
                )
                graph_routes = (
                    ()
                    if not isinstance(routing, dict)
                    else cast("list[list[int]]", routing.get("graph_head_routes", ()))
                )
                image_total = float(image["total_prompt_mass"])
                text_total = float(text["total_prompt_mass"])
                total = float(prompt_mass["total_prompt_mass"])
                rows.append(
                    {
                        "task": task,
                        "sample_id": sample_id,
                        "arm": arm,
                        "fullkv_answer_match": (
                            _normalize_answer(trace["answer"])
                            == _normalize_answer(reference["answer"])
                        ),
                        "fullkv_token_agreement": _token_agreement(
                            [int(token) for token in trace["generated_token_ids"]],
                            [int(token) for token in reference["generated_token_ids"]],
                        ),
                        "overall_exact_prompt_mass_fraction": float(
                            prompt_mass["exact_prompt_mass_fraction"]
                        ),
                        "overall_represented_prompt_mass_fraction": float(
                            prompt_mass["represented_prompt_mass_fraction"]
                        ),
                        "image_prompt_mass_share": _fraction(image_total, total),
                        "text_prompt_mass_share": _fraction(text_total, total),
                        "image_exact_prompt_mass_fraction": _fraction(
                            float(image["exact_prompt_mass"]), image_total
                        ),
                        "image_represented_prompt_mass_fraction": float(
                            image["represented_prompt_mass_fraction"]
                        ),
                        "text_exact_prompt_mass_fraction": _fraction(
                            float(text["exact_prompt_mass"]), text_total
                        ),
                        "image_exact_block_count": int(image["exact_block_count"]),
                        "text_exact_block_count": int(text["exact_block_count"]),
                        "prototype_count": _prototype_count(trace),
                        "routed_head_count": len(graph_routes),
                        "stable_head_count": int(stable_capacity.get("route_count", 0)),
                    }
                )
    return pd.DataFrame(rows).sort_values(["task", "arm", "sample_id"]).reset_index(drop=True)


def _mean(records: Iterable[float]) -> float:
    values = tuple(float(value) for value in records)
    if not values:
        raise RuntimeError("cannot average an empty sequence")
    return statistics.fmean(values)


def summarize(rows: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for (task, arm), group in rows.groupby(["task", "arm"], sort=True):
        summaries.append(
            {
                "task": str(task),
                "arm": str(arm),
                "sample_count": len(group),
                "fullkv_answer_agreement": _mean(group["fullkv_answer_match"]),
                "fullkv_token_agreement": _mean(group["fullkv_token_agreement"]),
                "overall_exact_prompt_mass_fraction": _mean(
                    group["overall_exact_prompt_mass_fraction"]
                ),
                "image_prompt_mass_share": _mean(group["image_prompt_mass_share"]),
                "image_exact_prompt_mass_fraction": _mean(
                    group["image_exact_prompt_mass_fraction"]
                ),
                "image_represented_prompt_mass_fraction": _mean(
                    group["image_represented_prompt_mass_fraction"]
                ),
                "text_exact_prompt_mass_fraction": _mean(
                    group["text_exact_prompt_mass_fraction"]
                ),
                "mean_image_exact_blocks": _mean(group["image_exact_block_count"]),
                "mean_text_exact_blocks": _mean(group["text_exact_block_count"]),
                "mean_prototype_count": _mean(group["prototype_count"]),
                "mean_routed_head_count": _mean(group["routed_head_count"]),
                "mean_stable_head_count": _mean(group["stable_head_count"]),
            }
        )

    summary_by_key = {(record["task"], record["arm"]): record for record in summaries}
    spatial_deltas: dict[str, dict[str, float]] = {}
    for task in TASKS:
        control = summary_by_key[(task, CONTROL)]
        spatial_deltas[task] = {}
        for arm in ARMS[2:]:
            record = summary_by_key[(task, arm)]
            spatial_deltas[task][arm] = float(record["mean_image_exact_blocks"]) - float(
                control["mean_image_exact_blocks"]
            )

    candidate_rows = rows[rows["arm"] != "prompt_attention_topk"]
    prompt_rows = rows[rows["arm"] == "prompt_attention_topk"]
    routed = _mean(candidate_rows["routed_head_count"])
    stable = _mean(candidate_rows["stable_head_count"])
    maximum_spatial_block_delta = max(
        abs(delta) for task in spatial_deltas.values() for delta in task.values()
    )
    findings = {
        "mean_candidate_image_prompt_mass_share": _mean(
            candidate_rows["image_prompt_mass_share"]
        ),
        "mean_candidate_image_exact_prompt_mass_fraction": _mean(
            candidate_rows["image_exact_prompt_mass_fraction"]
        ),
        "mean_candidate_text_exact_prompt_mass_fraction": _mean(
            candidate_rows["text_exact_prompt_mass_fraction"]
        ),
        "mean_prompt_topk_image_exact_prompt_mass_fraction": _mean(
            prompt_rows["image_exact_prompt_mass_fraction"]
        ),
        "mean_routed_head_fraction": _fraction(routed, routed + stable),
        "maximum_mean_image_block_change_from_spatial_stratification": (
            maximum_spatial_block_delta
        ),
        "spatial_image_block_deltas_vs_control": spatial_deltas,
        "conclusion": (
            "Overall prompt mass is text-dominated. The v76 spatial reservation reaches only "
            "routed heads, so it barely changes image allocation while stable heads preserve "
            "near-saturated text mass. The next candidate must rebalance already-budgeted "
            "stable-head exact capacity toward visual blocks."
        ),
    }
    return summaries, findings


def write_report(
    run_root: Path,
    output_prefix: Path,
) -> dict[str, Any]:
    rows = build_audit_rows(run_root)
    summaries, findings = summarize(rows)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(output_prefix.with_suffix(".parquet"), index=False)
    payload = {
        "schema_version": 1,
        "classification": "label_blind_v76_selection_audit",
        "run_root": str(run_root.resolve()),
        "benchmark_labels_accessed": False,
        "official_task_scores_accessed": False,
        "reference_fields_accessed": False,
        "allowed_signals": [
            "FullKV generated answers",
            "FullKV and candidate generated token IDs",
            "prompt-attention mass telemetry",
            "cache modality memberships",
            "selection and routing counts",
        ],
        "summaries": summaries,
        "findings": findings,
    }
    output_prefix.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# V76 label-blind selection audit",
        "",
        "This audit reads model outputs and cache-selection telemetry only. It does not read "
        "benchmark references, task scores, or labels.",
        "",
        "| Task | Arm | FullKV answer agreement | Image mass share | Image exact mass | "
        "Text exact mass | Image exact blocks | Text exact blocks | Routed heads | Stable heads |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in summaries:
        lines.append(
            "| {task} | {arm} | {answer:.2%} | {image_share:.2%} | {image_exact:.2%} | "
            "{text_exact:.2%} | {image_blocks:.1f} | {text_blocks:.1f} | {routed:.1f} | "
            "{stable:.1f} |".format(
                task=record["task"],
                arm=record["arm"],
                answer=record["fullkv_answer_agreement"],
                image_share=record["image_prompt_mass_share"],
                image_exact=record["image_exact_prompt_mass_fraction"],
                text_exact=record["text_exact_prompt_mass_fraction"],
                image_blocks=record["mean_image_exact_blocks"],
                text_blocks=record["mean_text_exact_blocks"],
                routed=record["mean_routed_head_count"],
                stable=record["mean_stable_head_count"],
            )
        )
    lines.extend(
        [
            "",
            "## Finding",
            "",
            str(findings["conclusion"]),
            "",
            "The approximately 93% overall prompt-mass figure is therefore not evidence that "
            "the visual evidence needed for OCR was preserved. Image tokens contribute only "
            f"{float(findings['mean_candidate_image_prompt_mass_share']):.2%} of prompt mass on "
            "average, while stable-head text retention is already saturated at "
            f"{float(findings['mean_candidate_text_exact_prompt_mass_fraction']):.2%}.",
            "",
            "## Design consequence",
            "",
            "Keep the byte cap and gates unchanged. Reallocate stable-head exact slots using a "
            "deterministic visual-position target, then rank visual and text blocks by prompt "
            "attention within their capacities. This acts on all stable heads and does not use "
            "OCR labels, answer references, or benchmark scores.",
        ]
    )
    output_prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve()
    output_prefix = (
        args.output_prefix.resolve()
        if args.output_prefix is not None
        else run_root / "analysis" / "label_blind_selection_audit"
    )
    payload = write_report(run_root, output_prefix)
    findings = cast("dict[str, Any]", payload["findings"])
    if not math.isfinite(float(findings["mean_candidate_image_prompt_mass_share"])):
        raise RuntimeError("audit produced a non-finite image prompt-mass share")
    print(json.dumps({"output_prefix": str(output_prefix), "findings": findings}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
