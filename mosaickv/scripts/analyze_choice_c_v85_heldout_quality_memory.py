#!/usr/bin/env python3
"""Analyze v85 while preserving the sealed historical v73 analyzer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import yaml
from analyze_choice_c_v73_clean_quality_memory import analyze as analyze_v73


def analyze(
    spec_path: Path,
    output_root: Path,
    *,
    run_prefix: str = "v85",
    candidate_effective_method: str = "mosaickv_exact",
) -> dict[str, Any]:
    spec = cast("dict[str, Any]", yaml.safe_load(spec_path.read_text(encoding="utf-8")))
    if spec.get("experiment_id") != "choice_c_v85_v84_frozen_untouched_heldout_quality_memory":
        raise RuntimeError("unexpected v85 experiment id")
    if (
        spec.get("candidate_protocol_method") != "mosaickv_full"
        or spec.get("candidate_effective_method") != candidate_effective_method
    ):
        raise RuntimeError("v85 candidate method labels differ from the preregistration")
    expected_frozen_sha = str(spec["frozen_candidate_sha256"])
    expected_selected_candidate = str(spec["selected_candidate"])
    for task_entry in cast("list[dict[str, Any]]", spec["tasks"]):
        task = str(task_entry["task"])
        paths = tuple(sorted((output_root / "telemetry").glob(f"{run_prefix}-{task}-*.json")))
        if len(paths) != 1:
            raise RuntimeError(f"expected one {task} v85 telemetry file")
        telemetry = cast("dict[str, Any]", json.loads(paths[0].read_text(encoding="utf-8")))
        if telemetry.get("frozen_candidate_sha256") != expected_frozen_sha:
            raise RuntimeError(f"{task} telemetry uses a different frozen candidate")
        if telemetry.get("selected_candidate") != expected_selected_candidate:
            raise RuntimeError(f"{task} telemetry uses a different selected candidate")

    result = analyze_v73(
        spec_path,
        output_root,
        run_prefix=run_prefix,
        candidate_effective_method=candidate_effective_method,
    )
    gate = cast("dict[str, Any]", spec["primary_gate"])
    minimum_total_reduction = float(gate["minimum_active_plus_residual_memory_reduction_fraction"])
    task_results = cast("dict[str, dict[str, Any]]", result["task_results"])
    for task_result in task_results.values():
        total_pass = bool(
            float(task_result["active_plus_residual_reduction_fraction_vs_prompt_topk"])
            >= minimum_total_reduction
        )
        task_result["active_plus_residual_memory_gate_pass"] = total_pass
        task_result["quality_memory_gate_pass"] = bool(
            task_result["quality_memory_gate_pass"] and total_pass
        )
    overall_pass = all(
        bool(task_result["quality_memory_gate_pass"]) for task_result in task_results.values()
    )
    result["quality_memory_gate_pass"] = overall_pass
    result["v85_additional_gate"] = {
        "minimum_active_plus_residual_memory_reduction_fraction": minimum_total_reduction,
        "all_tasks_pass": all(
            bool(task_result["active_plus_residual_memory_gate_pass"])
            for task_result in task_results.values()
        ),
    }
    analysis_root = output_root / "analysis"
    (analysis_root / "heldout_quality_memory_gate.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = [
        "# V85 frozen held-out quality-memory validation",
        "",
        f"- Overall gate: **{'PASS' if overall_pass else 'FAIL'}**",
        "- Both active and active-plus-residual reductions must reach 30%.",
        "- Paired quality gates are inherited unchanged from the sealed analyzer.",
        "- Latency is informational and cannot reject the run.",
        "",
        "| Task | Samples | Active reduction | Active+residual reduction | Gate |",
        "|---|---:|---:|---:|:---:|",
    ]
    for task, task_result in task_results.items():
        report.append(
            f"| {task} | {task_result['sample_count']} | "
            f"{float(task_result['active_memory_reduction_fraction_vs_prompt_topk']):.2%} | "
            f"{float(task_result['active_plus_residual_reduction_fraction_vs_prompt_topk']):.2%} | "
            f"{'PASS' if task_result['quality_memory_gate_pass'] else 'FAIL'} |"
        )
    (analysis_root / "heldout_quality_memory_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-prefix", default="v85")
    parser.add_argument("--candidate-effective-method", default="mosaickv_exact")
    args = parser.parse_args()
    result = analyze(
        args.spec.resolve(),
        args.output_root.resolve(),
        run_prefix=str(args.run_prefix),
        candidate_effective_method=str(args.candidate_effective_method),
    )
    print(json.dumps({"quality_memory_gate_pass": result["quality_memory_gate_pass"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
