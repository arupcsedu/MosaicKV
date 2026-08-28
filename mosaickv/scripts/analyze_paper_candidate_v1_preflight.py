#!/usr/bin/env python3
"""Evaluate the two preregistered gates from three immutable preflight workers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any, cast

from mosaickv.evaluation.paper_candidate import PAPER_CANDIDATE_TASKS


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_immutable(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != encoded:
        raise RuntimeError(f"refusing to mutate immutable preflight analysis: {path}")
    if not path.exists():
        path.write_text(encoded, encoding="utf-8")


def _required_row(value: object, *, method: str, sample_id: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{method}/{sample_id} row is not a mapping")
    row = cast("dict[str, Any]", value)
    if row.get("sample_id") != sample_id or row.get("status") != "completed":
        raise RuntimeError(f"{method}/{sample_id} is incomplete")
    for field in ("task_score", "active_kv_bytes"):
        observed = row.get(field)
        if isinstance(observed, bool) or not isinstance(observed, (int, float)):
            raise RuntimeError(f"{method}/{sample_id} lacks numeric {field}")
        if not math.isfinite(float(observed)):
            raise RuntimeError(f"{method}/{sample_id} has non-finite {field}")
    if int(row["active_kv_bytes"]) < 1:
        raise RuntimeError(f"{method}/{sample_id} has invalid active bytes")
    if not isinstance(row.get("reference"), str):
        raise RuntimeError(f"{method}/{sample_id} lacks reference provenance")
    return row


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Paper candidate v1 GPU preflight",
        "",
        f"Overall gate: **{'PASS' if report['passed'] else 'FAIL'}**",
        "",
        "| Task | N | Prompt Top-K score | MosaicKV score | Delta | "
        "Mean active reduction | Minimum active reduction | Memory | Quality |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for task in cast("list[dict[str, Any]]", report["tasks"]):
        lines.append(
            "| {task} | {sample_count} | {baseline_score:.6f} | "
            "{candidate_score:.6f} | {score_delta:+.6f} | "
            "{mean_active_reduction_fraction:.2%} | "
            "{minimum_sample_active_reduction_fraction:.2%} | {memory} | {quality} |".format(
                **task,
                memory="PASS" if task["memory_passed"] else "FAIL",
                quality="PASS" if task["quality_passed"] else "FAIL",
            )
        )
    lines.extend(
        [
            "",
            "Only measured attention-active K/V tensor-storage bytes are used for the memory gate.",
            "Task metrics are evaluated separately and are not averaged across benchmarks.",
            "Latency is not an acceptance gate. This analyzer cannot submit the paper campaign.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    telemetry_dir = output_root / "telemetry"
    paths = sorted(telemetry_dir.glob("paper-v1-*.json"))
    if len(paths) != len(PAPER_CANDIDATE_TASKS):
        raise RuntimeError(
            f"expected {len(PAPER_CANDIDATE_TASKS)} worker records, found {len(paths)}"
        )
    workers = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    indexed = {
        str(worker.get("task")): (path, worker)
        for path, worker in zip(paths, workers, strict=True)
    }
    if set(indexed) != set(PAPER_CANDIDATE_TASKS):
        raise RuntimeError("preflight worker task set is incomplete")
    source_shas = {
        str(cast("dict[str, Any]", worker["source_snapshot"])["commit"])
        for worker in workers
    }
    if len(source_shas) != 1:
        raise RuntimeError("preflight workers used different source commits")

    task_reports: list[dict[str, Any]] = []
    worker_records: list[dict[str, Any]] = []
    for task in PAPER_CANDIDATE_TASKS:
        path, worker = indexed[task]
        if worker.get("status") != "completed" or worker.get("canonical_eligible") is not True:
            raise RuntimeError(f"{task} worker is not a completed canonical record")
        if worker.get("paper_experiment_authorized") is not False:
            raise RuntimeError(f"{task} worker unexpectedly authorizes the paper campaign")
        pairs = worker.get("paired_samples")
        if not isinstance(pairs, list) or not pairs:
            raise RuntimeError(f"{task} has no paired samples")
        baseline_scores: list[float] = []
        candidate_scores: list[float] = []
        reductions: list[float] = []
        identities: list[str] = []
        for raw_pair in pairs:
            if not isinstance(raw_pair, dict) or not isinstance(raw_pair.get("sample_id"), str):
                raise RuntimeError(f"{task} has an invalid paired sample")
            pair = cast("dict[str, Any]", raw_pair)
            sample_id = str(pair["sample_id"])
            full = _required_row(pair.get("full_kv"), method="full_kv", sample_id=sample_id)
            baseline = _required_row(
                pair.get("prompt_attention_topk"),
                method="prompt_attention_topk",
                sample_id=sample_id,
            )
            candidate = _required_row(
                pair.get("mosaickv_full"), method="mosaickv_full", sample_id=sample_id
            )
            if not (full["reference"] == baseline["reference"] == candidate["reference"]):
                raise RuntimeError(f"{task}/{sample_id} references differ across methods")
            baseline_bytes = int(baseline["active_kv_bytes"])
            candidate_bytes = int(candidate["active_kv_bytes"])
            reduction = 1 - candidate_bytes / baseline_bytes
            recorded_reduction = float(pair.get("active_reduction_fraction", math.nan))
            if not math.isclose(reduction, recorded_reduction, rel_tol=0, abs_tol=1e-12):
                raise RuntimeError(f"{task}/{sample_id} memory reduction was altered")
            if candidate_bytes > int(pair["resolved_candidate_active_cap_bytes"]):
                raise RuntimeError(f"{task}/{sample_id} exceeded its resolved active cap")
            identities.append(sample_id)
            baseline_scores.append(float(baseline["task_score"]))
            candidate_scores.append(float(candidate["task_score"]))
            reductions.append(reduction)
        if len(identities) != len(set(identities)):
            raise RuntimeError(f"{task} contains duplicate sample identities")
        baseline_score = fmean(baseline_scores)
        candidate_score = fmean(candidate_scores)
        task_reports.append(
            {
                "task": task,
                "sample_count": len(identities),
                "baseline_score": baseline_score,
                "candidate_score": candidate_score,
                "score_delta": candidate_score - baseline_score,
                "mean_active_reduction_fraction": fmean(reductions),
                "minimum_sample_active_reduction_fraction": min(reductions),
                "memory_passed": min(reductions) > 0.30,
                "quality_passed": candidate_score >= baseline_score,
            }
        )
        worker_records.append({"path": str(path), "sha256": _sha(path)})
    passed = all(
        bool(task["memory_passed"]) and bool(task["quality_passed"])
        for task in task_reports
    )
    report = {
        "schema_version": 1,
        "experiment_id": "paper_quality_memory_v1_preflight",
        "source_git_sha": next(iter(source_shas)),
        "workers": worker_records,
        "tasks": task_reports,
        "passed": passed,
        "memory_gate": "every_sample_active_kv_bytes_reduction_strictly_greater_than_0.30",
        "quality_gate": (
            "candidate_mean_official_score_greater_than_or_equal_to_prompt_topk_per_task"
        ),
        "paper_experiment_authorized": False,
        "next_action": (
            "eligible_for_separate_paper_experiment_authorization"
            if passed
            else "paper_experiment_blocked"
        ),
    }
    analysis_dir = output_root / "analysis"
    _write_immutable(analysis_dir / "gate.json", report)
    markdown_path = analysis_dir / "gate.md"
    markdown = _markdown(report)
    if markdown_path.exists() and markdown_path.read_text(encoding="utf-8") != markdown:
        raise RuntimeError("refusing to mutate immutable preflight Markdown report")
    if not markdown_path.exists():
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(markdown, encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
