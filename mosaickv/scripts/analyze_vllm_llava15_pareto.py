#!/usr/bin/env python3
"""Strict paired quality/memory gate for the LLaVA vLLM development run."""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
from pathlib import Path
from typing import Any, cast

TASKS = ("mmstar", "textvqa", "docvqa")
METHODS = ("full_kv", "prompt_attention_topk", "mosaickv_exact")


def _rows(root: Path, task: str, method: str, backend: str) -> dict[str, dict[str, Any]]:
    paths = sorted(
        Path(path)
        for path in glob.glob(
            str(root / task / method / backend / "**" / "results.jsonl"),
            recursive=True,
        )
    )
    rows: dict[str, dict[str, Any]] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object result row in {path}")
            row = cast("dict[str, Any]", value)
            sample_id = str(row.get("sample_id", ""))
            if not sample_id or sample_id in rows:
                raise ValueError(
                    f"missing or duplicate sample ID {sample_id!r} for {task}/{method}"
                )
            rows[sample_id] = row
    if len(rows) != 2:
        raise ValueError(f"expected two rows for {task}/{method}/{backend}, found {len(rows)}")
    return rows


def _finite_completed(row: dict[str, Any]) -> bool:
    fields = (
        "active_kv_bytes",
        "task_score",
        "compression_time",
        "ttft",
        "decode_time",
        "end_to_end_time",
    )
    return (
        row.get("status") == "completed"
        and row.get("error") is None
        and all(
            isinstance(row.get(field), (int, float))
            and not isinstance(row.get(field), bool)
            and math.isfinite(float(row[field]))
            for field in fields
        )
    )


def analyze(root: Path, minimum_reduction: float) -> dict[str, Any]:
    task_records: list[dict[str, Any]] = []
    failures: list[str] = []
    for task in TASKS:
        vllm = {method: _rows(root, task, method, "vllm") for method in METHODS}
        expected_ids = set(vllm["full_kv"])
        if any(set(rows) != expected_ids for rows in vllm.values()):
            failures.append(f"{task}: methods used different samples")
            continue
        hf = {
            method: _rows(root, task, method, "hf")
            for method in ("prompt_attention_topk", "mosaickv_exact")
        }
        per_sample: list[dict[str, Any]] = []
        for sample_id in sorted(expected_ids):
            full = vllm["full_kv"][sample_id]
            prompt = vllm["prompt_attention_topk"][sample_id]
            mosaic = vllm["mosaickv_exact"][sample_id]
            for method, row in (
                ("full_kv", full),
                ("prompt_attention_topk", prompt),
                ("mosaickv_exact", mosaic),
            ):
                if not _finite_completed(row):
                    failures.append(f"{task}/{sample_id}/{method}: incomplete or non-finite row")
            prompt_bytes = int(prompt["active_kv_bytes"])
            mosaic_bytes = int(mosaic["active_kv_bytes"])
            reduction = 1.0 - mosaic_bytes / prompt_bytes
            answer_parity = mosaic.get("answer") == prompt.get("answer")
            score_parity = math.isclose(
                float(mosaic["task_score"]),
                float(prompt["task_score"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            hf_output_agreement = all(
                vllm[method][sample_id].get("answer") == hf[method][sample_id].get("answer")
                and math.isclose(
                    float(vllm[method][sample_id]["task_score"]),
                    float(hf[method][sample_id]["task_score"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                for method in ("prompt_attention_topk", "mosaickv_exact")
            )
            if reduction + 1e-12 < minimum_reduction:
                failures.append(
                    f"{task}/{sample_id}: active-memory reduction {reduction:.6f} "
                    f"is below {minimum_reduction:.6f}"
                )
            if not answer_parity or not score_parity:
                failures.append(f"{task}/{sample_id}: MosaicKV differs from Prompt Top-K")
            per_sample.append(
                {
                    "sample_id": sample_id,
                    "prompt_active_kv_bytes": prompt_bytes,
                    "mosaickv_active_kv_bytes": mosaic_bytes,
                    "active_memory_reduction_fraction": reduction,
                    "answer_parity": answer_parity,
                    "score_parity": score_parity,
                    "hf_vllm_output_agreement": hf_output_agreement,
                    "hf_vllm_output_agreement_is_gate": False,
                    "hf_vllm_comparability": (
                        "informational_only_different_cache_granularity_and_active_budget"
                    ),
                    "fullkv_answer_agreement": mosaic.get("answer") == full.get("answer"),
                }
            )
        task_records.append(
            {
                "task": task,
                "sample_count": len(per_sample),
                "prompt_score": statistics.mean(
                    float(row["task_score"])
                    for row in vllm["prompt_attention_topk"].values()
                ),
                "mosaickv_score": statistics.mean(
                    float(row["task_score"]) for row in vllm["mosaickv_exact"].values()
                ),
                "mean_active_memory_reduction_fraction": statistics.mean(
                    row["active_memory_reduction_fraction"] for row in per_sample
                ),
                "minimum_active_memory_reduction_fraction": min(
                    row["active_memory_reduction_fraction"] for row in per_sample
                ),
                "mean_prompt_active_kv_mib": statistics.mean(
                    float(row["active_kv_bytes"])
                    for row in vllm["prompt_attention_topk"].values()
                )
                / 2**20,
                "mean_mosaickv_active_kv_mib": statistics.mean(
                    float(row["active_kv_bytes"])
                    for row in vllm["mosaickv_exact"].values()
                )
                / 2**20,
                "mean_prompt_end_to_end_seconds": statistics.mean(
                    float(row["end_to_end_time"])
                    for row in vllm["prompt_attention_topk"].values()
                ),
                "mean_mosaickv_end_to_end_seconds": statistics.mean(
                    float(row["end_to_end_time"])
                    for row in vllm["mosaickv_exact"].values()
                ),
                "samples": per_sample,
            }
        )
    isolation_manifests = sorted(
        root.glob("textvqa/mosaickv_isolation/vllm/**/manifest.json")
    )
    isolation_verified = False
    if len(isolation_manifests) != 1:
        failures.append(
            "textvqa: expected one two-request MosaicKV isolation manifest, "
            f"found {len(isolation_manifests)}"
        )
    else:
        manifest = json.loads(isolation_manifests[0].read_text(encoding="utf-8"))
        execution = manifest.get("execution") if isinstance(manifest, dict) else None
        isolation_verified = (
            isinstance(execution, dict)
            and execution.get("request_isolation_verified") is True
        )
        if not isolation_verified:
            failures.append("textvqa: native MosaicKV A-B-A request isolation was not verified")
    return {
        "schema_version": 1,
        "classification": "retrospective_nonheldout_vllm_development_preflight",
        "model": "llava-hf/llava-1.5-7b-hf",
        "minimum_active_memory_reduction_fraction": minimum_reduction,
        "quality_gate": "exact paired Prompt Top-K answer and task-score parity",
        "backend_generalization_gate": (
            "within_backend_only; cross-backend outputs are informational unless cache "
            "granularity, selected positions, and active bytes match"
        ),
        "native_request_isolation_verified": isolation_verified,
        "passed": not failures,
        "failures": failures,
        "tasks": task_records,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# LLaVA-1.5 vLLM quality-memory development preflight",
        "",
        f"Gate: **{'PASS' if report['passed'] else 'FAIL'}**",
        "",
        "| Task | N | Prompt score | MosaicKV score | Prompt MiB | MosaicKV MiB | "
        "Mean smaller | Minimum smaller |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task in report["tasks"]:
        lines.append(
            f"| {task['task']} | {task['sample_count']} | {task['prompt_score']:.4f} | "
            f"{task['mosaickv_score']:.4f} | {task['mean_prompt_active_kv_mib']:.2f} | "
            f"{task['mean_mosaickv_active_kv_mib']:.2f} | "
            f"{100 * task['mean_active_memory_reduction_fraction']:.2f}% | "
            f"{100 * task['minimum_active_memory_reduction_fraction']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "This is a retrospective, non-held-out development result and is not paper evidence.",
        ]
    )
    if report["failures"]:
        lines.extend(["", "Failures:"])
        lines.extend(f"- {failure}" for failure in report["failures"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--minimum-reduction", type=float, default=0.30)
    args = parser.parse_args()
    report = analyze(args.root.resolve(), args.minimum_reduction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "output": str(args.output.resolve())}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
