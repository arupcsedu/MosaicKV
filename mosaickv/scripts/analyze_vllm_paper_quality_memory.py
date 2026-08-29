#!/usr/bin/env python3
"""Fail-closed paper gate for paired LLaVA vLLM quality-memory rows.

Official task scores are the primary quality endpoint. Exact answer agreement
is retained as a secondary diagnostic because semantically equivalent strings,
and task-specific partial-credit metrics, make raw string equality unsuitable
as the primary cross-task gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

TASKS = ("mmstar", "textvqa", "docvqa")
METHODS = ("full_kv", "prompt_attention_topk", "mosaickv_exact")
MODEL_ID = "llava-hf/llava-1.5-7b-hf"


@dataclass(frozen=True, slots=True)
class Observation:
    row: dict[str, Any]
    result_path: Path
    manifest: dict[str, Any]
    trace: dict[str, Any]


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return cast("dict[str, Any]", value)


def _finite(row: Mapping[str, Any], field: str) -> float:
    value = row.get(field)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"result field {field} is missing or non-finite")
    return float(value)


def _model_trace(method_root: Path, run_id: str, sample_id: str) -> dict[str, Any]:
    paths = sorted(method_root.glob(f"**/traces/{run_id}/*.json"))
    matches: list[dict[str, Any]] = []
    for path in paths:
        payload = _object(json.loads(path.read_text(encoding="utf-8")), str(path))
        if payload.get("run_id") == run_id and payload.get("sample_id") == sample_id:
            matches.append(payload)
    if len(matches) != 1:
        raise ValueError(
            f"expected one model trace for {run_id}/{sample_id}, found {len(matches)}"
        )
    return matches[0]


def _observations(
    root: Path,
    task: str,
    method: str,
    *,
    expected_samples: int,
    require_clean_source: bool,
) -> dict[str, Observation]:
    method_root = root / task / method / "vllm"
    result_paths = sorted(method_root.glob("**/results.jsonl"))
    if not result_paths:
        raise ValueError(f"no vLLM result files found for {task}/{method}")
    observations: dict[str, Observation] = {}
    for result_path in result_paths:
        encoded = result_path.read_bytes()
        for line in encoded.decode("utf-8").splitlines():
            row = _object(json.loads(line), str(result_path))
            sample_id = str(row.get("sample_id", ""))
            if not sample_id or sample_id in observations:
                raise ValueError(
                    f"missing or duplicate sample ID {sample_id!r} for {task}/{method}"
                )
            if row.get("status") != "completed" or row.get("error") is not None:
                raise ValueError(f"incomplete result for {task}/{method}/{sample_id}")
            if row.get("task") != task or row.get("backend") != "vllm":
                raise ValueError(f"task/backend mismatch for {task}/{method}/{sample_id}")
            if row.get("model") != MODEL_ID:
                raise ValueError(f"model mismatch for {task}/{method}/{sample_id}")
            effective_method = str(row.get("method", ""))
            if not effective_method.startswith(method):
                raise ValueError(
                    f"effective method {effective_method!r} is not {method!r} for {sample_id}"
                )
            for field in (
                "task_score",
                "active_kv_bytes",
                "residual_kv_bytes",
                "compression_time",
                "ttft",
                "decode_time",
                "end_to_end_time",
            ):
                _finite(row, field)
            if int(row["active_kv_bytes"]) < 1 or int(row["residual_kv_bytes"]) < 0:
                raise ValueError(f"invalid cache accounting for {task}/{method}/{sample_id}")
            manifest_path = Path(str(row.get("manifest_path", ""))).resolve()
            if not manifest_path.is_file():
                raise ValueError(f"missing manifest for {task}/{method}/{sample_id}")
            manifest = _object(
                json.loads(manifest_path.read_text(encoding="utf-8")), str(manifest_path)
            )
            source = _object(manifest.get("source"), f"{manifest_path}.source")
            if require_clean_source and (
                source.get("git_dirty") is not False
                or source.get("canonical_eligible") is not True
            ):
                raise ValueError(f"non-canonical source for {task}/{method}/{sample_id}")
            artifacts = _object(manifest.get("artifacts"), f"{manifest_path}.artifacts")
            if artifacts.get("raw_output_sha") != hashlib.sha256(encoded).hexdigest():
                raise ValueError(f"raw-result hash mismatch for {task}/{method}/{sample_id}")
            run_id = str(row.get("run_id", ""))
            trace = _model_trace(method_root, run_id, sample_id)
            if method == "full_kv":
                if trace.get("native_block_compaction") is not False:
                    raise ValueError(f"FullKV unexpectedly used native compaction for {sample_id}")
            else:
                if (
                    trace.get("native_block_compaction") is not True
                    or trace.get("native_scope")
                    != "hash_bound_common_selector_request_wide_projection"
                    or not isinstance(trace.get("selector_plan"), dict)
                ):
                    raise ValueError(f"{method}/{sample_id} did not use the common selector")
            if method == "mosaickv_exact" and (
                int(row.get("residual_kv_bytes", -1)) != 0
                or int(row.get("repair_count", -1)) != 0
            ):
                raise ValueError(
                    f"exact-only MosaicKV reported residual/repair state for {sample_id}"
                )
            observations[sample_id] = Observation(row, result_path, manifest, trace)
    if len(observations) != expected_samples:
        raise ValueError(
            f"expected {expected_samples} rows for {task}/{method}, found {len(observations)}"
        )
    return observations


def _controlled_signature(observation: Observation) -> dict[str, Any]:
    manifest = observation.manifest
    execution = _object(manifest.get("execution"), "manifest.execution")
    environment = _object(manifest.get("environment"), "manifest.environment")
    generation = _object(manifest.get("generation"), "manifest.generation")
    return {
        "model": manifest.get("model"),
        "dataset": manifest.get("dataset"),
        "generation_parameters_sha": generation.get("parameters_sha"),
        "hardware": manifest.get("hardware"),
        "software": manifest.get("software"),
        "environment_lock_sha256": environment.get("lock_sha256"),
        "backend": execution.get("backend"),
        "attention_implementation": execution.get("attention_implementation"),
        "cuda_graph": execution.get("cuda_graph"),
        "seed": execution.get("seed"),
    }


def _bootstrap_mean_difference(
    differences: Sequence[float], *, resamples: int, seed: int
) -> tuple[float, float]:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("paired bootstrap requires at least two paired samples")
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    batch = 256
    for start in range(0, resamples, batch):
        count = min(batch, resamples - start)
        indices = rng.integers(0, values.size, size=(count, values.size))
        means[start : start + count] = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _isolation_verified(root: Path, *, require_clean_source: bool) -> bool:
    manifests = sorted(root.glob("**/mosaickv_isolation/vllm/**/manifest.json"))
    for path in manifests:
        manifest = _object(json.loads(path.read_text(encoding="utf-8")), str(path))
        source = _object(manifest.get("source"), f"{path}.source")
        execution = _object(manifest.get("execution"), f"{path}.execution")
        if require_clean_source and (
            source.get("git_dirty") is not False
            or source.get("canonical_eligible") is not True
        ):
            continue
        if execution.get("request_isolation_verified") is True:
            return True
    return False


def analyze(
    root: Path,
    *,
    expected_samples: int,
    minimum_reduction: float,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    require_clean_source: bool,
) -> dict[str, Any]:
    if expected_samples < 2:
        raise ValueError("expected_samples must be at least two")
    if bootstrap_resamples < 1000:
        raise ValueError("paper analysis requires at least 1000 bootstrap resamples")
    if not math.isfinite(minimum_reduction) or not 0 < minimum_reduction < 1:
        raise ValueError("minimum_reduction must be finite and in (0, 1)")
    failures: list[str] = []
    task_reports: list[dict[str, Any]] = []
    source_shas: set[str] = set()
    for task_index, task in enumerate(TASKS):
        by_method = {
            method: _observations(
                root,
                task,
                method,
                expected_samples=expected_samples,
                require_clean_source=require_clean_source,
            )
            for method in METHODS
        }
        identities = tuple(sorted(by_method["full_kv"]))
        if any(tuple(sorted(rows)) != identities for rows in by_method.values()):
            raise ValueError(f"{task} methods do not cover identical samples")
        score_differences: list[float] = []
        reductions: list[float] = []
        answer_agreements: list[bool] = []
        fullkv_agreements: list[bool] = []
        sample_reports: list[dict[str, Any]] = []
        for sample_id in identities:
            observations = tuple(by_method[method][sample_id] for method in METHODS)
            signatures = tuple(_controlled_signature(value) for value in observations)
            if any(signature != signatures[0] for signature in signatures[1:]):
                failures.append(f"{task}/{sample_id}: controlled runtime provenance differs")
            source_shas.update(
                str(_object(value.manifest.get("source"), "manifest.source")["git_sha"])
                for value in observations
            )
            prompts = {str(value.trace.get("prompt_sha256", "")) for value in observations}
            media = {str(value.trace.get("media_sha256", "")) for value in observations}
            sampling = {
                json.dumps(value.trace.get("effective_sampling_parameters"), sort_keys=True)
                for value in observations
            }
            if len(prompts) != 1 or "" in prompts or len(media) != 1 or "" in media:
                failures.append(
                    f"{task}/{sample_id}: prompt/media provenance differs or is missing"
                )
            if len(sampling) != 1:
                failures.append(f"{task}/{sample_id}: deterministic generation settings differ")
            full = by_method["full_kv"][sample_id].row
            prompt = by_method["prompt_attention_topk"][sample_id].row
            mosaic = by_method["mosaickv_exact"][sample_id].row
            prompt_bytes = int(prompt["active_kv_bytes"])
            mosaic_bytes = int(mosaic["active_kv_bytes"])
            reduction = 1.0 - mosaic_bytes / prompt_bytes
            score_difference = float(mosaic["task_score"]) - float(prompt["task_score"])
            answer_agreement = mosaic.get("answer") == prompt.get("answer")
            fullkv_agreement = mosaic.get("answer") == full.get("answer")
            if reduction + 1e-12 < minimum_reduction:
                failures.append(
                    f"{task}/{sample_id}: active-memory reduction {reduction:.6f} "
                    f"is below {minimum_reduction:.6f}"
                )
            reductions.append(reduction)
            score_differences.append(score_difference)
            answer_agreements.append(answer_agreement)
            fullkv_agreements.append(fullkv_agreement)
            sample_reports.append(
                {
                    "sample_id": sample_id,
                    "active_memory_reduction_fraction": reduction,
                    "score_difference_vs_prompt_topk": score_difference,
                    "answer_agreement_with_prompt_topk": answer_agreement,
                    "answer_agreement_with_fullkv": fullkv_agreement,
                }
            )
        mean_score_difference = statistics.fmean(score_differences)
        if mean_score_difference < 0:
            failures.append(
                f"{task}: mean official-score difference {mean_score_difference:.6f} is negative"
            )
        interval = _bootstrap_mean_difference(
            score_differences,
            resamples=bootstrap_resamples,
            seed=bootstrap_seed + task_index,
        )
        prompt_rows = [by_method["prompt_attention_topk"][value].row for value in identities]
        mosaic_rows = [by_method["mosaickv_exact"][value].row for value in identities]
        task_reports.append(
            {
                "task": task,
                "sample_count": len(identities),
                "prompt_topk_mean_official_score": statistics.fmean(
                    float(value["task_score"]) for value in prompt_rows
                ),
                "mosaickv_mean_official_score": statistics.fmean(
                    float(value["task_score"]) for value in mosaic_rows
                ),
                "paired_mean_score_difference": mean_score_difference,
                "paired_bootstrap_score_difference_95_ci": interval,
                "mean_active_memory_reduction_fraction": statistics.fmean(reductions),
                "minimum_active_memory_reduction_fraction": min(reductions),
                "answer_agreement_with_prompt_topk": statistics.fmean(answer_agreements),
                "answer_agreement_with_fullkv": statistics.fmean(fullkv_agreements),
                "mean_prompt_topk_end_to_end_seconds": statistics.fmean(
                    float(value["end_to_end_time"]) for value in prompt_rows
                ),
                "mean_mosaickv_end_to_end_seconds": statistics.fmean(
                    float(value["end_to_end_time"]) for value in mosaic_rows
                ),
                "latency_is_gate": False,
                "samples": sample_reports,
            }
        )
    if len(source_shas) != 1:
        failures.append(f"rows used {len(source_shas)} source commits instead of one")
    isolation = any(
        _object(value.manifest.get("execution"), "manifest.execution").get(
            "request_isolation_verified"
        )
        is True
        for value in by_method["mosaickv_exact"].values()
    ) or _isolation_verified(root, require_clean_source=require_clean_source)
    if not isolation:
        failures.append("no persistent-engine A-B-A MosaicKV isolation evidence was recorded")
    return {
        "schema_version": 1,
        "classification": (
            "paired_llava_vllm_quality_memory_gate"
            if require_clean_source
            else "dirty_source_retrospective_development_gate"
        ),
        "canonical_eligible": require_clean_source,
        "model": MODEL_ID,
        "tasks": list(TASKS),
        "candidate_effective_scope": "mosaickv_exact__vllm_requestwide_projection",
        "minimum_active_memory_reduction_fraction": minimum_reduction,
        "quality_gate": "nonnegative paired mean official task-score difference per task",
        "quality_confidence_interval_policy": "reported_secondary_not_used_to_weaken_zero_margin",
        "exact_answer_agreement_policy": "secondary_diagnostic_not_primary_gate",
        "latency_policy": "reported_secondary_not_rejection_criterion",
        "bootstrap_resamples": bootstrap_resamples,
        "source_git_sha": next(iter(source_shas)) if len(source_shas) == 1 else None,
        "request_isolation_verified": isolation,
        "passed": not failures,
        "failures": failures,
        "task_results": task_reports,
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# LLaVA vLLM paired quality-memory gate",
        "",
        f"Gate: **{'PASS' if report['passed'] else 'FAIL'}**",
        "",
        "| Task | N | Prompt score | MosaicKV score | Delta | 95% CI | "
        "Mean smaller | Minimum smaller | Exact agreement |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task in cast("list[dict[str, Any]]", report["task_results"]):
        interval = task["paired_bootstrap_score_difference_95_ci"]
        lines.append(
            f"| {task['task']} | {task['sample_count']} | "
            f"{task['prompt_topk_mean_official_score']:.4f} | "
            f"{task['mosaickv_mean_official_score']:.4f} | "
            f"{task['paired_mean_score_difference']:+.4f} | "
            f"[{interval[0]:+.4f}, {interval[1]:+.4f}] | "
            f"{task['mean_active_memory_reduction_fraction']:.2%} | "
            f"{task['minimum_active_memory_reduction_fraction']:.2%} | "
            f"{task['answer_agreement_with_prompt_topk']:.2%} |"
        )
    lines.extend(
        [
            "",
            "Official per-task scores are primary; incompatible task metrics are not averaged.",
            "Exact-answer agreement and latency are reported secondary diagnostics.",
        ]
    )
    failures = cast("list[str]", report["failures"])
    if failures:
        lines.extend(["", "Failures:", *[f"- {value}" for value in failures]])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int, required=True)
    parser.add_argument("--minimum-reduction", type=float, default=0.30)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20270831)
    parser.add_argument("--allow-dirty-development", action="store_true")
    args = parser.parse_args()
    report = analyze(
        args.root.resolve(),
        expected_samples=args.expected_samples,
        minimum_reduction=args.minimum_reduction,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
        require_clean_source=not args.allow_dirty_development,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "output": str(args.output.resolve())}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
