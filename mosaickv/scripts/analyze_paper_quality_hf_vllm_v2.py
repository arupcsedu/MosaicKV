#!/usr/bin/env python3
"""Merge and gate the sharded HF/vLLM LLaVA paper-quality campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import yaml  # type: ignore[import-untyped]

TASKS = ("mmstar", "textvqa", "docvqa")
BACKENDS = ("hf_eager", "vllm")
METHODS = {
    "hf_eager": ("full_kv", "prompt_attention_topk", "mosaickv_full"),
    "vllm": ("full_kv", "prompt_attention_topk", "mosaickv_exact"),
}


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return cast("dict[str, Any]", value)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} is missing or nonnumeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{key} is nonfinite")
    return result


def _paths(row: Mapping[str, Any], method: str) -> tuple[Path, Path]:
    root = Path(str(row["output_root"])) / str(row["backend"]) / str(row["task"])
    root /= str(row["run_namespace"])
    namespace = str(row["run_namespace"])
    if row["backend"] == "hf_eager":
        run_ids = {
            "full_kv": f"{namespace}-full-kv",
            "prompt_attention_topk": f"{namespace}-prompt-attention-topk",
            "mosaickv_full": f"{namespace}-mosaickv-full",
        }
        method_root = root / method
    else:
        run_ids = {
            "full_kv": f"{namespace}-full-kv",
            "prompt_attention_topk": f"{namespace}-prompt-attention-topk",
            "mosaickv_exact": f"{namespace}-mosaickv-exact",
        }
        method_root = root / method / "vllm"
    run_root = method_root / run_ids[method]
    return run_root / "results.jsonl", run_root / "manifest.json"


def _bootstrap_interval(
    differences: Sequence[float], *, resamples: int, seed: int
) -> tuple[float, float]:
    values = np.asarray(differences, dtype=np.float64)
    if values.size < 2 or resamples < 1000:
        raise ValueError("paired bootstrap requires >=2 samples and >=1000 resamples")
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, 256):
        count = min(256, resamples - start)
        indices = rng.integers(0, values.size, size=(count, values.size))
        means[start : start + count] = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_task_summary(
    frame: pd.DataFrame,
    *,
    backend: str,
    task: str,
    minimum_reduction: float,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], list[str]]:
    """Return one backend/task paired summary and its explicit gate failures."""

    subset = frame[(frame.backend_scope == backend) & (frame.task == task)]
    candidate = "mosaickv_full" if backend == "hf_eager" else "mosaickv_exact"
    by_method = {
        method: subset[subset.protocol_method == method].set_index("sample_id")
        for method in METHODS[backend]
    }
    identities = tuple(sorted(by_method["full_kv"].index))
    if not identities or any(
        tuple(sorted(value.index)) != identities for value in by_method.values()
    ):
        raise ValueError(f"{backend}/{task} methods do not contain identical paired samples")
    prompt = by_method["prompt_attention_topk"].loc[list(identities)]
    mosaic = by_method[candidate].loc[list(identities)]
    full = by_method["full_kv"].loc[list(identities)]
    prompt_bytes = prompt.active_kv_bytes.to_numpy(dtype=np.float64)
    mosaic_bytes = mosaic.active_kv_bytes.to_numpy(dtype=np.float64)
    reductions = 1.0 - mosaic_bytes / prompt_bytes
    score_differences = mosaic.task_score.to_numpy(dtype=np.float64) - prompt.task_score.to_numpy(
        dtype=np.float64
    )
    failures = [
        f"{backend}/{task}/{sample_id}: active-memory reduction "
        f"{reduction:.6f} < {minimum_reduction:.6f}"
        for sample_id, reduction in zip(identities, reductions, strict=True)
        if reduction + 1e-12 < minimum_reduction
    ]
    mean_delta = float(score_differences.mean())
    if mean_delta < 0:
        failures.append(
            f"{backend}/{task}: paired official-score delta {mean_delta:.6f} is negative"
        )
    interval = _bootstrap_interval(
        score_differences,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    summary = {
        "backend": backend,
        "task": task,
        "sample_count": len(identities),
        "fullkv_mean_official_score": float(full.task_score.mean()),
        "prompt_topk_mean_official_score": float(prompt.task_score.mean()),
        "mosaickv_mean_official_score": float(mosaic.task_score.mean()),
        "paired_mean_score_delta": mean_delta,
        "paired_bootstrap_score_delta_95_ci_low": interval[0],
        "paired_bootstrap_score_delta_95_ci_high": interval[1],
        "mean_active_memory_reduction_fraction": float(reductions.mean()),
        "minimum_active_memory_reduction_fraction": float(reductions.min()),
        "mean_fullkv_active_kv_bytes": float(full.active_kv_bytes.mean()),
        "mean_prompt_topk_active_kv_bytes": float(prompt.active_kv_bytes.mean()),
        "mean_mosaickv_active_kv_bytes": float(mosaic.active_kv_bytes.mean()),
        "mean_mosaickv_residual_kv_bytes": float(mosaic.residual_kv_bytes.mean()),
        "answer_agreement_with_prompt_topk": float(
            (mosaic.answer.to_numpy() == prompt.answer.to_numpy()).mean()
        ),
        "answer_agreement_with_fullkv": float(
            (mosaic.answer.to_numpy() == full.answer.to_numpy()).mean()
        ),
        "mean_fullkv_end_to_end_seconds": float(full.end_to_end_time.mean()),
        "mean_prompt_topk_end_to_end_seconds": float(prompt.end_to_end_time.mean()),
        "mean_mosaickv_end_to_end_seconds": float(mosaic.end_to_end_time.mean()),
        "memory_gate_passed": bool(np.all(reductions + 1e-12 >= minimum_reduction)),
        "quality_gate_passed": mean_delta >= 0,
        "latency_is_gate": False,
    }
    return summary, failures


def analyze(
    index_path: Path, protocol_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    protocol = _object(yaml.safe_load(protocol_path.read_text(encoding="utf-8")), "protocol")
    experiment = _object(protocol.get("experiment"), "protocol.experiment")
    gates = _object(protocol.get("primary_gates"), "protocol.primary_gates")
    if protocol.get("execution_authorized") is not True:
        raise PermissionError("paper campaign analysis requires an authorized protocol")
    rows = [
        _object(json.loads(line), f"{index_path}:{number}")
        for number, line in enumerate(index_path.read_text(encoding="utf-8").splitlines(), 1)
        if line
    ]
    if len(rows) != 24 or tuple(sorted(set(str(row["backend"]) for row in rows))) != BACKENDS:
        raise ValueError("paper campaign index must contain 24 HF/vLLM jobs")
    expected_samples = int(experiment["sample_count_per_task"])
    records: list[dict[str, Any]] = []
    source_shas: set[str] = set()
    isolation_failures: list[str] = []
    for job in rows:
        completion = (
            Path(str(job["output_root"]))
            / str(job["backend"])
            / str(job["task"])
            / str(job["run_namespace"])
            / "job_complete.json"
        )
        complete = _object(json.loads(completion.read_text(encoding="utf-8")), str(completion))
        if (
            complete.get("status") != "complete"
            or complete.get("source_git_sha") != job["source_git_sha"]
        ):
            raise ValueError(f"invalid job completion seal: {completion}")
        ledger = _object(
            json.loads(Path(str(job["sample_ledger_path"])).read_text(encoding="utf-8")),
            "sample shard",
        )
        expected_ids = tuple(str(value) for value in cast("list[str]", ledger["sample_ids"]))
        for method in METHODS[str(job["backend"])]:
            result_path, manifest_path = _paths(job, method)
            manifest = _object(
                json.loads(manifest_path.read_text(encoding="utf-8")), str(manifest_path)
            )
            source = _object(manifest.get("source"), "manifest.source")
            if source.get("git_dirty") is not False or source.get("canonical_eligible") is not True:
                raise ValueError(f"noncanonical manifest: {manifest_path}")
            if source.get("git_sha") != job["source_git_sha"]:
                raise ValueError(f"manifest source differs from its job: {manifest_path}")
            source_shas.add(str(source["git_sha"]))
            dataset = _object(manifest.get("dataset"), "manifest.dataset")
            model = _object(manifest.get("model"), "manifest.model")
            if dataset.get("revision") != job["dataset_revision"]:
                raise ValueError(f"dataset revision changed: {manifest_path}")
            if (
                model.get("id") != protocol["candidate"]["model_id"]
                or model.get("revision") != protocol["candidate"]["model_revision"]
            ):
                raise ValueError(f"model identity changed: {manifest_path}")
            if job["backend"] == "vllm":
                execution = _object(manifest.get("execution"), "manifest.execution")
                if execution.get("request_isolation_verified") is not True:
                    isolation_failures.append(f"{job['task']}/{job['shard_index']}/{method}")
            observed: list[dict[str, Any]] = []
            for line_number, line in enumerate(
                result_path.read_text(encoding="utf-8").splitlines(), 1
            ):
                if not line:
                    continue
                value = _object(json.loads(line), f"{result_path}:{line_number}")
                observed.append(value)
            if len(observed) != len(expected_ids) or tuple(
                sorted(str(value["sample_id"]) for value in observed)
            ) != tuple(sorted(expected_ids)):
                raise ValueError(f"incomplete or wrong sample rows: {result_path}")
            for value in observed:
                if value.get("status") != "completed" or value.get("error") is not None:
                    raise ValueError(f"failed sample in canonical results: {result_path}")
                for key in (
                    "task_score",
                    "active_kv_bytes",
                    "residual_kv_bytes",
                    "compression_time",
                    "ttft",
                    "prefill_time",
                    "decode_time",
                    "end_to_end_time",
                    "generated_tokens",
                    "peak_gpu_memory",
                ):
                    _finite(value, key)
                effective_method = str(value.get("method", ""))
                if not effective_method.startswith(method):
                    raise ValueError(
                        f"effective method label differs from protocol method: {effective_method}"
                    )
                records.append(
                    {
                        **value,
                        "backend_scope": job["backend"],
                        "protocol_method": method,
                        "effective_method": effective_method,
                        "shard_index": job["shard_index"],
                        "source_git_sha": job["source_git_sha"],
                        "protocol_sha256": job["protocol_sha256"],
                        "sample_ledger_sha256": job["sample_ledger_sha256"],
                        "result_path": str(result_path),
                    }
                )
    frame = pd.DataFrame.from_records(records)
    if len(source_shas) != 1:
        raise ValueError("campaign results do not share one source commit")
    for task in TASKS:
        identities = {
            backend: set(
                str(value)
                for value in frame[
                    (frame.backend_scope == backend)
                    & (frame.task == task)
                    & (frame.protocol_method == "full_kv")
                ].sample_id
            )
            for backend in BACKENDS
        }
        if (
            identities["hf_eager"] != identities["vllm"]
            or len(identities["hf_eager"]) != expected_samples
        ):
            raise ValueError(f"HF/vLLM sample identity mismatch for {task}")
    failures = list(isolation_failures)
    summaries: list[dict[str, Any]] = []
    minimum_reduction = float(gates["minimum_per_sample_active_memory_reduction_fraction"])
    for backend_index, backend in enumerate(BACKENDS):
        for task_index, task in enumerate(TASKS):
            summary, task_failures = paired_task_summary(
                frame,
                backend=backend,
                task=task,
                minimum_reduction=minimum_reduction,
                bootstrap_resamples=int(experiment["bootstrap_resamples"]),
                bootstrap_seed=int(experiment["bootstrap_seed"]) + 10 * backend_index + task_index,
            )
            summaries.append(summary)
            failures.extend(task_failures)
    summary_frame = pd.DataFrame.from_records(summaries)
    report = {
        "schema_version": 1,
        "experiment_id": protocol["experiment_id"],
        "classification": protocol["classification"],
        "source_git_sha": next(iter(source_shas)),
        "protocol_path": str(protocol_path),
        "protocol_sha256": _sha(protocol_path),
        "sample_count_per_task": expected_samples,
        "primary_comparisons_are_backend_separated": True,
        "latency_is_secondary_nonblocking": True,
        "hf_candidate_scope": sorted(
            set(
                frame[
                    (frame.backend_scope == "hf_eager") & (frame.protocol_method == "mosaickv_full")
                ].effective_method
            )
        ),
        "vllm_candidate_scope": "mosaickv_exact__vllm_requestwide_projection",
        "passed": not failures,
        "failures": failures,
        "task_results": summaries,
    }
    return frame, summary_frame, report


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# HF/vLLM LLaVA-1.5 paper quality-memory analysis",
        "",
        f"Overall gate: **{'PASS' if report['passed'] else 'FAIL'}**",
        "",
        "Comparisons are paired within each backend; HF and vLLM memory figures are not mixed.",
        "",
        "| Backend | Task | N | FullKV | Prompt Top-K | MosaicKV | Score delta | "
        "95% CI | Mean smaller | Minimum smaller | Answer agreement |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in cast("list[dict[str, Any]]", report["task_results"]):
        lines.append(
            f"| {item['backend']} | {item['task']} | {item['sample_count']} | "
            f"{item['fullkv_mean_official_score']:.4f} | "
            f"{item['prompt_topk_mean_official_score']:.4f} | "
            f"{item['mosaickv_mean_official_score']:.4f} | "
            f"{item['paired_mean_score_delta']:+.4f} | "
            f"[{item['paired_bootstrap_score_delta_95_ci_low']:+.4f}, "
            f"{item['paired_bootstrap_score_delta_95_ci_high']:+.4f}] | "
            f"{item['mean_active_memory_reduction_fraction']:.2%} | "
            f"{item['minimum_active_memory_reduction_fraction']:.2%} | "
            f"{item['answer_agreement_with_prompt_topk']:.2%} |"
        )
    lines.extend(
        [
            "",
            "Official task scores and active KV bytes are primary. Latency is "
            "reported but is not a rejection gate.",
            "Incompatible task metrics are not macro-averaged.",
        ]
    )
    failures = cast("list[str]", report["failures"])
    if failures:
        lines.extend(["", "## Gate failures", "", *[f"- {value}" for value in failures]])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    frame, summary, report = analyze(args.index.resolve(), args.protocol.resolve())
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output / "paired_rows.parquet", index=False)
    summary.to_parquet(output / "backend_task_summary.parquet", index=False)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "report.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "output": str(output)}, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
