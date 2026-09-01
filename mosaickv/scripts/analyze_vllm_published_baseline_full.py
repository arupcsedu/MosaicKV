#!/usr/bin/env python3
"""Validate and aggregate the 9,000-sample published-baseline extension."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

TASKS = {"mmstar": 1500, "textvqa": 3000, "docvqa": 4500}
BASELINES = ("lookm_reimpl", "prefixkv_reimpl", "vl_cache_reimpl")
REUSED_METHODS = {
    "hf_eager": ("full_kv", "prompt_attention_topk", "mosaickv_full"),
    "vllm": ("full_kv", "prompt_attention_topk", "mosaickv_exact"),
}
MOSAIC_METHOD = {"hf_eager": "mosaickv_full", "vllm": "mosaickv_exact"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return cast(
        "list[dict[str, Any]]",
        [json.loads(line) for line in path.read_text().splitlines() if line.strip()],
    )


def _bootstrap_interval(values: np.ndarray, *, seed: int) -> tuple[float, float]:
    if not len(values):
        raise ValueError("cannot bootstrap an empty vector")
    rng = np.random.default_rng(seed)
    sample_count = len(values)
    means = np.empty(2000, dtype=np.float64)
    for index in range(len(means)):
        means[index] = float(values[rng.integers(0, sample_count, sample_count)].mean())
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def _validate_manifest(path: Path, *, source_git_sha: str, backend: str) -> None:
    manifest = json.loads(path.read_text())
    source = cast("dict[str, Any]", manifest["source"])
    if source.get("git_sha") != source_git_sha:
        raise RuntimeError(f"manifest source SHA changed: {path}")
    if source.get("git_dirty") is not False or source.get("canonical_eligible") is not True:
        raise RuntimeError(f"manifest is not canonical clean-source evidence: {path}")
    execution = cast("dict[str, Any]", manifest["execution"])
    expected_backend = "huggingface" if backend == "hf_eager" else "vllm"
    if execution.get("backend") != expected_backend:
        raise RuntimeError(f"manifest backend changed: {path}")
    if manifest.get("measurement_type") != "baseline_reimpl_measured":
        raise RuntimeError(f"manifest measurement type changed: {path}")


def _campaign_rows(root: Path) -> pd.DataFrame:
    index = json.loads((root / "inputs/job_index.json").read_text())
    if index.get("job_count") != 142 or len(index.get("jobs", [])) != 142:
        raise RuntimeError("campaign must contain exactly 142 immutable shard jobs")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for job in cast("list[dict[str, Any]]", index["jobs"]):
        task = str(job["task"])
        shard = int(job["shard_index"])
        source_sha = str(job["source_git_sha"])
        ledger = json.loads(Path(str(job["sample_ledger_path"])).read_text())
        expected_ids = {str(value) for value in ledger["sample_ids"]}
        namespace = f"baseline-full-{source_sha[:10]}-{task}-{shard:03d}"
        shard_root = root / "results" / task / f"{shard:03d}"
        marker = json.loads((shard_root / "job_complete.json").read_text())
        if marker.get("status") != "completed":
            raise RuntimeError(f"shard completion marker is not complete: {task}/{shard}")
        for backend in ("hf_eager", "vllm"):
            for method in BASELINES:
                run_id = f"{namespace}-{method}"
                if backend == "hf_eager":
                    path = shard_root / "hf" / run_id / "results.jsonl"
                else:
                    path = shard_root / "vllm" / method / f"{run_id}-vllm" / "results.jsonl"
                method_rows = _read_jsonl(path)
                observed_ids = {str(row.get("sample_id")) for row in method_rows}
                if len(method_rows) != len(expected_ids) or observed_ids != expected_ids:
                    raise RuntimeError(f"incomplete paired rows: {task}/{shard}/{backend}/{method}")
                if any(row.get("status") != "completed" for row in method_rows):
                    raise RuntimeError(f"failed sample: {task}/{shard}/{backend}/{method}")
                expected_effective = (
                    method if backend == "hf_eager" else f"{method}__vllm_requestwide_projection"
                )
                if {str(row.get("method")) for row in method_rows} != {expected_effective}:
                    raise RuntimeError(f"effective method label changed: {path}")
                for row in method_rows:
                    key = (task, str(row["sample_id"]), backend, method)
                    if key in seen:
                        raise RuntimeError(f"duplicate campaign row: {key}")
                    seen.add(key)
                    manifest_path = Path(str(row["manifest_path"]))
                    _validate_manifest(
                        manifest_path,
                        source_git_sha=source_sha,
                        backend=backend,
                    )
                    rows.append(
                        {
                            **row,
                            "task": task,
                            "backend_scope": backend,
                            "protocol_method": method,
                            "effective_method": expected_effective,
                            "shard_index": shard,
                            "source_git_sha": source_sha,
                            "result_path": str(path),
                            "cohort": "published_baseline_extension_9000",
                        }
                    )
    frame = pd.DataFrame(rows)
    if len(frame) != 54_000:
        raise RuntimeError(f"expected 54,000 measured extension rows, found {len(frame)}")
    observed = frame.groupby(["task", "backend_scope", "protocol_method"]).size()
    for task, sample_count in TASKS.items():
        for backend in REUSED_METHODS:
            for method in BASELINES:
                if int(observed.get((task, backend, method), 0)) != sample_count:
                    raise RuntimeError(f"row count changed: {task}/{backend}/{method}")
    return frame


def _reused_rows(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"task", "sample_id", "backend_scope", "protocol_method", "status"}
    if missing := required - set(frame.columns):
        raise RuntimeError(f"reused composite is missing columns: {sorted(missing)}")
    keep = pd.Series(False, index=frame.index)
    for backend, methods in REUSED_METHODS.items():
        keep |= frame["backend_scope"].eq(backend) & frame["protocol_method"].isin(methods)
    frame = frame[keep].copy()
    if len(frame) != 54_000 or set(frame["status"].astype(str)) != {"completed"}:
        raise RuntimeError("reused FullKV/Prompt/Mosaic composite is incomplete")
    frame["cohort"] = "reused_original_9000"
    return frame


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["task", "backend_scope", "protocol_method"], as_index=False)
        .agg(
            sample_count=("sample_id", "size"),
            task_score=("task_score", "mean"),
            active_kv_bytes=("active_kv_bytes", "mean"),
            residual_kv_bytes=("residual_kv_bytes", "mean"),
            compression_seconds=("compression_time", "mean"),
            end_to_end_seconds=("end_to_end_time", "mean"),
            failures=("status", lambda values: int((values != "completed").sum())),
        )
        .sort_values(["backend_scope", "task", "protocol_method"])
        .reset_index(drop=True)
    )


def _comparisons(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    competitors = ("prompt_attention_topk", *BASELINES)
    for backend, mosaic_method in MOSAIC_METHOD.items():
        for task in TASKS:
            subset = frame[frame["backend_scope"].eq(backend) & frame["task"].eq(task)]
            mosaic = subset[subset["protocol_method"].eq(mosaic_method)].set_index("sample_id")
            for competitor in competitors:
                other = subset[subset["protocol_method"].eq(competitor)].set_index("sample_id")
                if len(mosaic) != TASKS[task] or not mosaic.index.equals(other.index):
                    mosaic = mosaic.sort_index()
                    other = other.sort_index()
                if len(mosaic) != TASKS[task] or not mosaic.index.equals(other.index):
                    raise RuntimeError(f"pairing changed: {task}/{backend}/{competitor}")
                score_delta = (
                    mosaic["task_score"].astype(float).to_numpy()
                    - other["task_score"].astype(float).to_numpy()
                )
                low, high = _bootstrap_interval(
                    score_delta,
                    seed=20271001 + len(records),
                )
                mosaic_bytes = float(mosaic["active_kv_bytes"].astype(float).mean())
                other_bytes = float(other["active_kv_bytes"].astype(float).mean())
                reduction = 1.0 - mosaic_bytes / other_bytes
                records.append(
                    {
                        "task": task,
                        "backend_scope": backend,
                        "competitor": competitor,
                        "sample_count": len(mosaic),
                        "mosaickv_method": mosaic_method,
                        "mosaickv_score": float(mosaic["task_score"].astype(float).mean()),
                        "competitor_score": float(other["task_score"].astype(float).mean()),
                        "paired_score_delta": float(score_delta.mean()),
                        "paired_score_delta_ci95_low": low,
                        "paired_score_delta_ci95_high": high,
                        "mosaickv_active_kv_bytes": mosaic_bytes,
                        "competitor_active_kv_bytes": other_bytes,
                        "active_memory_reduction_fraction": reduction,
                        "answer_agreement_fraction": float(
                            (
                                mosaic["answer"].astype(str).to_numpy()
                                == other["answer"].astype(str).to_numpy()
                            ).mean()
                        ),
                        "memory_gate_30pct": reduction >= 0.30 - 1e-12,
                        "quality_noninferiority_point_gate": float(score_delta.mean()) >= -1e-12,
                    }
                )
    return pd.DataFrame(records)


def analyze(campaign_root: Path, reused_composite: Path) -> dict[str, Any]:
    campaign = _campaign_rows(campaign_root)
    reused = _reused_rows(reused_composite)
    combined = pd.concat([reused, campaign], ignore_index=True, sort=False)
    key_columns = ["task", "sample_id", "backend_scope", "protocol_method"]
    if combined.duplicated(key_columns).any():
        raise RuntimeError("combined result contains duplicate sample/method rows")
    summary = _summary(combined)
    comparisons = _comparisons(combined)
    analysis_root = campaign_root / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)
    campaign.to_parquet(analysis_root / "published_baseline_rows_54000.parquet", index=False)
    combined.to_parquet(analysis_root / "combined_rows_108000.parquet", index=False)
    summary.to_parquet(analysis_root / "summary.parquet", index=False)
    comparisons.to_parquet(analysis_root / "paired_comparisons.parquet", index=False)
    payload = {
        "schema_version": 1,
        "classification": "canonical_published_baseline_extension_to_existing_9000",
        "campaign_rows": len(campaign),
        "reused_rows": len(reused),
        "combined_rows": len(combined),
        "failed_rows": int((combined["status"] != "completed").sum()),
        "tasks": TASKS,
        "backends": list(REUSED_METHODS),
        "published_baselines": list(BASELINES),
        "semantic_limit": (
            "vLLM LOOK-M, PrefixKV, and VL-Cache results are explicitly labeled measured "
            "request-wide whole-block projections, not official/native per-layer engine methods"
        ),
    }
    (analysis_root / "validation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# Published-baseline full campaign",
        "",
        f"Validated **{len(combined):,}** rows with **{payload['failed_rows']}** failures.",
        "",
        "The active-KV byte count is the primary comparison budget. vLLM published-baseline "
        "rows are measured request-wide whole-block projections and are not described as "
        "official or native per-layer engine implementations.",
        "",
        "## Per-task and backend summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Paired MosaicKV comparisons",
        "",
        comparisons.to_markdown(index=False),
        "",
    ]
    (analysis_root / "report.md").write_text("\n".join(lines))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--reused-composite", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.campaign_root.resolve(), args.reused_composite.resolve())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
