#!/usr/bin/env python3
"""Validate and aggregate the paired HF byte-budget selector curves."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

TASKS = ("mmstar", "textvqa", "docvqa")
TASK_LABELS = {"mmstar": "MMStar", "textvqa": "TextVQA", "docvqa": "DocVQA"}
METHODS = ("prompt_attention_topk", "mosaickv_exact")
METHOD_LABELS = {
    "prompt_attention_topk": "Prompt Top-K",
    "mosaickv_exact": "MosaicKV exact",
}
BUDGETS_MIB = (16, 22, 30, 40, 60)
CLASSIFICATION = "canonical_byte_matched_selector_curve"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_index(path: Path) -> list[dict[str, Any]]:
    rows = cast(
        "list[dict[str, Any]]",
        [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ],
    )
    if len(rows) != 48 or [int(row["array_index"]) for row in rows] != list(range(48)):
        raise RuntimeError("byte-curve index must contain exactly 48 ordered jobs")
    return rows


def _validate_manifest(
    path: Path,
    *,
    source_sha: str,
    config_sha: str,
    budget_bytes: int,
    dataset: dict[str, Any],
) -> dict[str, Any]:
    manifest = cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))
    source = cast("dict[str, Any]", manifest["source"])
    if source.get("git_sha") != source_sha or source.get("git_dirty") is not False:
        raise RuntimeError(f"manifest does not record the clean execution source: {path}")
    if source.get("canonical_eligible") is not True or source.get("config_sha") != config_sha:
        raise RuntimeError(f"manifest is not canonical for its resolved configuration: {path}")
    execution = cast("dict[str, Any]", manifest["execution"])
    expected_execution = {
        "backend": "huggingface",
        "attention_implementation": "eager",
        "decode_attention_implementation": "eager",
        "deterministic_algorithms": True,
        "seed": 20271001,
    }
    if any(execution.get(key) != value for key, value in expected_execution.items()):
        raise RuntimeError(f"manifest backend controls changed: {path}")
    model = cast("dict[str, Any]", manifest["model"])
    expected_model = {
        "id": "llava-hf/llava-1.5-7b-hf",
        "revision": "b234b804b114d9e37bb655e11cbbb5f5e971b7a9",
        "precision": "fp16",
    }
    if any(model.get(key) != value for key, value in expected_model.items()):
        raise RuntimeError(f"manifest model controls changed: {path}")
    observed_dataset = cast("dict[str, Any]", manifest["dataset"])
    expected_dataset = {
        "id": dataset["dataset_id"],
        "revision": dataset["dataset_revision"],
        "split": dataset["split"],
    }
    if any(observed_dataset.get(key) != value for key, value in expected_dataset.items()):
        raise RuntimeError(f"manifest dataset controls changed: {path}")
    cache = cast("dict[str, Any]", manifest["cache"])
    if (
        cache.get("budget_unit") != "bytes"
        or cache.get("budget_value") != budget_bytes
        or cache.get("retention_ratio") != 0.99
    ):
        raise RuntimeError(f"manifest byte budget changed: {path}")
    resolved = cast("dict[str, Any]", manifest["resolved_config"])
    resolved_cache = cast("dict[str, Any]", resolved["cache"])
    expected_cache = {
        "block_size": 4,
        "physical_budget_block_size": 4,
        "physical_budget_mandatory_policy": "last_token",
        "exact_gather_mode": "source_views",
        "packing_mode": "preallocated_layer",
    }
    if any(resolved_cache.get(key) != value for key, value in expected_cache.items()):
        raise RuntimeError(f"manifest physical cache policy changed: {path}")
    return manifest


def collect(index_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    jobs = _read_index(index_path)
    result_rows: list[dict[str, Any]] = []
    byte_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int, str]] = set()
    input_hashes: dict[tuple[str, str], dict[str, str]] = {}
    source_shas: set[str] = set()
    protocol_hashes: set[str] = set()
    candidate_hashes: set[str] = set()
    generation_hashes: set[str] = set()
    for job in jobs:
        task = str(job["task"])
        shard = int(job["shard_index"])
        if task not in TASKS:
            raise RuntimeError(f"unsupported byte-curve task: {task}")
        if tuple(int(value) for value in job["budgets_mib"]) != BUDGETS_MIB:
            raise RuntimeError("job budget sweep changed")
        if tuple(str(value) for value in job["methods"]) != METHODS:
            raise RuntimeError("job method order changed")
        job_path = Path(str(job["job_spec_path"])).resolve()
        if _sha(job_path) != job["job_spec_sha256"]:
            raise RuntimeError(f"job specification hash changed: {job_path}")
        ledger_path = Path(str(job["sample_ledger_path"])).resolve()
        if _sha(ledger_path) != job["sample_ledger_sha256"]:
            raise RuntimeError(f"sample ledger hash changed: {ledger_path}")
        ledger = cast("dict[str, Any]", json.loads(ledger_path.read_text(encoding="utf-8")))
        expected_ids = tuple(str(value) for value in cast("list[str]", ledger["sample_ids"]))
        completion_path = (
            Path(str(job["output_root"])) / task / f"{shard:02d}" / "job_complete.json"
        )
        completion = cast("dict[str, Any]", json.loads(completion_path.read_text(encoding="utf-8")))
        if completion.get("status") != "completed":
            raise RuntimeError(f"byte-curve job did not complete: {task}/{shard}")
        if completion.get("job_spec_sha256") != job["job_spec_sha256"]:
            raise RuntimeError(f"completion seal changed identity: {task}/{shard}")
        source_sha = str(completion["source_git_sha"])
        source_shas.add(source_sha)
        protocol_hashes.add(str(completion["protocol_sha256"]))
        candidate_hashes.add(str(completion["candidate_config_sha256"]))
        records = cast("list[dict[str, Any]]", completion["records"])
        if len(records) != len(METHODS) * len(BUDGETS_MIB):
            raise RuntimeError(f"incomplete method/budget records: {task}/{shard}")
        record_keys: set[tuple[int, str]] = set()
        for record in records:
            budget_mib = int(record["budget_mib"])
            budget_bytes = int(record["budget_bytes"])
            method = str(record["method"])
            if budget_mib not in BUDGETS_MIB or budget_bytes != budget_mib * 2**20:
                raise RuntimeError(f"record has a non-protocol budget: {task}/{shard}")
            if method not in METHODS or (budget_mib, method) in record_keys:
                raise RuntimeError(f"record has a duplicate or unsupported method: {task}/{shard}")
            record_keys.add((budget_mib, method))
            config_sha = str(record["config_sha256"])
            rows = cast("list[dict[str, Any]]", record["rows"])
            if len(rows) != len(expected_ids) or {str(row["sample_id"]) for row in rows} != set(
                expected_ids
            ):
                raise RuntimeError(f"incomplete sample rows: {task}/{shard}/{budget_mib}/{method}")
            for row in rows:
                sample_id = str(row["sample_id"])
                key = (task, sample_id, budget_mib, method)
                if key in seen:
                    raise RuntimeError(f"duplicate byte-curve row: {key}")
                seen.add(key)
                if row.get("status") != "completed" or row.get("error") is not None:
                    raise RuntimeError(f"failed byte-curve sample: {key}")
                required_numeric = (
                    "task_score",
                    "active_kv_bytes",
                    "compression_time",
                    "decode_time",
                    "end_to_end_time",
                    "generated_tokens",
                )
                if any(pd.isna(row.get(field)) for field in required_numeric):
                    raise RuntimeError(f"missing required metric for {key}")
                if int(row["active_kv_bytes"]) > budget_bytes:
                    raise RuntimeError(f"active KV bytes exceed the hard cap for {key}")
                if int(row["generated_tokens"]) != int(job["output_length"]):
                    raise RuntimeError(f"output length changed for {key}")
                manifest = _validate_manifest(
                    Path(str(row["manifest_path"])),
                    source_sha=source_sha,
                    config_sha=config_sha,
                    budget_bytes=budget_bytes,
                    dataset=ledger,
                )
                generation_hashes.add(
                    str(cast("dict[str, Any]", manifest["generation"])["parameters_sha"])
                )
                inputs = cast("dict[str, str]", manifest["inputs"])
                input_key = (task, sample_id)
                hashes = {
                    name: str(inputs[name])
                    for name in (
                        "prompt_set_sha",
                        "media_set_sha",
                        "tokenization_sha",
                        "preprocessing_sha",
                    )
                }
                previous = input_hashes.setdefault(input_key, hashes)
                if previous != hashes:
                    raise RuntimeError(
                        f"paired prompt/media/tokenization hashes differ: {input_key}"
                    )
                result_rows.append(
                    {
                        **row,
                        "task": task,
                        "shard_index": shard,
                        "protocol_method": method,
                        "budget_mib": budget_mib,
                        "configured_budget_bytes": budget_bytes,
                        "config_sha256": config_sha,
                        "source_git_sha": source_sha,
                        "protocol_sha256": str(completion["protocol_sha256"]),
                        "sample_ledger_sha256": str(completion["sample_ledger_sha256"]),
                    }
                )
        checks = cast("list[dict[str, Any]]", completion["byte_match_checks"])
        if len(checks) != len(expected_ids) * len(BUDGETS_MIB):
            raise RuntimeError(f"incomplete byte checks: {task}/{shard}")
        for check in checks:
            if (
                check.get("exact_byte_match") is not True
                or check.get("within_hard_cap") is not True
            ):
                raise RuntimeError(f"failed byte-match check: {check}")
            byte_rows.append({"task": task, "shard_index": shard, **check})

    if (
        len(source_shas) != 1
        or len(protocol_hashes) != 1
        or len(candidate_hashes) != 1
        or len(generation_hashes) != 1
    ):
        raise RuntimeError("campaign mixes source, protocol, candidate, or generation hashes")
    rows = pd.DataFrame(result_rows)
    bytes_frame = pd.DataFrame(byte_rows)
    expected_per_task = 256 * len(METHODS) * len(BUDGETS_MIB)
    if len(rows) != len(TASKS) * expected_per_task:
        raise RuntimeError(f"expected 7,680 result rows, found {len(rows)}")
    if len(bytes_frame) != len(TASKS) * 256 * len(BUDGETS_MIB):
        raise RuntimeError("byte validation row count changed")
    metadata = {
        "source_git_sha": next(iter(source_shas)),
        "protocol_sha256": next(iter(protocol_hashes)),
        "candidate_config_sha256": next(iter(candidate_hashes)),
        "generation_parameters_sha256": next(iter(generation_hashes)),
        "job_count": len(jobs),
    }
    return rows, bytes_frame, metadata


def _bootstrap(
    values: np.ndarray[Any, np.dtype[np.float64]], *, seed: int, count: int
) -> tuple[float, float]:
    if values.ndim != 1 or not len(values):
        raise ValueError("paired bootstrap requires a nonempty vector")
    rng = np.random.default_rng(seed)
    means = np.empty(count, dtype=np.float64)
    for index in range(count):
        means[index] = float(values[rng.integers(0, len(values), len(values))].mean())
    quantiles = np.quantile(means, (0.025, 0.975))
    return float(quantiles[0]), float(quantiles[1])


def summarize(
    rows: pd.DataFrame,
    *,
    bootstrap_seed: int = 20271012,
    bootstrap_resamples: int = 10000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        rows.groupby(["task", "budget_mib", "protocol_method"], as_index=False)
        .agg(
            sample_count=("sample_id", "size"),
            task_score=("task_score", "mean"),
            active_kv_bytes=("active_kv_bytes", "mean"),
            compression_seconds=("compression_time", "mean"),
            decode_seconds=("decode_time", "mean"),
            end_to_end_seconds=("end_to_end_time", "mean"),
        )
        .sort_values(["task", "budget_mib", "protocol_method"])
        .reset_index(drop=True)
    )
    comparisons: list[dict[str, Any]] = []
    for task_index, task in enumerate(TASKS):
        for budget_index, budget_mib in enumerate(BUDGETS_MIB):
            subset = rows[rows["task"].eq(task) & rows["budget_mib"].eq(budget_mib)]
            prompt = (
                subset[subset["protocol_method"].eq("prompt_attention_topk")]
                .set_index("sample_id")
                .sort_index()
            )
            mosaic = (
                subset[subset["protocol_method"].eq("mosaickv_exact")]
                .set_index("sample_id")
                .sort_index()
            )
            if not len(prompt) or not prompt.index.equals(mosaic.index):
                raise RuntimeError(f"paired curve identities changed: {task}/{budget_mib}")
            if not np.array_equal(
                prompt["reference"].astype(str).to_numpy(),
                mosaic["reference"].astype(str).to_numpy(),
            ):
                raise RuntimeError(f"paired references differ: {task}/{budget_mib}")
            prompt_bytes = prompt["active_kv_bytes"].astype(int).to_numpy()
            mosaic_bytes = mosaic["active_kv_bytes"].astype(int).to_numpy()
            if not np.array_equal(prompt_bytes, mosaic_bytes):
                raise RuntimeError(f"realized active bytes differ: {task}/{budget_mib}")
            delta = (
                mosaic["task_score"].astype(float).to_numpy()
                - prompt["task_score"].astype(float).to_numpy()
            )
            low, high = _bootstrap(
                delta,
                seed=bootstrap_seed + task_index * len(BUDGETS_MIB) + budget_index,
                count=bootstrap_resamples,
            )
            comparisons.append(
                {
                    "task": task,
                    "budget_mib": budget_mib,
                    "sample_count": len(prompt),
                    "configured_budget_bytes": budget_mib * 2**20,
                    "realized_active_kv_bytes": float(prompt_bytes.mean()),
                    "realized_active_kv_mib": float(prompt_bytes.mean() / 2**20),
                    "exact_per_sample_byte_match": True,
                    "prompt_score": float(prompt["task_score"].astype(float).mean()),
                    "mosaickv_score": float(mosaic["task_score"].astype(float).mean()),
                    "paired_score_delta": float(delta.mean()),
                    "paired_score_delta_ci95_low": low,
                    "paired_score_delta_ci95_high": high,
                    "answer_agreement_fraction": float(
                        (
                            prompt["answer"].astype(str).to_numpy()
                            == mosaic["answer"].astype(str).to_numpy()
                        ).mean()
                    ),
                }
            )
    return summary, pd.DataFrame(comparisons)


def _latex_table(comparisons: pd.DataFrame) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        (
            r"\caption{Byte-matched HF selector comparison. Both methods use the exact "
            r"same per-sample active KV bytes. Score differences and paired 95\% bootstrap "
            r"intervals are in percentage points.}"
        ),
        r"\label{tab:hf-byte-matched-curves}",
        r"\scriptsize",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        (
            r"Task & Cap (MiB) & Realized (MiB) & Prompt Top-K & MosaicKV & "
            r"$\Delta$ score & 95\% CI & Agreement \\"
        ),
        r"\midrule",
    ]
    for task_index, task in enumerate(TASKS):
        subset = comparisons[comparisons["task"].eq(task)].sort_values("budget_mib")
        for row in subset.itertuples():
            lines.append(
                f"{TASK_LABELS[task]} & {int(row.budget_mib)} & {row.realized_active_kv_mib:.2f} & "
                f"{100 * row.prompt_score:.2f} & {100 * row.mosaickv_score:.2f} & "
                f"{100 * row.paired_score_delta:+.2f} & "
                f"$[{100 * row.paired_score_delta_ci95_low:.2f}, "
                f"{100 * row.paired_score_delta_ci95_high:.2f}]$ & "
                f"{100 * row.answer_agreement_fraction:.2f} \\\\"
            )
        if task_index != len(TASKS) - 1:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def _tikz(summary: pd.DataFrame) -> str:
    lines = [
        r"\begin{figure*}[t]",
        r"\centering",
        r"\begin{tikzpicture}",
        (
            r"\begin{groupplot}[group style={group size=3 by 1,horizontal sep=1.0cm},"
            r"width=0.31\textwidth,height=0.24\textwidth,"
            r"xlabel={Measured active KV (MiB)},ylabel={Task score (\%)},grid=both]"
        ),
    ]
    for task in TASKS:
        task_rows = summary[summary["task"].eq(task)]
        values = task_rows["task_score"].astype(float) * 100
        pad = max(0.5, float(values.max() - values.min()) * 0.15)
        ymin = max(0.0, float(values.min() - pad))
        ymax = min(100.0, float(values.max() + pad))
        lines.append(
            f"\\nextgroupplot[title={{{TASK_LABELS[task]}}},ymin={ymin:.3f},ymax={ymax:.3f}]"
        )
        for method, style in (
            ("prompt_attention_topk", "promptstyle"),
            ("mosaickv_exact", "mosaicstyle"),
        ):
            subset = task_rows[task_rows["protocol_method"].eq(method)].sort_values(
                "active_kv_bytes"
            )
            coordinates = " ".join(
                f"({float(row.active_kv_bytes) / 2**20:.4f},{100 * float(row.task_score):.4f})"
                for row in subset.itertuples()
            )
            lines.append(f"\\addplot[{style}] coordinates {{{coordinates}}};")
            if task == TASKS[0]:
                lines.append(f"\\addlegendentry{{{METHOD_LABELS[method]}}}")
    lines.extend(
        [
            r"\end{groupplot}",
            r"\end{tikzpicture}",
            (
                r"\caption{Byte-matched HF quality--memory curves. Each paired point shares "
                r"the same samples, hard byte cap, four-token block granularity, "
                r"mandatory-token policy, model, processor, and deterministic generation "
                r"settings.}"
            ),
            r"\label{fig:hf-byte-matched-curves}",
            r"\end{figure*}",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(index_path: Path, analysis_root: Path) -> dict[str, Any]:
    if analysis_root.exists():
        raise FileExistsError(f"refusing to replace analysis directory: {analysis_root}")
    rows, byte_checks, metadata = collect(index_path)
    summary, comparisons = summarize(rows)
    analysis_root.mkdir(parents=True, exist_ok=False)
    rows.to_parquet(analysis_root / "rows.parquet", index=False)
    byte_checks.to_parquet(analysis_root / "byte_match_validation.parquet", index=False)
    summary.to_parquet(analysis_root / "summary.parquet", index=False)
    comparisons.to_parquet(analysis_root / "paired_comparisons.parquet", index=False)
    (analysis_root / "byte_matched_table.tex").write_text(
        _latex_table(comparisons), encoding="utf-8"
    )
    (analysis_root / "figure3_byte_matched_curves.tex").write_text(_tikz(summary), encoding="utf-8")
    winning = comparisons[comparisons["paired_score_delta"].ge(0)]
    payload = {
        "schema_version": 1,
        "classification": CLASSIFICATION,
        **metadata,
        "result_rows": len(rows),
        "failed_rows": 0,
        "tasks": list(TASKS),
        "methods": list(METHODS),
        "budgets_mib": list(BUDGETS_MIB),
        "exact_per_sample_byte_match": bool(comparisons["exact_per_sample_byte_match"].all()),
        "mosaickv_nondegenerate_point_wins": len(winning),
        "comparison_count": len(comparisons),
        "claim_scope": (
            "paired HF eager selector-policy curve; not a held-out algorithm-tuning result"
        ),
    }
    (analysis_root / "validation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# HF byte-matched selector curves",
        "",
        "This analysis compares MosaicKV exact selection with Prompt Top-K at identical "
        "per-sample active KV byte counts. It is a fixed curve-characterization subset of "
        "previously evaluated examples, not a new held-out model-quality claim.",
        "",
        "## Paired comparison",
        "",
        comparisons.to_markdown(index=False),
        "",
        "## Per-method curve points",
        "",
        summary.to_markdown(index=False),
        "",
    ]
    (analysis_root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.index.resolve(), args.analysis_root.resolve())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
