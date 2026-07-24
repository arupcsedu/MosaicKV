#!/usr/bin/env python3
"""Derive paired quality-memory tables and Pareto analyses from a complete campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from mosaickv.config import load_config
from mosaickv.evaluation.storage import load_jsonl
from mosaickv.evaluation.tasks import default_task_registry
from mosaickv.experiment_matrix import verify_expanded_index


@dataclass(frozen=True, slots=True)
class Observation:
    model: str
    model_key: str
    task: str
    method: str
    effective_method: str
    retention_ratio: float
    block_size: int
    output_length: int
    seed: int
    sample_id: str
    task_score: float
    active_kv_bytes: int
    residual_kv_bytes: int
    total_latency: float
    decode_latency: float
    compression_time: float
    peak_gpu_memory: int
    generated_token_ids: tuple[int, ...]
    logit_kl: float | None
    attention_output_error: float | None
    run_id: str
    manifest_path: str


def _canonical_line(value: Mapping[str, object]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _atomic_text(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite derived artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = handle.name
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
        raise


def _finite(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be a finite measured number")
    return float(value)


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a measured integer")
    return value


def _analysis_source() -> tuple[str, bool]:
    repository = Path(__file__).resolve().parents[2]
    sha = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return sha, bool(status.strip())


def _trace_for_sample(trace_root: Path, run_id: str, sample_id: str) -> dict[str, Any]:
    matches: list[tuple[int, dict[str, Any]]] = []
    for path in sorted((trace_root / run_id).glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("sample_id") == sample_id and value.get("status") == "completed":
            marker = "-attempt-"
            attempt = 0 if marker not in path.stem else int(path.stem.rsplit(marker, 1)[1])
            matches.append((attempt, cast("dict[str, Any]", value)))
    if not matches:
        raise ValueError(f"no completed trace for {run_id}/{sample_id}")
    # A killed Slurm attempt can commit a trace before lmms-eval commits its scored
    # row. The successful retry always receives the highest immutable attempt ID.
    return max(matches, key=lambda item: item[0])[1]


def _load_observations(index: Path, run_root: Path) -> tuple[list[Observation], dict[str, Any]]:
    verify_expanded_index(index)
    manifest = json.loads((index.parent / "matrix_manifest.json").read_text(encoding="utf-8"))
    observations: list[Observation] = []
    seen: set[tuple[str, str]] = set()
    for line in index.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        job = json.loads(line)
        run_id = str(job["run_id"])
        run_directory = run_root / run_id
        raw_path = run_directory / "results.jsonl"
        manifest_path = run_directory / "manifest.json"
        if not raw_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"incomplete campaign job: {run_id}")
        run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if run_manifest.get("source", {}).get("git_dirty") is not False:
            raise ValueError(f"campaign manifest is dirty: {run_id}")
        if run_manifest.get("source", {}).get("config_sha") != job["config_sha"]:
            raise ValueError(f"config SHA mismatch: {run_id}")
        if (
            run_manifest.get("artifacts", {}).get("raw_output_sha")
            != hashlib.sha256(raw_path.read_bytes()).hexdigest()
        ):
            raise ValueError(f"raw output SHA mismatch: {run_id}")
        sample_payload = json.loads(Path(job["sample_id_file"]).read_text(encoding="utf-8"))
        expected_ids = tuple(str(value) for value in sample_payload["sample_ids"])
        rows = load_jsonl(raw_path)
        if {row.sample_id for row in rows} != set(expected_ids) or len(rows) != len(expected_ids):
            raise ValueError(f"sample shard coverage mismatch: {run_id}")
        config = load_config(job["config_path"])
        for row in rows:
            identity = (run_id, row.sample_id)
            if identity in seen:
                raise ValueError(f"duplicate run/sample row: {identity}")
            seen.add(identity)
            if row.status.value != "completed":
                raise ValueError(
                    f"failed main-quality sample: {run_id}/{row.sample_id}: {row.error}"
                )
            trace = _trace_for_sample(Path(job["trace_directory"]), run_id, row.sample_id)
            if trace.get("active_kv_bytes") != row.active_kv_bytes:
                raise ValueError(f"selected trace/result byte mismatch: {run_id}/{row.sample_id}")
            token_ids = trace.get("generated_token_ids")
            if not isinstance(token_ids, list) or not all(
                isinstance(value, int) and not isinstance(value, bool) for value in token_ids
            ):
                raise ValueError(f"missing generated token IDs: {run_id}/{row.sample_id}")
            if len(token_ids) != config.generation.max_new_tokens:
                raise ValueError(f"output length mismatch: {run_id}/{row.sample_id}")
            observations.append(
                Observation(
                    model=row.model,
                    model_key=str(job["model_key"]),
                    task=str(job["task_key"]),
                    method=str(job["method"]),
                    effective_method=row.method,
                    retention_ratio=float(job["retention_ratio"]),
                    block_size=config.cache.block_size,
                    output_length=config.generation.max_new_tokens,
                    seed=int(job["selection_seed"]),
                    sample_id=row.sample_id,
                    task_score=_finite(row.task_score, "task_score"),
                    active_kv_bytes=_integer(row.active_kv_bytes, "active_kv_bytes"),
                    residual_kv_bytes=_integer(row.residual_kv_bytes, "residual_kv_bytes"),
                    total_latency=_finite(row.end_to_end_time, "end_to_end_time"),
                    decode_latency=_finite(row.decode_time, "decode_time"),
                    compression_time=_finite(row.compression_time, "compression_time"),
                    peak_gpu_memory=_integer(row.peak_gpu_memory, "peak_gpu_memory"),
                    generated_token_ids=tuple(token_ids),
                    logit_kl=(None if row.logit_kl is None else _finite(row.logit_kl, "logit_kl")),
                    attention_output_error=(
                        None
                        if row.attention_output_error is None
                        else _finite(row.attention_output_error, "attention_output_error")
                    ),
                    run_id=run_id,
                    manifest_path=str(manifest_path),
                )
            )
    return observations, cast("dict[str, Any]", manifest)


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise ValueError("cannot average an empty measured group")
    return statistics.fmean(materialized)


def _bootstrap_quality(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    seed: int,
    iterations: int,
) -> dict[str, float]:
    if candidate.shape != reference.shape or candidate.ndim != 1 or candidate.size < 2:
        raise ValueError("paired bootstrap requires equal one-dimensional cohorts of size >= 2")
    rng = np.random.default_rng(seed)
    candidate_means = np.empty(iterations, dtype=np.float64)
    delta_means = np.empty(iterations, dtype=np.float64)
    delta = candidate - reference
    batch_size = 128
    for start in range(0, iterations, batch_size):
        count = min(batch_size, iterations - start)
        indices = rng.integers(0, candidate.size, size=(count, candidate.size))
        candidate_means[start : start + count] = candidate[indices].mean(axis=1)
        delta_means[start : start + count] = delta[indices].mean(axis=1)
    return {
        "task_score_ci_low": float(np.quantile(candidate_means, 0.025)),
        "task_score_ci_high": float(np.quantile(candidate_means, 0.975)),
        "score_delta_ci_low": float(np.quantile(delta_means, 0.025)),
        "score_delta_ci_high": float(np.quantile(delta_means, 0.975)),
    }


def _derive_per_sample(observations: Sequence[Observation]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, float, int, str], list[Observation]] = defaultdict(list)
    references: dict[tuple[str, str, int, str], Observation] = {}
    for observation in observations:
        group = (
            observation.model,
            observation.task,
            observation.method,
            observation.retention_ratio,
            observation.output_length,
            observation.sample_id,
        )
        grouped[group].append(observation)
        if observation.method == "full_kv":
            key = (
                observation.model,
                observation.task,
                observation.output_length,
                observation.sample_id,
            )
            if key in references:
                raise ValueError(f"multiple FullKV references for {key}")
            references[key] = observation
    derived: list[dict[str, object]] = []
    for group, candidates in sorted(grouped.items()):
        model, task, method, ratio, output_length, sample_id = group
        reference_key = (model, task, output_length, sample_id)
        reference = references.get(reference_key)
        if reference is None:
            raise ValueError(f"missing paired FullKV reference for {reference_key}")
        active_values = {candidate.active_kv_bytes for candidate in candidates}
        if len(active_values) != 1:
            raise ValueError(f"selection seeds produced different active byte budgets: {group}")
        active_bytes = next(iter(active_values))
        candidate_tokens = [candidate.generated_token_ids for candidate in candidates]
        if any(len(tokens) != len(reference.generated_token_ids) for tokens in candidate_tokens):
            raise ValueError(f"generated token lengths differ from FullKV: {group}")
        token_agreements = [
            _mean(
                float(left == right)
                for left, right in zip(tokens, reference.generated_token_ids, strict=True)
            )
            for tokens in candidate_tokens
        ]
        sequence_agreements = [
            float(tokens == reference.generated_token_ids) for tokens in candidate_tokens
        ]
        diagnostic_kl = [
            value for candidate in candidates if (value := candidate.logit_kl) is not None
        ]
        diagnostic_attention = [
            value
            for candidate in candidates
            if (value := candidate.attention_output_error) is not None
        ]
        derived.append(
            {
                "classification": "derived_from_measured_main_quality",
                "model": model,
                "model_key": candidates[0].model_key,
                "task": task,
                "metric": default_task_registry()
                .resolve("video_mme" if task == "videomme" else task)
                .sample_metric,
                "method": method,
                "effective_methods": json.dumps(
                    sorted({candidate.effective_method for candidate in candidates}),
                    separators=(",", ":"),
                ),
                "retention_ratio": ratio,
                "block_size": candidates[0].block_size,
                "output_length": output_length,
                "sample_id": sample_id,
                "selection_seeds": json.dumps(
                    sorted({candidate.seed for candidate in candidates}), separators=(",", ":")
                ),
                "task_score": _mean(candidate.task_score for candidate in candidates),
                "fullkv_task_score": reference.task_score,
                "score_delta": _mean(candidate.task_score for candidate in candidates)
                - reference.task_score,
                "active_kv_bytes": active_bytes,
                "residual_kv_bytes": _mean(candidate.residual_kv_bytes for candidate in candidates),
                "total_kv_bytes": active_bytes
                + _mean(candidate.residual_kv_bytes for candidate in candidates),
                "fullkv_active_kv_bytes": reference.active_kv_bytes,
                "compression_ratio": active_bytes / reference.active_kv_bytes,
                "token_agreement": _mean(token_agreements),
                "sequence_agreement": _mean(sequence_agreements),
                "logit_kl": None if not diagnostic_kl else _mean(diagnostic_kl),
                "attention_output_error": (
                    None if not diagnostic_attention else _mean(diagnostic_attention)
                ),
                "total_latency": _mean(candidate.total_latency for candidate in candidates),
                "decode_latency": _mean(candidate.decode_latency for candidate in candidates),
                "compression_time": _mean(candidate.compression_time for candidate in candidates),
                "peak_gpu_memory": _mean(candidate.peak_gpu_memory for candidate in candidates),
                "manifest_paths": json.dumps(
                    sorted({candidate.manifest_path for candidate in candidates}),
                    separators=(",", ":"),
                ),
            }
        )
    return derived


def _summarize(
    rows: Sequence[Mapping[str, object]], *, bootstrap_iterations: int, bootstrap_seed: int
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["model"],
                row["model_key"],
                row["task"],
                row["metric"],
                row["method"],
                row["effective_methods"],
                row["retention_ratio"],
                row["block_size"],
                row["output_length"],
                row["selection_seeds"],
            )
        ].append(row)
    summaries: list[dict[str, object]] = []
    for group_index, (key, samples) in enumerate(sorted(grouped.items())):
        (
            model,
            model_key,
            task,
            metric,
            method,
            effective_methods,
            ratio,
            block_size,
            output_length,
            selection_seeds,
        ) = key
        candidate = np.asarray([row["task_score"] for row in samples], dtype=np.float64)
        reference = np.asarray([row["fullkv_task_score"] for row in samples], dtype=np.float64)
        ci = _bootstrap_quality(
            candidate,
            reference,
            seed=bootstrap_seed + group_index,
            iterations=bootstrap_iterations,
        )
        summary: dict[str, object] = {
            "classification": "derived_from_measured_main_quality",
            "model": model,
            "model_key": model_key,
            "task": task,
            "metric": metric,
            "method": method,
            "effective_methods": effective_methods,
            "retention_ratio": ratio,
            "block_size": block_size,
            "output_length": output_length,
            "selection_seeds": selection_seeds,
            "sample_count": len(samples),
            "diagnostic_sample_count": sum(row["logit_kl"] is not None for row in samples),
            "task_score": _mean(cast("float", row["task_score"]) for row in samples),
            "score_delta": _mean(cast("float", row["score_delta"]) for row in samples),
            "active_kv_bytes": _mean(cast("float", row["active_kv_bytes"]) for row in samples),
            "residual_kv_bytes": _mean(cast("float", row["residual_kv_bytes"]) for row in samples),
            "total_kv_bytes": _mean(cast("float", row["total_kv_bytes"]) for row in samples),
            "compression_ratio": _mean(cast("float", row["compression_ratio"]) for row in samples),
            "token_agreement": _mean(cast("float", row["token_agreement"]) for row in samples),
            "sequence_agreement": _mean(
                cast("float", row["sequence_agreement"]) for row in samples
            ),
            "logit_kl": (
                None
                if not any(row["logit_kl"] is not None for row in samples)
                else _mean(
                    cast("float", row["logit_kl"]) for row in samples if row["logit_kl"] is not None
                )
            ),
            "attention_output_error": (
                None
                if not any(row["attention_output_error"] is not None for row in samples)
                else _mean(
                    cast("float", row["attention_output_error"])
                    for row in samples
                    if row["attention_output_error"] is not None
                )
            ),
            "total_latency": _mean(cast("float", row["total_latency"]) for row in samples),
            "decode_latency": _mean(cast("float", row["decode_latency"]) for row in samples),
            "compression_time": _mean(cast("float", row["compression_time"]) for row in samples),
            "peak_gpu_memory": _mean(cast("float", row["peak_gpu_memory"]) for row in samples),
            "bootstrap_iterations": bootstrap_iterations,
            **ci,
        }
        summaries.append(summary)
    return summaries


def _pareto(summaries: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    resources = {
        "quality_vs_active_kv_memory": "active_kv_bytes",
        "quality_vs_total_latency": "total_latency",
        "quality_vs_decode_latency": "decode_latency",
    }
    strata: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in summaries:
        strata[(row["model"], row["task"], row["metric"], row["output_length"])].append(row)
    for stratum, points in sorted(strata.items()):
        for analysis, resource in resources.items():
            for point in points:
                quality = cast("float", point["task_score"])
                cost = cast("float", point[resource])
                dominators = [
                    other
                    for other in points
                    if cast("float", other["task_score"]) >= quality
                    and cast("float", other[resource]) <= cost
                    and (
                        cast("float", other["task_score"]) > quality
                        or cast("float", other[resource]) < cost
                    )
                ]
                result.append(
                    {
                        "classification": "derived_from_measured_main_quality",
                        "model": stratum[0],
                        "task": stratum[1],
                        "metric": stratum[2],
                        "output_length": stratum[3],
                        "analysis": analysis,
                        "method": point["method"],
                        "retention_ratio": point["retention_ratio"],
                        "task_score": quality,
                        "resource_value": cost,
                        "pareto_optimal": not dominators,
                        "dominated_by": json.dumps(
                            sorted(
                                f"{other['method']}@{other['retention_ratio']}"
                                for other in dominators
                            ),
                            separators=(",", ":"),
                        ),
                    }
                )
    return result


def _write_parquet(
    path: Path, rows: Sequence[Mapping[str, object]], metadata: Mapping[str, str]
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite derived artifact: {path}")
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(list(rows))
    schema_metadata = dict(table.schema.metadata or {})
    schema_metadata.update({key.encode(): value.encode() for key, value in metadata.items()})
    table = table.replace_schema_metadata(schema_metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = handle.name
        pq.write_table(table, temporary, compression="zstd")
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
        raise


def _latex_escape(value: object) -> str:
    return str(value).replace("_", r"\_").replace("%", r"\%")


def _quality_table(rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        r"\begin{longtable}{lllrrrr}",
        (
            r"Model & Task (metric) & Method & Ret. & Score [95\% CI] & "
            r"$\Delta$ [95\% CI] & Seq. agree \\"
        ),
        r"\hline",
    ]
    for row in sorted(
        rows,
        key=lambda item: (
            str(item["model_key"]),
            str(item["task"]),
            str(item["method"]),
            -float(cast("float", item["retention_ratio"])),
        ),
    ):
        lines.append(
            f"{_latex_escape(row['model_key'])} & "
            f"{_latex_escape(row['task'])} ({_latex_escape(row['metric'])}) & "
            f"{_latex_escape(row['method'])} & "
            f"{float(cast('float', row['retention_ratio'])):.2f} & "
            f"{float(cast('float', row['task_score'])):.4f} "
            f"[{float(cast('float', row['task_score_ci_low'])):.4f}, "
            f"{float(cast('float', row['task_score_ci_high'])):.4f}] & "
            f"{float(cast('float', row['score_delta'])):+.4f} "
            f"[{float(cast('float', row['score_delta_ci_low'])):+.4f}, "
            f"{float(cast('float', row['score_delta_ci_high'])):+.4f}] & "
            f"{float(cast('float', row['sequence_agreement'])):.4f} \\\\"
        )
    lines.append(r"\end{longtable}")
    return "\n".join(lines) + "\n"


def _memory_table(rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        r"\begin{longtable}{lllrrrrr}",
        r"Model & Task & Method & Ret. & Active MiB & Residual MiB & Ratio & Total latency (s) \\",
        r"\hline",
    ]
    for row in sorted(
        rows,
        key=lambda item: (
            str(item["model_key"]),
            str(item["task"]),
            str(item["method"]),
            -float(cast("float", item["retention_ratio"])),
        ),
    ):
        lines.append(
            f"{_latex_escape(row['model_key'])} & {_latex_escape(row['task'])} & "
            f"{_latex_escape(row['method'])} & "
            f"{float(cast('float', row['retention_ratio'])):.2f} & "
            f"{float(cast('float', row['active_kv_bytes'])) / 2**20:.2f} & "
            f"{float(cast('float', row['residual_kv_bytes'])) / 2**20:.2f} & "
            f"{float(cast('float', row['compression_ratio'])):.4f} & "
            f"{float(cast('float', row['total_latency'])):.4f} \\\\"
        )
    lines.append(r"\end{longtable}")
    return "\n".join(lines) + "\n"


def _plot(
    path: Path,
    summaries: Sequence[Mapping[str, object]],
    *,
    x_field: str,
    title: str,
    x_label: str,
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite derived artifact: {path}")
    from PIL import Image, ImageDraw, ImageFont

    strata = sorted({(str(row["model_key"]), str(row["task"])) for row in summaries})
    columns = 3
    rows_count = (len(strata) + columns - 1) // columns
    panel_width, panel_height = 900, 650
    image = Image.new("RGB", (columns * panel_width, rows_count * panel_height + 100), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((30, 25), title, fill="black", font=font)
    methods = sorted({str(row["method"]) for row in summaries})
    palette = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b")
    colors = {method: palette[index % len(palette)] for index, method in enumerate(methods)}
    for panel_index, (model, task) in enumerate(strata):
        column = panel_index % columns
        panel_row = panel_index // columns
        left = column * panel_width + 60
        top = panel_row * panel_height + 100
        right = (column + 1) * panel_width - 40
        bottom = top + panel_height - 80
        points = [row for row in summaries if row["model_key"] == model and row["task"] == task]
        x_values = [float(cast("float", row[x_field])) for row in points]
        if x_field == "active_kv_bytes":
            x_values = [value / 2**20 for value in x_values]
        y_values = [float(cast("float", row["task_score"])) for row in points]
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)
        if x_min == x_max:
            x_min, x_max = x_min - 0.5, x_max + 0.5
        if y_min == y_max:
            y_min, y_max = y_min - 0.01, y_max + 0.01
        draw.rectangle((left, top, right, bottom), outline="#333333", width=2)
        draw.text((left + 8, top + 8), f"{model} / {task}", fill="black", font=font)
        draw.text((left + 8, bottom + 10), x_label, fill="#333333", font=font)
        for row, x_value, y_value in zip(points, x_values, y_values, strict=True):
            x = left + 30 + (x_value - x_min) / (x_max - x_min) * (right - left - 50)
            y = bottom - 25 - (y_value - y_min) / (y_max - y_min) * (bottom - top - 60)
            color = colors[str(row["method"])]
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".pdf", delete=False
        ) as handle:
            temporary = handle.name
        image.save(temporary, "PDF", resolution=150.0)
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
        raise


def _analysis_markdown(
    summaries: Sequence[Mapping[str, object]],
    pareto: Sequence[Mapping[str, object]],
    *,
    index: Path,
    measurement_git_shas: Sequence[str],
    analysis_sha: str,
) -> str:
    mosaic = [row for row in pareto if str(row["method"]).startswith("mosaickv")]
    optimal = [row for row in mosaic if row["pareto_optimal"]]
    nonoptimal = [row for row in mosaic if not row["pareto_optimal"]]
    block_sizes: dict[str, set[int]] = defaultdict(set)
    for row in summaries:
        block_sizes[str(row["method"])].add(int(cast("int", row["block_size"])))
    block_policy = ", ".join(
        f"{method}={sorted(sizes)}" for method, sizes in sorted(block_sizes.items())
    )
    lines = [
        "# Main quality-versus-memory analysis",
        "",
        "All values are derived from measured, immutable per-example rows. Metrics are kept "
        "separate by model and task; no incompatible benchmark metrics are macro-averaged.",
        "",
        "## Provenance",
        "",
        f"- Campaign index: `{index}`",
        f"- Measurement git SHA(s): `{', '.join(measurement_git_shas)}`",
        f"- Analysis git SHA: `{analysis_sha}`",
        f"- Measured model/task/method operating points: `{len(summaries)}`",
        "",
        "## Pareto audit",
        "",
        f"- MosaicKV Pareto-optimal projections: `{len(optimal)}`",
        f"- MosaicKV non-Pareto projections: `{len(nonoptimal)}`",
        "",
        "### MosaicKV non-Pareto operating points",
        "",
    ]
    if not nonoptimal:
        lines.append("- None.")
    else:
        for row in nonoptimal:
            lines.append(
                f"- `{row['model']}` / `{row['task']}` / `{row['analysis']}` / "
                f"retention `{row['retention_ratio']}` is dominated by {row['dominated_by']}."
            )
    lines.extend(
        [
            "",
            "## Interpretation constraints",
            "",
            "- Score confidence intervals use paired example bootstrap against the matching "
            "FullKV answer on the same pinned sample ledger.",
            "- Pareto flags use measured means and do not convert nominal retention into memory.",
            f"- Cache block sizes are explicit in every Parquet row: {block_policy}.",
            "- Paper-faithful token methods and common block methods retain their native "
            "granularity; Pareto comparisons use realized bytes, and this granularity "
            "difference remains a disclosed design distinction.",
            "- Diagnostic KL and attention-output error use only the fixed sample IDs embedded "
            "in every resolved configuration.",
            "- Results with generalized or safety-fallback effective labels remain explicitly "
            "identified in Parquet and must not be described as the requested algorithm.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--results-directory", type=Path, default=Path("results/quality_memory"))
    parser.add_argument("--tables-directory", type=Path, default=Path("tables"))
    parser.add_argument("--figures-directory", type=Path, default=Path("figures"))
    parser.add_argument("--analysis-path", type=Path, default=Path("analysis/quality_memory.md"))
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=2027)
    args = parser.parse_args()
    if args.bootstrap_iterations < 1000:
        raise ValueError("main analysis requires at least 1000 bootstrap iterations")
    observations, campaign_manifest = _load_observations(
        args.index.resolve(), args.run_root.resolve()
    )
    per_sample = _derive_per_sample(observations)
    summaries = _summarize(
        per_sample,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
    )
    pareto = _pareto(summaries)
    analysis_sha, analysis_dirty = _analysis_source()
    if analysis_dirty:
        raise RuntimeError("main-quality analysis requires a clean worktree")
    measurement_shas = sorted(
        {
            json.loads(Path(observation.manifest_path).read_text(encoding="utf-8"))["source"][
                "git_sha"
            ]
            for observation in observations
        }
    )
    metadata = {
        "classification": "derived_from_measured_main_quality",
        "campaign_index_sha256": str(campaign_manifest["index_sha256"]),
        "measurement_git_shas": json.dumps(measurement_shas, separators=(",", ":")),
        "analysis_git_sha": analysis_sha,
        "bootstrap_iterations": str(args.bootstrap_iterations),
    }
    results = args.results_directory.resolve()
    _write_parquet(results / "per_sample.parquet", per_sample, metadata)
    _write_parquet(results / "summary.parquet", summaries, metadata)
    _write_parquet(results / "pareto.parquet", pareto, metadata)
    _atomic_text(
        args.tables_directory.resolve() / "quality_by_model_task.tex", _quality_table(summaries)
    )
    _atomic_text(
        args.tables_directory.resolve() / "memory_by_model_task.tex", _memory_table(summaries)
    )
    _plot(
        args.figures_directory.resolve() / "quality_memory_pareto.pdf",
        summaries,
        x_field="active_kv_bytes",
        title="Main quality versus measured active KV memory",
        x_label="mean active KV (MiB)",
    )
    _plot(
        args.figures_directory.resolve() / "quality_retention_curves.pdf",
        summaries,
        x_field="retention_ratio",
        title="Main quality versus configured retention",
        x_label="retention ratio (diagnostic axis only)",
    )
    _atomic_text(
        args.analysis_path.resolve(),
        _analysis_markdown(
            summaries,
            pareto,
            index=args.index.resolve(),
            measurement_git_shas=measurement_shas,
            analysis_sha=analysis_sha,
        ),
    )
    print(
        _canonical_line(
            {
                "observations": len(observations),
                "paired_samples": len(per_sample),
                "summary_points": len(summaries),
                "pareto_rows": len(pareto),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
