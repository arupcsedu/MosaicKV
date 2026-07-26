#!/usr/bin/env python3
# ruff: noqa: E501  # Markdown/LaTeX table rows are intentionally kept as literal lines.
"""Merge Experiment-2 trials and generate tables, plots, and an audit narrative."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

GROUP_KEYS = (
    "model",
    "model_revision",
    "backend",
    "attention_implementation",
    "method",
    "retention_ratio",
    "resolution",
    "image_count",
    "visual_token_count",
    "output_length",
    "batch_size",
)
METRICS = (
    "image_encoder_time",
    "projector_time",
    "prefill_time",
    "compression_time",
    "ttft",
    "inter_token_latency",
    "decode_time",
    "end_to_end_time",
    "tokens_per_second",
    "requests_per_second",
    "active_kv_bytes",
    "residual_kv_bytes",
    "peak_gpu_memory_allocated",
    "peak_gpu_memory_reserved",
    "residual_transfer_time",
    "repair_time",
    "repair_frequency",
    "prefix_cache_hit_rate",
    "encoder_cache_hit_rate",
)


def _atomic_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows), temporary, compression="zstd")
    os.replace(temporary, path)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _finite(values: Iterable[Any]) -> list[float]:
    result: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        number = float(value)
        if math.isfinite(number):
            result.append(number)
    return result


def _summary_value(values: list[float], quantile: float) -> float | None:
    return float(np.quantile(values, quantile)) if values else None


def _token_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["model"],
        row["method"],
        row["retention_ratio"],
        row["resolution"],
        row["image_count"],
        row["output_length"],
        row["batch_size"],
        row["trial_index"],
    )


def _apply_parity(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    eager = {
        _token_key(row): row.get("token_ids")
        for row in rows
        if row.get("backend") == "hf_eager"
        and row.get("method") == "full_kv"
        and row.get("status") == "completed"
    }
    verdicts: dict[str, dict[str, Any]] = {}
    for backend in ("hf_sdpa", "hf_flash_attention_2", "vllm", "sglang"):
        candidates = [
            row
            for row in rows
            if row.get("backend") == backend
            and row.get("method") == "full_kv"
            and row.get("status") == "completed"
            and row.get("batch_size") == 1
        ]
        compared = 0
        matched = 0
        for row in candidates:
            reference = eager.get(_token_key(row))
            if reference is None:
                continue
            compared += 1
            matched += row.get("token_ids") == reference
        passed = compared > 0 and compared == matched
        verdicts[backend] = {
            "compared_trials": compared,
            "matched_trials": matched,
            "passed": passed,
        }
    for row in rows:
        backend = row.get("backend")
        if backend == "hf_eager":
            row["backend_parity_verified"] = True
        elif backend in verdicts:
            row["backend_parity_verified"] = verdicts[backend]["passed"]
        else:
            row["backend_parity_verified"] = False
    return verdicts


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in GROUP_KEYS)].append(row)
    summaries: list[dict[str, Any]] = []
    for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        summary = dict(zip(GROUP_KEYS, key, strict=True))
        completed = [row for row in group if row.get("status") == "completed"]
        summary.update(
            trials=len(group),
            completed_trials=len(completed),
            failed_trials=len(group) - len(completed),
            backend_parity_verified=all(
                bool(row.get("backend_parity_verified")) for row in completed
            )
            if completed
            else False,
            paper_eligible=all(not bool(row.get("git_dirty")) for row in completed)
            if completed
            else False,
        )
        for metric in METRICS:
            values = _finite(row.get(metric) for row in completed)
            summary[f"{metric}_p50"] = _summary_value(values, 0.50)
            summary[f"{metric}_p95"] = _summary_value(values, 0.95)
            summary[f"{metric}_mean"] = statistics.fmean(values) if values else None
        summaries.append(summary)
    return summaries


COLORS = (
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
    "#8c564b",
)


def _plot_pdf(
    path: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    series: dict[str, list[tuple[float, float]]],
) -> None:
    width, height = 1400, 900
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    left, right, top, bottom = 120, width - 60, 90, height - 120
    draw.text((left, 25), title, fill="#111111", font=font)
    draw.text((left, height - 42), x_label, fill="#222222", font=font)
    draw.text((10, top), y_label, fill="#222222", font=font)
    all_points = [point for points in series.values() for point in points]
    if all_points:
        x_values = [point[0] for point in all_points]
        y_values = [point[1] for point in all_points]
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)
        if x_min == x_max:
            x_min, x_max = x_min - 0.5, x_max + 0.5
        if y_min == y_max:
            y_min, y_max = y_min - 0.5, y_max + 0.5
        x_pad = (x_max - x_min) * 0.05
        y_pad = (y_max - y_min) * 0.08
        x_min, x_max = x_min - x_pad, x_max + x_pad
        y_min, y_max = max(0.0, y_min - y_pad), y_max + y_pad
        for tick in range(6):
            fraction = tick / 5
            px = left + fraction * (right - left)
            py = bottom - fraction * (bottom - top)
            draw.line((px, top, px, bottom), fill="#eeeeee", width=1)
            draw.line((left, py, right, py), fill="#eeeeee", width=1)
            draw.text(
                (px - 20, bottom + 10),
                f"{x_min + fraction * (x_max - x_min):.3g}",
                fill="#444444",
                font=font,
            )
            draw.text(
                (left - 90, py - 6),
                f"{y_min + fraction * (y_max - y_min):.3g}",
                fill="#444444",
                font=font,
            )
        draw.line((left, bottom, right, bottom), fill="#111111", width=2)
        draw.line((left, top, left, bottom), fill="#111111", width=2)
        legend_y = top
        for index, (label, points) in enumerate(sorted(series.items())):
            color = COLORS[index % len(COLORS)]
            ordered = sorted(points)
            pixels: list[tuple[float, float]] = []
            for x_value, y_value in ordered:
                px = left + (x_value - x_min) / (x_max - x_min) * (right - left)
                py = bottom - (y_value - y_min) / (y_max - y_min) * (bottom - top)
                pixels.append((px, py))
                draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=color)
            if len(pixels) > 1:
                draw.line(pixels, fill=color, width=3)
            draw.rectangle((right - 290, legend_y, right - 275, legend_y + 10), fill=color)
            draw.text((right - 265, legend_y - 2), label[:42], fill="#222222", font=font)
            legend_y += 18
    else:
        draw.text((left, top), "No verified measured rows available", fill="#aa0000", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".pdf", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        image.save(temporary, "PDF", resolution=150.0)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _series(
    summaries: Sequence[dict[str, Any]],
    x_name: str,
    y_name: str,
    *,
    compressed_only: bool = False,
    predicate: Callable[[dict[str, Any]], bool] | None = None,
    label_output_length: bool = False,
) -> dict[str, list[tuple[float, float]]]:
    result: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in summaries:
        if not row["backend_parity_verified"] or row["failed_trials"]:
            continue
        if compressed_only and row["method"] == "full_kv":
            continue
        if predicate is not None and not predicate(row):
            continue
        x_value, y_value = row.get(x_name), row.get(y_name)
        if not isinstance(x_value, (int, float)) or not isinstance(y_value, (int, float)):
            continue
        label = f"{row['model'].split('/')[-1]} | {row['backend']} | {row['method']}"
        if label_output_length:
            label += f" | out={row['output_length']}"
        result[label].append((float(x_value), float(y_value)))
    return dict(result)


def _default_visual_context(row: dict[str, Any]) -> bool:
    expected_resolution = 336 if str(row["model"]).startswith("llava-hf/") else 448
    return (
        row["resolution"] == expected_resolution
        and row["image_count"] == 1
        and row["batch_size"] == 1
    )


def _overhead_regimes(summaries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    references = {
        (
            row["model"],
            row["backend"],
            row["resolution"],
            row["image_count"],
            row["output_length"],
            row["batch_size"],
        ): row
        for row in summaries
        if row["method"] == "full_kv" and row["backend_parity_verified"]
    }
    regimes: list[dict[str, Any]] = []
    for row in summaries:
        if row["method"] == "full_kv" or not row["backend_parity_verified"]:
            continue
        identity = (
            row["model"],
            row["backend"],
            row["resolution"],
            row["image_count"],
            row["output_length"],
            row["batch_size"],
        )
        reference = references.get(identity)
        if reference is None:
            continue
        compression = row.get("compression_time_p50")
        reference_decode = reference.get("decode_time_p50")
        method_decode = row.get("decode_time_p50")
        if not all(
            isinstance(value, (int, float))
            for value in (compression, reference_decode, method_decode)
        ):
            continue
        decode_savings = float(reference_decode) - float(method_decode)
        regimes.append(
            {
                "model": row["model"],
                "backend": row["backend"],
                "method": row["method"],
                "retention_ratio": row["retention_ratio"],
                "output_length": row["output_length"],
                "compression_p50_seconds": compression,
                "decode_savings_p50_seconds": decode_savings,
                "overhead_outweighs_decode_savings": float(compression) > max(0.0, decode_savings),
                "end_to_end_delta_seconds": (
                    float(row["end_to_end_time_p50"]) - float(reference["end_to_end_time_p50"])
                ),
            }
        )
    return regimes


def _format(value: Any, digits: int = 3) -> str:
    return "--" if value is None else f"{float(value):.{digits}f}"


def _latex(summaries: Sequence[dict[str, Any]]) -> str:
    lines = [
        r"\begin{tabular}{lllrrrrrrrrrr}",
        r"\toprule",
        r"Model & Backend & Method & $r$ & Px & Visual & Images & Out & Batch & TTFT$_{50}$ & TTFT$_{95}$ & Decode$_{50}$ & KV MiB \\",
        r"\midrule",
    ]
    for row in summaries:
        if not row["backend_parity_verified"] or row["failed_trials"]:
            continue
        model = row["model"].split("/")[-1].replace("_", r"\_")
        backend = row["backend"].replace("_", r"\_")
        method = row["method"].replace("_", r"\_")
        kv = row.get("active_kv_bytes_p50")
        kv_mib = None if kv is None else float(kv) / 2**20
        lines.append(
            " & ".join(
                (
                    model,
                    backend,
                    method,
                    str(row["retention_ratio"]),
                    str(row["resolution"]),
                    str(row["visual_token_count"]),
                    str(row["image_count"]),
                    str(row["output_length"]),
                    str(row["batch_size"]),
                    _format(row.get("ttft_p50")),
                    _format(row.get("ttft_p95")),
                    _format(row.get("decode_time_p50")),
                    _format(kv_mib, 1),
                )
            )
            + r" \\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}", ""))
    return "\n".join(lines)


def _analysis(
    rows: Sequence[dict[str, Any]],
    summaries: Sequence[dict[str, Any]],
    parity: dict[str, dict[str, Any]],
    regimes: Sequence[dict[str, Any]],
    manifest_patch_shas: Sequence[str],
) -> str:
    completed = sum(row.get("status") == "completed" for row in rows)
    failed = len(rows) - completed
    paper_eligible = all(not bool(row.get("git_dirty")) for row in rows)
    lines = [
        "# Experiment 2: systems microbenchmark validation",
        "",
        "## Scope and eligibility",
        "",
        f"The bounded campaign contains {len(rows)} timed trial rows: {completed} completed and {failed} failed. "
        "Every point used 5 warmups and 20 timed repetitions.",
        "",
        (
            "The source tree was clean, so these rows are eligible for a paper campaign."
            if paper_eligible
            else "The source tree contained the explicitly staged implementation patch. These are validation rows, not paper-eligible final measurements; every manifest records the Git SHA and patch SHA."
        ),
        "",
        f"The immutable point manifests contain {len(manifest_patch_shas)} worktree fingerprint(s): "
        + ", ".join(f"`{value}`" for value in manifest_patch_shas)
        + ". Results with different fingerprints remain validation-only and are not pooled as a clean paper campaign.",
        "",
        "All reported p50/p95 values are derived from per-trial measurements. Missing backend telemetry remains missing; it was not synthesized.",
        "",
        "## Backend parity gates",
        "",
        "| Backend | Compared trials | Matching trials | Verdict |",
        "|---|---:|---:|---|",
    ]
    for backend, verdict in parity.items():
        lines.append(
            f"| {backend} | {verdict['compared_trials']} | {verdict['matched_trials']} | "
            f"{'passed' if verdict['passed'] else 'failed / excluded'} |"
        )
    lines.extend(
        (
            "",
            "A backend contributes tables and figures only when its completed FullKV token IDs match the paired HF-eager trials. A missing or failed gate is excluded, never replaced by a simulated row. Native vLLM/SGLang MosaicKV remains unsupported.",
            "",
            "## Supported coverage",
            "",
            "HF eager measures FullKV and exact-only MosaicKV for Qwen-3B/7B. LLaVA uses block size 64 for every row and a bounded pair of compressed points to control its 32-head descriptor cost. HF SDPA and FlashAttention-2, vLLM, and SGLang contribute FullKV rows only after parity. Prototype merging and repair are absent because all audited production adapters expose post-RoPE cached keys and no safe phase-alignment transform has passed correctness.",
            "",
            "HF adapters are batch-size one; no sequential proxy is labeled as batching. Serving-backend batch rows are concurrent engine requests.",
            "",
            "## Compression-overhead regimes",
            "",
            "Compression overhead is compared with the paired FullKV p50 decode-time saving at the same model, backend, resolution, image count, output length, and batch size.",
            "",
            "| Model | Method | Retention | Output | Compression p50 (s) | Decode saving p50 (s) | E2E delta (s) | Overhead outweighs saving |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        )
    )
    for regime in regimes:
        lines.append(
            f"| {regime['model'].split('/')[-1]} | {regime['method']} | {regime['retention_ratio']} | "
            f"{regime['output_length']} | {_format(regime['compression_p50_seconds'])} | "
            f"{_format(regime['decode_savings_p50_seconds'])} | {_format(regime['end_to_end_delta_seconds'])} | "
            f"{'yes' if regime['overhead_outweighs_decode_savings'] else 'no'} |"
        )
    if not regimes:
        lines.append("| -- | -- | -- | -- | -- | -- | -- | no paired verified rows |")
    decode_slowdowns = [
        -float(regime["decode_savings_p50_seconds"])
        for regime in regimes
        if float(regime["decode_savings_p50_seconds"]) < 0
    ]
    lines.extend(
        (
            "",
            "## Decode-latency interpretation",
            "",
            (
                f"{len(decode_slowdowns)} of {len(regimes)} paired compressed regimes decoded "
                "more slowly than FullKV. "
                f"The p50 slowdown ranged from {min(decode_slowdowns):.3f} to "
                f"{max(decode_slowdowns):.3f} seconds per sequence."
                if decode_slowdowns
                else "No paired compressed regime had a measurable decode slowdown."
            ),
            "",
            "The corrected runtime no longer clones the full KV cache before every non-repair token. Dense physical budgets are quantized to exactly fill every layer/head capacity, which removes per-head validity masks for the common exact-only path. Remaining costs include eager DynamicCache growth, per-token synchronized measurement boundaries, and attention kernels that may be compute- rather than memory-bound at short context. Compression overhead is reported separately and can still outweigh decode savings.",
        )
    )
    lines.extend(
        (
            "",
            "## Unsupported combinations",
            "",
            "| Combination | Status | Reason |",
            "|---|---|---|",
            "| HF SDPA/FlashAttention-2 compression | unsupported | Prompt-attention capture and arbitrary packed per-head masks remain eager-only. |",
            "| SGLang/vLLM compressed methods | unsupported | No safe native request-scoped KV mutation API exists in the audited versions. |",
            "| vLLM Qwen-7B / LLaVA-7B | unsupported in this bounded run | No matching current-lock parity pass was available before scheduling. |",
            "| Native MosaicKV on vLLM/SGLang | unsupported | No safe request-scoped KV mutation API exists in the audited versions. |",
            "| `mosaickv_proto` / `mosaickv_full` | unsupported | Safe post-RoPE prototype handling has not passed correctness. |",
            "| HF batch 2/4 | unsupported | Explicit HF adapters enforce batch size one. |",
            "",
            "## Interpretation limits",
            "",
            "This is a controlled synthetic visual workload intended to validate systems instrumentation and expose trends within 2.5 hours. It is not a quality benchmark and does not establish paper claims. Only rows passing the current-lock startup and token-parity gates are shown.",
            "",
        )
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    campaign_roots = tuple(Path(value).resolve() for value in args.campaign_root)
    raw_paths = sorted(
        {
            path
            for campaign_root in campaign_roots
            for path in campaign_root.glob("*/points/*/trials.jsonl")
        }
    )
    if not raw_paths:
        joined = ", ".join(str(path) for path in campaign_roots)
        raise FileNotFoundError(f"no point trial files under: {joined}")
    rows: list[dict[str, Any]] = []
    manifest_patch_shas: set[str] = set()
    for path in raw_paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
        manifest_patch_shas.add(str(manifest["source"]["patch_sha"]))
    parity = _apply_parity(rows)
    summaries = summarize(rows)
    regimes = _overhead_regimes(summaries)
    package_root = Path(args.package_root).resolve()
    results = package_root / "results" / "performance"
    figures = package_root / "figures"
    tables = package_root / "tables"
    analysis = package_root / "analysis"
    artifact_paths = (
        results / "trials.parquet",
        results / "summary.parquet",
        figures / "ttft_vs_visual_tokens.pdf",
        figures / "decode_latency_vs_memory.pdf",
        figures / "throughput_vs_retention.pdf",
        figures / "compression_overhead.pdf",
        tables / "performance.tex",
        analysis / "performance.md",
    )
    for path in artifact_paths:
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    _atomic_parquet(results / "trials.parquet", rows)
    _atomic_parquet(results / "summary.parquet", summaries)
    _plot_pdf(
        figures / "ttft_vs_visual_tokens.pdf",
        title="TTFT versus expanded visual-token count (p50)",
        x_label="Expanded visual tokens",
        y_label="TTFT p50 (seconds)",
        series=_series(
            summaries,
            "visual_token_count",
            "ttft_p50",
            predicate=lambda row: (
                row["method"] == "full_kv" and row["output_length"] == 64 and row["batch_size"] == 1
            ),
        ),
    )
    _plot_pdf(
        figures / "decode_latency_vs_memory.pdf",
        title="Decode latency versus active KV payload (p50)",
        x_label="Active KV bytes",
        y_label="Decode p50 (seconds)",
        series=_series(
            summaries,
            "active_kv_bytes_p50",
            "decode_time_p50",
            predicate=_default_visual_context,
            label_output_length=True,
        ),
    )
    _plot_pdf(
        figures / "throughput_vs_retention.pdf",
        title="Throughput versus retention ratio (p50)",
        x_label="Retention ratio",
        y_label="Tokens per second p50",
        series=_series(
            summaries,
            "retention_ratio",
            "tokens_per_second_p50",
            predicate=_default_visual_context,
            label_output_length=True,
        ),
    )
    _plot_pdf(
        figures / "compression_overhead.pdf",
        title="Compression overhead versus retention ratio (p50)",
        x_label="Retention ratio",
        y_label="Compression time p50 (seconds)",
        series=_series(
            summaries,
            "retention_ratio",
            "compression_time_p50",
            compressed_only=True,
            predicate=_default_visual_context,
            label_output_length=True,
        ),
    )
    _atomic_text(tables / "performance.tex", _latex(summaries))
    _atomic_text(
        analysis / "performance.md",
        _analysis(rows, summaries, parity, regimes, sorted(manifest_patch_shas)),
    )
    print(
        json.dumps(
            {
                "trial_rows": len(rows),
                "summary_rows": len(summaries),
                "failed_trials": sum(row.get("status") != "completed" for row in rows),
                "parity": parity,
                "overhead_regimes": len(regimes),
            },
            sort_keys=True,
        )
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--campaign-root",
        required=True,
        action="append",
        help="raw campaign directory; repeat to combine independently scheduled backends",
    )
    parser.add_argument("--package-root", required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="atomically replace derived artifacts generated from the same immutable raw trials",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
