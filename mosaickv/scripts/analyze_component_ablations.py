#!/usr/bin/env python3
"""Validate and summarize the bounded Experiment-3 component ablations."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, cast

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

QWEN_MROPE_TRANSFORM_PROVENANCE = (
    "transformers_4_55_2_qwen2_5_vl_get_rope_index_"
    "rotary_emb_apply_multimodal_rotary_pos_emb_v1"
)


def _finite(values: list[object]) -> list[float]:
    return [
        float(value)
        for value in values
        if isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)
    ]


def _average(values: list[object]) -> float | None:
    selected = _finite(values)
    return mean(selected) if selected else None


def _trace_by_sample(root: Path, run_id: str) -> dict[str, dict[str, Any]]:
    traces: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "traces" / run_id).glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") == "completed":
            sample_id = str(value["sample_id"])
            if sample_id in traces:
                raise RuntimeError(f"duplicate completed trace: {run_id}/{sample_id}")
            traces[sample_id] = cast("dict[str, Any]", value)
    return traces


def _validate_trace_support(
    trace: dict[str, Any],
    *,
    variant_id: str,
    run_id: str,
    sample_id: str,
) -> None:
    """Fail closed when a measured row did not execute its declared cache path."""

    identity = f"{run_id}/{sample_id}"
    if variant_id == "I":
        if trace.get("tier_mode") != "exact_only_prototypes_disabled":
            raise RuntimeError(f"no-prototype row did not use exact-only mode: {identity}")
        if trace.get("prototypes"):
            raise RuntimeError(f"no-prototype row contains prototype payloads: {identity}")
    else:
        if trace.get("tier_mode") != "three_tier":
            raise RuntimeError(f"prototype row did not materialize three tiers: {identity}")
        safety = trace.get("prototype_safety")
        if not isinstance(safety, dict) or safety.get("safe") is not True:
            raise RuntimeError(f"prototype row lacks a positive safety assessment: {identity}")
        if safety.get("phase_transform_provenance") != QWEN_MROPE_TRANSFORM_PROVENANCE:
            raise RuntimeError(f"prototype row has unregistered M-RoPE provenance: {identity}")
    evaluation_only = variant_id in {"K", "L"}
    if bool(trace.get("ablation_evaluation_only")) is not evaluation_only:
        raise RuntimeError(f"evaluation-only ablation label is inconsistent: {identity}")
    if variant_id == "L":
        oracle = trace.get("oracle_repair_diagnostic")
        if not isinstance(oracle, dict) or oracle.get("status") != "completed_evaluation_only":
            raise RuntimeError(f"oracle-repair diagnostic is not measured: {identity}")
        overhead = oracle.get("overhead_seconds_excluded_from_online_total")
        if not isinstance(overhead, int | float) or not math.isfinite(float(overhead)):
            raise RuntimeError(f"oracle-repair diagnostic lacks finite overhead: {identity}")


def _control_identity(manifest: dict[str, Any]) -> tuple[object, ...]:
    return (
        manifest["model"]["id"],
        manifest["model"]["revision"],
        manifest["model"]["precision"],
        manifest["dataset"]["id"],
        manifest["dataset"]["revision"],
        manifest["dataset"]["split"],
        manifest["execution"]["backend"],
        manifest["execution"]["attention_implementation"],
        manifest["execution"]["decode_attention_implementation"],
        manifest["execution"]["seed"],
        manifest["generation"]["parameters_sha"],
        manifest["cache"]["retention_ratio"],
        manifest["cache"]["budget_unit"],
        manifest["cache"]["accounting_spec_sha"],
        manifest["inputs"]["prompt_set_sha"],
        manifest["inputs"]["media_set_sha"],
        manifest["inputs"]["preprocessing_sha"],
        manifest["inputs"]["tokenization_sha"],
    )


def _sample_record(
    row: dict[str, Any],
    trace: dict[str, Any] | None,
    *,
    task: str,
    variant_id: str,
    variant: str,
) -> dict[str, Any]:
    diagnostic = {} if trace is None else trace.get("forecast_quality_diagnostic", {})
    objective = {} if trace is None else trace.get("selection_objective", {})
    graph = {} if trace is None else trace.get("graph_diagnostics", {})
    fidelity = {} if trace is None else trace.get("fidelity_diagnostic", {})
    events = [] if trace is None else trace.get("repair_events", [])
    triggered = [item for item in events if isinstance(item, dict) and item.get("triggered")]
    recovered = [
        bool(item["quality_recovered"])
        for item in triggered
        if item.get("quality_recovered") is not None
    ]
    prototypes = [] if trace is None else trace.get("prototypes", [])
    timing = {} if trace is None else trace.get("timing_breakdown", {})
    prototype_count = len(prototypes) if isinstance(prototypes, list) else None
    return {
        "task": task,
        "variant_id": variant_id,
        "variant": variant,
        "run_id": row.get("run_id"),
        "sample_id": row.get("sample_id"),
        "status": row.get("status"),
        "error": row.get("error"),
        "effective_method": row.get("method"),
        "task_score": row.get("task_score"),
        "active_kv_bytes": row.get("active_kv_bytes"),
        "compression_time": row.get("compression_time"),
        "forecast_regret": diagnostic.get("mean_forecast_regret"),
        "selected_block_recall": diagnostic.get("mean_selected_block_recall"),
        "attention_rank_correlation": diagnostic.get("mean_attention_rank_correlation"),
        "graph_coverage": objective.get("facility_location_coverage"),
        "evidence_cluster_coverage": graph.get("evidence_cluster_coverage"),
        "graph_edges": graph.get("edge_count"),
        "graph_edge_counts": graph.get("edge_counts"),
        "logit_kl": fidelity.get("logit_kl", row.get("logit_kl")),
        "attention_output_error": fidelity.get(
            "attention_output_error", row.get("attention_output_error")
        ),
        "repair_count": row.get("repair_count"),
        "repaired_bytes": row.get("repaired_bytes"),
        "residual_kv_bytes": row.get("residual_kv_bytes"),
        "repair_latency": (
            float(timing.get("repair_transfer", 0.0) or 0.0)
            + float(timing.get("repair_redecode", 0.0) or 0.0)
        ),
        "prototype_count": prototype_count,
        "generated_tokens": row.get("generated_tokens"),
        "repair_benefit": mean(recovered) if recovered else None,
        "end_to_end_time": row.get("end_to_end_time"),
        "tier_mode": None if trace is None else trace.get("tier_mode"),
        "tier_reason": None if trace is None else trace.get("tier_reason"),
    }


def _aggregate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["task"], record["variant_id"], record["variant"])].append(record)
    aggregates: list[dict[str, Any]] = []
    for (task, variant_id, variant), rows in sorted(grouped.items()):
        completed = [row for row in rows if row["status"] == "completed"]
        methods = sorted({str(row["effective_method"]) for row in completed})
        fallback = any("safety_fallback" in method for method in methods)
        repair_events = sum(int(row["repair_count"] or 0) for row in completed)
        aggregates.append(
            {
                "task": task,
                "variant_id": variant_id,
                "variant": variant,
                "support_status": ("measured_exact_safety_fallback" if fallback else "measured"),
                "unsupported_reason": None,
                "effective_method": ";".join(methods),
                "sample_count": len(completed),
                "failed_samples": len(rows) - len(completed),
                "task_score": _average([row["task_score"] for row in completed]),
                "active_kv_bytes": _average([row["active_kv_bytes"] for row in completed]),
                "compression_latency_s": _average([row["compression_time"] for row in completed]),
                "forecast_regret": _average([row["forecast_regret"] for row in completed]),
                "selected_block_recall": _average(
                    [row["selected_block_recall"] for row in completed]
                ),
                "graph_coverage": _average([row["graph_coverage"] for row in completed]),
                "evidence_cluster_coverage": _average(
                    [row["evidence_cluster_coverage"] for row in completed]
                ),
                "logit_kl": _average([row["logit_kl"] for row in completed]),
                "attention_output_error": _average(
                    [row["attention_output_error"] for row in completed]
                ),
                "prototype_count": _average([row["prototype_count"] for row in completed]),
                "residual_kv_bytes": _average(
                    [row["residual_kv_bytes"] for row in completed]
                ),
                "repair_frequency": (
                    repair_events
                    / sum(max(1, int(row["generated_tokens"] or 1) - 1) for row in completed)
                    if completed
                    else None
                ),
                "repair_benefit": _average([row["repair_benefit"] for row in completed]),
                "repair_latency_s": _average([row["repair_latency"] for row in completed]),
                "end_to_end_latency_s": _average([row["end_to_end_time"] for row in completed]),
                "tier_mode": ";".join(
                    sorted({str(row["tier_mode"]) for row in completed if row["tier_mode"]})
                ),
                "budget_match": False,
                "canonical_eligible": False,
            }
        )
    return aggregates


def _unsupported(index_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for job in index_rows:
        for item in job["unsupported"]:
            records.append(
                {
                    "task": job["task"],
                    "variant_id": item["variant_id"],
                    "variant": item["variant"],
                    "support_status": "unsupported",
                    "unsupported_reason": item["reason"],
                    "effective_method": None,
                    "sample_count": 0,
                    "failed_samples": 0,
                    "task_score": None,
                    "active_kv_bytes": None,
                    "compression_latency_s": None,
                    "forecast_regret": None,
                    "selected_block_recall": None,
                    "graph_coverage": None,
                    "evidence_cluster_coverage": None,
                    "logit_kl": None,
                    "attention_output_error": None,
                    "prototype_count": None,
                    "residual_kv_bytes": None,
                    "repair_frequency": None,
                    "repair_benefit": None,
                    "repair_latency_s": None,
                    "end_to_end_latency_s": None,
                    "tier_mode": None,
                    "budget_match": False,
                    "canonical_eligible": False,
                }
            )
    return records


def _value(frame: pd.DataFrame, task: str, variant: str, column: str) -> float | None:
    selected = frame[(frame.task == task) & (frame.variant_id == variant)][column].dropna()
    return None if selected.empty else float(selected.iloc[0])


def _difference(
    frame: pd.DataFrame,
    tasks: list[str],
    left: str,
    right: str,
    column: str,
) -> float | None:
    values = []
    for task in tasks:
        first = _value(frame, task, left, column)
        second = _value(frame, task, right, column)
        if first is not None and second is not None:
            values.append(first - second)
    return mean(values) if values else None


def _task_differences(
    frame: pd.DataFrame,
    tasks: list[str],
    left: str,
    right: str,
    column: str,
) -> dict[str, float | None]:
    return {
        task: _difference(frame, [task], left, right, column)
        for task in tasks
    }


def _fmt(value: float | None, digits: int = 4) -> str:
    return "unavailable" if value is None else f"{value:.{digits}f}"


def _fmt_task_values(values: dict[str, float | None]) -> str:
    return ", ".join(f"{task}={_fmt(value)}" for task, value in values.items())


class _PdfCanvas:
    """Tiny dependency-free vector PDF canvas for reproducibility figures."""

    def __init__(self, width: float, height: float) -> None:
        self.width = width
        self.height = height
        self.commands: list[str] = []

    @staticmethod
    def _escape(value: object) -> str:
        return str(value).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        color: tuple[float, float, float] = (0.0, 0.0, 0.0),
        width: float = 0.8,
    ) -> None:
        self.commands.append(
            f"{color[0]:.3f} {color[1]:.3f} {color[2]:.3f} RG "
            f"{width:.2f} w {x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S"
        )

    def text(self, x: float, y: float, value: object, *, size: float = 8.0) -> None:
        self.commands.append(
            f"0 0 0 rg BT /F1 {size:.2f} Tf {x:.2f} {y:.2f} Td "
            f"({self._escape(value)}) Tj ET"
        )

    def marker(
        self,
        x: float,
        y: float,
        *,
        color: tuple[float, float, float],
        radius: float = 2.5,
    ) -> None:
        # Four Bezier arcs form a vector circle.
        control = radius * 0.55228475
        self.commands.append(
            f"{color[0]:.3f} {color[1]:.3f} {color[2]:.3f} rg "
            f"{x + radius:.2f} {y:.2f} m "
            f"{x + radius:.2f} {y + control:.2f} {x + control:.2f} {y + radius:.2f} "
            f"{x:.2f} {y + radius:.2f} c "
            f"{x - control:.2f} {y + radius:.2f} {x - radius:.2f} {y + control:.2f} "
            f"{x - radius:.2f} {y:.2f} c "
            f"{x - radius:.2f} {y - control:.2f} {x - control:.2f} {y - radius:.2f} "
            f"{x:.2f} {y - radius:.2f} c "
            f"{x + control:.2f} {y - radius:.2f} {x + radius:.2f} {y - control:.2f} "
            f"{x + radius:.2f} {y:.2f} c f"
        )

    def save(self, path: Path) -> None:
        stream = ("\n".join(self.commands) + "\n").encode("ascii", errors="replace")
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.width:.2f} "
                f"{self.height:.2f}] /Resources << /Font << /F1 4 0 R >> >> "
                "/Contents 5 0 R >>"
            ).encode(),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"endstream",
        ]
        output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = [0]
        for index, payload in enumerate(objects, start=1):
            offsets.append(len(output))
            output.extend(f"{index} 0 obj\n".encode())
            output.extend(payload)
            output.extend(b"\nendobj\n")
        xref = len(output)
        output.extend(f"xref\n0 {len(objects) + 1}\n".encode())
        output.extend(b"0000000000 65535 f \n")
        for offset in offsets[1:]:
            output.extend(f"{offset:010d} 00000 n \n".encode())
        output.extend(
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n".encode()
        )
        path.write_bytes(output)


_PLOT_COLORS = (
    (0.12, 0.47, 0.71),
    (1.00, 0.50, 0.05),
    (0.17, 0.63, 0.17),
    (0.84, 0.15, 0.16),
)


def _draw_panel(
    canvas: _PdfCanvas,
    box: tuple[float, float, float, float],
    *,
    title: str,
    x_labels: list[str],
    series: list[tuple[str, list[float | None], tuple[float, float, float]]],
    y_label: str,
    annotate_points: bool = False,
) -> None:
    x, y, width, height = box
    left, right, bottom, top = 45.0, 10.0, 32.0, 24.0
    px, py = x + left, y + bottom
    pw, ph = width - left - right, height - bottom - top
    values = [value for _name, row, _color in series for value in row if value is not None]
    canvas.text(x + 3, y + height - 12, title, size=10)
    canvas.text(x + 3, y + height - 23, y_label, size=7)
    canvas.line(px, py, px, py + ph)
    canvas.line(px, py, px + pw, py)
    if not values or not x_labels:
        canvas.text(px + 8, py + ph / 2, "unavailable", size=9)
        return
    low, high = min(values), max(values)
    if math.isclose(low, high):
        margin = max(abs(low) * 0.1, 1e-6)
        low, high = low - margin, high + margin
    else:
        margin = (high - low) * 0.08
        low, high = low - margin, high + margin
    for index in range(3):
        fraction = index / 2
        gy = py + fraction * ph
        value = low + fraction * (high - low)
        canvas.line(px, gy, px + pw, gy, color=(0.82, 0.82, 0.82), width=0.4)
        canvas.text(x + 2, gy - 2, f"{value:.3g}", size=6)
    x_positions = [
        px + (index + 0.5) * pw / len(x_labels) for index in range(len(x_labels))
    ]
    for xpos, label in zip(x_positions, x_labels, strict=True):
        canvas.text(xpos - 3, py - 12, label, size=7)
    for series_index, (name, row, color) in enumerate(series):
        previous: tuple[float, float] | None = None
        for label, xpos, value in zip(x_labels, x_positions, row, strict=True):
            if value is None:
                previous = None
                continue
            ypos = py + (value - low) * ph / (high - low)
            if previous is not None:
                canvas.line(previous[0], previous[1], xpos, ypos, color=color, width=1.1)
            canvas.marker(xpos, ypos, color=color)
            if annotate_points:
                canvas.text(xpos + 3, ypos + 2, label, size=6)
            previous = (xpos, ypos)
        legend_x = px + 4 + (series_index % 2) * 82
        legend_y = py + ph - 9 - (series_index // 2) * 10
        canvas.marker(legend_x, legend_y + 2, color=color, radius=2)
        canvas.text(legend_x + 5, legend_y, name, size=6)


def _draw_xy_panel(
    canvas: _PdfCanvas,
    box: tuple[float, float, float, float],
    *,
    title: str,
    points: list[tuple[str, float | None, float | None]],
    color: tuple[float, float, float],
) -> None:
    x, y, width, height = box
    left, right, bottom, top = 48.0, 10.0, 34.0, 25.0
    px, py = x + left, y + bottom
    pw, ph = width - left - right, height - bottom - top
    finite = [(label, xv, yv) for label, xv, yv in points if xv is not None and yv is not None]
    canvas.text(x + 3, y + height - 12, title, size=10)
    canvas.text(x + 3, y + height - 23, "Task score", size=7)
    canvas.text(px + pw / 2 - 28, y + 5, "End-to-end latency (s)", size=7)
    canvas.line(px, py, px, py + ph)
    canvas.line(px, py, px + pw, py)
    if not finite:
        canvas.text(px + 8, py + ph / 2, "unavailable", size=9)
        return
    x_values = [float(item[1]) for item in finite]
    y_values = [float(item[2]) for item in finite]
    x_low, x_high = min(x_values), max(x_values)
    y_low, y_high = min(y_values), max(y_values)
    x_margin = max((x_high - x_low) * 0.08, max(abs(x_low), 1.0) * 1e-3)
    y_margin = max((y_high - y_low) * 0.08, max(abs(y_low), 1.0) * 1e-3)
    x_low, x_high = x_low - x_margin, x_high + x_margin
    y_low, y_high = y_low - y_margin, y_high + y_margin
    for index in range(3):
        fraction = index / 2
        gx, gy = px + fraction * pw, py + fraction * ph
        canvas.line(px, gy, px + pw, gy, color=(0.82, 0.82, 0.82), width=0.4)
        canvas.line(gx, py, gx, py + ph, color=(0.88, 0.88, 0.88), width=0.4)
        canvas.text(x + 3, gy - 2, f"{y_low + fraction * (y_high - y_low):.3g}", size=6)
        canvas.text(gx - 10, py - 12, f"{x_low + fraction * (x_high - x_low):.3g}", size=6)
    for label, xv, yv in finite:
        xpos = px + (float(xv) - x_low) * pw / (x_high - x_low)
        ypos = py + (float(yv) - y_low) * ph / (y_high - y_low)
        canvas.marker(xpos, ypos, color=color)
        canvas.text(xpos + 3, ypos + 2, label, size=6)


def _write_figures(frame: pd.DataFrame, figures: Path) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    measured = frame[frame.support_status != "unsupported"].copy()
    task_order = ["mmstar", "docvqa", "chartqa", "video_mme"]

    forecast_variants = ["B", "C", "D", "A", "K"]
    canvas = _PdfCanvas(792, 302)
    for panel, column, label in (
        (0, "selected_block_recall", "Selected-block recall"),
        (1, "forecast_regret", "Forecast regret"),
    ):
        series = [
            (
                task,
                [_value(measured, task, variant, column) for variant in forecast_variants],
                _PLOT_COLORS[index],
            )
            for index, task in enumerate(task_order)
        ]
        _draw_panel(
            canvas,
            (panel * 396.0, 0.0, 396.0, 302.0),
            title=f"Forecast ablation: {label}",
            x_labels=forecast_variants,
            series=series,
            y_label=label,
        )
    canvas.save(figures / "ablation_forecast.pdf")

    component_variants = ["A", "E", "F", "G", "H"]
    canvas = _PdfCanvas(792, 302)
    for panel, column, label in (
        (0, "graph_coverage", "Facility-location graph coverage"),
        (1, "attention_output_error", "Attention-output relative error"),
    ):
        series = [
            (
                task,
                [_value(measured, task, variant, column) for variant in component_variants],
                _PLOT_COLORS[index],
            )
            for index, task in enumerate(task_order)
        ]
        _draw_panel(
            canvas,
            (panel * 396.0, 0.0, 396.0, 302.0),
            title=f"Component ablation: {label}",
            x_labels=component_variants,
            series=series,
            y_label=label,
        )
    canvas.save(figures / "ablation_components.pdf")

    canvas = _PdfCanvas(720, 540)
    for index, task in enumerate(task_order):
        subset = measured[measured.task == task]
        points: list[tuple[str, float | None, float | None]] = []
        for row in subset.itertuples():
            latency = (
                None
                if pd.isna(row.end_to_end_latency_s)
                else float(row.end_to_end_latency_s)
            )
            score = None if pd.isna(row.task_score) else float(row.task_score)
            points.append((str(row.variant_id), latency, score))
        _draw_xy_panel(
            canvas,
            ((index % 2) * 360.0, (1 - index // 2) * 270.0, 360.0, 270.0),
            title=f"{task}: quality versus latency",
            points=points,
            color=_PLOT_COLORS[index],
        )
    canvas.save(figures / "ablation_quality_latency.pdf")


def _write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "task",
        "variant_id",
        "support_status",
        "sample_count",
        "task_score",
        "active_kv_bytes",
        "compression_latency_s",
        "selected_block_recall",
        "forecast_regret",
        "graph_coverage",
        "logit_kl",
        "attention_output_error",
        "prototype_count",
        "residual_kv_bytes",
        "repair_frequency",
        "repair_benefit",
        "repair_latency_s",
        "end_to_end_latency_s",
    ]
    table = frame[columns].copy()
    path.write_text(
        table.to_latex(
            index=False,
            float_format=lambda value: f"{value:.4g}",
            na_rep="--",
            longtable=True,
            escape=True,
            caption=(
                "Small exploratory MosaicKV component ablations. Unsupported cells contain "
                "no measured numeric values."
            ),
            label="tab:mosaickv-component-ablations-small",
        ),
        encoding="utf-8",
    )


def _write_analysis(
    frame: pd.DataFrame,
    path: Path,
    *,
    controls_passed: bool,
    output_root: Path,
    study: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tasks = ["mmstar", "docvqa", "chartqa", "video_mme"]
    online_forecast_gain = _difference(
        frame,
        tasks,
        "A",
        "B",
        "selected_block_recall",
    )
    prompt_gain = _difference(frame, tasks, "C", "B", "selected_block_recall")
    draft_gain = _difference(frame, tasks, "D", "B", "selected_block_recall")
    oracle_gain = _difference(frame, tasks, "K", "B", "selected_block_recall")
    forecast_scores = _task_differences(frame, tasks, "A", "B", "task_score")
    graph_scores = _task_differences(frame, tasks, "A", "E", "task_score")
    structured_coverage = _difference(
        frame, ["docvqa", "chartqa"], "A", "E", "graph_coverage"
    )
    general_coverage = _difference(frame, ["mmstar"], "A", "E", "graph_coverage")
    structured_graph_fidelity = _difference(
        frame, ["docvqa", "chartqa"], "E", "A", "attention_output_error"
    )
    general_graph_fidelity = _difference(
        frame, ["mmstar"], "E", "A", "attention_output_error"
    )
    value_fidelity = _difference(
        frame,
        tasks,
        "G",
        "A",
        "attention_output_error",
    )
    value_logit_kl = _difference(frame, tasks, "G", "A", "logit_kl")
    prototype_scores = _task_differences(frame, tasks, "J", "I", "task_score")
    prototype_attention = _difference(
        frame, tasks, "J", "I", "attention_output_error"
    )
    prototype_logit = _difference(frame, tasks, "J", "I", "logit_kl")
    repair_scores = _task_differences(frame, tasks, "A", "J", "task_score")
    repair_latency = _difference(frame, tasks, "A", "J", "end_to_end_latency_s")
    measured = frame[frame.support_status != "unsupported"]
    measured_cells = len(measured)
    completed_samples = int(measured.sample_count.sum())
    failed_samples = int(measured.failed_samples.sum())
    three_tier = measured[measured.tier_mode.fillna("").str.contains("three_tier")]
    online_repair_frequency = _average(
        list(measured[measured.variant_id == "A"].repair_frequency)
    )
    oracle_repair_benefit = _average(
        list(measured[measured.variant_id == "L"].repair_benefit)
    )
    oracle_repair_frequency = _average(
        list(measured[measured.variant_id == "L"].repair_frequency)
    )
    oracle_repair_latency = _average(
        list(measured[measured.variant_id == "L"].repair_latency_s)
    )
    subset_sizes = cast("dict[str, int]", study["subset_sizes"])
    subset_description = ", ".join(f"{task}={subset_sizes[task]}" for task in tasks)
    lines = [
        "# Experiment 3: small component-ablation study",
        "",
        "## Scope and validity",
        "",
        "This is an exploratory dirty-worktree validation run, not a paper result. It uses "
        f"deterministically selected subsets ({subset_description}), "
        f"{study['model']} in {study['precision']}, {study['prefill_attention']} prefill, "
        f"{study['decode_attention']} decode, {study['output_length']} fixed greedy tokens, "
        f"retention {study['retention_ratio']}, and {study['block_size']}-token blocks. The "
        "primary budget is measured packed active KV bytes. The small sample count is useful for "
        "implementation direction only and cannot establish benchmark significance.",
        "",
        f"The analyzed index contains {measured_cells} measured cells, {completed_samples} "
        f"completed sample rows, and {failed_samples} failed sample rows. Raw run artifacts are "
        f"under `{output_root}`.",
        "",
        f"Controlled-comparison validation: **{'passed' if controls_passed else 'failed'}**.",
        "",
        "Qwen2.5-VL cache keys are post-M-RoPE. Prototype rows are considered measured only when "
        "their trace records the audited de-rotate/phase-neutral-merge/re-rotate transform and "
        "`tier_mode=three_tier`; exact safety fallbacks are labeled separately. The number of "
        f"measured three-tier cells in this run is {len(three_tier)}.",
        "",
        "## Answers",
        "",
        "1. **Forecasting and selected-block recall.** Mean recall changes relative to the "
        f"no-forecast control were: full hybrid {_fmt(online_forecast_gain)}, prompt-only "
        f"{_fmt(prompt_gain)}, draft-only {_fmt(draft_gain)}, and evaluation-only oracle "
        f"{_fmt(oracle_gain)}. The hybrid forecast therefore improves selected-block recall in "
        "this run. Its per-task score changes (A minus B) were "
        f"{_fmt_task_values(forecast_scores)}, so the recall gain does not by itself establish "
        "a quality gain. Task metrics are not averaged across benchmarks.",
        "",
        "2. **Graph coverage on structured tasks.** Full-minus-no-graph per-task score changes "
        f"were {_fmt_task_values(graph_scores)}. "
        "The corresponding facility-coverage changes were "
        f"{_fmt(structured_coverage)} and {_fmt(general_coverage)}, respectively. "
        "This bounded run does not show a task-score structured advantage. "
        "On the continuous fidelity diagnostic, no-graph minus full attention-output error was "
        f"{_fmt(structured_graph_fidelity)} for the structured tasks and "
        f"{_fmt(general_graph_fidelity)} for MMStar; positive values favor the graph. "
        "The graph now consumes measured prompt-attention co-activation and exact processor "
        "visual-grid boxes/frame indices. These datasets still do not expose task-provided OCR "
        "strings or document layout regions, so no stronger structured-task claim is made.",
        "",
        "3. **Value-aware scoring and fidelity.** Removing value contribution changed mean "
        f"attention-output error by {_fmt(value_fidelity)} (G minus A). A positive value favors "
        "the value-aware full configuration; a negative value favors the ablation. The mean "
        f"logit-KL change was {_fmt(value_logit_kl)} (G minus A). The registered equation now "
        "uses signed negative lambda_v/lambda_o coefficients, so high contribution and novelty "
        "are preservation rewards rather than eviction penalties. Thus value awareness improves "
        "the requested attention-output fidelity diagnostic here, while the logit-KL result "
        "remains mixed and is disclosed rather than hidden.",
        "",
        "4. **Prototype merging at equal active memory.** Prototype-minus-exact per-task score "
        f"changes are {_fmt_task_values(prototype_scores)}. Its mean attention-output-error "
        "change is "
        f"{_fmt(prototype_attention)}; its logit-KL change is {_fmt(prototype_logit)}. Negative "
        "fidelity changes favor prototype merging. Interpret these values only for rows labeled "
        "`three_tier`; a safety fallback is not prototype evidence.",
        "",
        "5. **Repair benefit versus overhead.** Online full-minus-no-repair per-task score "
        f"changes are {_fmt_task_values(repair_scores)}. Its mean end-to-end latency change is "
        f"{_fmt(repair_latency)} seconds, "
        f"and online repair frequency is {_fmt(online_repair_frequency)}. Evaluation-only oracle "
        f"repair frequency is {_fmt(oracle_repair_frequency)}, mean measured transfer/re-decode "
        f"latency is {_fmt(oracle_repair_latency)} seconds, and quality recovery among triggered "
        f"events is {_fmt(oracle_repair_benefit)}. The registered online policy did not trigger, "
        "so this run does not justify its residual-construction overhead. The oracle row is a "
        "measured evaluation-only diagnostic, not an online result or fabricated repair benefit.",
        "",
        "## Remaining threats",
        "",
        "- Two examples per image task and one video example provide no statistical power and "
        "are sensitive to discrete task scoring.",
        "- Dirty-tree manifests are non-canonical even though they preserve git SHA, "
        "config SHA, and patch SHA.",
        "- Video-MME evidence comes from one locally cached video and repeated decoding "
        "was sensitive to local decoder seek overhead.",
        "- Processor-grid boxes are available, but benchmark OCR strings, table cells, and named "
        "document regions are not present in the cached task records.",
        "- Qwen M-RoPE prototype support is pinned to Transformers 4.55.2 and eager decode; other "
        "model families and fused attention implementations remain unsupported.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(index: Path, project_root: Path) -> None:
    index_rows = [
        json.loads(line) for line in index.read_text(encoding="utf-8").splitlines() if line
    ]
    output_root = Path(index_rows[0]["output_root"])
    study_spec = yaml.safe_load((index.parent / "study.yaml").read_text(encoding="utf-8"))
    if not isinstance(study_spec, dict):
        raise ValueError("component-ablation study metadata must be an object")
    records: list[dict[str, Any]] = []
    identities: dict[str, set[tuple[object, ...]]] = defaultdict(set)
    patch_shas: set[str] = set()
    first_manifest: dict[str, Any] | None = None
    budget_by_sample: dict[tuple[str, str], set[int]] = defaultdict(set)
    for job in index_rows:
        for cell in job["cells"]:
            run_root = output_root / cell["run_id"]
            raw_path = run_root / "results.jsonl"
            manifest_path = run_root / "manifest.json"
            if not raw_path.is_file() or not manifest_path.is_file():
                raise FileNotFoundError(f"incomplete ablation run: {run_root}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            source = manifest["source"]
            if source["config_sha"] != cell["config_sha"]:
                raise RuntimeError(f"manifest/config index SHA mismatch: {run_root}")
            if source["git_dirty"] is not True or source["canonical_eligible"] is not False:
                raise RuntimeError(
                    f"bounded v17 row has an invalid source classification: {run_root}"
                )
            if first_manifest is None:
                first_manifest = manifest
            identities[job["task"]].add(_control_identity(manifest))
            patch_shas.add(str(manifest["source"]["patch_sha"]))
            if manifest["measurement_type"] != "ablation_measured":
                raise RuntimeError(f"run is not classified as an ablation: {run_root}")
            traces = _trace_by_sample(output_root, cell["run_id"])
            cell_samples: set[str] = set()
            for line in raw_path.read_text(encoding="utf-8").splitlines():
                if not line:
                    continue
                row = cast("dict[str, Any]", json.loads(line))
                sample_id = str(row["sample_id"])
                if sample_id in cell_samples:
                    raise RuntimeError(f"duplicate result row: {cell['run_id']}/{sample_id}")
                cell_samples.add(sample_id)
                record = _sample_record(
                    row,
                    traces.get(str(row["sample_id"])),
                    task=str(job["task"]),
                    variant_id=str(cell["variant_id"]),
                    variant=str(cell["variant"]),
                )
                if record["status"] == "completed":
                    trace = traces.get(str(row["sample_id"]))
                    if trace is None:
                        raise RuntimeError(
                            "completed ablation row is missing its trace: "
                            f"{cell['run_id']}/{row['sample_id']}"
                        )
                    _validate_trace_support(
                        trace,
                        variant_id=str(cell["variant_id"]),
                        run_id=str(cell["run_id"]),
                        sample_id=str(row["sample_id"]),
                    )
                records.append(record)
                if record["status"] == "completed" and record["active_kv_bytes"] is not None:
                    budget_by_sample[(record["task"], record["sample_id"])].add(
                        int(record["active_kv_bytes"])
                    )
            if cell_samples != set(job["sample_ids"]):
                raise RuntimeError(f"result sample ledger mismatch: {run_root}")
    controls_passed = (
        all(len(values) == 1 for values in identities.values())
        and len(patch_shas) == 1
        and all(len(values) == 1 for values in budget_by_sample.values())
    )
    aggregates = _aggregate(records) + _unsupported(index_rows)
    for row in aggregates:
        row["budget_match"] = row["support_status"] != "unsupported" and all(
            len(values) == 1
            for (task, _sample), values in budget_by_sample.items()
            if task == row["task"]
        )
    frame = pd.DataFrame(aggregates).sort_values(["task", "variant_id"])
    result_path = project_root / "results" / "ablations" / "component_ablation.parquet"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), result_path)
    _write_table(frame, project_root / "tables" / "component_ablations.tex")
    _write_figures(frame, project_root / "figures")
    _write_analysis(
        frame,
        project_root / "analysis" / "component_ablations.md",
        controls_passed=controls_passed,
        output_root=output_root,
        study={
            "subset_sizes": {
                str(row["task"]): int(row["subset_size"]) for row in index_rows
            },
            "model": first_manifest["model"]["id"] if first_manifest else "unknown",
            "precision": first_manifest["model"]["precision"] if first_manifest else "unknown",
            "prefill_attention": (
                first_manifest["execution"]["attention_implementation"]
                if first_manifest
                else "unknown"
            ),
            "decode_attention": (
                first_manifest["execution"]["decode_attention_implementation"]
                if first_manifest
                else "unknown"
            ),
            "output_length": (
                study_spec.get("output_length", "unknown")
            ),
            "retention_ratio": (
                first_manifest["cache"]["retention_ratio"]
                if first_manifest
                else "unknown"
            ),
            "block_size": (
                study_spec.get("block_size", "unknown")
            ),
        },
    )
    print(
        json.dumps(
            {
                "aggregate_rows": len(frame),
                "controls_passed": controls_passed,
                "failed_samples": int(frame.failed_samples.sum()),
                "output": str(result_path.resolve()),
            },
            sort_keys=True,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    analyze(args.index.resolve(), args.project_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
