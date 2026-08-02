#!/usr/bin/env python3
"""Aggregate Experiment-5 calibration, frozen evaluation, parity, and support evidence."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, cast

import pandas as pd  # type: ignore[import-untyped]
import yaml  # type: ignore[import-untyped]
from PIL import Image, ImageDraw, ImageFont

from mosaickv.generalization import is_completed_forecast_diagnostic
from mosaickv.manifest import sha256_bytes

MODEL_FAMILY = {
    "qwen2_5_vl_3b": "Qwen2.5-VL-3B",
    "qwen2_5_vl_7b": "Qwen2.5-VL-7B",
    "llava_1_5_7b": "LLaVA-1.5-7B",
}


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _trace(trace_root: Path, run_id: str, sample_id: str) -> tuple[Path, dict[str, Any]]:
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted((trace_root / run_id).glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("sample_id") == sample_id and value.get("status") == "completed":
            matches.append((path, cast("dict[str, Any]", value)))
    if len(matches) != 1:
        raise RuntimeError(f"expected one trace for {run_id}/{sample_id}, found {len(matches)}")
    return matches[0]


def _mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else math.nan


def _token_agreement(first: list[int], second: list[int]) -> float:
    denominator = max(len(first), len(second))
    if denominator == 0:
        return 1.0
    matches = sum(left == right for left, right in zip(first, second, strict=False))
    return matches / denominator


def aggregate(
    index: Path,
    spec_path: Path,
    hf_logit_parity_path: Path,
    vllm_parity_path: Path,
    sglang_parity_paths: tuple[Path, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    jobs = _rows(index)
    matrix_manifest_path = index.parent / "manifest.json"
    matrix_manifest = json.loads(matrix_manifest_path.read_text(encoding="utf-8"))
    if matrix_manifest.get("phase") != "frozen_evaluation":
        raise RuntimeError("analysis requires the frozen Experiment-5 evaluation matrix")
    if sha256_bytes(index.read_bytes()) != matrix_manifest["index_sha"]:
        raise RuntimeError("evaluation index SHA mismatch")
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    frozen_path = index.parent / "frozen_hyperparameters.yaml"
    frozen = yaml.safe_load(frozen_path.read_text(encoding="utf-8"))
    if sha256_bytes(frozen_path.read_bytes()) != matrix_manifest["frozen_sha"]:
        raise RuntimeError("evaluation frozen-hyperparameter SHA mismatch")
    hf_logit_parity = json.loads(hf_logit_parity_path.read_text(encoding="utf-8"))
    if (
        hf_logit_parity.get("status") != "passed"
        or hf_logit_parity.get("measurement")
        != "same_prefix_full_vocabulary_fullkv_backend_logit_parity"
    ):
        raise RuntimeError("HF backend logit parity artifact is not passed")
    hf_logit_records = {
        str(record["candidate_attention"]): cast("dict[str, Any]", record)
        for record in cast("list[dict[str, Any]]", hf_logit_parity["records"])
    }
    if set(hf_logit_records) != {"eager", "sdpa", "flash_attention_2"}:
        raise RuntimeError("HF backend logit parity artifact has unexpected backends")
    for attention, record in hf_logit_records.items():
        result_path = Path(record["result_path"])
        manifest_path = Path(record["manifest_path"])
        if (
            sha256_bytes(result_path.read_bytes()) != record["result_sha"]
            or sha256_bytes(manifest_path.read_bytes()) != record["manifest_sha"]
        ):
            raise RuntimeError(f"HF backend logit parity artifact hash mismatch: {attention}")
    output_root = Path(matrix_manifest["output_root"])
    trace_root = output_root / "traces"
    sample_records: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = [
        {
            "combination": str(item["combination"]),
            "reason": str(item["reason"]),
            "source": "predeclared_capability_gate",
        }
        for item in cast("list[dict[str, Any]]", spec["unsupported"])
    ]
    source_provenance: set[tuple[str, bool, str]] = set()
    cell_records: dict[str, list[dict[str, Any]]] = {}
    for job in jobs:
        for cell in cast("list[dict[str, Any]]", job["cells"]):
            run_id = str(cell["run_id"])
            run_root = output_root / run_id
            result_path = run_root / "results.jsonl"
            manifest_path = run_root / "manifest.json"
            if not result_path.exists() or not manifest_path.exists():
                unsupported.append(
                    {
                        "combination": f"{job['model_key']}/{cell['cell_id']}",
                        "reason": "runtime cell did not produce an immutable result and manifest",
                        "source": "runtime_failure",
                    }
                )
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            source = manifest["source"]
            if source["config_sha"] != cell["config_sha"]:
                raise RuntimeError(f"config SHA mismatch: {run_id}")
            source_provenance.add(
                (str(source["git_sha"]), bool(source["git_dirty"]), str(source["patch_sha"]))
            )
            rows = _rows(result_path)
            current: list[dict[str, Any]] = []
            for row in rows:
                if row.get("status") != "completed" or row.get("error") is not None:
                    unsupported.append(
                        {
                            "combination": (
                                f"{job['model_key']}/{cell['cell_id']}/{row['sample_id']}"
                            ),
                            "reason": str(row.get("error") or "sample failed"),
                            "source": "runtime_failure",
                        }
                    )
                    continue
                trace_path, trace = _trace(trace_root, run_id, str(row["sample_id"]))
                fidelity = cast("dict[str, Any]", trace.get("fidelity_diagnostic", {}))
                forecast = cast("dict[str, Any]", trace.get("forecast_quality_diagnostic", {}))
                forecast_completed = is_completed_forecast_diagnostic(forecast)
                repair_events = cast("list[dict[str, Any]]", trace.get("repair_events", []))
                generated_ids = [int(item) for item in trace.get("generated_token_ids", [])]
                record = {
                    "model_key": str(job["model_key"]),
                    "model": str(job["model_id"]),
                    "model_family": MODEL_FAMILY[str(job["model_key"])],
                    "job_kind": str(job["job_kind"]),
                    "task": str(cell["task_key"]),
                    "sample_id": str(row["sample_id"]),
                    "cell_id": str(cell["cell_id"]),
                    "role": str(cell["role"]),
                    "method": str(cell["method"]),
                    "prefill_attention": str(cell["prefill_attention"]),
                    "decode_attention": str(cell["decode_attention"]),
                    "answer": str(row["answer"]),
                    "task_score": float(row["task_score"]),
                    "active_kv_bytes": int(row["active_kv_bytes"]),
                    "residual_kv_bytes": int(row["residual_kv_bytes"]),
                    "end_to_end_seconds": float(row["end_to_end_time"]),
                    "compression_seconds": float(row["compression_time"]),
                    "generated_tokens": int(row["generated_tokens"]),
                    "generated_token_ids_json": json.dumps(generated_ids, separators=(",", ":")),
                    "token_agreement_to_fullkv": (
                        float(fidelity["token_agreement"])
                        if fidelity.get("status") == "completed"
                        else math.nan
                    ),
                    "forecast_cosine_similarity": (
                        float(forecast["mean_cosine_similarity"])
                        if forecast_completed
                        else math.nan
                    ),
                    "selected_block_recall": (
                        float(forecast["mean_selected_block_recall"])
                        if forecast_completed
                        else math.nan
                    ),
                    "forecast_regret": (
                        float(forecast["mean_forecast_regret"]) if forecast_completed else math.nan
                    ),
                    "repair_events": sum(bool(event["triggered"]) for event in repair_events),
                    "repair_rate_per_token": (
                        sum(bool(event["triggered"]) for event in repair_events)
                        / len(generated_ids)
                        if generated_ids
                        else 0.0
                    ),
                    "manifest_path": str(manifest_path.resolve()),
                    "trace_path": str(trace_path.resolve()),
                    "config_sha": str(cell["config_sha"]),
                    "git_sha": str(source["git_sha"]),
                    "git_dirty": bool(source["git_dirty"]),
                    "git_patch_sha": str(source["patch_sha"]),
                }
                sample_records.append(record)
                current.append(record)
            cell_records[str(cell["cell_id"])] = current
    sample_frame = pd.DataFrame(sample_records)
    generalization_samples = sample_frame[sample_frame["job_kind"] == "model_generalization"]
    aggregate_records: list[dict[str, Any]] = []
    for (model_key, task, role), group in generalization_samples.groupby(
        ["model_key", "task", "role"], sort=False
    ):
        reference = generalization_samples[
            (generalization_samples["model_key"] == model_key)
            & (generalization_samples["task"] == task)
            & (generalization_samples["role"] == "full_reference")
        ]
        if len(reference) != len(group) and role != "full_reference":
            raise RuntimeError(f"unpaired generalization rows for {model_key}/{task}/{role}")
        full_score = float(reference["task_score"].mean())
        full_bytes = float(reference["active_kv_bytes"].mean())
        mean_score = float(group["task_score"].mean())
        mean_bytes = float(group["active_kv_bytes"].mean())
        aggregate_records.append(
            {
                "model_key": model_key,
                "model_family": MODEL_FAMILY[str(model_key)],
                "task": task,
                "role": role,
                "method": str(group.iloc[0]["method"]),
                "sample_count": len(group),
                "mean_task_score": mean_score,
                "fullkv_mean_task_score": full_score,
                "quality_loss": full_score - mean_score,
                "mean_active_kv_bytes": mean_bytes,
                "active_memory_reduction": 1.0 - mean_bytes / full_bytes,
                "mean_end_to_end_seconds": float(group["end_to_end_seconds"].mean()),
                "mean_compression_seconds": float(group["compression_seconds"].mean()),
                "mean_token_agreement": _mean(
                    [float(value) for value in group["token_agreement_to_fullkv"]]
                ),
                "mean_forecast_cosine_similarity": _mean(
                    [float(value) for value in group["forecast_cosine_similarity"]]
                ),
                "mean_selected_block_recall": _mean(
                    [float(value) for value in group["selected_block_recall"]]
                ),
                "mean_forecast_regret": _mean([float(value) for value in group["forecast_regret"]]),
                "repair_event_count": int(group["repair_events"].sum()),
                "repair_rate_per_token": float(group["repair_rate_per_token"].mean()),
            }
        )
    model_task = pd.DataFrame(aggregate_records)
    backend_samples = sample_frame[sample_frame["job_kind"] == "backend_parity"]
    backend_records: list[dict[str, Any]] = []
    for row in backend_samples.itertuples(index=False):
        anchor_id = "full_eager" if row.method == "full_kv" else "exact_eager"
        anchor_group = backend_samples[
            (backend_samples["cell_id"] == anchor_id)
            & (backend_samples["sample_id"] == row.sample_id)
        ]
        if len(anchor_group) != 1:
            raise RuntimeError(f"missing backend anchor {anchor_id}/{row.sample_id}")
        anchor = anchor_group.iloc[0]
        tokens = json.loads(row.generated_token_ids_json)
        anchor_tokens = json.loads(anchor["generated_token_ids_json"])
        agreement = _token_agreement(anchor_tokens, tokens)
        logit_record = (
            hf_logit_records[str(row.prefill_attention)] if row.method == "full_kv" else None
        )
        backend_records.append(
            {
                "backend": f"hf_{row.prefill_attention}",
                "decode_backend": f"hf_{row.decode_attention}",
                "method": row.method,
                "sample_id": row.sample_id,
                "status": "passed",
                "directly_comparable": True,
                "comparison_anchor": anchor_id,
                "token_agreement": agreement,
                "sequence_agreement": agreement == 1.0,
                "backend_token_deviation": 1.0 - agreement,
                "backend_numerical_deviation": (
                    float(logit_record["maximum_absolute_logit_difference"])
                    if logit_record is not None
                    else math.nan
                ),
                "backend_mean_absolute_logit_deviation": (
                    float(logit_record["mean_absolute_logit_difference"])
                    if logit_record is not None
                    else math.nan
                ),
                "backend_mean_logit_kl": (
                    float(logit_record["mean_reference_to_candidate_logit_kl"])
                    if logit_record is not None
                    else math.nan
                ),
                "backend_numerical_deviation_status": (
                    "measured_full_vocabulary_identical_prefix"
                    if logit_record is not None
                    else "not_measured_for_compressed_path"
                ),
                "task_score_delta": float(row.task_score) - float(anchor["task_score"]),
                "answer_equal": row.answer == anchor["answer"],
                "end_to_end_seconds": float(row.end_to_end_seconds),
                "evidence_source": "experiment_5_same_sample",
                "artifact_path": row.manifest_path,
            }
        )
    prior_paths = (vllm_parity_path, *sglang_parity_paths)
    for path in prior_paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") != "passed":
            raise RuntimeError(f"prior serving parity artifact is not passed: {path}")
        backend_name = "vllm" if "vllm" in value["comparison"] else "sglang"
        backend_records.append(
            {
                "backend": backend_name,
                "decode_backend": backend_name,
                "method": "full_kv",
                "sample_id": path.stem,
                "status": "passed_prior_artifact",
                "directly_comparable": False,
                "comparison_anchor": str(value["comparison"]),
                "token_agreement": float(value["token_agreement"]),
                "sequence_agreement": float(value["token_agreement"]) == 1.0,
                "backend_token_deviation": 1.0 - float(value["token_agreement"]),
                "backend_numerical_deviation": math.nan,
                "backend_mean_absolute_logit_deviation": math.nan,
                "backend_mean_logit_kl": math.nan,
                "backend_numerical_deviation_status": "not_measured_logits_not_persisted",
                "task_score_delta": math.nan,
                "answer_equal": bool(value["checks"]["decoded_text"]),
                "end_to_end_seconds": math.nan,
                "evidence_source": "prior_controlled_fullkv_parity_not_experiment_5_sample",
                "artifact_path": str(path.resolve()),
            }
        )
    backend_frame = pd.DataFrame(backend_records)
    unsupported_frame = pd.DataFrame(unsupported).drop_duplicates()
    metadata = {
        "classification": matrix_manifest["classification"],
        "evaluation_index_sha": matrix_manifest["index_sha"],
        "frozen_hyperparameter_sha": matrix_manifest["frozen_sha"],
        "frozen_winner": frozen["winner"],
        "calibration_model": frozen["calibration_model"],
        "calibration_task": frozen["calibration_task"],
        "calibration_sample_ids": frozen["calibration_sample_ids"],
        "frozen_hyperparameters": frozen["hyperparameters"],
        "analysis_script_sha": sha256_bytes(Path(__file__).read_bytes()),
        "evaluation_source_provenance": [
            {"git_sha": item[0], "git_dirty": item[1], "patch_sha": item[2]}
            for item in sorted(source_provenance)
        ],
        "prior_parity_artifacts": [
            {"path": str(path.resolve()), "sha256": sha256_bytes(path.read_bytes())}
            for path in prior_paths
        ],
        "hf_logit_parity_artifact": {
            "path": str(hf_logit_parity_path.resolve()),
            "sha256": sha256_bytes(hf_logit_parity_path.read_bytes()),
            "measurement": hf_logit_parity["measurement"],
        },
    }
    return sample_frame, model_task, backend_frame, unsupported_frame, metadata


def _latex(frame: pd.DataFrame, columns: list[str], headers: list[str]) -> str:
    lines = [
        "\\begin{tabular}{" + "l" * len(columns) + "}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    for row in frame.itertuples(index=False):
        values: list[str] = []
        mapping = row._asdict()
        for column in columns:
            value = mapping[column]
            if isinstance(value, float):
                values.append("--" if not math.isfinite(value) else f"{value:.4f}")
            else:
                values.append(str(value).replace("_", "\\_"))
        lines.append(" & ".join(values) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def _quality_loss_plot(frame: pd.DataFrame, path: Path) -> None:
    """Write the requested plot using the common environment's Pillow dependency."""

    width, height = 1050, 620
    left, right, top, bottom = 115, 55, 80, 125
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((left, 25), "Frozen MosaicKV quality loss by model family", fill="black", font=font)
    plot_left, plot_right = left, width - right
    plot_top, plot_bottom = top, height - bottom
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#222222", width=2)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#222222", width=2)
    models = [MODEL_FAMILY[key] for key in MODEL_FAMILY]
    tasks = sorted(str(task) for task in frame["task"].unique())
    values = [float(value) for value in frame["quality_loss"] if math.isfinite(float(value))]
    if not values:
        draw.text((plot_left + 20, plot_top + 20), "No finite results", fill="black", font=font)
    else:
        minimum = min(0.0, min(values))
        maximum = max(0.0, max(values))
        if math.isclose(minimum, maximum):
            maximum = minimum + 1.0
        margin = max((maximum - minimum) * 0.12, 0.01)
        minimum -= margin
        maximum += margin

        def x_position(index: int) -> float:
            if len(models) == 1:
                return (plot_left + plot_right) / 2
            return plot_left + index * (plot_right - plot_left) / (len(models) - 1)

        def y_position(value: float) -> float:
            return plot_bottom - (value - minimum) / (maximum - minimum) * (plot_bottom - plot_top)

        for tick in range(6):
            value = minimum + tick * (maximum - minimum) / 5
            y = y_position(value)
            draw.line((plot_left - 5, y, plot_right, y), fill="#dddddd", width=1)
            draw.text((20, y - 6), f"{value:.3f}", fill="#333333", font=font)
        zero_y = y_position(0.0)
        draw.line((plot_left, zero_y, plot_right, zero_y), fill="#333333", width=2)
        palette = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")
        for task_index, task in enumerate(tasks):
            color = palette[task_index % len(palette)]
            points: list[tuple[float, float]] = []
            for model_index, model in enumerate(models):
                selected = frame[(frame["model_family"] == model) & (frame["task"] == task)]
                if len(selected) != 1:
                    continue
                points.append(
                    (x_position(model_index), y_position(float(selected.iloc[0]["quality_loss"])))
                )
            if len(points) > 1:
                draw.line(points, fill=color, width=3)
            for x, y in points:
                draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color)
            legend_x = plot_left + task_index * 170
            draw.rectangle((legend_x, height - 45, legend_x + 15, height - 30), fill=color)
            draw.text((legend_x + 22, height - 44), task, fill="black", font=font)
        for model_index, model in enumerate(models):
            x = x_position(model_index)
            draw.text((x - 55, plot_bottom + 22), model, fill="black", font=font)
        draw.text((20, plot_top - 25), "FullKV score - compressed score", fill="black", font=font)
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


def _report(
    model_task: pd.DataFrame,
    backend: pd.DataFrame,
    unsupported: pd.DataFrame,
    metadata: dict[str, Any],
) -> str:
    exact = model_task[model_task["role"] == "frozen_algorithmic"]
    adapted = model_task[model_task["role"] == "architecture_adapted_full"]
    degraded_exact = exact[exact["quality_loss"] > 0]
    degraded_adapted = adapted[adapted["quality_loss"] > 0]
    exact_memory = [float(value) for value in exact["active_memory_reduction"]]
    exact_recall = [float(value) for value in exact["mean_selected_block_recall"]]
    direct_backend = backend[backend["directly_comparable"]]
    measured_logit_backend = direct_backend[
        direct_backend["backend_numerical_deviation"].map(lambda value: math.isfinite(float(value)))
    ]
    lines = [
        "# Experiment 5: hyperparameter generalization",
        "",
        "> Classification: exploratory dirty-tree, tiny-sample validation. These are not "
        "paper-scale estimates.",
        "",
        "## Protocol",
        "",
        f"Calibration used only `{metadata['calibration_model']}` on "
        f"`{metadata['calibration_task']}` samples {metadata['calibration_sample_ids']}. "
        f"The deterministic rule selected `{metadata['frozen_winner']}`. The frozen artifact "
        f"SHA is `{metadata['frozen_hyperparameter_sha']}`. Held-out configurations were "
        "materialized only after this artifact existed; calibration and evaluation ledgers are "
        "disjoint.",
        "",
        "## Algorithmic generalization",
        "",
        "The architecture-neutral comparison is `mosaickv_exact` versus FullKV with the same "
        "frozen forecast, graph, utility, selector, block size, byte ratio, and generation "
        "settings. Prototype availability is not allowed to change these rows.",
        "",
        f"Completed frozen exact model-task cells: {len(exact)}. "
        f"{len(exact) - len(degraded_exact)} had zero task-local score loss and "
        f"{len(degraded_exact)} had a nonzero loss. Active-memory reduction ranged from "
        f"{min(exact_memory):.4f} to {max(exact_memory):.4f}; selected-block recall ranged "
        f"from {min(exact_recall):.4f} to {max(exact_recall):.4f}. No task scores are "
        "macro-averaged.",
        "",
        "## Architecture-specific adaptations",
        "",
        "Qwen rows labeled `architecture_adapted_full` add the audited M-RoPE prototype "
        "transform and guarded residual repair without retuning the frozen core parameters. "
        "They are not pooled with architecture-neutral exact rows. In immutable v1, LLaVA "
        "remained exact-only because its phase transform had not yet passed the adapter gate. "
        "That limitation was resolved afterward and does not alter these historical rows.",
        "",
        f"Completed adapted model-task cells: {len(adapted)}; "
        f"{len(degraded_adapted)} had nonzero task-local loss. These rows triggered repair "
        f"{int(adapted['repair_event_count'].sum())} times and are reported separately.",
        "",
        "## Backend implementation support",
        "",
        "Pure HF eager/SDPA/FlashAttention-2 FullKV rows are compared within method. Compressed "
        "SDPA/FA2 rows use eager prefill plus the named decode backend and are labeled that way. "
        "At immutable-v1 run time, vLLM/SGLang had only FullKV parity artifacts, so no "
        "compressed serving row was emitted. Later jobs 17869843/17871006 validate "
        "version-pinned whole-block transport, but not common-selector equivalence.",
        "",
        f"Same-sample Experiment-5 backend rows: {len(direct_backend)}; "
        f"separate prior serving artifacts: {int((~backend['directly_comparable']).sum())}.",
        f"All direct rows had exact token-sequence parity: "
        f"{bool(direct_backend['sequence_agreement'].all())}. FullKV same-prefix, "
        f"full-vocabulary numerical deviation was measured for {len(measured_logit_backend)} "
        "backend rows; compressed-path logit deviation remains explicitly unavailable.",
        "",
        "## Unsupported combinations",
        "",
    ]
    for row in unsupported.itertuples(index=False):
        lines.append(f"- `{row.combination}`: {row.reason} ({row.source}).")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Two examples per model-task and one same-sample backend probe can validate plumbing, "
            "freezing, leakage controls, and gross transfer failures. They cannot establish "
            "population-level cross-model generalization. Failed and unsupported cells remain "
            "visible and no incompatible benchmark scores are macro-averaged.",
            "",
        ]
    )
    return "\n".join(lines)


def _architecture_report(
    model_task: pd.DataFrame,
    backend: pd.DataFrame,
    unsupported: pd.DataFrame,
    metadata: dict[str, Any],
) -> str:
    exact = model_task[model_task["role"] == "frozen_algorithmic"]
    adapted = model_task[model_task["role"] == "architecture_adapted_full"]
    parameters = cast("dict[str, Any]", metadata["frozen_hyperparameters"])
    direct_parity = bool(backend[backend["directly_comparable"]]["sequence_agreement"].all())
    measured_backend = backend[
        backend["backend_numerical_deviation"].map(lambda value: math.isfinite(float(value)))
    ]
    lines = [
        "# Architecture-specific adaptations",
        "",
        "> Exploratory dirty-tree evidence from two held-out examples per model/task. "
        "The report distinguishes uniform algorithm transfer from implementation support.",
        "",
        "## Frozen architecture-neutral algorithm",
        "",
        "The uniform comparison uses exact block selection only. Its frozen core is "
        f"retention `{parameters['retention_ratio']}`, block size `{parameters['block_size']}`, "
        f"`{parameters['forecast_mode']}` forecasting with `{parameters['draft_steps']}` draft "
        f"steps, prompt window `{parameters['prompt_window']}`, and "
        f"`{parameters['graph_max_neighbors']}` graph neighbors. No model or task was retuned.",
        "",
    ]
    for row in exact.itertuples(index=False):
        lines.append(
            f"- `{row.model_family}` / `{row.task}`: task-local loss "
            f"`{row.quality_loss:.4f}`, active-memory reduction "
            f"`{row.active_memory_reduction:.4f}`, latency "
            f"`{row.mean_end_to_end_seconds:.4f}s`, forecast cosine "
            f"`{row.mean_forecast_cosine_similarity:.4f}`, selected recall "
            f"`{row.mean_selected_block_recall:.4f}`, repair events `0`."
        )
    lines.extend(
        [
            "",
            "## Model-specific cache semantics",
            "",
            "- Qwen2.5-VL uses multimodal rotary positions. Its optional adapted rows use the "
            "audited M-RoPE anchor-frame prototype transform and eager-only attention capture "
            "for guarded residual repair.",
            "- In immutable v1, LLaVA-1.5 remained exact-selection only because its phase "
            "transform had not yet passed; no prototype or repair rows were emitted.",
            "- In immutable v1, the optional LLaVA-OneVision-7B checkpoint was neither cached "
            "nor registered, so no row was simulated.",
            "- After v1, exact Llama/Qwen2 phase transforms and uniform-BF16 OneVision-7B "
            "loading passed exploratory A100 parity, real-prototype, and repair gates. These "
            "new gates do not rewrite the historical experiment.",
            "",
            "## Adapted Qwen outcome",
            "",
        ]
    )
    for row in adapted.itertuples(index=False):
        lines.append(
            f"- `{row.model_family}` / `{row.task}`: task-local loss "
            f"`{row.quality_loss:.4f}`, latency `{row.mean_end_to_end_seconds:.4f}s`, "
            f"repair events `{row.repair_event_count}`."
        )
    lines.extend(
        [
            "",
            "On this tiny slice, the adaptation recovered the Qwen-3B ChartQA loss but "
            "introduced losses on Qwen-7B DocVQA and ChartQA and incurred substantial "
            "diagnostic-enabled latency. It therefore does not establish that prototype/repair "
            "hyperparameters generalize; the architecture-neutral exact rows are the valid "
            "cross-family result.",
            "LLaVA's zero losses on DocVQA and ChartQA are floor-limited because its matching "
            "FullKV rows also scored zero; they are parity checks, not positive quality evidence.",
            "",
            "## Backend support versus algorithm transfer",
            "",
            "HF eager supports the complete audited Qwen path. HF SDPA and FlashAttention-2 "
            "passed same-sample token parity for FullKV and for exact selection with eager "
            "prefill plus fused decode. They do not support prototype-risk attention capture "
            "or repair in this implementation. At immutable-v1 run time, vLLM and SGLang had "
            "FullKV parity only, so no compressed serving row was emitted. Later jobs "
            "17869843/17871006 validate version-pinned whole-block transport, but not "
            "common-selector equivalence or prototype/repair support.",
            "",
            f"Direct HF parity rows: `{int(backend['directly_comparable'].sum())}`. Explicit "
            f"unsupported combinations: `{len(unsupported)}`. FullKV same-prefix, "
            f"full-vocabulary logit deviation was measured for `{len(measured_backend)}` HF "
            "rows; compressed serving rows retain token parity only. All direct probes had "
            f"token-sequence parity `{direct_parity}`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument(
        "--hf-logit-parity",
        type=Path,
        default=Path("/scratch/djy8hg/runs/mosaickv/generalization-backend-logits-v1/summary.json"),
    )
    parser.add_argument(
        "--vllm-parity",
        type=Path,
        default=Path("/scratch/djy8hg/runs/mosaickv/vllm-fullkv-0.10.2-fix9/parity/ci-red.json"),
    )
    parser.add_argument(
        "--sglang-parity",
        type=Path,
        nargs="+",
        default=[
            Path("/scratch/djy8hg/runs/mosaickv/sglang-fullkv-0.5.1-fix8/parity-sdpa/ci-red.json"),
            Path(
                "/scratch/djy8hg/runs/mosaickv/sglang-fullkv-0.5.1-fix8/parity-sdpa/ci-white.json"
            ),
        ],
    )
    parser.add_argument(
        "--result-directory", type=Path, default=project_root / "results" / "generalization"
    )
    parser.add_argument(
        "--analysis",
        type=Path,
        default=project_root / "analysis" / "hyperparameter_generalization.md",
    )
    parser.add_argument(
        "--architecture-report",
        type=Path,
        default=project_root / "analysis" / "architecture_specific_adaptations.md",
    )
    args = parser.parse_args()
    samples, model_task, backend, unsupported, metadata = aggregate(
        args.index.resolve(),
        args.spec.resolve(),
        args.hf_logit_parity.resolve(),
        args.vllm_parity.resolve(),
        tuple(path.resolve() for path in args.sglang_parity),
    )
    destination = args.result_directory.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    samples.to_parquet(destination / "sample_results.parquet", index=False)
    model_task.to_parquet(destination / "model_task_generalization.parquet", index=False)
    backend.to_parquet(destination / "backend_parity.parquet", index=False)
    unsupported.to_parquet(destination / "unsupported_combinations.parquet", index=False)
    (destination / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    tables = project_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    (tables / "generalization_by_model_task.tex").write_text(
        _latex(
            model_task,
            [
                "model_family",
                "task",
                "role",
                "mean_task_score",
                "quality_loss",
                "active_memory_reduction",
                "mean_end_to_end_seconds",
                "mean_forecast_cosine_similarity",
                "mean_selected_block_recall",
                "mean_forecast_regret",
                "repair_rate_per_token",
            ],
            [
                "Model",
                "Task",
                "Role",
                "Score",
                "Loss",
                "Memory red.",
                "Latency",
                "Forecast cos.",
                "Recall",
                "Regret",
                "Repair",
            ],
        ),
        encoding="utf-8",
    )
    (tables / "backend_parity.tex").write_text(
        _latex(
            backend,
            [
                "backend",
                "decode_backend",
                "method",
                "status",
                "token_agreement",
                "backend_token_deviation",
                "backend_numerical_deviation",
                "backend_mean_absolute_logit_deviation",
                "backend_mean_logit_kl",
                "task_score_delta",
            ],
            [
                "Prefill",
                "Decode",
                "Method",
                "Status",
                "Token agr.",
                "Token dev.",
                "Max logit dev.",
                "Mean logit dev.",
                "Mean KL",
                "Score delta",
            ],
        ),
        encoding="utf-8",
    )
    figures = project_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    exact = model_task[model_task["role"] == "frozen_algorithmic"]
    _quality_loss_plot(exact, figures / "quality_loss_model_family.pdf")
    args.analysis.parent.mkdir(parents=True, exist_ok=True)
    args.analysis.write_text(_report(model_task, backend, unsupported, metadata), encoding="utf-8")
    args.architecture_report.parent.mkdir(parents=True, exist_ok=True)
    args.architecture_report.write_text(
        _architecture_report(model_task, backend, unsupported, metadata), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "backend_rows": len(backend),
                "model_task_rows": len(model_task),
                "sample_rows": len(samples),
                "unsupported_rows": len(unsupported),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
