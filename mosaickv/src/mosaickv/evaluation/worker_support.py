"""Shared, import-safe helpers for canonical HF experiment workers.

The Slurm entry points live in :mod:`mosaickv/scripts` so they can be invoked
directly.  They must not, however, import one another through the ambient
current directory: that pattern breaks unit tests and makes a submitted job
depend on its launch directory.  This module is the installed-package boundary
for the small set of common worker operations.
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pandas as pd  # type: ignore[import-untyped]
import torch
import yaml

from mosaickv.config import (
    DatasetConfig,
    ForecastingConfig,
    GraphConfig,
    PrototypeConfig,
    RepairConfig,
    ResidualConfig,
    SelectionConfig,
    canonical_config_json,
    config_sha256,
)
from mosaickv.evaluation.paper_candidate import PAPER_CANDIDATE_TASK_ROUTES
from mosaickv.hf_cli import run_hf_evaluation
from mosaickv.types import ForecastMode, MosaicKVMethod, RepairPolicy


def settle(adapter: Any) -> None:
    """Release Python temporaries and finish CUDA work before the next arm."""

    gc.collect()
    if torch.cuda.is_available() and getattr(adapter.device, "type", "cpu") == "cuda":
        torch.cuda.synchronize(adapter.device)


def control_configs(
    template: Any,
    *,
    task: str,
    seed: int,
    output_length: int,
    dataset: DatasetConfig | None = None,
) -> dict[str, Any]:
    """Construct FullKV and Prompt Top-K controls with identical inputs.

    This deliberately turns off MosaicKV-only components.  Derived baseline
    scripts may replace the method and cache policy afterwards, but they begin
    from this single controlled input/generation configuration.
    """

    if dataset is None:
        route = PAPER_CANDIDATE_TASK_ROUTES[task]
    else:
        from mosaickv.evaluation.tasks import default_task_registry

        spec = default_task_registry().resolve(task)
        if (dataset.id, dataset.split) != (spec.dataset_id, spec.split):
            raise ValueError("explicit dataset identity does not match the task registry")
        route = dataset
    common: dict[str, Any] = {
        "dataset": route,
        "execution": replace(template.execution, seed=seed),
        "generation": replace(template.generation, max_new_tokens=output_length),
        "forecasting": ForecastingConfig(
            enabled=False,
            mode=ForecastMode.PROMPT_WINDOW,
            prompt_window=16,
            draft_steps=0,
        ),
        "graph": GraphConfig(enabled=False),
        "selection": SelectionConfig(enabled=False),
        "prototypes": PrototypeConfig(enabled=False),
        "residual": ResidualConfig(enabled=False),
        "repair": RepairConfig(
            enabled=False,
            policy=RepairPolicy.NONE,
            max_blocks_per_step=0,
        ),
    }
    prompt = replace(
        template,
        method=MosaicKVMethod.PROMPT_ATTENTION_TOPK,
        cache=replace(
            template.cache,
            budget_value=2**63 - 1,
            total_storage_budget_bytes=None,
        ),
        **common,
    )
    full = replace(
        template,
        method=MosaicKVMethod.FULL_KV,
        bounded_decode=None,
        cache=replace(
            template.cache,
            budget_value=2**63 - 1,
            retention_ratio=1.0,
            total_storage_budget_bytes=None,
        ),
        **common,
    )
    return {"full_kv": full, "prompt_attention_topk": prompt}


def _write_config(config: Any, path: Path) -> None:
    payload = yaml.safe_dump(json.loads(canonical_config_json(config)), sort_keys=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != payload:
        raise RuntimeError(f"refusing to mutate resolved run configuration: {path}")
    if not path.exists():
        path.write_text(payload, encoding="utf-8")


def record_hf_run(
    config: Any,
    adapter: Any,
    *,
    task: str,
    run_id: str,
    output_root: Path,
    ledger_path: Path,
    ledger_sha: str,
    sample_ids: tuple[str, ...],
    phase: str,
    repetition: int,
    order: int,
    export_metrics: bool = False,
    materialized_milebench: bool = False,
    materialized_prefixkv: bool = False,
    materialized_coco: bool = False,
    fullkv_reference_input: str | Path | None = None,
    fullkv_reference_sha256: str | None = None,
    fullkv_reference_output: str | Path | None = None,
    comparison_decode_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run and validate one immutable, fully paired HF measurement arm."""

    if config.diagnostics.enabled:
        raise RuntimeError("canonical workers require online-only diagnostics")
    if materialized_coco and (
        task != "coco_caption"
        or not export_metrics
        or materialized_prefixkv
        or materialized_milebench
    ):
        raise ValueError("materialized COCO requires its task and metric exports")
    if materialized_milebench and (task != "milebench" or not export_metrics):
        raise ValueError("materialized MileBench requires its task and metric exports")
    if materialized_prefixkv and (
        task not in {"llava_description", "mmvet_prefixkv"}
        or not export_metrics
        or materialized_milebench
    ):
        raise ValueError("materialized PrefixKV requires its native task and metric exports")
    config_path = output_root / "configs" / f"{run_id}-{config_sha256(config)}.yaml"
    _write_config(config, config_path)
    arguments = argparse.Namespace(
        task=task,
        run_id=run_id,
        output_dir=str(output_root),
        raw_output=None,
        parquet_output=None,
        manifest=None,
        trace_directory=str(output_root / "traces"),
        subset_size=None,
        full_dataset=False,
        sample_offset=0,
        sample_id_file=str(ledger_path),
        sample_id_file_sha=ledger_sha,
        local_files_only=True,
        comparison_decode_policy=comparison_decode_policy,
    )
    if export_metrics:
        metric_root = output_root / run_id
        arguments.metric_output = str(metric_root / "metrics.jsonl")
        arguments.metric_observations_parquet_output = str(metric_root / "observations.parquet")
        arguments.metric_aggregates_parquet_output = str(metric_root / "metrics.parquet")
    if materialized_milebench or materialized_prefixkv or materialized_coco:
        arguments.sample_id_file = arguments.sample_id_file_sha = None
        arguments.paper_native_ledger = str(ledger_path)
        arguments.paper_native_ledger_sha = ledger_sha
        arguments.unified_milebench = materialized_milebench
        arguments.unified_prefixkv = materialized_prefixkv
        arguments.unified_coco = materialized_coco
        arguments.fullkv_reference_input = fullkv_reference_input
        arguments.fullkv_reference_sha = fullkv_reference_sha256
        arguments.fullkv_reference_output = fullkv_reference_output
    status, summary = run_hf_evaluation(arguments, config, adapter=adapter)
    failed_samples = summary.get("failed_samples", 0)
    if status or failed_samples or summary.get("metric_error") or summary.get("metric_errors"):
        raise RuntimeError(f"run failed or contains failed samples: {run_id}: {summary}")
    frame = pd.read_parquet(cast("str", summary["parquet_output"]))
    if (
        len(frame) != len(sample_ids)
        or set(str(value) for value in frame["sample_id"]) != set(sample_ids)
        or set(str(value) for value in frame["status"]) != {"completed"}
    ):
        raise RuntimeError(f"run is incomplete: {run_id}")
    rows = [
        {
            "sample_id": str(row.sample_id),
            "answer": str(row.answer),
            "reference": str(row.reference),
            # Corpus CIDEr has no independently averaged per-sample task_score.
            "task_score": None if pd.isna(row.task_score) else float(row.task_score),
            "active_kv_bytes": int(row.active_kv_bytes),
            **{
                field: None if pd.isna(getattr(row, field, None)) else int(getattr(row, field))
                for field in (
                    "initial_active_kv_storage_bytes",
                    "final_active_kv_storage_bytes",
                    "peak_active_kv_storage_bytes",
                )
            },
            "residual_kv_bytes": int(row.residual_kv_bytes),
            "compression_time": float(row.compression_time),
            **{
                field: None if pd.isna(getattr(row, field, None)) else float(getattr(row, field))
                for field in ("prefill_time", "ttft")
            },
            "decode_time": float(row.decode_time),
            "end_to_end_time": float(row.end_to_end_time),
            "generated_tokens": int(row.generated_tokens),
            "repair_count": int(row.repair_count),
            "status": str(row.status),
            "error": None if pd.isna(row.error) else str(row.error),
            "manifest_path": str(row.manifest_path),
        }
        for row in frame.itertuples()
    ]
    return {
        "phase": phase,
        "repetition": repetition,
        "order": order,
        "method": config.method.value,
        "run_id": run_id,
        "rows": rows,
        "metric_observations_parquet_output": summary.get("metric_observations_parquet_output"),
        "metric_aggregates_parquet_output": summary.get("metric_aggregates_parquet_output"),
    }
