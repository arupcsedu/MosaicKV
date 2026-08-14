#!/usr/bin/env python3
"""Run one preregistered held-out quality-memory task on an isolated GPU."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import pandas as pd  # type: ignore[import-untyped]
import torch
import yaml
from run_choice_c_gate import _load_ledger, _sha
from run_choice_c_latency_factorial_worker import _run_online, _write_telemetry
from run_choice_c_v46_prompt_mosaic_latency_worker import _source_snapshot
from run_choice_c_v63_anchor_routing_worker import _settle, _write_immutable
from run_choice_c_v67_matched_sdpa_worker import METHOD_ORDER, _matched_configs

from mosaickv.adapters.huggingface import load_hf_adapter
from mosaickv.config import load_config
from mosaickv.hf_cli import _model_kwargs
from mosaickv.measurements.telemetry import capture_gpu_environment
from mosaickv.types import MosaicKVMethod


def _resolve(package_root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("preregistered path must be a non-empty string")
    path = Path(value)
    return (package_root / path).resolve() if not path.is_absolute() else path.resolve()


def _task_entry(spec: dict[str, Any], index: int) -> dict[str, Any]:
    tasks = cast("list[dict[str, Any]]", spec.get("tasks"))
    if not 0 <= index < len(tasks):
        raise ValueError(f"task index must be in [0, {len(tasks) - 1}]")
    return tasks[index]


def _record(
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
) -> dict[str, Any]:
    frame = _run_online(
        config,
        adapter,
        task=task,
        run_id=run_id,
        output_root=output_root,
        ledger_path=ledger_path,
        ledger_sha=ledger_sha,
        expected_sample_ids=sample_ids,
    )
    rows = [
        {
            "sample_id": str(row.sample_id),
            "answer": str(row.answer),
            "reference": str(row.reference),
            "task_score": float(row.task_score),
            "active_kv_bytes": int(row.active_kv_bytes),
            "residual_kv_bytes": int(row.residual_kv_bytes),
            "compression_time": float(row.compression_time),
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
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--task-index", required=True, type=int)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    package_root = project_root / "mosaickv"
    spec_path = args.spec.resolve()
    output_root = args.output_root.resolve()
    spec = cast("dict[str, Any]", yaml.safe_load(spec_path.read_text()))
    if spec.get("experiment_id") != "choice_c_v72_heldout_quality_memory":
        raise RuntimeError("worker requires the frozen v72 experiment protocol")
    if spec.get("frozen_before_sample_materialization") is not True:
        raise RuntimeError("v72 protocol was not frozen before sample materialization")

    entry = _task_entry(spec, args.task_index)
    task = str(entry["task"])
    ledger_path = _resolve(package_root, entry["ledger"])
    ledger, ledger_sha = _load_ledger(ledger_path)
    sample_ids = tuple(cast("list[str]", ledger["sample_ids"]))
    expected_count = int(entry["sample_count"])
    expected_ledger = {
        "task": task,
        "dataset_id": entry["dataset_id"],
        "dataset_revision": entry["dataset_revision"],
        "split": entry["split"],
        "selection_seed": entry["selection_seed"],
        "sample_count": expected_count,
    }
    if any(ledger.get(key) != value for key, value in expected_ledger.items()):
        raise RuntimeError(f"{task} ledger differs from the preregistered protocol")
    if len(sample_ids) != expected_count or len(set(sample_ids)) != expected_count:
        raise RuntimeError(f"{task} ledger does not contain {expected_count} unique samples")
    if ledger.get("classification") not in {
        "fresh_disjoint_heldout_quality_validation",
        "exploratory_dirty_tree_fresh_disjoint_heldout_quality_validation",
    }:
        raise RuntimeError(f"{task} ledger is not a declared fresh held-out ledger")

    base_path = _resolve(package_root, spec["candidate_config"])
    configs = _matched_configs(load_config(base_path), routed_head_only=True)
    for method, config in tuple(configs.items()):
        configs[method] = replace(
            config,
            dataset=replace(
                config.dataset,
                id=str(entry["dataset_id"]),
                revision=str(entry["dataset_revision"]),
                split=str(entry["split"]),
            ),
            execution=replace(config.execution, seed=int(spec["seed"])),
            generation=replace(
                config.generation,
                max_new_tokens=int(spec["output_length"]),
            ),
        )
    declared_methods = tuple(cast("list[str]", spec["methods"]))
    if declared_methods != METHOD_ORDER or set(configs) != set(METHOD_ORDER):
        raise RuntimeError("v72 method order differs from the frozen paired comparison")

    source_snapshot = _source_snapshot(project_root)
    worker_id = f"v72-{task}-{_sha(spec_path)[:10]}"
    telemetry_path = output_root / "telemetry" / f"{worker_id}.json"
    if telemetry_path.exists():
        prior = json.loads(telemetry_path.read_text())
        if prior.get("status") == "completed":
            print(f"v72 already completed: {worker_id}", flush=True)
            return 0
        raise RuntimeError(f"v72 has non-complete immutable telemetry: {telemetry_path}")

    warmups = int(spec["warmups_per_method"])
    repetitions = int(spec["measured_repetitions_per_method"])
    telemetry: dict[str, Any] = {
        "schema_version": 1,
        "classification": ledger["classification"],
        "canonical_eligible": bool(ledger.get("canonical_eligible", False)),
        "worker_id": worker_id,
        "task": task,
        "task_index": args.task_index,
        "spec": str(spec_path),
        "spec_sha256": _sha(spec_path),
        "base_config": str(base_path),
        "base_config_sha256": _sha(base_path),
        "ledger": str(ledger_path),
        "ledger_sha256": ledger_sha,
        "dataset_id": entry["dataset_id"],
        "dataset_revision": entry["dataset_revision"],
        "sample_ids": list(sample_ids),
        "warmups_per_method": warmups,
        "measured_repetitions_per_method": repetitions,
        "trial_order": "alternating_prompt_first_then_mosaic_first",
        "primary_gate": spec["primary_gate"],
        "latency_policy": cast("dict[str, Any]", spec["latency_report"])["policy"],
        "source_snapshot": source_snapshot,
        "before_model_load": asdict(capture_gpu_environment(torch)),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID", "not_used"),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", "not_used"),
            "node": os.environ.get("SLURMD_NODENAME", "not_used"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
        },
        "trials": [],
        "status": "started",
    }
    try:
        model_config = configs[MosaicKVMethod.MOSAICKV_FULL.value]
        adapter = load_hf_adapter(
            model_config.model.id,
            revision=model_config.model.revision,
            model_kwargs=_model_kwargs(model_config, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        warmup = dict(ledger)
        warmup.update({"sample_count": 1, "sample_ids": [sample_ids[0]]})
        warmup_path = output_root / "warmup_ledgers" / f"{task}.json"
        _write_immutable(warmup_path, warmup)
        warmup_sha = _sha(warmup_path)
        for warmup_index in range(warmups):
            order = METHOD_ORDER if warmup_index % 2 == 0 else tuple(reversed(METHOD_ORDER))
            for order_index, method in enumerate(order):
                _record(
                    configs[method],
                    adapter,
                    task=task,
                    run_id=f"v72-{task}-warmup-{warmup_index:02d}-{method}",
                    output_root=output_root,
                    ledger_path=warmup_path,
                    ledger_sha=warmup_sha,
                    sample_ids=(sample_ids[0],),
                    phase="warmup",
                    repetition=warmup_index,
                    order=order_index,
                )
                _settle(adapter)
        telemetry["before_measurement"] = asdict(capture_gpu_environment(torch))
        for repetition in range(repetitions):
            order = METHOD_ORDER if repetition % 2 == 0 else tuple(reversed(METHOD_ORDER))
            for order_index, method in enumerate(order):
                if _source_snapshot(project_root) != source_snapshot:
                    raise RuntimeError("repository source changed during v72 validation")
                record = _record(
                    configs[method],
                    adapter,
                    task=task,
                    run_id=f"v72-{task}-trial-{repetition:02d}-{method}",
                    output_root=output_root,
                    ledger_path=ledger_path,
                    ledger_sha=ledger_sha,
                    sample_ids=sample_ids,
                    phase="measured",
                    repetition=repetition,
                    order=order_index,
                )
                cast("list[dict[str, Any]]", telemetry["trials"]).append(record)
                _settle(adapter)
        telemetry["status"] = "completed"
    except Exception as error:
        telemetry["status"] = "failed"
        telemetry["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        telemetry["after_measurement"] = asdict(capture_gpu_environment(torch))
        _write_telemetry(telemetry_path, telemetry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
