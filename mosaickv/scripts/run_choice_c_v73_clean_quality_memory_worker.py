#!/usr/bin/env python3
"""Run one clean v73 FullKV, Prompt Top-K, and MosaicKV held-out task."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import torch
import yaml
from run_choice_c_gate import _load_ledger, _sha
from run_choice_c_latency_factorial_worker import _write_telemetry
from run_choice_c_v46_prompt_mosaic_latency_worker import _source_snapshot
from run_choice_c_v63_anchor_routing_worker import _settle, _write_immutable
from run_choice_c_v67_matched_sdpa_worker import _matched_configs
from run_choice_c_v72_heldout_quality_memory_worker import _record, _resolve, _task_entry

from mosaickv.adapters.huggingface import load_hf_adapter
from mosaickv.config import load_config
from mosaickv.hf_cli import _model_kwargs
from mosaickv.measurements.telemetry import capture_gpu_environment
from mosaickv.types import MosaicKVMethod

METHOD_ORDER = ("full_kv", "prompt_attention_topk", "mosaickv_full")


def _require_clean_worktree(project_root: Path) -> None:
    status = subprocess.run(
        ["git", "-C", str(project_root), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise RuntimeError("v73 requires a completely clean tracked and untracked worktree")


def _configs(base: Any) -> dict[str, Any]:
    paired = _matched_configs(base, routed_head_only=True)
    prompt = paired[MosaicKVMethod.PROMPT_ATTENTION_TOPK.value]
    full = replace(
        prompt,
        method=MosaicKVMethod.FULL_KV,
        cache=replace(prompt.cache, retention_ratio=1.0),
    )
    return {
        MosaicKVMethod.FULL_KV.value: full,
        MosaicKVMethod.PROMPT_ATTENTION_TOPK.value: prompt,
        MosaicKVMethod.MOSAICKV_FULL.value: paired[MosaicKVMethod.MOSAICKV_FULL.value],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--task-index", required=True, type=int)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    package_root = project_root / "mosaickv"
    _require_clean_worktree(project_root)
    source_snapshot = _source_snapshot(project_root)
    spec_path = args.spec.resolve()
    output_root = args.output_root.resolve()
    spec = cast("dict[str, Any]", yaml.safe_load(spec_path.read_text()))
    if spec.get("experiment_id") != "choice_c_v73_clean_heldout_quality_memory":
        raise RuntimeError("worker requires the frozen v73 clean experiment protocol")
    if spec.get("frozen_before_sample_materialization") is not True:
        raise RuntimeError("v73 protocol was not frozen before sample materialization")

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
        raise RuntimeError(f"{task} ledger differs from the preregistered v73 protocol")
    if len(sample_ids) != expected_count or len(set(sample_ids)) != expected_count:
        raise RuntimeError(f"{task} ledger does not contain {expected_count} unique samples")
    if ledger.get("classification") != "fresh_disjoint_heldout_quality_validation":
        raise RuntimeError(f"{task} ledger is not clean-source held-out data")
    if ledger.get("canonical_eligible") is not True or ledger.get("source_git_dirty") is not False:
        raise RuntimeError(f"{task} ledger is not canonical eligible")

    base_path = _resolve(package_root, spec["candidate_config"])
    configs = _configs(load_config(base_path))
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
            generation=replace(config.generation, max_new_tokens=int(spec["output_length"])),
        )
    declared_methods = tuple(cast("list[str]", spec["methods"]))
    if declared_methods != METHOD_ORDER or set(configs) != set(METHOD_ORDER):
        raise RuntimeError("v73 method order differs from the frozen comparison")
    if configs["full_kv"].cache.retention_ratio != 1.0:
        raise RuntimeError("v73 FullKV is not configured at retention 1.0")

    worker_id = f"v73-{task}-{_sha(spec_path)[:10]}"
    telemetry_path = output_root / "telemetry" / f"{worker_id}.json"
    if telemetry_path.exists():
        prior = json.loads(telemetry_path.read_text())
        if prior.get("status") == "completed":
            print(f"v73 already completed: {worker_id}", flush=True)
            return 0
        raise RuntimeError(f"v73 has non-complete immutable telemetry: {telemetry_path}")

    warmups = int(spec["warmups_per_method"])
    repetitions = int(spec["measured_repetitions_per_method"])
    telemetry: dict[str, Any] = {
        "schema_version": 1,
        "classification": spec["classification"],
        "canonical_eligible": True,
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
        "trial_order": "forward_then_reverse_for_each_pair_of_repetitions",
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
                    run_id=f"v73-{task}-warmup-{warmup_index:02d}-{method}",
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
                _require_clean_worktree(project_root)
                if _source_snapshot(project_root) != source_snapshot:
                    raise RuntimeError("repository source changed during v73 validation")
                record = _record(
                    configs[method],
                    adapter,
                    task=task,
                    run_id=f"v73-{task}-trial-{repetition:02d}-{method}",
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
