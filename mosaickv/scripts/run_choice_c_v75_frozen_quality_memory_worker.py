#!/usr/bin/env python3
"""Evaluate a frozen label-blind candidate on new, bound held-out samples."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import torch
import yaml  # type: ignore[import-untyped]
from choice_c_v74_protocol import apply_candidate, apply_common_cache_policy
from run_choice_c_gate import _load_ledger, _sha
from run_choice_c_latency_factorial_worker import _write_telemetry
from run_choice_c_v46_prompt_mosaic_latency_worker import _source_snapshot
from run_choice_c_v63_anchor_routing_worker import _settle, _write_immutable
from run_choice_c_v72_heldout_quality_memory_worker import _record, _resolve, _task_entry
from run_choice_c_v73_clean_quality_memory_worker import (
    METHOD_ORDER,
    _configs,
    _require_clean_worktree,
)

from mosaickv.adapters.huggingface import load_hf_adapter
from mosaickv.config import load_config
from mosaickv.hf_cli import _model_kwargs
from mosaickv.measurements.telemetry import capture_gpu_environment
from mosaickv.types import MosaicKVMethod


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--task-index", required=True, type=int)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-prefix", default="v75")
    parser.add_argument(
        "--expected-experiment-id",
        default="choice_c_v75_frozen_heldout_quality_memory",
    )
    parser.add_argument(
        "--expected-frozen-classification",
        default="frozen_label_blind_calibrated_candidate",
    )
    parser.add_argument(
        "--expected-ledger-classification",
        default="fresh_disjoint_heldout_quality_validation",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    package_root = project_root / "mosaickv"
    _require_clean_worktree(project_root)
    source_snapshot = _source_snapshot(project_root)
    spec_path = args.spec.resolve()
    output_root = args.output_root.resolve()
    spec = cast("dict[str, Any]", yaml.safe_load(spec_path.read_text(encoding="utf-8")))
    run_prefix = str(args.run_prefix)
    if not run_prefix or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in run_prefix
    ):
        raise ValueError("run-prefix must contain only lowercase letters, digits, '_' or '-'")
    if spec.get("experiment_id") != args.expected_experiment_id:
        raise RuntimeError("worker experiment differs from the requested held-out protocol")
    if spec.get("frozen_before_sample_materialization") is not True:
        raise RuntimeError("v75 candidate was not frozen before sample materialization")

    freeze_path = _resolve(package_root, spec["frozen_candidate"])
    freeze_sha = _sha(freeze_path)
    freeze = cast("dict[str, Any]", yaml.safe_load(freeze_path.read_text(encoding="utf-8")))
    if (
        freeze.get("classification") != args.expected_frozen_classification
        or freeze.get("labels_accessed") is not False
        or freeze.get("official_task_scores_accessed") is not False
    ):
        raise RuntimeError("v75 frozen candidate lacks label-blind calibration provenance")
    parent_seal_raw = spec.get("parent_negative_result_seal")
    if parent_seal_raw is not None:
        parent_seal = _resolve(package_root, parent_seal_raw)
        if freeze.get("parent_negative_result_seal") != str(parent_seal) or freeze.get(
            "parent_negative_result_seal_sha256"
        ) != _sha(parent_seal):
            raise RuntimeError("frozen candidate does not bind the sealed parent result")
    if freeze.get("calibration_source_snapshot_sha256") != source_snapshot["snapshot_sha256"]:
        raise RuntimeError("v75 source snapshot differs from the calibration snapshot")
    calibration_spec_path = _resolve(package_root, spec["calibration_spec"])
    if freeze.get("calibration_spec_sha256") != _sha(calibration_spec_path):
        raise RuntimeError("v75 calibration spec differs from the frozen candidate")
    calibration_spec = cast(
        "dict[str, Any]", yaml.safe_load(calibration_spec_path.read_text(encoding="utf-8"))
    )
    common_cache_policy = cast("dict[str, Any]", calibration_spec["common_cache_policy"])
    if freeze.get("common_cache_policy") != common_cache_policy:
        raise RuntimeError("frozen candidate does not bind the common cache policy")
    selected_id = str(freeze["selected_candidate"])
    declared_candidate = next(
        (
            candidate
            for candidate in cast("list[dict[str, Any]]", calibration_spec["candidates"])
            if str(candidate["id"]) == selected_id
        ),
        None,
    )
    if declared_candidate is None or freeze.get("candidate") != declared_candidate:
        raise RuntimeError("frozen candidate does not match the preregistered candidate catalog")

    entry = _task_entry(spec, args.task_index)
    task = str(entry["task"])
    ledger_path = _resolve(package_root, entry["ledger"])
    ledger, ledger_sha = _load_ledger(ledger_path)
    sample_ids = tuple(cast("list[str]", ledger["sample_ids"]))
    expected = {
        "task": task,
        "dataset_id": entry["dataset_id"],
        "dataset_revision": entry["dataset_revision"],
        "split": entry["split"],
        "selection_seed": entry["selection_seed"],
        "sample_count": entry["sample_count"],
        "classification": args.expected_ledger_classification,
        "canonical_eligible": True,
        "source_git_dirty": False,
        "selection_freeze_sha256": freeze_sha,
    }
    if any(ledger.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"{task} ledger differs from the bound held-out protocol")
    if len(sample_ids) != int(entry["sample_count"]) or len(set(sample_ids)) != len(sample_ids):
        raise RuntimeError(f"{task} v75 ledger has the wrong unique sample count")

    base_path = _resolve(package_root, spec["candidate_config"])
    if freeze.get("candidate_config_sha256") != _sha(base_path):
        raise RuntimeError("v75 base candidate config differs from calibration")
    configs = {
        method: apply_common_cache_policy(config, common_cache_policy)
        for method, config in _configs(load_config(base_path)).items()
    }
    configs[MosaicKVMethod.MOSAICKV_FULL.value] = apply_candidate(
        configs[MosaicKVMethod.MOSAICKV_FULL.value], declared_candidate
    )
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
    if tuple(cast("list[str]", spec["methods"])) != METHOD_ORDER:
        raise RuntimeError("v75 method order differs from the frozen comparison")

    worker_id = f"{run_prefix}-{task}-{_sha(spec_path)[:10]}"
    telemetry_path = output_root / "telemetry" / f"{worker_id}.json"
    if telemetry_path.exists():
        prior = json.loads(telemetry_path.read_text(encoding="utf-8"))
        if prior.get("status") == "completed":
            print(f"held-out worker already completed: {worker_id}", flush=True)
            return 0
        raise RuntimeError(f"v75 has non-complete immutable telemetry: {telemetry_path}")

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
        "frozen_candidate": str(freeze_path),
        "frozen_candidate_sha256": freeze_sha,
        "selected_candidate": selected_id,
        "base_config": str(base_path),
        "base_config_sha256": _sha(base_path),
        "ledger": str(ledger_path),
        "ledger_sha256": ledger_sha,
        "dataset_id": entry["dataset_id"],
        "dataset_revision": entry["dataset_revision"],
        "sample_ids": list(sample_ids),
        "warmups_per_method": warmups,
        "measured_repetitions_per_method": repetitions,
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
                    run_id=f"{run_prefix}-{task}-warmup-{warmup_index:02d}-{method}",
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
                    raise RuntimeError("repository source changed during held-out validation")
                record = _record(
                    configs[method],
                    adapter,
                    task=task,
                    run_id=f"{run_prefix}-{task}-trial-{repetition:02d}-{method}",
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
