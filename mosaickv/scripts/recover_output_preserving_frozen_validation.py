#!/usr/bin/env python3
"""Recover post-processing after completed, immutable GPU inference.

This recovery is deliberately narrow.  It accepts only the known
paper-native effective-method validation failure, copies every measured row
without changing any measured value, reruns corpus metric reduction, and
writes a new manifest and worker record.  The original failed attempt remains
immutable and is hash-bound from the recovery record.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import NoReturn

import pandas as pd
from prepare_output_preserving_sweep import source_digest
from run_coco_shared_bytes import gpu_snapshot
from run_prefixkv_decode_preflight import digest, write_once

from mosaickv.config import config_sha256, load_config
from mosaickv.evaluation.coco_shared_bytes import validate_caption_rows
from mosaickv.evaluation.model import EvaluationRequest
from mosaickv.evaluation.paper_native_lanes import (
    load_paper_native_ledger,
    run_materialized_paper_native_evaluation,
)
from mosaickv.evaluation.storage import JsonlResultStore, load_jsonl
from mosaickv.evaluation.tasks import default_task_registry
from mosaickv.hf_cli import (
    _input_provenance,
    _paper_evaluation_provenance,
    _trace_sha,
)
from mosaickv.manifest import ArtifactProvenance, RunManifestWriter, sha256_bytes, sha256_text
from mosaickv.types import MeasurementType, ResultCategory

EXPECTED_ERROR = "ValueError: paper-native run changed effective method or used a fallback"


@dataclass(slots=True)
class _ReplayOnlyModel:
    """Metadata-only model; generation is forbidden during recovery."""

    model_id: str
    backend: str
    method: str
    retention_ratio: float
    config: object
    supports_video: bool = False

    def generate(self, request: EvaluationRequest) -> NoReturn:
        del request
        raise AssertionError("recovery attempted new model inference")


def _one(items: list[Path], description: str) -> Path:
    if len(items) != 1:
        raise ValueError(f"expected one {description}, found {len(items)}")
    return items[0]


def _record_rows(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Recreate worker_support.record_hf_run's stable row projection."""

    return [
        {
            "sample_id": str(row.sample_id),
            "answer": str(row.answer),
            "reference": str(row.reference),
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


def recover(args: argparse.Namespace) -> None:
    import torch

    started = time.monotonic()
    root = args.root.resolve()
    if digest(root / "plan.json") != args.plan_sha256:
        raise ValueError("plan hash mismatch")
    plan = json.loads((root / "plan.json").read_text())
    if plan["classification"] != "preregistered_fresh_development_validation_not_paper_evidence":
        raise ValueError("recovery is restricted to the frozen development validation")
    if args.index != 3 or args.index >= len(plan["jobs"]):
        raise ValueError("only the registered output-preserving candidate is recoverable")
    job = plan["jobs"][args.index]
    if job["effective_method"] != "mosaickv_exact__output_preserving_v1__draft_gate":
        raise ValueError("unexpected candidate effective method")
    source = Path(__file__).resolve().parents[2]
    observed_sha = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    observed_dirty = subprocess.run(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if observed_sha != args.corrective_git_sha or observed_dirty:
        raise ValueError("recovery requires its clean immutable corrective source")
    if source_digest(source) != args.source_content_sha256:
        raise ValueError("corrective source snapshot changed")
    config = load_config(job["config"])
    if config_sha256(config) != job["config_sha256"]:
        raise ValueError("config hash mismatch")
    ledger_path = Path(plan["ledger"])
    if digest(ledger_path) != plan["ledger_sha256"]:
        raise ValueError("ledger hash mismatch")
    task = default_task_registry().resolve("coco_caption")
    ledger = load_paper_native_ledger(
        ledger_path,
        expected_sha256=plan["ledger_sha256"],
        task=task,
        dataset_revision=plan["dataset_revision"],
    )
    expected_ids = tuple(plan["sample_ids"])
    if tuple(sample.sample_id for sample in ledger.samples) != expected_ids:
        raise ValueError("ledger identities changed")

    output = Path(job["output"])
    failed_worker_path = _one(
        [
            path
            for path in sorted(output.glob("attempt-*/worker.json"))
            if json.loads(path.read_text()).get("status") == "failed"
        ],
        "failed worker",
    )
    failed_worker = json.loads(failed_worker_path.read_text())
    if failed_worker.get("error") != EXPECTED_ERROR:
        raise ValueError("worker did not fail at the registered post-processing boundary")
    if failed_worker.get("source_base_git_sha") != plan["source_base_git_sha"]:
        raise ValueError("failed inference source does not match the plan")
    old_attempt = failed_worker_path.parent
    old_raw = _one(
        list((old_attempt / "measurement").glob("quality-*/results.jsonl")),
        "raw result",
    )
    old_traces = sorted((old_attempt / "measurement/traces").rglob("*.json"))
    old_rows = load_jsonl(old_raw)
    old_ids = {row.sample_id for row in old_rows}
    if len(old_rows) != len(expected_ids) or old_ids != set(expected_ids):
        raise ValueError("completed inference rows do not match the registered cohort")
    if any(
        row.status.value != "completed"
        or row.error is not None
        or row.method != job["effective_method"]
        for row in old_rows
    ):
        raise ValueError("raw inference contains failed rows or an unexpected method")
    trace_by_sample: dict[str, str] = {}
    for path in old_traces:
        payload = json.loads(path.read_text())
        if payload.get("status") != "completed":
            continue
        sample_id = payload.get("sample_id")
        if sample_id in trace_by_sample:
            raise ValueError(f"duplicate completed trace: {sample_id}")
        if payload.get("effective_method") != job["effective_method"]:
            raise ValueError("trace effective method differs from the registered candidate")
        trace_by_sample[str(sample_id)] = str(path)
    if set(trace_by_sample) != set(expected_ids):
        raise ValueError("completed inference is missing registered traces")

    if torch.cuda.device_count() != 1:
        raise RuntimeError("recovery requires one A100 for faithful hardware provenance")
    attempt = output / f"attempt-recovery-{os.getenv('SLURM_JOB_ID', 'local')}-{time.time_ns()}"
    run_id = old_rows[0].run_id
    run_root = attempt / "measurement" / run_id
    raw_path = run_root / "results.jsonl"
    parquet_path = run_root / "aggregate.parquet"
    manifest_path = run_root / "manifest.json"
    metric_path = run_root / "metrics.jsonl"
    observations_path = run_root / "observations.parquet"
    metrics_path = run_root / "metrics.parquet"
    store = JsonlResultStore(raw_path)
    for row in old_rows:
        store.append(replace(row, manifest_path=str(manifest_path)))

    replay = _ReplayOnlyModel(
        model_id=config.model.id,
        backend=config.execution.backend.value,
        method=config.method.value,
        retention_ratio=config.cache.retention_ratio,
        config=config,
    )
    generation_sha = sha256_text(
        json.dumps(asdict(config.generation), sort_keys=True, separators=(",", ":"), default=str)
    )
    corrective_sha = args.corrective_git_sha
    summary = run_materialized_paper_native_evaluation(
        run_id=run_id,
        task=task,
        ledger=ledger,
        model=replay,
        raw_output=raw_path,
        parquet_output=parquet_path,
        manifest_path=str(manifest_path),
        metric_output=metric_path,
        metric_observations_parquet_output=observations_path,
        metric_aggregates_parquet_output=metrics_path,
        seed=config.execution.seed,
        subset_size=None,
        evaluator_commit=corrective_sha,
        model_revision=config.model.revision,
        tokenizer_revision=config.model.revision,
        generation_config_sha256=generation_sha,
        result_category=ResultCategory.UNIFIED_COMPARISON,
    )
    recovered_rows = load_jsonl(raw_path)
    if any(
        replace(recovered, manifest_path=original.manifest_path) != original
        for original, recovered in zip(old_rows, recovered_rows, strict=True)
    ):
        raise ValueError("recovery changed a measured inference field")
    artifacts = ArtifactProvenance(
        raw_output_sha=sha256_bytes(raw_path.read_bytes()),
        metrics_sha=sha256_bytes(parquet_path.read_bytes()),
        log_sha=_trace_sha(old_attempt / "measurement/traces", run_id, raw_path),
    )
    RunManifestWriter(repo_root=source).write(
        manifest_path,
        config,
        MeasurementType.METHOD,
        _input_provenance(config, task.name, raw_path, input_binding=plan["ledger_sha256"]),
        artifacts,
        run_id=run_id,
        execution_metadata={
            "postprocessing_recovery": True,
            "inference_source_git_sha": plan["source_base_git_sha"],
            "inference_raw_sha256": sha256_bytes(old_raw.read_bytes()),
            "failed_worker_sha256": sha256_bytes(failed_worker_path.read_bytes()),
        },
        result_category=ResultCategory.UNIFIED_COMPARISON,
        evaluation_provenance=_paper_evaluation_provenance(
            None, task, config.method.value, recovered_rows, summary
        ),
    )
    frame = pd.read_parquet(parquet_path)
    if len(frame) != len(expected_ids) or set(frame["sample_id"].astype(str)) != set(expected_ids):
        raise ValueError("recovered aggregate is incomplete")
    record_rows = _record_rows(frame)
    validate_caption_rows(record_rows, json.loads(ledger_path.read_text())["samples"], 64)
    recovery = {
        "schema_version": 1,
        "recovery_kind": "postprocessing_only_no_gpu_inference",
        "original_failed_worker": str(failed_worker_path),
        "original_failed_worker_sha256": sha256_bytes(failed_worker_path.read_bytes()),
        "original_raw_output": str(old_raw),
        "original_raw_output_sha256": sha256_bytes(old_raw.read_bytes()),
        "recovered_raw_output_sha256": sha256_bytes(raw_path.read_bytes()),
        "corrective_git_sha": corrective_sha,
        "measured_fields_changed": False,
    }
    write_once(attempt / "recovery.json", recovery)
    worker = {
        "status": "passed",
        "classification": plan["classification"],
        "heldout_claim": False,
        "plan_sha256": args.plan_sha256,
        "source_base_git_sha": plan["source_base_git_sha"],
        "source_content_sha256": plan["source_content_sha256"],
        "source_worktree_dirty": False,
        "postprocessing_source_git_sha": corrective_sha,
        "postprocessing_source_content_sha256": args.source_content_sha256,
        "postprocessing_recovery": recovery,
        "index": args.index,
        "job": job,
        "gpu_before": failed_worker["gpu_before"],
        "gpu_after": gpu_snapshot(torch),
        "record": {
            "phase": "quality",
            "repetition": 0,
            "order": args.index,
            "method": config.method.value,
            "run_id": run_id,
            "rows": record_rows,
            "metric_observations_parquet_output": str(observations_path),
            "metric_aggregates_parquet_output": str(metrics_path),
        },
        "traces": trace_by_sample,
        "worker_wall_seconds": failed_worker["worker_wall_seconds"],
        "postprocessing_recovery_wall_seconds": time.monotonic() - started,
    }
    write_once(attempt / "worker.json", worker)
    print(json.dumps(worker, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--index", type=int, default=3)
    parser.add_argument("--corrective-git-sha", required=True)
    parser.add_argument("--source-content-sha256", required=True)
    recover(parser.parse_args())
