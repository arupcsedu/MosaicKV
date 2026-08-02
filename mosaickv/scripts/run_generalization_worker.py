#!/usr/bin/env python3
"""Run one Experiment-5 calibration, model-family, or backend-parity job."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import torch

from mosaickv.adapters.huggingface import load_hf_adapter
from mosaickv.config import config_sha256, load_config
from mosaickv.hf_cli import _model_kwargs, run_hf_evaluation


def _job(index: Path, array_index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines() if line]
    manifest = json.loads((index.parent / "manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256(index.read_bytes()).hexdigest() != manifest.get("index_sha"):
        raise RuntimeError("Experiment-5 index SHA does not match its immutable manifest")
    if not 0 <= array_index < len(rows) or rows[array_index].get("array_index") != array_index:
        raise IndexError(f"Experiment-5 array index {array_index} is invalid")
    return cast("dict[str, Any]", rows[array_index]), cast("dict[str, Any]", manifest)


def _task_record(job: dict[str, Any], task_key: str) -> dict[str, Any]:
    if "tasks" in job:
        matches = [
            item
            for item in cast("list[dict[str, Any]]", job["tasks"])
            if item["task_key"] == task_key
        ]
        if len(matches) != 1:
            raise RuntimeError(f"job has no unique task record for {task_key}")
        return matches[0]
    if job.get("task_key") != task_key:
        raise RuntimeError(f"job task {job.get('task_key')} does not match cell task {task_key}")
    return job


def run(index: Path, array_index: int) -> int:
    job, matrix_manifest = _job(index, array_index)
    cells = cast("list[dict[str, Any]]", job["cells"])
    configs = [load_config(cell["config_path"]) for cell in cells]
    for cell, config in zip(cells, configs, strict=True):
        if config_sha256(config) != cell["config_sha"]:
            raise RuntimeError(f"config SHA mismatch: {cell['config_path']}")
        task = _task_record(job, str(cell["task_key"]))
        if config.dataset.revision != task["dataset_revision"]:
            raise RuntimeError("worker config dataset revision mismatch")
        if config.diagnostics.enabled:
            if tuple(config.diagnostics.sample_ids) != tuple(task["sample_ids"]):
                raise RuntimeError("worker diagnostics and immutable sample ledger differ")
            if config.diagnostics.sample_ledger_sha != task["sample_ledger_sha"]:
                raise RuntimeError("worker diagnostics ledger SHA mismatch")
    first = configs[0]
    adapter = load_hf_adapter(
        first.model.id,
        revision=first.model.revision,
        model_kwargs=_model_kwargs(first, local_files_only=True),
        processor_kwargs={"local_files_only": True},
    )
    if any(config.model != first.model for config in configs):
        raise RuntimeError("one Experiment-5 worker may load only one pinned model")
    output_root = Path(job["output_root"])
    errors: list[dict[str, str]] = []
    summaries: list[dict[str, object]] = []
    for cell, config in zip(cells, configs, strict=True):
        task = _task_record(job, str(cell["task_key"]))
        args = argparse.Namespace(
            task=task["task"],
            run_id=cell["run_id"],
            output_dir=job["output_root"],
            raw_output=None,
            parquet_output=None,
            manifest=None,
            trace_directory=job["trace_directory"],
            subset_size=None,
            full_dataset=False,
            sample_offset=0,
            sample_id_file=task["sample_ledger_path"],
            sample_id_file_sha=task["sample_ledger_sha"],
            local_files_only=True,
        )
        try:
            # A preceding hybrid cell may leave the shared model on its decode
            # implementation. Restore the declared prefill implementation at
            # every immutable cell boundary.
            adapter.set_attention_implementation(config.execution.attention_implementation)
            status, summary = run_hf_evaluation(args, config, adapter=adapter)
            summary.update(
                {
                    "cell_id": str(cell["cell_id"]),
                    "role": str(cell["role"]),
                    "config_sha": str(cell["config_sha"]),
                    "job_kind": str(job["job_kind"]),
                }
            )
            summaries.append(cast("dict[str, object]", summary))
            if status:
                errors.append(
                    {
                        "run_id": str(cell["run_id"]),
                        "error": f"evaluation returned status {status}",
                    }
                )
        except Exception as error:
            errors.append(
                {
                    "run_id": str(cell["run_id"]),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    record = {
        "array_index": array_index,
        "classification": matrix_manifest["classification"],
        "errors": errors,
        "job_kind": job["job_kind"],
        "model_key": job["model_key"],
        "summaries": summaries,
    }
    path = output_root / "workers" / f"{array_index:02d}-{job['job_kind']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    return int(bool(errors))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--array-index", required=True, type=int)
    args = parser.parse_args()
    return run(args.index.resolve(), args.array_index)


if __name__ == "__main__":
    raise SystemExit(main())
