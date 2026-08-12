#!/usr/bin/env python3
"""Run one mixed-task continuation shard queue while reusing one LLaVA model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, cast

import torch

from mosaickv.adapters.huggingface import load_hf_adapter
from mosaickv.config import config_sha256, load_config
from mosaickv.hf_cli import _model_kwargs, run_hf_evaluation


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _load_job(index: Path, worker_index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines() if line]
    manifest = json.loads((index.parent / "continuation_manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256(index.read_bytes()).hexdigest() != manifest.get("index_sha256"):
        raise RuntimeError("continuation index SHA mismatch")
    if not 0 <= worker_index < len(rows) or rows[worker_index]["worker_index"] != worker_index:
        raise IndexError(f"invalid continuation worker index: {worker_index}")
    return cast("dict[str, Any]", rows[worker_index]), cast("dict[str, Any]", manifest)


def run(index: Path, worker_index: int) -> int:
    job, manifest = _load_job(index, worker_index)
    cells = cast("list[dict[str, Any]]", job["cells"])
    if not cells:
        raise RuntimeError("continuation worker has an empty queue")
    config_cache: dict[str, Any] = {}
    for cell in cells:
        path = str(cell["config_path"])
        if path not in config_cache:
            config_cache[path] = load_config(path)
        config = config_cache[path]
        if config_sha256(config) != cell["config_sha"]:
            raise RuntimeError(f"continuation config SHA mismatch: {path}")
    first = config_cache[str(cells[0]["config_path"])]
    for config in config_cache.values():
        if (config.model.id, config.model.revision) != (first.model.id, first.model.revision):
            raise RuntimeError("one continuation worker cannot mix model checkpoints")

    adapter = load_hf_adapter(
        first.model.id,
        revision=first.model.revision,
        model_kwargs=_model_kwargs(first, local_files_only=True),
        processor_kwargs={"local_files_only": True},
    )
    output_root = Path(str(job["output_root"])).resolve()
    record_path = output_root / "workers" / f"worker-{worker_index:02d}.json"
    errors: list[dict[str, str]] = []
    completed: list[dict[str, object]] = []

    def checkpoint(status: str) -> None:
        processed = {str(item["run_id"]) for item in completed} | {
            str(item["run_id"]) for item in errors
        }
        _atomic_json(
            record_path,
            {
                "schema_version": 1,
                "classification": manifest["classification"],
                "worker_index": worker_index,
                "status": status,
                "queue_size": len(cells),
                "processed_shards": len(processed),
                "completed": completed,
                "errors": errors,
            },
        )

    checkpoint("model_loaded")
    for cell in cells:
        config = config_cache[str(cell["config_path"])]
        args = argparse.Namespace(
            task=cell["task"],
            run_id=cell["run_id"],
            output_dir=str(output_root),
            raw_output=None,
            parquet_output=None,
            manifest=None,
            trace_directory=job["trace_directory"],
            subset_size=None,
            sample_id_file=cell["sample_id_file"],
            sample_id_file_sha=cell["sample_id_file_sha"],
            full_dataset=False,
            sample_offset=0,
            local_files_only=True,
        )
        try:
            status, summary = run_hf_evaluation(args, config, adapter=adapter)
            completed.append(
                {
                    "run_id": cell["run_id"],
                    "logical_run_id": cell["logical_run_id"],
                    "shard_index": cell["shard_index"],
                    "status": status,
                    "summary": summary,
                }
            )
            if status:
                errors.append(
                    {
                        "run_id": str(cell["run_id"]),
                        "error": f"evaluation returned status {status}",
                    }
                )
        except Exception as error:  # preserve every failure and continue independent shards
            errors.append(
                {
                    "run_id": str(cell["run_id"]),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        finally:
            checkpoint("running")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    checkpoint("completed" if not errors else "completed_with_errors")
    return int(bool(errors))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--worker-index", required=True, type=int)
    args = parser.parse_args()
    status = run(args.index.resolve(), args.worker_index)
    if not math.isfinite(float(status)):
        raise RuntimeError("invalid worker status")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
