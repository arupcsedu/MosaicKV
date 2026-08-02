#!/usr/bin/env python3
"""Run all Experiment-4 repair variants for one task with one model load."""

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
        raise RuntimeError("repair index SHA does not match its immutable manifest")
    if not 0 <= array_index < len(rows) or rows[array_index].get("array_index") != array_index:
        raise IndexError(f"repair array index {array_index} is invalid")
    return cast("dict[str, Any]", rows[array_index]), cast("dict[str, Any]", manifest)


def run(index: Path, array_index: int) -> int:
    job, matrix_manifest = _job(index, array_index)
    cells = cast("list[dict[str, Any]]", job["cells"])
    configs = [load_config(cell["config_path"]) for cell in cells]
    for cell, config in zip(cells, configs, strict=True):
        if config_sha256(config) != cell["config_sha"]:
            raise RuntimeError(f"config SHA mismatch: {cell['config_path']}")
        if config.dataset.revision != job["dataset_revision"]:
            raise RuntimeError("worker config dataset revision mismatch")
    first = configs[0]
    adapter = load_hf_adapter(
        first.model.id,
        revision=first.model.revision,
        model_kwargs=_model_kwargs(first, local_files_only=True),
        processor_kwargs={"local_files_only": True},
    )
    output_root = Path(job["output_root"])
    errors: list[dict[str, str]] = []
    summaries: list[dict[str, object]] = []
    for cell, config in zip(cells, configs, strict=True):
        args = argparse.Namespace(
            task=job["task"],
            run_id=cell["run_id"],
            output_dir=job["output_root"],
            raw_output=None,
            parquet_output=None,
            manifest=None,
            trace_directory=job["trace_directory"],
            subset_size=None,
            full_dataset=False,
            sample_offset=0,
            sample_id_file=job["sample_id_file"],
            sample_id_file_sha=job["sample_id_file_sha"],
            local_files_only=True,
        )
        try:
            status, summary = run_hf_evaluation(args, config, adapter=adapter)
            summary.update(
                {
                    "variant_id": str(cell["variant_id"]),
                    "variant": str(cell["variant"]),
                    "config_sha": str(cell["config_sha"]),
                }
            )
            summaries.append(cast("dict[str, object]", summary))
            if status:
                errors.append({"run_id": str(cell["run_id"]), "error": f"status {status}"})
        except Exception as error:
            errors.append(
                {"run_id": str(cell["run_id"]), "error": f"{type(error).__name__}: {error}"}
            )
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    record = {
        "array_index": array_index,
        "classification": matrix_manifest["classification"],
        "errors": errors,
        "summaries": summaries,
        "task": job["task"],
    }
    path = output_root / "workers" / f"{array_index:02d}-{job['task_key']}.json"
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
