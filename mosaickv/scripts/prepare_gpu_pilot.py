#!/usr/bin/env python3
"""Materialize the diagnostic Qwen2.5-VL-3B pilot as immutable task configs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, cast

import yaml

from mosaickv.config import canonical_config_json, config_sha256
from mosaickv.experiment_matrix import TASK_CATALOG, _base_config

MODEL_KEY = "qwen2_5_vl_3b"
BACKEND_KEY = "hf_eager"
TASK_KEYS = ("mmstar", "textvqa", "chartqa")
METHODS = (
    "full_kv",
    "random_kv",
    "prompt_attention_topk",
    "prefixkv_reimpl",
    "vl_cache_reimpl",
    "mosaickv_exact",
    "mosaickv_proto",
    "mosaickv_full",
)
RATIOS = (1.0, 0.5, 0.2, 0.1)
SEED = 17
SUBSET_SIZE = 20
OUTPUT_LENGTH = 32


def _write_once(path: Path, payload: bytes, mode: int = 0o444) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable pilot artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
        raise


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _spec() -> dict[str, object]:
    return {
        "schema_version": 1,
        "pilot_revision": 5,
        "classification": "diagnostic_gpu_pilot_not_paper_result",
        "model_key": MODEL_KEY,
        "backend_key": BACKEND_KEY,
        "task_keys": TASK_KEYS,
        "methods": METHODS,
        "retention_ratios": RATIOS,
        "seed": SEED,
        "subset_size": SUBSET_SIZE,
        "output_length": OUTPUT_LENGTH,
        "ordinary_generate_reference": True,
        "excluded": {
            "lookm_reimpl": (
                "The paper-faithful LOOK-M path requires LLaVA-1.5 multi-head attention; "
                "Qwen2.5-VL uses grouped-query attention. The runtime rejects this combination."
            )
        },
        "expected_safety_fallbacks": {
            "mosaickv_proto": "Qwen cached keys are post-RoPE; prototype merging is disabled",
            "mosaickv_full": (
                "Qwen cached keys are post-RoPE and residual repair capability is disabled"
            ),
        },
        "expected_effective_methods": {
            "prefixkv_reimpl": ["generalized_prefixkv_reimpl"],
            "mosaickv_proto": [
                "mosaickv_proto__retention_one_exact",
                "mosaickv_proto__mosaickv_exact_safety_fallback",
            ],
            "mosaickv_full": [
                "mosaickv_full__retention_one_exact",
                "mosaickv_full__mosaickv_exact_safety_fallback",
            ],
        },
        "primary_budget": "packed_prefill_active_kv_bytes",
        "budget_unit": "bytes",
        "block_size": 1,
    }


def materialize(destination: Path, output_root: Path) -> Path:
    spec = _spec()
    spec_bytes = _json_bytes(spec)
    spec_sha = hashlib.sha256(spec_bytes).hexdigest()
    if destination.exists():
        manifest = json.loads((destination / "pilot_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("spec_sha") != spec_sha:
            raise RuntimeError("existing pilot expansion was generated from a different spec")
        return destination / "jobs.jsonl"

    destination.mkdir(parents=True, exist_ok=False)
    _write_once(destination / "pilot_spec.json", spec_bytes)
    jobs: list[dict[str, Any]] = []
    config_counter = 0
    for task_index, task_key in enumerate(TASK_KEYS):
        cells: list[dict[str, object]] = []
        for method in METHODS:
            ratios = (1.0,) if method == "full_kv" else RATIOS
            for ratio in ratios:
                config = _base_config(
                    model_key=MODEL_KEY,
                    backend_key=BACKEND_KEY,
                    method_name=method,
                    task_key=task_key,
                    retention_ratio=ratio,
                    output_length=OUTPUT_LENGTH,
                    seed=SEED,
                )
                digest = config_sha256(config)
                config_path = destination / "configs" / f"{config_counter:04d}-{digest}.yaml"
                payload = cast("dict[str, object]", json.loads(canonical_config_json(config)))
                _write_once(
                    config_path,
                    yaml.safe_dump(payload, sort_keys=True).encode(),
                )
                ratio_label = str(ratio).replace(".", "p")
                cells.append(
                    {
                        "config_path": str(config_path.resolve()),
                        "config_sha": digest,
                        "requested_method": method,
                        "retention_ratio": ratio,
                        "role": "primary",
                        "run_id": (f"gpu-pilot-{spec_sha[:10]}-{task_key}-{method}-{ratio_label}"),
                    }
                )
                config_counter += 1
        if task_key == "mmstar":
            primary = next(
                cell
                for cell in cells
                if cell["requested_method"] == "random_kv" and cell["retention_ratio"] == 0.5
            )
            duplicate = dict(primary)
            duplicate["role"] = "random_reproducibility_duplicate"
            duplicate["run_id"] = f"{primary['run_id']}-repeat"
            cells.append(duplicate)
        task = TASK_CATALOG[task_key]
        jobs.append(
            {
                "array_index": task_index,
                "backend": BACKEND_KEY,
                "pilot_revision": spec["pilot_revision"],
                "cells": cells,
                "dataset_id": task.dataset_id,
                "dataset_revision": task.revision,
                "output_root": str(output_root.resolve()),
                "seed": SEED,
                "subset_size": SUBSET_SIZE,
                "task": task.task_name,
                "task_key": task_key,
                "trace_directory": str((output_root / "traces").resolve()),
            }
        )
    index_payload = b"".join(_json_bytes(row) for row in jobs)
    _write_once(destination / "jobs.jsonl", index_payload)
    _write_once(
        destination / "pilot_manifest.json",
        _json_bytes(
            {
                "classification": spec["classification"],
                "config_count": config_counter,
                "index_sha": hashlib.sha256(index_payload).hexdigest(),
                "job_count": len(jobs),
                "output_root": str(output_root.resolve()),
                "pilot_revision": spec["pilot_revision"],
                "spec_sha": spec_sha,
            }
        ),
    )
    return destination / "jobs.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("/scratch/djy8hg/runs/mosaickv/gpu-pilot-matrix-v5"),
    )
    parser.add_argument(
        "--run-output-root",
        type=Path,
        default=Path("/scratch/djy8hg/runs/mosaickv/gpu-pilot-v5"),
    )
    args = parser.parse_args()
    index = materialize(args.output_directory.resolve(), args.run_output_root.resolve())
    print(str(index.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
