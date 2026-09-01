#!/usr/bin/env python3
"""Materialize the exact 9,000-sample baseline-extension shard ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

TASKS: dict[str, dict[str, Any]] = {
    "mmstar": {
        "dataset_id": "Lin-Chen/MMStar",
        "dataset_revision": "bc98d668301da7b14f648724866e57302778ab27",
        "split": "val",
        "sample_count": 1500,
    },
    "textvqa": {
        "dataset_id": "lmms-lab/textvqa",
        "dataset_revision": "9c0699cd19768ac5ab97568f6b3cbac4c0062884",
        "split": "validation",
        "sample_count": 3000,
    },
    "docvqa": {
        "dataset_id": "lmms-lab/DocVQA",
        "dataset_revision": "539088ef8a8ada01ac8e2e6d4e372586748a265e",
        "split": "validation",
        "sample_count": 4500,
    },
}
SOURCE_METHODS_BY_BACKEND = {
    "hf_eager": frozenset({"full_kv", "prompt_attention_topk", "mosaickv_full"}),
    "vllm": frozenset({"full_kv", "prompt_attention_topk", "mosaickv_exact"}),
}
SOURCE_BACKENDS = frozenset(SOURCE_METHODS_BY_BACKEND)
BASELINES = ("lookm_reimpl", "prefixkv_reimpl", "vl_cache_reimpl")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_once(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    if path.exists():
        if path.read_bytes() != encoded:
            raise FileExistsError(f"refusing to replace immutable campaign input: {path}")
        return hashlib.sha256(encoded).hexdigest()
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(encoded).hexdigest()


def prepare(
    source: Path,
    output_root: Path,
    *,
    source_git_sha: str,
    shard_size: int,
) -> dict[str, Any]:
    if len(source_git_sha) != 40 or any(char not in "0123456789abcdef" for char in source_git_sha):
        raise ValueError("source_git_sha must be a lowercase 40-character Git SHA")
    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    frame = pd.read_parquet(source)
    required = {"task", "sample_id", "backend_scope", "protocol_method", "status"}
    if missing := required - set(frame.columns):
        raise ValueError(f"source composite is missing columns: {sorted(missing)}")
    if set(frame["status"].astype(str)) != {"completed"}:
        raise ValueError("source composite contains non-completed rows")
    observed_tasks = set(frame["task"].astype(str))
    if observed_tasks != set(TASKS):
        raise ValueError(f"source task set changed: {sorted(observed_tasks)}")

    jobs: list[dict[str, Any]] = []
    source_sha = _sha(source)
    for task, route in TASKS.items():
        task_frame = frame[frame["task"].astype(str).eq(task)]
        if set(task_frame["backend_scope"].astype(str)) != SOURCE_BACKENDS:
            raise ValueError(f"{task} source backend set changed")
        for backend, methods in SOURCE_METHODS_BY_BACKEND.items():
            backend_frame = task_frame[task_frame["backend_scope"].astype(str).eq(backend)]
            if set(backend_frame["protocol_method"].astype(str)) != methods:
                raise ValueError(f"{task}/{backend} source method set changed")
        counts = task_frame.groupby("sample_id", sort=False).size()
        if len(counts) != int(route["sample_count"]) or set(counts.astype(int)) != {6}:
            raise ValueError(f"{task} does not contain the expected six paired source rows")
        sample_ids = tuple(str(value) for value in counts.index)
        for shard_index in range(math.ceil(len(sample_ids) / shard_size)):
            members = sample_ids[shard_index * shard_size : (shard_index + 1) * shard_size]
            ledger = {
                "schema_version": 2,
                "task": task,
                "dataset_id": route["dataset_id"],
                "dataset_revision": route["dataset_revision"],
                "split": route["split"],
                "selection_seed": 20271001,
                "sample_count": len(members),
                "sample_ids": list(members),
                "source_composite": str(source.resolve()),
                "source_composite_sha256": source_sha,
                "source_git_sha": source_git_sha,
                "classification": "canonical_published_baseline_extension_to_existing_9000",
            }
            ledger_path = output_root / "inputs" / task / f"{shard_index:03d}.json"
            ledger_sha = _write_once(ledger_path, ledger)
            job = {
                "schema_version": 1,
                "array_index": len(jobs),
                "task": task,
                "shard_index": shard_index,
                "shard_count": math.ceil(len(sample_ids) / shard_size),
                "sample_count": len(members),
                "sample_ledger_path": str(ledger_path.resolve()),
                "sample_ledger_sha256": ledger_sha,
                "dataset_revision": route["dataset_revision"],
                "source_git_sha": source_git_sha,
                "methods": list(BASELINES),
                "backends": ["hf_eager", "vllm"],
                "retention_ratio": 0.13,
                "output_length": 64,
                "seed": 20271001,
                "hf_block_size": 1,
                "vllm_block_size": 16,
                "classification": "canonical_published_baseline_extension_to_existing_9000",
            }
            job_path = output_root / "inputs" / "jobs" / f"{len(jobs):04d}.json"
            job["job_spec_path"] = str(job_path.resolve())
            job_sha = _write_once(job_path, job)
            job["job_spec_sha256"] = job_sha
            jobs.append(job)

    expected_jobs = sum(
        math.ceil(int(route["sample_count"]) / shard_size) for route in TASKS.values()
    )
    if len(jobs) != expected_jobs:
        raise RuntimeError("campaign job count is inconsistent")
    protocol = {
        "schema_version": 1,
        "experiment_id": "vllm_published_baseline_full_9000_v1",
        "classification": "canonical_published_baseline_extension_to_existing_9000",
        "source_git_sha": source_git_sha,
        "source_composite": str(source.resolve()),
        "source_composite_sha256": source_sha,
        "model_id": "llava-hf/llava-1.5-7b-hf",
        "model_revision": "b234b804b114d9e37bb655e11cbbb5f5e971b7a9",
        "precision": "fp16",
        "attention_backend": "eager",
        "methods": list(BASELINES),
        "backends": ["hf_eager", "vllm"],
        "retention_ratio": 0.13,
        "output_length": 64,
        "seed": 20271001,
        "shard_size": shard_size,
        "job_count": len(jobs),
        "maximum_concurrent_a100s": 8,
        "cpus_per_job": 12,
        "time_limit": "02:00:00",
        "reuse": {
            "methods": ["prompt_attention_topk", "mosaickv_exact"],
            "reason": "reuse exact paired 9,000-sample measured rows; do not rerun",
        },
    }
    protocol_sha = _write_once(output_root / "inputs" / "protocol.json", protocol)
    index_payload = {
        "schema_version": 1,
        "protocol_sha256": protocol_sha,
        "job_count": len(jobs),
        "jobs": jobs,
    }
    _write_once(output_root / "inputs" / "job_index.json", index_payload)
    return protocol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-git-sha", required=True)
    parser.add_argument("--shard-size", type=int, default=64)
    args = parser.parse_args()
    payload = prepare(
        args.source.resolve(),
        args.output_root.resolve(),
        source_git_sha=args.source_git_sha,
        shard_size=args.shard_size,
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
