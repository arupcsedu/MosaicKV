#!/usr/bin/env python3
"""Materialize a six-GPU sharded continuation of the timed-out LLaVA campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, cast

import yaml
from prepare_llava_full_quality_8gpu import (
    _canonical_bytes,
    _source_provenance,
    _write_once,
)

from mosaickv.config import config_sha256, load_config
from mosaickv.evaluation.storage import load_jsonl


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_spec(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("continuation spec must be a schema-version-1 mapping")
    if value.get("workers") != 6 or value.get("gpus") != 6:
        raise ValueError("continuation requires exactly six workers and six GPUs")
    shard_size = value.get("shard_size")
    sample_count = value.get("samples_per_logical_cell")
    if not isinstance(shard_size, int) or not isinstance(sample_count, int):
        raise ValueError("shard and logical sample counts must be integers")
    if shard_size < 1 or sample_count < 1 or sample_count % shard_size:
        raise ValueError("logical sample count must be divisible by a positive shard size")
    return cast("dict[str, Any]", value)


def _assert_parent_terminal(spec: dict[str, Any]) -> None:
    parent = cast("dict[str, Any]", spec["parent"])
    job_id = str(parent["slurm_job_id"])
    output = subprocess.run(
        ["sacct", "-j", job_id, "-X", "-n", "-P", "-o", "State"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    states = [line.split("|", 1)[0].split("+", 1)[0] for line in output.splitlines() if line]
    if not states or states[0] != parent["terminal_state"]:
        raise RuntimeError(f"parent Slurm job is not in the expected terminal state: {states}")


def _behavior_hashes(spec: dict[str, Any]) -> dict[str, str]:
    repository = Path(__file__).resolve().parents[1]
    identity = cast("dict[str, str]", spec["runtime_identity"])
    paths = {
        "hf_runtime_sha256": repository / "src/mosaickv/backends/hf_runtime.py",
        "config_sha256": repository / "src/mosaickv/config.py",
    }
    observed = {name: _sha(path.read_bytes()) for name, path in paths.items()}
    if observed != identity:
        raise RuntimeError(
            "behavior-affecting source changed since the parent campaign: "
            f"expected={identity}, observed={observed}"
        )
    return observed


def _parent_jobs(spec: dict[str, Any]) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    parent = cast("dict[str, Any]", spec["parent"])
    index = Path(str(parent["index"])).resolve()
    payload = index.read_bytes()
    if _sha(payload) != parent["expected_index_sha256"]:
        raise RuntimeError("parent index SHA does not match the continuation spec")
    manifest = json.loads((index.parent / "pilot_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("index_sha") != _sha(payload):
        raise RuntimeError("parent index and manifest disagree")
    jobs = [json.loads(line) for line in payload.decode().splitlines() if line]
    return index, cast("list[dict[str, Any]]", jobs), cast("dict[str, Any]", manifest)


def _sample_ids(job: dict[str, Any]) -> tuple[str, ...]:
    path = Path(str(job["sample_id_file"])).resolve()
    payload = path.read_bytes()
    if _sha(payload) != job["sample_id_file_sha"]:
        raise RuntimeError(f"parent sample ledger SHA mismatch: {path}")
    value = json.loads(payload)
    ids = value.get("sample_ids")
    if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
        raise RuntimeError(f"invalid parent sample ledger: {path}")
    result = tuple(cast("list[str]", ids))
    if len(result) != len(set(result)):
        raise RuntimeError(f"duplicate IDs in parent sample ledger: {path}")
    return result


def _reusable_cell(job: dict[str, Any], cell: dict[str, Any], sample_ids: tuple[str, ...]) -> bool:
    root = Path(str(job["output_root"])) / str(cell["run_id"])
    raw_path = root / "results.jsonl"
    manifest_path = root / "manifest.json"
    parquet_path = root / "aggregate.parquet"
    if not (raw_path.is_file() and manifest_path.is_file() and parquet_path.is_file()):
        return False
    rows = load_jsonl(raw_path)
    if len(rows) != len(sample_ids) or {row.sample_id for row in rows} != set(sample_ids):
        return False
    if any(row.status.value != "completed" for row in rows):
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return bool(
        manifest.get("source", {}).get("config_sha") == cell["config_sha"]
        and manifest.get("artifacts", {}).get("raw_output_sha") == _sha(raw_path.read_bytes())
    )


def _trace_totals(trace_root: Path, run_id: str) -> dict[str, float]:
    totals: dict[str, tuple[int, float]] = {}
    for path in (trace_root / run_id).glob("*.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") != "completed":
            continue
        total = value.get("timing_breakdown", {}).get("total")
        sample_id = value.get("sample_id")
        if not isinstance(sample_id, str) or not isinstance(total, int | float):
            continue
        marker = "-attempt-"
        attempt = 0 if marker not in path.stem else int(path.stem.rsplit(marker, 1)[1])
        if sample_id not in totals or attempt > totals[sample_id][0]:
            totals[sample_id] = (attempt, float(total))
    return {sample_id: item[1] for sample_id, item in totals.items()}


def _timing_models(
    parent_jobs: list[dict[str, Any]],
) -> tuple[dict[tuple[str, str, float], dict[str, float]], dict[tuple[str, str, float], float]]:
    per_sample: dict[tuple[str, str, float], dict[str, float]] = {}
    p95: dict[tuple[str, str, float], float] = {}
    for job in parent_jobs:
        trace_root = Path(str(job["trace_directory"])).resolve()
        for cell in cast("list[dict[str, Any]]", job["cells"]):
            key = (
                str(job["task_key"]),
                str(cell["requested_method"]),
                float(cell["retention_ratio"]),
            )
            totals = _trace_totals(trace_root, str(cell["run_id"]))
            if totals:
                per_sample[key] = totals
                ordered = sorted(totals.values())
                p95[key] = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
    return per_sample, p95


def _estimate_sample_seconds(
    *,
    task: str,
    method: str,
    ratio: float,
    sample_id: str,
    per_sample: dict[tuple[str, str, float], dict[str, float]],
    p95: dict[tuple[str, str, float], float],
) -> float:
    key = (task, method, ratio)
    if sample_id in per_sample.get(key, {}):
        return per_sample[key][sample_id]
    if method in {"mosaickv_full", "mosaickv_proto"} and ratio == 0.5:
        return p95.get(key, 1_400.0)
    if method == "mosaickv_proto" and ratio in {0.1, 0.2}:
        proxy = (task, "mosaickv_full", ratio)
        return per_sample.get(proxy, {}).get(sample_id, p95.get(proxy, 60.0))
    if method == "prefixkv_reimpl":
        return 180.0
    if method == "vl_cache_reimpl":
        return 120.0
    if method in {"prompt_attention_topk", "random_kv"}:
        return 45.0
    if method == "full_kv":
        return 5.0
    return p95.get(key, 90.0)


def _balance(cells: list[dict[str, Any]], workers: int) -> list[list[dict[str, Any]]]:
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(workers)]
    totals = [0.0] * workers
    for cell in sorted(
        cells,
        key=lambda item: (-float(item["estimated_seconds"]), str(item["run_id"])),
    ):
        worker = min(range(workers), key=lambda index: (totals[index], index))
        buckets[worker].append(cell)
        totals[worker] += float(cell["estimated_seconds"])
    for bucket in buckets:
        bucket.sort(key=lambda item: (-float(item["estimated_seconds"]), str(item["run_id"])))
    if max(totals) - min(totals) > max(float(cell["estimated_seconds"]) for cell in cells):
        raise RuntimeError(f"continuation worker balance is invalid: {totals}")
    return buckets


def materialize(spec_path: Path, destination: Path) -> Path:
    spec = _load_spec(spec_path)
    spec_payload = spec_path.read_bytes()
    spec_sha = _sha(spec_payload)
    manifest_path = destination / "continuation_manifest.json"
    index_path = destination / "jobs.jsonl"
    if destination.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("spec_sha256") != spec_sha:
            raise RuntimeError("existing continuation was generated from a different spec")
        if _sha(index_path.read_bytes()) != manifest.get("index_sha256"):
            raise RuntimeError("existing continuation index SHA mismatch")
        return index_path

    _assert_parent_terminal(spec)
    behavior_hashes = _behavior_hashes(spec)
    parent_index, parent_jobs, parent_manifest = _parent_jobs(spec)
    per_sample, p95 = _timing_models(parent_jobs)
    git_sha, git_dirty, source_patch = _source_provenance()
    if not git_dirty:
        raise RuntimeError("continuation is versioned as an exploratory dirty-tree run")
    output_root = Path(str(spec["continuation_output_root"])).resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to reuse continuation output root: {output_root}")

    destination.mkdir(parents=True, exist_ok=False)
    _write_once(destination / "continuation_spec.yaml", spec_payload)
    _write_once(destination / "source.patch", source_patch)
    source_patch_sha = _sha(source_patch)
    shard_size = int(spec["shard_size"])
    expected_samples = int(spec["samples_per_logical_cell"])
    shard_count = expected_samples // shard_size
    task_shards: dict[tuple[str, int], tuple[Path, str, tuple[str, ...]]] = {}
    parent_samples: dict[str, tuple[str, ...]] = {}
    for job in parent_jobs:
        task_key = str(job["task_key"])
        ids = _sample_ids(job)
        if len(ids) != expected_samples:
            raise RuntimeError(f"parent task {task_key} does not contain {expected_samples} IDs")
        if task_key in parent_samples and parent_samples[task_key] != ids:
            raise RuntimeError(f"parent workers disagree on {task_key} sample IDs")
        parent_samples[task_key] = ids
    for task_key, ids in sorted(parent_samples.items()):
        source_job = next(job for job in parent_jobs if job["task_key"] == task_key)
        for shard_index in range(shard_count):
            selected = ids[shard_index * shard_size : (shard_index + 1) * shard_size]
            payload = _canonical_bytes(
                {
                    "schema_version": 1,
                    "task": source_job["task"],
                    "dataset_id": source_job["dataset_id"],
                    "dataset_revision": source_job["dataset_revision"],
                    "split": load_config(source_job["cells"][0]["config_path"]).dataset.split,
                    "parent_sample_id_file": source_job["sample_id_file"],
                    "parent_sample_id_file_sha": source_job["sample_id_file_sha"],
                    "shard_index": shard_index,
                    "shard_count": shard_count,
                    "sample_ids": list(selected),
                }
            )
            path = destination / "sample_shards" / task_key / f"{shard_index:02d}.json"
            _write_once(path, payload)
            task_shards[(task_key, shard_index)] = (path.resolve(), _sha(payload), selected)

    reused: list[dict[str, Any]] = []
    continuation_cells: list[dict[str, Any]] = []
    copied_configs: dict[str, Path] = {}
    logical_remaining: set[str] = set()
    for job in parent_jobs:
        task_key = str(job["task_key"])
        ids = parent_samples[task_key]
        for cell in cast("list[dict[str, Any]]", job["cells"]):
            if _reusable_cell(job, cell, ids):
                reused.append(
                    {
                        "logical_run_id": cell["run_id"],
                        "task_key": task_key,
                        "method": cell["requested_method"],
                        "retention_ratio": cell["retention_ratio"],
                        "seed": cell["seed"],
                        "raw_output": str(
                            Path(str(job["output_root"])) / cell["run_id"] / "results.jsonl"
                        ),
                        "manifest": str(
                            Path(str(job["output_root"])) / cell["run_id"] / "manifest.json"
                        ),
                    }
                )
                continue
            logical_id = str(cell["run_id"])
            logical_remaining.add(logical_id)
            config_path = Path(str(cell["config_path"])).resolve()
            config = load_config(config_path)
            if config_sha256(config) != cell["config_sha"]:
                raise RuntimeError(f"parent config SHA mismatch: {config_path}")
            digest = str(cell["config_sha"])
            if digest not in copied_configs:
                copy_path = destination / "configs" / f"{len(copied_configs):03d}-{digest}.yaml"
                _write_once(copy_path, config_path.read_bytes())
                copied_configs[digest] = copy_path.resolve()
            for shard_index in range(shard_count):
                sample_path, sample_sha, selected = task_shards[(task_key, shard_index)]
                estimate = sum(
                    _estimate_sample_seconds(
                        task=task_key,
                        method=str(cell["requested_method"]),
                        ratio=float(cell["retention_ratio"]),
                        sample_id=sample_id,
                        per_sample=per_sample,
                        p95=p95,
                    )
                    for sample_id in selected
                )
                continuation_cells.append(
                    {
                        "config_path": str(copied_configs[digest]),
                        "config_sha": digest,
                        "dataset_id": job["dataset_id"],
                        "dataset_revision": job["dataset_revision"],
                        "estimated_seconds": estimate,
                        "logical_run_id": logical_id,
                        "method": cell["requested_method"],
                        "retention_ratio": cell["retention_ratio"],
                        "run_id": (
                            f"cont6-v1-{logical_id}-shard-{shard_index:02d}-of-{shard_count:02d}"
                        ),
                        "sample_id_file": str(sample_path),
                        "sample_id_file_sha": sample_sha,
                        "seed": cell["seed"],
                        "shard_count": shard_count,
                        "shard_index": shard_index,
                        "task": job["task"],
                        "task_key": task_key,
                    }
                )

    expected_parent = cast("dict[str, Any]", spec["parent"])
    if len(reused) != int(expected_parent["committed_cells"]):
        raise RuntimeError(f"reusable cell count changed: {len(reused)}")
    if len(logical_remaining) != int(expected_parent["uncommitted_cells"]):
        raise RuntimeError(f"remaining logical cell count changed: {len(logical_remaining)}")
    expected_shards = len(logical_remaining) * shard_count
    if len(continuation_cells) != expected_shards:
        raise RuntimeError("continuation shard count mismatch")

    workers = int(spec["workers"])
    buckets = _balance(continuation_cells, workers)
    jobs: list[dict[str, Any]] = []
    for rank, cells in enumerate(buckets):
        jobs.append(
            {
                "worker_index": rank,
                "classification": spec["classification"],
                "cells": cells,
                "output_root": str(output_root),
                "source_git_sha": git_sha,
                "source_git_dirty": git_dirty,
                "source_patch_sha256": source_patch_sha,
                "trace_directory": str(output_root / "traces"),
            }
        )
    index_payload = b"".join(_canonical_bytes(job) for job in jobs)
    _write_once(index_path, index_payload)
    _write_once(destination / "reused_cells.json", _canonical_bytes(reused))
    worker_seconds = [
        sum(float(cell["estimated_seconds"]) for cell in bucket) for bucket in buckets
    ]
    manifest = {
        "schema_version": 1,
        "classification": spec["classification"],
        "spec_sha256": spec_sha,
        "index_sha256": _sha(index_payload),
        "parent_index": str(parent_index),
        "parent_manifest_sha256": _sha((parent_index.parent / "pilot_manifest.json").read_bytes()),
        "parent_source_patch_sha256": parent_manifest["source_patch_sha256"],
        "source_git_sha": git_sha,
        "source_git_dirty": git_dirty,
        "source_patch_sha256": source_patch_sha,
        "behavior_hashes": behavior_hashes,
        "worker_count": workers,
        "shard_size": shard_size,
        "shards_per_logical_cell": shard_count,
        "reused_logical_cells": len(reused),
        "reused_rows": len(reused) * expected_samples,
        "continuation_logical_cells": len(logical_remaining),
        "continuation_shard_runs": len(continuation_cells),
        "continuation_rows": len(logical_remaining) * expected_samples,
        "expected_combined_rows": (len(reused) + len(logical_remaining)) * expected_samples,
        "output_root": str(output_root),
        "worker_shard_counts": [len(bucket) for bucket in buckets],
        "worker_estimated_hours": [seconds / 3600 for seconds in worker_seconds],
        "maximum_estimated_worker_hours": max(worker_seconds) / 3600,
        "maximum_estimated_shard_hours": max(
            float(cell["estimated_seconds"]) for cell in continuation_cells
        )
        / 3600,
    }
    _write_once(manifest_path, _canonical_bytes(manifest))
    return index_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("configs/experiments/llava_full_quality_continuation_6gpu_v1.yaml"),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path(
            "/scratch/djy8hg/runs/mosaickv/campaigns/llava-full-quality-continuation-6gpu-v1"
        ),
    )
    args = parser.parse_args()
    print(str(materialize(args.spec.resolve(), args.output_directory.resolve())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
