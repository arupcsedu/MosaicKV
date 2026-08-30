#!/usr/bin/env python3
"""Create immutable continuation shards after validating the reused V2 prefix."""

from __future__ import annotations

import argparse
import json
import math
import stat
import subprocess
from pathlib import Path
from typing import Any, cast

import pandas as pd  # type: ignore[import-untyped]
from prepare_paper_quality_hf_vllm_v2 import (
    BACKENDS,
    TASKS,
    _canonical_bytes,
    _load_protocol,
    _object,
    _sha,
    _sha_bytes,
    _source_snapshot,
    _write_once,
)

CLASSIFICATION = "fixed_preregistered_paper_quality_memory_continuation"
LEDGER_CLASSIFICATION = "fixed_preregistered_paper_quality_memory"


def _parse_paths(values: list[str], option: str) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for value in values:
        task, separator, path = value.partition("=")
        if not separator or task not in TASKS or not path or task in paths:
            raise ValueError(f"{option} requires one task=/absolute/path for every task")
        paths[task] = Path(path).resolve()
    if set(paths) != set(TASKS):
        raise ValueError(f"{option} requires MMStar, TextVQA, and DocVQA paths")
    return paths


def _read_ledger(path: Path) -> tuple[dict[str, Any], str]:
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise ValueError(f"ledger is not immutable: {path}")
    payload = path.read_bytes()
    value = _object(json.loads(payload), str(path))
    ids = value.get("sample_ids")
    if (
        not isinstance(ids, list)
        or value.get("sample_count") != len(ids)
        or len(ids) != len(set(str(item) for item in ids))
    ):
        raise ValueError(f"ledger sample IDs are invalid: {path}")
    return value, _sha_bytes(payload)


def prepare_continuation(
    *,
    protocol_path: Path,
    full_ledger_paths: dict[str, Path],
    destination: Path,
    output_root: Path,
    repository: Path,
) -> Path:
    source_sha, dirty = _source_snapshot(repository.resolve())
    if dirty:
        raise RuntimeError("continuation preparation requires a clean worktree")
    protocol_path = protocol_path.resolve()
    protocol = _load_protocol(protocol_path, require_execution_authorized=True)
    if protocol.get("experiment_id") != "paper_quality_hf_vllm_llava15_v3_continuation":
        raise ValueError("continuation preparer received the wrong protocol")
    if protocol.get("classification") != CLASSIFICATION:
        raise ValueError("continuation classification changed")
    protocol_sha = _sha(protocol_path)
    experiment = _object(protocol["experiment"], "protocol.experiment")
    reuse = _object(protocol["reuse"], "protocol.reuse")
    candidate = _object(protocol["candidate"], "protocol.candidate")
    candidate_path = (protocol_path.parents[2] / str(candidate["config"])).resolve()
    if _sha(candidate_path) != candidate["config_sha256"]:
        raise ValueError("candidate configuration hash changed")
    if subprocess.run(
        (
            "git",
            "merge-base",
            "--is-ancestor",
            str(candidate["frozen_algorithm_commit"]),
            source_sha,
        ),
        cwd=repository,
        check=False,
    ).returncode:
        raise ValueError("source does not descend from the frozen candidate")

    prior_root = Path(str(reuse["prior_campaign_root"])).resolve()
    prior_index = prior_root / "campaign_inputs/jobs.jsonl"
    prior_report = prior_root / "analysis/report.json"
    prior_rows = prior_root / "analysis/paired_rows.parquet"
    expected_hashes = {
        prior_index: reuse["prior_index_sha256"],
        prior_report: reuse["prior_report_sha256"],
        prior_rows: reuse["prior_paired_rows_sha256"],
    }
    for path, digest in expected_hashes.items():
        if _sha(path) != digest:
            raise ValueError(f"reused campaign artifact hash changed: {path}")
    prior_frame = pd.read_parquet(prior_rows)
    if len(prior_frame) != 4608 or set(prior_frame["status"]) != {"completed"}:
        raise ValueError("reused campaign does not contain 4,608 complete primary rows")

    target_counts = _object(experiment["target_sample_counts"], "target counts")
    continuation_counts = _object(experiment["continuation_sample_counts"], "continuation counts")
    task_contracts = _object(experiment["tasks"], "tasks")
    prior_hashes = _object(reuse["prior_master_ledger_sha256"], "prior ledger hashes")
    reused_count = int(reuse["reused_samples_per_task"])
    shard_size = int(experiment["samples_per_continuation_shard"])
    full_ledgers: dict[str, tuple[dict[str, Any], str]] = {}
    suffixes: dict[str, tuple[str, ...]] = {}
    prior_ledgers: dict[str, tuple[dict[str, Any], str]] = {}
    for task in TASKS:
        prior_path = prior_root / "master_ledgers" / f"{task}.json"
        prior_ledger, prior_sha = _read_ledger(prior_path)
        if prior_sha != prior_hashes[task]:
            raise ValueError(f"reused {task} ledger hash changed")
        full_ledger, full_sha = _read_ledger(full_ledger_paths[task])
        contract = _object(task_contracts[task], task)
        expected = {
            "task": task,
            "dataset_id": contract["dataset_id"],
            "dataset_revision": contract["dataset_revision"],
            "split": contract["split"],
            "classification": LEDGER_CLASSIFICATION,
            "canonical_eligible": True,
            "source_git_sha": source_sha,
            "source_git_dirty": False,
            "selection_freeze_sha256": protocol_sha,
            "sample_count": int(target_counts[task]),
        }
        if any(full_ledger.get(key) != value for key, value in expected.items()):
            raise ValueError(f"full {task} ledger differs from the continuation protocol")
        prior_ids = tuple(str(value) for value in cast("list[str]", prior_ledger["sample_ids"]))
        full_ids = tuple(str(value) for value in cast("list[str]", full_ledger["sample_ids"]))
        if len(prior_ids) != reused_count or full_ids[:reused_count] != prior_ids:
            raise ValueError(f"full {task} ledger does not exactly reuse the V2 prefix")
        suffix = full_ids[reused_count:]
        if len(suffix) != int(continuation_counts[task]) or set(suffix) & set(prior_ids):
            raise ValueError(f"{task} continuation suffix count/overlap is invalid")
        prior_ledgers[task] = (prior_ledger, prior_sha)
        full_ledgers[task] = (full_ledger, full_sha)
        suffixes[task] = suffix

    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"continuation input root already exists: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    shard_records: dict[tuple[str, int], dict[str, Any]] = {}
    shard_counts: dict[str, int] = {}
    for task in TASKS:
        full_ledger, full_sha = full_ledgers[task]
        suffix = suffixes[task]
        shard_count = math.ceil(len(suffix) / shard_size)
        shard_counts[task] = shard_count
        for shard_index in range(shard_count):
            selected = suffix[shard_index * shard_size : (shard_index + 1) * shard_size]
            payload = _canonical_bytes(
                {
                    "schema_version": 1,
                    "classification": CLASSIFICATION,
                    "task": task,
                    "dataset_id": full_ledger["dataset_id"],
                    "dataset_revision": full_ledger["dataset_revision"],
                    "split": full_ledger["split"],
                    "selection_seed": experiment["selection_seed"],
                    "source_git_sha": source_sha,
                    "source_git_dirty": False,
                    "canonical_eligible": True,
                    "selection_freeze_sha256": protocol_sha,
                    "parent_ledger_sha256": full_sha,
                    "reused_prefix_count": reused_count,
                    "shard_index": shard_index,
                    "shard_count": shard_count,
                    "sample_count": len(selected),
                    "sample_ids": list(selected),
                }
            )
            path = destination / "sample_shards" / task / f"{shard_index:03d}.json"
            _write_once(path, payload)
            shard_records[(task, shard_index)] = {
                "path": str(path.resolve()),
                "sha256": _sha_bytes(payload),
                "sample_count": len(selected),
            }

    rows: list[dict[str, Any]] = []
    backends = _object(experiment["backends"], "backends")
    for backend in BACKENDS:
        for task in TASKS:
            contract = _object(task_contracts[task], task)
            for shard_index in range(shard_counts[task]):
                array_index = len(rows)
                shard = shard_records[(task, shard_index)]
                row = {
                    "schema_version": 1,
                    "array_index": array_index,
                    "experiment_id": protocol["experiment_id"],
                    "classification": CLASSIFICATION,
                    "source_git_sha": source_sha,
                    "protocol_path": str(protocol_path),
                    "protocol_sha256": protocol_sha,
                    "candidate_config_path": str(candidate_path),
                    "candidate_config_sha256": candidate["config_sha256"],
                    "backend": backend,
                    "task": task,
                    "dataset_revision": contract["dataset_revision"],
                    "shard_index": shard_index,
                    "shard_count": shard_counts[task],
                    "sample_count": shard["sample_count"],
                    "sample_ledger_path": shard["path"],
                    "sample_ledger_sha256": shard["sha256"],
                    "selection_seed": experiment["selection_seed"],
                    "output_length": experiment["output_length"],
                    "precision": experiment["precision"],
                    "output_root": str(output_root.resolve()),
                    "run_namespace": (
                        f"{protocol['experiment_id']}-{backend}-{task}-"
                        f"continuation-{shard_index:03d}-of-{shard_counts[task]:03d}-"
                        f"{source_sha[:10]}"
                    ),
                    "backend_config": backends[backend],
                }
                spec_payload = _canonical_bytes(row)
                spec_path = destination / "jobs" / f"{array_index:04d}.json"
                _write_once(spec_path, spec_payload)
                row["job_spec_path"] = str(spec_path.resolve())
                row["job_spec_sha256"] = _sha_bytes(spec_payload)
                rows.append(row)
    if len(rows) != int(protocol["execution_plan"]["continuation_backend_jobs"]):
        raise RuntimeError("continuation job count differs from the protocol")
    index_payload = b"".join(_canonical_bytes(row) for row in rows)
    index_path = destination / "jobs.jsonl"
    _write_once(index_path, index_payload)
    manifest = {
        "schema_version": 1,
        "experiment_id": protocol["experiment_id"],
        "classification": CLASSIFICATION,
        "source_git_sha": source_sha,
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_sha,
        "candidate_config_sha256": candidate["config_sha256"],
        "prior_campaign_root": str(prior_root),
        "prior_paired_rows_sha256": reuse["prior_paired_rows_sha256"],
        "reused_samples_per_task": reused_count,
        "target_sample_counts": target_counts,
        "continuation_sample_counts": continuation_counts,
        "continuation_shard_counts": shard_counts,
        "full_ledgers": {
            task: {"path": str(full_ledger_paths[task]), "sha256": full_ledgers[task][1]}
            for task in TASKS
        },
        "sample_shard_count": len(shard_records),
        "job_count": len(rows),
        "maximum_concurrent_a100_jobs": 6,
        "index_path": str(index_path.resolve()),
        "index_sha256": _sha_bytes(index_payload),
        "output_root": str(output_root.resolve()),
    }
    _write_once(destination / "campaign_manifest.json", _canonical_bytes(manifest))
    return index_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--full-ledger", action="append", default=[])
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    index = prepare_continuation(
        protocol_path=args.protocol,
        full_ledger_paths=_parse_paths(args.full_ledger, "--full-ledger"),
        destination=args.destination,
        output_root=args.output_root,
        repository=repository,
    )
    print(
        json.dumps(
            {"index": str(index), "job_count": len(index.read_text().splitlines())},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
