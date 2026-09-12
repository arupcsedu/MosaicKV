#!/usr/bin/env python3
"""Prepare immutable shards for the published-baseline fidelity diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any, cast

import pandas as pd  # type: ignore[import-untyped]
import yaml

TASKS = ("textvqa", "docvqa")
FAMILIES = ("prefixkv", "vl_cache")
CLASSIFICATION = "canonical_published_baseline_fidelity_diagnostic"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return cast("dict[str, Any]", value)


def _write_once(path: Path, value: object) -> str:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    digest = hashlib.sha256(payload).hexdigest()
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace immutable input: {path}")
        return digest
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, stat.S_IRUSR | stat.S_IRGRP)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def _snapshot(repository: Path) -> tuple[str, bool]:
    sha = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=repository, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=all"),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return sha, bool(status.strip())


def load_protocol(path: Path, *, require_authorized: bool = True) -> dict[str, Any]:
    protocol = _object(yaml.safe_load(path.read_text(encoding="utf-8")), "protocol")
    if (
        protocol.get("schema_version") != 1
        or protocol.get("experiment_id") != "baseline_fidelity_review_v1"
    ):
        raise ValueError("wrong baseline-fidelity protocol")
    if protocol.get("classification") != CLASSIFICATION:
        raise ValueError("wrong baseline-fidelity classification")
    if require_authorized and protocol.get("execution_authorized") is not True:
        raise PermissionError("baseline-fidelity execution is not authorized")
    if tuple(_object(protocol.get("tasks"), "tasks")) != TASKS:
        raise ValueError("task order changed")
    if tuple(_object(protocol.get("families"), "families")) != FAMILIES:
        raise ValueError("family order changed")
    evaluation = _object(protocol.get("evaluation"), "evaluation")
    if int(evaluation.get("sample_count_per_task", 0)) != 64:
        raise ValueError("sample count changed")
    if int(evaluation.get("samples_per_shard", 0)) != 8:
        raise ValueError("shard size changed")
    if int(evaluation.get("block_size", 0)) != 1:
        raise ValueError("baseline diagnostic requires token-sized blocks")
    if evaluation.get("budget_unit") != "bytes":
        raise ValueError("baseline diagnostic requires physical byte budgets")
    controls = _object(protocol.get("controls"), "controls")
    if any(value is not True for value in controls.values()):
        raise ValueError("every baseline-fidelity control must remain enabled")
    return protocol


def select_ids(frame: pd.DataFrame, *, task: str, count: int, seed: int) -> tuple[str, ...]:
    required = {"task", "sample_id", "backend_scope", "protocol_method", "status"}
    if missing := required - set(frame.columns):
        raise ValueError(f"identity artifact is missing columns: {sorted(missing)}")
    subset = frame[
        frame["task"].astype(str).eq(task)
        & frame["backend_scope"].astype(str).eq("hf_eager")
        & frame["protocol_method"].astype(str).eq("full_kv")
        & frame["status"].astype(str).eq("completed")
    ]
    identities = sorted(set(str(value) for value in subset["sample_id"]))
    if len(identities) < count:
        raise ValueError(f"{task} has only {len(identities)} completed FullKV identities")
    identities.sort(
        key=lambda sample_id: (
            hashlib.sha256(f"{seed}:{task}:{sample_id}".encode()).digest(),
            sample_id,
        )
    )
    return tuple(identities[:count])


def prepare(
    protocol_path: Path, destination: Path, output_root: Path, repository: Path
) -> dict[str, Any]:
    protocol_path = protocol_path.resolve()
    destination = destination.resolve()
    output_root = output_root.resolve()
    repository = repository.resolve()
    source_sha, dirty = _snapshot(repository)
    if dirty:
        raise RuntimeError("canonical baseline-fidelity preparation requires a clean worktree")
    protocol = load_protocol(protocol_path)
    protocol_sha = _sha(protocol_path)
    source = _object(protocol["source"], "source")
    artifact = Path(str(source["sample_identity_artifact"])).resolve()
    if _sha(artifact) != source["sample_identity_sha256"]:
        raise ValueError("sample identity artifact hash changed")
    if destination.exists():
        raise FileExistsError(f"campaign input destination already exists: {destination}")
    frame = pd.read_parquet(
        artifact, columns=["task", "sample_id", "backend_scope", "protocol_method", "status"]
    )
    evaluation = _object(protocol["evaluation"], "evaluation")
    count = int(evaluation["sample_count_per_task"])
    shard_size = int(evaluation["samples_per_shard"])
    seed = int(evaluation["sample_selection_seed"])
    destination.mkdir(parents=True)
    selected: dict[str, list[str]] = {}
    jobs: list[dict[str, Any]] = []
    for family in FAMILIES:
        contract = _object(_object(protocol["families"], "families")[family], family)
        for task in TASKS:
            if task not in selected:
                selected[task] = list(select_ids(frame, task=task, count=count, seed=seed))
            for shard_index, offset in enumerate(range(0, count, shard_size)):
                members = selected[task][offset : offset + shard_size]
                ledger = {
                    "schema_version": 1,
                    "classification": CLASSIFICATION,
                    "source_git_sha": source_sha,
                    "source_git_dirty": False,
                    "protocol_sha256": protocol_sha,
                    "family": family,
                    "task": task,
                    "sample_ids": members,
                    "sample_count": len(members),
                    "shard_index": shard_index,
                    "shard_count": count // shard_size,
                    "sample_selection_seed": seed,
                    "dataset": _object(protocol["tasks"], "tasks")[task],
                }
                ledger_path = destination / "ledgers" / family / task / f"{shard_index:02d}.json"
                ledger_sha = _write_once(ledger_path, ledger)
                jobs.append(
                    {
                        "schema_version": 1,
                        "array_index": len(jobs),
                        "classification": CLASSIFICATION,
                        "source_git_sha": source_sha,
                        "protocol_path": str(protocol_path),
                        "protocol_sha256": protocol_sha,
                        "family": family,
                        "task": task,
                        "model_id": contract["model_id"],
                        "model_revision": contract["model_revision"],
                        "methods": contract["methods"],
                        "retention_ratios": contract["retention_ratios"],
                        "sample_ledger_path": str(ledger_path),
                        "sample_ledger_sha256": ledger_sha,
                        "shard_index": shard_index,
                        "sample_count": len(members),
                        "selection_seed": evaluation["selection_seed"],
                        "output_length": evaluation["output_length"],
                        "output_root": str(output_root),
                    }
                )
    if len(jobs) != 32:
        raise RuntimeError(f"expected 32 jobs, generated {len(jobs)}")
    index: list[dict[str, Any]] = []
    for job in jobs:
        path = destination / "jobs" / f"{int(job['array_index']):03d}.json"
        digest = _write_once(path, job)
        index.append({**job, "job_spec_path": str(path), "job_spec_sha256": digest})
    index_path = destination / "jobs.jsonl"
    lines = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode() for row in index
    )
    if index_path.exists():
        raise FileExistsError(index_path)
    index_path.write_bytes(lines)
    manifest = {
        "schema_version": 1,
        "classification": CLASSIFICATION,
        "source_git_sha": source_sha,
        "source_git_dirty": False,
        "protocol_sha256": protocol_sha,
        "sample_identity_artifact_sha256": _sha(artifact),
        "selection_reads_outcome_columns": False,
        "selected_sample_ids": selected,
        "job_count": len(jobs),
        "jobs_index": str(index_path),
        "jobs_index_sha256": _sha(index_path),
    }
    _write_once(destination / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args.protocol, args.destination, args.output_root, args.repository), indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
