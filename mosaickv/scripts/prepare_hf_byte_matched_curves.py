#!/usr/bin/env python3
"""Prepare immutable jobs for the paired HF byte-budget curve experiment."""

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

TASKS = ("mmstar", "textvqa", "docvqa")
METHODS = ("prompt_attention_topk", "mosaickv_exact")
SOURCE_METHODS = frozenset({"full_kv", "prompt_attention_topk", "mosaickv_full"})
BUDGETS_MIB = (16, 22, 30, 40, 60)
CLASSIFICATION = "canonical_byte_matched_selector_curve"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _write_once(path: Path, value: object) -> str:
    return _write_bytes_once(path, _canonical_bytes(value))


def _write_bytes_once(path: Path, payload: bytes) -> str:
    digest = hashlib.sha256(payload).hexdigest()
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace immutable input: {path}")
        return digest
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
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


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return cast("dict[str, Any]", value)


def _source_snapshot(repository: Path) -> tuple[str, bool]:
    sha = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=all"),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return sha, bool(status.strip())


def load_protocol(path: Path, *, require_authorized: bool) -> dict[str, Any]:
    protocol = _object(yaml.safe_load(path.read_text(encoding="utf-8")), "protocol")
    if protocol.get("schema_version") != 1:
        raise ValueError("unsupported byte-curve protocol schema")
    if protocol.get("experiment_id") != "hf_byte_matched_curves_llava15_v1":
        raise ValueError("wrong byte-curve experiment ID")
    if protocol.get("classification") != CLASSIFICATION:
        raise ValueError("wrong byte-curve classification")
    if require_authorized and protocol.get("execution_authorized") is not True:
        raise PermissionError("byte-curve execution is not authorized")
    evaluation = _object(protocol.get("evaluation"), "evaluation")
    if tuple(evaluation.get("methods", ())) != METHODS:
        raise ValueError("byte-curve method order changed")
    if tuple(int(value) for value in evaluation.get("budgets_mib", ())) != BUDGETS_MIB:
        raise ValueError("byte-curve budgets changed")
    if evaluation.get("budget_unit") != "bytes":
        raise ValueError("byte-curve budget unit must be bytes")
    if int(evaluation.get("block_size", 0)) != 4:
        raise ValueError("byte-curve block size must be four")
    if evaluation.get("physical_budget_mandatory_policy") != "last_token":
        raise ValueError("byte-curve mandatory policy changed")
    if int(evaluation.get("sample_count_per_task", 0)) != 256:
        raise ValueError("byte-curve sample count changed")
    if int(evaluation.get("samples_per_shard", 0)) != 16:
        raise ValueError("byte-curve shard size changed")
    if tuple(_object(protocol.get("tasks"), "tasks")) != TASKS:
        raise ValueError("byte-curve task order changed")
    controls = _object(protocol.get("paired_controls"), "paired_controls")
    required_controls = (
        "require_identical_prompts_media_and_sample_ids",
        "require_identical_processor_and_chat_template",
        "require_identical_model_revision_and_precision",
        "require_identical_generation_parameters",
        "require_identical_block_and_mandatory_policy",
        "require_identical_configured_byte_cap",
        "require_exact_per_sample_active_byte_match",
        "require_complete_rows",
        "require_clean_source",
    )
    if any(controls.get(name) is not True for name in required_controls):
        raise ValueError("every paired byte-curve control must be enabled")
    return protocol


def select_paired_sample_ids(
    frame: pd.DataFrame,
    *,
    task: str,
    sample_count: int,
    seed: int,
) -> tuple[str, ...]:
    """Select IDs without reading scores, answers, timing, or memory outcomes."""

    required = {"task", "sample_id", "backend_scope", "protocol_method", "status"}
    if missing := required - set(frame.columns):
        raise ValueError(f"source artifact is missing columns: {sorted(missing)}")
    subset = frame[
        frame["task"].astype(str).eq(task)
        & frame["backend_scope"].astype(str).eq("hf_eager")
        & frame["protocol_method"].astype(str).isin(SOURCE_METHODS)
    ]
    if set(subset["status"].astype(str)) != {"completed"}:
        raise ValueError(f"{task} source contains incomplete rows")
    observed = subset.groupby("sample_id")["protocol_method"].agg(
        lambda values: frozenset(str(value) for value in values)
    )
    valid = [str(sample_id) for sample_id, methods in observed.items() if methods == SOURCE_METHODS]
    if len(valid) < sample_count:
        raise ValueError(f"{task} has only {len(valid)} completely paired source identities")
    ranked = sorted(
        valid,
        key=lambda sample_id: (
            hashlib.sha256(f"{seed}:{task}:{sample_id}".encode()).digest(),
            sample_id,
        ),
    )
    return tuple(ranked[:sample_count])


def build_job_records(
    *,
    protocol: dict[str, Any],
    protocol_path: Path,
    protocol_sha256: str,
    source_git_sha: str,
    source_artifact_sha256: str,
    candidate_path: Path,
    candidate_sha256: str,
    output_root: Path,
    ledger_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    evaluation = _object(protocol["evaluation"], "evaluation")
    jobs: list[dict[str, Any]] = []
    for ledger in ledger_records:
        jobs.append(
            {
                "schema_version": 1,
                "array_index": len(jobs),
                "experiment_id": protocol["experiment_id"],
                "classification": CLASSIFICATION,
                "source_git_sha": source_git_sha,
                "source_git_dirty": False,
                "protocol_path": str(protocol_path),
                "protocol_sha256": protocol_sha256,
                "source_artifact_sha256": source_artifact_sha256,
                "candidate_config_path": str(candidate_path),
                "candidate_config_sha256": candidate_sha256,
                "task": ledger["task"],
                "shard_index": ledger["shard_index"],
                "shard_count": ledger["shard_count"],
                "sample_count": ledger["sample_count"],
                "sample_ledger_path": ledger["path"],
                "sample_ledger_sha256": ledger["sha256"],
                "methods": list(METHODS),
                "budgets_mib": list(BUDGETS_MIB),
                "retention_ratio_upper_bound": evaluation["retention_ratio_upper_bound"],
                "selection_seed": evaluation["selection_seed"],
                "output_length": evaluation["output_length"],
                "output_root": str(output_root),
            }
        )
    return jobs


def prepare(
    *,
    protocol_path: Path,
    destination: Path,
    output_root: Path,
    repository: Path,
) -> dict[str, Any]:
    protocol_path = protocol_path.resolve()
    destination = destination.resolve()
    output_root = output_root.resolve()
    repository = repository.resolve()
    source_git_sha, dirty = _source_snapshot(repository)
    if dirty:
        raise RuntimeError("canonical byte-curve preparation requires a clean worktree")
    protocol = load_protocol(protocol_path, require_authorized=True)
    protocol_sha = _sha(protocol_path)
    source = _object(protocol["source"], "source")
    source_path = Path(str(source["paired_result_artifact"])).resolve()
    source_sha = _sha(source_path)
    if source_sha != source["paired_result_sha256"]:
        raise ValueError("paired source artifact hash changed")
    package_root = repository / "mosaickv"
    candidate_path = (package_root / str(source["candidate_config"])).resolve()
    candidate_sha = _sha(candidate_path)
    if candidate_sha != source["candidate_config_sha256"]:
        raise ValueError("frozen candidate configuration hash changed")
    ancestor = subprocess.run(
        (
            "git",
            "merge-base",
            "--is-ancestor",
            str(source["frozen_algorithm_commit"]),
            source_git_sha,
        ),
        cwd=repository,
        check=False,
    )
    if ancestor.returncode:
        raise ValueError("execution source does not descend from the frozen algorithm commit")
    if destination.exists():
        raise FileExistsError(f"campaign input destination already exists: {destination}")

    columns = ["task", "sample_id", "backend_scope", "protocol_method", "status"]
    frame = pd.read_parquet(source_path, columns=columns)
    evaluation = _object(protocol["evaluation"], "evaluation")
    task_contracts = _object(protocol["tasks"], "tasks")
    sample_count = int(evaluation["sample_count_per_task"])
    shard_size = int(evaluation["samples_per_shard"])
    selection_seed = int(evaluation["sample_selection_seed"])
    destination.mkdir(parents=True, exist_ok=False)
    ledger_records: list[dict[str, Any]] = []
    selected_by_task: dict[str, list[str]] = {}
    for task in TASKS:
        selected = select_paired_sample_ids(
            frame,
            task=task,
            sample_count=sample_count,
            seed=selection_seed,
        )
        selected_by_task[task] = list(selected)
        contract = _object(task_contracts[task], f"tasks.{task}")
        shard_count = sample_count // shard_size
        for shard_index in range(shard_count):
            members = selected[shard_index * shard_size : (shard_index + 1) * shard_size]
            ledger = {
                "schema_version": 1,
                "classification": CLASSIFICATION,
                "task": task,
                "dataset_id": contract["dataset_id"],
                "dataset_revision": contract["dataset_revision"],
                "split": contract["split"],
                "sample_count": len(members),
                "sample_ids": list(members),
                "sample_selection_seed": selection_seed,
                "source_result_artifact": str(source_path),
                "source_result_sha256": source_sha,
                "source_git_sha": source_git_sha,
                "source_git_dirty": False,
                "protocol_sha256": protocol_sha,
                "shard_index": shard_index,
                "shard_count": shard_count,
            }
            ledger_path = destination / "ledgers" / task / f"{shard_index:02d}.json"
            ledger_sha = _write_once(ledger_path, ledger)
            ledger_records.append(
                {
                    "task": task,
                    "shard_index": shard_index,
                    "shard_count": shard_count,
                    "sample_count": len(members),
                    "path": str(ledger_path),
                    "sha256": ledger_sha,
                }
            )

    jobs = build_job_records(
        protocol=protocol,
        protocol_path=protocol_path,
        protocol_sha256=protocol_sha,
        source_git_sha=source_git_sha,
        source_artifact_sha256=source_sha,
        candidate_path=candidate_path,
        candidate_sha256=candidate_sha,
        output_root=output_root,
        ledger_records=ledger_records,
    )
    if len(jobs) != 48:
        raise RuntimeError(f"expected 48 jobs, generated {len(jobs)}")
    index_rows: list[dict[str, Any]] = []
    for job in jobs:
        path = destination / "jobs" / f"{int(job['array_index']):03d}.json"
        digest = _write_once(path, job)
        index_rows.append({**job, "job_spec_path": str(path), "job_spec_sha256": digest})
    index_path = destination / "jobs.jsonl"
    index_payload = b"".join(_canonical_bytes(row) for row in index_rows)
    _write_bytes_once(index_path, index_payload)
    selection_manifest = {
        "schema_version": 1,
        "classification": CLASSIFICATION,
        "source_git_sha": source_git_sha,
        "protocol_sha256": protocol_sha,
        "source_result_sha256": source_sha,
        "selection_reads_outcome_columns": False,
        "selected_sample_ids": selected_by_task,
        "job_count": len(jobs),
        "index_path": str(index_path),
        "index_sha256": _sha(index_path),
    }
    _write_once(destination / "selection_manifest.json", selection_manifest)
    return selection_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    result = prepare(
        protocol_path=args.protocol,
        destination=args.destination,
        output_root=args.output_root,
        repository=args.repository,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
