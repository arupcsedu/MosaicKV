#!/usr/bin/env python3
"""Create and verify immutable HF/vLLM paper-campaign shard specifications."""

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

import yaml  # type: ignore[import-untyped]

TASKS = ("mmstar", "textvqa", "docvqa")
BACKENDS = ("hf_eager", "vllm")
CLASSIFICATION = "fixed_preregistered_paper_quality_memory"
CONTINUATION_CLASSIFICATION = "fixed_preregistered_paper_quality_memory_continuation"
PROTOCOL_IDS = {
    "paper_quality_hf_vllm_llava15_v2",
    "paper_quality_hf_vllm_llava15_v3_continuation",
}


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _write_once(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable campaign input: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary_name = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_name, 0o444)
        os.replace(temporary_name, path)
    except BaseException:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise


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


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return cast("dict[str, Any]", value)


def _load_protocol(path: Path, *, require_execution_authorized: bool) -> dict[str, Any]:
    protocol = _object(yaml.safe_load(path.read_text(encoding="utf-8")), "protocol")
    if protocol.get("schema_version") != 1:
        raise ValueError("unsupported paper protocol schema")
    if protocol.get("experiment_id") not in PROTOCOL_IDS:
        raise ValueError("wrong paper protocol")
    if protocol.get("classification") not in {
        CLASSIFICATION,
        CONTINUATION_CLASSIFICATION,
    }:
        raise ValueError("wrong paper protocol classification")
    if require_execution_authorized and protocol.get("execution_authorized") is not True:
        raise PermissionError("paper campaign execution has not been authorized")
    experiment = _object(protocol.get("experiment"), "protocol.experiment")
    if tuple(_object(experiment.get("tasks"), "protocol.experiment.tasks")) != TASKS:
        raise ValueError("paper protocol task order changed")
    if tuple(_object(experiment.get("backends"), "protocol.experiment.backends")) != BACKENDS:
        raise ValueError("paper protocol backend order changed")
    if protocol.get("experiment_id") == "paper_quality_hf_vllm_llava15_v2":
        sample_count = int(experiment.get("sample_count_per_task", 0))
        shard_count = int(experiment.get("shards_per_task", 0))
        shard_size = int(experiment.get("samples_per_shard", 0))
        if (
            sample_count != shard_count * shard_size
            or min(sample_count, shard_count, shard_size) < 1
        ):
            raise ValueError("sample count must equal shards_per_task * samples_per_shard")
    else:
        target_counts = _object(
            experiment.get("target_sample_counts"),
            "protocol.experiment.target_sample_counts",
        )
        continuation_counts = _object(
            experiment.get("continuation_sample_counts"),
            "protocol.experiment.continuation_sample_counts",
        )
        reuse = _object(protocol.get("reuse"), "protocol.reuse")
        reused = int(reuse.get("reused_samples_per_task", 0))
        if reused < 1 or int(experiment.get("samples_per_continuation_shard", 0)) < 1:
            raise ValueError("continuation reuse and shard sizes must be positive")
        reuse_hashes = {
            str(value)
            for key, value in reuse.items()
            if key.endswith("_sha256") and isinstance(value, str)
        }
        reuse_hashes.update(
            str(value)
            for value in _object(
                reuse.get("prior_master_ledger_sha256"),
                "protocol.reuse.prior_master_ledger_sha256",
            ).values()
        )
        if any(
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
            for value in reuse_hashes
        ):
            raise ValueError("continuation reuse contains an invalid SHA-256 digest")
        for task in TASKS:
            if int(target_counts.get(task, 0)) != reused + int(continuation_counts.get(task, 0)):
                raise ValueError(f"continuation sample arithmetic differs for {task}")
    candidate = _object(protocol.get("candidate"), "protocol.candidate")
    commit = str(candidate.get("frozen_algorithm_commit", ""))
    if len(commit) != 40:
        raise ValueError("candidate frozen_algorithm_commit must be a full SHA")
    digest = str(candidate.get("config_sha256", ""))
    if len(digest) != 64:
        raise ValueError("candidate config_sha256 must be a SHA-256 digest")
    return protocol


def _parse_ledgers(values: list[str]) -> dict[str, Path]:
    ledgers: dict[str, Path] = {}
    for value in values:
        task, separator, path = value.partition("=")
        if not separator or task not in TASKS or not path:
            raise ValueError("--ledger must use mmstar|textvqa|docvqa=/absolute/path.json")
        if task in ledgers:
            raise ValueError(f"duplicate ledger for {task}")
        ledgers[task] = Path(path).resolve()
    if tuple(task for task in TASKS if task in ledgers) != TASKS:
        raise ValueError("one ledger is required for each protocol task")
    return ledgers


def _load_ledger(
    path: Path,
    *,
    task: str,
    task_contract: dict[str, Any],
    sample_count: int,
    protocol_sha: str,
    source_sha: str,
) -> tuple[dict[str, Any], str]:
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise ValueError(f"master sample ledger is not read-only: {path}")
    payload = path.read_bytes()
    ledger = _object(json.loads(payload), str(path))
    expected = {
        "schema_version": 1,
        "task": task,
        "dataset_id": task_contract["dataset_id"],
        "dataset_revision": task_contract["dataset_revision"],
        "split": task_contract["split"],
        "classification": CLASSIFICATION,
        "canonical_eligible": True,
        "source_git_sha": source_sha,
        "source_git_dirty": False,
        "sample_count": sample_count,
        "selection_freeze_sha256": protocol_sha,
    }
    if any(ledger.get(key) != value for key, value in expected.items()):
        raise ValueError(f"master sample ledger differs from the protocol: {path}")
    ids = ledger.get("sample_ids")
    if (
        not isinstance(ids, list)
        or len(ids) != sample_count
        or len(set(str(item) for item in ids)) != sample_count
        or not all(isinstance(item, str) and item for item in ids)
    ):
        raise ValueError(f"master sample ledger IDs are invalid: {path}")
    return ledger, _sha_bytes(payload)


def prepare_campaign(
    *,
    protocol_path: Path,
    ledger_paths: dict[str, Path],
    destination: Path,
    repository: Path,
    output_root: Path,
) -> Path:
    """Create 12 sample shards and 24 immutable backend job records."""

    protocol_path = protocol_path.resolve()
    destination = destination.resolve()
    repository = repository.resolve()
    output_root = output_root.resolve()
    source_sha, dirty = _source_snapshot(repository)
    if dirty:
        raise RuntimeError("paper campaign preparation requires a clean worktree")
    protocol = _load_protocol(protocol_path, require_execution_authorized=True)
    protocol_sha = _sha(protocol_path)
    experiment = _object(protocol["experiment"], "protocol.experiment")
    candidate = _object(protocol["candidate"], "protocol.candidate")
    candidate_path = (protocol_path.parents[2] / str(candidate["config"])).resolve()
    if _sha(candidate_path) != candidate["config_sha256"]:
        raise ValueError("candidate configuration hash differs from the protocol")
    ancestor = subprocess.run(
        (
            "git",
            "merge-base",
            "--is-ancestor",
            str(candidate["frozen_algorithm_commit"]),
            source_sha,
        ),
        cwd=repository,
        check=False,
    )
    if ancestor.returncode != 0:
        raise ValueError("execution source does not descend from the frozen candidate commit")
    if destination.exists():
        raise FileExistsError(f"campaign destination already exists: {destination}")

    sample_count = int(experiment["sample_count_per_task"])
    shard_count = int(experiment["shards_per_task"])
    shard_size = int(experiment["samples_per_shard"])
    task_contracts = _object(experiment["tasks"], "protocol.experiment.tasks")
    loaded = {
        task: _load_ledger(
            ledger_paths[task],
            task=task,
            task_contract=_object(task_contracts[task], f"task {task}"),
            sample_count=sample_count,
            protocol_sha=protocol_sha,
            source_sha=source_sha,
        )
        for task in TASKS
    }

    destination.mkdir(parents=True, exist_ok=False)
    shard_records: dict[tuple[str, int], dict[str, Any]] = {}
    for task in TASKS:
        ledger, ledger_sha = loaded[task]
        ids = cast("list[str]", ledger["sample_ids"])
        for shard_index in range(shard_count):
            selected = ids[shard_index * shard_size : (shard_index + 1) * shard_size]
            payload = _canonical_bytes(
                {
                    "schema_version": 1,
                    "classification": CLASSIFICATION,
                    "task": task,
                    "dataset_id": ledger["dataset_id"],
                    "dataset_revision": ledger["dataset_revision"],
                    "split": ledger["split"],
                    "selection_seed": ledger["selection_seed"],
                    "source_git_sha": source_sha,
                    "source_git_dirty": False,
                    "canonical_eligible": True,
                    "selection_freeze_sha256": protocol_sha,
                    "parent_ledger_sha256": ledger_sha,
                    "shard_index": shard_index,
                    "shard_count": shard_count,
                    "sample_count": len(selected),
                    "sample_ids": selected,
                }
            )
            path = destination / "sample_shards" / task / f"{shard_index:02d}.json"
            _write_once(path, payload)
            shard_records[(task, shard_index)] = {
                "path": str(path.resolve()),
                "sha256": _sha_bytes(payload),
                "sample_count": len(selected),
            }

    rows: list[dict[str, Any]] = []
    for backend in BACKENDS:
        for task in TASKS:
            task_contract = _object(task_contracts[task], f"task {task}")
            for shard_index in range(shard_count):
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
                    "dataset_revision": task_contract["dataset_revision"],
                    "shard_index": shard_index,
                    "shard_count": shard_count,
                    "sample_count": shard["sample_count"],
                    "sample_ledger_path": shard["path"],
                    "sample_ledger_sha256": shard["sha256"],
                    "selection_seed": experiment["selection_seed"],
                    "output_length": experiment["output_length"],
                    "precision": experiment["precision"],
                    "output_root": str(output_root),
                    "run_namespace": (
                        f"{protocol['experiment_id']}-{backend}-{task}-"
                        f"shard-{shard_index:02d}-of-{shard_count:02d}-{source_sha[:10]}"
                    ),
                    "backend_config": _object(experiment["backends"], "backends")[backend],
                }
                payload = _canonical_bytes(row)
                spec_path = destination / "jobs" / f"{array_index:03d}.json"
                _write_once(spec_path, payload)
                row["job_spec_path"] = str(spec_path.resolve())
                row["job_spec_sha256"] = _sha_bytes(payload)
                rows.append(row)
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
        "master_ledgers": {
            task: {"path": str(ledger_paths[task]), "sha256": loaded[task][1]} for task in TASKS
        },
        "sample_shard_count": len(shard_records),
        "job_count": len(rows),
        "maximum_concurrent_a100_jobs": 4,
        "index_path": str(index_path.resolve()),
        "index_sha256": _sha_bytes(index_payload),
        "output_root": str(output_root),
    }
    _write_once(destination / "campaign_manifest.json", _canonical_bytes(manifest))
    return index_path


def validate_job(index_path: Path, array_index: int, repository: Path) -> dict[str, Any]:
    """Fail closed unless one job and all hash-bound inputs remain immutable."""

    source_sha, dirty = _source_snapshot(repository.resolve())
    if dirty:
        raise RuntimeError("paper campaign execution requires a clean worktree")
    rows = [
        _object(json.loads(line), f"{index_path}:{number}")
        for number, line in enumerate(index_path.read_text(encoding="utf-8").splitlines(), 1)
        if line
    ]
    if not 0 <= array_index < len(rows):
        raise ValueError(f"array index {array_index} is outside [0, {len(rows) - 1}]")
    row = rows[array_index]
    if row.get("array_index") != array_index or row.get("source_git_sha") != source_sha:
        raise ValueError("job record source/index identity changed")
    protocol_path = Path(str(row["protocol_path"]))
    if _sha(protocol_path) != row.get("protocol_sha256"):
        raise ValueError("paper protocol hash changed")
    _load_protocol(protocol_path, require_execution_authorized=True)
    candidate_path = Path(str(row["candidate_config_path"]))
    ledger_path = Path(str(row["sample_ledger_path"]))
    spec_path = Path(str(row["job_spec_path"]))
    if _sha(candidate_path) != row.get("candidate_config_sha256"):
        raise ValueError("candidate configuration hash changed")
    if _sha(ledger_path) != row.get("sample_ledger_sha256"):
        raise ValueError("sample-shard ledger hash changed")
    if _sha(spec_path) != row.get("job_spec_sha256"):
        raise ValueError("job specification hash changed")
    spec = _object(json.loads(spec_path.read_text(encoding="utf-8")), str(spec_path))
    for key, value in spec.items():
        if row.get(key) != value:
            raise ValueError(f"indexed job differs from its immutable spec at {key}")
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--protocol", type=Path, required=True)
    prepare.add_argument("--ledger", action="append", default=[])
    prepare.add_argument("--destination", type=Path, required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    validate = subparsers.add_parser("validate-job")
    validate.add_argument("--index", type=Path, required=True)
    validate.add_argument("--array-index", type=int, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    if args.command == "prepare":
        index = prepare_campaign(
            protocol_path=args.protocol,
            ledger_paths=_parse_ledgers(args.ledger),
            destination=args.destination,
            repository=repository,
            output_root=args.output_root,
        )
        payload = {"index": str(index), "job_count": len(index.read_text().splitlines())}
    else:
        payload = validate_job(args.index.resolve(), args.array_index, repository)
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
