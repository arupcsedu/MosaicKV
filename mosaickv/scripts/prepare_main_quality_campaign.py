#!/usr/bin/env python3
"""Explode the logical main-quality matrix into immutable, paired sample shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import yaml

from mosaickv.config import (
    FidelityDiagnosticsConfig,
    canonical_config_json,
    config_sha256,
)
from mosaickv.experiment_matrix import (
    TASK_CATALOG,
    expand_experiment_matrix,
    load_experiment_matrix,
    verify_expanded_index,
)


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_once(path: Path, payload: bytes, *, mode: int = 0o444) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable campaign artifact: {path}")
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


def _clean_source_sha() -> str:
    repository = Path(__file__).resolve().parents[2]
    status = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise RuntimeError("main-quality campaign materialization requires a clean worktree")
    return subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _parse_ledgers(values: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        task, separator, path = value.partition("=")
        if not separator or not task or not path:
            raise ValueError("--ledger values must use task=/absolute/path.json")
        if task in parsed:
            raise ValueError(f"duplicate ledger for task {task!r}")
        parsed[task] = Path(path).resolve()
    return parsed


def _load_ledger(task_key: str, path: Path) -> tuple[dict[str, Any], str]:
    task = TASK_CATALOG[task_key]
    payload = path.read_bytes()
    permissions = stat.S_IMODE(path.stat().st_mode)
    if permissions & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise ValueError(f"sample ledger must be read-only: {path}")
    value = json.loads(payload)
    expected = {
        "schema_version": 1,
        "task": task.task_name,
        "dataset_id": task.dataset_id,
        "dataset_revision": task.revision,
        "split": task.split,
    }
    if not isinstance(value, dict) or any(value.get(key) != item for key, item in expected.items()):
        raise ValueError(f"sample ledger identity mismatch for {task_key}: {path}")
    ids = value.get("sample_ids")
    if (
        not isinstance(ids, list)
        or not ids
        or not all(isinstance(sample_id, str) and sample_id.strip() for sample_id in ids)
    ):
        raise ValueError(f"sample ledger has invalid sample_ids: {path}")
    if len(ids) != len(set(ids)) or value.get("sample_count") != len(ids):
        raise ValueError(f"sample ledger count/uniqueness mismatch: {path}")
    source_sha = value.get("source_git_sha")
    if (
        not isinstance(source_sha, str)
        or len(source_sha) != 40
        or value.get("source_git_dirty") is not False
    ):
        raise ValueError(f"sample ledger lacks clean source provenance: {path}")
    return cast("dict[str, Any]", value), _sha(payload)


def materialize(
    *,
    matrix_path: Path,
    ledger_paths: dict[str, Path],
    destination: Path,
    shard_size: int,
    diagnostic_size: int,
    resume: bool,
    sample_limit: int | None = None,
    model_keys: frozenset[str] = frozenset(),
    task_keys: frozenset[str] = frozenset(),
    methods: frozenset[str] = frozenset(),
    retention_ratios: frozenset[float] = frozenset(),
    output_root: Path | None = None,
) -> Path:
    if shard_size < 1 or diagnostic_size < 1:
        raise ValueError("shard and diagnostic sizes must be positive")
    if sample_limit is not None and sample_limit < 1:
        raise ValueError("sample_limit must be positive when provided")
    matrix = load_experiment_matrix(matrix_path)
    resolved_output_root = (
        output_root.resolve() if output_root is not None else Path(matrix.run.output_root).resolve()
    )
    logical_runs = tuple(
        run
        for run in expand_experiment_matrix(matrix)
        if (not model_keys or run.model_key in model_keys)
        and (not task_keys or run.task_key in task_keys)
        and (not methods or run.config.method.value in methods)
        and (not retention_ratios or run.config.cache.retention_ratio in retention_ratios)
    )
    if not logical_runs:
        raise ValueError("campaign filters select no logical runs")
    required_tasks = {run.task_key for run in logical_runs}
    if set(ledger_paths) != required_tasks:
        raise ValueError(
            f"ledger task set mismatch: required={sorted(required_tasks)}, "
            f"provided={sorted(ledger_paths)}"
        )
    ledgers = {task: _load_ledger(task, ledger_paths[task]) for task in sorted(required_tasks)}
    if sample_limit is not None:
        for ledger, _ledger_sha in ledgers.values():
            ids = cast("list[str]", ledger["sample_ids"])
            ledger["sample_ids"] = ids[:sample_limit]
            ledger["sample_count"] = len(cast("list[str]", ledger["sample_ids"]))
    projected_jobs = sum(
        (len(cast("list[str]", ledgers[run.task_key][0]["sample_ids"])) + shard_size - 1)
        // shard_size
        for run in logical_runs
    )
    if projected_jobs > 10_000:
        raise ValueError(
            f"campaign expands to {projected_jobs} jobs, exceeding Slurm "
            "MaxArraySize=10000; increase --shard-size or split the versioned campaign"
        )
    index_path = destination / "jobs.jsonl"
    manifest_path = destination / "matrix_manifest.json"
    if destination.exists():
        if not resume:
            raise FileExistsError(
                f"campaign already exists; pass --resume to verify it: {destination}"
            )
        if not index_path.is_file() or not manifest_path.is_file():
            raise RuntimeError("existing campaign expansion is incomplete")
        verify_expanded_index(index_path)
        return index_path
    source_git_sha = _clean_source_sha()
    destination.mkdir(parents=True, exist_ok=False)
    _write_once(destination / "source_matrix.yaml", matrix_path.read_bytes())
    shard_files: dict[tuple[str, int], tuple[Path, str, int]] = {}
    for task_key, (ledger, _ledger_sha) in ledgers.items():
        ids = cast("list[str]", ledger["sample_ids"])
        shard_count = (len(ids) + shard_size - 1) // shard_size
        for shard_index in range(shard_count):
            selected = ids[shard_index * shard_size : (shard_index + 1) * shard_size]
            task = TASK_CATALOG[task_key]
            payload = _canonical_bytes(
                {
                    "schema_version": 1,
                    "task": task.task_name,
                    "dataset_id": task.dataset_id,
                    "dataset_revision": task.revision,
                    "split": task.split,
                    "parent_ledger_sha": _ledger_sha,
                    "shard_index": shard_index,
                    "shard_count": shard_count,
                    "sample_ids": selected,
                }
            )
            path = destination / "sample_shards" / task_key / f"{shard_index:05d}.json"
            _write_once(path, payload)
            shard_files[(task_key, shard_index)] = (path.resolve(), _sha(payload), shard_count)

    rows: list[dict[str, Any]] = []
    for logical_index, run in enumerate(logical_runs):
        ledger, ledger_sha = ledgers[run.task_key]
        ids = cast("list[str]", ledger["sample_ids"])
        diagnostic_ids = tuple(ids[: min(diagnostic_size, len(ids))])
        config = replace(
            run.config,
            diagnostics=FidelityDiagnosticsConfig(
                enabled=True,
                sample_ids=diagnostic_ids,
                sample_ledger_sha=ledger_sha,
            ),
        )
        config_digest = config_sha256(config)
        shard_count = (len(ids) + shard_size - 1) // shard_size
        logical_run_id = (
            f"{matrix.experiment_id}-{matrix.sha256[:10]}-logical-{logical_index:05d}-"
            f"{config_digest[:12]}"
        )
        for shard_index in range(shard_count):
            array_index = len(rows)
            sample_path, sample_sha, recorded_count = shard_files[(run.task_key, shard_index)]
            if recorded_count != shard_count:
                raise RuntimeError("internal shard-count mismatch")
            config_path = (
                destination
                / "configs"
                / f"{array_index:06d}-{config_digest}-{shard_index:05d}.yaml"
            )
            config_payload = cast("dict[str, Any]", json.loads(canonical_config_json(config)))
            _write_once(
                config_path,
                cast("str", yaml.safe_dump(config_payload, sort_keys=True)).encode(),
            )
            rows.append(
                {
                    "array_index": array_index,
                    "backend": run.backend_key,
                    "command": "evaluate",
                    "config_path": str(config_path.resolve()),
                    "config_sha": config_digest,
                    "diagnostic_sample_count": len(diagnostic_ids),
                    "experiment_id": matrix.experiment_id,
                    "full_sample_ledger_path": str(ledger_paths[run.task_key]),
                    "full_sample_ledger_sha": ledger_sha,
                    "logical_run_id": logical_run_id,
                    "matrix_revision": matrix.matrix_revision,
                    "matrix_sha": matrix.sha256,
                    "measurement_mode": matrix.run.measurement_mode,
                    "method": config.method.value,
                    "model": config.model.id,
                    "model_key": run.model_key,
                    "output_root": str(resolved_output_root),
                    "retention_ratio": config.cache.retention_ratio,
                    "run_id": f"{logical_run_id}-shard-{shard_index:05d}-of-{shard_count:05d}",
                    "sample_id_file": str(sample_path),
                    "sample_id_file_sha": sample_sha,
                    "selection_seed": run.selection_seed,
                    "shard_count": shard_count,
                    "shard_index": shard_index,
                    "subset_size": None,
                    "sweep": run.sweep_name,
                    "task": run.task_name,
                    "task_key": run.task_key,
                    "timed_repetitions": matrix.performance.timed_repetitions,
                    "trace_directory": str(resolved_output_root / "traces"),
                    "variant": run.variant,
                    "warmups": matrix.performance.warmups,
                }
            )
    index_payload = b"".join(_canonical_bytes(row) for row in rows)
    _write_once(index_path, index_payload)
    manifest = {
        "campaign_schema_version": 1,
        "experiment_id": matrix.experiment_id,
        "index_sha256": _sha(index_payload),
        "job_count": len(rows),
        "logical_run_count": len(logical_runs),
        "matrix_revision": matrix.matrix_revision,
        "matrix_sha": matrix.sha256,
        "sample_ledgers": {
            task: {"path": str(ledger_paths[task]), "sha256": ledgers[task][1]}
            for task in sorted(ledgers)
        },
        "shard_size": shard_size,
        "diagnostic_size": diagnostic_size,
        "sample_limit": sample_limit,
        "filters": {
            "model_keys": sorted(model_keys),
            "task_keys": sorted(task_keys),
            "methods": sorted(methods),
            "retention_ratios": sorted(retention_ratios),
        },
        "output_root": str(resolved_output_root),
        "source_git_sha": source_git_sha,
    }
    _write_once(manifest_path, _canonical_bytes(manifest))
    verify_expanded_index(index_path)
    return index_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--ledger", action="append", default=[])
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=100)
    parser.add_argument("--diagnostic-size", type=int, default=20)
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--model-key", action="append", default=[])
    parser.add_argument("--task-key", action="append", default=[])
    parser.add_argument("--method", action="append", default=[])
    parser.add_argument("--retention-ratio", action="append", type=float, default=[])
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Scratch run root for this campaign (defaults to the matrix run.output_root).",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    index = materialize(
        matrix_path=args.matrix.resolve(),
        ledger_paths=_parse_ledgers(args.ledger),
        destination=args.output_directory.resolve(),
        shard_size=args.shard_size,
        diagnostic_size=args.diagnostic_size,
        resume=bool(args.resume),
        sample_limit=args.sample_limit,
        model_keys=frozenset(args.model_key),
        task_keys=frozenset(args.task_key),
        methods=frozenset(args.method),
        retention_ratios=frozenset(args.retention_ratio),
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {"index_path": str(index), "job_count": verify_expanded_index(index)},
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
