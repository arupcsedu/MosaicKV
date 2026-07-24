#!/usr/bin/env python3
"""Materialize an immutable ordered sample ledger from one pinned lmms task split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from importlib.metadata import version
from pathlib import Path
from typing import Any

from mosaickv.evaluation.lmms_adapter import prepare_seeded_lmms_tasks
from mosaickv.evaluation.tasks import default_task_registry


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _write_once(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable sample ledger: {path}")
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
        os.chmod(temporary, 0o444)
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
        raise


def _source_provenance() -> tuple[str, bool]:
    repository = Path(__file__).resolve().parents[2]
    sha = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return sha, bool(status.strip())


def materialize(
    *,
    task_name: str,
    dataset_revision: str,
    model_name: str,
    seed: int,
    output: Path,
    resume: bool,
) -> dict[str, Any]:
    spec = default_task_registry().resolve(task_name)
    if spec.development_lmms_task is None:
        raise ValueError(f"task {task_name!r} is not provided by lmms-eval")
    if not spec.scoring_deterministic:
        raise ValueError(
            f"task {task_name!r} requires a nondeterministic external judge and is not "
            "eligible for this deterministic campaign"
        )
    source_git_sha, source_git_dirty = _source_provenance()
    if source_git_dirty:
        raise RuntimeError("sample ledger materialization requires a clean source worktree")
    if output.exists():
        if not resume:
            raise FileExistsError(f"sample ledger already exists: {output}")
        recorded = json.loads(output.read_text(encoding="utf-8"))
        expected = {
            "task": task_name,
            "dataset_id": spec.dataset_id,
            "dataset_revision": dataset_revision,
            "split": spec.split,
            "selection_seed": seed,
            "materialization_model_name": model_name,
        }
        if any(recorded.get(key) != value for key, value in expected.items()):
            raise ValueError("existing sample ledger has different immutable provenance")
        return {
            "status": "resumed",
            "path": str(output.resolve()),
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "sample_count": len(recorded["sample_ids"]),
        }
    _tasks, _prepared, selected_ids, _manager = prepare_seeded_lmms_tasks(
        (spec,),
        seed=seed,
        subset_size=None,
        model_name=model_name,
        dataset_revisions={spec.dataset_id: dataset_revision},
    )
    if not selected_ids:
        raise RuntimeError(f"pinned task split is empty: {task_name}")
    final_git_sha, final_git_dirty = _source_provenance()
    if final_git_sha != source_git_sha or final_git_dirty:
        raise RuntimeError(
            "source worktree changed during sample ledger materialization; refusing output"
        )
    payload = _canonical_bytes(
        {
            "schema_version": 1,
            "task": task_name,
            "dataset_id": spec.dataset_id,
            "dataset_revision": dataset_revision,
            "split": spec.split,
            "selection_seed": seed,
            "materialization_model_name": model_name,
            "lmms_eval_version": version("lmms-eval"),
            "source_git_sha": source_git_sha,
            "source_git_dirty": source_git_dirty,
            "sample_count": len(selected_ids),
            "sample_ids": list(selected_ids),
        }
    )
    _write_once(output, payload)
    return {
        "status": "materialized",
        "path": str(output.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "sample_count": len(selected_ids),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = materialize(
        task_name=args.task,
        dataset_revision=args.dataset_revision,
        model_name=args.model_name,
        seed=args.seed,
        output=args.output.resolve(),
        resume=bool(args.resume),
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
