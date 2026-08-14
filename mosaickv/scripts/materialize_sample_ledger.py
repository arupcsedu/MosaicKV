#!/usr/bin/env python3
"""Materialize an immutable ordered sample ledger from one pinned lmms task split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Sequence
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
    subset_size: int | None = None,
    exclude_ledger_directories: Sequence[Path] = (),
    allow_dirty_exploratory: bool = False,
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
    if source_git_dirty and not allow_dirty_exploratory:
        raise RuntimeError("sample ledger materialization requires a clean source worktree")
    if subset_size is not None and subset_size < 1:
        raise ValueError("subset_size must be positive when provided")

    exclusion_records: list[dict[str, Any]] = []
    excluded_sample_ids: set[str] = set()
    for directory in sorted(path.resolve() for path in exclude_ledger_directories):
        if not directory.is_dir():
            raise ValueError(f"exclusion ledger directory does not exist: {directory}")
        for ledger_path in sorted(directory.glob("*.json")):
            recorded = json.loads(ledger_path.read_text(encoding="utf-8"))
            if (
                recorded.get("task") != task_name
                or recorded.get("dataset_id") != spec.dataset_id
                or recorded.get("dataset_revision") != dataset_revision
                or recorded.get("split") != spec.split
            ):
                continue
            sample_ids = tuple(str(value) for value in recorded.get("sample_ids", ()))
            excluded_sample_ids.update(sample_ids)
            exclusion_records.append(
                {
                    "name": ledger_path.name,
                    "sha256": hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
                    "sample_count": len(sample_ids),
                }
            )
    exclusion_records.sort(key=lambda item: (str(item["name"]), str(item["sha256"])))
    exclusion_manifest_sha = hashlib.sha256(_canonical_bytes(exclusion_records)).hexdigest()
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
            "subset_size": subset_size,
            "exclusion_manifest_sha": exclusion_manifest_sha,
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
        prepare_samples=False,
    )
    selected_ids = tuple(
        sample_id for sample_id in selected_ids if sample_id not in excluded_sample_ids
    )
    if subset_size is not None:
        selected_ids = selected_ids[:subset_size]
    if not selected_ids:
        raise RuntimeError(f"pinned task split is empty: {task_name}")
    if subset_size is not None and len(selected_ids) != subset_size:
        raise RuntimeError(
            f"only {len(selected_ids)} fresh samples remain after exclusions; "
            f"requested {subset_size}"
        )
    final_git_sha, final_git_dirty = _source_provenance()
    if final_git_sha != source_git_sha or final_git_dirty != source_git_dirty:
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
            "classification": (
                "exploratory_dirty_tree_fresh_disjoint_heldout_quality_validation"
                if source_git_dirty
                else "fresh_disjoint_heldout_quality_validation"
            ),
            "canonical_eligible": not source_git_dirty,
            "selection_rule": (
                "first_N_in_seeded_task_order_after_excluding_every_matching_declared_ledger"
            ),
            "subset_size": subset_size,
            "excluded_unique_sample_count": len(excluded_sample_ids),
            "exclusion_manifest_sha": exclusion_manifest_sha,
            "exclusion_ledgers": exclusion_records,
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
    parser.add_argument("--subset-size", type=int)
    parser.add_argument("--exclude-ledger-directory", action="append", default=[], type=Path)
    parser.add_argument("--allow-dirty-exploratory", action="store_true")
    args = parser.parse_args()
    result = materialize(
        task_name=args.task,
        dataset_revision=args.dataset_revision,
        model_name=args.model_name,
        seed=args.seed,
        output=args.output.resolve(),
        resume=bool(args.resume),
        subset_size=args.subset_size,
        exclude_ledger_directories=tuple(args.exclude_ledger_directory),
        allow_dirty_exploratory=bool(args.allow_dirty_exploratory),
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
