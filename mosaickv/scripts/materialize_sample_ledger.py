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
    exclude_result_directories: Sequence[Path] = (),
    allow_dirty_exploratory: bool = False,
    classification: str = "fresh_disjoint_heldout_quality_validation",
    selection_freeze: Path | None = None,
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
    allowed_classifications = {
        "fresh_disjoint_heldout_quality_validation",
        "fresh_disjoint_label_blind_calibration",
        "fixed_preregistered_paper_quality_memory",
    }
    if classification not in allowed_classifications:
        raise ValueError(
            "classification must be fresh_disjoint_heldout_quality_validation, "
            "fresh_disjoint_label_blind_calibration, or "
            "fixed_preregistered_paper_quality_memory"
        )
    selection_freeze_record: dict[str, Any] | None = None
    if selection_freeze is not None:
        selection_freeze = selection_freeze.resolve()
        if not selection_freeze.is_file():
            raise ValueError(f"selection freeze does not exist: {selection_freeze}")
        selection_freeze_record = {
            "path": str(selection_freeze),
            "sha256": hashlib.sha256(selection_freeze.read_bytes()).hexdigest(),
        }

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
                    "kind": "sample_ledger",
                    "name": ledger_path.name,
                    "sha256": hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
                    "sample_count": len(sample_ids),
                }
            )
    for directory in sorted(path.resolve() for path in exclude_result_directories):
        if not directory.is_dir():
            raise ValueError(f"exclusion result directory does not exist: {directory}")
        for result_path in sorted(directory.rglob("results.jsonl")):
            manifest_path = result_path.parent / "manifest.json"
            matching_ids: set[str] = set()
            for line_number, line in enumerate(
                result_path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"invalid JSON in {result_path}:{line_number}: {error}"
                    ) from error
                if row.get("task") == task_name and row.get("sample_id") is not None:
                    matching_ids.add(str(row["sample_id"]))
            if not matching_ids:
                continue
            if not manifest_path.is_file():
                raise ValueError(
                    f"matching observed results lack a manifest: {result_path}"
                )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            dataset = manifest.get("dataset", {})
            if (
                dataset.get("id") != spec.dataset_id
                or dataset.get("revision") != dataset_revision
                or dataset.get("split") != spec.split
            ):
                continue
            excluded_sample_ids.update(matching_ids)
            exclusion_records.append(
                {
                    "kind": "observed_results",
                    "name": str(result_path.relative_to(directory)),
                    "sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
                    "manifest_sha256": hashlib.sha256(
                        manifest_path.read_bytes()
                    ).hexdigest(),
                    "sample_count": len(matching_ids),
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
            "classification": (
                f"exploratory_dirty_tree_{classification}"
                if source_git_dirty
                else classification
            ),
            "selection_freeze_sha256": (
                None
                if selection_freeze_record is None
                else selection_freeze_record["sha256"]
            ),
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
                f"exploratory_dirty_tree_{classification}"
                if source_git_dirty
                else classification
            ),
            "canonical_eligible": not source_git_dirty,
            "selection_rule": (
                "first_N_in_seeded_task_order_after_excluding_every_matching_declared_ledger"
            ),
            "subset_size": subset_size,
            "excluded_unique_sample_count": len(excluded_sample_ids),
            "exclusion_manifest_sha": exclusion_manifest_sha,
            "exclusion_ledgers": exclusion_records,
            "selection_freeze": selection_freeze_record,
            "selection_freeze_sha256": (
                None
                if selection_freeze_record is None
                else selection_freeze_record["sha256"]
            ),
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
    parser.add_argument("--exclude-result-directory", action="append", default=[], type=Path)
    parser.add_argument("--allow-dirty-exploratory", action="store_true")
    parser.add_argument(
        "--classification",
        choices=(
            "fresh_disjoint_heldout_quality_validation",
            "fresh_disjoint_label_blind_calibration",
            "fixed_preregistered_paper_quality_memory",
        ),
        default="fresh_disjoint_heldout_quality_validation",
    )
    parser.add_argument("--selection-freeze", type=Path)
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
        exclude_result_directories=tuple(args.exclude_result_directory),
        allow_dirty_exploratory=bool(args.allow_dirty_exploratory),
        classification=str(args.classification),
        selection_freeze=(
            None if args.selection_freeze is None else args.selection_freeze.resolve()
        ),
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
