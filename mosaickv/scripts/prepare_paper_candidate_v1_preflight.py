#!/usr/bin/env python3
"""Materialize or validate fresh immutable ledgers for paper-candidate preflight v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, cast

import yaml
from materialize_sample_ledger import _write_once, materialize

from mosaickv.evaluation.paper_candidate import (
    PAPER_CANDIDATE_MODEL_ID,
    PAPER_CANDIDATE_TASK_ROUTES,
    PAPER_CANDIDATE_TASKS,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _head(repository: Path) -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_clean(repository: Path) -> None:
    status = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=all"),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise RuntimeError("paper-candidate preflight requires a clean worktree")


def _protocol(package_root: Path) -> tuple[dict[str, Any], Path]:
    path = package_root / "configs/experiments/paper_quality_memory_v1.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("paper-candidate protocol must be a mapping")
    value = cast("dict[str, Any]", raw)
    if value.get("experiment_id") != "paper_quality_memory_v1":
        raise RuntimeError("wrong paper-candidate protocol")
    authorization = cast("dict[str, Any]", value["execution_authorization"])
    if authorization.get("gpu_preflight_authorized") is not True:
        raise PermissionError("paper-candidate GPU preflight is not authorized")
    if authorization.get("paper_experiment_authorized") is not False:
        raise RuntimeError("preflight protocol must not authorize a paper campaign")
    preflight = cast("dict[str, Any]", value["preflight"])
    if preflight.get("methods") != [
        "full_kv",
        "prompt_attention_topk",
        "mosaickv_full",
    ]:
        raise RuntimeError("preflight method scope changed")
    if preflight.get("automatic_paper_submission") is not False:
        raise RuntimeError("preflight must not automatically submit a paper campaign")
    return value, path


def _ledger_directories(repository: Path, run_root: Path) -> tuple[Path, ...]:
    directories = {repository / "mosaickv/configs/ledgers"}
    if run_root.is_dir():
        directories.update(path for path in run_root.rglob("inputs") if path.is_dir())
    return tuple(sorted(directories))


def _validate_ledger(
    path: Path,
    *,
    task: str,
    count: int,
    seed: int,
    source_sha: str,
    protocol_sha: str,
) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    route = PAPER_CANDIDATE_TASK_ROUTES[task]
    expected = {
        "task": task,
        "dataset_id": route.id,
        "dataset_revision": route.revision,
        "split": route.split,
        "selection_seed": seed,
        "sample_count": count,
        "source_git_sha": source_sha,
        "source_git_dirty": False,
        "canonical_eligible": True,
        "classification": "fresh_disjoint_heldout_quality_validation",
        "selection_freeze_sha256": protocol_sha,
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise RuntimeError(f"{task} ledger violates the immutable preflight protocol")
    sample_ids = value.get("sample_ids")
    if not isinstance(sample_ids, list) or len(sample_ids) != count:
        raise RuntimeError(f"{task} ledger must contain exactly {count} samples")
    if len(set(str(item) for item in sample_ids)) != count:
        raise RuntimeError(f"{task} ledger contains duplicate sample identities")
    return {
        "task": task,
        "path": str(path.resolve()),
        "sha256": _sha(path),
        "sample_ids": [str(item) for item in sample_ids],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[2]
    package_root = repository / "mosaickv"
    run_root = Path("/scratch/djy8hg/runs/mosaickv")
    output_root = args.output_root.resolve()
    protocol, protocol_path = _protocol(package_root)
    preflight = cast("dict[str, Any]", protocol["preflight"])
    count = int(preflight["sample_count_per_task"])
    seed = int(preflight["selection_seed"])
    _require_clean(repository)
    source_sha = _head(repository)
    protocol_sha = _sha(protocol_path)

    ledger_paths = {
        task: output_root / "inputs" / f"{task}.json" for task in PAPER_CANDIDATE_TASKS
    }
    if not args.validate_only:
        exclusion_ledgers = _ledger_directories(repository, run_root)
        exclusion_results = (run_root,) if run_root.is_dir() else ()
        for task in PAPER_CANDIDATE_TASKS:
            route = PAPER_CANDIDATE_TASK_ROUTES[task]
            materialize(
                task_name=task,
                dataset_revision=route.revision,
                model_name=PAPER_CANDIDATE_MODEL_ID,
                seed=seed,
                output=ledger_paths[task],
                resume=False,
                subset_size=count,
                exclude_ledger_directories=exclusion_ledgers,
                exclude_result_directories=exclusion_results,
                classification="fresh_disjoint_heldout_quality_validation",
                selection_freeze=protocol_path,
            )

    records = [
        _validate_ledger(
            ledger_paths[task],
            task=task,
            count=count,
            seed=seed,
            source_sha=source_sha,
            protocol_sha=protocol_sha,
        )
        for task in PAPER_CANDIDATE_TASKS
    ]
    manifest_path = output_root / "inputs/preflight_manifest.json"
    manifest = {
        "schema_version": 1,
        "experiment_id": "paper_quality_memory_v1_preflight",
        "classification": "fresh_disjoint_heldout_quality_validation",
        "source_git_sha": source_sha,
        "source_git_dirty": False,
        "protocol": str(protocol_path.resolve()),
        "protocol_sha256": protocol_sha,
        "candidate_config_sha256": _sha(
            package_root / "configs/paper_candidate_llava_v1.yaml"
        ),
        "tasks": records,
        "paper_experiment_authorized": False,
    }
    encoded = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if args.validate_only:
        if not manifest_path.is_file() or manifest_path.read_bytes() != encoded:
            raise RuntimeError("immutable preflight manifest is absent or differs")
    else:
        _write_once(manifest_path, encoded)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
