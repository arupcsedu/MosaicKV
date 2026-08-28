#!/usr/bin/env python3
"""Create label-blind, ledger-disjoint retrospective preflight identities."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import yaml

from mosaickv.evaluation.lmms_adapter import prepare_seeded_lmms_tasks
from mosaickv.evaluation.paper_candidate import (
    PAPER_CANDIDATE_MODEL_ID,
    PAPER_CANDIDATE_TASK_ROUTES,
    PAPER_CANDIDATE_TASKS,
)
from mosaickv.evaluation.tasks import default_task_registry

CLASSIFICATION = "retrospective_nonheldout_development_preflight"
EXPERIMENT_ID = "paper_quality_memory_v1_retrospective_preflight"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _write_once(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable preflight artifact: {path}")
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


def _source_state(repository: Path) -> tuple[str, bool]:
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


def _load_mapping(path: Path) -> dict[str, Any] | None:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError):
        return None
    return cast("dict[str, Any]", value) if isinstance(value, dict) else None


def collect_explicit_ledger_identities(
    *, repository: Path, run_root: Path, ignored_root: Path
) -> tuple[dict[str, set[str]], list[dict[str, Any]]]:
    """Conservatively collect IDs from every direct, explicit sample ledger."""

    identities = {task: set() for task in PAPER_CANDIDATE_TASKS}
    sources: list[dict[str, Any]] = []
    directories = {repository / "mosaickv/configs/ledgers"}
    if run_root.is_dir():
        directories.update(
            path
            for path in run_root.rglob("inputs")
            if path.is_dir() and ignored_root not in (path, *path.parents)
        )
    seen: set[Path] = set()
    for directory in sorted(directories):
        for path in sorted(
            (*directory.glob("*.json"), *directory.glob("*.yaml"), *directory.glob("*.yml"))
        ):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            value = _load_mapping(resolved)
            if value is None:
                continue
            task = str(value.get("task", "")).casefold()
            raw_ids = value.get("sample_ids")
            if task not in identities or not isinstance(raw_ids, list):
                continue
            sample_ids = {str(item) for item in raw_ids}
            identities[task].update(sample_ids)
            sources.append(
                {
                    "path": str(resolved),
                    "sha256": _sha(resolved),
                    "task": task,
                    "sample_count": len(sample_ids),
                }
            )
    sources.sort(key=lambda item: (str(item["path"]), str(item["sha256"])))
    return identities, sources


def _protocol(package_root: Path) -> tuple[dict[str, Any], Path]:
    path = (
        package_root
        / "configs/experiments/paper_quality_memory_v1_retrospective_preflight.yaml"
    )
    value = _load_mapping(path)
    if value is None or value.get("experiment_id") != EXPERIMENT_ID:
        raise RuntimeError("wrong retrospective paper-candidate protocol")
    authorization = cast("dict[str, Any]", value["execution_authorization"])
    if authorization.get("retrospective_gpu_preflight_authorized") is not True:
        raise PermissionError("retrospective GPU preflight is not authorized")
    if any(
        authorization.get(key) is not False
        for key in (
            "paper_experiment_authorized",
            "paper_result_generation_authorized",
            "automatic_follow_on_authorized",
        )
    ):
        raise RuntimeError("retrospective protocol authorizes a paper action")
    return value, path


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
        "classification": CLASSIFICATION,
        "canonical_eligible": False,
        "paper_claim_eligible": False,
        "selection_freeze_sha256": protocol_sha,
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise RuntimeError(f"{task} retrospective ledger differs from its protocol")
    sample_ids = value.get("sample_ids")
    if not isinstance(sample_ids, list) or len(sample_ids) != count:
        raise RuntimeError(f"{task} retrospective ledger has the wrong sample count")
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
    source_sha, source_dirty = _source_state(repository)
    if source_dirty:
        raise RuntimeError("retrospective preflight materialization requires a clean worktree")
    protocol_sha = _sha(protocol_path)
    ledgers = {task: output_root / "inputs" / f"{task}.json" for task in PAPER_CANDIDATE_TASKS}

    if not args.validate_only:
        excluded, sources = collect_explicit_ledger_identities(
            repository=repository,
            run_root=run_root,
            ignored_root=output_root,
        )
        source_manifest = {
            "schema_version": 1,
            "classification": CLASSIFICATION,
            "selection_boundary": "every_explicit_sample_ledger_by_task_label",
            "answers_references_scores_or_failure_identities_used": False,
            "source_git_sha": source_sha,
            "excluded_unique_identities": {
                task: len(excluded[task]) for task in PAPER_CANDIDATE_TASKS
            },
            "sources": sources,
        }
        source_manifest_path = output_root / "inputs/explicit_ledger_exclusions.json"
        _write_once(source_manifest_path, _canonical_bytes(source_manifest))
        for task in PAPER_CANDIDATE_TASKS:
            spec = default_task_registry().resolve(task)
            route = PAPER_CANDIDATE_TASK_ROUTES[task]
            _tasks, _prepared, selected, _manager = prepare_seeded_lmms_tasks(
                (spec,),
                seed=seed,
                subset_size=None,
                model_name=PAPER_CANDIDATE_MODEL_ID,
                dataset_revisions={route.id: route.revision},
                prepare_samples=False,
            )
            selected_ids = tuple(item for item in selected if item not in excluded[task])[:count]
            if len(selected_ids) != count:
                raise RuntimeError(f"{task} has fewer than {count} ledger-disjoint identities")
            payload = {
                "schema_version": 1,
                "task": task,
                "dataset_id": route.id,
                "dataset_revision": route.revision,
                "split": route.split,
                "selection_seed": seed,
                "materialization_model_name": PAPER_CANDIDATE_MODEL_ID,
                "lmms_eval_version": version("lmms-eval"),
                "source_git_sha": source_sha,
                "source_git_dirty": False,
                "classification": CLASSIFICATION,
                "canonical_eligible": False,
                "paper_claim_eligible": False,
                "selection_rule": (
                    "first_N_in_seeded_task_order_after_excluding_every_explicit_ledger_ID"
                ),
                "prior_outputs_exist_for_complete_split": True,
                "excluded_unique_sample_count": len(excluded[task]),
                "exclusion_manifest": str(source_manifest_path),
                "exclusion_manifest_sha256": _sha(source_manifest_path),
                "selection_freeze": str(protocol_path.resolve()),
                "selection_freeze_sha256": protocol_sha,
                "sample_count": len(selected_ids),
                "sample_ids": list(selected_ids),
            }
            _write_once(ledgers[task], _canonical_bytes(payload))

    records = [
        _validate_ledger(
            ledgers[task],
            task=task,
            count=count,
            seed=seed,
            source_sha=source_sha,
            protocol_sha=protocol_sha,
        )
        for task in PAPER_CANDIDATE_TASKS
    ]
    exclusion_path = output_root / "inputs/explicit_ledger_exclusions.json"
    manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "classification": CLASSIFICATION,
        "source_git_sha": source_sha,
        "source_git_dirty": False,
        "protocol": str(protocol_path.resolve()),
        "protocol_sha256": protocol_sha,
        "candidate_config_sha256": _sha(
            package_root / "configs/paper_candidate_llava_v1.yaml"
        ),
        "exclusion_manifest": str(exclusion_path),
        "exclusion_manifest_sha256": _sha(exclusion_path),
        "tasks": records,
        "paper_claim_eligible": False,
        "paper_experiment_authorized": False,
    }
    manifest_path = output_root / "inputs/preflight_manifest.json"
    encoded = _canonical_bytes(manifest)
    if args.validate_only:
        if not manifest_path.is_file() or manifest_path.read_bytes() != encoded:
            raise RuntimeError("retrospective preflight manifest is absent or differs")
    else:
        _write_once(manifest_path, encoded)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
