#!/usr/bin/env python3
"""Run the fail-closed V101 exclusion, materialization, and disjointness tools."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, cast

import yaml

from mosaickv.evaluation.disjoint_samples import (
    DatasetContract,
    check_disjointness,
    materialize_deterministic_ledger,
    require_protocol_authorization,
    scan_exclusion_sources,
    sha256_file,
    write_once,
)


def _git_provenance(repository: Path) -> tuple[str, bool]:
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


def _load_protocol(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("V101 protocol must contain an object")
    protocol = cast("dict[str, Any]", value)
    if protocol.get("experiment_id") != "choice_c_v101_v100_fresh_disjoint_calibration":
        raise ValueError("unexpected protocol experiment ID")
    return protocol


def _contracts(protocol: dict[str, Any]) -> tuple[DatasetContract, ...]:
    return tuple(
        DatasetContract(
            task=str(task["task"]),
            dataset_id=str(task["dataset_id"]),
            dataset_revision=str(task["dataset_revision"]),
            split=str(task["split"]),
        )
        for task in cast("list[dict[str, Any]]", protocol["tasks"])
    )


def _resolve_protocol_path(package_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (package_root / path).resolve()


def _require_execution_snapshot(
    protocol: dict[str, Any], *, repository: Path
) -> tuple[str, bool]:
    sha, dirty = _git_provenance(repository)
    source = cast("dict[str, Any]", protocol["candidate_source"])
    expected = str(source["tooling_implementation_git_sha"])
    if expected == "pending_post_review_commit":
        raise PermissionError("V101 tooling SHA remains pending its immutable seal")
    if expected == "pending_tooling_commit":
        raise PermissionError("V101 tooling SHA remains pending its immutable seal")
    ancestor = subprocess.run(
        ["git", "-C", str(repository), "merge-base", "--is-ancestor", expected, sha],
        check=False,
    )
    candidate = str(source["candidate_implementation_git_sha"])
    changed = subprocess.run(
        ["git", "-C", str(repository), "diff", "--name-only", f"{candidate}..{sha}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    allowlist = set(cast("list[str]", source["execution_diff_allowlist"]))
    unexpected = sorted(set(changed) - allowlist)
    if dirty or ancestor.returncode != 0 or unexpected:
        raise RuntimeError(
            "V101 requires a clean descendant of its tooling commit and permits only "
            f"allowlisted changes after candidate SHA {candidate}; observed sha={sha}, "
            f"dirty={dirty}, unexpected={unexpected}"
        )
    return sha, dirty


def _task_contract(
    protocol: dict[str, Any], task_name: str
) -> tuple[dict[str, Any], DatasetContract]:
    tasks = cast("list[dict[str, Any]]", protocol["tasks"])
    matches = [task for task in tasks if task.get("task") == task_name]
    if len(matches) != 1:
        raise ValueError(f"V101 has no unique task contract for {task_name!r}")
    task = matches[0]
    return task, DatasetContract(
        task=str(task["task"]),
        dataset_id=str(task["dataset_id"]),
        dataset_revision=str(task["dataset_revision"]),
        split=str(task["split"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    subparsers = parser.add_subparsers(dest="action", required=True)
    scan = subparsers.add_parser("scan")
    scan.add_argument("--output", required=True, type=Path)
    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--task", required=True, choices=("docvqa", "textvqa"))
    materialize.add_argument("--catalog", required=True, type=Path)
    materialize.add_argument("--exclusion-manifest", required=True, type=Path)
    materialize.add_argument("--output", required=True, type=Path)
    check = subparsers.add_parser("check")
    check.add_argument("--exclusion-manifest", required=True, type=Path)
    check.add_argument("--ledger", required=True, action="append", type=Path)
    check.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    package_root = Path(__file__).resolve().parents[1]
    repository = package_root.parent
    spec_path = args.spec.resolve()
    protocol = _load_protocol(spec_path)
    action = str(args.action)
    require_protocol_authorization(protocol, action)
    git_sha, git_dirty = _require_execution_snapshot(protocol, repository=repository)

    if action == "scan":
        exclusion = cast("dict[str, Any]", protocol["required_exclusion_snapshot"])
        roots = tuple(
            _resolve_protocol_path(package_root, str(value))
            for value in cast("list[str]", exclusion["recursively_scan_roots"])
        )
        ignored = {
            Path(str(args.output)).resolve(),
            _resolve_protocol_path(package_root, str(exclusion["disjointness_report_path"])),
            *(
                _resolve_protocol_path(package_root, str(task["ledger_path"]))
                for task in cast("list[dict[str, Any]]", protocol["tasks"])
            ),
        }
        payload = scan_exclusion_sources(
            roots=roots,
            contracts=_contracts(protocol),
            source_git_sha=git_sha,
            source_git_dirty=git_dirty,
            ignored_paths=ignored,
        )
        artifact_sha = write_once(Path(str(args.output)).resolve(), payload)
        print(json.dumps({"status": "created", "sha256": artifact_sha}, sort_keys=True))
        return 0

    if action == "materialize":
        task, contract = _task_contract(protocol, str(args.task))
        rule = cast("dict[str, Any]", protocol["future_materialization_rule"])
        source = cast("dict[str, Any]", protocol["candidate_source"])
        payload = materialize_deterministic_ledger(
            catalog_path=Path(str(args.catalog)).resolve(),
            exclusion_manifest_path=Path(str(args.exclusion_manifest)).resolve(),
            contract=contract,
            selection_seed=int(task["selection_seed"]),
            selection_salt=str(rule["hash_preimage"]).split("_null_", maxsplit=1)[0],
            sample_count=int(task["sample_count"]),
            candidate_implementation_git_sha=str(source["candidate_implementation_git_sha"]),
            materialization_git_sha=git_sha,
            materialization_git_dirty=git_dirty,
            protocol_sha256=sha256_file(spec_path),
        )
        artifact_sha = write_once(Path(str(args.output)).resolve(), payload)
        print(json.dumps({"status": "created", "sha256": artifact_sha}, sort_keys=True))
        return 0

    report = check_disjointness(
        exclusion_manifest_path=Path(str(args.exclusion_manifest)).resolve(),
        ledger_paths=tuple(Path(str(path)).resolve() for path in args.ledger),
        validation_git_sha=git_sha,
        validation_git_dirty=git_dirty,
    )
    artifact_sha = write_once(Path(str(args.output)).resolve(), report)
    print(
        json.dumps(
            {"status": report["status"], "sha256": artifact_sha}, sort_keys=True
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
