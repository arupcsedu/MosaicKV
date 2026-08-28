#!/usr/bin/env python3
"""Run the separately authorized RealWorldQA identity-only exclusion scan."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, cast

import yaml  # type: ignore[import-untyped]

from mosaickv.evaluation.disjoint_samples import DatasetContract, SampleProtocolError
from mosaickv.evaluation.identity_exclusion_scan import (
    scan_identity_sources,
    write_once_exclusive,
)

SCOPE_COMMIT = "15169135b96161b53bf1f56ab220b6a3f2217e8a"
SCOPE_PATH = Path("configs/ledgers/paper_quality_memory_v2_scan_only_scope.yaml")
SCOPE_SHA256 = "fb46ca1dbc030037b2b92b3182b661bdd0b4138f8eb060fd8f0bb4eb28a48110"
EXPECTED_OUTPUT = Path(
    "/scratch/djy8hg/runs/mosaickv/paper-quality-memory-v2-frozen-99a412b/"
    "inputs/exclusion_manifest_realworldqa.json"
)
AUTHORIZATION_RECORD_TYPE = "paper_quality_memory_v2_exclusion_scan_execution_authorization"
SCANNER_FILES = frozenset(
    {
        "src/mosaickv/evaluation/identity_exclusion_scan.py",
        "scripts/scan_paper_quality_memory_v2_exclusions.py",
        "tests/unit/test_paper_quality_memory_v2_exclusion_scan.py",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_mapping(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SampleProtocolError(f"authorization artifact must be an object: {path}")
    return cast("dict[str, Any]", value)


def load_canonical_scope(package_root: Path) -> dict[str, Any]:
    path = (package_root / SCOPE_PATH).resolve()
    if not path.is_file() or _sha256(path) != SCOPE_SHA256:
        raise SampleProtocolError("RealWorldQA scan scope differs from the reviewed boundary")
    scope = _load_mapping(path)
    reviewed = scope.get("reviewed_route_boundary")
    authorization = scope.get("authorization")
    if (
        scope.get("record_type") != "realworldqa_exclusion_scan_only_scope"
        or not isinstance(reviewed, dict)
        or reviewed.get("route_seal_commit_reviewed_by_user") is not True
        or not isinstance(authorization, dict)
        or authorization.get("exclusion_scan_execution_authorized") is not False
    ):
        raise SampleProtocolError("reviewed RealWorldQA scan scope has an invalid state")
    return scope


def require_execution_authorization(
    *,
    authorization_path: Path,
    scope: dict[str, Any],
    output: Path,
    package_root: Path,
) -> dict[str, Any]:
    """Require a future committed ledger that authorizes only this scan."""

    if output.resolve() != EXPECTED_OUTPUT.resolve():
        raise PermissionError(f"RealWorldQA scan output must be {EXPECTED_OUTPUT}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {output}")
    resolved_authorization = authorization_path.resolve()
    ledger_root = (package_root / "configs/ledgers").resolve()
    if not resolved_authorization.is_file() or not resolved_authorization.is_relative_to(
        ledger_root
    ):
        raise PermissionError("scan authorization must be a committed package ledger")
    authorization = _load_mapping(resolved_authorization)
    auth = authorization.get("authorization")
    if not isinstance(auth, dict):
        raise PermissionError("scan execution authorization is missing")
    if (
        authorization.get("record_type") != AUTHORIZATION_RECORD_TYPE
        or authorization.get("scope_commit_git_sha") != SCOPE_COMMIT
        or authorization.get("scope_file_sha256") != SCOPE_SHA256
        or authorization.get("output") != str(EXPECTED_OUTPUT)
        or authorization.get("reviewed_by_user") is not True
        or auth.get("exclusion_scan_execution_authorized") is not True
    ):
        raise PermissionError("RealWorldQA exclusion scan is not explicitly authorized")
    forbidden = (
        "catalog_access_or_construction_authorized",
        "sample_materialization_authorized",
        "disjointness_check_authorized",
        "model_or_dataset_execution_authorized",
        "gpu_evaluation_authorized",
        "slurm_or_other_job_submission_authorized",
        "result_analysis_authorized",
        "paper_result_claim_authorized",
        "automatic_follow_on_authorized",
    )
    if any(auth.get(field) is not False for field in forbidden):
        raise PermissionError("scan authorization enables a prohibited downstream action")
    scanner = authorization.get("scanner_boundary")
    if not isinstance(scanner, dict):
        raise PermissionError("scan authorization lacks a scanner boundary")
    scanner_commit = scanner.get("commit_git_sha")
    if not isinstance(scanner_commit, str) or len(scanner_commit) != 40:
        raise PermissionError("scan authorization lacks a full scanner commit SHA")
    expected_files = scanner.get("files")
    if not isinstance(expected_files, list) or not expected_files:
        raise PermissionError("scan authorization lacks scanner file hashes")
    observed_files: set[str] = set()
    repository = package_root.parent
    for record in expected_files:
        if not isinstance(record, dict):
            raise PermissionError("invalid scanner file record")
        relative = record.get("path")
        expected_sha = record.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected_sha, str):
            raise PermissionError("invalid scanner file binding")
        observed_files.add(relative)
        path = (package_root / relative).resolve()
        if not path.is_file() or not path.is_relative_to(package_root.resolve()):
            raise PermissionError("scanner file binding escapes the package")
        if _sha256(path) != expected_sha:
            raise PermissionError(f"scanner file differs from authorization: {relative}")
        committed = subprocess.run(
            ["git", "-C", str(repository), "show", f"{scanner_commit}:mosaickv/{relative}"],
            check=True,
            capture_output=True,
        ).stdout
        if hashlib.sha256(committed).hexdigest() != expected_sha:
            raise PermissionError(f"scanner Git object differs from authorization: {relative}")
    if observed_files != SCANNER_FILES:
        raise PermissionError("scan authorization does not bind the exact scanner file set")
    if scope["scan_scope"]["only_permitted_output"] != str(EXPECTED_OUTPUT):
        raise SampleProtocolError("scope output differs from the guarded scanner output")
    return authorization


def require_clean_snapshot(
    repository: Path,
    scanner_commit: str,
    authorization_path: Path,
) -> str:
    head = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    scope_ancestor = subprocess.run(
        ["git", "-C", str(repository), "merge-base", "--is-ancestor", SCOPE_COMMIT, head],
        check=False,
    ).returncode
    scanner_ancestor = subprocess.run(
        ["git", "-C", str(repository), "merge-base", "--is-ancestor", scanner_commit, head],
        check=False,
    ).returncode
    scanner_changes = set(
        subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "diff",
                "--name-only",
                f"{SCOPE_COMMIT}..{scanner_commit}",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    )
    expected_scanner_changes = {f"mosaickv/{path}" for path in SCANNER_FILES}
    later_changes = set(
        subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "diff",
                "--name-only",
                f"{scanner_commit}..{head}",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    )
    authorization_relative = authorization_path.resolve().relative_to(repository.resolve())
    expected_scanner_changes.add(authorization_relative.as_posix())
    if (
        dirty
        or len(head) != 40
        or scope_ancestor != 0
        or scanner_ancestor != 0
        or scanner_changes != expected_scanner_changes
        or later_changes != {authorization_relative.as_posix()}
    ):
        raise PermissionError("scan requires a clean committed descendant of reviewed boundaries")
    return head


def _target(scope: dict[str, Any]) -> DatasetContract:
    scan = cast("dict[str, Any]", scope["scan_scope"])
    return DatasetContract(
        task="realworldqa",
        dataset_id=str(scan["dataset_id"]),
        dataset_config=str(scan["dataset_config"]),
        dataset_revision=str(scan["dataset_revision"]),
        split=str(scan["split"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    package_root = Path(__file__).resolve().parents[1]
    repository = package_root.parent
    scope = load_canonical_scope(package_root)
    output = args.output.resolve()
    authorization = require_execution_authorization(
        authorization_path=args.authorization,
        scope=scope,
        output=output,
        package_root=package_root,
    )
    scanner_commit = str(authorization["scanner_boundary"]["commit_git_sha"])
    head = require_clean_snapshot(repository, scanner_commit, args.authorization.resolve())
    roots = tuple(Path(value).resolve() for value in scope["scan_scope"]["allowed_read_roots"])
    payload = scan_identity_sources(
        roots=roots,
        target=_target(scope),
        source_git_sha=head,
        source_git_dirty=False,
        ignored=(output,),
    )
    payload["scope_commit_git_sha"] = SCOPE_COMMIT
    payload["scope_file_sha256"] = SCOPE_SHA256
    payload["scanner_commit_git_sha"] = scanner_commit
    payload["execution_authorization_path"] = str(args.authorization.resolve())
    payload["execution_authorization_sha256"] = _sha256(args.authorization.resolve())
    payload["manifest_content_sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in payload.items() if key != "manifest_content_sha256"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    ).hexdigest()
    artifact_sha = write_once_exclusive(output, payload)
    print(json.dumps({"sha256": artifact_sha, "status": "created"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
