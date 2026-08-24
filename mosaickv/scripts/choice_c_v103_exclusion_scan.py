#!/usr/bin/env python3
"""Run only the authorized V103 identity/provenance exclusion scan."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, cast

import yaml
from validate_choice_c_v103_protocol import validate as validate_protocol

from mosaickv.evaluation.disjoint_samples import (
    DatasetContract,
    SampleProtocolError,
    scan_exclusion_sources,
    sha256_file,
    write_once,
)

EXPERIMENT_ID = "choice_c_v103_fresh_identity_calibration"
SEAL_COMMIT = "07958152271b8a8d72b51679cbb92cecf20cc005"
CANONICAL_SPEC = Path("configs/experiments/choice_c_v103_fresh_identity_calibration_protocol.yaml")
EXPECTED_OUTPUT = Path(
    "/scratch/djy8hg/runs/mosaickv/choice-c-v103-fresh-identity-calibration-v1/"
    "inputs/exclusion_manifest.json"
)
ALLOWED_CHANGES_AFTER_SEAL = {
    "mosaickv/analysis/choice_c_v103_protocol_draft.md",
    "mosaickv/configs/experiments/choice_c_v103_fresh_identity_calibration_protocol.yaml",
    "mosaickv/configs/ledgers/choice_c_v103_protocol_scan_scope_seal.yaml",
    "mosaickv/scripts/choice_c_v103_exclusion_scan.py",
    "mosaickv/scripts/validate_choice_c_v103_protocol.py",
    "mosaickv/tests/unit/test_choice_c_v103_exclusion_scan.py",
    "mosaickv/tests/unit/test_choice_c_v103_protocol.py",
}


def _load_protocol(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("experiment_id") != EXPERIMENT_ID:
        raise SampleProtocolError("unexpected V103 protocol")
    return cast("dict[str, Any]", value)


def _contracts(protocol: dict[str, Any]) -> tuple[DatasetContract, ...]:
    return tuple(
        DatasetContract(
            task=str(task["task"]),
            dataset_id=str(task["dataset_id"]),
            dataset_config=str(task["dataset_config"]),
            dataset_revision=str(task["dataset_revision"]),
            split=str(task["split"]),
        )
        for task in cast("list[dict[str, Any]]", protocol["tasks"])
    )


def require_scan_request(*, spec_path: Path, output: Path, package_root: Path) -> dict[str, Any]:
    canonical = (package_root / CANONICAL_SPEC).resolve()
    if spec_path.resolve() != canonical:
        raise PermissionError(f"V103 scan requires canonical protocol path: {canonical}")
    if output.resolve() != EXPECTED_OUTPUT.resolve():
        raise PermissionError(f"V103 scan output must be {EXPECTED_OUTPUT}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {output}")
    result = validate_protocol(canonical)
    if (
        result.get("only_local_cpu_exclusion_scan_authorized") is not True
        or result.get("exclusion_scan_authorized") is not True
        or result.get("exclusion_scan_performed") is not False
        or result.get("downstream_actions_authorized") is not False
    ):
        raise PermissionError("V103 protocol does not authorize exactly one unexecuted CPU scan")
    return _load_protocol(canonical)


def _git_snapshot(repository: Path) -> str:
    sha = subprocess.run(
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
    ancestor = subprocess.run(
        ["git", "-C", str(repository), "merge-base", "--is-ancestor", SEAL_COMMIT, sha],
        check=False,
    )
    changed = set(
        subprocess.run(
            ["git", "-C", str(repository), "diff", "--name-only", f"{SEAL_COMMIT}..{sha}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    )
    unexpected = sorted(changed - ALLOWED_CHANGES_AFTER_SEAL)
    if len(sha) != 40 or dirty or ancestor.returncode != 0 or unexpected:
        raise RuntimeError(
            "V103 scan requires a clean committed descendant of the reviewed seal with only "
            f"the scan-authorization update; dirty={bool(dirty)}, unexpected={unexpected}"
        )
    return sha


def _verify_bound_sources(protocol: dict[str, Any], package_root: Path) -> None:
    scan = cast("dict[str, Any]", protocol["fresh_exclusion_scan"])
    for source in cast("list[dict[str, Any]]", scan["mandatory_bound_sources"]):
        path = Path(str(source["path"]))
        resolved = path.resolve() if path.is_absolute() else (package_root / path).resolve()
        if not resolved.is_file() or sha256_file(resolved) != source["sha256"]:
            raise SampleProtocolError(f"mandatory V103 exclusion source differs: {resolved}")


def catalog_paths_to_ignore(
    protocol: dict[str, Any], roots: tuple[Path, ...], package_root: Path
) -> tuple[Path, ...]:
    ignored = {
        path.resolve()
        for root in roots
        for path in root.rglob("*")
        if path.is_file() and "catalog" in path.name.casefold()
    }
    reviewed = cast("dict[str, Any]", protocol["reviewed_id_catalogs"])
    for record in cast("list[dict[str, Any]]", reviewed["catalogs"]):
        path = Path(str(record["path"]))
        ignored.add(path.resolve() if path.is_absolute() else (package_root / path).resolve())
    return tuple(sorted(ignored, key=str))


def verify_payload_coverage(
    protocol: dict[str, Any], payload: dict[str, Any], package_root: Path
) -> None:
    scan = cast("dict[str, Any]", protocol["fresh_exclusion_scan"])
    expected_counts: dict[Path, int] = {}
    for source in cast("list[dict[str, Any]]", scan["mandatory_bound_sources"]):
        count = source.get("expected_unique_identities")
        if count is None:
            continue
        path = Path(str(source["path"]))
        resolved = path.resolve() if path.is_absolute() else (package_root / path).resolve()
        expected_counts[resolved] = int(count)
    observed_counts = {
        Path(str(source["path"])).resolve(): int(source["sample_count"])
        for source in cast("list[dict[str, Any]]", payload.get("sources", []))
        if "path" in source and "sample_count" in source
    }
    if any(observed_counts.get(path) != count for path, count in expected_counts.items()):
        raise SampleProtocolError("V103 manifest does not bind every mandatory V102 ledger")
    if (
        payload.get("excluded_unique_sample_count", -1)
        < scan["required_v102_unique_identity_count"]
    ):
        raise SampleProtocolError("V103 manifest omits mandatory V102 sample identities")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    package_root = Path(__file__).resolve().parents[1]
    repository = package_root.parent
    output = args.output.resolve()
    protocol = require_scan_request(
        spec_path=args.spec.resolve(), output=output, package_root=package_root
    )
    git_sha = _git_snapshot(repository)
    _verify_bound_sources(protocol, package_root)
    scan = cast("dict[str, Any]", protocol["fresh_exclusion_scan"])
    roots = tuple(
        (Path(value).resolve() if Path(value).is_absolute() else (package_root / value).resolve())
        for value in cast("list[str]", scan["recursively_scan_roots"])
    )
    ignored_catalogs = catalog_paths_to_ignore(protocol, roots, package_root)
    payload = scan_exclusion_sources(
        roots=roots,
        contracts=_contracts(protocol),
        source_git_sha=git_sha,
        source_git_dirty=False,
        ignored_paths=(output, *ignored_catalogs),
    )
    if payload.get("schema_version") != 2:
        raise SampleProtocolError("V103 requires a configuration-aware exclusion manifest")
    verify_payload_coverage(protocol, payload, package_root)
    artifact_sha = write_once(output, payload)
    print(json.dumps({"sha256": artifact_sha, "status": "created"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
