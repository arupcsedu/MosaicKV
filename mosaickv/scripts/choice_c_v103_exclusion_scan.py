#!/usr/bin/env python3
"""Run only the authorized V103 identity/provenance exclusion scan."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, cast

import yaml
from validate_choice_c_v103_protocol import validate as validate_protocol

from mosaickv.evaluation.disjoint_samples import (
    DatasetContract,
    DatasetProvenanceResolution,
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
RESOLUTION_LEDGER = Path("configs/ledgers/choice_c_v103_legacy_dataset_config_resolutions.yaml")
ALLOWED_CHANGES_AFTER_SEAL = {
    "mosaickv/analysis/choice_c_v103_protocol_draft.md",
    "mosaickv/configs/experiments/choice_c_v103_fresh_identity_calibration_protocol.yaml",
    "mosaickv/configs/ledgers/choice_c_v103_protocol_scan_scope_seal.yaml",
    "mosaickv/configs/ledgers/choice_c_v103_legacy_dataset_config_resolutions.yaml",
    "mosaickv/scripts/choice_c_v103_exclusion_scan.py",
    "mosaickv/scripts/validate_choice_c_v103_protocol.py",
    "mosaickv/src/mosaickv/evaluation/disjoint_samples.py",
    "mosaickv/tests/unit/test_choice_c_v103_exclusion_scan.py",
    "mosaickv/tests/unit/test_choice_c_v103_protocol.py",
    "mosaickv/tests/unit/test_disjoint_samples.py",
}


def _load_protocol(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("experiment_id") != EXPERIMENT_ID:
        raise SampleProtocolError("unexpected V103 protocol")
    return cast("dict[str, Any]", value)


def _sha256_at_commit(repository: Path, commit: str, relative_path: str) -> str:
    content = subprocess.run(
        ["git", "-C", str(repository), "show", f"{commit}:{relative_path}"],
        check=True,
        capture_output=True,
    ).stdout
    return hashlib.sha256(content).hexdigest()


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
        result.get("exclusion_scan_retry_authorized") is not True
        or result.get("exclusion_scan_authorized") is not True
        or result.get("exclusion_scan_performed") is not False
        or result.get("downstream_actions_authorized") is not False
    ):
        raise PermissionError("V103 protocol does not authorize an exclusion-scan retry")
    return _load_protocol(canonical)


def load_provenance_resolutions(
    protocol: dict[str, Any], package_root: Path, *, require_reviewed: bool
) -> tuple[DatasetProvenanceResolution, ...]:
    boundary = cast("dict[str, Any]", protocol["legacy_dataset_config_resolution"])
    if boundary.get("path") != RESOLUTION_LEDGER.as_posix():
        raise SampleProtocolError("V103 provenance-resolution ledger path differs")
    ledger_path = (package_root / RESOLUTION_LEDGER).resolve()
    ledger = cast("dict[str, Any]", yaml.safe_load(ledger_path.read_text(encoding="utf-8")))
    authorization = cast("dict[str, Any]", ledger["authorization"])
    if require_reviewed:
        if (
            ledger.get("status") != "committed_reviewed_scan_retry_authorized_not_executed"
            or ledger.get("immutable") is not True
            or authorization.get("corrective_commit_created") is not True
            or authorization.get("corrective_commit_reviewed_after_commit") is not True
            or authorization.get("scan_retry_authorized") is not True
        ):
            raise PermissionError(
                "V103 provenance resolution is not committed and retry-authorized"
            )
    elif (
        ledger.get("status") != "authored_uncommitted_unreviewed_scan_retry_disabled"
        or ledger.get("immutable") is not False
        or any(authorization.values())
    ):
        raise SampleProtocolError("V103 draft provenance-resolution authorization differs")
    if any(
        authorization.get(key) is not False
        for key in (
            "catalog_access_authorized",
            "materialization_authorized",
            "disjointness_check_authorized",
            "gpu_or_slurm_authorized",
            "experiment_authorized",
        )
    ):
        raise PermissionError("V103 provenance resolution authorizes a downstream action")

    tasks = {str(task["task"]): task for task in cast("list[dict[str, Any]]", protocol["tasks"])}
    evidence = cast("dict[str, Any]", ledger["protocol_evidence"])
    evidence_path = str(evidence["path"])
    evidence_sha = str(evidence["file_sha256_at_execution"])
    repository = package_root.parent
    commits = cast("list[str]", evidence["source_commits"])
    if commits != [
        "4449e6d673dba5b0f65235f9967f7b6da0b00444",
        "b29eec97d230811070628f15bc966ce26e9f4373",
    ] or any(
        _sha256_at_commit(repository, commit, f"mosaickv/{evidence_path}") != evidence_sha
        for commit in commits
    ):
        raise SampleProtocolError("V103 original execution protocol evidence differs")

    task_evidence = {
        str(record["task"]): record
        for record in cast("list[dict[str, Any]]", evidence["task_contracts"])
    }
    if set(task_evidence) != set(tasks):
        raise SampleProtocolError("V103 provenance task evidence differs")
    for name, record in task_evidence.items():
        task = tasks[name]
        if any(
            record.get(field) != task.get(field)
            for field in ("dataset_id", "dataset_config", "dataset_revision", "split")
        ):
            raise SampleProtocolError("V103 provenance task contract conflicts with V103")
        calibration = Path(str(record["calibration_ledger_path"]))
        if (
            not calibration.is_file()
            or sha256_file(calibration) != record["calibration_ledger_sha256"]
        ):
            raise SampleProtocolError("V103 calibration-ledger provenance differs")

    run_evidence = cast("list[dict[str, Any]]", ledger["run_evidence"])
    if len(run_evidence) != 2:
        raise SampleProtocolError("V103 run-evidence count differs")
    for run in run_evidence:
        if run.get("source_git_dirty") is not False or run.get("spec_sha256") != evidence_sha:
            raise SampleProtocolError("V103 run provenance is not a clean pinned execution")
        source_commit = str(run["source_git_sha"])
        if source_commit not in commits:
            raise SampleProtocolError("V103 run source commit differs")
        for name, raw_record in cast("dict[str, dict[str, Any]]", run["task_evidence"]).items():
            if name not in tasks:
                raise SampleProtocolError("V103 run evidence names an unexpected task")
            record = raw_record
            telemetry_path = Path(str(record["telemetry_path"]))
            warmup_path = Path(str(record["warmup_ledger_path"]))
            task_record = task_evidence[name]
            if (
                not telemetry_path.is_file()
                or sha256_file(telemetry_path) != record["telemetry_sha256"]
                or not warmup_path.is_file()
                or sha256_file(warmup_path) != task_record["warmup_ledger_sha256"]
            ):
                raise SampleProtocolError("V103 run telemetry or warmup-ledger evidence differs")
            telemetry = cast(
                "dict[str, Any]", json.loads(telemetry_path.read_text(encoding="utf-8"))
            )
            warmup = cast("dict[str, Any]", json.loads(warmup_path.read_text(encoding="utf-8")))
            source_snapshot = cast("dict[str, Any]", telemetry.get("source_snapshot", {}))
            if (
                telemetry.get("task") != name
                or telemetry.get("dataset_id") != tasks[name]["dataset_id"]
                or telemetry.get("dataset_revision") != tasks[name]["dataset_revision"]
                or telemetry.get("spec_sha256") != evidence_sha
                or telemetry.get("ledger_sha256") != task_record["calibration_ledger_sha256"]
                or source_snapshot.get("commit") != source_commit
                or warmup.get("task") != name
                or any(
                    warmup.get(field) != tasks[name][field]
                    for field in ("dataset_id", "dataset_config", "dataset_revision", "split")
                )
            ):
                raise SampleProtocolError("V103 run evidence does not support its resolution")

    records = cast("list[dict[str, Any]]", ledger["resolutions"])
    if len(records) != boundary.get("resolution_count") or len(records) != 10:
        raise SampleProtocolError("V103 provenance-resolution count differs")
    resolutions: list[DatasetProvenanceResolution] = []
    paths: set[Path] = set()
    for record in records:
        task_contract = tasks.get(str(record.get("task")))
        if task_contract is None or any(
            record.get(field) != task_contract.get(field)
            for field in ("dataset_id", "dataset_config", "dataset_revision", "split")
        ):
            raise SampleProtocolError("V103 provenance resolution conflicts with task contract")
        path = Path(str(record["source_path"])).resolve()
        if path in paths:
            raise SampleProtocolError(f"duplicate V103 provenance-resolution path: {path}")
        paths.add(path)
        resolutions.append(
            DatasetProvenanceResolution(
                source_path=path,
                source_sha256=str(record["source_sha256"]),
                dataset_id=str(record["dataset_id"]),
                dataset_config=str(record["dataset_config"]),
                dataset_revision=str(record["dataset_revision"]),
                split=str(record["split"]),
            )
        )
    return tuple(resolutions)


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
    protocol: dict[str, Any],
    payload: dict[str, Any],
    package_root: Path,
    resolutions: tuple[DatasetProvenanceResolution, ...],
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
    observed_manifests: set[Path] = set()
    for source in cast("list[dict[str, Any]]", payload.get("sources", [])):
        if "manifest_path" in source:
            observed_manifests.add(Path(str(source["manifest_path"])).resolve())
        observed_manifests.update(
            Path(str(record["path"])).resolve()
            for record in cast("list[dict[str, Any]]", source.get("manifests", []))
        )
    expected_manifests = {resolution.source_path.resolve() for resolution in resolutions}
    if not expected_manifests.issubset(observed_manifests):
        raise SampleProtocolError("V103 manifest did not consume every reviewed resolution")


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
    resolutions = load_provenance_resolutions(protocol, package_root, require_reviewed=True)
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
        provenance_resolutions=resolutions,
    )
    if payload.get("schema_version") != 2:
        raise SampleProtocolError("V103 requires a configuration-aware exclusion manifest")
    verify_payload_coverage(protocol, payload, package_root, resolutions)
    artifact_sha = write_once(output, payload)
    print(json.dumps({"sha256": artifact_sha, "status": "created"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
