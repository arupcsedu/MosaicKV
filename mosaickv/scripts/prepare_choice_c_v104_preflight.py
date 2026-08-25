#!/usr/bin/env python3
"""Materialize or validate the authorized V104 four-sample GPU preflight."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import yaml  # type: ignore[import-untyped]

from mosaickv.evaluation.disjoint_samples import (
    DatasetContract,
    SampleIdentity,
    SampleProtocolError,
    canonical_bytes,
    check_disjointness,
    materialize_deterministic_ledger,
    sha256_file,
    validate_exclusion_manifest,
    validate_ledger,
    verify_exclusion_source_files,
    write_once,
)

EXPERIMENT_ID = "choice_c_v104_joint_reallocation_preflight"
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
PROTOCOL = PACKAGE_ROOT / "configs/experiments/choice_c_v104_joint_reallocation_preflight.yaml"
V103_SPEC = Path(
    "/scratch/djy8hg/runs/mosaickv/choice-c-v103-fresh-identity-calibration-v1/"
    "inputs/gpu_preflight_worker_spec.json"
)
OUTPUT_ROOT = Path(
    "/scratch/djy8hg/runs/mosaickv/choice-c-v104-joint-reallocation-preflight-v1"
)
INPUT_NAMES = {
    "exclusion": "exclusion_manifest.json",
    "textvqa_train_calibration": "preflight_textvqa_train.json",
    "infovqa": "preflight_infovqa.json",
    "disjointness": "preflight_disjointness_report.json",
    "worker_spec": "gpu_preflight_worker_spec.json",
}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SampleProtocolError(f"expected a JSON object: {path}")
    return cast("dict[str, Any]", value)


def _load_protocol(path: Path = PROTOCOL) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SampleProtocolError(f"expected a YAML object: {path}")
    return cast("dict[str, Any]", value)


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(PROJECT_ROOT), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _clean_git_sha() -> str:
    status = _git("status", "--porcelain", "--untracked-files=all")
    if status:
        raise SampleProtocolError("V104 materialization requires a clean Git worktree")
    sha = _git("rev-parse", "HEAD")
    if len(sha) != 40:
        raise SampleProtocolError("V104 materialization requires a full Git SHA")
    return sha


def _canonical_identities(values: list[SampleIdentity]) -> tuple[SampleIdentity, ...]:
    return tuple(sorted(set(values), key=SampleIdentity.canonical_key))


def _identity_digest(identities: tuple[SampleIdentity, ...]) -> str:
    return hashlib.sha256(canonical_bytes([item.as_dict() for item in identities])).hexdigest()


def build_augmented_exclusion_manifest(
    *,
    prior_manifest: Mapping[str, Any],
    prior_manifest_path: Path,
    additional_ledger_paths: list[Path],
    source_git_sha: str,
) -> dict[str, Any]:
    """Bind the reviewed exclusions plus every planned V103 preflight identity."""

    identities = list(validate_exclusion_manifest(prior_manifest))
    verify_exclusion_source_files(prior_manifest)
    sources: list[dict[str, Any]] = [
        {
            "kind": "reviewed_prior_exclusion_manifest",
            "path": str(prior_manifest_path.resolve()),
            "sha256": sha256_file(prior_manifest_path),
            "sample_count": len(identities),
            "identities_sha256": _identity_digest(_canonical_identities(identities)),
        }
    ]
    for ledger_path in additional_ledger_paths:
        ledger_identities = validate_ledger(_load_json(ledger_path))
        identities.extend(ledger_identities)
        sources.append(
            {
                "kind": "consumed_v103_preflight_ledger",
                "path": str(ledger_path.resolve()),
                "sha256": sha256_file(ledger_path),
                "sample_count": len(ledger_identities),
                "identities_sha256": _identity_digest(ledger_identities),
            }
        )
    canonical = _canonical_identities(identities)
    if len(canonical) != len(identities):
        raise SampleProtocolError("V103 preflight identities duplicate prior exclusions")
    payload: dict[str, Any] = {
        "schema_version": 2,
        "classification": "v104_augmented_label_blind_sample_exclusion_manifest",
        "source_git_sha": source_git_sha,
        "source_git_dirty": False,
        "scan_roots": cast("list[str]", prior_manifest.get("scan_roots", [])),
        "dataset_contracts": cast(
            "list[dict[str, Any]]", prior_manifest.get("dataset_contracts", [])
        ),
        "source_count": len(sources),
        "sources": sources,
        "excluded_unique_sample_count": len(canonical),
        "excluded_identities": [identity.as_dict() for identity in canonical],
    }
    payload["manifest_content_sha256"] = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    validate_exclusion_manifest(payload)
    return payload


def _contract(entry: Mapping[str, Any]) -> DatasetContract:
    return DatasetContract(
        task=str(entry["task"]),
        dataset_id=str(entry["dataset_id"]),
        dataset_revision=str(entry["dataset_revision"]),
        split=str(entry["split"]),
        dataset_config=str(entry["dataset_config"]),
    )


def _rewrite_ledger_exclusion_path(
    ledger: dict[str, Any], *, canonical_manifest_path: Path
) -> dict[str, Any]:
    payload = dict(ledger)
    payload["exclusion_manifest_path"] = str(canonical_manifest_path.resolve())
    payload.pop("ledger_content_sha256", None)
    payload["ledger_content_sha256"] = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    validate_ledger(payload)
    return payload


def _canonical_disjointness_report(
    *,
    checked: Mapping[str, Any],
    manifest_path: Path,
    ledger_paths: list[Path],
) -> dict[str, Any]:
    payload = dict(checked)
    payload["exclusion_manifest_path"] = str(manifest_path.resolve())
    checked_by_name = {
        Path(str(record["path"])).name: record
        for record in cast("list[dict[str, Any]]", checked["ledgers"])
    }
    payload["ledgers"] = [
        {
            "path": str(path.resolve()),
            "sha256": checked_by_name[path.name]["sha256"],
            "sample_count": checked_by_name[path.name]["sample_count"],
        }
        for path in sorted(ledger_paths, key=lambda item: str(item.resolve()))
    ]
    payload.pop("report_content_sha256", None)
    payload["report_content_sha256"] = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    return payload


def _worker_spec(
    *,
    protocol: Mapping[str, Any],
    protocol_sha: str,
    git_sha: str,
    report_path: Path,
    report_sha: str,
    ledger_paths: Mapping[str, Path],
) -> dict[str, Any]:
    if sha256_file(V103_SPEC) != "03627579a82a6b87f42d1c2f9fedd2899ba962bd725944f7bf174bef6a3d4289":
        raise SampleProtocolError("sealed V103 worker-spec template changed")
    source = _load_json(V103_SPEC)
    spec = copy.deepcopy(source)
    spec.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "classification": "preregistered_v104_four_sample_gpu_preflight_not_calibration",
            "candidate_implementation_git_sha": git_sha,
            "candidate_source": {
                "candidate_implementation_git_sha": git_sha,
                "runtime_behavior_changes_after_candidate_commit_allowed": False,
            },
            "preflight_disjointness_report_path": str(report_path.resolve()),
            "preflight_disjointness_report_sha256": report_sha,
            "v104_protocol_path": str(PROTOCOL.resolve()),
            "v104_protocol_sha256": protocol_sha,
        }
    )
    spec.pop("v103_protocol_path", None)
    spec.pop("v103_protocol_sha256", None)
    candidates = cast("list[dict[str, Any]]", spec["candidates"])
    candidates[0]["id"] = "v104_hierarchical_fixed_capacity_joint_reallocation"
    candidates[0]["rationale"] = (
        "After a failed final joint probe, use only the remaining bounded probe allowance "
        "to test deterministic same-byte route-boundary reallocations in isolated states."
    )
    gate = cast("dict[str, Any]", spec["gpu_preflight_gate"])
    gate.update(
        {
            "status": "authorized_after_clean_immutable_commit",
            "gpu_preflight_authorized": True,
            "arms_required_on_identical_samples": [
                "full_kv",
                "prompt_attention_topk",
                "v104_hierarchical_fixed_capacity_joint_reallocation",
            ],
            "every_trace_must_record": cast(
                "dict[str, Any]", protocol["preflight_gate"]
            )["every_trace_must_record"],
        }
    )
    identity_plan = cast("dict[str, Any]", protocol["fresh_identity_plan"])
    task_entries: list[dict[str, Any]] = []
    for task in cast("list[dict[str, Any]]", identity_plan["tasks"]):
        name = str(task["task"])
        task_entries.append(
            {
                "task": name,
                "task_family": "textvqa" if name == "textvqa_train_calibration" else "ocr_document",
                "dataset_id": task["dataset_id"],
                "dataset_config": task["dataset_config"],
                "dataset_revision": task["dataset_revision"],
                "split": task["split"],
                "ledger": str(ledger_paths[name].resolve()),
                "sample_count": identity_plan["sample_count_per_task"],
                "selection_seed": identity_plan["selection_seed"],
            }
        )
    spec["tasks"] = task_entries
    return spec


def _validate_protocol_authorization(protocol: Mapping[str, Any]) -> None:
    if protocol.get("experiment_id") != EXPERIMENT_ID:
        raise SampleProtocolError("wrong V104 protocol")
    source = cast("Mapping[str, Any]", protocol["candidate_source"])
    if (
        source.get("implementation_git_sha_resolution")
        != "commit_containing_this_protocol_and_runtime"
        or source.get("source_worktree_must_be_clean") is not True
        or source.get("runtime_behavior_changes_after_implementation_commit_allowed") is not False
    ):
        raise SampleProtocolError("V104 candidate-source freeze is incomplete")
    identity = cast("Mapping[str, Any]", protocol["fresh_identity_plan"])
    if (
        identity.get("ledger_materialization_authorized") is not True
        or identity.get("gpu_preflight_authorized") is not True
    ):
        raise SampleProtocolError("V104 preflight is not narrowly authorized")
    downstream = cast("Mapping[str, Any]", protocol["downstream_actions"])
    if any(value is not False for value in downstream.values()):
        raise SampleProtocolError("V104 downstream calibration or held-out work is authorized")


def _validate_outputs(output_root: Path, *, git_sha: str) -> dict[str, Any]:
    inputs = output_root / "inputs"
    protocol = _load_protocol()
    _validate_protocol_authorization(protocol)
    spec = _load_json(inputs / INPUT_NAMES["worker_spec"])
    if (
        spec.get("candidate_implementation_git_sha") != git_sha
        or cast("Mapping[str, Any]", spec.get("candidate_source", {})).get(
            "candidate_implementation_git_sha"
        )
        != git_sha
    ):
        raise SampleProtocolError("V104 worker spec is not pinned to current clean HEAD")
    report_path = inputs / INPUT_NAMES["disjointness"]
    if spec.get("preflight_disjointness_report_sha256") != sha256_file(report_path):
        raise SampleProtocolError("V104 worker spec binds a different disjointness report")
    ledger_paths = [
        inputs / INPUT_NAMES["textvqa_train_calibration"],
        inputs / INPUT_NAMES["infovqa"],
    ]
    checked = check_disjointness(
        exclusion_manifest_path=inputs / INPUT_NAMES["exclusion"],
        ledger_paths=ledger_paths,
        validation_git_sha=git_sha,
        validation_git_dirty=False,
    )
    stored = _load_json(report_path)
    if checked != stored or checked.get("status") != "passed":
        raise SampleProtocolError("V104 disjointness report differs from independent validation")
    selected: list[str] = []
    for task, path in zip(
        ("textvqa_train_calibration", "infovqa"), ledger_paths, strict=True
    ):
        ledger = _load_json(path)
        validate_ledger(ledger)
        if ledger.get("candidate_implementation_git_sha") != git_sha:
            raise SampleProtocolError(f"{task} ledger is pinned to a different candidate")
        selected.extend(cast("list[str]", ledger["sample_ids"]))
    return {
        "candidate_implementation_git_sha": git_sha,
        "disjointness_status": checked["status"],
        "gpu_preflight_authorized": True,
        "selected_sample_ids": selected,
        "worker_spec": str((inputs / INPUT_NAMES["worker_spec"]).resolve()),
        "worker_spec_sha256": sha256_file(inputs / INPUT_NAMES["worker_spec"]),
    }


def materialize(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    git_sha = _clean_git_sha()
    protocol = _load_protocol()
    _validate_protocol_authorization(protocol)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite immutable V104 root: {output_root}")
    identity_plan = cast("dict[str, Any]", protocol["fresh_identity_plan"])
    prior_path = Path(str(identity_plan["predecessor_exclusion_manifest"]))
    prior = _load_json(prior_path)
    additional = [
        Path(str(path))
        for path in cast("list[str]", identity_plan["additionally_excluded_v103_preflight_ledgers"])
    ]
    manifest = build_augmented_exclusion_manifest(
        prior_manifest=prior,
        prior_manifest_path=prior_path,
        additional_ledger_paths=additional,
        source_git_sha=git_sha,
    )
    protocol_sha = sha256_file(PROTOCOL)
    output_parent = output_root.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_parent))
    try:
        inputs = staging / "inputs"
        manifest_path = inputs / INPUT_NAMES["exclusion"]
        write_once(manifest_path, manifest)
        canonical_manifest = output_root / "inputs" / INPUT_NAMES["exclusion"]
        ledgers: dict[str, Path] = {}
        for task in cast("list[dict[str, Any]]", identity_plan["tasks"]):
            task_name = str(task["task"])
            catalog_path = Path(str(task["id_only_catalog"]))
            if sha256_file(catalog_path) != task["id_only_catalog_sha256"]:
                raise SampleProtocolError(f"reviewed ID-only catalog changed: {task_name}")
            ledger = materialize_deterministic_ledger(
                catalog_path=catalog_path,
                exclusion_manifest_path=manifest_path,
                contract=_contract(task),
                selection_seed=int(identity_plan["selection_seed"]),
                selection_salt=str(identity_plan["selection_salt"]),
                sample_count=int(identity_plan["sample_count_per_task"]),
                candidate_implementation_git_sha=git_sha,
                materialization_git_sha=git_sha,
                materialization_git_dirty=False,
                protocol_sha256=protocol_sha,
            )
            ledger = _rewrite_ledger_exclusion_path(
                ledger, canonical_manifest_path=canonical_manifest
            )
            path = inputs / INPUT_NAMES[task_name]
            write_once(path, ledger)
            ledgers[task_name] = path
        checked = check_disjointness(
            exclusion_manifest_path=manifest_path,
            ledger_paths=list(ledgers.values()),
            validation_git_sha=git_sha,
            validation_git_dirty=False,
        )
        canonical_ledgers = {
            task: output_root / "inputs" / path.name for task, path in ledgers.items()
        }
        report_path = output_root / "inputs" / INPUT_NAMES["disjointness"]
        report = _canonical_disjointness_report(
            checked=checked,
            manifest_path=canonical_manifest,
            ledger_paths=list(canonical_ledgers.values()),
        )
        staged_report = inputs / INPUT_NAMES["disjointness"]
        write_once(staged_report, report)
        spec = _worker_spec(
            protocol=protocol,
            protocol_sha=protocol_sha,
            git_sha=git_sha,
            report_path=report_path,
            report_sha=sha256_file(staged_report),
            ledger_paths=canonical_ledgers,
        )
        write_once(inputs / INPUT_NAMES["worker_spec"], spec)
        os.replace(staging, output_root)
        os.chmod(output_root / "inputs", 0o555)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return _validate_outputs(output_root, git_sha=git_sha)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    git_sha = _clean_git_sha()
    result = (
        _validate_outputs(args.output_root.resolve(), git_sha=git_sha)
        if args.validate_only
        else materialize(args.output_root.resolve())
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
