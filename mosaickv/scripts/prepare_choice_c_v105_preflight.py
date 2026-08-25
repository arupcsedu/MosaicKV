#!/usr/bin/env python3
"""Materialize or validate V105 only after its narrow protocol authorization."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, cast

import prepare_choice_c_v104_preflight as support
import yaml

from mosaickv.evaluation.disjoint_samples import (
    SampleProtocolError,
    check_disjointness,
    materialize_deterministic_ledger,
    sha256_file,
    validate_ledger,
    write_once,
)

EXPERIMENT_ID = "choice_c_v105_reserved_beam_preflight"
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = PACKAGE_ROOT / "configs/experiments/choice_c_v105_reserved_beam_preflight.yaml"
V104_SPEC = Path(
    "/scratch/djy8hg/runs/mosaickv/choice-c-v104-joint-reallocation-preflight-v1/"
    "inputs/gpu_preflight_worker_spec.json"
)
V104_SPEC_SHA256 = "a9b9c500514482577436e1499c246aae9bce71ce9c6e3b5d524c40d2afe47a4b"
OUTPUT_ROOT = Path(
    "/scratch/djy8hg/runs/mosaickv/choice-c-v105-reserved-beam-preflight-v2"
)
INPUT_NAMES = support.INPUT_NAMES


def _protocol() -> dict[str, Any]:
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SampleProtocolError("V105 protocol must be a YAML object")
    return cast("dict[str, Any]", value)


def require_authorization(protocol: dict[str, Any]) -> None:
    """Fail closed until one reviewed update authorizes only the preflight."""

    if protocol.get("experiment_id") != EXPERIMENT_ID:
        raise SampleProtocolError("wrong V105 protocol")
    if protocol.get("protocol_status") != (
        "corrective_retry_materialization_and_gpu_execution_authorized_after_clean_commit"
    ):
        raise PermissionError("V105 materialization is not authorized")
    retry = cast("dict[str, Any]", protocol.get("corrective_retry"))
    if (
        not isinstance(retry, dict)
        or retry.get("retry_id") != 1
        or Path(str(retry.get("output_root"))) != OUTPUT_ROOT
        or retry.get("predecessor_gpu_execution_started") is not False
        or retry.get("selected_identities_remain_unexecuted_and_label_blind") is not True
        or retry.get("candidate_algorithm_or_threshold_changed") is not False
    ):
        raise SampleProtocolError("V105 corrective-retry provenance is incomplete")
    source = cast("dict[str, Any]", protocol["candidate_source"])
    if (
        source.get("implementation_git_sha_resolution")
        != "commit_containing_this_protocol_and_runtime"
        or source.get("source_worktree_must_be_clean") is not True
        or source.get("runtime_behavior_changes_after_implementation_commit_allowed") is not False
    ):
        raise SampleProtocolError("V105 candidate-source freeze is incomplete")
    identity = cast("dict[str, Any]", protocol["fresh_identity_plan"])
    if (
        identity.get("ledger_materialization_authorized") is not True
        or identity.get("gpu_preflight_authorized") is not True
    ):
        raise PermissionError("V105 ledger or GPU preflight authorization is absent")
    downstream = cast("dict[str, Any]", protocol["downstream_actions"])
    if any(value is not False for value in downstream.values()):
        raise SampleProtocolError("V105 downstream calibration or held-out work is authorized")


def _worker_spec(
    *,
    protocol: dict[str, Any],
    protocol_sha: str,
    git_sha: str,
    exclusion_path: Path,
    exclusion_sha: str,
    report_path: Path,
    report_sha: str,
    ledger_paths: dict[str, Path],
) -> dict[str, Any]:
    if sha256_file(V104_SPEC) != V104_SPEC_SHA256:
        raise SampleProtocolError("sealed V104 worker-spec template changed")
    spec = copy.deepcopy(support._load_json(V104_SPEC))
    spec.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "classification": "preregistered_v105_four_sample_gpu_preflight_not_calibration",
            "candidate_implementation_git_sha": git_sha,
            "candidate_source": {
                "candidate_implementation_git_sha": git_sha,
                "runtime_behavior_changes_after_candidate_commit_allowed": False,
            },
            "exclusion_manifest_path": str(exclusion_path.resolve()),
            "exclusion_manifest_sha256": exclusion_sha,
            "preflight_disjointness_report_path": str(report_path.resolve()),
            "preflight_disjointness_report_sha256": report_sha,
            "v105_protocol_path": str(PROTOCOL.resolve()),
            "v105_protocol_sha256": protocol_sha,
        }
    )
    spec.pop("v104_protocol_path", None)
    spec.pop("v104_protocol_sha256", None)
    candidates = cast("list[dict[str, Any]]", spec["candidates"])
    candidate = candidates[0]
    candidate["id"] = "v105_reserved_label_blind_reallocation_beam"
    candidate["rationale"] = (
        "Reserve post-joint probe capacity, rank failed same-byte single-route swaps only "
        "by token/KL/logit stability, and probe bounded width-two/three compositions."
    )
    overrides = cast("dict[str, Any]", candidate["overrides"])
    allocation = cast("dict[str, Any]", protocol["probe_allocation"])
    overrides.update(
        {
            "stable_head_hierarchical_reserved_reallocation_evaluations": allocation[
                "explicitly_reserved_post_joint_evaluations"
            ],
            "stable_head_hierarchical_reallocation_beam_width": allocation[
                "reallocation_beam_width"
            ],
            "stable_head_hierarchical_reallocation_maximum_combination_width": allocation[
                "reallocation_maximum_combination_width"
            ],
        }
    )
    gate = cast("dict[str, Any]", spec["gpu_preflight_gate"])
    gate.update(
        {
            "status": "authorized_after_clean_immutable_commit",
            "gpu_preflight_authorized": True,
            "arms_required_on_identical_samples": [
                "full_kv",
                "prompt_attention_topk",
                "v105_reserved_label_blind_reallocation_beam",
            ],
            "calibration_may_be_authorized_only_after_separate_positive_preflight_review": True,
            "every_trace_must_record": cast(
                "dict[str, Any]", protocol["preflight_gate"]
            )["every_trace_must_record"],
        }
    )
    identity = cast("dict[str, Any]", protocol["fresh_identity_plan"])
    tasks: list[dict[str, Any]] = []
    for entry in cast("list[dict[str, Any]]", identity["tasks"]):
        task = str(entry["task"])
        tasks.append(
            {
                "task": task,
                "task_family": "textvqa" if task == "textvqa_train_calibration" else "ocr_document",
                "dataset_id": entry["dataset_id"],
                "dataset_config": entry["dataset_config"],
                "dataset_revision": entry["dataset_revision"],
                "split": entry["split"],
                "ledger": str(ledger_paths[task].resolve()),
                "sample_count": identity["sample_count_per_task"],
                "selection_seed": identity["selection_seed"],
            }
        )
    spec["tasks"] = tasks
    return spec


def validate_outputs(output_root: Path, *, git_sha: str) -> dict[str, Any]:
    protocol = _protocol()
    require_authorization(protocol)
    inputs = output_root / "inputs"
    spec_path = inputs / INPUT_NAMES["worker_spec"]
    spec = support._load_json(spec_path)
    if spec.get("candidate_implementation_git_sha") != git_sha:
        raise SampleProtocolError("V105 worker spec is pinned to a different candidate")
    report_path = inputs / INPUT_NAMES["disjointness"]
    if spec.get("preflight_disjointness_report_sha256") != sha256_file(report_path):
        raise SampleProtocolError("V105 worker spec binds a different disjointness report")
    ledgers = [
        inputs / INPUT_NAMES["textvqa_train_calibration"],
        inputs / INPUT_NAMES["infovqa"],
    ]
    checked = check_disjointness(
        exclusion_manifest_path=inputs / INPUT_NAMES["exclusion"],
        ledger_paths=ledgers,
        validation_git_sha=git_sha,
        validation_git_dirty=False,
    )
    if checked != support._load_json(report_path) or checked.get("status") != "passed":
        raise SampleProtocolError("V105 disjointness report failed independent validation")
    selected: list[str] = []
    for ledger_path in ledgers:
        ledger = support._load_json(ledger_path)
        validate_ledger(ledger)
        if ledger.get("candidate_implementation_git_sha") != git_sha:
            raise SampleProtocolError("V105 ledger is pinned to a different candidate")
        selected.extend(cast("list[str]", ledger["sample_ids"]))
    return {
        "candidate_implementation_git_sha": git_sha,
        "disjointness_status": "passed",
        "gpu_preflight_authorized": True,
        "selected_sample_ids": selected,
        "worker_spec": str(spec_path.resolve()),
        "worker_spec_sha256": sha256_file(spec_path),
    }


def materialize(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    git_sha = support._clean_git_sha()
    protocol = _protocol()
    require_authorization(protocol)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite immutable V105 root: {output_root}")
    identity = cast("dict[str, Any]", protocol["fresh_identity_plan"])
    prior_path = Path(str(identity["predecessor_exclusion_manifest"]))
    additional = [
        Path(str(path))
        for path in cast("list[str]", identity["additionally_excluded_v104_preflight_ledgers"])
    ]
    manifest = support.build_augmented_exclusion_manifest(
        prior_manifest=support._load_json(prior_path),
        prior_manifest_path=prior_path,
        additional_ledger_paths=additional,
        source_git_sha=git_sha,
    )
    protocol_sha = sha256_file(PROTOCOL)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        inputs = staging / "inputs"
        staged_manifest = inputs / INPUT_NAMES["exclusion"]
        write_once(staged_manifest, manifest)
        canonical_manifest = output_root / "inputs" / INPUT_NAMES["exclusion"]
        staged_ledgers: dict[str, Path] = {}
        for entry in cast("list[dict[str, Any]]", identity["tasks"]):
            task = str(entry["task"])
            catalog = Path(str(entry["id_only_catalog"]))
            if sha256_file(catalog) != entry["id_only_catalog_sha256"]:
                raise SampleProtocolError(f"reviewed ID-only catalog changed: {task}")
            ledger = materialize_deterministic_ledger(
                catalog_path=catalog,
                exclusion_manifest_path=staged_manifest,
                contract=support._contract(entry),
                selection_seed=int(identity["selection_seed"]),
                selection_salt=str(identity["selection_salt"]),
                sample_count=int(identity["sample_count_per_task"]),
                candidate_implementation_git_sha=git_sha,
                materialization_git_sha=git_sha,
                materialization_git_dirty=False,
                protocol_sha256=protocol_sha,
            )
            ledger = support._rewrite_ledger_exclusion_path(
                ledger, canonical_manifest_path=canonical_manifest
            )
            path = inputs / INPUT_NAMES[task]
            write_once(path, ledger)
            staged_ledgers[task] = path
        checked = check_disjointness(
            exclusion_manifest_path=staged_manifest,
            ledger_paths=list(staged_ledgers.values()),
            validation_git_sha=git_sha,
            validation_git_dirty=False,
        )
        canonical_ledgers = {
            task: output_root / "inputs" / path.name
            for task, path in staged_ledgers.items()
        }
        report_path = output_root / "inputs" / INPUT_NAMES["disjointness"]
        report = support._canonical_disjointness_report(
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
            exclusion_path=canonical_manifest,
            exclusion_sha=sha256_file(staged_manifest),
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
    return validate_outputs(output_root, git_sha=git_sha)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    git_sha = support._clean_git_sha()
    result = (
        validate_outputs(args.output_root.resolve(), git_sha=git_sha)
        if args.validate_only
        else materialize(args.output_root.resolve())
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
