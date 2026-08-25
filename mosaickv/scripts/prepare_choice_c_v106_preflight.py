#!/usr/bin/env python3
"""Fail-closed materializer for the fresh V106 four-sample preflight."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, cast

import prepare_choice_c_v104_preflight as ledger_support
import yaml

from mosaickv.evaluation.disjoint_samples import (
    SampleProtocolError,
    check_disjointness,
    materialize_deterministic_ledger,
    sha256_file,
    validate_ledger,
    write_once,
)

EXPERIMENT_ID = "choice_c_v106_logit_feedback_restore_donor_preflight"
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    PACKAGE_ROOT
    / "configs/experiments/choice_c_v106_logit_feedback_restore_donor_preflight.yaml"
)
V105_SPEC = Path(
    "/scratch/djy8hg/runs/mosaickv/choice-c-v105-reserved-beam-preflight-v2/"
    "inputs/gpu_preflight_worker_spec.json"
)
V105_SPEC_SHA256 = "2668976809c1e7b34fc3b48c098e19f42bd4bbc69fccfcff289cdd399799ddcb"
OUTPUT_ROOT = Path(
    "/scratch/djy8hg/runs/mosaickv/choice-c-v106-logit-feedback-restore-donor-preflight-v1"
)
INPUT_NAMES = ledger_support.INPUT_NAMES


def _protocol() -> dict[str, Any]:
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SampleProtocolError("V106 protocol must be a YAML object")
    return cast("dict[str, Any]", value)


def validate_protocol(protocol: dict[str, Any]) -> None:
    """Validate the draft without granting execution authority."""

    if protocol.get("experiment_id") != EXPERIMENT_ID:
        raise SampleProtocolError("wrong V106 protocol")
    boundary = cast("dict[str, Any]", protocol.get("scientific_boundary"))
    if (
        not isinstance(boundary, dict)
        or boundary.get("predecessor_result") != "sealed_negative"
        or boundary.get("all_v105_planned_sample_identities_consumed") is not True
        or boundary.get(
            "v105_task_labels_answers_scores_or_failure_identities_used_for_v106_design"
        )
        is not False
        or boundary.get("thresholds_weakened") is not False
    ):
        raise SampleProtocolError("V106 scientific boundary is incomplete")
    policy = cast("dict[str, Any]", protocol.get("restore_donor_policy"))
    if (
        not isinstance(policy, dict)
        or policy.get("exact_logical_bytes_added_must_equal_bytes_removed") is not True
        or policy.get(
            "every_final_proposal_checked_against_every_route_position_limit_before_probe"
        )
        is not True
        or policy.get("benchmark_labels_references_scores_answers_and_failure_identities_accessed")
        is not False
    ):
        raise SampleProtocolError("V106 restore/donor policy is incomplete")
    downstream = cast("dict[str, Any]", protocol.get("downstream_actions"))
    if not isinstance(downstream, dict):
        raise SampleProtocolError("V106 downstream actions must be explicit")
    non_preflight = {
        key: value
        for key, value in downstream.items()
        if key not in {"preflight_materialization_authorized", "preflight_execution_authorized"}
    }
    if any(value is not False for value in non_preflight.values()):
        raise SampleProtocolError("V106 calibration or held-out work is authorized")


def require_authorization(protocol: dict[str, Any]) -> None:
    """Fail closed until a reviewed clean commit authorizes preflight only."""

    validate_protocol(protocol)
    identity = cast("dict[str, Any]", protocol["fresh_identity_plan"])
    gate = cast("dict[str, Any]", protocol["preflight_gate"])
    downstream = cast("dict[str, Any]", protocol["downstream_actions"])
    if (
        protocol.get("protocol_status")
        != "fresh_preflight_materialization_and_gpu_execution_authorized_after_clean_commit"
        or identity.get("ledger_materialization_authorized") is not True
        or identity.get("gpu_preflight_authorized") is not True
        or gate.get("submission_authorized_after_clean_immutable_commit") is not True
        or downstream.get("preflight_materialization_authorized") is not True
        or downstream.get("preflight_execution_authorized") is not True
    ):
        raise PermissionError("V106 preflight materialization or execution is not authorized")


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
    if sha256_file(V105_SPEC) != V105_SPEC_SHA256:
        raise SampleProtocolError("sealed V105 worker-spec template changed")
    spec = copy.deepcopy(ledger_support._load_json(V105_SPEC))
    spec.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "classification": "preregistered_v106_four_sample_gpu_preflight_not_calibration",
            "candidate_implementation_git_sha": git_sha,
            "candidate_source": {
                "candidate_implementation_git_sha": git_sha,
                "runtime_behavior_changes_after_candidate_commit_allowed": False,
            },
            "exclusion_manifest_path": str(exclusion_path.resolve()),
            "exclusion_manifest_sha256": exclusion_sha,
            "preflight_disjointness_report_path": str(report_path.resolve()),
            "preflight_disjointness_report_sha256": report_sha,
            "v106_protocol_path": str(PROTOCOL.resolve()),
            "v106_protocol_sha256": protocol_sha,
        }
    )
    spec.pop("v105_protocol_path", None)
    spec.pop("v105_protocol_sha256", None)
    candidate = cast("list[dict[str, Any]]", spec["candidates"])[0]
    candidate["id"] = "v106_fullkv_logit_feedback_restore_equal_byte_donor"
    candidate["rationale"] = (
        "Use only FullKV-relative screening token/KL/logit feedback to restore the most "
        "output-sensitive route while compressing an equal-byte low-impact donor; reject "
        "every physically infeasible final proposal before probing."
    )
    cast("dict[str, Any]", candidate["overrides"])[
        "stable_head_hierarchical_logit_feedback_restore_donor"
    ] = True
    gate = cast("dict[str, Any]", spec["gpu_preflight_gate"])
    gate.update(
        {
            "status": "authorized_after_clean_immutable_commit",
            "gpu_preflight_authorized": True,
            "arms_required_on_identical_samples": [
                "full_kv",
                "prompt_attention_topk",
                "v106_fullkv_logit_feedback_restore_equal_byte_donor",
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
    spec = ledger_support._load_json(spec_path)
    if spec.get("candidate_implementation_git_sha") != git_sha:
        raise SampleProtocolError("V106 worker spec is pinned to a different candidate")
    report_path = inputs / INPUT_NAMES["disjointness"]
    if spec.get("preflight_disjointness_report_sha256") != sha256_file(report_path):
        raise SampleProtocolError("V106 worker spec binds a different disjointness report")
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
    if checked != ledger_support._load_json(report_path) or checked.get("status") != "passed":
        raise SampleProtocolError("V106 disjointness report failed independent validation")
    selected: list[str] = []
    for ledger_path in ledgers:
        ledger = ledger_support._load_json(ledger_path)
        validate_ledger(ledger)
        if ledger.get("candidate_implementation_git_sha") != git_sha:
            raise SampleProtocolError("V106 ledger is pinned to a different candidate")
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
    git_sha = ledger_support._clean_git_sha()
    protocol = _protocol()
    require_authorization(protocol)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite immutable V106 root: {output_root}")
    identity = cast("dict[str, Any]", protocol["fresh_identity_plan"])
    prior_path = Path(str(identity["predecessor_exclusion_manifest"]))
    additional = [
        Path(str(path))
        for path in cast("list[str]", identity["additionally_excluded_v105_preflight_ledgers"])
    ]
    manifest = ledger_support.build_augmented_exclusion_manifest(
        prior_manifest=ledger_support._load_json(prior_path),
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
                contract=ledger_support._contract(entry),
                selection_seed=int(identity["selection_seed"]),
                selection_salt=str(identity["selection_salt"]),
                sample_count=int(identity["sample_count_per_task"]),
                candidate_implementation_git_sha=git_sha,
                materialization_git_sha=git_sha,
                materialization_git_dirty=False,
                protocol_sha256=protocol_sha,
            )
            ledger = ledger_support._rewrite_ledger_exclusion_path(
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
            task: output_root / "inputs" / path.name for task, path in staged_ledgers.items()
        }
        report_path = output_root / "inputs" / INPUT_NAMES["disjointness"]
        report = ledger_support._canonical_disjointness_report(
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
    parser.add_argument("--validate-protocol-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    if args.validate_protocol_only:
        validate_protocol(_protocol())
        print(json.dumps({"status": "draft_protocol_validated_execution_disabled"}))
        return 0
    git_sha = ledger_support._clean_git_sha()
    result = (
        validate_outputs(args.output_root.resolve(), git_sha=git_sha)
        if args.validate_only
        else materialize(args.output_root.resolve())
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
