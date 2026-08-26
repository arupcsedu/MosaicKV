#!/usr/bin/env python3
"""Fail-closed materializer for the fresh V107 four-sample preflight."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, cast

import prepare_choice_c_v106_preflight as predecessor
import yaml  # type: ignore[import-untyped]

from mosaickv.evaluation.disjoint_samples import (
    SampleProtocolError,
    check_disjointness,
    materialize_deterministic_ledger,
    sha256_file,
    validate_ledger,
    write_once,
)

EXPERIMENT_ID = "choice_c_v107_feasible_route_preflight"
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = PACKAGE_ROOT / "configs/experiments/choice_c_v107_feasible_route_preflight.yaml"
OUTPUT_ROOT = Path(
    "/scratch/djy8hg/runs/mosaickv/choice-c-v107-feasible-route-preflight-v1"
)
INPUT_NAMES = predecessor.INPUT_NAMES
ADDITIONAL_EXCLUSION_KEY = "additionally_excluded_v106_preflight_ledgers"


def _protocol() -> dict[str, Any]:
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SampleProtocolError("V107 protocol must be a YAML object")
    return cast("dict[str, Any]", value)


def validate_protocol(protocol: dict[str, Any]) -> None:
    """Validate V107 without granting materialization or execution authority."""

    if protocol.get("experiment_id") != EXPERIMENT_ID:
        raise SampleProtocolError("wrong V107 protocol")
    boundary = cast("dict[str, Any]", protocol.get("scientific_boundary"))
    if (
        not isinstance(boundary, dict)
        or boundary.get("predecessor_result") != "sealed_negative"
        or boundary.get("all_v106_planned_sample_identities_consumed") is not True
        or boundary.get(
            "v106_task_labels_answers_scores_or_failure_identities_used_for_v107_design"
        )
        is not False
        or boundary.get("thresholds_weakened") is not False
    ):
        raise SampleProtocolError("V107 scientific boundary is incomplete")
    policy = cast("dict[str, Any]", protocol.get("feasible_route_policy"))
    if (
        not isinstance(policy, dict)
        or policy.get("fullkv_eligible_for_final_compressed_proposals") is not False
        or policy.get("unresolved_route_baseline")
        != "largest_position_feasible_level"
        or policy.get(
            "restore_donor_levels_must_belong_to_filtered_feasible_ladders"
        )
        is not True
        or policy.get(
            "benchmark_labels_references_scores_answers_and_failure_identities_accessed"
        )
        is not False
    ):
        raise SampleProtocolError("V107 feasible-route policy is incomplete")
    downstream = cast("dict[str, Any]", protocol.get("downstream_actions"))
    if not isinstance(downstream, dict):
        raise SampleProtocolError("V107 downstream actions must be explicit")
    non_preflight = {
        key: value
        for key, value in downstream.items()
        if key not in {"preflight_materialization_authorized", "preflight_execution_authorized"}
    }
    if any(value is not False for value in non_preflight.values()):
        raise SampleProtocolError("V107 calibration or held-out work is authorized")


def require_authorization(protocol: dict[str, Any]) -> None:
    """Fail closed until a reviewed clean commit authorizes V107 preflight only."""

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
        raise PermissionError("V107 preflight materialization or execution is not authorized")


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
    spec = predecessor._worker_spec(
        protocol=protocol,
        protocol_sha=protocol_sha,
        git_sha=git_sha,
        exclusion_path=exclusion_path,
        exclusion_sha=exclusion_sha,
        report_path=report_path,
        report_sha=report_sha,
        ledger_paths=ledger_paths,
    )
    spec.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "classification": "preregistered_v107_four_sample_gpu_preflight_not_calibration",
            "v107_protocol_path": str(PROTOCOL.resolve()),
            "v107_protocol_sha256": protocol_sha,
        }
    )
    spec.pop("v106_protocol_path", None)
    spec.pop("v106_protocol_sha256", None)
    candidate = cast("list[dict[str, Any]]", spec["candidates"])[0]
    candidate["id"] = "v107_position_feasible_route_baseline_logit_feedback_restore_donor"
    candidate["rationale"] = (
        "Remove route levels that exceed physical position limits before final planning, "
        "use the largest feasible unresolved-route baseline, and exchange only feasible "
        "FullKV-logit-ranked recipient/donor levels at equal logical bytes."
    )
    gate = cast("dict[str, Any]", spec["gpu_preflight_gate"])
    gate["arms_required_on_identical_samples"] = [
        "full_kv",
        "prompt_attention_topk",
        "v107_position_feasible_route_baseline_logit_feedback_restore_donor",
    ]
    return spec


def validate_outputs(output_root: Path, *, git_sha: str) -> dict[str, Any]:
    protocol = _protocol()
    require_authorization(protocol)
    inputs = output_root / "inputs"
    spec_path = inputs / INPUT_NAMES["worker_spec"]
    spec = predecessor.ledger_support._load_json(spec_path)
    if spec.get("candidate_implementation_git_sha") != git_sha:
        raise SampleProtocolError("V107 worker spec is pinned to a different candidate")
    report_path = inputs / INPUT_NAMES["disjointness"]
    if spec.get("preflight_disjointness_report_sha256") != sha256_file(report_path):
        raise SampleProtocolError("V107 worker spec binds a different disjointness report")
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
    if checked != predecessor.ledger_support._load_json(report_path):
        raise SampleProtocolError("V107 disjointness report failed independent validation")
    if checked.get("status") != "passed":
        raise SampleProtocolError("V107 disjointness did not pass")
    selected: list[str] = []
    for ledger_path in ledgers:
        ledger = predecessor.ledger_support._load_json(ledger_path)
        validate_ledger(ledger)
        if ledger.get("candidate_implementation_git_sha") != git_sha:
            raise SampleProtocolError("V107 ledger is pinned to a different candidate")
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
    git_sha = predecessor.ledger_support._clean_git_sha()
    protocol = _protocol()
    require_authorization(protocol)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite immutable V107 root: {output_root}")
    identity = cast("dict[str, Any]", protocol["fresh_identity_plan"])
    prior_path = Path(str(identity["predecessor_exclusion_manifest"]))
    additional = [
        Path(str(path))
        for path in cast(
            "list[str]", identity[ADDITIONAL_EXCLUSION_KEY]
        )
    ]
    manifest = predecessor.ledger_support.build_augmented_exclusion_manifest(
        prior_manifest=predecessor.ledger_support._load_json(prior_path),
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
                contract=predecessor.ledger_support._contract(entry),
                selection_seed=int(identity["selection_seed"]),
                selection_salt=str(identity["selection_salt"]),
                sample_count=int(identity["sample_count_per_task"]),
                candidate_implementation_git_sha=git_sha,
                materialization_git_sha=git_sha,
                materialization_git_dirty=False,
                protocol_sha256=protocol_sha,
            )
            ledger = predecessor.ledger_support._rewrite_ledger_exclusion_path(
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
        report = predecessor.ledger_support._canonical_disjointness_report(
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
        protocol = _protocol()
        validate_protocol(protocol)
        print(
            json.dumps(
                {
                    "protocol_status": protocol["protocol_status"],
                    "status": "protocol_validated",
                },
                sort_keys=True,
            )
        )
        return 0
    git_sha = predecessor.ledger_support._clean_git_sha()
    result = (
        validate_outputs(args.output_root.resolve(), git_sha=git_sha)
        if args.validate_only
        else materialize(args.output_root.resolve())
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
