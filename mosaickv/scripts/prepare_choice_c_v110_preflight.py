#!/usr/bin/env python3
"""Fail-closed materializer for the fresh V110 four-sample preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import prepare_choice_c_v104_preflight as ledger_support
import prepare_choice_c_v109_preflight as support
import yaml  # type: ignore[import-untyped]

from mosaickv.evaluation.disjoint_samples import SampleProtocolError, sha256_file

EXPERIMENT_ID = "choice_c_v110_adaptive_relative_impact_preflight"
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    PACKAGE_ROOT
    / "configs/experiments/choice_c_v110_adaptive_relative_impact_preflight.yaml"
)
OUTPUT_ROOT = Path(
    "/scratch/djy8hg/runs/mosaickv/choice-c-v110-adaptive-relative-impact-preflight-v1"
)
ADDITIONAL_EXCLUSION_KEY = "additionally_excluded_v109_preflight_ledgers"
_V109_WORKER_SPEC = support._worker_spec


def _protocol() -> dict[str, Any]:
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SampleProtocolError("V110 protocol must be a YAML object")
    return cast("dict[str, Any]", value)


def validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("experiment_id") != EXPERIMENT_ID:
        raise SampleProtocolError("wrong V110 protocol")
    boundary = cast("dict[str, Any]", protocol.get("scientific_boundary"))
    if (
        not isinstance(boundary, dict)
        or boundary.get("predecessor_result") != "sealed_negative"
        or boundary.get("all_v109_planned_sample_identities_consumed") is not True
        or boundary.get(
            "v109_task_labels_answers_scores_or_failure_identities_used_for_v110_design"
        )
        is not False
        or boundary.get("thresholds_weakened") is not False
    ):
        raise SampleProtocolError("V110 scientific boundary is incomplete")
    policy = cast("dict[str, Any]", protocol.get("adaptive_probe_policy"))
    required_true = {
        "activate_only_when_no_partition_passes",
        "refine_high_and_low_impact_branches",
        "relative_impact_uses_fullkv_relative_token_kl_and_logit_only",
        "relative_impact_changes_candidate_order_only",
        "final_candidate_uses_absolute_fullkv_gate",
        "global_and_per_route_bytes_must_remain_unchanged",
    }
    if (
        not isinstance(policy, dict)
        or any(policy.get(key) is not True for key in required_true)
        or policy.get("reclaimed_reallocation_evaluations") != 9
        or policy.get("retained_final_reallocation_evaluations") != 3
        or policy.get("maximum_refinement_frontier_per_round") != 4
        or policy.get("benchmark_labels_answers_references_scores_or_failure_identities_accessed")
        is not False
    ):
        raise SampleProtocolError("V110 adaptive probe policy is incomplete")
    candidate = cast("dict[str, Any]", protocol.get("candidate"))
    if (
        candidate.get("hierarchical_maximum_refinement_depth") != 6
        or candidate.get("hierarchical_maximum_candidate_evaluations") != 24
        or candidate.get("hierarchical_maximum_sequential_rounds") != 6
    ):
        raise SampleProtocolError("V110 bounded hierarchy is incomplete")
    gates = cast("dict[str, Any]", protocol.get("unchanged_memory_and_fidelity_gates"))
    if (
        gates.get("minimum_active_reduction_fraction_vs_prompt_topk") != 0.30
        or gates.get("minimum_active_plus_residual_reduction_fraction_vs_prompt_topk")
        != 0.30
        or gates.get("fullkv_answer_noninferiority_margin_vs_prompt_topk") != 0.05
        or gates.get("fullkv_token_noninferiority_margin_vs_prompt_topk") != 0.05
        or gates.get("maximum_forward_kl") != 0.02
        or gates.get("maximum_absolute_logit_difference") != 1.0
        or gates.get("greedy_token_match_required") is not True
    ):
        raise SampleProtocolError("V110 changed a preregistered gate")
    preflight = cast("dict[str, Any]", protocol.get("preflight_gate"))
    if preflight.get("fallback_trace_must_record") != [
        "fullkv_fallback_accounting",
        "rejected_compressed_proposal_id",
        "rejected_compressed_proposal_accounting",
    ]:
        raise SampleProtocolError("V110 fallback trace contract is incomplete")
    downstream = cast("dict[str, Any]", protocol.get("downstream_actions"))
    if any(
        value is not False
        for key, value in downstream.items()
        if key not in {"preflight_materialization_authorized", "preflight_execution_authorized"}
    ):
        raise SampleProtocolError("V110 calibration or held-out work is authorized")


def require_authorization(protocol: dict[str, Any]) -> None:
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
        raise PermissionError("V110 preflight is not authorized")


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
    spec = _V109_WORKER_SPEC(
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
            "classification": "preregistered_v110_four_sample_gpu_preflight_not_calibration",
            "v110_protocol_path": str(PROTOCOL.resolve()),
            "v110_protocol_sha256": protocol_sha,
        }
    )
    spec.pop("v109_protocol_path", None)
    spec.pop("v109_protocol_sha256", None)
    candidate = cast("list[dict[str, Any]]", spec["candidates"])[0]
    candidate["id"] = "v110_adaptive_relative_impact_restore_donor"
    candidate["rationale"] = (
        "Reclaim nine stranded probes for high/low-impact deep refinement, then "
        "check same-byte evidence-boundary swaps against unchanged FullKV gates."
    )
    overrides = cast("dict[str, Any]", candidate["overrides"])
    overrides["stable_head_hierarchical_maximum_refinement_depth"] = 6
    gate = cast("dict[str, Any]", spec["gpu_preflight_gate"])
    protocol_gate = cast("dict[str, Any]", protocol["preflight_gate"])
    gate.update(
        {
            "status": "authorized_after_clean_immutable_commit",
            "gpu_preflight_authorized": True,
            "arms_required_on_identical_samples": [
                "full_kv",
                "prompt_attention_topk",
                "v110_adaptive_relative_impact_restore_donor",
            ],
            "every_trace_must_record": protocol_gate["every_trace_must_record"],
            "adaptive_trace_must_record": protocol_gate["adaptive_trace_must_record"],
            "accounting_trace_must_record": protocol_gate["accounting_trace_must_record"],
            "calibration_may_be_authorized_only_after_separate_positive_preflight_review": True,
        }
    )
    return spec


def _configure_support() -> None:
    support.EXPERIMENT_ID = EXPERIMENT_ID
    support.PROTOCOL = PROTOCOL
    support.OUTPUT_ROOT = OUTPUT_ROOT
    support.ADDITIONAL_EXCLUSION_KEY = ADDITIONAL_EXCLUSION_KEY
    support._protocol = _protocol
    support.validate_protocol = validate_protocol
    support.require_authorization = require_authorization
    support._worker_spec = _worker_spec


def validate_outputs(output_root: Path, *, git_sha: str) -> dict[str, Any]:
    _configure_support()
    result = support.validate_outputs(output_root, git_sha=git_sha)
    result.pop("v109_protocol_sha256", None)
    result["v110_protocol_sha256"] = sha256_file(PROTOCOL)
    return result


def materialize(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    _configure_support()
    return support.materialize(output_root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-protocol-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    if args.validate_protocol_only:
        protocol = _protocol()
        validate_protocol(protocol)
        print(json.dumps({"status": "protocol_validated"}, sort_keys=True))
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
