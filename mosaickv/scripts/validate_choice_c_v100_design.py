#!/usr/bin/env python3
"""Validate the V100 protocol and its narrowly authorized GPU debug gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import yaml


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(spec_path: Path) -> dict[str, Any]:
    spec = cast("dict[str, Any]", yaml.safe_load(spec_path.read_text(encoding="utf-8")))
    package_root = Path(__file__).resolve().parents[1]
    required = {
        "experiment_id": "choice_c_v100_hierarchical_head_local_logit_safe_calibration",
        "classification": "preregistered_gpu_debug_verified_calibration_unexecuted",
        "implementation_status": "bounded_hf_probe_loop_gpu_debug_verified",
        "execution_status": "retention_one_and_one_sample_gpu_debug_passed",
        "submission_authorized": False,
        "candidate_freeze_authorized": False,
        "heldout_evaluation_authorized": False,
    }
    if any(spec.get(key) != value for key, value in required.items()):
        raise RuntimeError("V100 must remain GPU-debug-verified and calibration-blocked")

    authorization = cast("dict[str, Any]", spec["gpu_debug_authorization"])
    if authorization != {
        "authorized_scope": "retention_one_parity_then_one_compressed_synthetic_sample",
        "environment_gate": "MOSAICKV_ENABLE_V100_GPU_DEBUG=1",
        "calibration_or_heldout_authorized": False,
        "retention_one_must_pass_before_compressed_debug": True,
        "compressed_debug_must_use_afterok_dependency": True,
    }:
        raise RuntimeError("V100 GPU debug authorization scope changed")

    evidence = cast("dict[str, Any]", spec["gpu_debug_evidence"])
    retention = cast("dict[str, Any]", evidence["retention_one"])
    compressed = cast("dict[str, Any]", evidence["compressed_one_sample"])
    if (
        evidence.get("evidence_scope") != "implementation_validation_smoke_only"
        or evidence.get("paper_quality_claim_authorized") is not False
        or evidence.get("runtime_git_sha")
        != "1f90a5921ad80def808cd0a0165c42166ef9fc71"
        or evidence.get("validation_git_sha")
        != "3d7b23e3a7b76e58c9e9bc31c6cf6cf543fab203"
        or retention.get("slurm_job_id") != "18770404"
        or retention.get("scheduler_state") != "COMPLETED"
        or retention.get("report_sha256")
        != "9dfcae7046413ad33a18357d150a7ff647d1b22c6f612667f2a18b48f3dd5f0f"
        or retention.get("token_agreement") != 1.0
        or retention.get("maximum_logit_difference") != 0.0
        or compressed.get("slurm_job_id") != "18770405"
        or compressed.get("scheduler_state") != "FAILED"
        or compressed.get("runtime_completed_before_postprocessing_failure") is not True
        or compressed.get("postprocessing_failure")
        != "obsolete_selection_configuration_key"
        or compressed.get("corrected_key")
        != "online_uncertainty_routing.hierarchical_gpu_debug_authorized"
        or compressed.get("summary_regenerated_without_gpu_rerun") is not True
        or compressed.get("report_sha256")
        != "9dd138a4f802ed21ff14e39d96f967313fcd6353648fffe09a561f315d517231"
        or compressed.get("artifact_hashes_verified") is not True
        or compressed.get("required_trace_fields_verified") is not True
        or compressed.get("probe_evaluations_used") != 9
        or compressed.get("maximum_probe_evaluations") != 24
        or compressed.get("probe_rounds_used") != 2
        or compressed.get("maximum_probe_rounds") != 6
        or compressed.get("maximum_observed_kl") > compressed.get("maximum_allowed_kl")
        or compressed.get("maximum_observed_logit_difference")
        > compressed.get("maximum_allowed_logit_difference")
        or compressed.get("probe_state_discarded_before_measured_decode") is not True
        or compressed.get("target_reached") is not True
        or compressed.get("final_joint_probe_passed") is not True
        or compressed.get("used_fullkv_fallback") is not False
        or compressed.get("active_kv_bytes") > compressed.get("configured_active_byte_cap")
        or compressed.get("final_decode_active_kv_bytes")
        > compressed.get("configured_active_byte_cap")
        or compressed.get("residual_kv_bytes") != 0
    ):
        raise RuntimeError("V100 GPU debug evidence is incomplete or inconsistent")

    parent = cast("dict[str, Any]", spec["parent_negative_result_seal"])
    parent_path = package_root / str(parent["path"])
    if not parent_path.is_file() or _sha(parent_path) != parent["sha256"]:
        raise RuntimeError("V100 parent V99 negative seal hash differs")
    boundary = cast("dict[str, Any]", spec["scientific_boundary"])
    forbidden = (
        "v99_answers_used_for_design",
        "v99_references_used_for_design",
        "v99_task_scores_used_for_design",
        "v99_failure_identities_used_for_design",
        "v99_generated_token_ids_used_for_design",
        "v99_per_sample_telemetry_used_for_design",
        "benchmark_labels_accessed",
        "official_task_scores_accessed",
    )
    if any(boundary.get(key) is not False for key in forbidden):
        raise RuntimeError("V100 scientific boundary permits label or per-sample leakage")
    if boundary.get("calibration_must_use_fresh_disjoint_ledgers") is not True:
        raise RuntimeError("V100 must require a fresh disjoint calibration ledger")

    policy = cast("dict[str, Any]", spec["unchanged_policy"])
    if (
        policy.get("configured_active_byte_cap") != 28_500_000
        or policy.get("common_block_size_tokens") != 4
        or policy.get("candidate_tier_policy") != "exact_only"
        or any(
            policy.get(key) is not False
            for key in (
                "prototype_merging_enabled",
                "residual_storage_enabled",
                "repair_enabled",
            )
        )
        or policy.get("second_prefill_forbidden") is not True
    ):
        raise RuntimeError("V100 changed the frozen cache or tier policy")

    algorithm = cast("dict[str, Any]", spec["algorithm"])
    fail_closed = cast("dict[str, Any]", algorithm["fail_closed_behavior"])
    if (
        algorithm.get("initial_state") != "exact_fullkv"
        or cast("dict[str, Any]", algorithm["route_capacity_ladders"]).get("nested") is not True
        or cast("dict[str, Any]", algorithm["global_assembly"]).get("solver")
        != "deterministic_multiple_choice_knapsack"
        or cast("dict[str, Any]", algorithm["joint_verification"]).get("required") is not True
        or fail_closed.get("unverified_target_cache_fallback_forbidden") is not True
        or fail_closed.get("record_actual_active_bytes") is not True
    ):
        raise RuntimeError("V100 hierarchy or fail-closed design changed")

    probes = cast("dict[str, Any]", spec["probe_protocol"])
    if (
        probes.get("horizon_steps") != 2
        or probes.get("maximum_forward_kl") != 0.02
        or probes.get("maximum_absolute_logit_difference") != 1.0
        or probes.get("greedy_token_match_required") is not True
        or probes.get("maximum_candidate_cache_evaluations") != 24
        or probes.get("maximum_sequential_decision_rounds") != 6
        or probes.get("fullkv_reference_rollout_count") != 1
        or probes.get("every_sample_must_reach_target") is not True
    ):
        raise RuntimeError("V100 probe limits or unchanged logit gates changed")

    runtime = cast("dict[str, Any]", spec["strict_runtime_configuration"])
    expected_runtime = {
        "stable_head_hierarchical_logit_safe_eviction": True,
        "stable_head_hierarchical_partition_count": 8,
        "stable_head_hierarchical_maximum_refinement_depth": 2,
        "stable_head_hierarchical_maximum_candidate_evaluations": 24,
        "stable_head_hierarchical_maximum_sequential_rounds": 6,
        "stable_head_hierarchical_maximum_rollback_rounds": 2,
        "stable_head_hierarchical_horizon": 2,
        "stable_head_hierarchical_maximum_kl": 0.02,
        "stable_head_hierarchical_maximum_logit_difference": 1.0,
        "mutually_exclusive_with_v98_shortlist": True,
        "mutually_exclusive_with_v99_progressive": True,
    }
    if runtime != expected_runtime:
        raise RuntimeError("V100 strict runtime controls changed")

    calibration = cast("dict[str, Any]", spec["planned_calibration"])
    tasks = cast("list[dict[str, Any]]", calibration["tasks"])
    gates = cast("dict[str, Any]", spec["unchanged_freeze_gates"])
    implementation = cast("dict[str, Any]", spec["implementation_gate_before_submission"])
    submission = cast("dict[str, Any]", spec["submission_policy"])
    completed_cpu_gates = {
        "strict_config_fields_added",
        "hierarchical_search_implemented",
        "route_capacity_ladder_unit_tests_passed",
        "deterministic_partition_tests_passed",
        "multiple_choice_knapsack_tests_passed",
        "atomic_group_invariant_tests_passed",
        "probe_budget_tests_passed",
        "isolated_cache_state_tests_passed",
        "bounded_hf_probe_loop_integrated",
        "mocked_isolated_probe_loop_tests_passed",
        "synthetic_interaction_rollback_tests_passed",
    }
    completed_backend_gates = {
        "retention_one_fullkv_parity_passed",
        "one_sample_gpu_debug_passed",
    }
    if (
        calibration.get("status") != "unmaterialized"
        or calibration.get("arms")
        != ["full_kv", "prompt_attention_topk", "hierarchical_head_local_logit_safe_exact"]
        or {task.get("task") for task in tasks} != {"docvqa", "textvqa"}
        or {task.get("sample_count") for task in tasks} != {32}
        or any(
            task.get("ledger_status") != "must_be_materialized_from_a_future_clean_commit"
            for task in tasks
        )
        or gates.get("minimum_active_reduction_fraction_vs_prompt_topk") != 0.30
        or gates.get("minimum_active_plus_residual_reduction_fraction_vs_prompt_topk") != 0.30
        or gates.get("fullkv_answer_noninferiority_margin_vs_prompt_topk") != 0.05
        or gates.get("fullkv_token_noninferiority_margin_vs_prompt_topk") != 0.05
        or gates.get("minimum_per_sample_group_reservation_utilization") != 0.75
        or gates.get("require_every_sample_to_reach_target") is not True
        or implementation.get("all_conditions_required") is not True
        or any(implementation.get(key) is not True for key in completed_cpu_gates)
        or any(implementation.get(key) is not True for key in completed_backend_gates)
        or implementation.get("clean_commit_created") is not True
        or set(implementation)
        != completed_cpu_gates
        | completed_backend_gates
        | {"clean_commit_created", "all_conditions_required"}
        or any(value is not False for key, value in submission.items() if key.endswith("_allowed"))
    ):
        raise RuntimeError("V100 calibration, implementation, or submission gates changed")
    return {
        "experiment_id": spec["experiment_id"],
        "classification": spec["classification"],
        "task_count": len(tasks),
        "samples_per_task": tasks[0]["sample_count"],
        "maximum_candidate_cache_evaluations": probes["maximum_candidate_cache_evaluations"],
        "maximum_sequential_decision_rounds": probes["maximum_sequential_decision_rounds"],
        "submission_authorized": False,
        "heldout_evaluation_authorized": False,
        "status": "valid_gpu_debug_passed_calibration_blocked_protocol",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.spec.resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
