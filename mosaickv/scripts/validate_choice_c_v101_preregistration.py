#!/usr/bin/env python3
"""Fail-closed validation for the sealed V101 feasibility-negative protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, cast

import yaml

SOURCE_SHA = "820037afb3cc74a9f949df1555a2441680f9660e"
PARENT_SHA = "69aed21ee82c5cf338eeb4ecff2a8dd0481d4d04fb378831339da0cfa9effac4"
FEASIBILITY_SEAL_GIT_SHA = "9eafc34af15676ac36a0327ff91a492feb6d862d"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_at_commit(repository: Path, commit: str, path: Path) -> str:
    """Hash the sealed Git blob even when later protocols evolve the live file."""

    relative = path.resolve().relative_to(repository.resolve())
    content = subprocess.run(
        ["git", "-C", str(repository), "show", f"{commit}:{relative.as_posix()}"],
        check=True,
        capture_output=True,
    ).stdout
    return hashlib.sha256(content).hexdigest()


def validate(spec_path: Path, *, verify_artifact: bool = False) -> dict[str, Any]:
    spec = cast("dict[str, Any]", yaml.safe_load(spec_path.read_text(encoding="utf-8")))
    package_root = Path(__file__).resolve().parents[1]
    required_top_level = {
        "experiment_id": "choice_c_v101_v100_fresh_disjoint_calibration",
        "classification": (
            "sealed_feasibility_negative_unmaterialized_unsubmitted_label_blind_calibration"
        ),
        "protocol_status": "sealed_feasibility_negative_full_validation_splits_exhausted",
        "selection_rule_frozen_before_materialization": True,
        "exclusion_scan_authorized": False,
        "materialization_authorized": False,
        "disjointness_check_authorized": False,
        "calibration_execution_authorized": False,
        "candidate_freeze_authorized": False,
        "heldout_evaluation_authorized": False,
    }
    if any(spec.get(key) != value for key, value in required_top_level.items()):
        raise RuntimeError("V101 must remain sealed, infeasible, unmaterialized, and unsubmitted")

    source = cast("dict[str, Any]", spec["candidate_source"])
    parent_path = package_root / str(source["v100_protocol_path"])
    if (
        source.get("candidate_implementation_git_sha") != SOURCE_SHA
        or source.get("protocol_base_git_sha") != SOURCE_SHA
        or source.get("preregistration_git_sha") != "c110d1ddbe141f78d18412d2e230502f788c2fff"
        or source.get("v100_protocol_sha256") != PARENT_SHA
        or _sha256(parent_path) != PARENT_SHA
        or source.get("tooling_implementation_git_sha")
        != "25e173af78fb2e748794755575a8c34f98d1dde8"
        or source.get("execution_git_sha_policy")
        != "clean_descendant_of_tooling_commit_with_allowlisted_diff_from_candidate"
        or set(cast("list[str]", source.get("execution_diff_allowlist")))
        != {
            "mosaickv/configs/experiments/choice_c_v101_v100_fresh_disjoint_calibration.yaml",
            "mosaickv/scripts/choice_c_v101_sample_tooling.py",
            "mosaickv/scripts/validate_choice_c_v101_preregistration.py",
            "mosaickv/src/mosaickv/evaluation/disjoint_samples.py",
            "mosaickv/tests/unit/test_choice_c_v101_protocol.py",
            "mosaickv/tests/unit/test_disjoint_samples.py",
        }
        or source.get("runtime_behavior_changes_after_candidate_commit_allowed") is not False
        or source.get("execution_manifest_must_record_candidate_tooling_and_execution_git_shas")
        is not True
    ):
        raise RuntimeError("V101 is not pinned to the verified V100 source and protocol")

    tooling = cast("dict[str, Any]", spec["sample_tooling_implementation"])
    tooling_files = {
        "core_module": "a16507c2cc1e199cd2fb5d6b4d4a9a9389d43f7ffcb78d5a840aa0d788127a8e",
        "command_wrapper": "371212d43c09c0065937a5ea5c4c91f6927e0ddc5906cda3bfaa06decd2753d9",
        "synthetic_regression": "f4f850bf74f5fa0f988a94597a4180adaaaaf97fb81821870af86757878d55c5",
    }
    repository = package_root.parent
    for field, expected_sha in tooling_files.items():
        path = package_root / str(tooling[field])
        if not path.is_file() or _sha256_at_commit(
            repository, FEASIBILITY_SEAL_GIT_SHA, path
        ) != expected_sha:
            raise RuntimeError(f"V101 tooling hash differs for {field}")
        if tooling.get(f"{field}_sha256") != expected_sha:
            raise RuntimeError(f"V101 recorded tooling hash differs for {field}")
    required_true_tooling = (
        "exclusion_scanner_implemented",
        "deterministic_materializer_implemented",
        "disjointness_checker_implemented",
        "candidate_catalog_restricted_to_ids_only",
        "overlap_failures_recorded",
        "immutable_write_once_artifacts",
        "source_hashes_and_clean_git_provenance_required",
        "synthetic_cpu_tests_passed",
        "real_cli_actions_fail_closed_while_unauthorized",
        "dataset_metadata_only_review_performed",
        "real_exclusion_scan_performed",
        "real_output_directory_created",
    )
    required_false_tooling = (
        "real_dataset_access_performed",
        "real_dataset_samples_or_labels_accessed",
        "real_ledger_materialized",
        "slurm_job_submitted",
    )
    if (
        tooling.get("implementation_status")
        != "committed_sealed_synthetic_tested_exclusion_scan_completed"
        or tooling.get("tooling_commit_created") is not True
        or tooling.get("tooling_commit_sha") != "25e173af78fb2e748794755575a8c34f98d1dde8"
        or any(tooling.get(key) is not True for key in required_true_tooling)
        or any(tooling.get(key) is not False for key in required_false_tooling)
    ):
        raise RuntimeError("V101 tooling status is incomplete or claims unauthorized work")
    parent = cast("dict[str, Any]", yaml.safe_load(parent_path.read_text(encoding="utf-8")))
    parent_gate = cast("dict[str, Any]", parent["implementation_gate_before_submission"])
    if (
        parent_gate.get("retention_one_fullkv_parity_passed") is not True
        or parent_gate.get("one_sample_gpu_debug_passed") is not True
        or parent.get("submission_authorized") is not False
        or parent.get("candidate_freeze_authorized") is not False
        or parent.get("heldout_evaluation_authorized") is not False
    ):
        raise RuntimeError("V101 parent lacks verified GPU debug gates or remains unsafe")

    boundary = cast("dict[str, Any]", spec["scientific_boundary"])
    forbidden_access = (
        "prior_answers_used_for_candidate_or_sample_selection",
        "prior_references_used_for_candidate_or_sample_selection",
        "prior_task_scores_used_for_candidate_or_sample_selection",
        "prior_failure_identities_used_for_candidate_or_sample_selection",
        "prior_generated_tokens_used_for_candidate_or_sample_selection",
        "calibration_answers_accessed_before_execution",
        "calibration_references_accessed",
        "calibration_official_task_scores_accessed",
        "benchmark_labels_required_online",
    )
    if (
        any(boundary.get(key) is not False for key in forbidden_access)
        or boundary.get("all_prior_choice_c_runs_used_only_as_sample_exclusion_boundaries")
        is not True
        or boundary.get("method_is_training_free") is not True
    ):
        raise RuntimeError("V101 scientific boundary permits label or prior-outcome leakage")

    tasks = cast("list[dict[str, Any]]", spec["tasks"])
    expected_revisions = {
        "docvqa": "539088ef8a8ada01ac8e2e6d4e372586748a265e",
        "textvqa": "9c0699cd19768ac5ab97568f6b3cbac4c0062884",
    }
    if (
        {str(task.get("task")) for task in tasks} != set(expected_revisions)
        or any(
            task.get("dataset_revision") != expected_revisions[str(task.get("task"))]
            for task in tasks
        )
        or {int(task.get("selection_seed", -1)) for task in tasks} != {20310821}
        or {int(task.get("sample_count", -1)) for task in tasks} != {32}
        or any(
            task.get("ledger_status") != "blocked_zero_eligible_validation_ids" for task in tasks
        )
    ):
        raise RuntimeError("V101 task, revision, seed, or unmaterialized ledger contract changed")

    materialization = cast("dict[str, Any]", spec["future_materialization_rule"])
    exclusion = cast("dict[str, Any]", spec["required_exclusion_snapshot"])
    if (
        materialization.get("authorized") is not False
        or materialization.get("ordering") != "sha256_digest_ascending_then_canonical_sample_id"
        or materialization.get("inspect_media_or_prompt_content_for_selection") is not False
        or materialization.get(
            "inspect_answers_references_scores_or_failure_identity_for_selection"
        )
        is not False
        or materialization.get("choose_first_n_eligible_after_exclusion") is not True
        or materialization.get("fail_if_fewer_than_requested_eligible_samples") is not True
        or materialization.get("blocked_reason")
        != "exclusion_manifest_covers_every_pinned_validation_example"
        or exclusion.get("status") != "materialized_reviewed_read_only_feasibility_negative"
        or exclusion.get("exclude_all_repository_ledgers") is not True
        or exclusion.get("exclude_every_prior_choice_c_input_ledger") is not True
        or exclusion.get("exclude_every_prior_choice_c_observed_result_row") is not True
        or exclusion.get("require_source_path_sha256_and_sample_count_per_exclusion_source")
        is not True
        or exclusion.get("require_zero_overlap_for_each_task") is not True
        or exclusion.get("exclusion_manifest_status") != "reviewed_sealed_read_only"
        or exclusion.get("disjointness_report_status")
        != "not_materialized_not_applicable_zero_eligible_ids"
    ):
        raise RuntimeError("V101 negative materialization or exclusion seal changed")

    seal = cast("dict[str, Any]", spec["feasibility_negative_seal"])
    coverage = cast("list[dict[str, Any]]", seal.get("split_coverage"))
    expected_coverage = {"docvqa": 5349, "textvqa": 5000}
    if (
        seal.get("outcome") != "infeasible_no_untouched_ids_in_either_pinned_validation_split"
        or seal.get("materialization_must_remain_disabled") is not True
        or seal.get("calibration_or_submission_performed") is not False
        or seal.get("labels_references_scores_media_or_failure_identities_accessed_for_review")
        is not False
        or seal.get("review_used_only_ids_provenance_hashes_and_cached_dataset_cardinalities")
        is not True
        or {str(item.get("task")) for item in coverage} != set(expected_coverage)
        or any(
            item.get("pinned_split_cardinality") != expected_coverage[str(item.get("task"))]
            or item.get("excluded_unique_ids") != expected_coverage[str(item.get("task"))]
            or item.get("eligible_unique_ids") != 0
            or item.get("coverage_fraction") != 1.0
            for item in coverage
        )
        or exclusion.get("exclusion_manifest_file_sha256")
        != "39a068c8863ea4afd936ce698d933fff5abed31e6f43d1874f778039ef139ca2"
        or exclusion.get("exclusion_manifest_content_sha256")
        != "ea217365356ea199de1080245618c12502857be4071f001e0fbbd3ce0a06a91b"
        or exclusion.get("exclusion_manifest_source_git_sha")
        != "36659a51354e48c3ad8bdaa983ef530ba15fe842"
        or exclusion.get("exclusion_source_count") != 2410
        or exclusion.get("excluded_unique_sample_count") != 10349
        or exclusion.get("all_bound_source_hashes_verified") is not True
    ):
        raise RuntimeError("V101 feasibility-negative evidence changed")
    if verify_artifact:
        manifest_path = Path(str(exclusion["exclusion_manifest_path"]))
        if not manifest_path.is_file() or _sha256(manifest_path) != exclusion.get(
            "exclusion_manifest_file_sha256"
        ):
            raise RuntimeError("V101 exclusion manifest is absent or differs from its seal")
        manifest = cast("dict[str, Any]", json.loads(manifest_path.read_text(encoding="utf-8")))
        if (
            manifest.get("manifest_content_sha256")
            != exclusion.get("exclusion_manifest_content_sha256")
            or manifest.get("source_git_sha") != exclusion.get("exclusion_manifest_source_git_sha")
            or manifest.get("source_git_dirty") is not False
            or manifest.get("source_count") != exclusion.get("exclusion_source_count")
            or manifest.get("excluded_unique_sample_count")
            != exclusion.get("excluded_unique_sample_count")
            or manifest_path.stat().st_mode & 0o222
        ):
            raise RuntimeError("V101 exclusion manifest content or permissions differ")
        for item in coverage:
            evidence_path = Path(str(item["cardinality_evidence_path"]))
            if not evidence_path.is_file() or _sha256(evidence_path) != item.get(
                "cardinality_evidence_sha256"
            ):
                raise RuntimeError("V101 cached cardinality evidence is absent or differs")
            dataset_info = cast(
                "dict[str, Any]", json.loads(evidence_path.read_text(encoding="utf-8"))
            )
            observed = cast("dict[str, Any]", dataset_info["splits"])["validation"]
            if cast("dict[str, Any]", observed).get("num_examples") != item.get(
                "pinned_split_cardinality"
            ):
                raise RuntimeError("V101 cached split cardinality differs")

    cache = cast("dict[str, Any]", spec["cache_contract"])
    gates = cast("dict[str, Any]", spec["unchanged_memory_and_fidelity_gates"])
    expected_gates: dict[str, Any] = {
        "minimum_active_reduction_fraction_vs_prompt_topk": 0.30,
        "minimum_active_plus_residual_reduction_fraction_vs_prompt_topk": 0.30,
        "fullkv_answer_noninferiority_margin_vs_prompt_topk": 0.05,
        "fullkv_token_noninferiority_margin_vs_prompt_topk": 0.05,
        "require_every_task": True,
        "robust_fold_count": 4,
        "expected_samples_per_fold": 8,
        "require_every_fold": True,
        "maximum_prototype_logical_fraction": 0.0,
        "require_zero_residual_bytes": True,
        "minimum_per_sample_group_reservation_utilization": 0.75,
        "require_every_sample_to_reach_target": True,
        "require_every_final_joint_logit_gate": True,
        "maximum_forward_kl": 0.02,
        "maximum_absolute_logit_difference": 1.0,
        "require_greedy_token_match": True,
        "require_all_rows_complete": True,
        "forbid_nan_or_missing_measurements": True,
    }
    if (
        any(gates.get(key) != value for key, value in expected_gates.items())
        or cache.get("common_block_size_tokens") != 4
        or cache.get("physical_budget_block_size_tokens") != 4
        or cache.get("mandatory_policy") != "last_token"
        or cache.get("candidate_active_byte_cap") != 28_500_000
        or cache.get("candidate_tier_policy") != "exact_only"
        or any(
            cache.get(key) is not False
            for key in (
                "prototype_merging_enabled",
                "residual_storage_enabled",
                "repair_enabled",
            )
        )
        or cache.get("nominal_retention_comparisons_forbidden") is not True
    ):
        raise RuntimeError("V101 weakened the frozen cache, memory, or fidelity contract")

    review = cast("dict[str, Any]", spec["review_gate_before_materialization"])
    submission = cast("dict[str, Any]", spec["submission_policy"])
    if (
        review.get("protocol_reviewed") is not True
        or review.get("protocol_commit_created") is not True
        or review.get("materializer_implemented_and_tested") is not True
        or review.get("exclusion_scanner_implemented_and_tested") is not True
        or review.get("disjointness_checker_implemented_and_tested") is not True
        or review.get("tooling_commit_sealed") is not True
        or review.get("exclusion_scan_scope_reviewed") is not True
        or review.get("exclusion_manifest_reviewed") is not True
        or review.get("exclusion_manifest_hashes_verified") is not True
        or review.get("split_cardinalities_verified") is not True
        or review.get("materialization_feasible") is not False
        or review.get("protocol_terminated_as_feasibility_negative") is not True
        or any(value is not False for key, value in submission.items() if key.endswith("_allowed"))
    ):
        raise RuntimeError("V101 incorrectly authorizes materialization or submission")
    return {
        "experiment_id": spec["experiment_id"],
        "candidate_implementation_git_sha": SOURCE_SHA,
        "tasks": {str(task["task"]): int(task["sample_count"]) for task in tasks},
        "selection_seed": 20310821,
        "active_memory_reduction_gate": 0.30,
        "fidelity_noninferiority_margin": 0.05,
        "exclusion_scan_authorized": False,
        "materialization_authorized": False,
        "calibration_execution_authorized": False,
        "excluded_unique_sample_count": 10349,
        "eligible_unique_sample_count": 0,
        "status": "valid_sealed_feasibility_negative_protocol",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--verify-artifact", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            validate(args.spec.resolve(), verify_artifact=bool(args.verify_artifact)),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
