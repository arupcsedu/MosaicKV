#!/usr/bin/env python3
"""Validate the authorization-disabled V103 fresh-identity protocol draft."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import yaml

V102_SEAL_SHA = "1f100eee354f553d16e8a41921b7bc0edec559a62a9090bbceb650496c394ab8"
V102_PROTOCOL_SHA = "d68bc737aecdc9383dbb47ee2eea830a1d9aeaf24c22b9973be59cc299e337c3"
V100_PROTOCOL_SHA = "69aed21ee82c5cf338eeb4ecff2a8dd0481d4d04fb378831339da0cfa9effac4"
V72_CONFIG_SHA = "8bb0add49a858df8bfa936bc932e919ce847511cdccd5e3ce5407b91cd1996b9"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(path: Path) -> dict[str, Any]:
    spec = cast("dict[str, Any]", yaml.safe_load(path.read_text(encoding="utf-8")))
    package = Path(__file__).resolve().parents[1]
    expected = {
        "experiment_id": "choice_c_v103_fresh_identity_calibration",
        "classification": "preregistered_draft_unscanned_unmaterialized_unsubmitted",
        "protocol_status": "draft_review_required_all_actions_disabled",
    }
    if any(spec.get(key) != value for key, value in expected.items()):
        raise RuntimeError("V103 is not an authorization-disabled protocol draft")

    authorization = cast("dict[str, Any]", spec["authorization"])
    if any(value is not False for value in authorization.values()):
        raise RuntimeError("V103 draft authorizes an action")
    submission = cast("dict[str, Any]", spec["submission_policy"])
    if any(value is not False for key, value in submission.items() if key != "reason"):
        raise RuntimeError("V103 draft permits submission or materialization")

    boundary = cast("dict[str, Any]", spec["version_boundary"])
    pinned_files = {
        package / str(boundary["v102_negative_seal_path"]): V102_SEAL_SHA,
        package / str(boundary["v102_protocol_path"]): V102_PROTOCOL_SHA,
        package / str(spec["candidate_boundary"]["algorithm_protocol_path"]): V100_PROTOCOL_SHA,
        package / str(spec["candidate_boundary"]["base_config_path"]): V72_CONFIG_SHA,
    }
    if any(
        not file.is_file() or _sha256(file) != expected_sha
        for file, expected_sha in pinned_files.items()
    ):
        raise RuntimeError("V103 predecessor or candidate pin differs")
    if (
        boundary.get("protocol_draft_parent_git_sha") != "3563c669b70cc923eec1d76a4cb0f53bc1144a6a"
        or boundary.get("protocol_commit_git_sha") is not None
        or boundary.get("v102_is_sealed_negative") is not True
        or boundary.get("v102_frozen_candidate_exists") is not False
        or boundary.get("v102_execution_must_remain_disabled") is not True
    ):
        raise RuntimeError("V103 does not preserve the V102 negative boundary")

    candidate_boundary = cast("dict[str, Any]", spec["candidate_boundary"])
    if (
        candidate_boundary.get(
            "v102_answers_references_scores_or_failure_identities_used_for_algorithm_or_hyperparameters"
        )
        is not False
        or candidate_boundary.get(
            "v102_engineering_defects_used_to_define_fail_closed_trace_requirements"
        )
        is not True
        or candidate_boundary.get("algorithm_or_hyperparameter_change_allowed") is not False
        or candidate_boundary.get("single_fixed_candidate_no_search") is not True
    ):
        raise RuntimeError("V103 candidate boundary is scientifically inconsistent")

    freshness = cast("dict[str, Any]", spec["freshness_definition"])
    if (
        freshness.get("canonical_identity_fields")
        != ["dataset_id", "dataset_config", "dataset_revision", "split", "sample_id"]
        or freshness.get("all_v102_ledger_identities_mandatorily_excluded") is not True
        or freshness.get("sample_level_freshness_claimed_before_scan_review") is not False
        or freshness.get("any_ambiguous_relevant_artifact_blocks_materialization") is not True
    ):
        raise RuntimeError("V103 weakens the fresh-identity contract")

    scan = cast("dict[str, Any]", spec["fresh_exclusion_scan"])
    sources = cast("list[dict[str, Any]]", scan["mandatory_bound_sources"])
    expected_sources = {
        "v102_textvqa_complete_selected_ledger": (
            "a0667e45729639b1830a308d97d46662f3e6c6c76ddda97749d0663fc329b242",
            32,
        ),
        "v102_infovqa_complete_selected_ledger": (
            "3308465e540f98dac1003e85a78403968a90ed19bd7d23fe5b2ec6ad8da258c5",
            32,
        ),
        "v102_negative_seal": (V102_SEAL_SHA, None),
    }
    observed_sources = {
        str(source["role"]): (source.get("sha256"), source.get("expected_unique_identities"))
        for source in sources
    }
    if (
        observed_sources != expected_sources
        or scan.get("required_v102_unique_identity_count") != 64
        or scan.get("inspect_answers_references_scores_media_prompts_or_failure_outcomes")
        is not False
        or scan.get("scan_authorized") is not False
        or scan.get("scan_performed") is not False
        or scan.get("manifest_reviewed") is not False
        or scan.get("v103_tooling_compatibility_status") != "not_reviewed_not_authorized"
        or scan.get("v102_specific_authorization_guards_must_not_be_bypassed") is not True
    ):
        raise RuntimeError("V103 exclusion scan is incomplete or authorized")

    tasks = cast("list[dict[str, Any]]", spec["tasks"])
    by_task = {str(task["task"]): task for task in tasks}
    if set(by_task) != {"textvqa_train_calibration", "infovqa"}:
        raise RuntimeError("V103 task set changed")
    for task in tasks:
        if task.get("preflight_sample_count") != 2 or task.get("calibration_sample_count") != 32:
            raise RuntimeError("V103 preflight or calibration size changed")
    text = by_task["textvqa_train_calibration"]
    info = by_task["infovqa"]
    if (text.get("dataset_config"), text.get("split"), text.get("reference_provenance_field")) != (
        "default",
        "train",
        "answers",
    ) or (info.get("dataset_config"), info.get("split")) != ("InfographicVQA", "validation"):
        raise RuntimeError("V103 dataset route or TextVQA provenance changed")

    comparison = cast("dict[str, Any]", spec["controlled_comparison_contract"])
    provenance = cast("dict[str, Any]", spec["required_run_provenance"])
    required_provenance = {
        "git_sha",
        "config_sha",
        "model_id",
        "model_revision",
        "dataset_id",
        "dataset_config",
        "dataset_revision",
        "dataset_split",
        "cuda_runtime_version",
        "nvidia_driver_version",
        "pytorch_version",
        "transformers_version",
        "vllm_version_or_not_used",
        "sglang_version_or_not_used",
        "gpu_type",
        "gpu_count",
        "backend",
        "prefill_attention_implementation",
        "decode_attention_implementation",
        "seed",
        "measurement_type",
        "worktree_dirty",
    }
    if (
        comparison.get("active_kv_bytes_are_the_primary_budget") is not True
        or comparison.get("pair_every_arm_on_the_same_canonical_sample_identity") is not True
        or comparison.get("failed_samples_must_be_preserved_as_error_rows") is not True
        or set(cast("list[str]", provenance["every_row_or_immutable_manifest_must_record"]))
        != required_provenance
        or provenance.get("clean_worktree_required") is not True
        or provenance.get("missing_provenance_fails_the_row") is not True
    ):
        raise RuntimeError("V103 comparison or provenance contract is incomplete")

    selection = cast("dict[str, Any]", spec["deterministic_selection"])
    materialization = cast("dict[str, Any]", spec["ledger_materialization"])
    disjointness = cast("dict[str, Any]", spec["new_disjointness_review"])
    if (
        selection.get("selection_seed") != 496373287
        or selection.get("exclude_before_ranking") is not True
        or selection.get("no_replacement_after_outcome_inspection") is not True
        or materialization.get("total_planned_unique_identities") != 68
        or materialization.get("calibration_ledgers_materialized_before_gpu_preflight") is not False
        or materialization.get("calibration_materialization_requires_sealed_positive_preflight")
        is not True
        or materialization.get("failed_preflight_must_leave_calibration_ledgers_absent") is not True
        or materialization.get("materialization_performed") is not False
        or disjointness.get("require_all_68_selected_identities_unique") is not True
        or disjointness.get("require_v102_64_identity_exclusion_coverage") is not True
        or disjointness.get("checker_authorized") is not False
    ):
        raise RuntimeError("V103 selection or disjointness contract changed")
    preflight_report = cast("dict[str, Any]", disjointness["preflight_report"])
    calibration_report = cast("dict[str, Any]", disjointness["calibration_report"])
    if (
        preflight_report.get("selected_identity_count") != 4
        or preflight_report.get("required_before_gpu_preflight") is not True
        or preflight_report.get("checker_performed") is not False
        or calibration_report.get("selected_identity_count_including_preflight") != 68
        or calibration_report.get("require_zero_overlap_with_preflight_ledgers") is not True
        or calibration_report.get("required_before_full_calibration") is not True
        or calibration_report.get("checker_performed") is not False
    ):
        raise RuntimeError("V103 staged disjointness review changed")

    preflight = cast("dict[str, Any]", spec["gpu_preflight_gate"])
    required_trace_fields = {
        "status",
        "effective_method",
        "used_fullkv_fallback",
        "attempted_active_kv_bytes",
        "hard_active_kv_budget_bytes",
        "probe_decisions",
        "timing_breakdown",
        "byte_budget_materialization",
    }
    if (
        set(cast("list[str]", preflight["every_trace_must_record"])) != required_trace_fields
        or preflight.get("fullkv_fallback_passes_preflight") is not False
        or preflight.get("gpu_preflight_authorized") is not False
    ):
        raise RuntimeError("V103 preflight does not fail closed")

    gates = cast("dict[str, Any]", spec["unchanged_memory_and_fidelity_gates"])
    if (
        gates.get("minimum_active_reduction_fraction_vs_prompt_topk") != 0.30
        or gates.get("minimum_active_plus_residual_reduction_fraction_vs_prompt_topk") != 0.30
        or gates.get("fullkv_answer_noninferiority_margin_vs_prompt_topk") != 0.05
        or gates.get("fullkv_token_noninferiority_margin_vs_prompt_topk") != 0.05
        or gates.get("require_every_task") is not True
        or gates.get("require_every_fold") is not True
        or gates.get("require_all_rows_complete") is not True
    ):
        raise RuntimeError("V103 weakens an unchanged scientific gate")

    outputs = cast("dict[str, Any]", spec["outputs"])
    return {
        "experiment_id": spec["experiment_id"],
        "status": "valid_authorization_disabled_v103_draft",
        "selection_seed": selection["selection_seed"],
        "tasks": {name: 32 for name in sorted(by_task)},
        "preflight_samples": 4,
        "calibration_samples": 64,
        "planned_unique_identities": materialization["total_planned_unique_identities"],
        "minimum_v102_exclusions": scan["required_v102_unique_identity_count"],
        "output_root": outputs["output_root"],
        "all_actions_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.spec.resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
