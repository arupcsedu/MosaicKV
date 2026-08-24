#!/usr/bin/env python3
"""Validate the authorization-disabled V103 fresh-identity protocol draft."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, cast

import yaml

V102_SEAL_SHA = "1f100eee354f553d16e8a41921b7bc0edec559a62a9090bbceb650496c394ab8"
V102_PROTOCOL_SHA = "d68bc737aecdc9383dbb47ee2eea830a1d9aeaf24c22b9973be59cc299e337c3"
V100_PROTOCOL_SHA = "69aed21ee82c5cf338eeb4ecff2a8dd0481d4d04fb378831339da0cfa9effac4"
V72_CONFIG_SHA = "8bb0add49a858df8bfa936bc932e919ce847511cdccd5e3ce5407b91cd1996b9"
V103_PROTOCOL_COMMIT = "75736bbc115d2ad5d82ae7c7a3a26df41c89d8fe"
V103_PROTOCOL_COMMIT_FILE_SHA = "c87e76a7555aeea7c7722e5b0cd8604134f5da94fad5fa0f156cea0e4ce26e9f"
V103_EXCLUSION_OUTPUT = (
    "/scratch/djy8hg/runs/mosaickv/choice-c-v103-fresh-identity-calibration-v1/"
    "inputs/exclusion_manifest.json"
)
EXPECTED_BOUND_SOURCES = {
    "v102_textvqa_complete_selected_ledger": (
        "a0667e45729639b1830a308d97d46662f3e6c6c76ddda97749d0663fc329b242",
        32,
    ),
    "v102_infovqa_complete_selected_ledger": (
        "3308465e540f98dac1003e85a78403968a90ed19bd7d23fe5b2ec6ad8da258c5",
        32,
    ),
    "v102_negative_seal": (
        "1f100eee354f553d16e8a41921b7bc0edec559a62a9090bbceb650496c394ab8",
        None,
    ),
}
SCAN_ALLOWED_FIELDS = {
    "canonical_sample_identity",
    "dataset_id",
    "dataset_config",
    "dataset_revision",
    "split",
    "sample_id",
    "task_name",
    "source_path",
    "source_sha256",
    "source_record_count",
    "identity_count",
    "schema_version",
}
SCAN_FORBIDDEN_FIELDS = {
    "answer",
    "answers",
    "reference",
    "references",
    "task_score",
    "generated_tokens",
    "logits",
    "media",
    "prompt",
    "error",
    "failure_identity",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_at_commit(repository: Path, commit: str, path: Path) -> str:
    relative = path.resolve().relative_to(repository.resolve())
    content = subprocess.run(
        ["git", "-C", str(repository), "show", f"{commit}:{relative.as_posix()}"],
        check=True,
        capture_output=True,
    ).stdout
    return hashlib.sha256(content).hexdigest()


def validate(path: Path) -> dict[str, Any]:
    spec = cast("dict[str, Any]", yaml.safe_load(path.read_text(encoding="utf-8")))
    package = Path(__file__).resolve().parents[1]
    expected = {
        "experiment_id": "choice_c_v103_fresh_identity_calibration",
        "classification": "preregistered_protocol_commit_reviewed_scan_scope_defined_unexecuted",
        "protocol_status": "scan_scope_seal_authored_pending_commit_all_execution_disabled",
    }
    if any(spec.get(key) != value for key, value in expected.items()):
        raise RuntimeError("V103 is not an authorization-disabled protocol draft")

    authorization = cast("dict[str, Any]", spec["authorization"])
    expected_authorization = {
        "protocol_commit_created": True,
        "protocol_reviewed_after_commit": True,
        "exclusion_scan_authorized": False,
        "id_catalog_reuse_review_authorized": False,
        "preflight_ledger_materialization_authorized": False,
        "calibration_ledger_materialization_authorized": False,
        "disjointness_check_authorized": False,
        "gpu_preflight_authorized": False,
        "calibration_execution_authorized": False,
        "candidate_freeze_authorized": False,
        "heldout_materialization_authorized": False,
        "heldout_evaluation_authorized": False,
        "slurm_or_other_job_submission_authorized": False,
    }
    if authorization != expected_authorization:
        raise RuntimeError("V103 seal update changes an operational authorization")
    submission = cast("dict[str, Any]", spec["submission_policy"])
    if any(value is not False for key, value in submission.items() if key != "reason"):
        raise RuntimeError("V103 draft permits submission or materialization")

    boundary = cast("dict[str, Any]", spec["version_boundary"])
    canonical_protocol = (
        package / "configs/experiments/choice_c_v103_fresh_identity_calibration_protocol.yaml"
    )
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
        or boundary.get("protocol_commit_git_sha") != V103_PROTOCOL_COMMIT
        or boundary.get("protocol_file_sha256_at_protocol_commit") != V103_PROTOCOL_COMMIT_FILE_SHA
        or boundary.get("protocol_commit_reviewed_by_user") is not True
        or _sha256_at_commit(package.parent, V103_PROTOCOL_COMMIT, canonical_protocol)
        != V103_PROTOCOL_COMMIT_FILE_SHA
        or boundary.get("v102_is_sealed_negative") is not True
        or boundary.get("v102_frozen_candidate_exists") is not False
        or boundary.get("v102_execution_must_remain_disabled") is not True
    ):
        raise RuntimeError("V103 does not preserve the V102 negative boundary")

    scan_scope = cast("dict[str, Any]", spec["scan_scope_seal"])
    scan_scope_path = package / str(scan_scope["path"])
    seal = cast("dict[str, Any]", yaml.safe_load(scan_scope_path.read_text(encoding="utf-8")))
    seal_authorization = cast("dict[str, Any]", seal["authorization"])
    permitted = cast("dict[str, Any]", seal["permitted_future_scan_scope"])
    forbidden = cast("dict[str, Any]", seal["explicitly_forbidden_actions"])
    seal_sources = cast("list[dict[str, Any]]", permitted["mandatory_bound_sources"])
    observed_seal_sources = {
        str(source["role"]): (source.get("sha256"), source.get("expected_unique_identities"))
        for source in seal_sources
    }
    if (
        scan_scope.get("status") != "authored_uncommitted_unreviewed_after_commit"
        or scan_scope.get("protocol_commit_git_sha") != V103_PROTOCOL_COMMIT
        or scan_scope.get("protocol_file_sha256_at_protocol_commit")
        != V103_PROTOCOL_COMMIT_FILE_SHA
        or scan_scope.get("exclusion_scan_execution_authorized") is not False
        or scan_scope.get("seal_commit_git_sha") is not None
        or scan_scope.get("seal_reviewed_after_commit") is not False
        or seal.get("status") != "authored_uncommitted_unreviewed_after_commit"
        or seal.get("immutable") is not False
        or permitted.get("execution_mode") != "local_cpu_process_only"
        or permitted.get("allowed_read_roots")
        != ["configs/ledgers", "/scratch/djy8hg/runs/mosaickv"]
        or observed_seal_sources != EXPECTED_BOUND_SOURCES
        or set(cast("list[str]", permitted["allowed_fields"])) != SCAN_ALLOWED_FIELDS
        or set(cast("list[str]", permitted["forbidden_fields"])) != SCAN_FORBIDDEN_FIELDS
        or permitted.get("only_permitted_output") != V103_EXCLUSION_OUTPUT
        or any(value is not True for value in forbidden.values())
        or seal_authorization.get("scan_scope_definition_reviewed") is not True
        or any(
            seal_authorization.get(key) is not False
            for key in (
                "seal_commit_created",
                "seal_reviewed_after_commit",
                "exclusion_scan_authorized",
                "catalog_access_authorized",
                "materialization_authorized",
                "disjointness_check_authorized",
                "experiment_authorized",
            )
        )
    ):
        raise RuntimeError("V103 scan-scope seal is not fail-closed")

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
    observed_sources = {
        str(source["role"]): (source.get("sha256"), source.get("expected_unique_identities"))
        for source in sources
    }
    if (
        observed_sources != EXPECTED_BOUND_SOURCES
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
        "status": "valid_uncommitted_v103_scan_scope_seal_all_execution_disabled",
        "protocol_commit_git_sha": V103_PROTOCOL_COMMIT,
        "protocol_commit_reviewed": True,
        "scan_scope_seal_commit_created": False,
        "exclusion_scan_authorized": False,
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
