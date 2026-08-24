#!/usr/bin/env python3
"""Fail-closed validation for the unmaterialized V102 calibration protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, cast

import yaml

SOURCE_SHA = "820037afb3cc74a9f949df1555a2441680f9660e"
V100_PROTOCOL_SHA = "69aed21ee82c5cf338eeb4ecff2a8dd0481d4d04fb378831339da0cfa9effac4"
V101_PROTOCOL_SHA = "76af1327a62f94cbbc3b8cd4f59ab317d67e0395c2ed65244ba879db4e2d192e"
TEXTVQA_REVISION = "9c0699cd19768ac5ab97568f6b3cbac4c0062884"
DOCVQA_REVISION = "539088ef8a8ada01ac8e2e6d4e372586748a265e"
TOOLING_SHA = "97f3f770da3681ff4a2879da328d6dc2bb30713d"
TOOLING_SEAL_SHA = "763b92fa01fcfeaa4db880a4d9f46b128217ea1b"
SCAN_AUTHORIZATION_SHA = "31ed230e8294e17e27bbabc5f6d008aa33f02a79"
EXCLUSION_MANIFEST_FILE_SHA = "128cc7028b86fd6965e23a345181c8ad8762cd0a6b1816bfa45da1b0f2833d14"
EXCLUSION_MANIFEST_CONTENT_SHA = "3d9c7219e54ad4bb2942164bc92efd550e7adb15ee29923a471964e24b738276"
EXCLUSION_MANIFEST_SEAL_SHA = "a7a146fd8d58aa5c1d0720f2040c6bf8588dbf2a"
CATALOG_TOOLING_SHA = "9cfd6822d968b593c98e2faf2f95eadb7b585f39"
CATALOG_TOOLING_SEAL_SHA = "222647c7df2275c98d0f74a5c24ecb2675e3b0b9"
CATALOG_AUTHORIZATION_GUARD_SHA = "9d8bd919dceeb138344d8420bff594b5ec3b7407c80187305f61cb19c080199f"
PROJECTED_CATALOG_TOOLING_SHA = "67beb0af20022228643eb211d2c5c06395b98db4"
PROJECTED_AUTHORIZATION_GUARD_SHA = (
    "a915db22881aa1d68eefd1bb4d12a1f596352f26da558574d31db4b0ab67bf59"
)


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


def validate(
    spec_path: Path,
    *,
    verify_local_metadata: bool = False,
    verify_exclusion_manifest: bool = False,
    verify_projected_catalogs: bool = False,
    require_calibration_authorized: bool = False,
) -> dict[str, Any]:
    spec = cast("dict[str, Any]", yaml.safe_load(spec_path.read_text(encoding="utf-8")))
    package_root = Path(__file__).resolve().parents[1]
    if spec.get("classification") == "sealed_negative_fail_fast_debug_gate_result":
        expected = {
            "experiment_id": "choice_c_v102_v100_fresh_cross_dataset_calibration",
            "protocol_status": "negative_debug_gate_sealed_all_future_execution_disabled",
            "calibration_execution_authorized": False,
            "candidate_freeze_authorized": False,
            "heldout_evaluation_authorized": False,
        }
        if any(spec.get(key) != value for key, value in expected.items()):
            raise RuntimeError("V102 negative seal contains an enabled experiment action")
        submission = cast("dict[str, Any]", spec.get("submission_policy"))
        if any(
            submission.get(key) is not False
            for key in (
                "exclusion_scan_job_allowed",
                "materialization_job_allowed",
                "calibration_job_allowed",
                "freeze_job_allowed",
                "heldout_job_allowed",
            )
        ):
            raise RuntimeError("V102 negative seal permits job submission")
        negative = cast("dict[str, Any]", spec.get("negative_debug_gate_seal"))
        seal_path = package_root / str(negative.get("seal_path"))
        seal = cast(
            "dict[str, Any]",
            yaml.safe_load(seal_path.read_text(encoding="utf-8")),
        )
        if (
            negative.get("debug_gate_passed") is not False
            or negative.get("both_tasks_reached_compressed_target") is not False
            or negative.get("full_calibration_submitted") is not False
            or negative.get("frozen_candidate_created") is not False
            or negative.get("heldout_evaluation_submitted") is not False
            or negative.get("debug_outputs_inspected") is not True
            or negative.get("all_v102_ledger_samples_ineligible_for_future_calibration") is not True
            or negative.get("future_protocol_requires_fresh_samples") is not True
            or seal.get("overall_pass") is not False
            or seal.get("future_protocol_requires_fresh_samples") is not True
        ):
            raise RuntimeError("V102 negative result is not sealed fail-closed")
        if require_calibration_authorized:
            raise PermissionError("V102 is sealed negative; calibration submission is forbidden")
        return {
            "experiment_id": spec["experiment_id"],
            "candidate_implementation_git_sha": SOURCE_SHA,
            "tasks": {"textvqa_train_calibration": 32, "infovqa": 32},
            "active_memory_reduction_gate": 0.30,
            "fidelity_noninferiority_margin": 0.05,
            "all_actions_authorized": False,
            "calibration_execution_authorized": False,
            "candidate_freeze_authorized": False,
            "debug_gate_passed": False,
            "frozen_candidate_created": False,
            "status": "sealed_negative_debug_gate_v102",
        }
    expected_top_level = {
        "experiment_id": "choice_c_v102_v100_fresh_cross_dataset_calibration",
        "classification": "preregistered_unmaterialized_unsubmitted_label_blind_calibration",
        "protocol_status": "small_v100_calibration_and_conditional_freeze_authorized_not_submitted",
        "selection_rule_frozen_before_materialization": True,
        "tooling_implementation_authorized": True,
        "tooling_cpu_testing_authorized": True,
        "exclusion_scan_authorized": False,
        "candidate_catalog_construction_authorized": False,
        "materialization_authorized": False,
        "disjointness_check_authorized": False,
        "calibration_execution_authorized": True,
        "candidate_freeze_authorized": True,
        "heldout_evaluation_authorized": False,
    }
    if any(spec.get(key) != value for key, value in expected_top_level.items()):
        raise RuntimeError("V102 must remain unmaterialized, unsubmitted, and unauthorized")

    source = cast("dict[str, Any]", spec["candidate_source"])
    execution_allowlist = cast("list[str]", source.get("execution_diff_allowlist"))
    v100_path = package_root / str(source["v100_protocol_path"])
    if (
        source.get("candidate_implementation_git_sha") != SOURCE_SHA
        or source.get("protocol_base_git_sha") != SOURCE_SHA
        or source.get("v100_protocol_sha256") != V100_PROTOCOL_SHA
        or _sha256(v100_path) != V100_PROTOCOL_SHA
        or source.get("runtime_behavior_changes_after_candidate_commit_allowed") is not False
        or source.get("preregistration_git_sha") != "9eafc34af15676ac36a0327ff91a492feb6d862d"
        or source.get("future_tooling_must_record_preregistration_git_sha") is not True
        or source.get("tooling_implementation_git_sha") != TOOLING_SHA
        or source.get("catalog_constructor_implementation_git_sha") != PROJECTED_CATALOG_TOOLING_SHA
        or source.get("execution_git_sha_policy")
        != "clean_descendant_of_sealed_tooling_commit_with_allowlisted_diff"
        or "mosaickv/tests/unit/test_choice_c_v102_catalog_tooling.py" not in execution_allowlist
    ):
        raise RuntimeError("V102 is not pinned to the verified V100 candidate")

    predecessor = cast("dict[str, Any]", spec["predecessor_feasibility_negative"])
    predecessor_path = package_root / str(predecessor["protocol_path"])
    if (
        predecessor.get("experiment_id") != "choice_c_v101_v100_fresh_disjoint_calibration"
        or predecessor.get("protocol_sha256") != V101_PROTOCOL_SHA
        or _sha256(predecessor_path) != V101_PROTOCOL_SHA
        or predecessor.get("exclusion_manifest_file_sha256")
        != "39a068c8863ea4afd936ce698d933fff5abed31e6f43d1874f778039ef139ca2"
        or predecessor.get("excluded_docvqa_validation_ids") != 5349
        or predecessor.get("eligible_docvqa_validation_ids") != 0
        or predecessor.get("excluded_textvqa_validation_ids") != 5000
        or predecessor.get("eligible_textvqa_validation_ids") != 0
        or predecessor.get("v101_materialization_remains_disabled") is not True
    ):
        raise RuntimeError("V102 does not preserve the sealed V101 negative result")

    boundary = cast("dict[str, Any]", spec["scientific_boundary"])
    forbidden_access = (
        "prior_answers_used_for_candidate_or_sample_selection",
        "prior_references_used_for_candidate_or_sample_selection",
        "prior_task_scores_used_for_candidate_or_sample_selection",
        "prior_failure_identities_used_for_candidate_or_sample_selection",
        "prior_generated_tokens_used_for_candidate_or_sample_selection",
        "inspect_media_or_prompt_content_for_selection",
        "calibration_answers_accessed_before_execution",
        "calibration_references_accessed",
        "calibration_official_task_scores_accessed",
        "benchmark_labels_required_online",
    )
    if (
        any(boundary.get(key) is not False for key in forbidden_access)
        or boundary.get("v101_used_only_as_a_split_feasibility_boundary") is not True
        or boundary.get("method_is_training_free") is not True
    ):
        raise RuntimeError("V102 scientific boundary permits label or outcome leakage")

    metadata = cast("dict[str, Any]", spec["dataset_metadata_evidence"])
    text = cast("dict[str, Any]", metadata["textvqa"])
    info = cast("dict[str, Any]", metadata["infovqa"])
    if (
        metadata.get("metadata_only_no_dataset_rows_loaded") is not True
        or metadata.get("lmms_eval_version") != "0.7.2"
        or text.get("dataset_id") != "lmms-lab/textvqa"
        or text.get("dataset_config") != "default"
        or text.get("resolved_revision") != TEXTVQA_REVISION
        or text.get("train_examples") != 34602
        or info.get("dataset_id") != "lmms-lab-encoder/DocVQA"
        or info.get("dataset_config") != "InfographicVQA"
        or info.get("resolved_revision") != DOCVQA_REVISION
        or info.get("validation_examples") != 2801
        or info.get("source_lmms_task") != "infovqa_val"
    ):
        raise RuntimeError("V102 dataset metadata contract changed")
    if verify_local_metadata:
        text_info = Path(str(text["cached_dataset_info_path"]))
        if not text_info.is_file() or _sha256(text_info) != text.get("cached_dataset_info_sha256"):
            raise RuntimeError("V102 TextVQA cached metadata is absent or differs")
        text_value = cast("dict[str, Any]", json.loads(text_info.read_text(encoding="utf-8")))
        text_train = cast("dict[str, Any]", text_value["splits"])["train"]
        if cast("dict[str, Any]", text_train).get("num_examples") != 34602:
            raise RuntimeError("V102 TextVQA train cardinality differs")
        installed = {
            Path(str(text["installed_task_yaml_path"])): text.get("installed_task_yaml_sha256"),
            Path(str(info["installed_task_yaml_path"])): info.get("installed_task_yaml_sha256"),
        }
        for path, expected in installed.items():
            if not path.is_file() or _sha256(path) != expected:
                raise RuntimeError(f"V102 installed lmms-eval task metadata differs: {path}")

    audit = cast("dict[str, Any]", spec["preliminary_prior_metadata_audit"])
    if (
        audit.get("matching_prior_textvqa_train_metadata_records") != 0
        or audit.get("matching_prior_infovqa_validation_metadata_records") != 0
        or audit.get("exhaustive_formal_exclusion_scan_completed") is not False
        or audit.get("formal_scan_required_before_materialization") is not True
        or audit.get("no_freshness_claim_allowed_until_formal_scan_review") is not True
    ):
        raise RuntimeError("V102 preliminary audit overclaims freshness")

    tasks = cast("list[dict[str, Any]]", spec["tasks"])
    by_task = {str(task["task"]): task for task in tasks}
    if set(by_task) != {"textvqa_train_calibration", "infovqa"}:
        raise RuntimeError("V102 task set changed")
    expected_tasks = {
        "textvqa_train_calibration": (
            "lmms-lab/textvqa",
            "default",
            TEXTVQA_REVISION,
            "train",
            34602,
            "question_id",
            "data/train-*.parquet",
            20,
            "catalog_textvqa_train_id_projected_v2.json",
        ),
        "infovqa": (
            "lmms-lab-encoder/DocVQA",
            "InfographicVQA",
            DOCVQA_REVISION,
            "validation",
            2801,
            "questionId",
            "InfographicVQA/validation-*.parquet",
            4,
            "catalog_infovqa_validation_id_projected_v2.json",
        ),
    }
    for name, expected in expected_tasks.items():
        task = by_task[name]
        observed = (
            task.get("dataset_id"),
            task.get("dataset_config"),
            task.get("dataset_revision"),
            task.get("split"),
            task.get("expected_split_cardinality"),
            task.get("candidate_catalog_id_field"),
            task.get("candidate_catalog_parquet_glob"),
            task.get("candidate_catalog_expected_shard_count"),
            Path(str(task.get("candidate_catalog_path"))).name,
        )
        if (
            observed != expected
            or task.get("selection_seed") != 20320821
            or task.get("sample_count") != 32
            or task.get("ledger_status") != "materialized_reviewed_disjointness_passed_read_only"
            or task.get("ledger") != task.get("ledger_path")
            or task.get("candidate_catalog_expected_id_count") != expected[4]
            or task.get("candidate_catalog_status")
            != "projected_content_valid_scope_positive_read_only"
        ):
            raise RuntimeError(f"V102 task contract changed for {name}")

    tooling = cast("dict[str, Any]", spec["identity_and_tooling_requirements"])
    if (
        tooling.get("implementation_status") != "committed_sealed_cpu_tested_scan_disabled"
        or tooling.get("canonical_identity")
        != "dataset_id_dataset_config_dataset_revision_split_sample_id"
        or tooling.get("dataset_config_is_mandatory") is not True
        or tooling.get("extend_dataset_contract_and_sample_identity_before_scan") is not True
        or tooling.get("implementation_authorized_by_user") is not True
        or tooling.get("cpu_only_testing_authorized_by_user") is not True
        or tooling.get("tooling_commit_created") is not True
        or tooling.get("tooling_implementation_git_sha") != TOOLING_SHA
        or tooling.get("tooling_seal_created") is not True
        or any(
            tooling.get(key) is not True
            for key in (
                "configuration_aware_identity_implemented",
                "configuration_aware_scanner_implemented",
                "configuration_aware_materializer_implemented",
                "configuration_aware_disjointness_checker_implemented",
                "textvqa_train_split_override_implemented",
                "infovqa_task_route_implemented",
                "synthetic_cpu_tests_passed",
            )
        )
        or any(
            tooling.get(key) is not False
            for key in (
                "real_dataset_rows_media_labels_or_scores_accessed",
                "real_materialization_performed",
                "slurm_or_gpu_job_submitted",
            )
        )
        or tooling.get("real_exclusion_scan_performed") is not True
        or tooling.get("no_runtime_or_candidate_algorithm_change_allowed") is not True
    ):
        raise RuntimeError("V102 configuration-aware tooling boundary changed")

    seal = cast("dict[str, Any]", spec["configuration_aware_tooling_seal"])
    expected_hashes = {
        "pyproject.toml": "f27bf49babe0f645d22a66587b691aae79ad2ed2635066c497c09d0c18ee7072",
        "scripts/choice_c_v102_sample_tooling.py": (
            "2509ff71b2b5c226fbefd9aec1f7a83eb61c7b0a68df1bb3c1ac9b6c3e89c974"
        ),
        "src/mosaickv/evaluation/disjoint_samples.py": (
            "cae454bd7ccd99d1f88577fbf974443273f4918bfbf5b5b78242fc56718990ce"
        ),
        "src/mosaickv/evaluation/lmms_adapter.py": (
            "146666e8a1058f2e34f38b4326bb66c05d862832dc79147b957b442879308328"
        ),
        "src/mosaickv/evaluation/lmms_compat/infovqa/_default_template_infovqa_yaml": (
            "fdc268c05fe889cd6ddbed9bba67aa2f2f55777fb3d7ed09dbb173f43178bf4a"
        ),
        "src/mosaickv/evaluation/tasks.py": (
            "b7282228c6da9df9dcfa33c37997bd14a1406cc64882fc4715698f07ffe23e15"
        ),
        "tests/unit/test_choice_c_v102_protocol.py": (
            "0d02192f069cc04d5cfee4e338626bdfb00fdbe3b5f7b405f04addb77f1a9387"
        ),
        "tests/unit/test_disjoint_samples.py": (
            "4f41941df1151ffb68d2d47a798f8bec887e2289e621022fd1bb14b28e085d55"
        ),
        "tests/unit/test_evaluation_tasks.py": (
            "de5e14393c2c64f3f8fb6d20e619c9431007e46241a54ba05bd29b16ebdd1732"
        ),
        "tests/unit/test_lmms_adapter.py": (
            "f34417c3399f3e13b8b5962300759a3ab891c39b02147c29a41220c7ef092b7c"
        ),
    }
    records = cast("list[dict[str, Any]]", seal.get("committed_files"))
    recorded_hashes = {str(record.get("path")): record.get("sha256") for record in records}
    repository = package_root.parent
    if (
        seal.get("status") != "committed_cpu_tested_sealed_scan_disabled"
        or seal.get("implementation_git_sha") != TOOLING_SHA
        or seal.get("seal_git_sha") != TOOLING_SEAL_SHA
        or seal.get("seal_reviewed_after_commit") is not True
        or recorded_hashes != expected_hashes
        or any(
            _sha256_at_commit(repository, TOOLING_SHA, package_root / path) != expected_hash
            for path, expected_hash in expected_hashes.items()
        )
    ):
        raise RuntimeError("V102 tooling seal or committed file hashes differ")
    validation = cast("dict[str, Any]", seal["validation"])
    if (
        validation.get("full_cpu_pytest_passed") is not True
        or validation.get("full_cpu_pytest_passed_count") != 728
        or validation.get("expected_gpu_skips") != 2
        or validation.get("ruff_passed") is not True
        or validation.get("focused_mypy_passed") is not True
        or any(
            validation.get(key) is not False
            for key in (
                "real_dataset_rows_media_labels_or_scores_accessed",
                "exclusion_scan_performed",
                "materialization_performed",
                "slurm_or_gpu_job_submitted",
            )
        )
    ):
        raise RuntimeError("V102 tooling seal validation evidence differs")

    catalog_tooling = cast("dict[str, Any]", spec["initial_candidate_catalog_tooling_seal"])
    expected_catalog_tooling_hashes = {
        "src/mosaickv/evaluation/disjoint_samples.py": (
            "59937223084d8b0ef818218291393c6491b67ee3a631fc66b31d028c6e89d122"
        ),
        "scripts/choice_c_v102_sample_tooling.py": (
            "b4db0860c187d4aa77d087e5b460cfc022ca91d83ce6404a40acf7fdcad86deb"
        ),
        "tests/unit/test_disjoint_samples.py": (
            "111b16049a093211b89f0fffb687ef882a13a2cb3c56342f92052cd4070509a4"
        ),
        "tests/unit/test_choice_c_v102_catalog_tooling.py": (
            "41ee50ec4db8459eed6af465eea90c7b6e6a1e10747a42e9942bb32cc85c2030"
        ),
    }
    catalog_tooling_records = cast("list[dict[str, Any]]", catalog_tooling.get("committed_files"))
    recorded_catalog_tooling_hashes = {
        str(record.get("path")): record.get("sha256") for record in catalog_tooling_records
    }
    catalog_validation = cast("dict[str, Any]", catalog_tooling.get("validation"))
    if (
        catalog_tooling.get("status") != "committed_cpu_tested_sealed_catalogs_not_constructed"
        or catalog_tooling.get("implementation_git_sha") != CATALOG_TOOLING_SHA
        or catalog_tooling.get("seal_git_sha") != CATALOG_TOOLING_SEAL_SHA
        or catalog_tooling.get("implementation_commit_subject")
        != "Implement fail-closed V102 ID catalogs"
        or catalog_tooling.get("implementation_commit_was_clean_after_commit") is not True
        or catalog_tooling.get("seal_reviewed_after_commit") is not True
        or catalog_tooling.get("catalog_construction_execution_enabled") is not True
        or recorded_catalog_tooling_hashes != expected_catalog_tooling_hashes
        or any(
            _sha256_at_commit(repository, CATALOG_TOOLING_SHA, package_root / path) != expected_hash
            for path, expected_hash in expected_catalog_tooling_hashes.items()
        )
        or catalog_validation.get("synthetic_cpu_tests_passed") is not True
        or catalog_validation.get("focused_cpu_pytest_passed_count") != 33
        or catalog_validation.get("full_cpu_pytest_passed") is not True
        or catalog_validation.get("targeted_ruff_format_and_lint_passed") is not True
        or catalog_validation.get("focused_mypy_passed") is not True
        or catalog_validation.get("installed_datasets_version") != "4.8.4"
        or catalog_validation.get("textvqa_audited_id_field") != "question_id"
        or catalog_validation.get("infovqa_audited_id_field") != "questionId"
        or any(
            catalog_validation.get(key) is not False
            for key in (
                "real_candidate_catalogs_constructed",
                "real_dataset_rows_media_labels_or_scores_accessed",
                "ledger_materialization_performed",
                "slurm_or_gpu_job_submitted",
            )
        )
    ):
        raise RuntimeError("V102 ID-only catalog tooling seal or evidence differs")

    scan = cast("dict[str, Any]", spec["exclusion_scan_authorization"])
    downstream_authorizations = (
        "materialization_authorized",
        "disjointness_check_authorized",
        "calibration_execution_authorized",
        "candidate_freeze_authorized",
        "heldout_evaluation_authorized",
        "slurm_or_other_job_submission_authorized",
    )
    if (
        scan.get("status") != "consumed_after_successful_local_cpu_scan_reexecution_disabled"
        or scan.get("authorized_by_user") is not True
        or scan.get("reviewed_tooling_seal_git_sha") != TOOLING_SEAL_SHA
        or scan.get("authorized_action") != "exclusion_scan_only"
        or scan.get("execution_mode") != "local_cpu_process_only"
        or scan.get("recursively_scan_roots")
        != ["configs/ledgers", "/scratch/djy8hg/runs/mosaickv"]
        or scan.get("immutable_output_path")
        != (
            "/scratch/djy8hg/runs/mosaickv/"
            "choice-c-v102-v100-fresh-cross-dataset-calibration-v1/"
            "inputs/exclusion_manifest.json"
        )
        or scan.get("use_configuration_aware_identity") is not True
        or scan.get("inspect_only_ids_task_names_and_provenance") is not True
        or scan.get("inspect_answers_references_scores_media_or_failure_outcomes") is not False
        or scan.get("require_clean_committed_source_snapshot") is not True
        or scan.get("require_write_once_output") is not True
        or scan.get("authorization_commit_git_sha") != SCAN_AUTHORIZATION_SHA
        or scan.get("scan_execution_source_git_sha") != SCAN_AUTHORIZATION_SHA
        or scan.get("scan_execution_source_git_dirty") is not False
        or scan.get("scan_execution_performed") is not True
        or scan.get("scan_reexecution_authorized") is not False
        or scan.get("exclusion_manifest_reviewed") is not True
        or scan.get("exclusion_manifest_sealed") is not True
        or any(scan.get(key) is not False for key in downstream_authorizations)
    ):
        raise RuntimeError("V102 scan authorization exceeds its reviewed CPU-only scope")

    materialization = cast("dict[str, Any]", spec["future_materialization_rule"])
    exclusion = cast("dict[str, Any]", spec["required_exclusion_snapshot"])
    if (
        materialization.get("authorized") is not False
        or materialization.get("inspect_media_or_prompt_content_for_selection") is not False
        or materialization.get(
            "inspect_answers_references_scores_or_failure_identity_for_selection"
        )
        is not False
        or exclusion.get("status") != "reviewed_sealed_read_only_zero_prior_matching_identities"
        or exclusion.get("require_configuration_aware_canonical_identity_deduplication") is not True
        or exclusion.get("require_zero_overlap_for_each_task") is not True
        or exclusion.get("exclusion_manifest_status") != "reviewed_sealed_read_only"
        or exclusion.get("exclusion_manifest_file_sha256") != EXCLUSION_MANIFEST_FILE_SHA
        or exclusion.get("exclusion_manifest_content_sha256") != EXCLUSION_MANIFEST_CONTENT_SHA
        or exclusion.get("exclusion_manifest_source_git_sha") != SCAN_AUTHORIZATION_SHA
        or exclusion.get("exclusion_manifest_source_git_dirty") is not False
        or exclusion.get("exclusion_manifest_mode") != "0444"
        or exclusion.get("exclusion_manifest_bytes") != 840
        or exclusion.get("schema_version") != 2
        or exclusion.get("source_count") != 0
        or exclusion.get("excluded_unique_sample_count") != 0
        or exclusion.get("excluded_textvqa_train_ids") != 0
        or exclusion.get("excluded_infovqa_validation_ids") != 0
        or exclusion.get("scanner_candidate_files_examined") != 4226
        or exclusion.get("scanner_structured_ledger_or_input_candidates") != 125
        or exclusion.get("scanner_observed_result_jsonl_candidates") != 4101
        or exclusion.get("ambiguous_relevant_metadata_paths") != 0
        or exclusion.get("exact_contract_metadata_paths") != 0
        or exclusion.get("content_hash_validation_passed") is not True
        or exclusion.get("bound_source_hash_validation_passed") is not True
        or exclusion.get("provenance_seal_git_sha") != EXCLUSION_MANIFEST_SEAL_SHA
        or exclusion.get("provenance_seal_reviewed_after_commit") is not True
        or exclusion.get("candidate_availability_or_disjointness_claimed") is not True
        or exclusion.get("disjointness_report_status") != "passed_reviewed_read_only"
        or exclusion.get("disjointness_report_file_sha256")
        != "2f7f947ca016b8fb5ad425d9d9cfb8e304b9e3ffd95391216e6cdaf8117d36ac"
        or exclusion.get("disjointness_report_content_sha256")
        != "0f5d20b178c93b51d1cf21ee4c1d26a8553726d2e8bbc12746138bb296b5ef14"
        or exclusion.get("disjointness_report_validation_git_sha")
        != "11bf2f77bc62dbde82bfd8519f7715ba32850753"
        or exclusion.get("selected_unique_sample_count") != 64
        or exclusion.get("overlap_count") != 0
        or exclusion.get("cross_ledger_duplicate_count") != 0
    ):
        raise RuntimeError("V102 materialization or exclusion boundary changed")
    if verify_exclusion_manifest:
        manifest_path = Path(str(exclusion["exclusion_manifest_path"]))
        if (
            not manifest_path.is_file()
            or _sha256(manifest_path) != EXCLUSION_MANIFEST_FILE_SHA
            or (manifest_path.stat().st_mode & 0o777) != 0o444
            or manifest_path.stat().st_size != 840
        ):
            raise RuntimeError("V102 sealed exclusion manifest is absent or differs")
        manifest = cast("dict[str, Any]", json.loads(manifest_path.read_text(encoding="utf-8")))
        if (
            manifest.get("manifest_content_sha256") != EXCLUSION_MANIFEST_CONTENT_SHA
            or manifest.get("source_git_sha") != SCAN_AUTHORIZATION_SHA
            or manifest.get("source_git_dirty") is not False
            or manifest.get("schema_version") != 2
            or manifest.get("source_count") != 0
            or manifest.get("excluded_unique_sample_count") != 0
        ):
            raise RuntimeError("V102 sealed exclusion manifest content differs")

    catalog = cast("dict[str, Any]", spec["initial_candidate_catalog_construction_record"])
    expected_catalogs = [
        {
            "task": "textvqa_train_calibration",
            "dataset_id": "lmms-lab/textvqa",
            "dataset_config": "default",
            "dataset_revision": TEXTVQA_REVISION,
            "split": "train",
            "expected_id_count": 34602,
            "id_field": "question_id",
            "output_path": (
                "/scratch/djy8hg/runs/mosaickv/"
                "choice-c-v102-v100-fresh-cross-dataset-calibration-v1/"
                "inputs/catalog_textvqa_train.json"
            ),
            "observed_file_sha256": (
                "5a6f700a623da65ae62bf4abc7317dcea48f0a9388f4047d3de8cfd643aece8d"
            ),
            "observed_file_bytes": 1165531,
            "observed_mode": "0444",
            "observed_id_count": 34602,
            "content_review_status": "passed",
            "construction_scope_status": "negative",
        },
        {
            "task": "infovqa",
            "dataset_id": "lmms-lab-encoder/DocVQA",
            "dataset_config": "InfographicVQA",
            "dataset_revision": DOCVQA_REVISION,
            "split": "validation",
            "expected_id_count": 2801,
            "id_field": "questionId",
            "output_path": (
                "/scratch/djy8hg/runs/mosaickv/"
                "choice-c-v102-v100-fresh-cross-dataset-calibration-v1/"
                "inputs/catalog_infovqa_validation.json"
            ),
            "observed_file_sha256": (
                "38907d03d42a6c52d898b3a543ad2cd737b85569c465ad301ca148aa72e4308c"
            ),
            "observed_file_bytes": 45008,
            "observed_mode": "0444",
            "observed_id_count": 2801,
            "content_review_status": "passed",
            "construction_scope_status": "negative",
        },
    ]
    forbidden_catalog_actions = (
        "exclusion_scan_reexecution_authorized",
        "ledger_materialization_authorized",
        "disjointness_check_authorized",
        "calibration_execution_authorized",
        "candidate_freeze_authorized",
        "heldout_evaluation_authorized",
        "slurm_or_other_job_submission_authorized",
    )
    if (
        catalog.get("status")
        != "consumed_content_valid_construction_scope_negative_reexecution_disabled"
        or catalog.get("authorized_by_user") is not True
        or catalog.get("reviewed_exclusion_manifest_seal_git_sha") != EXCLUSION_MANIFEST_SEAL_SHA
        or catalog.get("reviewed_catalog_tooling_seal_git_sha") != CATALOG_TOOLING_SEAL_SHA
        or catalog.get("authorization_guard_path") != "scripts/choice_c_v102_sample_tooling.py"
        or catalog.get("authorization_guard_sha256") != CATALOG_AUTHORIZATION_GUARD_SHA
        or catalog.get("authorization_guard_git_sha") != "c14db3907dc226fc8cfa1c01e8fdd5276210acf7"
        or _sha256_at_commit(
            repository,
            str(catalog.get("authorization_guard_git_sha")),
            package_root / str(catalog.get("authorization_guard_path")),
        )
        != CATALOG_AUTHORIZATION_GUARD_SHA
        or catalog.get("require_canonical_committed_spec_path") is not True
        or catalog.get("require_exact_reviewed_seal_sha") is not True
        or catalog.get("authorized_action") != "construct_complete_candidate_id_catalogs_only"
        or catalog.get("execution_mode") != "local_cpu_process_only"
        or catalog.get("source_snapshot_must_be_clean_and_committed") is not True
        or catalog.get("pinned_contracts_only") is not True
        or catalog.get("catalogs") != expected_catalogs
        or catalog.get("catalog_schema_fields")
        != [
            "schema_version",
            "dataset_id",
            "dataset_config",
            "dataset_revision",
            "split",
            "sample_ids",
        ]
        or catalog.get("selection_or_ranking_during_catalog_construction_forbidden") is not True
        or catalog.get("require_every_split_id_exactly_once") is not True
        or catalog.get("require_unique_nonempty_string_ids") is not True
        or catalog.get("deterministic_order") != "canonical_sample_id_ascending"
        or catalog.get("allowed_dataset_access")
        != ["pinned_dataset_metadata", "split_cardinality", "sample_id_column_only"]
        or catalog.get(
            "prompt_question_answer_reference_score_media_or_failure_outcome_access_forbidden"
        )
        is not True
        or catalog.get("model_processor_or_tokenizer_loading_forbidden") is not True
        or catalog.get("model_weights_loading_forbidden") is not True
        or catalog.get("use_only_scratch_cache_paths") is not True
        or catalog.get("hf_token_from_environment_only") is not True
        or catalog.get("immutable_write_once_mode") != "0444"
        or catalog.get("catalog_construction_performed") is not True
        or catalog.get("catalogs_reviewed") is not True
        or catalog.get("construction_source_git_sha") != "bdf45126c995972e0d8f334366b2e2dbfc1eaf3e"
        or catalog.get("construction_source_git_dirty") is not False
        or catalog.get("construction_scope_passed") is not False
        or catalog.get("reexecution_authorized") is not False
        or catalog.get("replacement_catalogs_required") is not True
        or any(catalog.get(key) is not False for key in forbidden_catalog_actions)
    ):
        raise RuntimeError("V102 candidate-catalog authorization exceeds its ID-only scope")

    scope_negative = cast("dict[str, Any]", spec["initial_candidate_catalog_scope_negative"])
    if (
        scope_negative.get("status") != "sealed_content_valid_construction_scope_negative"
        or scope_negative.get("content_schema_cardinality_uniqueness_order_and_permissions_passed")
        is not True
        or scope_negative.get("cross_catalog_duplicate_count") != 0
        or scope_negative.get("mosaic_code_indexed_only_the_audited_id_column") is not True
        or scope_negative.get("huggingface_builder_prepared_only_requested_split") is not False
        or scope_negative.get("observed_unrequested_split") != "InfographicVQA_test"
        or scope_negative.get("observed_unrequested_test_examples_prepared") != 3288
        or scope_negative.get("prompt_answer_reference_score_or_media_used_for_selection")
        is not False
        or scope_negative.get("construction_scope_gate_passed") is not False
        or scope_negative.get("catalogs_eligible_for_materialization") is not False
        or scope_negative.get("materialization_performed") is not False
        or scope_negative.get("calibration_or_experiment_run") is not False
    ):
        raise RuntimeError("V102 initial catalog scope-negative record differs")

    replacement = cast("dict[str, Any]", spec["replacement_projected_catalog_tooling"])
    expected_projected_hashes = {
        "scripts/choice_c_v102_sample_tooling.py": (
            "c933776edf677f57df0adb18d027b56547464e55396cd8f20c0838e76bd711d7"
        ),
        "scripts/validate_choice_c_v102_preregistration.py": (
            "0572bb19d5ccd66a111d321e96aa5b35b5a6b9a0038b66cab6c6498d32daee95"
        ),
        "tests/unit/test_choice_c_v102_catalog_tooling.py": (
            "89b7ab4906d42b9bbd5e4638950daf4f469dd8c9da5f7bfd8326428998081595"
        ),
        "tests/unit/test_choice_c_v102_protocol.py": (
            "19d6c496d0b9482aeefabcb2902d1383b24307bb342bb68c28c1565f5da8fc12"
        ),
        "tests/unit/test_disjoint_samples.py": (
            "3ca0b314a4b6e456d930399fab7167671828ae79bdacc1c882ba7ae3b57a3ed0"
        ),
    }
    projected_records = cast("list[dict[str, Any]]", replacement.get("committed_files"))
    recorded_projected_hashes = {
        str(record.get("path")): record.get("sha256") for record in projected_records
    }
    if (
        replacement.get("status")
        != "committed_cpu_tested_sealed_projected_catalogs_constructed_reviewed"
        or replacement.get("implementation_authorized_by_user") is not True
        or replacement.get("cpu_testing_authorized_by_user") is not True
        or replacement.get("implementation_git_sha") != PROJECTED_CATALOG_TOOLING_SHA
        or replacement.get("implementation_commit_subject")
        != "Implement projected V102 ID catalog loader"
        or replacement.get("implementation_reviewed_after_commit") is not True
        or replacement.get("use_datasets_load_dataset") is not False
        or replacement.get("pinned_hub_repo_file_listing_only") is not True
        or replacement.get("exact_split_parquet_glob_required") is not True
        or replacement.get("expected_shard_count_required") is not True
        or replacement.get("pyarrow_columns_argument_must_equal_audited_id_field_only") is not True
        or replacement.get("pyarrow_use_threads") is not False
        or replacement.get("pyarrow_pre_buffer") is not False
        or replacement.get("unrequested_split_open_forbidden") is not True
        or replacement.get("model_processor_or_tokenizer_loading_forbidden") is not True
        or replacement.get("scratch_only_cache_environment_required") is not True
        or replacement.get("synthetic_cpu_tests_passed") is not True
        or replacement.get("focused_cpu_pytest_passed_count") != 36
        or replacement.get("full_cpu_pytest_passed") is not True
        or replacement.get("real_replacement_catalogs_constructed") is not True
        or replacement.get("tooling_sealed") is not True
        or replacement.get("replacement_construction_authorized") is not False
        or replacement.get("materialization_authorized") is not False
        or replacement.get("slurm_or_gpu_job_submitted") is not False
        or recorded_projected_hashes != expected_projected_hashes
        or any(
            _sha256_at_commit(repository, PROJECTED_CATALOG_TOOLING_SHA, package_root / path)
            != expected_hash
            for path, expected_hash in expected_projected_hashes.items()
        )
    ):
        raise RuntimeError("V102 projected replacement tooling boundary differs")

    projected_auth = cast("dict[str, Any]", spec["candidate_catalog_construction_authorization"])
    expected_projected_catalogs = [
        {
            "task": "textvqa_train_calibration",
            "dataset_id": "lmms-lab/textvqa",
            "dataset_config": "default",
            "dataset_revision": TEXTVQA_REVISION,
            "split": "train",
            "expected_id_count": 34602,
            "id_field": "question_id",
            "parquet_glob": "data/train-*.parquet",
            "expected_shard_count": 20,
            "output_path": (
                "/scratch/djy8hg/runs/mosaickv/"
                "choice-c-v102-v100-fresh-cross-dataset-calibration-v1/inputs/"
                "catalog_textvqa_train_id_projected_v2.json"
            ),
            "observed_file_sha256": (
                "5a6f700a623da65ae62bf4abc7317dcea48f0a9388f4047d3de8cfd643aece8d"
            ),
            "observed_file_bytes": 1165531,
            "observed_mode": "0444",
            "observed_id_count": 34602,
            "content_review_status": "passed",
            "construction_scope_status": "passed",
        },
        {
            "task": "infovqa",
            "dataset_id": "lmms-lab-encoder/DocVQA",
            "dataset_config": "InfographicVQA",
            "dataset_revision": DOCVQA_REVISION,
            "split": "validation",
            "expected_id_count": 2801,
            "id_field": "questionId",
            "parquet_glob": "InfographicVQA/validation-*.parquet",
            "expected_shard_count": 4,
            "output_path": (
                "/scratch/djy8hg/runs/mosaickv/"
                "choice-c-v102-v100-fresh-cross-dataset-calibration-v1/inputs/"
                "catalog_infovqa_validation_id_projected_v2.json"
            ),
            "observed_file_sha256": (
                "38907d03d42a6c52d898b3a543ad2cd737b85569c465ad301ca148aa72e4308c"
            ),
            "observed_file_bytes": 45008,
            "observed_mode": "0444",
            "observed_id_count": 2801,
            "content_review_status": "passed",
            "construction_scope_status": "passed",
        },
    ]
    projected_forbidden = (
        "exclusion_scan_reexecution_authorized",
        "ledger_materialization_authorized",
        "disjointness_check_authorized",
        "calibration_execution_authorized",
        "candidate_freeze_authorized",
        "heldout_evaluation_authorized",
        "slurm_or_other_job_submission_authorized",
    )
    if (
        projected_auth.get("status")
        != "consumed_projected_catalogs_content_valid_scope_positive_reexecution_disabled"
        or projected_auth.get("authorized_by_user") is not True
        or projected_auth.get("reviewed_projected_tooling_git_sha") != PROJECTED_CATALOG_TOOLING_SHA
        or projected_auth.get("authorization_guard_path")
        != "scripts/choice_c_v102_sample_tooling.py"
        or projected_auth.get("authorization_guard_sha256") != PROJECTED_AUTHORIZATION_GUARD_SHA
        or projected_auth.get("authorization_guard_git_sha")
        != "9a8551940c5f7166eaa360bcf3bfed4e3c194867"
        or _sha256_at_commit(
            repository,
            str(projected_auth.get("authorization_guard_git_sha")),
            package_root / str(projected_auth.get("authorization_guard_path")),
        )
        != PROJECTED_AUTHORIZATION_GUARD_SHA
        or projected_auth.get("require_canonical_committed_spec_path") is not True
        or projected_auth.get("authorized_action")
        != "construct_two_projected_candidate_id_catalogs_only"
        or projected_auth.get("execution_mode") != "local_cpu_process_only"
        or projected_auth.get("source_snapshot_must_be_clean_and_committed") is not True
        or projected_auth.get("pinned_contracts_only") is not True
        or projected_auth.get("catalogs") != expected_projected_catalogs
        or projected_auth.get("catalog_schema_fields")
        != [
            "schema_version",
            "dataset_id",
            "dataset_config",
            "dataset_revision",
            "split",
            "sample_ids",
        ]
        or projected_auth.get("exact_split_shards_only") is not True
        or projected_auth.get("id_column_projection_only") is not True
        or projected_auth.get("use_datasets_load_dataset") is not False
        or projected_auth.get("selection_or_ranking_forbidden") is not True
        or projected_auth.get("use_only_scratch_cache_paths") is not True
        or projected_auth.get("hf_token_from_environment_only") is not True
        or projected_auth.get("immutable_write_once_mode") != "0444"
        or projected_auth.get("catalog_construction_performed") is not True
        or projected_auth.get("catalogs_reviewed") is not True
        or projected_auth.get("construction_source_git_sha")
        != "09697c62699a872bb9044858084c05b601437ddc"
        or projected_auth.get("construction_source_git_dirty") is not False
        or projected_auth.get("construction_scope_passed") is not True
        or projected_auth.get("reexecution_authorized") is not False
        or projected_auth.get("replacement_materialization_eligible") is not False
        or any(projected_auth.get(key) is not False for key in projected_forbidden)
    ):
        raise RuntimeError("V102 projected catalog authorization exceeds its ID-only scope")
    if verify_projected_catalogs:
        observed_id_sets: list[set[str]] = []
        allowed_fields = set(cast("list[str]", projected_auth["catalog_schema_fields"]))
        for record in expected_projected_catalogs:
            catalog_path = Path(str(record["output_path"]))
            if (
                not catalog_path.is_file()
                or _sha256(catalog_path) != record["observed_file_sha256"]
                or catalog_path.stat().st_size != record["observed_file_bytes"]
                or (catalog_path.stat().st_mode & 0o777) != 0o444
            ):
                raise RuntimeError(f"V102 projected catalog is absent or differs: {catalog_path}")
            payload = cast("dict[str, Any]", json.loads(catalog_path.read_text(encoding="utf-8")))
            sample_ids = cast("list[Any]", payload.get("sample_ids"))
            if (
                set(payload) != allowed_fields
                or payload.get("schema_version") != 2
                or any(
                    payload.get(field) != record[field]
                    for field in (
                        "dataset_id",
                        "dataset_config",
                        "dataset_revision",
                        "split",
                    )
                )
                or len(sample_ids) != record["observed_id_count"]
                or any(not isinstance(value, str) or not value for value in sample_ids)
                or sample_ids != sorted(sample_ids)
                or len(set(sample_ids)) != len(sample_ids)
            ):
                raise RuntimeError(f"V102 projected catalog content differs: {catalog_path}")
            observed_id_sets.append(set(cast("list[str]", sample_ids)))
        if observed_id_sets[0].intersection(observed_id_sets[1]):
            raise RuntimeError("V102 projected catalogs contain cross-catalog duplicate IDs")

    environment = cast("dict[str, Any]", spec["common_environment_reseal"])
    environment_record = package_root / str(environment.get("record_path"))
    if (
        environment.get("status") != "committed_reviewed_exact_common_environment_cpu_gate_passed"
        or environment.get("verification_source_git_sha")
        != "309551eb4c1f107d5cd831f96a0baa58125a5d44"
        or environment.get("verification_source_git_dirty") is not False
        or environment.get("record_git_sha") != "29afaa6b307c2ff3c378a906b247d48d384c007d"
        or environment.get("record_sha256")
        != "73c4153c103c398f95b55a680acd40ceb212e12ed832bb5c171ab60f8d38bb4d"
        or _sha256(environment_record) != environment.get("record_sha256")
        or environment.get("record_reviewed_after_commit") is not True
        or environment.get("environment_path") != "/scratch/djy8hg/env/mosaickv"
        or environment.get("lock_path") != "env/common/requirements.lock"
        or environment.get("lock_sha256")
        != "fdc56ad9edb03e8bad5cb2108322c7a12b0f33bdd5533ed596dc6b695a2db9c0"
        or environment.get("locked_distribution_count") != 251
        or environment.get("python_version") != "3.11.15"
        or environment.get("transformers_version") != "4.55.2"
        or environment.get("tokenizers_version") != "0.21.4"
        or environment.get("verifier_error_count") != 0
        or environment.get("full_test_passed_count") != 743
        or environment.get("expected_gpu_skips") != 2
        or environment.get("full_test_failure_count") != 0
        or any(
            environment.get(key) is not True
            for key in (
                "pip_check_passed",
                "exact_lock_verification_passed",
                "required_cpu_imports_passed",
                "python_user_site_disabled",
                "scratch_rooted_cache_and_pytest_temp",
            )
        )
        or any(
            environment.get(key) is not False
            for key in (
                "packages_installed_upgraded_or_removed",
                "lock_modified",
                "cuda_validation_claimed",
            )
        )
    ):
        raise RuntimeError("V102 common-environment reseal differs")

    materialization = cast("dict[str, Any]", spec["ledger_materialization_authorization"])
    expected_ledgers = [
        {
            "task": "textvqa_train_calibration",
            "catalog_path": (
                "/scratch/djy8hg/runs/mosaickv/"
                "choice-c-v102-v100-fresh-cross-dataset-calibration-v1/inputs/"
                "catalog_textvqa_train_id_projected_v2.json"
            ),
            "catalog_sha256": ("5a6f700a623da65ae62bf4abc7317dcea48f0a9388f4047d3de8cfd643aece8d"),
            "output_path": (
                "/scratch/djy8hg/runs/mosaickv/"
                "choice-c-v102-v100-fresh-cross-dataset-calibration-v1/inputs/"
                "calibration_textvqa_train.json"
            ),
            "sample_count": 32,
            "selection_seed": 20320821,
            "observed_file_sha256": (
                "a0667e45729639b1830a308d97d46662f3e6c6c76ddda97749d0663fc329b242"
            ),
            "observed_file_bytes": 4854,
            "observed_mode": "0444",
            "observed_content_sha256": (
                "160842ae53f902e08d6e80f17a888ec783a38d4dac8fadbc0e3904471d7edbee"
            ),
        },
        {
            "task": "infovqa",
            "catalog_path": (
                "/scratch/djy8hg/runs/mosaickv/"
                "choice-c-v102-v100-fresh-cross-dataset-calibration-v1/inputs/"
                "catalog_infovqa_validation_id_projected_v2.json"
            ),
            "catalog_sha256": ("38907d03d42a6c52d898b3a543ad2cd737b85569c465ad301ca148aa72e4308c"),
            "output_path": (
                "/scratch/djy8hg/runs/mosaickv/"
                "choice-c-v102-v100-fresh-cross-dataset-calibration-v1/inputs/"
                "calibration_infovqa.json"
            ),
            "sample_count": 32,
            "selection_seed": 20320821,
            "observed_file_sha256": (
                "3308465e540f98dac1003e85a78403968a90ed19bd7d23fe5b2ec6ad8da258c5"
            ),
            "observed_file_bytes": 4290,
            "observed_mode": "0444",
            "observed_content_sha256": (
                "6da4d59676cf4def24c474feeef5c3e7fcf01062270f34b68a199a929b42c4fa"
            ),
        },
    ]
    if (
        materialization.get("status")
        != "consumed_two_deterministic_ledgers_reviewed_reexecution_disabled"
        or materialization.get("authorized_by_user") is not True
        or materialization.get("authorized_action")
        != "materialize_two_deterministic_id_ledgers_only"
        or materialization.get("execution_mode") != "local_cpu_process_only"
        or materialization.get("source_snapshot_must_be_clean_and_committed") is not True
        or materialization.get("reviewed_projected_catalog_review_git_sha")
        != "309551eb4c1f107d5cd831f96a0baa58125a5d44"
        or materialization.get("reviewed_environment_record_git_sha")
        != "29afaa6b307c2ff3c378a906b247d48d384c007d"
        or materialization.get("exclusion_manifest_sha256") != EXCLUSION_MANIFEST_FILE_SHA
        or materialization.get("ledgers") != expected_ledgers
        or materialization.get("deterministic_hash_selection_only") is not True
        or materialization.get(
            "prompt_answer_reference_score_media_or_failure_identity_access_forbidden"
        )
        is not True
        or materialization.get("immutable_write_once_mode") != "0444"
        or materialization.get("materialization_performed") is not True
        or materialization.get("materialization_source_git_sha")
        != "b5097f9bfd5770eda29fdfca6c08f7a19c865b6e"
        or materialization.get("materialization_source_git_dirty") is not False
        or materialization.get("materialization_protocol_sha256")
        != "d0db27ff5cd13cac863376252bd15deb8328394398cff60740b66aa7121e75d0"
        or materialization.get("ledgers_reviewed") is not True
        or materialization.get("materialization_reexecution_authorized") is not False
        or materialization.get("disjointness_check_authorized") is not True
        or any(
            materialization.get(key) is not False
            for key in (
                "calibration_execution_authorized",
                "candidate_freeze_authorized",
                "heldout_evaluation_authorized",
                "slurm_or_other_job_submission_authorized",
            )
        )
    ):
        raise RuntimeError("V102 ledger-materialization authorization exceeds its scope")

    disjointness = cast("dict[str, Any]", spec["disjointness_check_authorization"])
    expected_ledger_paths = [str(record["output_path"]) for record in expected_ledgers]
    expected_ledger_hashes = [str(record["observed_file_sha256"]) for record in expected_ledgers]
    if (
        disjointness.get("status")
        != "consumed_configuration_aware_disjointness_passed_reexecution_disabled"
        or disjointness.get("authorized_by_user") is not True
        or disjointness.get("authorized_action")
        != "check_two_reviewed_ledgers_against_sealed_exclusion_manifest_only"
        or disjointness.get("execution_mode") != "local_cpu_process_only"
        or disjointness.get("source_snapshot_must_be_clean_and_committed") is not True
        or disjointness.get("exclusion_manifest_sha256") != EXCLUSION_MANIFEST_FILE_SHA
        or disjointness.get("ledger_paths") != expected_ledger_paths
        or disjointness.get("ledger_sha256") != expected_ledger_hashes
        or disjointness.get("output_path")
        != (
            "/scratch/djy8hg/runs/mosaickv/"
            "choice-c-v102-v100-fresh-cross-dataset-calibration-v1/inputs/"
            "disjointness_report.json"
        )
        or disjointness.get("immutable_write_once_mode") != "0444"
        or disjointness.get("require_zero_overlap_with_exclusion_manifest") is not True
        or disjointness.get("require_zero_cross_ledger_duplicate_identities") is not True
        or disjointness.get("disjointness_check_performed") is not True
        or disjointness.get("validation_source_git_sha")
        != "11bf2f77bc62dbde82bfd8519f7715ba32850753"
        or disjointness.get("validation_source_git_dirty") is not False
        or disjointness.get("report_file_sha256")
        != "2f7f947ca016b8fb5ad425d9d9cfb8e304b9e3ffd95391216e6cdaf8117d36ac"
        or disjointness.get("report_content_sha256")
        != "0f5d20b178c93b51d1cf21ee4c1d26a8553726d2e8bbc12746138bb296b5ef14"
        or disjointness.get("report_bytes") != 1205
        or disjointness.get("report_mode") != "0444"
        or disjointness.get("selected_unique_sample_count") != 64
        or disjointness.get("overlap_count") != 0
        or disjointness.get("cross_ledger_duplicate_count") != 0
        or disjointness.get("report_reviewed") is not True
        or disjointness.get("reexecution_authorized") is not False
        or disjointness.get("calibration_execution_authorized") is not True
        or disjointness.get("candidate_freeze_authorized") is not True
        or any(
            disjointness.get(key) is not False
            for key in (
                "heldout_evaluation_authorized",
                "slurm_or_other_job_submission_authorized",
            )
        )
    ):
        raise RuntimeError("V102 disjointness-check authorization exceeds its scope")

    common_policy = cast("dict[str, Any]", spec["common_cache_policy"])
    candidates = cast("list[dict[str, Any]]", spec["candidates"])
    if len(candidates) != 1:
        raise RuntimeError("V102 must execute one frozen MosaicKV candidate")
    candidate = candidates[0]
    overrides = cast("dict[str, Any]", candidate.get("overrides"))
    expected_hierarchical = {
        "stable_head_hierarchical_logit_safe_eviction": True,
        "stable_head_hierarchical_partition_count": 8,
        "stable_head_hierarchical_maximum_refinement_depth": 2,
        "stable_head_hierarchical_maximum_candidate_evaluations": 24,
        "stable_head_hierarchical_maximum_sequential_rounds": 6,
        "stable_head_hierarchical_maximum_rollback_rounds": 2,
        "stable_head_hierarchical_horizon": 2,
        "stable_head_hierarchical_maximum_kl": 0.02,
        "stable_head_hierarchical_maximum_logit_difference": 1.0,
        "stable_head_query_coverage_cross_head_allocation": True,
        "stable_head_query_coverage_layout_group_atomic_selection": True,
        "stable_head_query_coverage_layout_group_atomic_selection_mode": (
            "pareto_joint_donor_knapsack"
        ),
        "stable_head_progressive_logit_safe_eviction": False,
        "stable_head_short_horizon_logit_stability": False,
        "maximum_prototypes_per_head": 0,
        "prototype_budget_planning_mode": "disabled_exact_only",
        "active_budget_bytes": 28_500_000,
    }
    robust = cast("dict[str, Any]", spec["robust_fidelity_analysis"])
    if (
        spec.get("candidate_implementation_git_sha") != SOURCE_SHA
        or spec.get("candidate_config") != "configs/experiments/choice_c_v72_frozen_candidate.yaml"
        or spec.get("model_id") != "llava-hf/llava-1.5-7b-hf"
        or spec.get("model_revision") != "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
        or spec.get("backend") != "hf_eager_prefill_sdpa_decode"
        or spec.get("seed") != 17
        or spec.get("output_length") != 64
        or spec.get("warmups_per_arm") != 1
        or spec.get("measured_repetitions_per_arm") != 1
        or common_policy
        != {
            "block_size": 4,
            "physical_budget_block_size": 4,
            "physical_budget_mandatory_policy": "last_token",
        }
        or candidate.get("id") != "v100_hierarchical_head_local_logit_safe_exact"
        or candidate.get("tier_policy") != "exact_only_replacement"
        or any(overrides.get(key) != value for key, value in expected_hierarchical.items())
        or robust.get("labels_accessed") is not False
        or robust.get("official_task_scores_accessed") is not False
        or robust.get("fold_count") != 4
        or robust.get("expected_samples_per_fold") != 8
        or robust.get("require_every_task_and_fold") is not True
        or robust.get("require_base_memory_and_fidelity_gates") is not True
        or robust.get("fullkv_answer_noninferiority_margin_vs_prompt_topk") != 0.05
        or robust.get("fullkv_token_noninferiority_margin_vs_prompt_topk") != 0.05
    ):
        raise RuntimeError("V102 calibration runtime or robust gate differs")

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
        "require_zero_residual_bytes": True,
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
    ):
        raise RuntimeError("V102 weakened the frozen cache, memory, or fidelity gates")

    review = cast("dict[str, Any]", spec["review_gate_before_any_action"])
    submission = cast("dict[str, Any]", spec["submission_policy"])
    if (
        review.get("protocol_authored") is not True
        or review.get("protocol_commit_created") is not True
        or review.get("protocol_reviewed_after_commit") is not True
        or review.get("configuration_aware_tooling_implemented_and_tested") is not True
        or review.get("configuration_aware_tooling_committed") is not True
        or review.get("configuration_aware_tooling_sealed") is not True
        or review.get("tooling_seal_reviewed_after_commit") is not True
        or review.get("exclusion_scan_scope_reviewed") is not True
        or review.get("exclusion_manifest_reviewed") is not True
        or review.get("exclusion_manifest_sealed") is not True
        or review.get("exclusion_manifest_provenance_seal_reviewed_after_commit") is not True
        or review.get("candidate_catalog_construction_scope_reviewed") is not True
        or review.get("candidate_catalog_tooling_committed_and_sealed") is not True
        or review.get("candidate_catalog_tooling_seal_reviewed_after_commit") is not True
        or review.get("candidate_catalogs_reviewed") is not True
        or review.get("candidate_catalogs_eligible_for_materialization") is not True
        or review.get("replacement_projected_tooling_committed_sealed_and_reviewed") is not True
        or review.get("replacement_catalog_construction_scope_authorized") is not True
        or review.get("replacement_catalogs_reviewed") is not True
        or review.get("replacement_catalogs_eligible_for_materialization") is not True
        or review.get("common_environment_reseal_reviewed") is not True
        or review.get("deterministic_ledger_materialization_scope_authorized") is not False
        or review.get("ledgers_reviewed") is not True
        or review.get("disjointness_check_scope_authorized") is not False
        or review.get("disjointness_report_reviewed") is not True
        or review.get("calibration_execution_scope_authorized") is not True
        or review.get("conditional_freeze_analysis_authorized") is not True
        or submission.get("calibration_job_allowed") is not True
        or submission.get("freeze_job_allowed") is not True
        or any(
            submission.get(key) is not False
            for key in (
                "exclusion_scan_job_allowed",
                "materialization_job_allowed",
                "heldout_job_allowed",
            )
        )
    ):
        raise RuntimeError("V102 incorrectly authorizes an action")

    return {
        "experiment_id": spec["experiment_id"],
        "candidate_implementation_git_sha": SOURCE_SHA,
        "tasks": {name: int(task["sample_count"]) for name, task in by_task.items()},
        "selection_seed": 20320821,
        "active_memory_reduction_gate": 0.30,
        "fidelity_noninferiority_margin": 0.05,
        "all_actions_authorized": False,
        "exclusion_scan_authorized": False,
        "candidate_catalog_construction_authorized": False,
        "materialization_authorized": False,
        "disjointness_check_authorized": False,
        "calibration_execution_authorized": True,
        "candidate_freeze_authorized": True,
        "exclusion_manifest_status": "reviewed_sealed_read_only",
        "status": "valid_small_calibration_and_conditional_freeze_authorized_v102_protocol",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--verify-local-metadata", action="store_true")
    parser.add_argument("--verify-exclusion-manifest", action="store_true")
    parser.add_argument("--verify-projected-catalogs", action="store_true")
    parser.add_argument("--require-calibration-authorized", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            validate(
                args.spec.resolve(),
                verify_local_metadata=bool(args.verify_local_metadata),
                verify_exclusion_manifest=bool(args.verify_exclusion_manifest),
                verify_projected_catalogs=bool(args.verify_projected_catalogs),
                require_calibration_authorized=bool(args.require_calibration_authorized),
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
