from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml


def _protocol(package: Path) -> dict:
    return yaml.safe_load(
        (
            package / "configs/experiments/choice_c_v101_v100_fresh_disjoint_calibration.yaml"
        ).read_text(encoding="utf-8")
    )


def _validator(package: Path) -> ModuleType:
    path = package / "scripts/validate_choice_c_v101_preregistration.py"
    spec = importlib.util.spec_from_file_location("mosaickv_test_v101_protocol", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_v101_is_sealed_feasibility_negative_unmaterialized_and_unsubmitted() -> None:
    package = Path(__file__).parents[2]
    protocol = _protocol(package)

    assert protocol["candidate_source"]["candidate_implementation_git_sha"] == (
        "820037afb3cc74a9f949df1555a2441680f9660e"
    )
    assert protocol["candidate_source"]["tooling_implementation_git_sha"] == (
        "25e173af78fb2e748794755575a8c34f98d1dde8"
    )
    assert protocol["protocol_status"] == (
        "sealed_feasibility_negative_full_validation_splits_exhausted"
    )
    assert protocol["selection_rule_frozen_before_materialization"] is True
    assert protocol["exclusion_scan_authorized"] is False
    assert protocol["materialization_authorized"] is False
    assert protocol["disjointness_check_authorized"] is False
    assert protocol["calibration_execution_authorized"] is False
    assert protocol["candidate_freeze_authorized"] is False
    assert protocol["heldout_evaluation_authorized"] is False
    assert all(
        value is False
        for key, value in protocol["submission_policy"].items()
        if key.endswith("_allowed")
    )
    tooling = protocol["sample_tooling_implementation"]
    assert tooling["exclusion_scanner_implemented"] is True
    assert tooling["deterministic_materializer_implemented"] is True
    assert tooling["disjointness_checker_implemented"] is True
    assert tooling["tooling_commit_created"] is True
    assert tooling["tooling_commit_sha"] == "25e173af78fb2e748794755575a8c34f98d1dde8"
    assert tooling["synthetic_cpu_tests_passed"] is True
    assert tooling["real_dataset_access_performed"] is False
    assert tooling["real_dataset_samples_or_labels_accessed"] is False
    assert tooling["real_exclusion_scan_performed"] is True
    assert tooling["real_output_directory_created"] is True
    assert tooling["real_ledger_materialized"] is False
    assert tooling["slurm_job_submitted"] is False


def test_v101_records_zero_eligible_docvqa_and_textvqa_ids() -> None:
    package = Path(__file__).parents[2]
    protocol = _protocol(package)
    tasks = protocol["tasks"]
    exclusion = protocol["required_exclusion_snapshot"]

    assert {task["task"] for task in tasks} == {"docvqa", "textvqa"}
    assert {task["sample_count"] for task in tasks} == {32}
    assert {task["selection_seed"] for task in tasks} == {20310821}
    assert all(task["ledger_status"] == "blocked_zero_eligible_validation_ids" for task in tasks)
    assert exclusion["exclude_all_repository_ledgers"] is True
    assert exclusion["exclude_every_prior_choice_c_input_ledger"] is True
    assert exclusion["exclude_every_prior_choice_c_observed_result_row"] is True
    assert exclusion["require_zero_overlap_for_each_task"] is True
    materialization = protocol["future_materialization_rule"]
    assert materialization["authorized"] is False
    assert materialization["blocked_reason"] == (
        "exclusion_manifest_covers_every_pinned_validation_example"
    )
    assert materialization["inspect_media_or_prompt_content_for_selection"] is False
    assert (
        materialization["inspect_answers_references_scores_or_failure_identity_for_selection"]
        is False
    )


def test_v101_keeps_v100_memory_and_fidelity_gates() -> None:
    package = Path(__file__).parents[2]
    protocol = _protocol(package)
    gates = protocol["unchanged_memory_and_fidelity_gates"]
    cache = protocol["cache_contract"]

    assert gates["minimum_active_reduction_fraction_vs_prompt_topk"] == 0.30
    assert gates["minimum_active_plus_residual_reduction_fraction_vs_prompt_topk"] == 0.30
    assert gates["fullkv_answer_noninferiority_margin_vs_prompt_topk"] == 0.05
    assert gates["fullkv_token_noninferiority_margin_vs_prompt_topk"] == 0.05
    assert gates["robust_fold_count"] == 4
    assert gates["expected_samples_per_fold"] == 8
    assert gates["require_every_task"] is True
    assert gates["require_every_fold"] is True
    assert gates["require_zero_residual_bytes"] is True
    assert gates["require_every_sample_to_reach_target"] is True
    assert gates["require_every_final_joint_logit_gate"] is True
    assert cache["common_block_size_tokens"] == 4
    assert cache["candidate_active_byte_cap"] == 28_500_000
    assert cache["candidate_tier_policy"] == "exact_only"
    assert cache["prototype_merging_enabled"] is False
    assert cache["residual_storage_enabled"] is False
    assert cache["repair_enabled"] is False


def test_v101_validator_verifies_seal_or_rejects_missing_git_archive() -> None:
    package = Path(__file__).parents[2]
    path = package / "configs/experiments/choice_c_v101_v100_fresh_disjoint_calibration.yaml"
    validator = _validator(package)
    archive = subprocess.run(
        [
            "git",
            "-C",
            str(package.parent),
            "cat-file",
            "-e",
            f"{validator.FEASIBILITY_SEAL_GIT_SHA}:mosaickv/src/mosaickv/evaluation/disjoint_samples.py",
        ],
        capture_output=True,
        check=False,
    )
    if archive.returncode:
        # A squashed/distributed checkout need not contain the old object. This
        # is an explicit rejection test, NOT a successful historical validation.
        with pytest.raises(subprocess.CalledProcessError) as error:
            validator.validate(path)
        assert validator.FEASIBILITY_SEAL_GIT_SHA in " ".join(error.value.cmd)
        assert error.value.returncode != 0
        return
    result = validator.validate(path)

    assert result == {
        "experiment_id": "choice_c_v101_v100_fresh_disjoint_calibration",
        "candidate_implementation_git_sha": ("820037afb3cc74a9f949df1555a2441680f9660e"),
        "tasks": {"docvqa": 32, "textvqa": 32},
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
