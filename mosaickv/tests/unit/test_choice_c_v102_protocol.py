from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml


def _protocol(package: Path) -> dict:
    return yaml.safe_load(
        (
            package / "configs/experiments/choice_c_v102_v100_fresh_cross_dataset_calibration.yaml"
        ).read_text(encoding="utf-8")
    )


def _validator(package: Path) -> ModuleType:
    path = package / "scripts/validate_choice_c_v102_preregistration.py"
    spec = importlib.util.spec_from_file_location("mosaickv_test_v102_protocol", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sample_tooling(package: Path) -> ModuleType:
    path = package / "scripts/choice_c_v102_sample_tooling.py"
    spec = importlib.util.spec_from_file_location("mosaickv_test_v102_reviewed_tooling", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _calibration_worker(package: Path) -> ModuleType:
    path = package / "scripts/run_choice_c_v74_label_blind_calibration_worker.py"
    spec = importlib.util.spec_from_file_location("mosaickv_test_v102_worker", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_v102_is_sealed_negative_and_disables_every_future_action() -> None:
    package = Path(__file__).parents[2]
    protocol = _protocol(package)

    assert protocol["classification"] == "sealed_negative_fail_fast_debug_gate_result"
    assert protocol["protocol_status"] == (
        "negative_debug_gate_sealed_all_future_execution_disabled"
    )
    assert protocol["tooling_implementation_authorized"] is True
    assert protocol["tooling_cpu_testing_authorized"] is True
    assert protocol["candidate_source"]["candidate_implementation_git_sha"] == (
        "820037afb3cc74a9f949df1555a2441680f9660e"
    )
    assert protocol["candidate_source"]["preregistration_git_sha"] == (
        "9eafc34af15676ac36a0327ff91a492feb6d862d"
    )
    assert protocol["candidate_source"]["catalog_constructor_implementation_git_sha"] == (
        "67beb0af20022228643eb211d2c5c06395b98db4"
    )
    assert (
        "mosaickv/tests/unit/test_choice_c_v102_catalog_tooling.py"
        in (protocol["candidate_source"]["execution_diff_allowlist"])
    )
    assert (
        "mosaickv/analysis/choice_c_v102_catalog_scope_negative.md"
        in (protocol["candidate_source"]["execution_diff_allowlist"])
    )
    assert (
        "mosaickv/analysis/choice_c_v102_projected_catalog_review.md"
        in (protocol["candidate_source"]["execution_diff_allowlist"])
    )
    assert (
        "mosaickv/analysis/choice_c_v102_common_environment_reseal.md"
        in (protocol["candidate_source"]["execution_diff_allowlist"])
    )
    assert (
        "mosaickv/analysis/choice_c_v102_disjointness_review.md"
        in (protocol["candidate_source"]["execution_diff_allowlist"])
    )
    for field in (
        "exclusion_scan_authorized",
        "materialization_authorized",
        "disjointness_check_authorized",
        "heldout_evaluation_authorized",
    ):
        assert protocol[field] is False
    assert protocol["calibration_execution_authorized"] is False
    assert protocol["candidate_freeze_authorized"] is False
    assert protocol["candidate_catalog_construction_authorized"] is False
    submission = protocol["submission_policy"]
    assert submission["calibration_job_allowed"] is False
    assert submission["freeze_job_allowed"] is False
    assert submission["heldout_job_allowed"] is False
    negative = protocol["negative_debug_gate_seal"]
    assert negative["debug_gate_passed"] is False
    assert negative["both_tasks_reached_compressed_target"] is False
    assert negative["full_calibration_submitted"] is False
    assert negative["frozen_candidate_created"] is False
    assert negative["all_v102_ledger_samples_ineligible_for_future_calibration"] is True
    assert negative["future_protocol_requires_fresh_samples"] is True


def test_v102_sealed_materialization_freeze_satisfies_worker_gate() -> None:
    package = Path(__file__).parents[2]
    protocol = _protocol(package)
    worker = _calibration_worker(package)

    assert protocol.get("selection_rule_frozen_before_calibration") is None
    assert worker._selection_rule_frozen(protocol) is True

    mismatched = copy.deepcopy(protocol)
    mismatched["candidate_source"]["candidate_implementation_git_sha"] = "0" * 40
    assert worker._selection_rule_frozen(mismatched) is False

    mutable = copy.deepcopy(protocol)
    mutable["candidate_source"]["runtime_behavior_changes_after_candidate_commit_allowed"] = True
    assert worker._selection_rule_frozen(mutable) is False


def test_v102_debug_job_is_one_sample_per_task_and_cannot_freeze() -> None:
    package = Path(__file__).parents[2]
    script = (package / "slurm/choice_c_v102_debug_a100.sbatch").read_text(encoding="utf-8")

    assert "#SBATCH --array=0-1%2" in script
    assert "#SBATCH --gres=gpu:a100:1" in script
    assert "--warmup-only" in script
    assert "choice-c-v102-v100-fresh-cross-dataset-calibration-v1-debug1" in script
    assert "frozen_candidate.yaml" not in script
    assert "heldout" not in script.casefold()
    assert "--require-calibration-authorized" in script
    guarded_paths = (
        package / "scripts/submit_choice_c_v102.sh",
        package / "slurm/choice_c_v102_calibration_a100.sbatch",
        package / "slurm/choice_c_v102_freeze.sbatch",
    )
    assert all(
        "--require-calibration-authorized" in path.read_text(encoding="utf-8")
        for path in guarded_paths
    )


def test_v102_uses_configuration_aware_untouched_candidate_pools() -> None:
    package = Path(__file__).parents[2]
    protocol = _protocol(package)
    tasks = {task["task"]: task for task in protocol["tasks"]}

    text = tasks["textvqa_train_calibration"]
    assert (text["dataset_id"], text["dataset_config"], text["split"]) == (
        "lmms-lab/textvqa",
        "default",
        "train",
    )
    assert text["expected_split_cardinality"] == 34602
    assert text["candidate_catalog_id_field"] == "question_id"
    assert text["candidate_catalog_parquet_glob"] == "data/train-*.parquet"
    assert text["candidate_catalog_expected_shard_count"] == 20
    assert text["candidate_catalog_path"].endswith("catalog_textvqa_train_id_projected_v2.json")
    info = tasks["infovqa"]
    assert (info["dataset_id"], info["dataset_config"], info["split"]) == (
        "lmms-lab-encoder/DocVQA",
        "InfographicVQA",
        "validation",
    )
    assert info["expected_split_cardinality"] == 2801
    assert info["candidate_catalog_id_field"] == "questionId"
    assert info["candidate_catalog_parquet_glob"] == "InfographicVQA/validation-*.parquet"
    assert info["candidate_catalog_expected_shard_count"] == 4
    assert info["candidate_catalog_path"].endswith(
        "catalog_infovqa_validation_id_projected_v2.json"
    )
    assert {task["sample_count"] for task in tasks.values()} == {32}
    assert {task["selection_seed"] for task in tasks.values()} == {20320821}
    assert all(
        task["ledger_status"] == "materialized_reviewed_disjointness_passed_read_only"
        for task in tasks.values()
    )
    assert all(
        task["candidate_catalog_status"] == "projected_content_valid_scope_positive_read_only"
        for task in tasks.values()
    )
    tooling = protocol["identity_and_tooling_requirements"]
    assert tooling["canonical_identity"] == (
        "dataset_id_dataset_config_dataset_revision_split_sample_id"
    )
    assert tooling["implementation_status"] == "committed_sealed_cpu_tested_scan_disabled"
    assert tooling["configuration_aware_identity_implemented"] is True
    assert tooling["textvqa_train_split_override_implemented"] is True
    assert tooling["infovqa_task_route_implemented"] is True
    assert tooling["real_exclusion_scan_performed"] is True
    review = protocol["review_gate_before_any_action"]
    assert review["configuration_aware_tooling_committed"] is True
    assert review["configuration_aware_tooling_sealed"] is True
    seal = protocol["configuration_aware_tooling_seal"]
    assert seal["implementation_git_sha"] == "97f3f770da3681ff4a2879da328d6dc2bb30713d"
    assert seal["seal_git_sha"] == "763b92fa01fcfeaa4db880a4d9f46b128217ea1b"
    assert seal["seal_reviewed_after_commit"] is True
    assert len(seal["committed_files"]) == 10
    assert seal["validation"]["exclusion_scan_performed"] is False
    scan = protocol["exclusion_scan_authorization"]
    assert scan["status"] == "consumed_after_successful_local_cpu_scan_reexecution_disabled"
    assert scan["execution_mode"] == "local_cpu_process_only"
    assert scan["scan_execution_performed"] is True
    assert scan["scan_reexecution_authorized"] is False
    assert scan["exclusion_manifest_reviewed"] is True
    assert scan["slurm_or_other_job_submission_authorized"] is False
    snapshot = protocol["required_exclusion_snapshot"]
    assert snapshot["exclusion_manifest_file_sha256"] == (
        "128cc7028b86fd6965e23a345181c8ad8762cd0a6b1816bfa45da1b0f2833d14"
    )
    assert snapshot["excluded_unique_sample_count"] == 0
    assert snapshot["provenance_seal_git_sha"] == ("a7a146fd8d58aa5c1d0720f2040c6bf8588dbf2a")
    assert snapshot["candidate_availability_or_disjointness_claimed"] is True
    assert snapshot["disjointness_report_status"] == "passed_reviewed_read_only"
    assert snapshot["overlap_count"] == 0
    assert snapshot["cross_ledger_duplicate_count"] == 0
    catalog = protocol["initial_candidate_catalog_construction_record"]
    assert catalog["status"] == (
        "consumed_content_valid_construction_scope_negative_reexecution_disabled"
    )
    assert catalog["reviewed_catalog_tooling_seal_git_sha"] == (
        "222647c7df2275c98d0f74a5c24ecb2675e3b0b9"
    )
    assert catalog["authorization_guard_sha256"] == (
        "9d8bd919dceeb138344d8420bff594b5ec3b7407c80187305f61cb19c080199f"
    )
    assert catalog["require_canonical_committed_spec_path"] is True
    assert catalog["execution_mode"] == "local_cpu_process_only"
    assert catalog["catalog_construction_performed"] is True
    assert catalog["catalogs_reviewed"] is True
    assert catalog["construction_scope_passed"] is False
    assert catalog["reexecution_authorized"] is False
    assert catalog["replacement_catalogs_required"] is True
    assert catalog["ledger_materialization_authorized"] is False
    assert len(catalog["catalogs"]) == 2
    assert {record["id_field"] for record in catalog["catalogs"]} == {
        "question_id",
        "questionId",
    }
    catalog_seal = protocol["initial_candidate_catalog_tooling_seal"]
    assert catalog_seal["implementation_git_sha"] == ("9cfd6822d968b593c98e2faf2f95eadb7b585f39")
    assert catalog_seal["seal_git_sha"] == ("222647c7df2275c98d0f74a5c24ecb2675e3b0b9")
    assert catalog_seal["seal_reviewed_after_commit"] is True
    assert catalog_seal["catalog_construction_execution_enabled"] is True
    assert len(catalog_seal["committed_files"]) == 4
    replacement = protocol["replacement_projected_catalog_tooling"]
    assert replacement["status"] == (
        "committed_cpu_tested_sealed_projected_catalogs_constructed_reviewed"
    )
    assert replacement["implementation_git_sha"] == ("67beb0af20022228643eb211d2c5c06395b98db4")
    assert replacement["use_datasets_load_dataset"] is False
    assert replacement["unrequested_split_open_forbidden"] is True
    assert replacement["real_replacement_catalogs_constructed"] is True
    assert replacement["tooling_sealed"] is True
    assert replacement["replacement_construction_authorized"] is False
    projected = protocol["candidate_catalog_construction_authorization"]
    assert projected["status"] == (
        "consumed_projected_catalogs_content_valid_scope_positive_reexecution_disabled"
    )
    assert projected["catalog_construction_performed"] is True
    assert projected["catalogs_reviewed"] is True
    assert projected["construction_scope_passed"] is True
    assert projected["reexecution_authorized"] is False
    assert projected["ledger_materialization_authorized"] is False
    assert len(projected["catalogs"]) == 2
    environment = protocol["common_environment_reseal"]
    assert environment["status"] == ("committed_reviewed_exact_common_environment_cpu_gate_passed")
    assert environment["record_git_sha"] == "29afaa6b307c2ff3c378a906b247d48d384c007d"
    assert environment["locked_distribution_count"] == 251
    assert environment["full_test_passed_count"] == 743
    assert environment["full_test_failure_count"] == 0
    materialization = protocol["ledger_materialization_authorization"]
    assert materialization["status"] == (
        "consumed_two_deterministic_ledgers_reviewed_reexecution_disabled"
    )
    assert materialization["materialization_performed"] is True
    assert materialization["ledgers_reviewed"] is True
    assert materialization["disjointness_check_authorized"] is True
    disjointness = protocol["disjointness_check_authorization"]
    assert disjointness["status"] == (
        "consumed_configuration_aware_disjointness_passed_reexecution_disabled"
    )
    assert disjointness["disjointness_check_performed"] is True
    assert disjointness["overlap_count"] == 0
    assert disjointness["cross_ledger_duplicate_count"] == 0
    assert disjointness["calibration_execution_authorized"] is True


def test_v102_preserves_v101_negative_and_unchanged_gates() -> None:
    package = Path(__file__).parents[2]
    protocol = _protocol(package)
    predecessor = protocol["predecessor_feasibility_negative"]
    gates = protocol["unchanged_memory_and_fidelity_gates"]

    assert predecessor["eligible_docvqa_validation_ids"] == 0
    assert predecessor["eligible_textvqa_validation_ids"] == 0
    assert predecessor["v101_materialization_remains_disabled"] is True
    assert gates["minimum_active_reduction_fraction_vs_prompt_topk"] == 0.30
    assert gates["minimum_active_plus_residual_reduction_fraction_vs_prompt_topk"] == 0.30
    assert gates["fullkv_answer_noninferiority_margin_vs_prompt_topk"] == 0.05
    assert gates["fullkv_token_noninferiority_margin_vs_prompt_topk"] == 0.05
    assert gates["require_every_task"] is True
    assert gates["require_every_fold"] is True


def test_v102_validator_accepts_negative_seal_but_rejects_submission() -> None:
    package = Path(__file__).parents[2]
    path = package / "configs/experiments/choice_c_v102_v100_fresh_cross_dataset_calibration.yaml"
    result = _validator(package).validate(path)

    assert result == {
        "experiment_id": "choice_c_v102_v100_fresh_cross_dataset_calibration",
        "candidate_implementation_git_sha": ("820037afb3cc74a9f949df1555a2441680f9660e"),
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
    with pytest.raises(PermissionError, match="sealed negative"):
        _validator(package).validate(path, require_calibration_authorized=True)


def test_v102_catalog_reexecution_is_blocked_after_review(
    tmp_path: Path,
) -> None:
    package = Path(__file__).parents[2]
    protocol = _protocol(package)
    tooling = _sample_tooling(package)
    canonical = (
        package / "configs/experiments/choice_c_v102_v100_fresh_cross_dataset_calibration.yaml"
    )

    with pytest.raises(PermissionError, match="not fully sealed"):
        tooling._require_catalog_authorization_seal(protocol)
    tooling._require_canonical_spec_path(canonical, package_root=package)

    unreviewed = copy.deepcopy(protocol)
    unreviewed["replacement_projected_catalog_tooling"]["implementation_reviewed_after_commit"] = (
        False
    )
    with pytest.raises(PermissionError, match="not fully sealed"):
        tooling._require_catalog_authorization_seal(unreviewed)
    with pytest.raises(PermissionError, match="canonical spec path"):
        tooling._require_canonical_spec_path(tmp_path / "forged.yaml", package_root=package)


def test_v102_disjointness_scope_requires_exact_reviewed_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = Path(__file__).parents[2]
    protocol = _protocol(package)
    tooling = _sample_tooling(package)
    authorization = protocol["disjointness_check_authorization"]
    authorization["status"] = "configuration_aware_cpu_disjointness_check_authorized_not_executed"
    authorization["disjointness_check_performed"] = False
    authorization["report_reviewed"] = False
    manifest = tmp_path / "manifest.json"
    ledgers = (tmp_path / "text.json", tmp_path / "info.json")
    output = tmp_path / "report.json"
    authorization["exclusion_manifest_path"] = str(manifest)
    authorization["exclusion_manifest_sha256"] = "manifest-sha"
    authorization["ledger_paths"] = [str(path) for path in ledgers]
    authorization["ledger_sha256"] = ["text-sha", "info-sha"]
    authorization["output_path"] = str(output)
    hashes = {
        manifest.resolve(): "manifest-sha",
        ledgers[0].resolve(): "text-sha",
        ledgers[1].resolve(): "info-sha",
    }
    monkeypatch.setattr(tooling, "sha256_file", lambda path: hashes[path.resolve()])

    tooling._require_disjointness_authorization(
        protocol,
        exclusion_manifest=manifest,
        ledger_paths=ledgers,
        output=output,
    )
    with pytest.raises(PermissionError, match="report path differs"):
        tooling._require_disjointness_authorization(
            protocol,
            exclusion_manifest=manifest,
            ledger_paths=ledgers,
            output=tmp_path / "wrong.json",
        )
