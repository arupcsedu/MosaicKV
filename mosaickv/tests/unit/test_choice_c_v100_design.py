from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml


def _validator(package: Path) -> ModuleType:
    path = package / "scripts/validate_choice_c_v100_design.py"
    spec = importlib.util.spec_from_file_location("mosaickv_test_v100_design", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _design(package: Path) -> dict:
    return yaml.safe_load(
        (
            package
            / "configs/experiments/choice_c_v100_hierarchical_head_local_logit_safe_design.yaml"
        ).read_text(encoding="utf-8")
    )


def _debug_runner(package: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(package / "scripts"))
    path = package / "scripts/run_choice_c_v100_gpu_debug.py"
    spec = importlib.util.spec_from_file_location("mosaickv_test_v100_debug", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_v100_gpu_debug_passed_but_calibration_submission_is_blocked() -> None:
    package = Path(__file__).parents[2]
    design = _design(package)

    assert design["implementation_status"] == "bounded_hf_probe_loop_gpu_debug_verified"
    assert design["execution_status"] == "retention_one_and_one_sample_gpu_debug_passed"
    assert design["submission_authorized"] is False
    assert design["candidate_freeze_authorized"] is False
    assert design["heldout_evaluation_authorized"] is False
    assert all(
        value is False
        for key, value in design["submission_policy"].items()
        if key.endswith("_allowed")
    )
    implementation = design["implementation_gate_before_submission"]
    assert implementation["strict_config_fields_added"] is True
    assert implementation["hierarchical_search_implemented"] is True
    assert implementation["probe_budget_tests_passed"] is True
    assert implementation["synthetic_interaction_rollback_tests_passed"] is True
    assert implementation["isolated_cache_state_tests_passed"] is True
    assert implementation["bounded_hf_probe_loop_integrated"] is True
    assert implementation["mocked_isolated_probe_loop_tests_passed"] is True
    assert implementation["clean_commit_created"] is True
    assert implementation["retention_one_fullkv_parity_passed"] is True
    assert implementation["one_sample_gpu_debug_passed"] is True
    evidence = design["gpu_debug_evidence"]
    assert evidence["paper_quality_claim_authorized"] is False
    assert evidence["retention_one"]["token_agreement"] == 1.0
    assert evidence["retention_one"]["maximum_logit_difference"] == 0.0
    assert evidence["compressed_one_sample"]["scheduler_state"] == "FAILED"
    assert (
        evidence["compressed_one_sample"]["postprocessing_failure"]
        == "obsolete_selection_configuration_key"
    )
    assert evidence["compressed_one_sample"]["summary_regenerated_without_gpu_rerun"] is True
    authorization = design["gpu_debug_authorization"]
    assert authorization["calibration_or_heldout_authorized"] is False
    assert authorization["retention_one_must_pass_before_compressed_debug"] is True
    assert authorization["compressed_debug_must_use_afterok_dependency"] is True


def test_v100_uses_no_v99_sample_level_information() -> None:
    package = Path(__file__).parents[2]
    boundary = _design(package)["scientific_boundary"]

    assert boundary["v99_used_only_as_exclusion_boundary"] is True
    assert boundary["v99_answers_used_for_design"] is False
    assert boundary["v99_failure_identities_used_for_design"] is False
    assert boundary["v99_generated_token_ids_used_for_design"] is False
    assert boundary["v99_per_sample_telemetry_used_for_design"] is False
    assert boundary["calibration_must_use_fresh_disjoint_ledgers"] is True


def test_v100_replaces_sequential_probing_without_weakening_gates() -> None:
    package = Path(__file__).parents[2]
    design = _design(package)
    algorithm = design["algorithm"]
    probes = design["probe_protocol"]
    gates = design["unchanged_freeze_gates"]

    assert algorithm["initial_state"] == "exact_fullkv"
    assert algorithm["route_capacity_ladders"]["nested"] is True
    assert algorithm["hierarchy"]["maximum_binary_refinement_depth"] == 2
    assert algorithm["global_assembly"]["solver"] == "deterministic_multiple_choice_knapsack"
    assert algorithm["joint_verification"]["required"] is True
    assert algorithm["fail_closed_behavior"]["unverified_target_cache_fallback_forbidden"] is True
    assert probes["maximum_candidate_cache_evaluations"] == 24
    assert probes["maximum_sequential_decision_rounds"] == 6
    assert probes["maximum_forward_kl"] == 0.02
    assert probes["maximum_absolute_logit_difference"] == 1.0
    assert gates["minimum_active_reduction_fraction_vs_prompt_topk"] == 0.30
    assert gates["fullkv_answer_noninferiority_margin_vs_prompt_topk"] == 0.05
    assert gates["fullkv_token_noninferiority_margin_vs_prompt_topk"] == 0.05
    assert design["strict_runtime_configuration"] == {
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


def test_v100_design_validator_accepts_protocol() -> None:
    package = Path(__file__).parents[2]
    validator = _validator(package)
    result = validator.validate(
        package / "configs/experiments/choice_c_v100_hierarchical_head_local_logit_safe_design.yaml"
    )

    assert result["status"] == "valid_gpu_debug_passed_calibration_blocked_protocol"
    assert result["task_count"] == 2
    assert result["samples_per_task"] == 32
    assert result["maximum_candidate_cache_evaluations"] == 24
    assert result["maximum_sequential_decision_rounds"] == 6
    assert result["submission_authorized"] is False


def test_v100_debug_uses_frozen_common_cache_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = Path(__file__).parents[2]
    config = _debug_runner(package, monkeypatch)._v100_config(package, max_new_tokens=4)

    assert config.cache.block_size == 4
    assert config.cache.physical_budget_block_size == 4
    assert config.cache.physical_budget_mandatory_policy == "last_token"
    assert config.cache.budget_value == 28_500_000
    assert config.cache.total_storage_budget_bytes == 28_500_000
    assert config.selection.stable_head_hierarchical_logit_safe_eviction is True
    assert config.selection.stable_head_progressive_logit_safe_eviction is False


def test_v100_artifact_validator_reads_runtime_routing_and_verifies_accounting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = Path(__file__).parents[2]
    runner = _debug_runner(package, monkeypatch)
    runtime_sha = "a" * 40
    validation_sha = "b" * 40
    run_id = f"v100-single-sample-{runtime_sha[:12]}"
    run_root = tmp_path / run_id
    trace_root = tmp_path / "traces" / run_id
    run_root.mkdir(parents=True)
    trace_root.mkdir(parents=True)
    row = {
        "status": "completed",
        "error": None,
        "active_kv_bytes": 96,
        "residual_kv_bytes": 0,
        "answer": "blue",
        "task_score": 1.0,
    }
    raw_path = run_root / "results.jsonl"
    raw_path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    parquet_path = run_root / "aggregate.parquet"
    parquet_path.write_bytes(b"representative-parquet")
    evaluation = {
        "passed": True,
        "proposal": {"proposal_id": "joint-0"},
        "result": {
            "evaluated_steps": 2,
            "greedy_token_match": True,
            "maximum_forward_kl": 0.01,
            "maximum_absolute_logit_difference": 0.5,
        },
    }
    trace = {
        "status": "completed",
        "answer": "blue",
        "active_kv_bytes": 96,
        "final_active_kv_bytes": 100,
        "byte_budget_materialization": {
            "logical_active_bytes": 92,
            "packed_padding_bytes": 4,
            "final_packed_active_bytes": 96,
            "configured_active_byte_cap": 100,
        },
        "forecast_statistics": {},
        "generated_token_ids": [1, 2, 3, 4],
        "graph_edges": [],
        "hierarchical_logit_safe_eviction": {
            "status": "completed_label_blind",
            "evaluations": [evaluation],
            "evaluations_used": 1,
            "rounds_used": 1,
            "probe_state_discarded_before_measured_decode": True,
            "selected_candidate_passed": True,
            "final_joint_probe_passed": True,
            "target_reached": True,
            "used_fullkv_fallback": False,
            "unresolved_exact_routes": [],
        },
        "online_uncertainty_routing": {"hierarchical_gpu_debug_authorized": True},
        "prototypes": [],
        "repair_events": [],
        "selected_blocks": [],
        "timing_breakdown": {},
    }
    trace_path = trace_root / "trace.json"
    trace_path.write_text(json.dumps(trace, sort_keys=True), encoding="utf-8")
    trace_digest = hashlib.sha256()
    trace_digest.update(trace_path.name.encode())
    trace_digest.update(trace_path.read_bytes())
    manifest = {
        "source": {"git_sha": runtime_sha, "git_dirty": False},
        "artifacts": {
            "raw_output_sha": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            "metrics_sha": hashlib.sha256(parquet_path.read_bytes()).hexdigest(),
            "log_sha": trace_digest.hexdigest(),
        },
        "hardware": {"gpu_type": "NVIDIA A100-SXM4-80GB"},
        "model": {"id": "llava-test", "revision": "c" * 40},
        "resolved_config": {
            "cache": {
                "block_size": 4,
                "physical_budget_block_size": 4,
                "physical_budget_mandatory_policy": "last_token",
            },
            "selection": {
                "stable_head_hierarchical_maximum_candidate_evaluations": 24,
                "stable_head_hierarchical_maximum_sequential_rounds": 6,
                "stable_head_hierarchical_maximum_kl": 0.02,
                "stable_head_hierarchical_maximum_logit_difference": 1.0,
                "stable_head_hierarchical_horizon": 2,
            },
        },
    }
    (run_root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )

    report = runner.validate_existing_debug_artifacts(
        tmp_path,
        runtime_git_sha=runtime_sha,
        validation_git_sha=validation_sha,
        source_slurm_job_id="123",
    )

    assert report["status"] == "passed"
    assert report["runtime_git_sha"] == runtime_sha
    assert report["validation_git_sha"] == validation_sha
    assert report["source_artifacts_reused_without_gpu_rerun"] is True
    assert report["probe_gate_summary"]["used_fullkv_fallback"] is False
    assert report["artifact_hashes_verified"] is True
