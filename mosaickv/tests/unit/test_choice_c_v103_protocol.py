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
            package / "configs/experiments/choice_c_v103_fresh_identity_calibration_protocol.yaml"
        ).read_text(encoding="utf-8")
    )


def _validator(package: Path) -> ModuleType:
    path = package / "scripts/validate_choice_c_v103_protocol.py"
    spec = importlib.util.spec_from_file_location("mosaickv_test_v103_protocol", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_v103_scan_scope_seal_disables_every_operational_action() -> None:
    package = Path(__file__).parents[2]
    protocol = _protocol(package)

    assert protocol["protocol_status"] == (
        "scan_scope_seal_authored_pending_commit_all_execution_disabled"
    )
    authorization = protocol["authorization"]
    assert authorization["protocol_commit_created"] is True
    assert authorization["protocol_reviewed_after_commit"] is True
    assert all(
        value is False
        for key, value in authorization.items()
        if key not in {"protocol_commit_created", "protocol_reviewed_after_commit"}
    )
    assert all(
        value is False for key, value in protocol["submission_policy"].items() if key != "reason"
    )
    freshness = protocol["freshness_definition"]
    assert freshness["all_v102_ledger_identities_mandatorily_excluded"] is True
    assert freshness["sample_level_freshness_claimed_before_scan_review"] is False
    assert freshness["any_ambiguous_relevant_artifact_blocks_materialization"] is True
    assert protocol["outputs"]["no_output_created_by_this_draft"] is True
    scan_scope = protocol["scan_scope_seal"]
    assert scan_scope["protocol_commit_git_sha"] == ("75736bbc115d2ad5d82ae7c7a3a26df41c89d8fe")
    assert scan_scope["exclusion_scan_execution_authorized"] is False
    assert scan_scope["seal_commit_git_sha"] is None
    assert scan_scope["seal_reviewed_after_commit"] is False


def test_v103_uses_separate_preflight_and_calibration_identity_sets() -> None:
    package = Path(__file__).parents[2]
    protocol = _protocol(package)
    materialization = protocol["ledger_materialization"]
    disjointness = protocol["new_disjointness_review"]

    assert materialization["preflight_ids_count_per_task"] == 2
    assert materialization["calibration_ids_count_per_task"] == 32
    assert materialization["total_planned_unique_identities"] == 68
    assert materialization["preflight_and_calibration_ledgers_are_separate"] is True
    assert materialization["calibration_ledgers_materialized_before_gpu_preflight"] is False
    assert materialization["calibration_materialization_requires_sealed_positive_preflight"] is True
    assert materialization["failed_preflight_must_leave_calibration_ledgers_absent"] is True
    assert disjointness["require_zero_overlap_between_preflight_and_calibration"] is True
    assert disjointness["require_zero_overlap_with_new_exclusion_manifest"] is True
    assert disjointness["require_v102_64_identity_exclusion_coverage"] is True
    assert disjointness["preflight_report"]["selected_identity_count"] == 4
    assert disjointness["preflight_report"]["required_before_gpu_preflight"] is True
    assert disjointness["calibration_report"]["selected_identity_count_including_preflight"] == 68
    assert disjointness["calibration_report"]["required_before_full_calibration"] is True


def test_v103_preserves_trace_reference_and_unchanged_scientific_gates() -> None:
    package = Path(__file__).parents[2]
    protocol = _protocol(package)
    tasks = {task["task"]: task for task in protocol["tasks"]}
    preflight = protocol["gpu_preflight_gate"]
    gates = protocol["unchanged_memory_and_fidelity_gates"]
    comparison = protocol["controlled_comparison_contract"]
    provenance = protocol["required_run_provenance"]

    assert tasks["textvqa_train_calibration"]["reference_provenance_field"] == "answers"
    assert tasks["textvqa_train_calibration"]["forbidden_literal_reference"] == "answer"
    assert preflight["fullkv_fallback_passes_preflight"] is False
    assert preflight["over_budget_final_cache_pack_fails_preflight"] is True
    assert {
        "used_fullkv_fallback",
        "attempted_active_kv_bytes",
        "probe_decisions",
        "timing_breakdown",
    }.issubset(preflight["every_trace_must_record"])
    assert gates["minimum_active_reduction_fraction_vs_prompt_topk"] == 0.30
    assert gates["minimum_active_plus_residual_reduction_fraction_vs_prompt_topk"] == 0.30
    assert gates["fullkv_answer_noninferiority_margin_vs_prompt_topk"] == 0.05
    assert gates["fullkv_token_noninferiority_margin_vs_prompt_topk"] == 0.05
    assert gates["require_every_task"] is True
    assert gates["require_every_fold"] is True
    assert comparison["active_kv_bytes_are_the_primary_budget"] is True
    assert comparison["failed_samples_must_be_preserved_as_error_rows"] is True
    assert {"git_sha", "config_sha", "dataset_revision", "worktree_dirty"}.issubset(
        provenance["every_row_or_immutable_manifest_must_record"]
    )
    assert provenance["clean_worktree_required"] is True


def test_v103_validator_accepts_draft_and_rejects_weakened_copies(tmp_path: Path) -> None:
    package = Path(__file__).parents[2]
    validator = _validator(package)
    canonical = (
        package / "configs/experiments/choice_c_v103_fresh_identity_calibration_protocol.yaml"
    )
    result = validator.validate(canonical)

    assert result["status"] == ("valid_uncommitted_v103_scan_scope_seal_all_execution_disabled")
    assert result["all_actions_authorized"] is False
    assert result["protocol_commit_reviewed"] is True
    assert result["scan_scope_seal_commit_created"] is False
    assert result["exclusion_scan_authorized"] is False
    assert result["preflight_samples"] == 4
    assert result["calibration_samples"] == 64
    assert result["planned_unique_identities"] == 68
    assert result["minimum_v102_exclusions"] == 64

    authorized = copy.deepcopy(_protocol(package))
    authorized["authorization"]["exclusion_scan_authorized"] = True
    authorized_path = tmp_path / "authorized.yaml"
    authorized_path.write_text(yaml.safe_dump(authorized), encoding="utf-8")
    with pytest.raises(RuntimeError, match="operational authorization"):
        validator.validate(authorized_path)

    widened = copy.deepcopy(_protocol(package))
    seal_path = package / widened["scan_scope_seal"]["path"]
    seal = yaml.safe_load(seal_path.read_text(encoding="utf-8"))
    seal["permitted_future_scan_scope"]["allowed_fields"].append("answer")
    widened_seal = tmp_path / "widened_seal.yaml"
    widened_seal.write_text(yaml.safe_dump(seal), encoding="utf-8")
    widened["scan_scope_seal"]["path"] = str(widened_seal)
    widened_path = tmp_path / "widened.yaml"
    widened_path.write_text(yaml.safe_dump(widened), encoding="utf-8")
    with pytest.raises(RuntimeError, match="scan-scope seal"):
        validator.validate(widened_path)

    weakened = copy.deepcopy(_protocol(package))
    weakened["unchanged_memory_and_fidelity_gates"][
        "minimum_active_reduction_fraction_vs_prompt_topk"
    ] = 0.29
    weakened_path = tmp_path / "weakened.yaml"
    weakened_path.write_text(yaml.safe_dump(weakened), encoding="utf-8")
    with pytest.raises(RuntimeError, match="weakens"):
        validator.validate(weakened_path)
