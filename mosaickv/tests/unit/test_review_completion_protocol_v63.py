"""Pure source-config checks; no dataset, model, scanner or scheduler access."""

import hashlib
import json
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[3]
PROTOCOL = REPOSITORY / "mosaickv/configs/experiments/evicastkv_fox/review_completion_v63.json"


@pytest.fixture
def protocol():
    return json.loads(PROTOCOL.read_text())


def test_protocol_is_disabled_planning_not_a_result_or_execution_config(protocol):
    assert protocol["schema"] == "evicastkv_disabled_coco_review_completion_v63"
    assert protocol["classification"] == "prospective_planning_only_not_measured_results"
    assert protocol["status"] == "NO_CONFIG_FROZEN"
    assert protocol["enabled"] is False
    assert protocol["execution_enabled"] is False
    assert protocol["execution"]["status"] == "DISABLED"
    assert protocol["execution"]["auto_start"] is False
    for field in ("runner", "launch_command", "scheduler_script"):
        assert protocol["execution"][field] is None


@pytest.mark.parametrize(
    "field",
    [
        "exclusion_scan_authorized",
        "materialization_authorized",
        "fresh_confirmation_authorized",
        "GPU_execution_authorized",
        "experiment_submission_authorized",
        "paper_admission_authorized",
        "metadata_scan_authorized",
        "media_scan_authorized",
        "dataset_access_authorized",
        "model_access_authorized",
        "inference_authorized",
        "paper_claim_authorized",
    ],
)
def test_every_data_execution_and_admission_authorization_is_literal_false(protocol, field):
    assert protocol["authorization"][field] is False


def test_authorization_map_contains_no_implicit_permission(protocol):
    assert protocol["authorization"]
    assert all(value is False for value in protocol["authorization"].values())


def test_unknown_candidate_and_cohort_cannot_execute(protocol):
    candidate = protocol["candidate"]
    assert candidate["status"] == "NO_CONFIG_FROZEN"
    assert candidate["qualified"] is False
    for field in (
        "selected_config",
        "selected_config_path",
        "config_path",
        "config_sha256",
        "resolved_policy_sha256",
        "routing_policy",
        "routing_fraction",
        "coefficients",
        "fixed_candidate_quota_contract_sha256",
    ):
        assert candidate[field] is None
    assert candidate["reference_config_is_candidate_selection"] is False
    assert candidate["confirmation_outcomes_may_select_or_repair_candidate"] is False
    assert "NO_CONFIG_FROZEN" in protocol["execution"]["blockers"]
    cohort = protocol["cohort"]
    assert cohort["freshness_certified"] is False
    assert cohort["adequate_power_established"] is False
    assert cohort["power_analysis_status"] == "UNRESOLVED"
    for field in ("candidate_sample_ids", "sample_ledger_path", "sample_ledger_sha256"):
        assert cohort[field] is None


@pytest.mark.parametrize(
    "method", ["full_kv", "prompt_attention_topk", "vl_cache_reimpl", "mosaickv_exact"]
)
def test_reference_values_come_from_complete_hash_pinned_actual_source_JSON(protocol, method):
    reference = protocol["source_references"]["configs"][method]
    raw = (REPOSITORY / reference["path"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == reference["file_sha256"]
    actual = json.loads(raw)
    recovered = protocol["recovered_reference_controls"]
    assert actual["method"] == method
    for section in ("model", "dataset", "execution", "generation"):
        assert actual[section] == recovered[section]
    assert actual["model"]["id"] == "llava-hf/llava-v1.6-mistral-7b-hf"
    assert actual["model"]["revision"] == "2424fdd47412fccc66d91719126b420e9fbd7065"
    assert actual["dataset"]["revision"] == "c3213b54e173ae86517d481de17f31439d6f8c72"
    assert actual["dataset"]["split"] == "val"
    assert recovered["dataset_config_name"] is None
    assert "config" not in actual["dataset"]
    assert actual["cache"]["accounting_spec_sha"] == recovered["cache_accounting_spec_sha256"]
    assert actual["cache"]["block_size"] == recovered["block_size"]
    assert actual["cache"]["physical_budget_block_size"] == recovered["physical_budget_block_size"]
    assert (
        actual["cache"]["physical_budget_mandatory_policy"]
        == recovered["mandatory_position_policy"]
    )
    assert actual["cache"]["exact_gather_mode"] == recovered["exact_gather_mode"]
    assert actual["cache"]["packing_mode"] == recovered["packing_mode"]
    assert actual["forecasting"]["draft_steps"] == 0
    assert actual["prototypes"]["enabled"] is False
    assert actual["residual"]["enabled"] is False
    assert actual["repair"]["enabled"] is False


def test_COCO_only_main_request_count_is_proposed_1792_not_an_actual_matrix(protocol):
    matrix = protocol["matrix"]
    assert matrix["status"] == "PROPOSED_NOT_EXECUTED"
    assert matrix["proposed_caps_MiB"] == [20, 40]
    assert matrix["proposed_caps_bytes"] == [cap * 2**20 for cap in [20, 40]]
    assert matrix["cap_compatibility_qualified"] is False
    assert matrix["compressed_methods"] == [
        "prompt_attention_topk",
        "vl_cache_reimpl",
        "mosaickv_exact",
    ]
    per_input = 1 + len(matrix["compressed_methods"]) * len(matrix["proposed_caps_MiB"])
    assert matrix["compressed_requests_per_input"] == 6
    assert matrix["total_requests_per_input"] == per_input == 7
    assert protocol["cohort"]["proposed_unique_COCO_inputs"] == 256
    assert matrix["planned_main_requests"] == 256 * per_input == 1792
    assert matrix["request_count_is_prospective_not_actual"] is True
    assert matrix["actual_requests_executed"] == 0


def test_FullKV_is_quality_reference_and_never_a_compressed_latency_competitor(protocol):
    reference = protocol["matrix"]["quality_reference"]
    assert reference["method"] == "full_kv"
    assert reference["requests_per_input"] == 1
    assert reference["matched_byte_competitor"] is False
    assert reference["compressed_latency_ranking_member"] is False
    assert protocol["timing"]["FullKV_is_compressed_latency_comparator"] is False


def test_controlled_mechanism_requests_have_a_512_additional_upper_bound(protocol):
    mechanism = protocol["controlled_mechanism"]
    assert mechanism["policies"] == ["complete", "routed_prompt_control"]
    assert mechanism["proposed_cap_MiB"] == 20
    assert mechanism["proposed_unique_inputs"] == 256
    maximum = mechanism["proposed_unique_inputs"] * len(mechanism["policies"])
    assert mechanism["planned_additional_requests_maximum"] == maximum == 512
    assert mechanism["planned_core_requests_maximum"] == 1792 + maximum == 2304
    assert mechanism["actual_requests_executed"] == 0
    assert (
        mechanism[
            "complete_row_reuse_requires_identical_protocol_hash_and_actual_candidate_quota_contract"
        ]
        is True
    )
    assert mechanism["controlled_study_timing_is_native_deployment_timing"] is False


def test_timing_is_complete_request_counterbalanced_input_level_not_repeat_quality_N(protocol):
    timing = protocol["timing"]
    assert (
        timing["warmups"],
        timing["input_count"],
        timing["repeats_per_primary_method_budget"],
    ) == (
        5,
        32,
        3,
    )
    assert timing["input_manifest_path"] is None
    assert timing["input_manifest_sha256"] is None
    assert timing["counterbalanced_method_order_required"] is True
    assert timing["complete_requests_recomputed_unprofiled"] is True
    assert timing["shared_prefill_or_cached_graph_replay_is_headline_timing"] is False
    assert timing["summarize_repeats_per_input_before_bootstrap"] is True
    assert timing["repeats_increase_unique_quality_N"] is False


def test_development_route_and_scoring_controls_are_disabled_not_a_candidate_selection(protocol):
    development = protocol["development_mechanism_plan"]
    assert development["status"] == "DISABLED_PROSPECTIVE_DEVELOPMENT_ONLY"
    assert development["proposed_unique_inputs"] == 64
    assert development["proposed_caps_MiB"] == [20, 40]
    assert development["proposed_route_fractions"] == [0, 1 / 256, 1 / 16, 1 / 4]
    assert development["route_count_rounding_rule"] is None
    assert development["selected_routing_fraction"] is None
    assert development["full_route_pilot_inputs_maximum"] == 8
    assert development["fixed_candidate_fixed_capacity_scoring_variants"] == [
        "complete",
        "routed_prompt_control",
        "coverage_off",
        "value_novelty_off",
    ]
    assert development["coverage_off_is_not_no_graph"] is True
    assert development["entropy_routing_status"] == "NOT_RUN_OPTIONAL_NOT_SELECTED"
    assert development["conditional_random_routing_control_seeds"] == [13, 29, 47]
    assert development["best_seed_selection_or_seed_pooling_as_independent_examples"] is False
    assert development["development_requests_are_not_confirmation_requests"] is True
    assert development["planned_total_requests"] is None
    assert development["actual_requests_executed"] == 0


def test_registered_six_contrast_statistics_and_strict_latency_threshold_are_unchanged(protocol):
    stats = protocol["statistics"]
    assert stats["bootstrap_replicates"] == 10000
    assert stats["bootstrap_seed"] == 20271014
    assert stats["registered_primary_quality_family_size"] == 6
    assert (
        stats["COCO_primary_contrasts"]
        + stats["MileBench_registered_but_NOT_RUN_development_only_contrasts"]
        == 6
    )
    assert stats["Bonferroni_interval_coverage_expression"] == "1 - 0.05 / 6"
    assert stats["registered_six_contrast_family_unchanged"] is True
    assert stats["adjusted_latency_ratio_upper_bound_maximum"] == 1.0
    assert stats["latency_ratio_1_05_is_acceptance"] is False
    assert stats["zero_crossing_CI_is_equivalence_or_noninferiority"] is False
    assert stats["corpus_CIDEr_recomputed_each_draw_with_unique_resample_IDs"] is True
    assert stats["mean_shard_CIDEr_is_corpus_CIDEr"] is False
    assert stats["confirmation_outcome_dependent_family_change_or_optional_stopping"] is False


def test_MileBench_remains_NOT_RUN_development_only(protocol):
    milebench = protocol["matrix"]["MileBench"]
    assert milebench["status"] == "NOT_RUN_DEVELOPMENT_ONLY"
    assert milebench["confirmation_requests_planned"] == 0
    assert milebench["source_group_certification_required_before_any_future_fresh_claim"] is True
    assert milebench["fresh_confirmation_authorized"] is False


def test_scoring_and_boundary_diagnostics_do_not_claim_baseline_or_quality_validation(protocol):
    scoring = protocol["scoring_and_boundaries"]
    assert scoring["metric"] == "pinned_lmms_eval_PTBTokenizer_COCO_corpus_CIDEr_x100"
    assert scoring["validation_inputs_proposed"] == 32
    assert scoring["baseline_fidelity_inputs_or_saved_attention_fixtures_proposed"] == 8
    assert scoring["official_baseline_reproduction_claimed"] is False
    assert scoring["boundary_B_status"] == "DIAGNOSTIC_ONLY_BLOCKED_UNTIL_ADAPTER_VALID"
    assert scoring["boundary_B_is_changed_evaluation_protocol"] is True
    assert scoring["boundary_B_is_information_leak_fix"] is False
    assert scoring["boundary_B_changes_headline_protocol"] is False
    assert scoring["compressed_correctness_reference"] == (
        "equivalent_masked_same_selected_original_source_KV_not_unrestricted_FullKV"
    )


def test_exact_selection_and_storage_variant_do_not_claim_six_mechanism_equivalence(protocol):
    variant = protocol["variant_contract"]
    assert variant["generated_history"] == "append_only"
    assert variant["full_six_mechanism_benefit_claim"] is False
    assert variant["draft_steps"] == 0
    for field in (
        "draft_generation_enabled",
        "prototypes_enabled",
        "residual_repair_enabled",
        "new_backend_enabled",
        "new_batching_scheduler_enabled",
    ):
        assert variant[field] is False
    storage = variant["storage_variant"]
    assert storage["config_path"] is None
    assert storage["FP16_GQA_compatibility_established"] is False
    assert storage["proposed_20_40_MiB_cap_compatibility_established"] is False
    assert storage["can_replace_main_exact_selection_candidate"] is False
    assert storage["counts_as_six_mechanism_evidence"] is False


def test_actual_byte_pairing_and_step_schedule_cannot_be_replaced_by_mean_caps(protocol):
    memory = protocol["memory_and_pairing"]
    assert memory["initial_final_and_peak_active_unique_backing_bytes_required"] is True
    assert memory["padding_preallocation_metadata_and_transient_owned_storage_charged"] is True
    assert memory["per_input_initial_physical_byte_equality_required"] is True
    assert memory["aggregate_mean_byte_equality_is_not_pairing"] is True
    assert memory["final_peak_equality_requires_genuinely_matched_forward_step_schedule"] is True
    assert memory["otherwise_report_actual_growth_and_step_aligned_storage_separately"] is True
    assert (memory["maximum_predictions"], memory["continuation_appends"]) == (64, 63)


def test_proposed_resource_ceiling_never_grants_GPU_or_scheduler_authority(protocol):
    resources = protocol["proposed_resources"]
    assert resources["maximum_aggregate_A100s"] == 6
    assert resources["GPUs_per_shard"] == 1
    assert resources["CPUs_per_shard"] == 12
    assert resources["shard_time_limit"] == "02:00:00"
    assert resources["maximum_RAM_GiB"] == 32
    assert resources["maximum_analysis_workers"] == 8
    assert resources["resource_ceiling_is_execution_permission"] is False
    assert resources["GPU_execution_authorized"] is False
