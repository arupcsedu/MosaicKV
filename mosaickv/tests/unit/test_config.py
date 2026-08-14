from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from mosaickv.config import (
    AblationConfig,
    CacheConfig,
    ConfigurationError,
    FidelityDiagnosticsConfig,
    ForecastingConfig,
    GraphConfig,
    LookMConfig,
    PrefixKVConfig,
    PrototypeConfig,
    ResidualConfig,
    RunConfig,
    SelectionConfig,
    UtilityConfig,
    VLCacheConfig,
    canonical_config_json,
    config_sha256,
    load_config,
    synthetic_smoke_config,
)
from mosaickv.types import (
    AblationVariant,
    BudgetUnit,
    ForecastMode,
    LookMMergeStrategy,
    MosaicKVMethod,
    PrefixKVProfileMode,
    RepairPolicy,
    ResidualStorageDType,
)


def test_smoke_config_is_stable_and_valid() -> None:
    config = synthetic_smoke_config(seed=7)
    assert config.execution.seed == 7
    assert len(config_sha256(config)) == 64
    assert canonical_config_json(config) == canonical_config_json(config)
    assert "ablation" not in json.loads(canonical_config_json(config))


def test_oracle_ablation_requires_explicit_evaluation_only_gate() -> None:
    with pytest.raises(ConfigurationError, match="evaluation_only=true"):
        AblationConfig(variant=AblationVariant.ORACLE_FUTURE_QUERY)
    config = AblationConfig(
        variant=AblationVariant.ORACLE_FUTURE_QUERY,
        evaluation_only=True,
    )
    assert config.evaluation_only


def test_fidelity_diagnostic_ledger_is_strict_and_hashed() -> None:
    config = FidelityDiagnosticsConfig(
        enabled=True,
        sample_ids=("task:1", "task:2"),
        sample_ledger_sha="a" * 64,
        forecast_oracle_steps=8,
    )
    assert config.enabled
    assert config.forecast_oracle_steps == 8
    with pytest.raises(ConfigurationError, match="cannot contain duplicates"):
        FidelityDiagnosticsConfig(
            enabled=True,
            sample_ids=("task:1", "task:1"),
            sample_ledger_sha="a" * 64,
        )
    with pytest.raises(ConfigurationError, match="sample_ledger_sha"):
        FidelityDiagnosticsConfig(enabled=True, sample_ids=("task:1",))
    with pytest.raises(ConfigurationError, match="forecast_oracle_steps"):
        FidelityDiagnosticsConfig(forecast_oracle_steps=1)


def test_selection_evidence_can_be_captured_without_oracle_diagnostics() -> None:
    config = FidelityDiagnosticsConfig.from_mapping({"capture_selection_evidence": True})
    assert not config.enabled
    assert config.capture_selection_evidence


def test_per_head_prototype_routes_are_strict_and_resolved() -> None:
    config = PrototypeConfig(
        group_size=15,
        max_position_span=255,
        exact_head_routes=("0:1",),
        head_group_sizes=("0:2:11",),
        head_position_spans=("0:2:127",),
        adaptive_grouping=True,
    )
    assert config.requires_exact_head(0, 1)
    assert config.group_size_for(0, 2) == 11
    assert config.group_size_for(1, 2) == 15
    assert config.position_span_for(0, 2) == 127
    assert config.uses_prototype_head(0, 2)
    routed = PrototypeConfig(
        selection_only_head_routes=("0:2",),
        maximum_prototypes_per_head=2,
        uncertain_head_only=True,
        uncertain_head_fraction=0.25,
        source_modalities=("image",),
    )
    assert not routed.uses_prototype_head(0, 2)
    assert routed.maximum_prototypes_per_head == 2
    assert routed.uses_prototype_block(0, 1, "image")
    assert not routed.uses_prototype_block(0, 1, "text")
    with pytest.raises(ConfigurationError, match="cannot repeat"):
        PrototypeConfig(head_group_sizes=("0:1:4", "0:1:8"))
    with pytest.raises(ConfigurationError, match="must be disjoint"):
        PrototypeConfig(exact_head_routes=("0:1",), selection_only_head_routes=("0:1",))
    with pytest.raises(ConfigurationError, match="maximum_prototypes_per_head"):
        PrototypeConfig(maximum_prototypes_per_head=0)
    with pytest.raises(ConfigurationError, match="source_modalities"):
        PrototypeConfig(source_modalities=("audio",))


@pytest.mark.parametrize("scale", [-0.1, 1.1, float("inf")])
def test_prototype_attention_log_bias_scale_is_strict(scale: float) -> None:
    with pytest.raises(ConfigurationError, match="attention_log_bias_scale"):
        PrototypeConfig(attention_log_bias_scale=scale)

    resolved = PrototypeConfig.from_mapping({"attention_log_bias_scale": 0.5})
    assert resolved.attention_log_bias_scale == 0.5


def test_forecast_attention_prototype_controls_are_strictly_coupled() -> None:
    resolved = PrototypeConfig.from_mapping(
        {
            "pooling_mode": "forecast_attention_token",
            "attention_bias_basis": "forecast_attention_entropy",
            "anchor_position_strategy": "forecast_weighted_member_medoid",
        }
    )
    assert resolved.pooling_mode == "forecast_attention_token"
    assert resolved.attention_bias_basis == "forecast_attention_entropy"
    assert resolved.anchor_position_strategy == "forecast_weighted_member_medoid"
    with pytest.raises(ConfigurationError, match="requires the forecast-attention entropy"):
        PrototypeConfig(pooling_mode="forecast_attention_token")
    with pytest.raises(ConfigurationError, match="requires forecast-attention token pooling"):
        PrototypeConfig(attention_bias_basis="forecast_attention_entropy")
    with pytest.raises(ConfigurationError, match="anchor_position_strategy"):
        PrototypeConfig(anchor_position_strategy="free_rope_phase")
    with pytest.raises(ConfigurationError, match="source_gather_mode"):
        PrototypeConfig(source_gather_mode="per_block")

    gather_config = PrototypeConfig.from_mapping({"source_gather_mode": "per_head_reference"})
    assert gather_config.source_gather_mode == "per_head_reference"
    assert PrototypeConfig().budget_planning_mode == "one_pass_anchor_cover"
    with pytest.raises(ConfigurationError, match="budget_planning_mode"):
        PrototypeConfig(budget_planning_mode="unbounded_retry")


@pytest.mark.parametrize("ratio", [0.0, -0.1, 1.1])
def test_invalid_retention_ratio_has_useful_path(ratio: float) -> None:
    with pytest.raises(ConfigurationError, match=r"cache\.retention_ratio"):
        CacheConfig(
            budget_value=8,
            budget_unit=BudgetUnit.RETAINED_SLOTS,
            retention_ratio=ratio,
        )


def test_exact_tier_gather_mode_is_strict_and_resolved() -> None:
    resolved = CacheConfig.from_mapping(
        {
            "budget_value": 8,
            "budget_unit": "retained_slots",
            "retention_ratio": 0.5,
            "exact_gather_mode": "per_head_reference",
        }
    )
    assert resolved.exact_gather_mode == "per_head_reference"
    assert (
        CacheConfig.from_mapping(
            {
                "budget_value": 8,
                "budget_unit": "retained_slots",
                "retention_ratio": 0.5,
                "exact_gather_mode": "source_views",
            }
        ).exact_gather_mode
        == "source_views"
    )
    with pytest.raises(ConfigurationError, match="exact_gather_mode"):
        CacheConfig(
            budget_value=8,
            budget_unit=BudgetUnit.RETAINED_SLOTS,
            retention_ratio=0.5,
            exact_gather_mode="per_block",
        )


def test_runtime_packing_mode_is_strict_and_resolved() -> None:
    assert (
        CacheConfig(
            budget_value=8,
            budget_unit=BudgetUnit.RETAINED_SLOTS,
            retention_ratio=0.5,
        ).packing_mode
        == "per_head_reference"
    )
    resolved = CacheConfig.from_mapping(
        {
            "budget_value": 8,
            "budget_unit": "retained_slots",
            "retention_ratio": 0.5,
            "packing_mode": "per_head_reference",
        }
    )
    assert resolved.packing_mode == "per_head_reference"
    assert (
        CacheConfig.from_mapping(
            {
                "budget_value": 8,
                "budget_unit": "retained_slots",
                "retention_ratio": 0.5,
                "packing_mode": "preallocated_layer",
            }
        ).packing_mode
        == "preallocated_layer"
    )
    with pytest.raises(ConfigurationError, match="packing_mode"):
        CacheConfig(
            budget_value=8,
            budget_unit=BudgetUnit.RETAINED_SLOTS,
            retention_ratio=0.5,
            packing_mode="per_block",
        )


def test_total_and_residual_storage_budgets_are_strict() -> None:
    cache = CacheConfig.from_mapping(
        {
            "budget_value": 8,
            "budget_unit": "bytes",
            "retention_ratio": 0.5,
            "total_storage_budget_bytes": 12,
        }
    )
    assert cache.total_storage_budget_bytes == 12
    with pytest.raises(ConfigurationError, match="total_storage_budget_bytes"):
        CacheConfig(
            budget_value=8,
            budget_unit=BudgetUnit.BYTES,
            retention_ratio=0.5,
            total_storage_budget_bytes=0,
        )

    residual = ResidualConfig.from_mapping(
        {
            "max_bytes": 4,
            "max_source_blocks": 1,
            "selection_policy": "risk_weighted_reconstruction_error",
        }
    )
    assert residual.max_bytes == 4
    assert residual.max_source_blocks == 1
    assert (
        ResidualConfig(selection_policy="prototype_coverage_then_risk").selection_policy
        == "prototype_coverage_then_risk"
    )
    with pytest.raises(ConfigurationError, match=r"residual\.max_bytes"):
        ResidualConfig(max_bytes=-1)
    with pytest.raises(ConfigurationError, match=r"residual\.selection_policy"):
        ResidualConfig(selection_policy="highest_node_id")


def test_draft_kl_repair_requires_an_online_draft_forecast() -> None:
    config = synthetic_smoke_config()
    with pytest.raises(ConfigurationError, match="draft_steps >= 1"):
        replace(
            config,
            method=MosaicKVMethod.MOSAICKV_FULL,
            forecasting=ForecastingConfig(
                mode=ForecastMode.PROMPT_WINDOW,
                prompt_window=4,
                draft_steps=0,
            ),
            repair=replace(
                config.repair,
                policy=RepairPolicy.DECISION_AWARE,
                draft_kl_threshold=0.1,
            ),
        )


def test_draft_kl_repair_accepts_independent_counterfactual_draft() -> None:
    config = synthetic_smoke_config()
    resolved = replace(
        config,
        method=MosaicKVMethod.MOSAICKV_FULL,
        forecasting=ForecastingConfig(
            mode=ForecastMode.PROMPT_WINDOW,
            prompt_window=4,
            draft_steps=0,
        ),
        repair=replace(
            config.repair,
            policy=RepairPolicy.DECISION_AWARE,
            draft_kl_threshold=0.1,
            counterfactual_draft_steps=4,
            draft_require_token_mismatch=True,
            counterfactual_route_heads=True,
        ),
    )
    assert resolved.repair.counterfactual_draft_steps == 4
    assert resolved.repair.draft_require_token_mismatch
    assert resolved.repair.counterfactual_route_heads


def test_stable_head_prompt_routing_is_strictly_serialized() -> None:
    selection = SelectionConfig.from_mapping(
        {
            "stable_head_prompt_routing": True,
            "physical_capacity_fraction": 0.7,
        }
    )
    assert selection.stable_head_prompt_routing
    assert selection.physical_capacity_fraction == 0.7


@pytest.mark.parametrize("value", (0.0, -0.1, 1.01, float("inf"), float("nan")))
def test_physical_capacity_fraction_is_strict(value: float) -> None:
    with pytest.raises(ConfigurationError, match="physical_capacity_fraction"):
        SelectionConfig(physical_capacity_fraction=value)


def test_routed_head_only_planning_requires_stable_head_routing() -> None:
    with pytest.raises(ConfigurationError, match="requires stable_head_prompt_routing"):
        SelectionConfig.from_mapping({"routed_head_only_planning": True})
    selection = SelectionConfig.from_mapping(
        {
            "stable_head_prompt_routing": True,
            "routed_head_only_planning": True,
        }
    )
    assert selection.routed_head_only_planning


def test_unknown_configuration_field_is_rejected() -> None:
    payload = json.loads(canonical_config_json(synthetic_smoke_config()))
    payload["model"]["typo_revision"] = "bad"
    with pytest.raises(ConfigurationError, match=r"model contains unknown field.*typo_revision"):
        RunConfig.from_mapping(payload)


def test_graph_attention_host_transfer_mode_is_strict() -> None:
    assert GraphConfig().attention_host_transfer_mode == "deferred_layer"
    assert (
        GraphConfig.from_mapping(
            {"attention_host_transfer_mode": "per_group_reference"}
        ).attention_host_transfer_mode
        == "per_group_reference"
    )
    with pytest.raises(ConfigurationError, match="attention_host_transfer_mode"):
        GraphConfig(attention_host_transfer_mode="unknown")
    shortlist = GraphConfig.from_mapping(
        {
            "similarity_shortlist_multiplier": 4.0,
            "similarity_shortlist_min_nodes": 32,
            "candidate_descriptor_pooling": True,
        }
    )
    assert shortlist.similarity_shortlist_multiplier == 4.0
    assert shortlist.similarity_shortlist_min_nodes == 32
    assert shortlist.candidate_descriptor_pooling is True
    with pytest.raises(ConfigurationError, match="similarity_shortlist_multiplier"):
        GraphConfig(similarity_shortlist_multiplier=-1)
    with pytest.raises(ConfigurationError, match="candidate_descriptor_pooling"):
        GraphConfig(candidate_descriptor_pooling=True)
    routed = GraphConfig.from_mapping(
        {
            "similarity_shortlist_multiplier": 3.0,
            "candidate_descriptor_pooling": True,
            "candidate_graph_mode": "shortlist_rerank",
            "candidate_neighbor_hops": 1,
            "screening_group_blocks": 4,
            "uncertain_head_only": True,
            "uncertain_head_fraction": 0.5,
            "uncertain_head_threshold": 0.25,
        }
    )
    assert routed.candidate_graph_mode == "shortlist_rerank"
    assert routed.screening_group_blocks == 4
    assert routed.uncertain_head_only is True
    with pytest.raises(ConfigurationError, match="shortlist_rerank"):
        GraphConfig(candidate_graph_mode="shortlist_rerank")
    with pytest.raises(ConfigurationError, match="uncertain_head_only"):
        GraphConfig(uncertain_head_only=True)


def test_compute_execution_modes_are_strict() -> None:
    assert ForecastingConfig().centroid_execution_mode == "batched_layer"
    assert UtilityConfig().execution_mode == "vectorized_edges"
    assert (
        ForecastingConfig.from_mapping(
            {"centroid_execution_mode": "reference_per_head"}
        ).centroid_execution_mode
        == "reference_per_head"
    )
    assert (
        UtilityConfig.from_mapping({"execution_mode": "reference_sparse"}).execution_mode
        == "reference_sparse"
    )
    with pytest.raises(ConfigurationError, match="centroid_execution_mode"):
        ForecastingConfig(centroid_execution_mode="per_query")
    with pytest.raises(ConfigurationError, match="execution_mode"):
        UtilityConfig(execution_mode="scalar_dense")


def test_vectorized_selector_and_gpu_rescue_configuration_is_strict() -> None:
    selection = SelectionConfig.from_mapping(
        {"algorithm": "vectorized_coverage", "coverage_correction_rounds": 3}
    )
    assert selection.algorithm == "vectorized_coverage"
    assert selection.coverage_correction_rounds == 3
    with pytest.raises(ConfigurationError, match=r"selection\.algorithm"):
        SelectionConfig(algorithm="beam_search")
    with pytest.raises(ConfigurationError, match="coverage_correction_rounds"):
        SelectionConfig(coverage_correction_rounds=9)

    residual = ResidualConfig.from_mapping(
        {"gpu_rescue_max_blocks": 2, "gpu_rescue_max_bytes": 4096}
    )
    assert residual.gpu_rescue_max_blocks == 2
    assert residual.gpu_rescue_max_bytes == 4096
    with pytest.raises(ConfigurationError, match="gpu_rescue_max_bytes"):
        ResidualConfig(gpu_rescue_max_bytes=0)
    medoid = PrototypeConfig(
        pooling_mode="exact_medoid_key_forecast_value",
        attention_bias_basis="forecast_attention_entropy",
        anchor_position_strategy="forecast_weighted_member_medoid",
    )
    assert medoid.pooling_mode == "exact_medoid_key_forecast_value"
    with pytest.raises(ConfigurationError, match="anchor_position_strategy"):
        PrototypeConfig(
            pooling_mode="exact_medoid_key_forecast_value",
            attention_bias_basis="forecast_attention_entropy",
        )


def test_common_physical_budget_controls_are_strict() -> None:
    resolved = CacheConfig.from_mapping(
        {
            "budget_value": 1024,
            "budget_unit": "bytes",
            "retention_ratio": 0.2,
            "block_size": 1,
            "physical_budget_block_size": 16,
            "physical_budget_mandatory_policy": "last_token",
        }
    )
    assert resolved.physical_budget_block_size == 16
    assert resolved.physical_budget_mandatory_policy == "last_token"
    with pytest.raises(ConfigurationError, match="physical_budget_block_size"):
        replace(resolved, physical_budget_block_size=0)
    with pytest.raises(ConfigurationError, match="physical_budget_mandatory_policy"):
        replace(resolved, physical_budget_mandatory_policy="baseline_native")


def test_uncertainty_exact_routing_requires_adaptive_forecasting() -> None:
    with pytest.raises(ConfigurationError, match="requires adaptive_head_routing"):
        ForecastingConfig(uncertainty_exact_head_fraction=0.02)
    resolved = ForecastingConfig(
        adaptive_head_routing=True,
        uncertainty_exact_head_fraction=0.02,
    )
    assert resolved.uncertainty_exact_head_fraction == 0.02


def test_invalid_enum_lists_allowed_values() -> None:
    payload = json.loads(canonical_config_json(synthetic_smoke_config()))
    payload["execution"]["backend"] = "unknown-backend"
    with pytest.raises(ConfigurationError, match=r"execution\.backend must be one of"):
        RunConfig.from_mapping(payload)


def test_load_toml_example() -> None:
    project_root = Path(__file__).resolve().parents[2]
    config = load_config(project_root / "configs" / "smoke.toml")
    assert config.cache.retention_ratio == 1.0
    assert config.model.id == "synthetic/smoke"
    assert config.forecasting.mode is ForecastMode.HYBRID
    assert config.forecasting.prompt_window == 16


def test_load_hf_yaml_example() -> None:
    project_root = Path(__file__).resolve().parents[2]
    config = load_config(project_root / "configs" / "hf_mosaickv.yaml")
    assert config.method is MosaicKVMethod.MOSAICKV_FULL
    assert config.model.id == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert len(config.model.revision) == 40
    assert config.execution.attention_implementation == "eager"
    assert config.execution.decode_attention_implementation is None


def test_execution_accepts_separate_decode_attention_backend() -> None:
    payload = json.loads(canonical_config_json(synthetic_smoke_config()))
    payload["execution"]["decode_attention_implementation"] = "sdpa"

    config = RunConfig.from_mapping(payload)

    assert config.execution.decode_attention_implementation == "sdpa"


def test_load_simple_baseline_yaml_example() -> None:
    project_root = Path(__file__).resolve().parents[2]
    config = load_config(project_root / "configs" / "hf_simple_baseline.yaml")
    assert config.method is MosaicKVMethod.RANDOM_KV
    assert config.cache.retention_ratio == 0.5
    assert not config.forecasting.enabled
    assert not config.graph.enabled
    assert not config.selection.enabled
    assert not config.prototypes.enabled
    assert not config.residual.enabled
    assert not config.repair.enabled


def test_residual_index_mode_is_strict() -> None:
    assert ResidualConfig().index_mode == "compact"
    assert ResidualConfig(index_mode="materialized").index_mode == "materialized"
    with pytest.raises(ConfigurationError, match=r"residual\.index_mode"):
        ResidualConfig(index_mode="unknown")


def test_graph_edge_accumulation_mode_is_strict() -> None:
    assert GraphConfig().edge_accumulation_mode == "preselected"
    assert GraphConfig(edge_accumulation_mode="dictionary").edge_accumulation_mode == ("dictionary")
    with pytest.raises(ConfigurationError, match=r"graph\.edge_accumulation_mode"):
        GraphConfig(edge_accumulation_mode="unknown")


def test_load_prefixkv_yaml_example() -> None:
    project_root = Path(__file__).resolve().parents[2]
    config = load_config(project_root / "configs" / "hf_prefixkv_reimpl.yaml")
    assert config.method is MosaicKVMethod.PREFIXKV_REIMPL
    assert config.cache.block_size == 1
    assert config.prefixkv.enabled
    assert config.prefixkv.profile_mode is PrefixKVProfileMode.FIXED_GLOBAL


def test_load_vl_cache_yaml_example() -> None:
    project_root = Path(__file__).resolve().parents[2]
    config = load_config(project_root / "configs" / "hf_vl_cache_reimpl.yaml")
    assert config.method is MosaicKVMethod.VL_CACHE_REIMPL
    assert config.cache.block_size == 1
    assert config.vl_cache.enabled
    assert config.vl_cache.sparsity_threshold == pytest.approx(0.01)


def test_vl_cache_calibration_provenance_is_strict() -> None:
    with pytest.raises(ConfigurationError, match="require dataset ID, revision, and split"):
        VLCacheConfig(enabled=True, calibration_sample_ids=("cal-0",))
    with pytest.raises(ConfigurationError, match="cannot contain duplicates"):
        VLCacheConfig(
            enabled=True,
            calibration_dataset_id="dataset",
            calibration_dataset_revision="revision",
            calibration_split="train",
            calibration_sample_ids=("cal-0", "cal-0"),
        )


def test_simple_baseline_rejects_mosaickv_stages() -> None:
    project_root = Path(__file__).resolve().parents[2]
    config = load_config(project_root / "configs" / "hf_simple_baseline.yaml")
    payload = json.loads(canonical_config_json(config))
    payload["prototypes"]["enabled"] = True
    with pytest.raises(
        ConfigurationError,
        match=r"method='random_kv' requires disabled MosaicKV stages: prototypes",
    ):
        RunConfig.from_mapping(payload)


def test_full_kv_requires_retention_one() -> None:
    project_root = Path(__file__).resolve().parents[2]
    config = load_config(project_root / "configs" / "hf_simple_baseline.yaml")
    payload = json.loads(canonical_config_json(config))
    payload["method"] = "full_kv"
    with pytest.raises(ConfigurationError, match=r"full_kv.*retention_ratio=1\.0"):
        RunConfig.from_mapping(payload)


def test_lookm_configuration_is_strict_and_explicit() -> None:
    config = LookMConfig(
        enabled=True,
        recent_ratio=0.1,
        important_ratio=0.1,
        merge_strategy=LookMMergeStrategy.PIVOTAL,
    )
    assert config.recent_ratio + config.important_ratio == pytest.approx(0.2)
    with pytest.raises(ConfigurationError, match="cannot exceed 1"):
        LookMConfig(enabled=True, recent_ratio=0.6, important_ratio=0.5)


def test_prefixkv_offline_profile_is_explicit_and_fixed_global_needs_no_file() -> None:
    with pytest.raises(ConfigurationError, match=r"prefixkv\.profile_path is required"):
        PrefixKVConfig(enabled=True)
    config = PrefixKVConfig(
        enabled=True,
        profile_mode=PrefixKVProfileMode.FIXED_GLOBAL,
    )
    assert config.profile_path is None


def test_forecasting_configuration_rejects_ambiguous_unused_sources() -> None:
    with pytest.raises(ConfigurationError, match="prompt_window mode"):
        ForecastingConfig(
            mode=ForecastMode.PROMPT_WINDOW,
            prompt_window=4,
            draft_steps=1,
        )


def test_forecasting_configuration_parses_conditional_draft_gate_strictly() -> None:
    config = ForecastingConfig.from_mapping(
        {
            "mode": "hybrid",
            "prompt_window": 8,
            "draft_steps": 2,
            "conditional_draft_enabled": True,
            "conditional_draft_uncertainty_threshold": 0.2,
        }
    )
    assert config.conditional_draft_enabled
    assert config.conditional_draft_uncertainty_threshold == 0.2
    with pytest.raises(ConfigurationError, match="must be in \\[0, 1\\]"):
        ForecastingConfig(conditional_draft_uncertainty_threshold=1.01)


def test_prototype_configuration_rejects_invalid_modality_pairs() -> None:
    with pytest.raises(ConfigurationError, match=r"prototypes\.allowed_modality_pairs"):
        PrototypeConfig(allowed_modality_pairs=("image:audio",))


def test_residual_configuration_parses_storage_dtype_strictly() -> None:
    config = ResidualConfig.from_mapping(
        {
            "enabled": True,
            "rank": 8,
            "storage_dtype": "int8",
            "require_pinned_memory": False,
        }
    )
    assert config.storage_dtype is ResidualStorageDType.INT8
    with pytest.raises(ConfigurationError, match=r"residual\.storage_dtype must be one of"):
        ResidualConfig.from_mapping({"storage_dtype": "int4"})


def test_unsupported_config_format_fails(tmp_path: Path) -> None:
    path = tmp_path / "config.ini"
    path.write_text("schema_version: 1\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match=r"use \.json, \.toml, \.yaml, or \.yml"):
        load_config(path)
