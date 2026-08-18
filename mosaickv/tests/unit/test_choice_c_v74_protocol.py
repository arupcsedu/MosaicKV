from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from mosaickv.config import load_config
from mosaickv.types import MosaicKVMethod


def _load_script() -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / "choice_c_v74_protocol.py"
    spec = importlib.util.spec_from_file_location("mosaickv_test_v74_protocol", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PROTOCOL = _load_script()
COMMON_CACHE_POLICY = {
    "block_size": 4,
    "physical_budget_block_size": 4,
    "physical_budget_mandatory_policy": "last_token",
}


def test_v74_common_block_policy_is_applied_without_changing_the_byte_budget() -> None:
    root = Path(__file__).parents[2]
    base = load_config(root / "configs/experiments/choice_c_v72_frozen_candidate.yaml")

    revised = PROTOCOL.apply_common_cache_policy(base, COMMON_CACHE_POLICY)

    assert revised.cache.block_size == 4
    assert revised.cache.physical_budget_block_size == 4
    assert revised.cache.physical_budget_mandatory_policy == "last_token"
    assert revised.cache.budget_value == base.cache.budget_value
    assert revised.cache.retention_ratio == base.cache.retention_ratio


def test_v74_common_block_policy_rejects_mismatched_accounting_granularity() -> None:
    root = Path(__file__).parents[2]
    base = load_config(root / "configs/experiments/choice_c_v72_frozen_candidate.yaml")

    with pytest.raises(ValueError, match="must equal block_size"):
        PROTOCOL.apply_common_cache_policy(
            base,
            {
                **COMMON_CACHE_POLICY,
                "physical_budget_block_size": 8,
            },
        )


def test_v74_calibration_uses_one_block_policy_for_controls_and_candidates() -> None:
    root = Path(__file__).parents[2]
    spec = yaml.safe_load(
        (root / "configs/experiments/choice_c_v74_label_blind_calibration.yaml").read_text(
            encoding="utf-8"
        )
    )
    base = load_config(root / spec["candidate_config"])

    configs = PROTOCOL.calibration_configs(
        base,
        spec["candidates"],
        spec["common_cache_policy"],
    )

    assert {config.cache.block_size for config in configs.values()} == {4}
    assert {config.cache.physical_budget_block_size for config in configs.values()} == {4}
    assert {config.cache.physical_budget_mandatory_policy for config in configs.values()} == {
        "last_token"
    }
    candidates = {
        arm: config
        for arm, config in configs.items()
        if arm not in {"full_kv", "prompt_attention_topk"}
    }
    assert {config.cache.budget_value for config in candidates.values()} == {27_500_000}
    assert {config.prototypes.budget_planning_mode for config in candidates.values()} == {
        "one_pass_bounded_anchor_cover"
    }
    assert {config.prototypes.maximum_prototypes_per_head for config in candidates.values()} == {
        1,
        2,
    }


def test_v74_candidate_applies_declared_evidence_and_byte_budget_controls() -> None:
    root = Path(__file__).parents[2]
    base = load_config(root / "configs/experiments/choice_c_v72_frozen_candidate.yaml")
    candidate = {
        "id": "spatial",
        "tier_policy": "exact_dominant_minimal_prototype",
        "overrides": {
            "spatial_weight": 1.0,
            "spatial_max_distance": 0.25,
            "candidate_nonmandatory_fraction": 0.08,
            "routed_head_fraction": 0.03125,
            "prompt_anchor_fraction": 0.25,
            "evidence_preservation_fraction": 0.35,
            "minimum_stratum_fraction": 0.20,
            "physical_capacity_fraction": 1.0,
            "maximum_prototypes_per_head": 2,
            "prototype_budget_planning_mode": "one_pass_bounded_anchor_cover",
            "active_budget_bytes": 28_000_000,
        },
    }

    revised = PROTOCOL.apply_candidate(base, candidate)

    assert revised.cache.budget_value == 28_000_000
    assert revised.cache.total_storage_budget_bytes == 28_000_000
    assert revised.cache.retention_ratio == base.cache.retention_ratio
    assert revised.model == base.model
    assert revised.execution == base.execution
    assert revised.graph.spatial_weight == 1.0
    assert revised.selection.evidence_preservation_fraction == 0.35
    assert revised.selection.minimum_stratum_fraction == 0.20
    assert revised.selection.physical_capacity_fraction == 1.0
    assert revised.prototypes.maximum_prototypes_per_head == 2
    assert revised.prototypes.budget_planning_mode == "one_pass_bounded_anchor_cover"
    assert revised.graph.uncertain_head_fraction == revised.prototypes.uncertain_head_fraction
    assert revised.repair.enabled is False
    assert revised.residual.enabled is False
    assert revised.residual.max_bytes == 0
    assert revised.residual.max_source_blocks == 0


def test_versioned_candidate_can_enable_label_independent_spatial_strata() -> None:
    root = Path(__file__).parents[2]
    base = load_config(root / "configs/experiments/choice_c_v72_frozen_candidate.yaml")
    spec = yaml.safe_load(
        (root / "configs/experiments/choice_c_v76_disjoint_ocr_calibration.yaml").read_text(
            encoding="utf-8"
        )
    )
    candidate = next(item for item in spec["candidates"] if item["id"].startswith("ocr_grid6"))

    revised = PROTOCOL.apply_candidate(base, candidate)

    assert revised.cache.budget_value == 27_500_000
    assert revised.selection.evidence_reservation_mode == "spatial_stratified_prompt"
    assert revised.selection.evidence_grid_rows == 6
    assert revised.selection.evidence_grid_columns == 6
    assert revised.prototypes.maximum_prototypes_per_head == 2
    assert revised.residual.enabled is False


def test_versioned_candidate_can_rebalance_stable_head_visual_capacity() -> None:
    root = Path(__file__).parents[2]
    base = load_config(root / "configs/experiments/choice_c_v72_frozen_candidate.yaml")
    spec = yaml.safe_load(
        (
            root / "configs/experiments/choice_c_v78_disjoint_visual_balance_calibration.yaml"
        ).read_text(encoding="utf-8")
    )
    candidate = next(
        item for item in spec["candidates"] if item["id"].startswith("stable_visual50")
    )

    revised = PROTOCOL.apply_candidate(base, candidate)

    assert revised.cache.budget_value == 27_500_000
    assert revised.selection.stable_head_prompt_routing is True
    assert revised.selection.stable_head_visual_fraction == 0.5
    assert revised.prototypes.maximum_prototypes_per_head == 2
    assert revised.residual.enabled is False


def test_versioned_candidate_can_enable_adaptive_stable_head_swaps() -> None:
    root = Path(__file__).parents[2]
    base = load_config(root / "configs/experiments/choice_c_v72_frozen_candidate.yaml")
    spec = yaml.safe_load(
        (
            root / "configs/experiments/choice_c_v79_disjoint_adaptive_swap_calibration.yaml"
        ).read_text(encoding="utf-8")
    )
    candidate = next(
        item for item in spec["candidates"] if item["id"].startswith("adaptive_swap2_text9975")
    )

    revised = PROTOCOL.apply_candidate(base, candidate)

    assert revised.cache.budget_value == 27_500_000
    assert revised.selection.stable_head_visual_fraction == 0.0
    assert revised.selection.stable_head_adaptive_visual_swaps is True
    assert revised.selection.stable_head_minimum_text_mass_fraction == 0.9975
    assert revised.selection.stable_head_visual_gain_ratio == 1.5
    assert revised.selection.stable_head_maximum_visual_swaps == 2


def test_versioned_candidate_can_build_same_byte_exact_only_future_rerank() -> None:
    root = Path(__file__).parents[2]
    base = load_config(root / "configs/experiments/choice_c_v72_frozen_candidate.yaml")
    candidate = {
        "id": "exact_future_fixture",
        "tier_policy": "exact_only_replacement",
        "overrides": {
            "spatial_weight": 1.0,
            "spatial_max_distance": 0.125,
            "candidate_nonmandatory_fraction": 0.04,
            "routed_head_fraction": 0.03125,
            "prompt_anchor_fraction": 0.5,
            "evidence_preservation_fraction": 0.35,
            "minimum_stratum_fraction": 0.25,
            "physical_capacity_fraction": 1.0,
            "maximum_prototypes_per_head": 0,
            "prototype_budget_planning_mode": "disabled_exact_only",
            "active_budget_bytes": 27_500_000,
            "stable_head_future_query_reranking": True,
            "stable_head_future_minimum_prompt_mass_fraction": 0.995,
            "stable_head_future_gain_ratio": 1.5,
            "stable_head_maximum_future_replacements": 2,
            "stable_head_future_divergence_fail_closed": True,
            "stable_head_future_minimum_prompt_forecast_tv": 0.01,
            "stable_head_future_risk_gate": True,
            "stable_head_future_maximum_normalized_prompt_loss": 0.005,
            "stable_head_future_minimum_normalized_gain": 0.0005,
            "forecast_mode": "hybrid",
            "forecast_prompt_window": 16,
            "forecast_draft_steps": 2,
        },
    }

    revised = PROTOCOL.apply_candidate(base, candidate)

    assert revised.method is MosaicKVMethod.MOSAICKV_EXACT
    assert revised.prototypes.enabled is False
    assert revised.prototypes.maximum_prototypes_per_head is None
    assert revised.residual.enabled is False
    assert revised.repair.enabled is False
    assert revised.selection.stable_head_future_query_reranking is True
    assert revised.selection.stable_head_maximum_future_replacements == 2
    assert revised.selection.stable_head_future_divergence_fail_closed
    assert revised.selection.stable_head_future_minimum_prompt_forecast_tv == 0.01
    assert revised.selection.stable_head_future_risk_gate
    assert revised.selection.stable_head_future_maximum_normalized_prompt_loss == 0.005
    assert revised.selection.stable_head_future_minimum_normalized_gain == 0.0005
    assert revised.forecasting.mode.value == "hybrid"
    assert revised.forecasting.draft_steps == 2
    assert revised.residual.enabled is False


def test_versioned_candidate_can_build_prompt_query_coverage_reranker() -> None:
    root = Path(__file__).parents[2]
    base = load_config(root / "configs/experiments/choice_c_v72_frozen_candidate.yaml")
    spec = yaml.safe_load(
        (
            root / "configs/experiments/choice_c_v87_prompt_query_coverage_development.yaml"
        ).read_text(encoding="utf-8")
    )
    candidate = spec["candidates"][0]

    revised = PROTOCOL.apply_candidate(base, candidate)

    assert revised.method is MosaicKVMethod.MOSAICKV_EXACT
    assert revised.selection.stable_head_query_coverage_reranking
    assert revised.selection.stable_head_query_coverage_minimum_prompt_mass_fraction == 0.9975
    assert revised.selection.stable_head_query_coverage_maximum_replacements == 1
    assert revised.selection.stable_head_query_coverage_minimum_normalized_gain == 0.001
    assert not revised.selection.stable_head_future_query_reranking
    assert not revised.selection.stable_head_adaptive_visual_swaps
    assert revised.forecasting.mode.value == "prompt_window"
    assert revised.forecasting.prompt_window == 16
    assert revised.forecasting.draft_steps == 0


def test_versioned_candidate_rejects_string_adaptive_swap_flag() -> None:
    root = Path(__file__).parents[2]
    base = load_config(root / "configs/experiments/choice_c_v72_frozen_candidate.yaml")
    spec = yaml.safe_load(
        (
            root / "configs/experiments/choice_c_v79_disjoint_adaptive_swap_calibration.yaml"
        ).read_text(encoding="utf-8")
    )
    candidate = next(
        item for item in spec["candidates"] if item["id"].startswith("adaptive_swap2_text9975")
    )
    candidate["overrides"]["stable_head_adaptive_visual_swaps"] = "false"

    with pytest.raises(
        ValueError,
        match="stable_head_adaptive_visual_swaps must be a boolean",
    ):
        PROTOCOL.apply_candidate(base, candidate)


def test_versioned_candidate_applies_future_query_swap_risk_gate() -> None:
    root = Path(__file__).parents[2]
    base = load_config(root / "configs/experiments/choice_c_v72_frozen_candidate.yaml")
    spec = yaml.safe_load(
        (
            root / "configs/experiments/choice_c_v80_disjoint_risk_gated_swap_calibration.yaml"
        ).read_text(encoding="utf-8")
    )
    candidate = next(
        item for item in spec["candidates"] if item["id"].startswith("forecast_risk_strict")
    )

    revised = PROTOCOL.apply_candidate(base, candidate)

    assert revised.selection.stable_head_swap_risk_gate is True
    assert revised.selection.stable_head_maximum_normalized_text_loss == 0.0005
    assert revised.selection.stable_head_forecast_gain_ratio == 2.0
    assert revised.selection.stable_head_maximum_visual_swaps == 1


def test_versioned_candidate_rejects_string_swap_risk_flag() -> None:
    root = Path(__file__).parents[2]
    base = load_config(root / "configs/experiments/choice_c_v72_frozen_candidate.yaml")
    spec = yaml.safe_load(
        (
            root / "configs/experiments/choice_c_v80_disjoint_risk_gated_swap_calibration.yaml"
        ).read_text(encoding="utf-8")
    )
    candidate = next(
        item for item in spec["candidates"] if item["id"].startswith("forecast_risk_strict")
    )
    candidate["overrides"]["stable_head_swap_risk_gate"] = "true"

    with pytest.raises(ValueError, match="stable_head_swap_risk_gate must be a boolean"):
        PROTOCOL.apply_candidate(base, candidate)


def test_v74_candidate_rejects_prototype_heavy_or_double_capped_variants() -> None:
    root = Path(__file__).parents[2]
    base = load_config(root / "configs/experiments/choice_c_v72_frozen_candidate.yaml")
    candidate = {
        "id": "invalid",
        "tier_policy": "exact_dominant_minimal_prototype",
        "overrides": {
            "spatial_weight": 1.0,
            "spatial_max_distance": 0.125,
            "candidate_nonmandatory_fraction": 0.04,
            "routed_head_fraction": 0.125,
            "prompt_anchor_fraction": 0.25,
            "evidence_preservation_fraction": 0.35,
            "minimum_stratum_fraction": 0.20,
            "physical_capacity_fraction": 0.70,
            "maximum_prototypes_per_head": 4,
            "prototype_budget_planning_mode": "one_pass_bounded_anchor_cover",
            "active_budget_bytes": 28_000_000,
        },
    }

    with pytest.raises(ValueError, match="at most 1/32"):
        PROTOCOL.apply_candidate(base, candidate)

    candidate["overrides"].update(
        {
            "routed_head_fraction": 0.03125,
            "maximum_prototypes_per_head": 2,
        }
    )
    with pytest.raises(ValueError, match=r"must equal 1\.0"):
        PROTOCOL.apply_candidate(base, candidate)


def test_v74_token_agreement_penalizes_length_mismatch() -> None:
    assert PROTOCOL.token_agreement([1, 2], [1, 2]) == 1.0
    assert PROTOCOL.token_agreement([1, 2], [1, 3]) == 0.5
    assert PROTOCOL.token_agreement([1], [1, 2]) == 0.5
    assert PROTOCOL.token_agreement([], []) == 1.0
