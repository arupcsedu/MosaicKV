"""Versioned, label-blind v74 candidate construction helpers."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

from mosaickv.types import BudgetUnit, ForecastMode, MosaicKVMethod, RepairPolicy

REQUIRED_OVERRIDE_KEYS = frozenset(
    {
        "spatial_weight",
        "spatial_max_distance",
        "candidate_nonmandatory_fraction",
        "routed_head_fraction",
        "prompt_anchor_fraction",
        "evidence_preservation_fraction",
        "minimum_stratum_fraction",
        "physical_capacity_fraction",
        "maximum_prototypes_per_head",
        "prototype_budget_planning_mode",
        "active_budget_bytes",
    }
)
OPTIONAL_OVERRIDE_KEYS = frozenset(
    {
        "evidence_reservation_mode",
        "evidence_grid_rows",
        "evidence_grid_columns",
        "stable_head_visual_fraction",
        "stable_head_adaptive_visual_swaps",
        "stable_head_minimum_text_mass_fraction",
        "stable_head_visual_gain_ratio",
        "stable_head_maximum_visual_swaps",
        "stable_head_swap_risk_gate",
        "stable_head_maximum_normalized_text_loss",
        "stable_head_forecast_gain_ratio",
        "stable_head_future_query_reranking",
        "stable_head_future_minimum_prompt_mass_fraction",
        "stable_head_future_gain_ratio",
        "stable_head_maximum_future_replacements",
        "stable_head_future_divergence_fail_closed",
        "stable_head_future_minimum_prompt_forecast_tv",
        "stable_head_future_risk_gate",
        "stable_head_future_maximum_normalized_prompt_loss",
        "stable_head_future_minimum_normalized_gain",
        "stable_head_query_coverage_reranking",
        "stable_head_query_coverage_minimum_prompt_mass_fraction",
        "stable_head_query_coverage_maximum_replacements",
        "stable_head_query_coverage_minimum_normalized_gain",
        "stable_head_query_coverage_initial_selection",
        "stable_head_query_coverage_initial_fraction",
        "stable_head_query_coverage_initial_prompt_weight",
        "stable_head_query_coverage_initial_minimum_prompt_mass_fraction",
        "forecast_mode",
        "forecast_prompt_window",
        "forecast_draft_steps",
    }
)
OVERRIDE_KEYS = REQUIRED_OVERRIDE_KEYS | OPTIONAL_OVERRIDE_KEYS


def apply_common_cache_policy(config: Any, policy: dict[str, Any]) -> Any:
    """Apply one shared block/accounting policy to every comparison arm."""

    expected = {
        "block_size",
        "physical_budget_block_size",
        "physical_budget_mandatory_policy",
    }
    if not isinstance(policy, dict) or set(policy) != expected:
        raise ValueError(f"common_cache_policy must define exactly {sorted(expected)}")
    block_size = policy["block_size"]
    physical_block_size = policy["physical_budget_block_size"]
    mandatory_policy = policy["physical_budget_mandatory_policy"]
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
        raise ValueError("common block_size must be a positive integer")
    if physical_block_size != block_size:
        raise ValueError(
            "physical_budget_block_size must equal block_size for the common comparison"
        )
    if mandatory_policy != "last_token":
        raise ValueError("v74.6 common mandatory policy must be 'last_token'")
    return replace(
        config,
        cache=replace(
            config.cache,
            block_size=block_size,
            physical_budget_block_size=physical_block_size,
            physical_budget_mandatory_policy=mandatory_policy,
        ),
    )


def apply_candidate(base_mosaic: Any, candidate: dict[str, Any]) -> Any:
    """Apply one completely declared, byte-capped candidate."""

    candidate_id = candidate.get("id")
    overrides = cast("dict[str, Any]", candidate.get("overrides"))
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("candidate id must be a non-empty string")
    tier_policy = candidate.get("tier_policy")
    if tier_policy not in {
        "exact_dominant_minimal_prototype",
        "exact_only_replacement",
    }:
        raise ValueError(f"candidate {candidate_id!r} has unsupported tier_policy {tier_policy!r}")
    if not isinstance(overrides, dict):
        raise ValueError(f"candidate {candidate_id!r} overrides must be a mapping")
    override_keys = set(overrides)
    missing = REQUIRED_OVERRIDE_KEYS - override_keys
    unknown = override_keys - OVERRIDE_KEYS
    if missing or unknown:
        raise ValueError(
            f"candidate {candidate_id!r} has missing overrides {sorted(missing)} "
            f"and unknown overrides {sorted(unknown)}"
        )
    if base_mosaic.cache.budget_unit is not BudgetUnit.BYTES:
        raise ValueError("v74 byte-budget calibration requires cache.budget_unit='bytes'")
    active_budget_bytes = overrides["active_budget_bytes"]
    if (
        isinstance(active_budget_bytes, bool)
        or not isinstance(active_budget_bytes, int)
        or active_budget_bytes < 1
    ):
        raise ValueError("active_budget_bytes must be a positive integer")
    repair_disabled = (
        not base_mosaic.repair.enabled or base_mosaic.repair.policy is RepairPolicy.NONE
    )
    routed_fraction = float(overrides["routed_head_fraction"])
    maximum_prototypes = int(overrides["maximum_prototypes_per_head"])
    prototype_budget_planning_mode = str(overrides["prototype_budget_planning_mode"])
    physical_capacity_fraction = float(overrides["physical_capacity_fraction"])
    adaptive_visual_swaps = overrides.get(
        "stable_head_adaptive_visual_swaps",
        base_mosaic.selection.stable_head_adaptive_visual_swaps,
    )
    if not isinstance(adaptive_visual_swaps, bool):
        raise ValueError("stable_head_adaptive_visual_swaps must be a boolean")
    swap_risk_gate = overrides.get(
        "stable_head_swap_risk_gate",
        base_mosaic.selection.stable_head_swap_risk_gate,
    )
    if not isinstance(swap_risk_gate, bool):
        raise ValueError("stable_head_swap_risk_gate must be a boolean")
    future_query_reranking = overrides.get(
        "stable_head_future_query_reranking",
        base_mosaic.selection.stable_head_future_query_reranking,
    )
    if not isinstance(future_query_reranking, bool):
        raise ValueError("stable_head_future_query_reranking must be a boolean")
    divergence_fail_closed = overrides.get(
        "stable_head_future_divergence_fail_closed",
        base_mosaic.selection.stable_head_future_divergence_fail_closed,
    )
    if not isinstance(divergence_fail_closed, bool):
        raise ValueError("stable_head_future_divergence_fail_closed must be a boolean")
    future_risk_gate = overrides.get(
        "stable_head_future_risk_gate",
        base_mosaic.selection.stable_head_future_risk_gate,
    )
    if not isinstance(future_risk_gate, bool):
        raise ValueError("stable_head_future_risk_gate must be a boolean")
    query_coverage_reranking = overrides.get(
        "stable_head_query_coverage_reranking",
        base_mosaic.selection.stable_head_query_coverage_reranking,
    )
    if not isinstance(query_coverage_reranking, bool):
        raise ValueError("stable_head_query_coverage_reranking must be a boolean")
    query_coverage_initial_selection = overrides.get(
        "stable_head_query_coverage_initial_selection",
        base_mosaic.selection.stable_head_query_coverage_initial_selection,
    )
    if not isinstance(query_coverage_initial_selection, bool):
        raise ValueError("stable_head_query_coverage_initial_selection must be a boolean")
    forecast_mode_raw = overrides.get("forecast_mode", base_mosaic.forecasting.mode.value)
    try:
        forecast_mode = ForecastMode(str(forecast_mode_raw))
    except ValueError as error:
        raise ValueError(f"unsupported forecast_mode {forecast_mode_raw!r}") from error
    forecast_prompt_window = overrides.get(
        "forecast_prompt_window",
        base_mosaic.forecasting.prompt_window,
    )
    forecast_draft_steps = overrides.get(
        "forecast_draft_steps",
        base_mosaic.forecasting.draft_steps,
    )
    for name, value in (
        ("forecast_prompt_window", forecast_prompt_window),
        ("forecast_draft_steps", forecast_draft_steps),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if divergence_fail_closed and forecast_mode is ForecastMode.PROMPT_WINDOW:
        raise ValueError("fail-closed future reranking requires draft or hybrid forecast evidence")
    if not 0 < routed_fraction <= 0.03125:
        raise ValueError(
            "exact-dominant candidates may route at most 1/32 of layer/head identities"
        )
    if tier_policy == "exact_dominant_minimal_prototype":
        if not 1 <= maximum_prototypes <= 2:
            raise ValueError(
                "exact-dominant candidates require one or two prototypes per routed head"
            )
        if prototype_budget_planning_mode != "one_pass_bounded_anchor_cover":
            raise ValueError(
                "prototype candidates require prototype_budget_planning_mode="
                "'one_pass_bounded_anchor_cover'"
            )
        if future_query_reranking:
            raise ValueError("v81 future-query reranking is isolated to exact-only candidates")
    else:
        if maximum_prototypes != 0:
            raise ValueError("exact-only candidates must declare maximum_prototypes_per_head=0")
        if prototype_budget_planning_mode != "disabled_exact_only":
            raise ValueError(
                "exact-only candidates require prototype_budget_planning_mode='disabled_exact_only'"
            )
    if physical_capacity_fraction != 1.0:
        raise ValueError(
            "v74.6 uses the byte cap as the sole physical budget; "
            "physical_capacity_fraction must equal 1.0"
        )
    return replace(
        base_mosaic,
        method=(
            MosaicKVMethod.MOSAICKV_EXACT
            if tier_policy == "exact_only_replacement"
            else MosaicKVMethod.MOSAICKV_FULL
        ),
        cache=replace(
            base_mosaic.cache,
            budget_value=active_budget_bytes,
            total_storage_budget_bytes=active_budget_bytes,
        ),
        forecasting=replace(
            base_mosaic.forecasting,
            enabled=True,
            mode=forecast_mode,
            prompt_window=int(forecast_prompt_window),
            draft_steps=int(forecast_draft_steps),
        ),
        graph=replace(
            base_mosaic.graph,
            spatial_weight=float(overrides["spatial_weight"]),
            spatial_max_distance=float(overrides["spatial_max_distance"]),
            candidate_nonmandatory_fraction=float(overrides["candidate_nonmandatory_fraction"]),
            uncertain_head_fraction=routed_fraction,
        ),
        selection=replace(
            base_mosaic.selection,
            prompt_anchor_fraction=float(overrides["prompt_anchor_fraction"]),
            evidence_preservation_fraction=float(overrides["evidence_preservation_fraction"]),
            minimum_stratum_fraction=float(overrides["minimum_stratum_fraction"]),
            physical_capacity_fraction=physical_capacity_fraction,
            evidence_reservation_mode=str(
                overrides.get(
                    "evidence_reservation_mode",
                    base_mosaic.selection.evidence_reservation_mode,
                )
            ),
            evidence_grid_rows=int(
                overrides.get(
                    "evidence_grid_rows",
                    base_mosaic.selection.evidence_grid_rows,
                )
            ),
            evidence_grid_columns=int(
                overrides.get(
                    "evidence_grid_columns",
                    base_mosaic.selection.evidence_grid_columns,
                )
            ),
            stable_head_visual_fraction=float(
                overrides.get(
                    "stable_head_visual_fraction",
                    base_mosaic.selection.stable_head_visual_fraction,
                )
            ),
            stable_head_adaptive_visual_swaps=adaptive_visual_swaps,
            stable_head_minimum_text_mass_fraction=float(
                overrides.get(
                    "stable_head_minimum_text_mass_fraction",
                    base_mosaic.selection.stable_head_minimum_text_mass_fraction,
                )
            ),
            stable_head_visual_gain_ratio=float(
                overrides.get(
                    "stable_head_visual_gain_ratio",
                    base_mosaic.selection.stable_head_visual_gain_ratio,
                )
            ),
            stable_head_maximum_visual_swaps=int(
                overrides.get(
                    "stable_head_maximum_visual_swaps",
                    base_mosaic.selection.stable_head_maximum_visual_swaps,
                )
            ),
            stable_head_swap_risk_gate=swap_risk_gate,
            stable_head_maximum_normalized_text_loss=float(
                overrides.get(
                    "stable_head_maximum_normalized_text_loss",
                    base_mosaic.selection.stable_head_maximum_normalized_text_loss,
                )
            ),
            stable_head_forecast_gain_ratio=float(
                overrides.get(
                    "stable_head_forecast_gain_ratio",
                    base_mosaic.selection.stable_head_forecast_gain_ratio,
                )
            ),
            stable_head_future_query_reranking=future_query_reranking,
            stable_head_future_minimum_prompt_mass_fraction=float(
                overrides.get(
                    "stable_head_future_minimum_prompt_mass_fraction",
                    base_mosaic.selection.stable_head_future_minimum_prompt_mass_fraction,
                )
            ),
            stable_head_future_gain_ratio=float(
                overrides.get(
                    "stable_head_future_gain_ratio",
                    base_mosaic.selection.stable_head_future_gain_ratio,
                )
            ),
            stable_head_maximum_future_replacements=int(
                overrides.get(
                    "stable_head_maximum_future_replacements",
                    base_mosaic.selection.stable_head_maximum_future_replacements,
                )
            ),
            stable_head_future_divergence_fail_closed=divergence_fail_closed,
            stable_head_future_minimum_prompt_forecast_tv=float(
                overrides.get(
                    "stable_head_future_minimum_prompt_forecast_tv",
                    base_mosaic.selection.stable_head_future_minimum_prompt_forecast_tv,
                )
            ),
            stable_head_future_risk_gate=future_risk_gate,
            stable_head_future_maximum_normalized_prompt_loss=float(
                overrides.get(
                    "stable_head_future_maximum_normalized_prompt_loss",
                    base_mosaic.selection.stable_head_future_maximum_normalized_prompt_loss,
                )
            ),
            stable_head_future_minimum_normalized_gain=float(
                overrides.get(
                    "stable_head_future_minimum_normalized_gain",
                    base_mosaic.selection.stable_head_future_minimum_normalized_gain,
                )
            ),
            stable_head_query_coverage_reranking=query_coverage_reranking,
            stable_head_query_coverage_minimum_prompt_mass_fraction=float(
                overrides.get(
                    "stable_head_query_coverage_minimum_prompt_mass_fraction",
                    base_mosaic.selection.stable_head_query_coverage_minimum_prompt_mass_fraction,
                )
            ),
            stable_head_query_coverage_maximum_replacements=int(
                overrides.get(
                    "stable_head_query_coverage_maximum_replacements",
                    base_mosaic.selection.stable_head_query_coverage_maximum_replacements,
                )
            ),
            stable_head_query_coverage_minimum_normalized_gain=float(
                overrides.get(
                    "stable_head_query_coverage_minimum_normalized_gain",
                    base_mosaic.selection.stable_head_query_coverage_minimum_normalized_gain,
                )
            ),
            stable_head_query_coverage_initial_selection=(query_coverage_initial_selection),
            stable_head_query_coverage_initial_fraction=float(
                overrides.get(
                    "stable_head_query_coverage_initial_fraction",
                    base_mosaic.selection.stable_head_query_coverage_initial_fraction,
                )
            ),
            stable_head_query_coverage_initial_prompt_weight=float(
                overrides.get(
                    "stable_head_query_coverage_initial_prompt_weight",
                    base_mosaic.selection.stable_head_query_coverage_initial_prompt_weight,
                )
            ),
            stable_head_query_coverage_initial_minimum_prompt_mass_fraction=float(
                overrides.get(
                    "stable_head_query_coverage_initial_minimum_prompt_mass_fraction",
                    base_mosaic.selection.stable_head_query_coverage_initial_minimum_prompt_mass_fraction,
                )
            ),
        ),
        prototypes=replace(
            base_mosaic.prototypes,
            enabled=tier_policy == "exact_dominant_minimal_prototype",
            uncertain_head_fraction=routed_fraction,
            maximum_prototypes_per_head=(
                maximum_prototypes if tier_policy == "exact_dominant_minimal_prototype" else None
            ),
            budget_planning_mode=(
                prototype_budget_planning_mode
                if tier_policy == "exact_dominant_minimal_prototype"
                else "one_pass_anchor_cover"
            ),
        ),
        residual=(
            replace(
                base_mosaic.residual,
                enabled=False,
                max_bytes=0,
                max_source_blocks=0,
                gpu_rescue_max_blocks=0,
                gpu_rescue_max_bytes=None,
            )
            if repair_disabled
            else base_mosaic.residual
        ),
        repair=(
            replace(base_mosaic.repair, enabled=False) if repair_disabled else base_mosaic.repair
        ),
    )


def calibration_configs(
    base: Any,
    candidates: list[dict[str, Any]],
    common_cache_policy: dict[str, Any],
) -> dict[str, Any]:
    """Return common references plus all preregistered MosaicKV candidates."""

    from run_choice_c_v73_clean_quality_memory_worker import _configs as v73_configs

    controls = {
        arm: apply_common_cache_policy(config, common_cache_policy)
        for arm, config in v73_configs(base).items()
    }
    mosaic = controls[MosaicKVMethod.MOSAICKV_FULL.value]
    result = {
        MosaicKVMethod.FULL_KV.value: controls[MosaicKVMethod.FULL_KV.value],
        MosaicKVMethod.PROMPT_ATTENTION_TOPK.value: controls[
            MosaicKVMethod.PROMPT_ATTENTION_TOPK.value
        ],
    }
    for candidate in candidates:
        candidate_id = str(candidate["id"])
        if candidate_id in result:
            raise ValueError(f"duplicate or reserved candidate id: {candidate_id}")
        result[candidate_id] = apply_candidate(mosaic, candidate)
    if len(result) != len(candidates) + 2:
        raise ValueError("candidate ids must be unique")
    return result


def token_agreement(left: list[int], right: list[int]) -> float:
    """Position-wise token agreement with length mismatch counted as disagreement."""

    denominator = max(len(left), len(right))
    if denominator == 0:
        return 1.0
    return sum(a == b for a, b in zip(left, right, strict=False)) / denominator


__all__ = [
    "OPTIONAL_OVERRIDE_KEYS",
    "OVERRIDE_KEYS",
    "REQUIRED_OVERRIDE_KEYS",
    "apply_candidate",
    "apply_common_cache_policy",
    "calibration_configs",
    "token_agreement",
]
