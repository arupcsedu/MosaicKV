"""Fail-closed evidence contracts for the original six-mechanism runtime.

No storage-variant module is imported or changed. Controlled repair is a
mechanism stress test, never evidence that an online policy improves quality.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from mosaickv.config import (
    AblationConfig,
    CacheConfig,
    DatasetConfig,
    ExecutionConfig,
    ForecastingConfig,
    GenerationConfig,
    GraphConfig,
    ModelConfig,
    PrototypeConfig,
    RepairConfig,
    ResidualConfig,
    RunConfig,
    SelectionConfig,
    UtilityConfig,
)
from mosaickv.types import (
    AblationVariant,
    Backend,
    BudgetUnit,
    ForecastMode,
    MosaicKVMethod,
    Precision,
    RepairPolicy,
)

ARMS = (
    "full_kv",
    "retention_one",
    "prompt_topk",
    "full_mechanisms",
    "controlled_repair",
    "prompt_only",
    "graph_coverage_off",
    "value_contribution_off",
    "exact_only",
    "no_repair",
)


def build_config(protocol: Mapping[str, Any], arm: str) -> RunConfig:
    """Freeze all comparison controls; only the named mechanism may change."""
    if arm not in ARMS:
        raise ValueError(f"unregistered six-mechanism arm: {arm}")
    cfg = RunConfig(
        model=ModelConfig(protocol["model_id"], protocol["model_revision"], Precision.BF16),
        dataset=DatasetConfig(protocol["dataset_id"], protocol["dataset_revision"], "val"),
        execution=ExecutionConfig(Backend.HUGGINGFACE, "eager", protocol["seed"], True),
        generation=GenerationConfig(max_new_tokens=protocol["output_tokens"]),
        cache=CacheConfig(
            budget_value=protocol["prompt_cap_bytes"],
            budget_unit=BudgetUnit.BYTES,
            retention_ratio=0.5,
            block_size=1,
            physical_budget_block_size=1,
            physical_budget_mandatory_policy="first_and_last",
            packing_mode="preallocated_layer",
            exact_gather_mode="source_views",
            accounting_spec_sha=hashlib.sha256(
                json.dumps(protocol["accounting"], sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        ),
        method=MosaicKVMethod.MOSAICKV_FULL,
        forecasting=ForecastingConfig(
            mode=ForecastMode.HYBRID,
            prompt_window=protocol["prompt_window"],
            draft_steps=protocol["draft_steps"],
            centroid_count=2,
            centroid_iterations=2,
        ),
        graph=GraphConfig(
            max_neighbors=8,
            similarity_chunk_size=128,
            similarity_shortlist_multiplier=2.0,
            similarity_shortlist_min_nodes=32,
            candidate_descriptor_pooling=True,
            candidate_graph_mode="shortlist_rerank",
            edge_accumulation_mode="preselected",
        ),
        utility=UtilityConfig(lambda_q=1.0, lambda_v=-0.25, lambda_o=-0.25),
        selection=SelectionConfig(
            lambda_g=-0.25,
            lambda_m=-0.25,
            stop_on_nonpositive_gain=False,
            minimum_stratum_fraction=0.0,
            routed_head_only_planning=True,
            stable_head_prompt_routing=True,
            stable_head_uniform_fill=True,
            prompt_anchor_fraction=0.25,
        ),
        prototypes=PrototypeConfig(
            group_size=protocol["maximum_members_per_prototype"],
            max_position_span=1024,
            source_modalities=("image",),
            uncertain_head_only=True,
            uncertain_head_fraction=protocol["routed_head_fraction"],
            maximum_prototypes_per_head=protocol["maximum_prototypes_per_head"],
            budget_planning_mode="one_pass_bounded_anchor_cover",
        ),
        residual=ResidualConfig(require_pinned_memory=True),
        repair=RepairConfig(max_blocks_per_step=protocol["max_restored_blocks_per_step"]),
    )
    if arm == "retention_one":
        return replace(cfg, cache=replace(cfg.cache, retention_ratio=1.0, budget_value=2**31 - 1))
    if arm == "controlled_repair":
        cfg = replace(
            cfg,
            repair=replace(
                cfg.repair,
                policy=RepairPolicy.ENTROPY,
                entropy_threshold=protocol["controlled_entropy_threshold"],
            ),
        )
    elif arm == "prompt_only":
        cfg = replace(
            cfg,
            forecasting=replace(cfg.forecasting, mode=ForecastMode.PROMPT_WINDOW, draft_steps=0),
            ablation=AblationConfig(AblationVariant.PROMPT_WINDOW_ONLY),
        )
    elif arm == "graph_coverage_off":
        # Graph descriptors remain necessary for prototypes and value novelty.
        # This ablates only the facility-coverage reward, not the whole graph.
        cfg = replace(cfg, selection=replace(cfg.selection, lambda_g=0.0))
    elif arm == "value_contribution_off":
        cfg = replace(
            cfg,
            utility=replace(cfg.utility, lambda_v=0.0),
            ablation=AblationConfig(AblationVariant.NO_VALUE_AWARE_CONTRIBUTION),
        )
    elif arm == "no_repair":
        cfg = replace(
            cfg,
            repair=replace(cfg.repair, enabled=False, policy=RepairPolicy.NONE),
            residual=replace(cfg.residual, enabled=False),
            ablation=AblationConfig(AblationVariant.NO_RESIDUAL_REPAIR),
        )
    elif arm in {"exact_only", "prompt_topk", "full_kv"}:
        method = {
            "exact_only": MosaicKVMethod.MOSAICKV_EXACT,
            "prompt_topk": MosaicKVMethod.PROMPT_ATTENTION_TOPK,
            "full_kv": MosaicKVMethod.FULL_KV,
        }[arm]
        cfg = replace(
            cfg,
            method=method,
            cache=(
                replace(cfg.cache, retention_ratio=1.0, budget_value=2**31 - 1)
                if arm == "full_kv"
                else cfg.cache
            ),
            forecasting=replace(cfg.forecasting, enabled=arm == "exact_only"),
            graph=replace(cfg.graph, enabled=arm == "exact_only"),
            prototypes=replace(cfg.prototypes, enabled=False),
            residual=replace(cfg.residual, enabled=False),
            repair=replace(cfg.repair, enabled=False, policy=RepairPolicy.NONE),
            selection=replace(
                cfg.selection,
                enabled=arm == "exact_only",
                routed_head_only_planning=arm == "exact_only",
                stable_head_prompt_routing=arm == "exact_only",
                stable_head_uniform_fill=arm == "exact_only",
            ),
        )
        if arm == "full_kv":
            cfg = replace(
                cfg, cache=replace(cfg.cache, retention_ratio=1.0, budget_value=2**31 - 1)
            )
        if arm == "exact_only":
            cfg = replace(cfg, ablation=AblationConfig(AblationVariant.NO_PROTOTYPE_MERGING))
    return cfg


def catalog_count(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        return int(value.get("record_count", -1))
    return -1


def validate_evidence(
    trace: Mapping[str, Any], operations: Mapping[str, Any], *, controlled: bool = False
) -> dict[str, bool]:
    """Require executed work, real state transitions, and finite measured fields."""
    forecast = trace.get("forecast_statistics", {}).get("provenance", {})
    utility = trace.get("utility_diagnostics", {})
    tier = trace.get("tier_storage_accounting", {})
    residual = trace.get("residual_configuration", {}) or {}
    events = trace.get("repair_events", [])
    checks = {
        "completed": trace.get("status") == "completed",
        "one_original_prefill": operations.get("prefill") == 1,
        "no_exact_or_fullkv_fallback": trace.get("tier_mode") == "three_tier"
        and trace.get("effective_method") == "mosaickv_full",
        "hybrid_forecast_executed": forecast.get("mode") == "hybrid"
        and forecast.get("completed_draft_steps", 0) >= 1
        and forecast.get("reused_original_prefill") is True
        and forecast.get("draft_cache_isolated") is True
        and operations.get("forecast", 0) > 0,
        "cross_modal_graph_executed": operations.get("graph", 0) > 0
        and operations.get("cross_modal_edges", 0) > 0
        and catalog_count(trace.get("graph_edges")) > 0,
        "value_aware_utility_executed": operations.get("utility", 0) > 0
        and utility.get("weights", {}).get("lambda_v", 0) != 0
        and utility.get("value_contribution", {}).get("maximum", 0) > 0,
        "budgeted_selection_executed": operations.get("planner", 0) > 0
        and operations.get("selector", 0) > 0
        and catalog_count(trace.get("selected_blocks")) > 0,
        "three_real_tiers": operations.get("prototype_members", 0) > 0
        and catalog_count(trace.get("prototypes")) > 0
        and tier.get("exact_active_bytes", 0) > 0
        and tier.get("prototype_active_bytes", 0) > 0
        and tier.get("residual_cpu_bytes", 0) > 0,
        "lossless_pinned_residuals": residual.get("storage_dtype") == "lossless"
        and residual.get("all_payloads_pinned") is True,
        "repair_controller_executed": operations.get("repair", 0) > 0 and bool(events),
        "logical_positions_and_finite_logits": bool(operations.get("decode_checks"))
        and all(operations["decode_checks"]),
        "actual_rope_positions_preserved": bool(operations.get("rope_position_checks"))
        and all(operations["rope_position_checks"]),
        "inactive_source_accounting_present": trace.get("request_KV_ownership", {}).get(
            "includes_registered_inactive_source_and_transient_KV"
        )
        is True,
        "mandatory_and_storage_invariants": operations.get("plan_invariants") is True
        and operations.get("packed_storage_checks") is True,
        "finite_phase_measurements": bool(trace.get("timing_breakdown"))
        and all(
            isinstance(trace["timing_breakdown"].get(k), (int, float))
            and math.isfinite(trace["timing_breakdown"][k])
            and trace["timing_breakdown"][k] >= 0
            for k in (
                "forecast",
                "graph",
                "utility_and_selection",
                "tier_construction",
                "cache_packing",
                "decode",
                "total",
            )
        ),
    }
    if controlled:
        triggered = [event for event in events if event.get("triggered")]
        checks["controlled_real_promotion_and_redecode"] = bool(triggered) and all(
            event.get("restored_bytes", 0) > 0
            and event.get("restored_block_ids")
            and event.get("re_decode_count") == 1
            for event in triggered
        )
        checks["controlled_gpu_transfer"] = bool(triggered) and all(
            event.get("transfer_was_asynchronous") is True for event in triggered
        )
        checks["controlled_physical_cap"] = operations.get("repair_physical_cap_checks") is True
        checks["promoted_original_kv_exact"] = bool(
            operations.get("promoted_kv_exact_checks")
        ) and all(operations["promoted_kv_exact_checks"])
    return checks


def all_pass(checks: Mapping[str, bool]) -> bool:
    return bool(checks) and all(type(value) is bool and value for value in checks.values())
