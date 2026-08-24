#!/usr/bin/env python3
"""Select and freeze one v74 candidate using label-blind FullKV fidelity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path
from typing import Any, cast

import pandas as pd  # type: ignore[import-untyped]
import yaml
from choice_c_v74_protocol import token_agreement
from run_choice_c_gate import _sha
from run_choice_c_v72_heldout_quality_memory_worker import _resolve

CONTROL_ARMS = ("full_kv", "prompt_attention_topk")


def _normalize(value: object) -> str:
    return " ".join(str(value).strip().casefold().split())


def _mean_metric(frame: pd.DataFrame, column: str) -> float | None:
    """Return a finite mean while treating pandas missing scalars uniformly."""

    if column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce")
    values = values[~pd.isna(values)]
    if values.empty:
        return None
    return float(values.mean())


def _format_mb(value: object) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(cast('Any', value)) / 1_000_000:.2f} MB"


def _format_fraction(value: object) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(cast('Any', value)):.2%}"


def _boolean_metric(
    frame: pd.DataFrame,
    column: str,
    *,
    missing_default: bool,
) -> tuple[pd.Series, pd.Series]:
    """Normalize booleans without allowing pandas missing values to become true."""

    raw = (
        frame[column]
        if column in frame
        else pd.Series(pd.NA, index=frame.index, dtype="object")
    )
    missing = raw.map(pd.isna).astype(bool)
    normalized = raw.map(
        lambda value: missing_default if pd.isna(value) else bool(value)
    ).astype(bool)
    return normalized, missing


def _write_frozen(path: Path, payload: dict[str, Any]) -> None:
    encoded = yaml.safe_dump(payload, sort_keys=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"refusing to overwrite a different frozen candidate: {path}")
        return
    path.write_text(encoded, encoding="utf-8")
    os.chmod(path, 0o444)


def _load_telemetry(
    output_root: Path,
    *,
    task: str,
    expected_arms: tuple[str, ...],
    expected_samples: int,
    run_prefix: str = "v74",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    paths = tuple(sorted((output_root / "telemetry").glob(f"{run_prefix}-{task}-*.json")))
    if len(paths) != 1:
        raise RuntimeError(f"expected one {run_prefix} {task} telemetry file, found {len(paths)}")
    telemetry = cast("dict[str, Any]", json.loads(paths[0].read_text(encoding="utf-8")))
    if telemetry.get("status") != "completed" or telemetry.get("canonical_eligible") is not True:
        raise RuntimeError(f"{task} calibration telemetry is incomplete or noncanonical")
    if (
        telemetry.get("labels_accessed_for_selection") is not False
        or telemetry.get("official_task_scores_accessed_for_selection") is not False
    ):
        raise RuntimeError(f"{task} telemetry does not preserve the label-blind boundary")
    sample_ids = tuple(cast("list[str]", telemetry["sample_ids"]))
    if len(sample_ids) != expected_samples or len(set(sample_ids)) != expected_samples:
        raise RuntimeError(f"{task} calibration telemetry has the wrong sample set")
    if tuple(cast("list[str]", telemetry["arm_order"])) != expected_arms:
        raise RuntimeError(f"{task} calibration arms differ from the frozen protocol")
    rows: list[dict[str, Any]] = []
    trials = cast("list[dict[str, Any]]", telemetry["trials"])
    if len(trials) != len(expected_arms):
        raise RuntimeError(f"{task} calibration has an incomplete arm set")
    for trial in trials:
        arm = str(trial["arm"])
        if arm not in expected_arms:
            raise RuntimeError(f"{task} contains unknown calibration arm {arm!r}")
        trial_rows = cast("list[dict[str, Any]]", trial["rows"])
        if {str(row["sample_id"]) for row in trial_rows} != set(sample_ids):
            raise RuntimeError(f"{task}/{arm} does not cover the fixed calibration samples")
        for row in trial_rows:
            if "reference" in row or "task_score" in row:
                raise RuntimeError("label-bearing fields entered the calibration telemetry")
            if row.get("status") != "completed" or row.get("error") not in {None, "nan"}:
                raise RuntimeError(f"failed calibration row: {task}/{arm}/{row['sample_id']}")
            rows.append({**row, "task": task, "arm": arm})
    return telemetry, rows


def analyze(
    spec_path: Path,
    output_root: Path,
    frozen_output: Path,
    *,
    run_prefix: str = "v74",
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    package_root = project_root / "mosaickv"
    spec = cast("dict[str, Any]", yaml.safe_load(spec_path.read_text(encoding="utf-8")))
    candidates = cast("list[dict[str, Any]]", spec["candidates"])
    common_cache_policy = cast("dict[str, Any]", spec["common_cache_policy"])
    candidate_ids = tuple(str(candidate["id"]) for candidate in candidates)
    candidates_by_id = {str(candidate["id"]): candidate for candidate in candidates}
    arms = (*CONTROL_ARMS, *candidate_ids)
    objective = cast("dict[str, Any]", spec["calibration_objective"])
    if (
        objective.get("labels_accessed") is not False
        or objective.get("official_task_scores_accessed") is not False
    ):
        raise RuntimeError("calibration objective is not label-blind")

    telemetry_by_task: dict[str, dict[str, Any]] = {}
    all_rows: list[dict[str, Any]] = []
    tasks = cast("list[dict[str, Any]]", spec["tasks"])
    for entry in tasks:
        task = str(entry["task"])
        telemetry, rows = _load_telemetry(
            output_root,
            task=task,
            expected_arms=arms,
            expected_samples=int(entry["sample_count"]),
            run_prefix=run_prefix,
        )
        telemetry_by_task[task] = telemetry
        if telemetry.get("common_cache_policy") != common_cache_policy:
            raise RuntimeError(f"{task} did not use the frozen common cache policy")
        all_rows.extend(rows)
    snapshots = {
        str(cast("dict[str, Any]", telemetry["source_snapshot"])["snapshot_sha256"])
        for telemetry in telemetry_by_task.values()
    }
    if len(snapshots) != 1:
        raise RuntimeError("calibration workers used different source snapshots")

    frame = pd.DataFrame(all_rows).sort_values(["task", "arm", "sample_id"])
    analysis_root = output_root / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(analysis_root / "label_blind_calibration_trials.parquet", index=False)

    minimum_active = float(objective["minimum_active_reduction_fraction_vs_prompt_topk"])
    minimum_total = float(
        objective["minimum_active_plus_residual_reduction_fraction_vs_prompt_topk"]
    )
    fidelity_margin = float(objective["fullkv_answer_noninferiority_margin_vs_prompt_topk"])
    if not 0 <= fidelity_margin <= 1:
        raise ValueError("fullkv_answer_noninferiority_margin_vs_prompt_topk must be in [0, 1]")
    maximum_prototype_fraction_raw = objective.get("maximum_prototype_logical_fraction")
    maximum_prototype_fraction = (
        None if maximum_prototype_fraction_raw is None else float(maximum_prototype_fraction_raw)
    )
    if maximum_prototype_fraction is not None and not 0 <= maximum_prototype_fraction <= 1:
        raise ValueError("maximum_prototype_logical_fraction must be in [0, 1]")
    require_zero_residual_bytes_raw = objective.get("require_zero_residual_bytes", False)
    if not isinstance(require_zero_residual_bytes_raw, bool):
        raise ValueError("require_zero_residual_bytes must be a boolean")
    require_zero_residual_bytes = require_zero_residual_bytes_raw
    divergence_protocol = cast("dict[str, Any]", spec.get("forecast_divergence_analysis", {}))
    if divergence_protocol:
        if divergence_protocol.get("metric") != "prompt_forecast_total_variation":
            raise ValueError("unsupported forecast divergence metric")
        if (
            divergence_protocol.get("require_at_least_one_eligible_stable_route_per_sample")
            is not True
        ):
            raise ValueError("forecast divergence protocol must fail closed per sample")
    risk_gate_protocol = cast("dict[str, Any]", spec.get("future_rerank_risk_analysis", {}))
    risk_control_arm: str | None = None
    risk_candidate_ids: tuple[str, ...] = ()
    risk_control_margin = 0.0
    if risk_gate_protocol:
        risk_control_arm = str(risk_gate_protocol.get("control_arm", ""))
        risk_candidate_arms_raw = risk_gate_protocol.get("candidate_arms", [])
        if not isinstance(risk_candidate_arms_raw, list) or not all(
            isinstance(value, str) and value for value in risk_candidate_arms_raw
        ):
            raise ValueError("future-rerank risk candidate_arms must be a list of names")
        risk_candidate_ids = tuple(cast("list[str]", risk_candidate_arms_raw))
        if risk_control_arm not in candidate_ids:
            raise ValueError("future-rerank risk control must name a preregistered candidate")
        if not risk_candidate_ids or any(
            value not in candidate_ids for value in risk_candidate_ids
        ):
            raise ValueError("future-rerank risk candidates must name preregistered candidates")
        if risk_control_arm in risk_candidate_ids or len(set(risk_candidate_ids)) != len(
            risk_candidate_ids
        ):
            raise ValueError("future-rerank risk control and candidate arms must be distinct")
        if risk_gate_protocol.get("require_positive_accepted_reranks_each_task") is not True:
            raise ValueError("future-rerank risk analysis must require accepted reranks")
        if risk_gate_protocol.get("require_positive_risk_rejections_each_task") is not True:
            raise ValueError("future-rerank risk analysis must require risk rejections")
        if risk_gate_protocol.get("require_not_below_control_each_task") is not True:
            raise ValueError("future-rerank risk analysis must require control noninferiority")
        risk_control_margin = float(
            risk_gate_protocol.get("fullkv_answer_noninferiority_margin_vs_control", 0.0)
        )
        if not 0 <= risk_control_margin <= 1:
            raise ValueError("future-rerank control noninferiority margin must be in [0, 1]")
    logit_protocol = cast("dict[str, Any]", spec.get("short_horizon_logit_stability_analysis", {}))
    if logit_protocol:
        if logit_protocol.get("labels_accessed") is not False:
            raise ValueError("short-horizon logit stability must be label blind")
        if logit_protocol.get("require_every_sample_to_pass") is not True:
            raise ValueError("short-horizon logit stability must fail closed per sample")
        if logit_protocol.get("require_fullkv_prefill_reuse") is not True:
            raise ValueError("short-horizon logit stability must reuse the FullKV prefill")
        if logit_protocol.get("forbid_second_prefill") is not True:
            raise ValueError("short-horizon logit stability must forbid a second prefill")
        if logit_protocol.get("require_probe_state_discard_before_measured_decode") is not True:
            raise ValueError("short-horizon logit stability must discard probe states")
        if logit_protocol.get("require_greedy_token_match") is not True:
            raise ValueError("short-horizon logit stability must require greedy-token parity")
    progressive_protocol = cast(
        "dict[str, Any]",
        spec.get("progressive_logit_safe_eviction_analysis", {}),
    )
    if progressive_protocol:
        required_true = {
            "require_start_from_fullkv": "start from FullKV",
            "require_every_sample_to_reach_target": "reach the target for every sample",
            "require_every_accepted_checkpoint_to_pass": "pass every accepted checkpoint",
            "require_fullkv_prefill_reuse": "reuse the FullKV prefill",
            "forbid_second_prefill": "forbid a second prefill",
            "require_probe_state_discard_before_measured_decode": "discard probe states",
            "require_greedy_token_match": "require greedy-token parity",
        }
        if progressive_protocol.get("labels_accessed") is not False:
            raise ValueError("progressive logit-safe eviction must be label blind")
        for field, description in required_true.items():
            if progressive_protocol.get(field) is not True:
                raise ValueError(f"progressive logit-safe eviction must {description}")
    task_results: dict[str, dict[str, Any]] = {}
    decomposition = cast("dict[str, Any]", spec.get("decomposition_analysis", {}))
    prototype_decomposition_arm = decomposition.get("prototype_arm")
    exact_decomposition_arm = decomposition.get("exact_only_arm")
    if decomposition:
        if (
            prototype_decomposition_arm not in candidate_ids
            or exact_decomposition_arm not in candidate_ids
        ):
            raise ValueError("decomposition arms must name preregistered candidates")
        if decomposition.get("require_paired_samples") is not True:
            raise ValueError("v81 decomposition must require paired samples")
        if decomposition.get("require_equal_configured_byte_cap") is not True:
            raise ValueError("v81 decomposition must require an equal configured byte cap")
    candidate_ranking: list[dict[str, Any]] = []
    for task_entry in tasks:
        task = str(task_entry["task"])
        task_frame = frame[frame["task"] == task]
        by_arm = {
            arm: task_frame[task_frame["arm"] == arm].set_index("sample_id").sort_index()
            for arm in arms
        }
        ids = tuple(by_arm["full_kv"].index)
        if any(tuple(table.index) != ids for table in by_arm.values()):
            raise RuntimeError(f"{task} calibration arms are not sample-paired")
        prompt = by_arm["prompt_attention_topk"]
        full = by_arm["full_kv"]
        prompt_full_answer = statistics.fmean(
            _normalize(prompt.at[sample_id, "answer"]) == _normalize(full.at[sample_id, "answer"])
            for sample_id in ids
        )
        prompt_full_token = statistics.fmean(
            token_agreement(
                cast("list[int]", prompt.at[sample_id, "generated_token_ids"]),
                cast("list[int]", full.at[sample_id, "generated_token_ids"]),
            )
            for sample_id in ids
        )
        candidate_results: dict[str, Any] = {}
        for candidate_id in candidate_ids:
            candidate = by_arm[candidate_id]
            candidate_spec = candidates_by_id[candidate_id]
            candidate_overrides = cast("dict[str, Any]", candidate_spec["overrides"])
            answer_agreement = statistics.fmean(
                _normalize(candidate.at[sample_id, "answer"])
                == _normalize(full.at[sample_id, "answer"])
                for sample_id in ids
            )
            token_fidelity = statistics.fmean(
                token_agreement(
                    cast("list[int]", candidate.at[sample_id, "generated_token_ids"]),
                    cast("list[int]", full.at[sample_id, "generated_token_ids"]),
                )
                for sample_id in ids
            )
            prompt_bytes = float(prompt["active_kv_bytes"].mean())
            active_bytes = float(candidate["active_kv_bytes"].mean())
            residual_bytes = float(candidate["residual_kv_bytes"].mean())
            active_reduction = 1.0 - active_bytes / prompt_bytes
            total_reduction = 1.0 - (active_bytes + residual_bytes) / prompt_bytes
            memory_eligible = (
                active_reduction >= minimum_active and total_reduction >= minimum_total
            )
            fidelity_delta = answer_agreement - prompt_full_answer
            fidelity_eligible = fidelity_delta >= -fidelity_margin
            configured_cap = _mean_metric(candidate, "byte_budget_configured_active_byte_cap")
            attainable_cap = _mean_metric(candidate, "byte_budget_dense_attainable_budget_bytes")
            quantization_underfill = _mean_metric(
                candidate, "byte_budget_dense_quantization_underfill_bytes"
            )
            capacity_fraction_reduction = _mean_metric(
                candidate, "byte_budget_capacity_fraction_reduction_bytes"
            )
            logical_underfill = _mean_metric(
                candidate, "byte_budget_logical_underfill_vs_dense_budget_bytes"
            )
            final_underfill = _mean_metric(
                candidate, "byte_budget_final_underfill_vs_dense_budget_bytes"
            )
            padding_bytes = _mean_metric(candidate, "byte_budget_packed_padding_bytes")
            external_requested_positions = _mean_metric(
                candidate, "external_stable_requested_positions"
            )
            external_selected_positions = _mean_metric(
                candidate, "external_stable_selected_positions"
            )
            external_shortfall_positions = _mean_metric(
                candidate, "external_stable_shortfall_positions"
            )
            external_shortfall_routes = _mean_metric(
                candidate, "external_stable_routes_with_shortfall_count"
            )
            external_adaptive_swaps = _mean_metric(candidate, "external_stable_adaptive_swap_count")
            baseline_text_mass = _mean_metric(
                candidate, "external_stable_baseline_text_prompt_mass"
            )
            selected_text_mass = _mean_metric(
                candidate, "external_stable_selected_text_prompt_mass"
            )
            text_mass_preservation = (
                None
                if baseline_text_mass is None
                or selected_text_mass is None
                or baseline_text_mass <= 0
                else selected_text_mass / baseline_text_mass
            )
            normalized_visual_gain = _mean_metric(
                candidate, "external_stable_normalized_visual_evidence_gain"
            )
            normalized_text_loss = _mean_metric(
                candidate, "external_stable_normalized_text_prompt_mass_loss"
            )
            normalized_forecast_gain = _mean_metric(
                candidate, "external_stable_normalized_forecast_visual_gain"
            )
            normalized_forecast_loss = _mean_metric(
                candidate, "external_stable_normalized_forecast_text_loss"
            )
            adaptive_candidates = _mean_metric(
                candidate, "external_stable_adaptive_swap_candidate_count"
            )
            adaptive_risk_rejections = _mean_metric(
                candidate, "external_stable_adaptive_swap_risk_rejected_count"
            )
            adaptive_routes = _mean_metric(
                candidate, "external_stable_routes_with_adaptive_swaps_count"
            )
            risk_rejected_routes = _mean_metric(
                candidate, "external_stable_routes_with_risk_rejections_count"
            )
            future_rerank_count = _mean_metric(candidate, "external_stable_future_rerank_count")
            future_baseline_prompt_mass = _mean_metric(
                candidate, "external_stable_baseline_selected_prompt_mass"
            )
            future_selected_prompt_mass = _mean_metric(
                candidate, "external_stable_selected_prompt_mass_after_future_rerank"
            )
            future_prompt_preservation = (
                None
                if future_baseline_prompt_mass is None
                or future_selected_prompt_mass is None
                or future_baseline_prompt_mass <= 0
                else future_selected_prompt_mass / future_baseline_prompt_mass
            )
            normalized_future_query_gain = _mean_metric(
                candidate, "external_stable_normalized_future_query_gain"
            )
            normalized_future_prompt_loss = _mean_metric(
                candidate, "external_stable_normalized_future_prompt_mass_loss"
            )
            future_rerank_candidates = _mean_metric(
                candidate, "external_stable_future_rerank_candidate_count"
            )
            future_rerank_rejections = _mean_metric(
                candidate, "external_stable_future_rerank_rejected_count"
            )
            future_rerank_risk_rejections = _mean_metric(
                candidate, "external_stable_future_rerank_risk_rejected_count"
            )
            future_rerank_routes = _mean_metric(
                candidate, "external_stable_routes_with_future_reranking_count"
            )
            future_rejected_routes = _mean_metric(
                candidate,
                "external_stable_routes_with_future_rerank_rejections_count",
            )
            future_risk_rejected_routes = _mean_metric(
                candidate,
                "external_stable_routes_with_future_rerank_risk_rejections_count",
            )
            forecast_divergence_routes = _mean_metric(
                candidate,
                "external_stable_forecast_divergence_eligible_route_count",
            )
            prompt_forecast_tv_min = _mean_metric(
                candidate,
                "external_stable_prompt_forecast_total_variation_min",
            )
            prompt_forecast_tv_mean = _mean_metric(
                candidate,
                "external_stable_prompt_forecast_total_variation_mean",
            )
            prompt_forecast_tv_max = _mean_metric(
                candidate,
                "external_stable_prompt_forecast_total_variation_max",
            )
            query_coverage_rerank_count = _mean_metric(
                candidate,
                "external_stable_query_coverage_rerank_count",
            )
            baseline_query_coverage = _mean_metric(
                candidate,
                "external_stable_baseline_query_coverage",
            )
            selected_query_coverage = _mean_metric(
                candidate,
                "external_stable_selected_query_coverage",
            )
            normalized_query_coverage_gain = _mean_metric(
                candidate,
                "external_stable_normalized_query_coverage_gain",
            )
            query_coverage_baseline_prompt_mass = _mean_metric(
                candidate,
                "external_stable_query_coverage_baseline_prompt_mass",
            )
            query_coverage_selected_prompt_mass = _mean_metric(
                candidate,
                "external_stable_query_coverage_selected_prompt_mass",
            )
            query_coverage_prompt_mass_preservation = (
                None
                if query_coverage_baseline_prompt_mass is None
                or query_coverage_selected_prompt_mass is None
                or query_coverage_baseline_prompt_mass <= 0
                else (query_coverage_selected_prompt_mass / query_coverage_baseline_prompt_mass)
            )
            query_coverage_normalized_prompt_loss = _mean_metric(
                candidate,
                "external_stable_query_coverage_normalized_prompt_loss",
            )
            query_coverage_candidate_count = _mean_metric(
                candidate,
                "external_stable_query_coverage_candidate_count",
            )
            query_coverage_rejected_count = _mean_metric(
                candidate,
                "external_stable_query_coverage_rejected_count",
            )
            query_coverage_routes = _mean_metric(
                candidate,
                "external_stable_routes_with_query_coverage_reranking_count",
            )
            query_coverage_rejected_routes = _mean_metric(
                candidate,
                "external_stable_routes_with_query_coverage_rejections_count",
            )
            initial_query_coverage_reserved = _mean_metric(
                candidate,
                "external_stable_query_coverage_initial_reserved_count",
            )
            initial_query_coverage_changed = _mean_metric(
                candidate,
                "external_stable_query_coverage_initial_changed_count",
            )
            initial_baseline_query_coverage = _mean_metric(
                candidate,
                "external_stable_query_coverage_initial_baseline",
            )
            initial_selected_query_coverage = _mean_metric(
                candidate,
                "external_stable_query_coverage_initial_selected",
            )
            initial_normalized_query_coverage_gain = _mean_metric(
                candidate,
                "external_stable_query_coverage_initial_normalized_gain",
            )
            initial_baseline_prompt_mass = _mean_metric(
                candidate,
                "external_stable_query_coverage_initial_baseline_prompt_mass",
            )
            initial_selected_prompt_mass = _mean_metric(
                candidate,
                "external_stable_query_coverage_initial_selected_prompt_mass",
            )
            initial_prompt_mass_preservation = (
                None
                if initial_baseline_prompt_mass is None
                or initial_selected_prompt_mass is None
                or initial_baseline_prompt_mass <= 0
                else initial_selected_prompt_mass / initial_baseline_prompt_mass
            )
            initial_normalized_prompt_loss = _mean_metric(
                candidate,
                "external_stable_query_coverage_initial_normalized_prompt_loss",
            )
            initial_query_coverage_candidates = _mean_metric(
                candidate,
                "external_stable_query_coverage_initial_candidate_count",
            )
            initial_query_coverage_reserved_bytes = _mean_metric(
                candidate,
                "external_stable_query_coverage_initial_reserved_bytes",
            )
            initial_query_coverage_allocated_bytes = _mean_metric(
                candidate,
                "external_stable_query_coverage_initial_allocated_bytes",
            )
            initial_query_coverage_cross_head_risk = _mean_metric(
                candidate,
                "external_stable_query_coverage_initial_cross_head_risk",
            )
            initial_query_coverage_proposed_gain = _mean_metric(
                candidate,
                "external_stable_query_coverage_initial_proposed_normalized_gain",
            )
            initial_query_coverage_proposed_prompt_loss = _mean_metric(
                candidate,
                "external_stable_query_coverage_initial_proposed_normalized_prompt_loss",
            )
            initial_query_coverage_proposed_output_impact_risk = _mean_metric(
                candidate,
                "external_stable_query_coverage_initial_proposed_output_impact_risk",
            )
            initial_query_coverage_output_impact_donor_count = _mean_metric(
                candidate,
                "external_stable_query_coverage_initial_output_impact_donor_count",
            )
            initial_query_coverage_output_impact_protected_count = _mean_metric(
                candidate,
                "external_stable_query_coverage_initial_output_impact_protected_count",
            )
            initial_query_coverage_output_impact_protected_bytes = _mean_metric(
                candidate,
                "external_stable_query_coverage_initial_output_impact_protected_bytes",
            )
            initial_query_coverage_output_impact_protected_score = _mean_metric(
                candidate,
                "external_stable_query_coverage_initial_output_impact_protected_score",
            )
            initial_layout_group_candidates = _mean_metric(
                candidate,
                "external_stable_query_coverage_initial_layout_group_candidate_count",
            )
            initial_layout_group_count = _mean_metric(
                candidate,
                "external_stable_query_coverage_initial_layout_group_protected_group_count",
            )
            initial_layout_group_blocks = _mean_metric(
                candidate,
                "external_stable_query_coverage_initial_layout_group_protected_block_count",
            )
            initial_layout_group_bytes = _mean_metric(
                candidate,
                "external_stable_query_coverage_initial_layout_group_protected_bytes",
            )
            initial_layout_group_score = _mean_metric(
                candidate,
                "external_stable_query_coverage_initial_layout_group_protected_score",
            )
            initial_layout_group_target_bytes = _mean_metric(
                candidate,
                "external_stable_query_coverage_initial_layout_group_target_bytes",
            )
            initial_layout_group_accepted_candidates = _mean_metric(
                candidate,
                "external_stable_query_coverage_initial_layout_group_accepted_candidate_count",
            )
            initial_layout_group_risk_rejections = _mean_metric(
                candidate,
                "external_stable_query_coverage_initial_layout_group_risk_rejected_count",
            )
            initial_layout_group_pareto_candidates = _mean_metric(
                candidate,
                "external_stable_query_coverage_initial_layout_group_pareto_candidate_count",
            )
            initial_layout_group_knapsack_selected = _mean_metric(
                candidate,
                "external_stable_query_coverage_initial_layout_group_knapsack_selected_count",
            )
            initial_layout_group_joint_donor_candidates = _mean_metric(
                candidate,
                "external_stable_query_coverage_initial_layout_group_joint_donor_candidate_count",
            )
            initial_layout_group_query_gain = _mean_metric(
                candidate,
                "external_stable_query_coverage_initial_layout_group_normalized_query_gain",
            )
            initial_layout_group_prompt_risk = _mean_metric(
                candidate,
                "external_stable_query_coverage_initial_layout_group_normalized_prompt_risk",
            )
            initial_layout_group_output_risk = _mean_metric(
                candidate,
                "external_stable_query_coverage_initial_layout_group_output_impact_risk",
            )
            initial_query_coverage_stability_rejections = _mean_metric(
                candidate,
                "external_stable_query_coverage_initial_stability_rejected_route_count",
            )
            initial_query_coverage_global_routes = _mean_metric(
                candidate,
                "external_stable_query_coverage_initial_global_allocation_route_count",
            )
            initial_query_coverage_allocated_routes = _mean_metric(
                candidate,
                "external_stable_query_coverage_initial_allocated_route_count",
            )
            initial_query_coverage_routes = _mean_metric(
                candidate,
                "external_stable_routes_with_query_coverage_initial_selection_count",
            )
            exact_prompt_mass_fraction = _mean_metric(
                candidate, "prompt_mass_exact_prompt_mass_fraction"
            )
            represented_prompt_mass_fraction = _mean_metric(
                candidate, "prompt_mass_represented_prompt_mass_fraction"
            )
            prototype_count = _mean_metric(candidate, "prototype_count")
            exact_logical_bytes = _mean_metric(candidate, "byte_budget_exact_logical_active_bytes")
            prototype_logical_bytes = _mean_metric(
                candidate, "byte_budget_prototype_logical_active_bytes"
            )
            logical_accounted_bytes = _mean_metric(candidate, "byte_budget_logical_active_bytes")
            prototype_fraction = (
                None
                if prototype_logical_bytes is None
                or logical_accounted_bytes is None
                or logical_accounted_bytes <= 0
                else prototype_logical_bytes / logical_accounted_bytes
            )
            exact_dominant_eligible = (
                True
                if maximum_prototype_fraction is None
                else prototype_fraction is not None
                and prototype_fraction <= maximum_prototype_fraction
            )
            residual_storage_eligible = bool(not require_zero_residual_bytes or residual_bytes == 0)
            requires_forecast_divergence = bool(
                candidate_overrides.get("stable_head_future_divergence_fail_closed", False)
            )
            minimum_forecast_tv = float(
                candidate_overrides.get("stable_head_future_minimum_prompt_forecast_tv", 0.0)
            )
            divergence_route_series = pd.to_numeric(
                candidate.get(
                    "external_stable_forecast_divergence_eligible_route_count",
                    pd.Series(index=candidate.index, dtype=float),
                ),
                errors="coerce",
            )
            divergence_max_series = pd.to_numeric(
                candidate.get(
                    "external_stable_prompt_forecast_total_variation_max",
                    pd.Series(index=candidate.index, dtype=float),
                ),
                errors="coerce",
            )
            forecast_divergence_eligible = bool(
                not requires_forecast_divergence
                or (
                    not divergence_route_series.isna().any()
                    and bool((divergence_route_series > 0).all())
                    and not divergence_max_series.isna().any()
                    and bool((divergence_max_series + 1e-12 >= minimum_forecast_tv).all())
                )
            )
            requires_layout_groups = bool(
                candidate_overrides.get(
                    "stable_head_query_coverage_layout_group_reservation",
                    False,
                )
            )
            requires_atomic_layout_groups = bool(
                candidate_overrides.get(
                    "stable_head_query_coverage_layout_group_atomic_selection",
                    False,
                )
            )
            minimum_layout_utilization = float(
                candidate_overrides.get(
                    "stable_head_query_coverage_layout_group_minimum_utilization_fraction",
                    0.0,
                )
            )
            layout_target_series = pd.to_numeric(
                candidate.get(
                    "external_stable_query_coverage_initial_layout_group_target_bytes",
                    pd.Series(index=candidate.index, dtype=float),
                ),
                errors="coerce",
            )
            layout_protected_series = pd.to_numeric(
                candidate.get(
                    "external_stable_query_coverage_initial_layout_group_protected_bytes",
                    pd.Series(index=candidate.index, dtype=float),
                ),
                errors="coerce",
            )
            layout_utilization_series = layout_protected_series / layout_target_series
            layout_group_utilization_eligible = bool(
                not requires_atomic_layout_groups
                or (
                    not layout_target_series.isna().any()
                    and bool((layout_target_series > 0).all())
                    and not layout_protected_series.isna().any()
                    and bool(
                        (layout_utilization_series + 1e-12 >= minimum_layout_utilization).all()
                    )
                )
            )
            atomic_selection_mode = candidate_overrides.get(
                "stable_head_query_coverage_layout_group_atomic_selection_mode",
                "topk_greedy",
            )
            requires_pareto_knapsack = atomic_selection_mode in {
                "pareto_knapsack",
                "pareto_joint_donor_knapsack",
            }
            requires_joint_group_donors = atomic_selection_mode == "pareto_joint_donor_knapsack"
            defers_surrogate_stability_to_logits = bool(
                candidate_overrides.get("stable_head_short_horizon_logit_stability", False)
                or candidate_overrides.get(
                    "stable_head_progressive_logit_safe_eviction",
                    False,
                )
            )
            pareto_knapsack_activity_eligible = bool(
                not requires_pareto_knapsack
                or (
                    initial_layout_group_pareto_candidates is not None
                    and initial_layout_group_pareto_candidates > 0
                    and initial_layout_group_knapsack_selected is not None
                    and initial_layout_group_knapsack_selected > 0
                    and (
                        defers_surrogate_stability_to_logits
                        or (
                            initial_query_coverage_output_impact_donor_count is not None
                            and initial_query_coverage_output_impact_donor_count > 0
                        )
                    )
                )
            )
            joint_group_donor_activity_eligible = bool(
                not requires_joint_group_donors
                or (
                    initial_layout_group_joint_donor_candidates is not None
                    and initial_layout_group_joint_donor_candidates > 0
                )
            )
            layout_group_activity_eligible = bool(
                not requires_layout_groups
                or (
                    initial_layout_group_candidates is not None
                    and initial_layout_group_candidates > 0
                    and initial_layout_group_count is not None
                    and initial_layout_group_count > 0
                    and initial_layout_group_blocks is not None
                    and initial_layout_group_blocks > initial_layout_group_count
                    and initial_layout_group_bytes is not None
                    and initial_layout_group_bytes > 0
                    and initial_layout_group_score is not None
                    and initial_layout_group_score > 0
                )
            )
            requires_logit_stability = bool(
                candidate_overrides.get("stable_head_short_horizon_logit_stability", False)
            )
            configured_logit_shortlist = int(
                candidate_overrides.get("stable_head_logit_stability_shortlist_size", 0)
            )
            configured_logit_horizon = int(
                candidate_overrides.get("stable_head_logit_stability_horizon", 0)
            )
            configured_maximum_kl = float(
                candidate_overrides.get("stable_head_logit_stability_maximum_kl", 0.0)
            )
            configured_maximum_logit_difference = float(
                candidate_overrides.get(
                    "stable_head_logit_stability_maximum_logit_difference",
                    0.0,
                )
            )
            logit_enabled, logit_enabled_missing = _boolean_metric(
                candidate,
                "logit_stability_enabled",
                missing_default=False,
            )
            logit_selected_passed, logit_selected_passed_missing = _boolean_metric(
                candidate,
                "logit_stability_selected_candidate_passed",
                missing_default=False,
            )
            logit_greedy_match, logit_greedy_match_missing = _boolean_metric(
                candidate,
                "logit_stability_selected_greedy_token_match",
                missing_default=False,
            )
            logit_prefill_reused, logit_prefill_reused_missing = _boolean_metric(
                candidate,
                "logit_stability_fullkv_prefill_reused",
                missing_default=False,
            )
            logit_second_prefill, logit_second_prefill_missing = _boolean_metric(
                candidate,
                "logit_stability_second_prefill_performed",
                missing_default=True,
            )
            logit_probe_states_discarded, logit_probe_states_discarded_missing = (
                _boolean_metric(
                    candidate,
                    "logit_stability_probe_states_discarded_before_measured_decode",
                    missing_default=False,
                )
            )
            logit_labels_accessed, logit_labels_accessed_missing = _boolean_metric(
                candidate,
                "logit_stability_benchmark_labels_accessed",
                missing_default=True,
            )
            logit_shortlist_count = pd.to_numeric(
                candidate.get(
                    "logit_stability_evaluated_candidate_count",
                    pd.Series(index=candidate.index, dtype=float),
                ),
                errors="coerce",
            )
            logit_horizon = pd.to_numeric(
                candidate.get(
                    "logit_stability_horizon",
                    pd.Series(index=candidate.index, dtype=float),
                ),
                errors="coerce",
            )
            logit_maximum_kl = pd.to_numeric(
                candidate.get(
                    "logit_stability_selected_maximum_kl",
                    pd.Series(index=candidate.index, dtype=float),
                ),
                errors="coerce",
            )
            logit_maximum_difference = pd.to_numeric(
                candidate.get(
                    "logit_stability_selected_maximum_logit_difference",
                    pd.Series(index=candidate.index, dtype=float),
                ),
                errors="coerce",
            )
            logit_reported_kl_threshold = pd.to_numeric(
                candidate.get(
                    "logit_stability_maximum_kl_threshold",
                    pd.Series(index=candidate.index, dtype=float),
                ),
                errors="coerce",
            )
            logit_reported_difference_threshold = pd.to_numeric(
                candidate.get(
                    "logit_stability_maximum_logit_difference_threshold",
                    pd.Series(index=candidate.index, dtype=float),
                ),
                errors="coerce",
            )
            logit_fallback, _logit_fallback_missing = _boolean_metric(
                candidate,
                "logit_stability_used_fail_closed_scalar_anchor",
                missing_default=False,
            )
            logit_stability_eligible = bool(
                not requires_logit_stability
                or (
                    not bool(logit_enabled_missing.any())
                    and not bool(logit_selected_passed_missing.any())
                    and not bool(logit_greedy_match_missing.any())
                    and not bool(logit_prefill_reused_missing.any())
                    and not bool(logit_second_prefill_missing.any())
                    and not bool(logit_probe_states_discarded_missing.any())
                    and not bool(logit_labels_accessed_missing.any())
                    and bool(logit_enabled.all())
                    and not logit_shortlist_count.isna().any()
                    and bool((logit_shortlist_count == configured_logit_shortlist).all())
                    and not logit_horizon.isna().any()
                    and bool((logit_horizon == configured_logit_horizon).all())
                    and bool(logit_selected_passed.all())
                    and bool(logit_greedy_match.all())
                    and bool(logit_prefill_reused.all())
                    and not bool(logit_second_prefill.any())
                    and bool(logit_probe_states_discarded.all())
                    and not bool(logit_labels_accessed.any())
                    and not logit_reported_kl_threshold.isna().any()
                    and bool(
                        (
                            (logit_reported_kl_threshold - configured_maximum_kl).abs()
                            <= 1e-12
                        ).all()
                    )
                    and not logit_reported_difference_threshold.isna().any()
                    and bool(
                        (
                            (
                                logit_reported_difference_threshold
                                - configured_maximum_logit_difference
                            ).abs()
                            <= 1e-12
                        ).all()
                    )
                    and not logit_maximum_kl.isna().any()
                    and bool((logit_maximum_kl <= configured_maximum_kl + 1e-12).all())
                    and not logit_maximum_difference.isna().any()
                    and bool(
                        (
                            logit_maximum_difference
                            <= configured_maximum_logit_difference + 1e-12
                        ).all()
                    )
                )
            )
            requires_progressive_eviction = bool(
                candidate_overrides.get(
                    "stable_head_progressive_logit_safe_eviction",
                    False,
                )
            )
            configured_progressive_horizon = int(
                candidate_overrides.get("stable_head_progressive_eviction_horizon", 0)
            )
            configured_progressive_maximum_probes = int(
                candidate_overrides.get("stable_head_progressive_eviction_maximum_probes", 0)
            )
            configured_progressive_maximum_kl = float(
                candidate_overrides.get("stable_head_progressive_eviction_maximum_kl", 0.0)
            )
            configured_progressive_maximum_difference = float(
                candidate_overrides.get(
                    "stable_head_progressive_eviction_maximum_logit_difference",
                    0.0,
                )
            )
            configured_progressive_target_raw = candidate_overrides.get("active_budget_bytes")
            configured_progressive_target_bytes = (
                None
                if configured_progressive_target_raw is None
                else int(configured_progressive_target_raw)
            )
            progressive_boolean_fields = {
                "enabled": ("progressive_eviction_enabled", False),
                "started_from_fullkv": (
                    "progressive_eviction_started_from_fullkv",
                    False,
                ),
                "target_reached": ("progressive_eviction_target_reached", False),
                "selected_passed": (
                    "progressive_eviction_selected_candidate_passed",
                    False,
                ),
                "used_fallback": ("progressive_eviction_used_target_fallback", True),
                "accepted_greedy": (
                    "progressive_eviction_accepted_greedy_token_match",
                    False,
                ),
                "accepted_decreasing": (
                    "progressive_eviction_accepted_bytes_strictly_decrease",
                    False,
                ),
                "prefill_reused": (
                    "progressive_eviction_fullkv_prefill_reused",
                    False,
                ),
                "second_prefill": (
                    "progressive_eviction_second_prefill_performed",
                    True,
                ),
                "states_discarded": (
                    "progressive_eviction_probe_states_discarded_before_measured_decode",
                    False,
                ),
                "labels_accessed": (
                    "progressive_eviction_benchmark_labels_accessed",
                    True,
                ),
            }
            progressive_booleans: dict[str, pd.Series] = {}
            progressive_missing: dict[str, pd.Series] = {}
            for name, (column, default) in progressive_boolean_fields.items():
                values, missing = _boolean_metric(
                    candidate,
                    column,
                    missing_default=default,
                )
                progressive_booleans[name] = values
                progressive_missing[name] = missing

            def progressive_numeric(
                field: str,
                candidate_frame: pd.DataFrame = candidate,
            ) -> pd.Series:
                return pd.to_numeric(
                    candidate_frame.get(
                        f"progressive_eviction_{field}",
                        pd.Series(index=candidate_frame.index, dtype=float),
                    ),
                    errors="coerce",
                )

            progressive_horizon = progressive_numeric("horizon")
            progressive_reported_kl = progressive_numeric("maximum_kl_threshold")
            progressive_reported_difference = progressive_numeric(
                "maximum_logit_difference_threshold"
            )
            progressive_maximum_probes = progressive_numeric("maximum_probes")
            progressive_probes_used = progressive_numeric("probes_used")
            progressive_checkpoints = progressive_numeric("checkpoint_count")
            progressive_accepted = progressive_numeric("accepted_checkpoints")
            progressive_rejected = progressive_numeric("rejected_checkpoints")
            progressive_initial_bytes = progressive_numeric("initial_fullkv_logical_bytes")
            progressive_target_bytes = progressive_numeric("target_active_bytes")
            progressive_final_bytes = progressive_numeric("final_logical_active_bytes")
            progressive_accepted_kl = progressive_numeric("accepted_maximum_kl")
            progressive_accepted_difference = progressive_numeric(
                "accepted_maximum_logit_difference"
            )
            progressive_group_bytes = progressive_numeric("protected_group_bytes")
            progressive_group_target = progressive_numeric("protected_group_target_bytes")
            progressive_group_count = progressive_numeric("protected_group_count")
            progressive_group_utilization = progressive_group_bytes / progressive_group_target
            progressive_eviction_eligible = bool(
                not requires_progressive_eviction
                or (
                    progressive_protocol
                    and all(not bool(series.any()) for series in progressive_missing.values())
                    and bool(progressive_booleans["enabled"].all())
                    and bool(progressive_booleans["started_from_fullkv"].all())
                    and bool(progressive_booleans["target_reached"].all())
                    and bool(progressive_booleans["selected_passed"].all())
                    and not bool(progressive_booleans["used_fallback"].any())
                    and bool(progressive_booleans["accepted_greedy"].all())
                    and bool(progressive_booleans["accepted_decreasing"].all())
                    and bool(progressive_booleans["prefill_reused"].all())
                    and not bool(progressive_booleans["second_prefill"].any())
                    and bool(progressive_booleans["states_discarded"].all())
                    and not bool(progressive_booleans["labels_accessed"].any())
                    and not progressive_horizon.isna().any()
                    and bool((progressive_horizon == configured_progressive_horizon).all())
                    and not progressive_reported_kl.isna().any()
                    and bool(
                        (
                            (progressive_reported_kl - configured_progressive_maximum_kl).abs()
                            <= 1e-12
                        ).all()
                    )
                    and not progressive_reported_difference.isna().any()
                    and bool(
                        (
                            (
                                progressive_reported_difference
                                - configured_progressive_maximum_difference
                            ).abs()
                            <= 1e-12
                        ).all()
                    )
                    and not progressive_maximum_probes.isna().any()
                    and bool(
                        (progressive_maximum_probes == configured_progressive_maximum_probes).all()
                    )
                    and not progressive_probes_used.isna().any()
                    and bool((progressive_probes_used > 0).all())
                    and bool(
                        (progressive_probes_used <= configured_progressive_maximum_probes).all()
                    )
                    and not progressive_checkpoints.isna().any()
                    and bool((progressive_checkpoints == progressive_probes_used).all())
                    and not progressive_accepted.isna().any()
                    and bool((progressive_accepted > 0).all())
                    and not progressive_rejected.isna().any()
                    and bool(
                        (
                            progressive_accepted + progressive_rejected
                            == progressive_probes_used
                        ).all()
                    )
                    and not progressive_initial_bytes.isna().any()
                    and not progressive_target_bytes.isna().any()
                    and (
                        configured_progressive_target_bytes is None
                        or bool(
                            (
                                progressive_target_bytes
                                == configured_progressive_target_bytes
                            ).all()
                        )
                    )
                    and not progressive_final_bytes.isna().any()
                    and bool((progressive_initial_bytes > progressive_target_bytes).all())
                    and bool((progressive_final_bytes <= progressive_target_bytes).all())
                    and bool((progressive_final_bytes > 0).all())
                    and not progressive_accepted_kl.isna().any()
                    and bool(
                        (
                            progressive_accepted_kl
                            <= configured_progressive_maximum_kl + 1e-12
                        ).all()
                    )
                    and not progressive_accepted_difference.isna().any()
                    and bool(
                        (
                            progressive_accepted_difference
                            <= configured_progressive_maximum_difference + 1e-12
                        ).all()
                    )
                    and not progressive_group_target.isna().any()
                    and bool((progressive_group_target > 0).all())
                    and not progressive_group_bytes.isna().any()
                    and not progressive_group_count.isna().any()
                    and bool((progressive_group_count > 0).all())
                    and bool(
                        (
                            progressive_group_utilization + 1e-12
                            >= minimum_layout_utilization
                        ).all()
                    )
                )
            )
            candidate_results[candidate_id] = {
                "fullkv_exact_answer_agreement": answer_agreement,
                "fullkv_token_agreement": token_fidelity,
                "prompt_topk_exact_answer_agreement": statistics.fmean(
                    _normalize(candidate.at[sample_id, "answer"])
                    == _normalize(prompt.at[sample_id, "answer"])
                    for sample_id in ids
                ),
                "mean_active_kv_bytes": active_bytes,
                "mean_residual_kv_bytes": residual_bytes,
                "active_reduction_fraction_vs_prompt_topk": active_reduction,
                "active_plus_residual_reduction_fraction_vs_prompt_topk": total_reduction,
                "fullkv_answer_agreement_delta_vs_prompt_topk": fidelity_delta,
                "mean_configured_active_byte_cap": configured_cap,
                "mean_dense_attainable_budget_bytes": attainable_cap,
                "mean_dense_quantization_underfill_bytes": quantization_underfill,
                "mean_capacity_fraction_reduction_bytes": capacity_fraction_reduction,
                "mean_logical_underfill_vs_dense_budget_bytes": logical_underfill,
                "mean_final_underfill_vs_dense_budget_bytes": final_underfill,
                "mean_packed_padding_bytes": padding_bytes,
                "mean_external_stable_requested_positions": external_requested_positions,
                "mean_external_stable_selected_positions": external_selected_positions,
                "mean_external_stable_shortfall_positions": external_shortfall_positions,
                "mean_external_stable_routes_with_shortfall_count": external_shortfall_routes,
                "mean_external_stable_adaptive_swap_count": external_adaptive_swaps,
                "mean_external_stable_text_mass_preservation_fraction": (text_mass_preservation),
                "mean_external_stable_normalized_visual_evidence_gain": (normalized_visual_gain),
                "mean_external_stable_normalized_text_prompt_mass_loss": (normalized_text_loss),
                "mean_external_stable_normalized_forecast_visual_gain": (normalized_forecast_gain),
                "mean_external_stable_normalized_forecast_text_loss": (normalized_forecast_loss),
                "mean_external_stable_adaptive_swap_candidate_count": adaptive_candidates,
                "mean_external_stable_adaptive_swap_risk_rejected_count": (
                    adaptive_risk_rejections
                ),
                "mean_external_stable_routes_with_adaptive_swaps_count": adaptive_routes,
                "mean_external_stable_routes_with_risk_rejections_count": (risk_rejected_routes),
                "mean_external_stable_future_rerank_count": future_rerank_count,
                "mean_external_stable_future_prompt_mass_preservation_fraction": (
                    future_prompt_preservation
                ),
                "mean_external_stable_normalized_future_query_gain": (normalized_future_query_gain),
                "mean_external_stable_normalized_future_prompt_mass_loss": (
                    normalized_future_prompt_loss
                ),
                "mean_external_stable_future_rerank_candidate_count": (future_rerank_candidates),
                "mean_external_stable_future_rerank_rejected_count": (future_rerank_rejections),
                "mean_external_stable_future_rerank_risk_rejected_count": (
                    future_rerank_risk_rejections
                ),
                "mean_external_stable_routes_with_future_reranking_count": (future_rerank_routes),
                "mean_external_stable_routes_with_future_rerank_rejections_count": (
                    future_rejected_routes
                ),
                "mean_external_stable_routes_with_future_rerank_risk_rejections_count": (
                    future_risk_rejected_routes
                ),
                "mean_external_stable_forecast_divergence_eligible_route_count": (
                    forecast_divergence_routes
                ),
                "mean_external_stable_prompt_forecast_total_variation_min": (
                    prompt_forecast_tv_min
                ),
                "mean_external_stable_prompt_forecast_total_variation_mean": (
                    prompt_forecast_tv_mean
                ),
                "mean_external_stable_prompt_forecast_total_variation_max": (
                    prompt_forecast_tv_max
                ),
                "mean_external_stable_query_coverage_rerank_count": (query_coverage_rerank_count),
                "mean_external_stable_baseline_query_coverage": baseline_query_coverage,
                "mean_external_stable_selected_query_coverage": selected_query_coverage,
                "mean_external_stable_normalized_query_coverage_gain": (
                    normalized_query_coverage_gain
                ),
                "mean_external_stable_query_coverage_prompt_mass_preservation_fraction": (
                    query_coverage_prompt_mass_preservation
                ),
                "mean_external_stable_query_coverage_normalized_prompt_loss": (
                    query_coverage_normalized_prompt_loss
                ),
                "mean_external_stable_query_coverage_candidate_count": (
                    query_coverage_candidate_count
                ),
                "mean_external_stable_query_coverage_rejected_count": (
                    query_coverage_rejected_count
                ),
                "mean_external_stable_routes_with_query_coverage_reranking_count": (
                    query_coverage_routes
                ),
                "mean_external_stable_routes_with_query_coverage_rejections_count": (
                    query_coverage_rejected_routes
                ),
                "mean_external_stable_query_coverage_initial_reserved_count": (
                    initial_query_coverage_reserved
                ),
                "mean_external_stable_query_coverage_initial_changed_count": (
                    initial_query_coverage_changed
                ),
                "mean_external_stable_query_coverage_initial_baseline": (
                    initial_baseline_query_coverage
                ),
                "mean_external_stable_query_coverage_initial_selected": (
                    initial_selected_query_coverage
                ),
                "mean_external_stable_query_coverage_initial_normalized_gain": (
                    initial_normalized_query_coverage_gain
                ),
                "mean_external_stable_query_coverage_initial_prompt_mass_preservation_fraction": (
                    initial_prompt_mass_preservation
                ),
                "mean_external_stable_query_coverage_initial_normalized_prompt_loss": (
                    initial_normalized_prompt_loss
                ),
                "mean_external_stable_query_coverage_initial_candidate_count": (
                    initial_query_coverage_candidates
                ),
                "mean_external_stable_query_coverage_initial_reserved_bytes": (
                    initial_query_coverage_reserved_bytes
                ),
                "mean_external_stable_query_coverage_initial_allocated_bytes": (
                    initial_query_coverage_allocated_bytes
                ),
                "mean_external_stable_query_coverage_initial_cross_head_risk": (
                    initial_query_coverage_cross_head_risk
                ),
                "mean_external_stable_query_coverage_initial_proposed_normalized_gain": (
                    initial_query_coverage_proposed_gain
                ),
                "mean_external_stable_query_coverage_initial_proposed_normalized_prompt_loss": (
                    initial_query_coverage_proposed_prompt_loss
                ),
                "mean_external_stable_query_coverage_initial_proposed_output_impact_risk": (
                    initial_query_coverage_proposed_output_impact_risk
                ),
                "mean_external_stable_query_coverage_initial_output_impact_donor_count": (
                    initial_query_coverage_output_impact_donor_count
                ),
                "mean_external_stable_query_coverage_initial_output_impact_protected_count": (
                    initial_query_coverage_output_impact_protected_count
                ),
                "mean_external_stable_query_coverage_initial_output_impact_protected_bytes": (
                    initial_query_coverage_output_impact_protected_bytes
                ),
                "mean_external_stable_query_coverage_initial_output_impact_protected_score": (
                    initial_query_coverage_output_impact_protected_score
                ),
                "mean_external_stable_query_coverage_initial_layout_group_candidate_count": (
                    initial_layout_group_candidates
                ),
                "mean_external_stable_query_coverage_initial_layout_group_protected_group_count": (
                    initial_layout_group_count
                ),
                "mean_external_stable_query_coverage_initial_layout_group_protected_block_count": (
                    initial_layout_group_blocks
                ),
                "mean_external_stable_query_coverage_initial_layout_group_protected_bytes": (
                    initial_layout_group_bytes
                ),
                "mean_external_stable_query_coverage_initial_layout_group_protected_score": (
                    initial_layout_group_score
                ),
                "mean_external_stable_query_coverage_initial_layout_group_target_bytes": (
                    initial_layout_group_target_bytes
                ),
                (
                    "mean_external_stable_query_coverage_initial_layout_group_"
                    "accepted_candidate_count"
                ): (initial_layout_group_accepted_candidates),
                "mean_external_stable_query_coverage_initial_layout_group_risk_rejected_count": (
                    initial_layout_group_risk_rejections
                ),
                "mean_external_stable_query_coverage_initial_layout_group_pareto_candidate_count": (
                    initial_layout_group_pareto_candidates
                ),
                (
                    "mean_external_stable_query_coverage_initial_layout_group_"
                    "knapsack_selected_count"
                ): initial_layout_group_knapsack_selected,
                (
                    "mean_external_stable_query_coverage_initial_layout_group_"
                    "joint_donor_candidate_count"
                ): initial_layout_group_joint_donor_candidates,
                "mean_external_stable_query_coverage_initial_layout_group_normalized_query_gain": (
                    initial_layout_group_query_gain
                ),
                "mean_external_stable_query_coverage_initial_layout_group_normalized_prompt_risk": (
                    initial_layout_group_prompt_risk
                ),
                "mean_external_stable_query_coverage_initial_layout_group_output_impact_risk": (
                    initial_layout_group_output_risk
                ),
                "minimum_external_stable_query_coverage_initial_layout_group_utilization": (
                    None
                    if layout_utilization_series.isna().any()
                    else float(layout_utilization_series.min())
                ),
                "mean_external_stable_query_coverage_initial_stability_rejected_route_count": (
                    initial_query_coverage_stability_rejections
                ),
                "mean_external_stable_query_coverage_initial_global_allocation_route_count": (
                    initial_query_coverage_global_routes
                ),
                "mean_external_stable_query_coverage_initial_allocated_route_count": (
                    initial_query_coverage_allocated_routes
                ),
                "mean_external_stable_routes_with_query_coverage_initial_selection_count": (
                    initial_query_coverage_routes
                ),
                "mean_exact_prompt_mass_fraction": exact_prompt_mass_fraction,
                "mean_represented_prompt_mass_fraction": represented_prompt_mass_fraction,
                "mean_prototype_count": prototype_count,
                "mean_exact_logical_active_bytes": exact_logical_bytes,
                "mean_prototype_logical_active_bytes": prototype_logical_bytes,
                "mean_prototype_logical_fraction": prototype_fraction,
                "memory_eligible": memory_eligible,
                "fidelity_eligible": fidelity_eligible,
                "exact_dominant_eligible": exact_dominant_eligible,
                "residual_storage_eligible": residual_storage_eligible,
                "forecast_divergence_required": requires_forecast_divergence,
                "minimum_prompt_forecast_total_variation": minimum_forecast_tv,
                "forecast_divergence_eligible": forecast_divergence_eligible,
                "future_rerank_risk_activity_required": candidate_id in risk_candidate_ids,
                "future_rerank_risk_activity_eligible": True,
                "future_rerank_control_fidelity_eligible": True,
                "layout_group_activity_required": requires_layout_groups,
                "layout_group_activity_eligible": layout_group_activity_eligible,
                "layout_group_utilization_required": requires_atomic_layout_groups,
                "layout_group_utilization_eligible": layout_group_utilization_eligible,
                "pareto_knapsack_activity_required": requires_pareto_knapsack,
                "pareto_knapsack_activity_eligible": pareto_knapsack_activity_eligible,
                "joint_group_donor_activity_required": requires_joint_group_donors,
                "joint_group_donor_activity_eligible": joint_group_donor_activity_eligible,
                "short_horizon_logit_stability_required": requires_logit_stability,
                "short_horizon_logit_stability_eligible": logit_stability_eligible,
                "mean_short_horizon_logit_stability_maximum_kl": (
                    None if logit_maximum_kl.isna().all() else float(logit_maximum_kl.mean())
                ),
                "mean_short_horizon_logit_stability_maximum_logit_difference": (
                    None
                    if logit_maximum_difference.isna().all()
                    else float(logit_maximum_difference.mean())
                ),
                "short_horizon_logit_scalar_fallback_fraction": (
                    None if not requires_logit_stability else float(logit_fallback.mean())
                ),
                "progressive_logit_safe_eviction_required": requires_progressive_eviction,
                "progressive_logit_safe_eviction_eligible": progressive_eviction_eligible,
                "mean_progressive_eviction_probes_used": (
                    None
                    if progressive_probes_used.isna().all()
                    else float(progressive_probes_used.mean())
                ),
                "mean_progressive_eviction_accepted_checkpoints": (
                    None
                    if progressive_accepted.isna().all()
                    else float(progressive_accepted.mean())
                ),
                "mean_progressive_eviction_rejected_checkpoints": (
                    None
                    if progressive_rejected.isna().all()
                    else float(progressive_rejected.mean())
                ),
                "mean_progressive_eviction_accepted_maximum_kl": (
                    None
                    if progressive_accepted_kl.isna().all()
                    else float(progressive_accepted_kl.mean())
                ),
                "mean_progressive_eviction_accepted_maximum_logit_difference": (
                    None
                    if progressive_accepted_difference.isna().all()
                    else float(progressive_accepted_difference.mean())
                ),
                "minimum_progressive_eviction_group_utilization": (
                    None
                    if progressive_group_utilization.isna().all()
                    else float(progressive_group_utilization.min())
                ),
                "gate_eligible": False,
            }
        if risk_gate_protocol:
            assert risk_control_arm is not None
            control_agreement = float(
                candidate_results[risk_control_arm]["fullkv_exact_answer_agreement"]
            )
            for candidate_id in candidate_ids:
                result = cast("dict[str, Any]", candidate_results[candidate_id])
                if candidate_id in risk_candidate_ids:
                    accepted = result["mean_external_stable_future_rerank_count"]
                    rejected = result["mean_external_stable_future_rerank_risk_rejected_count"]
                    result["future_rerank_risk_activity_eligible"] = bool(
                        accepted is not None
                        and float(accepted) > 0
                        and rejected is not None
                        and float(rejected) > 0
                    )
                    result["future_rerank_control_fidelity_eligible"] = bool(
                        float(result["fullkv_exact_answer_agreement"]) + risk_control_margin
                        >= control_agreement
                    )
        for candidate_id in candidate_ids:
            result = cast("dict[str, Any]", candidate_results[candidate_id])
            result["gate_eligible"] = bool(
                result["memory_eligible"]
                and result["fidelity_eligible"]
                and result["exact_dominant_eligible"]
                and result["residual_storage_eligible"]
                and result["forecast_divergence_eligible"]
                and result["future_rerank_risk_activity_eligible"]
                and result["future_rerank_control_fidelity_eligible"]
                and result["layout_group_activity_eligible"]
                and result["layout_group_utilization_eligible"]
                and result["pareto_knapsack_activity_eligible"]
                and result["joint_group_donor_activity_eligible"]
                and result["short_horizon_logit_stability_eligible"]
                and result["progressive_logit_safe_eviction_eligible"]
            )
        decomposition_result: dict[str, Any] | None = None
        if decomposition:
            prototype_frame = by_arm[str(prototype_decomposition_arm)]
            exact_frame = by_arm[str(exact_decomposition_arm)]
            packed_equal = all(
                int(prototype_frame.at[sample_id, "active_kv_bytes"])
                == int(exact_frame.at[sample_id, "active_kv_bytes"])
                for sample_id in ids
            )
            if (
                decomposition.get("require_equal_packed_active_bytes_within_task") is True
                and not packed_equal
            ):
                raise RuntimeError(
                    f"{task} prototype/exact decomposition did not use identical packed bytes"
                )
            prototype_result = candidate_results[str(prototype_decomposition_arm)]
            exact_result = candidate_results[str(exact_decomposition_arm)]
            decomposition_result = {
                "prototype_arm": prototype_decomposition_arm,
                "exact_only_arm": exact_decomposition_arm,
                "paired_packed_active_bytes_equal": packed_equal,
                "mean_packed_active_bytes": float(prototype_frame["active_kv_bytes"].mean()),
                "prototype_fullkv_answer_agreement": prototype_result[
                    "fullkv_exact_answer_agreement"
                ],
                "exact_only_fullkv_answer_agreement": exact_result["fullkv_exact_answer_agreement"],
                "exact_minus_prototype_answer_agreement": (
                    float(exact_result["fullkv_exact_answer_agreement"])
                    - float(prototype_result["fullkv_exact_answer_agreement"])
                ),
                "prototype_fullkv_token_agreement": prototype_result["fullkv_token_agreement"],
                "exact_only_fullkv_token_agreement": exact_result["fullkv_token_agreement"],
                "exact_minus_prototype_token_agreement": (
                    float(exact_result["fullkv_token_agreement"])
                    - float(prototype_result["fullkv_token_agreement"])
                ),
                "prototype_logical_bytes": prototype_result["mean_prototype_logical_active_bytes"],
                "exact_only_prototype_logical_bytes": exact_result[
                    "mean_prototype_logical_active_bytes"
                ],
                "prototype_residual_bytes": prototype_result["mean_residual_kv_bytes"],
                "exact_only_residual_bytes": exact_result["mean_residual_kv_bytes"],
            }
        task_results[task] = {
            "sample_count": len(ids),
            "prompt_topk_fullkv_exact_answer_agreement": prompt_full_answer,
            "prompt_topk_fullkv_token_agreement": prompt_full_token,
            "candidates": candidate_results,
            "same_byte_decomposition": decomposition_result,
        }

    for candidate_id in candidate_ids:
        records = [
            cast("dict[str, Any]", result["candidates"])[candidate_id]
            for result in task_results.values()
        ]
        memory_eligible = all(bool(record["memory_eligible"]) for record in records)
        fidelity_eligible = all(bool(record["fidelity_eligible"]) for record in records)
        exact_dominant_eligible = all(bool(record["exact_dominant_eligible"]) for record in records)
        residual_storage_eligible = all(
            bool(record["residual_storage_eligible"]) for record in records
        )
        forecast_divergence_eligible = all(
            bool(record["forecast_divergence_eligible"]) for record in records
        )
        future_rerank_risk_activity_eligible = all(
            bool(record["future_rerank_risk_activity_eligible"]) for record in records
        )
        future_rerank_control_fidelity_eligible = all(
            bool(record["future_rerank_control_fidelity_eligible"]) for record in records
        )
        layout_group_activity_eligible = all(
            bool(record["layout_group_activity_eligible"]) for record in records
        )
        layout_group_activity_required = any(
            bool(record["layout_group_activity_required"]) for record in records
        )
        layout_group_utilization_eligible = all(
            bool(record["layout_group_utilization_eligible"]) for record in records
        )
        layout_group_utilization_required = any(
            bool(record["layout_group_utilization_required"]) for record in records
        )
        pareto_knapsack_activity_eligible = all(
            bool(record["pareto_knapsack_activity_eligible"]) for record in records
        )
        pareto_knapsack_activity_required = any(
            bool(record["pareto_knapsack_activity_required"]) for record in records
        )
        joint_group_donor_activity_eligible = all(
            bool(record["joint_group_donor_activity_eligible"]) for record in records
        )
        joint_group_donor_activity_required = any(
            bool(record["joint_group_donor_activity_required"]) for record in records
        )
        short_horizon_logit_stability_eligible = all(
            bool(record["short_horizon_logit_stability_eligible"]) for record in records
        )
        short_horizon_logit_stability_required = any(
            bool(record["short_horizon_logit_stability_required"]) for record in records
        )
        progressive_logit_safe_eviction_eligible = all(
            bool(record["progressive_logit_safe_eviction_eligible"]) for record in records
        )
        progressive_logit_safe_eviction_required = any(
            bool(record["progressive_logit_safe_eviction_required"]) for record in records
        )
        candidate_ranking.append(
            {
                "candidate_id": candidate_id,
                "memory_eligible": memory_eligible,
                "fidelity_eligible": fidelity_eligible,
                "exact_dominant_eligible": exact_dominant_eligible,
                "residual_storage_eligible": residual_storage_eligible,
                "forecast_divergence_eligible": forecast_divergence_eligible,
                "future_rerank_risk_activity_required": candidate_id in risk_candidate_ids,
                "future_rerank_risk_activity_eligible": (future_rerank_risk_activity_eligible),
                "future_rerank_control_fidelity_eligible": (
                    future_rerank_control_fidelity_eligible
                ),
                "layout_group_activity_required": layout_group_activity_required,
                "layout_group_activity_eligible": layout_group_activity_eligible,
                "layout_group_utilization_required": layout_group_utilization_required,
                "layout_group_utilization_eligible": layout_group_utilization_eligible,
                "pareto_knapsack_activity_required": pareto_knapsack_activity_required,
                "pareto_knapsack_activity_eligible": pareto_knapsack_activity_eligible,
                "joint_group_donor_activity_required": joint_group_donor_activity_required,
                "joint_group_donor_activity_eligible": joint_group_donor_activity_eligible,
                "short_horizon_logit_stability_required": (
                    short_horizon_logit_stability_required
                ),
                "short_horizon_logit_stability_eligible": (
                    short_horizon_logit_stability_eligible
                ),
                "progressive_logit_safe_eviction_required": (
                    progressive_logit_safe_eviction_required
                ),
                "progressive_logit_safe_eviction_eligible": (
                    progressive_logit_safe_eviction_eligible
                ),
                "gate_eligible": (
                    memory_eligible
                    and fidelity_eligible
                    and exact_dominant_eligible
                    and residual_storage_eligible
                    and forecast_divergence_eligible
                    and future_rerank_risk_activity_eligible
                    and future_rerank_control_fidelity_eligible
                    and layout_group_activity_eligible
                    and layout_group_utilization_eligible
                    and pareto_knapsack_activity_eligible
                    and joint_group_donor_activity_eligible
                    and short_horizon_logit_stability_eligible
                    and progressive_logit_safe_eviction_eligible
                ),
                "worst_task_fullkv_exact_answer_agreement": min(
                    float(record["fullkv_exact_answer_agreement"]) for record in records
                ),
                "macro_fullkv_token_agreement": statistics.fmean(
                    float(record["fullkv_token_agreement"]) for record in records
                ),
                "worst_task_active_plus_residual_reduction": min(
                    float(record["active_plus_residual_reduction_fraction_vs_prompt_topk"])
                    for record in records
                ),
                "worst_task_fullkv_answer_agreement_delta_vs_prompt_topk": min(
                    float(record["fullkv_answer_agreement_delta_vs_prompt_topk"])
                    for record in records
                ),
                "mean_configured_active_byte_cap": statistics.fmean(
                    float(record["mean_configured_active_byte_cap"])
                    for record in records
                    if record["mean_configured_active_byte_cap"] is not None
                )
                if any(record["mean_configured_active_byte_cap"] is not None for record in records)
                else None,
                "mean_dense_attainable_budget_bytes": statistics.fmean(
                    float(record["mean_dense_attainable_budget_bytes"])
                    for record in records
                    if record["mean_dense_attainable_budget_bytes"] is not None
                )
                if any(
                    record["mean_dense_attainable_budget_bytes"] is not None for record in records
                )
                else None,
                "mean_final_underfill_vs_dense_budget_bytes": statistics.fmean(
                    float(record["mean_final_underfill_vs_dense_budget_bytes"])
                    for record in records
                    if record["mean_final_underfill_vs_dense_budget_bytes"] is not None
                )
                if any(
                    record["mean_final_underfill_vs_dense_budget_bytes"] is not None
                    for record in records
                )
                else None,
                "mean_packed_padding_bytes": statistics.fmean(
                    float(record["mean_packed_padding_bytes"])
                    for record in records
                    if record["mean_packed_padding_bytes"] is not None
                )
                if any(record["mean_packed_padding_bytes"] is not None for record in records)
                else None,
                "mean_external_stable_shortfall_positions": statistics.fmean(
                    float(record["mean_external_stable_shortfall_positions"])
                    for record in records
                    if record["mean_external_stable_shortfall_positions"] is not None
                )
                if any(
                    record["mean_external_stable_shortfall_positions"] is not None
                    for record in records
                )
                else None,
                "mean_external_stable_routes_with_shortfall_count": statistics.fmean(
                    float(record["mean_external_stable_routes_with_shortfall_count"])
                    for record in records
                    if record["mean_external_stable_routes_with_shortfall_count"] is not None
                )
                if any(
                    record["mean_external_stable_routes_with_shortfall_count"] is not None
                    for record in records
                )
                else None,
                "minimum_layout_group_reservation_utilization": min(
                    float(
                        record[
                            "minimum_external_stable_query_coverage_initial_layout_group_utilization"
                        ]
                    )
                    for record in records
                    if record[
                        "minimum_external_stable_query_coverage_initial_layout_group_utilization"
                    ]
                    is not None
                )
                if any(
                    record[
                        "minimum_external_stable_query_coverage_initial_layout_group_utilization"
                    ]
                    is not None
                    for record in records
                )
                else None,
                "mean_layout_group_accepted_candidate_count": statistics.fmean(
                    float(
                        record[
                            "mean_external_stable_query_coverage_initial_layout_group_accepted_candidate_count"
                        ]
                    )
                    for record in records
                    if record[
                        "mean_external_stable_query_coverage_initial_layout_group_accepted_candidate_count"
                    ]
                    is not None
                )
                if any(
                    record[
                        "mean_external_stable_query_coverage_initial_layout_group_accepted_candidate_count"
                    ]
                    is not None
                    for record in records
                )
                else None,
                "mean_layout_group_risk_rejected_count": statistics.fmean(
                    float(
                        record[
                            "mean_external_stable_query_coverage_initial_layout_group_risk_rejected_count"
                        ]
                    )
                    for record in records
                    if record[
                        "mean_external_stable_query_coverage_initial_layout_group_risk_rejected_count"
                    ]
                    is not None
                )
                if any(
                    record[
                        "mean_external_stable_query_coverage_initial_layout_group_risk_rejected_count"
                    ]
                    is not None
                    for record in records
                )
                else None,
                "mean_exact_prompt_mass_fraction": statistics.fmean(
                    float(record["mean_exact_prompt_mass_fraction"])
                    for record in records
                    if record["mean_exact_prompt_mass_fraction"] is not None
                )
                if any(record["mean_exact_prompt_mass_fraction"] is not None for record in records)
                else None,
                "mean_represented_prompt_mass_fraction": statistics.fmean(
                    float(record["mean_represented_prompt_mass_fraction"])
                    for record in records
                    if record["mean_represented_prompt_mass_fraction"] is not None
                )
                if any(
                    record["mean_represented_prompt_mass_fraction"] is not None
                    for record in records
                )
                else None,
                "mean_prototype_count": statistics.fmean(
                    float(record["mean_prototype_count"])
                    for record in records
                    if record["mean_prototype_count"] is not None
                )
                if any(record["mean_prototype_count"] is not None for record in records)
                else None,
                "worst_task_prototype_logical_fraction": max(
                    float(record["mean_prototype_logical_fraction"])
                    for record in records
                    if record["mean_prototype_logical_fraction"] is not None
                )
                if any(record["mean_prototype_logical_fraction"] is not None for record in records)
                else None,
                "mean_layout_group_candidate_count": statistics.fmean(
                    float(
                        record[
                            "mean_external_stable_query_coverage_initial_layout_group_candidate_count"
                        ]
                    )
                    for record in records
                    if record[
                        "mean_external_stable_query_coverage_initial_layout_group_candidate_count"
                    ]
                    is not None
                )
                if any(
                    record[
                        "mean_external_stable_query_coverage_initial_layout_group_candidate_count"
                    ]
                    is not None
                    for record in records
                )
                else None,
                "mean_layout_group_protected_group_count": statistics.fmean(
                    float(
                        record[
                            "mean_external_stable_query_coverage_initial_layout_group_protected_group_count"
                        ]
                    )
                    for record in records
                    if record[
                        "mean_external_stable_query_coverage_initial_layout_group_protected_group_count"
                    ]
                    is not None
                )
                if any(
                    record[
                        "mean_external_stable_query_coverage_initial_layout_group_protected_group_count"
                    ]
                    is not None
                    for record in records
                )
                else None,
                "mean_layout_group_protected_block_count": statistics.fmean(
                    float(
                        record[
                            "mean_external_stable_query_coverage_initial_layout_group_protected_block_count"
                        ]
                    )
                    for record in records
                    if record[
                        "mean_external_stable_query_coverage_initial_layout_group_protected_block_count"
                    ]
                    is not None
                )
                if any(
                    record[
                        "mean_external_stable_query_coverage_initial_layout_group_protected_block_count"
                    ]
                    is not None
                    for record in records
                )
                else None,
                "mean_progressive_eviction_probes_used": statistics.fmean(
                    float(record["mean_progressive_eviction_probes_used"])
                    for record in records
                    if record["mean_progressive_eviction_probes_used"] is not None
                )
                if any(
                    record["mean_progressive_eviction_probes_used"] is not None
                    for record in records
                )
                else None,
                "mean_progressive_eviction_accepted_checkpoints": statistics.fmean(
                    float(record["mean_progressive_eviction_accepted_checkpoints"])
                    for record in records
                    if record["mean_progressive_eviction_accepted_checkpoints"] is not None
                )
                if any(
                    record["mean_progressive_eviction_accepted_checkpoints"] is not None
                    for record in records
                )
                else None,
                "mean_progressive_eviction_rejected_checkpoints": statistics.fmean(
                    float(record["mean_progressive_eviction_rejected_checkpoints"])
                    for record in records
                    if record["mean_progressive_eviction_rejected_checkpoints"] is not None
                )
                if any(
                    record["mean_progressive_eviction_rejected_checkpoints"] is not None
                    for record in records
                )
                else None,
                "minimum_progressive_eviction_group_utilization": min(
                    float(record["minimum_progressive_eviction_group_utilization"])
                    for record in records
                    if record["minimum_progressive_eviction_group_utilization"] is not None
                )
                if any(
                    record["minimum_progressive_eviction_group_utilization"] is not None
                    for record in records
                )
                else None,
            }
        )
    eligible = [record for record in candidate_ranking if record["gate_eligible"]]
    if not eligible:
        selected: dict[str, Any] | None = None
    else:
        selected = sorted(
            eligible,
            key=lambda record: (
                -float(record["worst_task_fullkv_exact_answer_agreement"]),
                -float(record["macro_fullkv_token_agreement"]),
                -float(record["worst_task_active_plus_residual_reduction"]),
                str(record["candidate_id"]),
            ),
        )[0]
    selected_id = None if selected is None else str(selected["candidate_id"])
    result = {
        "schema_version": 1,
        "experiment_id": spec["experiment_id"],
        "classification": spec["classification"],
        "canonical_eligible": True,
        "labels_accessed": False,
        "official_task_scores_accessed": False,
        "common_cache_policy": common_cache_policy,
        "source_snapshot_sha256": next(iter(snapshots)),
        "task_results": task_results,
        "candidate_ranking": candidate_ranking,
        "selected_candidate": selected_id,
        "memory_and_fidelity_eligible_candidate_found": selected is not None,
        "all_configured_gates_eligible_candidate_found": selected is not None,
        "forecast_divergence_analysis": divergence_protocol or None,
        "future_rerank_risk_analysis": risk_gate_protocol or None,
        "short_horizon_logit_stability_analysis": logit_protocol or None,
        "progressive_logit_safe_eviction_analysis": progressive_protocol or None,
    }
    (analysis_root / "label_blind_calibration.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pd.DataFrame(candidate_ranking).to_parquet(
        analysis_root / "label_blind_candidate_ranking.parquet", index=False
    )
    if selected_id is not None:
        selected_spec = next(
            candidate for candidate in candidates if str(candidate["id"]) == selected_id
        )
        ledger_records = {
            task: {
                "path": telemetry["ledger"],
                "sha256": telemetry["ledger_sha256"],
            }
            for task, telemetry in telemetry_by_task.items()
        }
        frozen = {
            "schema_version": 1,
            "classification": spec.get(
                "frozen_candidate_classification",
                "frozen_label_blind_calibrated_candidate",
            ),
            "selected_candidate": selected_id,
            "candidate": selected_spec,
            "common_cache_policy": common_cache_policy,
            "selection_objective": spec["calibration_objective"],
            "labels_accessed": False,
            "official_task_scores_accessed": False,
            "calibration_spec": str(spec_path.resolve()),
            "calibration_spec_sha256": _sha(spec_path),
            "calibration_source_snapshot_sha256": next(iter(snapshots)),
            "calibration_ledgers": ledger_records,
            "calibration_result_sha256": hashlib.sha256(
                (analysis_root / "label_blind_calibration.json").read_bytes()
            ).hexdigest(),
            "candidate_config": str(_resolve(package_root, spec["candidate_config"])),
            "candidate_config_sha256": _sha(_resolve(package_root, spec["candidate_config"])),
        }
        parent_seal_raw = spec.get("parent_negative_result_seal")
        if parent_seal_raw is not None:
            parent_seal = _resolve(package_root, parent_seal_raw)
            frozen["parent_negative_result_seal"] = str(parent_seal)
            frozen["parent_negative_result_seal_sha256"] = _sha(parent_seal)
        _write_frozen(frozen_output, frozen)

    report = [
        f"# {spec.get('report_title', f'{run_prefix.upper()} label-blind calibration')}",
        "",
        "Candidate selection used FullKV output/token fidelity and measured KV bytes "
        "only. Benchmark references and official task scores were excluded from "
        "calibration telemetry.",
        "",
        f"- Common block policy: `{common_cache_policy}`",
        f"- Selected candidate: **{selected_id or 'NONE'}**",
        "- Memory eligibility required at least 30% lower active bytes and at least "
        "30% lower active-plus-residual bytes than Prompt Top-K on every calibration task.",
        f"- Fidelity eligibility required FullKV answer agreement no more than "
        f"{fidelity_margin:.2%} below Prompt Top-K on every calibration task.",
        "- Byte-cap telemetry separates configured, dense-attainable, planner, logical, "
        "padding, and final packed bytes; `physical_capacity_fraction` is not allowed "
        "to silently apply a second cap.",
        (
            "- Exact-dominant eligibility limits prototype logical bytes to "
            f"{maximum_prototype_fraction:.2%} on every task."
            if maximum_prototype_fraction is not None
            else "- Exact-dominant eligibility was not configured for this analysis."
        ),
        (
            "- Residual-storage eligibility requires exactly zero residual bytes."
            if require_zero_residual_bytes
            else "- Zero residual storage was not configured as a separate gate."
        ),
        (
            "- Fail-closed forecast divergence required at least one eligible stable "
            "route on every sample."
            if divergence_protocol
            else "- No separate forecast-divergence gate was configured."
        ),
        (
            "- Risk-gated rerank arms required both accepted and risk-rejected swaps on "
            "every task, and could not fall below the no-rerank control in FullKV "
            "answer agreement."
            if risk_gate_protocol
            else "- No separate future-rerank risk/activity gate was configured."
        ),
        (
            "- Coherent layout-group candidates must protect at least one multi-block "
            "group with positive bytes and score on every task when enabled."
            if any(record["layout_group_activity_required"] for record in candidate_ranking)
            else "- No coherent layout-group activity gate was configured."
        ),
        (
            "- Atomic layout-group reservation must meet its preregistered minimum "
            "utilization on every sample."
            if any(record["layout_group_utilization_required"] for record in candidate_ranking)
            else "- No atomic layout-group utilization gate was configured."
        ),
        (
            "- Pareto-knapsack candidates require positive Pareto items, selected "
            "knapsack groups, and audited non-group output-impact donors on every task."
            if any(record["pareto_knapsack_activity_required"] for record in candidate_ranking)
            else "- No Pareto-knapsack mechanism-activity gate was configured."
        ),
        (
            "- Joint group-and-donor candidates must be constructed on every task before "
            "Pareto filtering and atomic allocation."
            if any(record["joint_group_donor_activity_required"] for record in candidate_ranking)
            else "- No joint group-and-donor mechanism-activity gate was configured."
        ),
        (
            "- Short-horizon logit stability requires every selected cache to preserve "
            "the FullKV greedy token and remain below both preregistered logit thresholds."
            if any(
                record["short_horizon_logit_stability_required"]
                for record in candidate_ranking
            )
            else "- No short-horizon FullKV-logit gate was configured."
        ),
        (
            "- Progressive logit-safe eviction must start from FullKV, reach the fixed "
            "memory target, and pass the greedy-token, KL, and absolute-logit gates at "
            "every accepted checkpoint."
            if any(
                record["progressive_logit_safe_eviction_required"]
                for record in candidate_ranking
            )
            else "- No progressive FullKV-to-target eviction gate was configured."
        ),
        "",
        "| Candidate | Memory | Fidelity | Exact-dominant | Zero residual | "
        "Forecast divergence | "
        "Risk activity | Control safety | Layout groups | Group utilization | Pareto knapsack | "
        "Joint donors | Logit stability | Progressive eviction | "
        "Worst answer agreement | "
        "Macro token agreement | Worst total reduction | Mean configured cap | "
        "Mean attainable cap | Mean final underfill | Mean packed padding | "
        "Mean stable-head shortfall | Exact prompt mass | Represented prompt mass | "
        "Mean prototypes |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in candidate_ranking:
        stable_shortfall = record["mean_external_stable_shortfall_positions"]
        stable_shortfall_text = (
            "n/a" if stable_shortfall is None else f"{float(stable_shortfall):.2f}"
        )
        prototype_count = record["mean_prototype_count"]
        prototype_count_text = "n/a" if prototype_count is None else f"{float(prototype_count):.2f}"
        report.append(
            f"| {record['candidate_id']} | {'yes' if record['memory_eligible'] else 'no'} | "
            f"{'yes' if record['fidelity_eligible'] else 'no'} | "
            f"{'yes' if record['exact_dominant_eligible'] else 'no'} | "
            f"{'yes' if record['residual_storage_eligible'] else 'no'} | "
            f"{'yes' if record['forecast_divergence_eligible'] else 'no'} | "
            f"{'yes' if record['future_rerank_risk_activity_eligible'] else 'no'} | "
            f"{'yes' if record['future_rerank_control_fidelity_eligible'] else 'no'} | "
            f"{'yes' if record['layout_group_activity_eligible'] else 'no'} | "
            f"{'yes' if record['layout_group_utilization_eligible'] else 'no'} | "
            f"{'yes' if record['pareto_knapsack_activity_eligible'] else 'no'} | "
            f"{'yes' if record['joint_group_donor_activity_eligible'] else 'no'} | "
            f"{'yes' if record['short_horizon_logit_stability_eligible'] else 'no'} | "
            f"{'yes' if record['progressive_logit_safe_eviction_eligible'] else 'no'} | "
            f"{float(record['worst_task_fullkv_exact_answer_agreement']):.2%} | "
            f"{float(record['macro_fullkv_token_agreement']):.2%} | "
            f"{float(record['worst_task_active_plus_residual_reduction']):.2%} | "
            f"{_format_mb(record['mean_configured_active_byte_cap'])} | "
            f"{_format_mb(record['mean_dense_attainable_budget_bytes'])} | "
            f"{_format_mb(record['mean_final_underfill_vs_dense_budget_bytes'])} | "
            f"{_format_mb(record['mean_packed_padding_bytes'])} | "
            f"{stable_shortfall_text} | "
            f"{_format_fraction(record['mean_exact_prompt_mass_fraction'])} | "
            f"{_format_fraction(record['mean_represented_prompt_mass_fraction'])} | "
            f"{prototype_count_text} |"
        )
    (analysis_root / "label_blind_calibration.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--frozen-output", required=True, type=Path)
    parser.add_argument("--run-prefix", default="v74")
    args = parser.parse_args()
    result = analyze(
        args.spec.resolve(),
        args.output_root.resolve(),
        args.frozen_output.resolve(),
        run_prefix=str(args.run_prefix),
    )
    print(
        json.dumps(
            {
                "selected_candidate": result["selected_candidate"],
                "labels_accessed": False,
            }
        )
    )
    return 0 if result["selected_candidate"] is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
