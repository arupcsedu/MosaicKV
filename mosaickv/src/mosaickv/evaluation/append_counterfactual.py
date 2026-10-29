"""Evaluation-only history-loss intervention, not a byte-matched method.

The finite incremental cache is append-only over the complete registered
teacher trajectory when its capacity equals target_count - 1. Extra lifetime
budget preserves the original prompt allocation; runtime evidence must still
verify identical selected prompt positions and the pre-eviction logits.
"""

from dataclasses import replace
from math import isfinite

from mosaickv.config import RunConfig


def append_config(cfg: RunConfig, target_count: int, position_costs: tuple[int, ...]) -> RunConfig:
    bounded = cfg.bounded_decode
    if (
        bounded is None
        or bounded.incremental_history is None
        or not bounded.incremental_history.fixed_prompt_reservation
        or bounded.incremental_history.policy != "sink_recent"
        or not position_costs
        or any(type(c) is not int or c < 1 for c in position_costs)
        or len(set(position_costs)) != 1
        or type(target_count) is not int
        or target_count - 1 <= bounded.generated_capacity
        or cfg.cache.budget_unit.value != "bytes"
        or cfg.repair.enabled
        or cfg.prototypes.enabled
        or cfg.residual.enabled
        or (cfg.forecasting.enabled and cfg.forecasting.draft_steps)
    ):
        raise ValueError("fixed-prompt FIFO, longer target and uniform layer costs required")
    history = target_count - 1
    delta = (history - bounded.generated_capacity) * (sum(position_costs) + max(position_costs))
    return replace(
        cfg,
        cache=replace(cfg.cache, budget_value=cfg.cache.budget_value + delta),
        bounded_decode=replace(bounded, generated_capacity=history),
    )


def verify_intervention(detail, fifo_detail, trace, fifo_trace, *, history_capacity):
    """Fail closed on changed prompt, target, logical positions or pre-ring output."""
    records, fifo = detail["records"], fifo_detail["records"]
    if not records or len(records) != len(fifo) or history_capacity != len(records) - 1:
        raise ValueError("complete paired teacher trajectory and append capacity required")
    for field in (
        "selected_blocks",
        "selected_layer_prompt_capacities",
        "packed_prefill_active_kv_bytes",
        "packed_padding_kv_bytes",
        "generated_token_ids",
        "source_kv_layout",
    ):
        if field not in trace or trace[field] != fifo_trace.get(field):
            raise ValueError(f"same selected prompt/target invariant failed: {field}")
    if trace["packed_padding_kv_bytes"] != 0:
        raise ValueError("unexpected physical padding")
    if detail["source_prompt_logical_length"] != fifo_detail["source_prompt_logical_length"]:
        raise ValueError("source logical length changed")
    boundary = next(
        (r["prediction_position"] for r in fifo if any(r["generated_evictions_visible_per_layer"])),
        None,
    )
    if boundary is None:
        raise ValueError("FIFO control never evicted history")
    for i, (a, b) in enumerate(zip(records, fifo, strict=True), 1):
        for field in (
            "prediction_position",
            "target_token_id",
            "next_logical_position_after_forward",
        ):
            if a[field] != b[field]:
                raise ValueError(f"teacher alignment changed: {field}")
        if a["prediction_position"] != i or not isfinite(a["nll"]) or not isfinite(b["nll"]):
            raise ValueError("invalid per-token record")
        if any(a["generated_evictions_visible_per_layer"]) or any(
            a["prompt_evictions_visible_per_layer"]
        ):
            raise ValueError("append-only counterfactual evicted cache entries")
        if any(a["removed_generated_positions_in_this_forward"]):
            raise ValueError("append-only history removed positions")
        if i < boundary and (a["nll"] != b["nll"] or a["greedy_token_id"] != b["greedy_token_id"]):
            raise ValueError("pre-eviction NLL/greedy output changed")
    old, new = fifo_trace["cache_lifecycle"], trace["cache_lifecycle"]
    if old["initial_active_kv_storage_bytes"] != new["initial_active_kv_storage_bytes"]:
        raise ValueError("initial prompt storage changed")
    if new["final_active_kv_storage_bytes"] <= old["final_active_kv_storage_bytes"]:
        raise ValueError("counterfactual did not retain additional history storage")
    return dict(
        same_selected_prompt=True,
        identical_pre_eviction_nll_and_greedy=True,
        no_prompt_or_generated_evictions=True,
        first_fifo_affected_prediction=boundary,
        identical_prefix_prediction_count=boundary - 1,
        generated_capacity=history_capacity,
        target_count=len(records),
        byte_matched_comparison=False,
        whole_request_peak_certified=False,
    )
