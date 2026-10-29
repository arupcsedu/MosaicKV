"""Synthetic intervention controls; no model/dataset downloads or quality claims."""

from copy import deepcopy
from dataclasses import replace

import pytest

from mosaickv.config import (
    BoundedDecodeConfig,
    CacheConfig,
    DatasetConfig,
    ExecutionConfig,
    ForecastingConfig,
    GenerationConfig,
    GraphConfig,
    IncrementalHistoryConfig,
    ModelConfig,
    PrototypeConfig,
    RepairConfig,
    ResidualConfig,
    RunConfig,
    SelectionConfig,
)
from mosaickv.evaluation.append_counterfactual import append_config, verify_intervention
from mosaickv.selection.profile_lifetime import allocate_profile_lifetime
from mosaickv.types import Backend, BudgetUnit, MosaicKVMethod, Precision, RepairPolicy


def config():
    return RunConfig(
        model=ModelConfig("llava-hf/llava-1.5-7b-hf", "a" * 40, Precision.BF16),
        dataset=DatasetConfig("synthetic", "b" * 40, "test"),
        execution=ExecutionConfig(Backend.HUGGINGFACE, "eager"),
        generation=GenerationConfig(),
        cache=CacheConfig(289 * 2**20, BudgetUnit.BYTES, 0.9999999999999999, 1),
        method=MosaicKVMethod.PROMPT_ATTENTION_TOPK,
        forecasting=ForecastingConfig(enabled=False, draft_steps=0),
        graph=GraphConfig(enabled=False),
        selection=SelectionConfig(enabled=False),
        prototypes=PrototypeConfig(enabled=False),
        residual=ResidualConfig(enabled=False),
        repair=RepairConfig(enabled=False, policy=RepairPolicy.NONE),
        bounded_decode=BoundedDecodeConfig(
            256,
            4,
            incremental_history=IncrementalHistoryConfig(
                policy="sink_recent", fixed_prompt_reservation=True, borrow_prefill_snapshot=True
            ),
        ),
    )


def test_adds_only_geometry_required_history_and_scratch_without_changing_prompt():
    cfg, costs = config(), (16384,) * 32
    candidate = append_config(cfg, 593, costs)
    assert candidate.cache.budget_value == int(462.25 * 2**20)
    assert candidate.bounded_decode.generated_capacity == 592
    assert replace(candidate, cache=cfg.cache, bounded_decode=cfg.bounded_decode) == cfg
    ratios = [0.05 + i * 0.02 for i in range(32)]
    before, after = [
        allocate_profile_lifetime(
            ratios,
            [604] * 32,
            [1] * 32,
            costs,
            cap_bytes=c.cache.budget_value,
            generated_capacity=c.bounded_decode.generated_capacity,
        )
        for c in (cfg, candidate)
    ]
    assert before.prompt_positions == after.prompt_positions
    assert before.prompt_bytes == after.prompt_bytes
    assert after.history_bytes - before.history_bytes == 336 * sum(costs)
    assert after.growth_scratch_bytes - before.growth_scratch_bytes == 336 * max(costs)


@pytest.mark.parametrize(
    "target,costs", [(257, (16, 16)), (593, (16, 32)), (593, ()), (593, (True,))]
)
def test_invalid_geometry_is_not_silently_adjusted(target, costs):
    with pytest.raises(ValueError):
        append_config(config(), target, costs)


def captures():
    records = [
        dict(
            prediction_position=i,
            target_token_id=3,
            greedy_token_id=3,
            nll=1.0,
            next_logical_position_after_forward=10 + i,
            generated_evictions_visible_per_layer=[0],
            prompt_evictions_visible_per_layer=[0],
            removed_generated_positions_in_this_forward=[[]],
        )
        for i in range(1, 7)
    ]
    append = dict(records=deepcopy(records), source_prompt_logical_length=10)
    fifo = dict(records=deepcopy(records), source_prompt_logical_length=10)
    for r in fifo["records"][4:]:
        r["generated_evictions_visible_per_layer"] = [1]
    trace = dict(
        selected_blocks=[1, 2],
        selected_layer_prompt_capacities=[2],
        packed_prefill_active_kv_bytes=32,
        packed_padding_kv_bytes=0,
        generated_token_ids=[3] * 6,
        source_kv_layout=[{"shape": [1, 2, 10, 4]}],
        cache_lifecycle=dict(initial_active_kv_storage_bytes=32, final_active_kv_storage_bytes=112),
    )
    old_trace = deepcopy(trace)
    old_trace["cache_lifecycle"]["final_active_kv_storage_bytes"] = 80
    return append, fifo, trace, old_trace


def test_accepts_only_post_fifo_boundary_loss_changes():
    a, b, t, u = captures()
    a["records"][4]["nll"] = 0.75
    checks = verify_intervention(a, b, t, u, history_capacity=5)
    assert checks["first_fifo_affected_prediction"] == 5
    assert checks["identical_prefix_prediction_count"] == 4
    assert checks["byte_matched_comparison"] is False


@pytest.mark.parametrize(
    "mutation", ["prompt", "target", "early_nll", "logical", "eviction", "padding", "storage"]
)
def test_rejects_confounded_interventions(mutation):
    a, b, t, u = captures()
    if mutation == "prompt":
        t["selected_blocks"] = [2, 3]
    elif mutation == "target":
        a["records"][5]["target_token_id"] = 7
    elif mutation == "early_nll":
        a["records"][0]["nll"] = 1.01
    elif mutation == "logical":
        a["records"][3]["next_logical_position_after_forward"] += 1
    elif mutation == "eviction":
        a["records"][5]["generated_evictions_visible_per_layer"] = [1]
    elif mutation == "padding":
        t["packed_padding_kv_bytes"] = 1
    else:
        t["cache_lifecycle"]["final_active_kv_storage_bytes"] = 80
    with pytest.raises(ValueError):
        verify_intervention(a, b, t, u, history_capacity=5)
