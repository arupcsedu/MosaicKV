"""Synthetic transport/accounting checks, not benchmark-quality evidence."""

from types import SimpleNamespace

import pytest
import torch
from transformers import DynamicCache
from transformers.cache_utils import Cache

from mosaickv.adapters.huggingface.incremental_history_cache import (
    IncrementalHistoryLayer,
    incremental_history_budget,
    update_incremental_importance,
)
from mosaickv.config import ConfigurationError, IncrementalHistoryConfig
from mosaickv.measurements.kv_ownership import KVOwnershipLedger
from mosaickv.measurements.kv_transactions import DynamicKVTransactions


def test_fixed_reservation_does_not_use_teacher_target_length_to_change_prompt():
    a = incremental_history_budget(
        208 * 2**20, [16384] * 32, 128, 1023, fixed_prompt_reservation=True
    )
    b = incremental_history_budget(
        208 * 2**20, [16384] * 32, 128, 35, fixed_prompt_reservation=True
    )
    c = incremental_history_budget(208 * 2**20, [16384] * 32, 128, 0, fixed_prompt_reservation=True)
    assert a.prompt_budget_bytes == b.prompt_budget_bytes == c.prompt_budget_bytes
    assert a.prompt_budget_bytes == 137.5 * 2**20
    assert b.generated_positions == 35 and c.generated_positions == 0
    assert a.resident_ceiling_bytes + a.growth_scratch_reserve_bytes <= a.cap_bytes


def test_transaction_observer_counts_concat_before_old_owners_are_replaced():
    k, v = torch.ones(1, 1, 3, 4), torch.ones(1, 1, 3, 4)
    cache = DynamicCache()
    ledger = KVOwnershipLedger()
    old_update, old_cat = Cache.update, torch.cat
    with DynamicKVTransactions(ledger) as monitor:
        cache.update(k, v, 0)
        new = torch.ones(1, 1, 1, 4)
        cache.update(new, new, 0)
    assert Cache.update is old_update and torch.cat is old_cat
    assert monitor.update_count == 2 and monitor.concat_count == 2
    # At V replacement: newly stored K (64), old V (48), input (16), new V (64).
    assert ledger.peak_storage_bytes >= 192
    torch.testing.assert_close(cache.layers[0].keys, torch.ones(1, 1, 4, 4))
    with pytest.raises(ValueError), DynamicKVTransactions(ledger):
        raise ValueError("synthetic exit")
    assert Cache.update is old_update and torch.cat is old_cat


def test_batched_history_keeps_recent_and_important_tokens_without_per_token_ranking():
    config = IncrementalHistoryConfig(2, 2, "recent_importance", 1.0, importance_refresh_tokens=2)
    layers = [
        IncrementalHistoryLayer(
            torch.zeros(1, 2, 3, 4),
            torch.zeros(1, 2, 3, 4),
            generated_capacity=8,
            sink_tokens=1,
            next_logical_position=50,
            config=config,
        )
        for _ in range(2)
    ]
    scores = torch.zeros(2, 8)
    cache = SimpleNamespace(layers=layers, _mosaickv_history_scores=scores)
    state = SimpleNamespace(past_key_values=cache)
    for i, layer in enumerate(layers):
        layer._scores = scores[i]
    for step in range(28):
        weights = []
        for layer in layers:
            layer.update(
                torch.full((1, 2, 1, 4), float(50 + step)),
                torch.full((1, 2, 1, 4), float(150 + step)),
            )
            pos = layer.generated_logical_positions
            assert {50, 50 + step}.issubset(pos)
            if step >= 1:
                assert 49 + step in pos
            p = torch.ones(1, 2, 1, layer.get_seq_length())
            if 51 in pos:
                p[..., 3 + pos.index(51)] = 1000
            weights.append(p / p.sum(-1, keepdim=True))
        update_incremental_importance(state, weights)
        if step > 8:
            assert all(51 in x.generated_logical_positions for x in layers)
    assert all(x.eviction_count == 20 for x in layers)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fixed_prompt_reservation": 1},
        {"borrow_prefill_snapshot": "true"},
        {"importance_refresh_tokens": True},
        {"importance_refresh_tokens": 0},
    ],
)
def test_v2_configuration_is_strict(kwargs):
    with pytest.raises(ConfigurationError):
        IncrementalHistoryConfig(**kwargs)
