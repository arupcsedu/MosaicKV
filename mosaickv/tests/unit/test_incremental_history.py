"""Synthetic allocation/retention evidence, never pretrained quality claims."""

import gc
from dataclasses import replace

import pytest
import torch

from mosaickv.adapters.huggingface.incremental_history_cache import (
    IncrementalHistoryLayer,
    incremental_history_budget,
)
from mosaickv.config import BoundedDecodeConfig, ConfigurationError, IncrementalHistoryConfig
from mosaickv.measurements.cache_lifecycle import CacheLifecycle
from mosaickv.measurements.kv_ownership import KVOwnershipLedger


def test_physical_budget_includes_largest_layer_reallocation():
    b = incremental_history_budget(192 * 2**20, [16384] * 32, 192, 1023)
    assert b.resident_ceiling_bytes + b.growth_scratch_reserve_bytes <= b.cap_bytes
    assert b.prompt_budget_bytes == 90 * 2**20
    assert b.resident_ceiling_bytes == 186 * 2**20
    assert b.prompt_budget_bytes + b.reserved_generated_bytes == b.resident_ceiling_bytes
    assert incremental_history_budget(10000, [64, 128], 20, 0).generated_positions == 0
    for args in [(100, [64, 64], 10, 100), (100, [], 1, 1), (True, [4], 1, 1)]:
        with pytest.raises(ValueError):
            incremental_history_budget(*args)


@pytest.mark.parametrize("policy", ["sink_recent", "recent_importance"])
def test_incremental_growth_and_retention_preserve_exact_kv_and_positions(policy):
    keys = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)
    values = keys + 100
    config = IncrementalHistoryConfig(2, 2, policy, 1.0)
    layer = IncrementalHistoryLayer(
        keys, values, generated_capacity=6, sink_tokens=1, next_logical_position=50, config=config
    )
    payload, initial = CacheLifecycle.count(((layer.keys, layer.values),))
    assert initial == payload == keys.numel() * keys.element_size() * 2
    for step in range(25):
        layer.update(
            torch.full((1, 2, 1, 4), float(50 + step)), torch.full((1, 2, 1, 4), float(150 + step))
        )
        torch.testing.assert_close(layer.keys[..., :3, :], keys, atol=0, rtol=0)
        torch.testing.assert_close(layer.values[..., :3, :], values, atol=0, rtol=0)
        positions = layer.generated_logical_positions
        assert 50 in positions  # Protected generated sink.
        assert set(range(max(50, 50 + step - 1), 51 + step)).issubset(positions)
        for slot, pos in enumerate(positions, start=3):
            assert bool((layer.keys[..., slot, :] == pos).all())
            assert bool((layer.values[..., slot, :] == pos + 100).all())
        if policy == "recent_importance":
            attention = torch.ones(1, 2, 1, layer.get_seq_length())
            # Keep one old generated token important even after it leaves recent.
            if 51 in positions:
                attention[..., 3 + positions.index(51)] = 100
            victim, finite = layer.observe_attention(attention / attention.sum(-1, keepdim=True))
            assert bool(finite)
            if victim is not None:
                layer._next_victim = int(victim)
        if policy == "recent_importance" and step > 5:
            assert 51 in positions
    assert layer.eviction_count == 25 - 6
    assert layer._allocated == 9
    assert len(layer.growth_events) == 3


def test_ownership_counts_aliased_allocations_once_and_does_not_keep_sources_alive():
    ledger = KVOwnershipLedger()
    k = torch.ones(1, 1, 8, 4)
    v = torch.ones_like(k)
    ledger.track("prefill", ((k, v),))
    ledger.track("alias", ((k[..., :2, :], v[..., :2, :]),))
    original = ledger.observe()
    assert original == 256
    new = (torch.ones_like(k), torch.ones_like(v))
    assert ledger.observe(replacement=(new,)) == 512
    del k, v
    gc.collect()
    assert ledger.observe() == 0
    assert ledger.peak_storage_bytes == 512


def test_growth_observer_counts_old_and_new_buffers_before_publication():
    ledger = KVOwnershipLedger()
    keys = torch.ones(1, 1, 2, 4)
    layer = None
    observed = []

    def callback(new):
        ledger.track("current", ((layer.keys, layer.values),))
        observed.append(ledger.observe(replacement=new))

    layer = IncrementalHistoryLayer(
        keys,
        keys.clone(),
        generated_capacity=8,
        sink_tokens=1,
        next_logical_position=10,
        config=IncrementalHistoryConfig(2, 2),
        transaction_observer=callback,
    )
    layer.update(torch.ones(1, 1, 1, 4), torch.ones(1, 1, 1, 4))
    assert observed == [(2 + 4) * 4 * 4 * 2]


def test_invalid_configuration_is_rejected():
    for kwargs in (
        {"chunk_tokens": True},
        {"recent_tokens": 0},
        {"policy": "hidden"},
        {"importance_decay": float("nan")},
    ):
        with pytest.raises(ConfigurationError):
            IncrementalHistoryConfig(**kwargs)
    with pytest.raises(ConfigurationError, match="capacity beyond"):
        BoundedDecodeConfig(4, 1, incremental_history=IncrementalHistoryConfig(recent_tokens=3))
    valid = BoundedDecodeConfig(8, 1, incremental_history=IncrementalHistoryConfig(recent_tokens=2))
    with pytest.raises(ConfigurationError, match="uniform prompt"):
        replace(valid, history_fraction=0.2)
    assert (
        BoundedDecodeConfig.from_mapping(
            {"generated_capacity": 8, "sink_tokens": 1, "incremental_history": {"recent_tokens": 2}}
        )
        == valid
    )
