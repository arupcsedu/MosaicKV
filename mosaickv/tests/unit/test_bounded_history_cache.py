"""Synthetic correctness fixtures; no pretrained quality or performance claims."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
import torch

from mosaickv.adapters.huggingface.bounded_history_cache import (
    BoundedHistoryLayer,
    bounded_history_budget,
)
from mosaickv.config import (
    BoundedDecodeConfig,
    CacheConfig,
    ConfigurationError,
    RunConfig,
    canonical_config,
    config_sha256,
    synthetic_evaluation_config,
)
from mosaickv.measurements.cache_lifecycle import CacheLifecycle
from mosaickv.types import BudgetUnit


@pytest.mark.parametrize("capacity,sinks", [(1, 0), (2, 0), (3, 1), (8, 2)])
def test_ring_preserves_prompt_sinks_recent_and_exact_values(capacity, sinks):
    keys = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)
    values = keys + 100
    layer = BoundedHistoryLayer(
        keys, values, generated_capacity=capacity, sink_tokens=sinks, next_logical_position=50
    )
    pointers = (layer.keys.untyped_storage().data_ptr(), layer.values.untyped_storage().data_ptr())
    lifecycle = CacheLifecycle.start(((layer.keys, layer.values),))
    for step in range(40):
        key = torch.full((1, 2, 1, 4), float(50 + step))
        val = key + 1000
        layer.update(key, val, {"cache_position": torch.tensor([50 + step])})
        torch.testing.assert_close(layer.keys[..., :3, :], keys, rtol=0, atol=0)
        torch.testing.assert_close(layer.values[..., :3, :], values, rtol=0, atol=0)
        assert layer.keys.untyped_storage().data_ptr() == pointers[0]
        assert layer.values.untyped_storage().data_ptr() == pointers[1]
        positions = layer.generated_logical_positions
        seen = list(range(50, 51 + step))
        expected = seen if len(seen) <= capacity else seen[:sinks] + seen[-(capacity - sinks) :]
        assert sorted(positions) == expected
        for slot, position in enumerate(positions, start=3):
            assert torch.all(layer.keys[..., slot, :] == position)
            assert torch.all(layer.values[..., slot, :] == position + 1000)
        lifecycle.observe(((layer.keys, layer.values),))
    assert lifecycle.initial_storage_bytes == lifecycle.final_storage_bytes
    assert lifecycle.peak_storage_bytes == (3 + capacity) * 2 * 4 * 4 * 2
    assert layer.evicted_positions == 40 - capacity


def test_budget_reserves_generation_before_prompt_and_counts_rounding():
    budget = bounded_history_budget(1001, 64, generated_capacity=4, decode_updates=1023)
    assert budget.prompt_budget_bytes == 11 * 64
    assert budget.reserved_generated_bytes == 4 * 64
    assert budget.lattice_shortfall_bytes == 41
    assert budget.prompt_budget_bytes + budget.reserved_generated_bytes <= 1001
    assert bounded_history_budget(1001, 64, 4, 1).generated_positions == 1
    assert bounded_history_budget(1001, 64, 4, 0).reserved_generated_bytes == 0
    with pytest.raises(ValueError, match="cannot fit"):
        bounded_history_budget(256, 64, 4, 100)


def test_ring_instances_do_not_alias_each_other_or_source():
    keys = torch.arange(8, dtype=torch.float32).reshape(1, 1, 2, 4)
    first, second = [
        BoundedHistoryLayer(
            keys, keys, generated_capacity=2, sink_tokens=0, next_logical_position=9
        )
        for _ in range(2)
    ]
    first.update(torch.ones(1, 1, 1, 4), torch.ones(1, 1, 1, 4))
    assert second.get_seq_length() == 2
    assert second.generated_logical_positions == ()
    assert torch.equal(keys, second.keys)
    assert first.keys.untyped_storage().data_ptr() != keys.untyped_storage().data_ptr()


def test_ring_rejects_batch_chunk_update_crop_and_reorder():
    keys = torch.ones(1, 1, 2, 4)
    layer = BoundedHistoryLayer(
        keys, keys, generated_capacity=2, sink_tokens=0, next_logical_position=9
    )
    with pytest.raises(ValueError, match="exactly one token"):
        layer.update(torch.ones(1, 1, 2, 4), torch.ones(1, 1, 2, 4))
    with pytest.raises(NotImplementedError, match="rollback"):
        layer.crop(1)
    with pytest.raises(NotImplementedError, match="reordering"):
        layer.reorder_cache(torch.tensor([0]))


@pytest.mark.parametrize("capacity,sinks", [(0, 0), (2, 2), (True, 0), (2, -1), (2, True)])
def test_strict_configuration_rejects_invalid_fields(capacity, sinks):
    with pytest.raises(ConfigurationError, match="bounded_decode"):
        BoundedDecodeConfig(capacity, sinks)
    with pytest.raises(ConfigurationError, match="unknown"):
        BoundedDecodeConfig.from_mapping({"invented": True})


def test_old_config_identity_and_strict_roundtrip_remain_compatible():
    config = synthetic_evaluation_config()
    payload = canonical_config(config)
    assert "bounded_decode" not in payload
    assert config_sha256(RunConfig.from_mapping(payload)) == config_sha256(config)
    assert config_sha256(RunConfig.from_mapping(json.loads(json.dumps(payload)))) == config_sha256(
        config
    )
    with pytest.raises(ConfigurationError, match="bounded_decode requires"):
        replace(config, bounded_decode=BoundedDecodeConfig())
    with pytest.raises(ConfigurationError, match="bounded_decode requires"):
        replace(
            config,
            cache=CacheConfig(1000, BudgetUnit.BYTES, 1.0, 1),
            bounded_decode=BoundedDecodeConfig(),
        )
