"""Synthetic invariants for audited published KV-cache reimplementations."""

from __future__ import annotations

import pytest
import torch

from mosaickv.adapters.huggingface import (
    CachedKeyState,
    CacheLayerSnapshot,
    CacheSnapshot,
    DecodeState,
)
from mosaickv.backends.hf_runtime import maintain_elastic_cache_decode_cache
from mosaickv.baselines.published_kv import (
    PublishedKVReimplementationError,
    build_published_kv_reimpl_plan,
    published_kv_runtime_payloads,
)
from mosaickv.cache_state import FullKVState
from mosaickv.config import CacheConfig, PublishedKVConfig
from mosaickv.types import BudgetUnit, MosaicKVMethod


def _state(*, layers: int = 3, heads: int = 2, sequence: int = 16) -> FullKVState:
    tensors = tuple(
        (
            torch.arange(heads * sequence * 4, dtype=torch.float32).reshape(1, heads, sequence, 4)
            + layer * 10_000,
            torch.arange(heads * sequence * 4, dtype=torch.float32).reshape(1, heads, sequence, 4)
            + layer * 20_000,
        )
        for layer in range(layers)
    )
    return FullKVState.from_tensors(
        tensors,
        block_size=1,
        sequence_dimension=-2,
        head_dimension=-3,
        mandatory_logical_positions=(sequence - 1,),
        cached_key_state="post_rope",
    )


def _attentions(
    *, layers: int = 3, query_heads: int = 4, sequence: int = 16
) -> tuple[torch.Tensor, ...]:
    base = torch.arange(1, sequence + 1, dtype=torch.float32)
    rows = []
    for layer in range(layers):
        values = base.repeat(query_heads, sequence, 1)
        values = values + torch.arange(sequence, dtype=torch.float32).view(1, sequence, 1)
        values = values + layer / 100
        values = values / values.sum(dim=-1, keepdim=True)
        rows.append(values.unsqueeze(0))
    return tuple(rows)


@pytest.mark.parametrize(
    "method",
    [
        MosaicKVMethod.ELASTIC_CACHE_REIMPL,
        MosaicKVMethod.H2O_REIMPL,
        MosaicKVMethod.STREAMINGLLM_REIMPL,
        MosaicKVMethod.PYRAMIDKV_REIMPL,
        MosaicKVMethod.SNAPKV_REIMPL,
        MosaicKVMethod.ROCO_REIMPL,
    ],
)
def test_published_policy_is_deterministic_and_obeys_exact_byte_budget(
    method: MosaicKVMethod,
) -> None:
    state = _state()
    # Twelve positions per route permit the eight-token observation window
    # plus scored historical tokens for H2O/SnapKV/PyramidKV.
    slots = 3 * 2 * 12
    slot_bytes = state.blocks[0].byte_size
    cache = CacheConfig(
        budget_value=slots * slot_bytes,
        budget_unit=BudgetUnit.BYTES,
        retention_ratio=0.75,
        block_size=1,
    )
    config = PublishedKVConfig(
        enabled=True,
        observation_window=8,
        sink_tokens=4,
        attention_pool_kernel=5,
    )
    first = build_published_kv_reimpl_plan(state, _attentions(), method, config, cache)
    second = build_published_kv_reimpl_plan(state, _attentions(), method, config, cache)
    assert first == second
    assert first.active_slots == slots
    assert first.retained_bytes == slots * slot_bytes
    assert first.retained_bytes <= cache.budget_value
    payloads, records = published_kv_runtime_payloads(first)
    assert len(payloads) == 3 * 2
    assert len(records) == slots
    assert all(
        record["implementation_status"] == "local_reimplementation_not_official"
        for record in records
    )


@pytest.mark.parametrize(
    "method",
    [
        MosaicKVMethod.ELASTIC_CACHE_REIMPL,
        MosaicKVMethod.H2O_REIMPL,
        MosaicKVMethod.STREAMINGLLM_REIMPL,
        MosaicKVMethod.PYRAMIDKV_REIMPL,
        MosaicKVMethod.SNAPKV_REIMPL,
        MosaicKVMethod.ROCO_REIMPL,
    ],
)
def test_published_policy_retention_one_is_identity(method: MosaicKVMethod) -> None:
    state = _state()
    cache = CacheConfig(
        budget_value=state.active_bytes,
        budget_unit=BudgetUnit.BYTES,
        retention_ratio=1.0,
        block_size=1,
    )
    plan = build_published_kv_reimpl_plan(
        state,
        _attentions(),
        method,
        PublishedKVConfig(enabled=True, observation_window=8),
        cache,
    )
    assert plan.retained_bytes == state.active_bytes
    assert all(
        head.selected_physical_positions == tuple(range(head.source_positions))
        for head in plan.heads
    )
    payloads, records = published_kv_runtime_payloads(plan)
    assert len(records) == len(state.blocks)
    for layer_index, layer in enumerate(state.layers):
        for kv_head in range(layer.kv_heads):
            entries = payloads[(layer_index, kv_head)]
            reconstructed_key = torch.cat([entry[3] for entry in entries], dim=-2)
            reconstructed_value = torch.cat([entry[4] for entry in entries], dim=-2)
            assert torch.equal(
                reconstructed_key,
                layer.key[:, kv_head : kv_head + 1, :, :],
            )
            assert torch.equal(
                reconstructed_value,
                layer.value[:, kv_head : kv_head + 1, :, :],
            )


def test_observation_policy_rejects_capacity_smaller_than_protected_window() -> None:
    state = _state()
    slot_bytes = state.blocks[0].byte_size
    cache = CacheConfig(
        budget_value=6 * 3 * 2 * slot_bytes,
        budget_unit=BudgetUnit.BYTES,
        retention_ratio=6 / 16,
        block_size=1,
    )
    with pytest.raises(PublishedKVReimplementationError, match="outside feasible range"):
        build_published_kv_reimpl_plan(
            state,
            _attentions(),
            MosaicKVMethod.SNAPKV_REIMPL,
            PublishedKVConfig(enabled=True, observation_window=8),
            cache,
        )


def test_pyramidkv_allocates_more_capacity_to_early_layers() -> None:
    state = _state(layers=4)
    slot_bytes = state.blocks[0].byte_size
    slots = 4 * 2 * 12
    plan = build_published_kv_reimpl_plan(
        state,
        _attentions(layers=4),
        MosaicKVMethod.PYRAMIDKV_REIMPL,
        PublishedKVConfig(enabled=True, observation_window=8, pyramid_beta=20),
        CacheConfig(
            budget_value=slots * slot_bytes,
            budget_unit=BudgetUnit.BYTES,
            retention_ratio=0.75,
            block_size=1,
        ),
    )
    layer_capacities = [plan.heads[layer * 2].capacity for layer in range(4)]
    assert layer_capacities == sorted(layer_capacities, reverse=True)


def test_streamingllm_keeps_attention_sinks_and_recent_window() -> None:
    state = _state(layers=1, heads=1, sequence=10)
    slot_bytes = state.blocks[0].byte_size
    plan = build_published_kv_reimpl_plan(
        state,
        (),
        MosaicKVMethod.STREAMINGLLM_REIMPL,
        PublishedKVConfig(enabled=True, sink_tokens=2),
        CacheConfig(
            budget_value=6 * slot_bytes,
            budget_unit=BudgetUnit.BYTES,
            retention_ratio=0.6,
            block_size=1,
        ),
    )
    assert plan.heads[0].selected_physical_positions == (0, 1, 6, 7, 8, 9)
    assert plan.heads[0].protected_positions == (0, 1, 6, 7, 8, 9)


@pytest.mark.parametrize(
    "method",
    [MosaicKVMethod.H2O_REIMPL, MosaicKVMethod.SNAPKV_REIMPL],
)
def test_observation_attention_formula_selects_known_heavy_hitters(
    method: MosaicKVMethod,
) -> None:
    state = _state(layers=1, heads=1, sequence=8)
    attention = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
    attention[0, 0, -2:, 1] = 0.9
    attention[0, 0, -2:, 3] = 0.8
    attention[0, 0, -2:, 5] = 0.7
    slot_bytes = state.blocks[0].byte_size
    plan = build_published_kv_reimpl_plan(
        state,
        (attention,),
        method,
        PublishedKVConfig(
            enabled=True,
            observation_window=2,
            attention_pool_kernel=1,
        ),
        CacheConfig(
            budget_value=5 * slot_bytes,
            budget_unit=BudgetUnit.BYTES,
            retention_ratio=5 / 8,
            block_size=1,
        ),
    )
    assert plan.heads[0].selected_physical_positions == (1, 3, 5, 6, 7)
    assert plan.heads[0].protected_positions == (6, 7)


def test_roco_protects_recent_and_high_variance_tokens() -> None:
    state = _state(layers=1, heads=1, sequence=8)
    attention = torch.full((1, 1, 8, 8), 0.1, dtype=torch.float32)
    attention[0, 0, ::2, 0] = 1.0
    attention[0, 0, 1::2, 0] = 0.0
    attention[0, 0, :, 1] = 0.9
    slot_bytes = state.blocks[0].byte_size
    plan = build_published_kv_reimpl_plan(
        state,
        (attention,),
        MosaicKVMethod.ROCO_REIMPL,
        PublishedKVConfig(
            enabled=True,
            recent_fraction=0.2,
            roco_protected_fraction=0.2,
        ),
        CacheConfig(
            budget_value=5 * slot_bytes,
            budget_unit=BudgetUnit.BYTES,
            retention_ratio=5 / 8,
            block_size=1,
        ),
    )
    assert {0, 7} <= set(plan.heads[0].protected_positions)
    assert {0, 1, 7} <= set(plan.heads[0].selected_physical_positions)


def test_elastic_cache_uses_shared_layer_anchors_and_fp32_bucket_means() -> None:
    state = _state(layers=1, heads=2, sequence=8)
    attention = torch.zeros((1, 4, 8, 8), dtype=torch.float32)
    attention[:, :, :, 1] = 0.9
    attention[:, :, :, 5] = 0.8
    slot_bytes = state.blocks[0].byte_size
    plan = build_published_kv_reimpl_plan(
        state,
        (attention,),
        MosaicKVMethod.ELASTIC_CACHE_REIMPL,
        PublishedKVConfig(
            enabled=True,
            elastic_start_tokens=1,
            elastic_protect_tokens=1,
        ),
        CacheConfig(
            budget_value=2 * 4 * slot_bytes,
            budget_unit=BudgetUnit.BYTES,
            retention_ratio=0.5,
            block_size=1,
        ),
    )
    assert all(head.selected_physical_positions == (0, 1, 5, 7) for head in plan.heads)
    assert all(
        head.merge_memberships == ((0,), (1, 2, 3), (4, 5, 6), (7,))
        for head in plan.heads
    )
    payloads, records = published_kv_runtime_payloads(plan)
    merged_key = payloads[(0, 0)][1][3]
    expected = state.layers[0].key[:, 0:1, 1:4, :].float().mean(dim=-2, keepdim=True)
    assert torch.equal(merged_key, expected.to(dtype=state.layers[0].key.dtype))
    assert records[1]["source_membership"] == [1, 2, 3]
    assert records[1]["tier"] == "elastic_cache_reimpl_merged"
    payload_bytes = sum(
        int(key.numel() * key.element_size() + value.numel() * value.element_size())
        for entries in payloads.values()
        for _logical, _tier, _source, key, value in entries
    )
    assert payload_bytes == plan.retained_bytes


def test_elastic_cache_decode_uses_fixed_point_and_preserves_logical_length() -> None:
    state = _state(layers=1, heads=1, sequence=8)
    attention = torch.ones((1, 1, 8, 8), dtype=torch.float32)
    slot_bytes = state.blocks[0].byte_size
    plan = build_published_kv_reimpl_plan(
        state,
        (attention,),
        MosaicKVMethod.ELASTIC_CACHE_REIMPL,
        PublishedKVConfig(enabled=True, elastic_decode_offset=-2),
        CacheConfig(
            budget_value=4 * slot_bytes,
            budget_unit=BudgetUnit.BYTES,
            retention_ratio=0.5,
            block_size=1,
        ),
    )
    cache = CacheSnapshot(
        (
            CacheLayerSnapshot(
                torch.arange(20, dtype=torch.float32).reshape(1, 1, 5, 4),
                torch.arange(20, dtype=torch.float32).reshape(1, 1, 5, 4),
                -2,
            ),
        ),
        tuple,
        "legacy",
        5,
        CachedKeyState.POST_ROPE,
    )

    class FakeAdapter:
        def extract_past_key_values(self, value: object) -> CacheSnapshot:
            assert isinstance(value, CacheSnapshot)
            return value

        def inject_past_key_values(self, snapshot: CacheSnapshot) -> object:
            return snapshot

    decode_state = DecodeState(
        past_key_values=cache,
        attention_mask=torch.ones((1, 5), dtype=torch.long),
        active_cache_length=5,
        logical_sequence_length=9,
        next_decode_position=9,
        modality_map=(),
        model_state={"mosaickv_packed_all_valid": True},
    )
    updated, events = maintain_elastic_cache_decode_cache(
        FakeAdapter(),  # type: ignore[arg-type]
        decode_state,
        plan,
        step_index=1,
    )
    assert updated.active_cache_length == 4
    assert updated.logical_sequence_length == 9
    assert updated.next_decode_position == 9
    assert events[0]["fixed_truncation_rank"] == 2
    assert events[0]["removed_physical_position"] == 2
