"""Experimental exact prompt + incrementally allocated generated history.

Single-request LLaVA-1.5 eager only. No source K/V transformation, model
updates, benchmark annotations, extra model probes, or position renumbering.
Recent/important retention is related to H2O, not claimed as a new principle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from mosaickv.adapters.huggingface.bounded_history_cache import BoundedHistoryLayer


@dataclass(frozen=True)
class IncrementalHistoryBudget:
    cap_bytes: int
    bytes_per_position: int
    generated_positions: int
    reserved_generated_bytes: int
    prompt_budget_bytes: int
    lattice_shortfall_bytes: int
    growth_scratch_reserve_bytes: int
    resident_ceiling_bytes: int


def incremental_history_budget(
    cap_bytes,
    layer_position_bytes,
    generated_capacity,
    decode_updates,
    *,
    fixed_prompt_reservation=False,
):
    costs = tuple(layer_position_bytes)
    if (
        type(cap_bytes) is not int
        or cap_bytes < 1
        or not costs
        or any(type(n) is not int or n < 1 for n in costs)
        or type(generated_capacity) is not int
        or generated_capacity < 1
        or type(decode_updates) is not int
        or decode_updates < 0
    ):
        raise ValueError("invalid incremental history physical budget")
    total, largest = sum(costs), max(costs)
    # At most one layer's previous allocation coexists with its replacement.
    slots = cap_bytes // (total + largest)
    generated = min(generated_capacity, decode_updates)
    reserved = generated_capacity if fixed_prompt_reservation else generated
    if slots <= reserved:
        raise ValueError("incremental cap cannot fit history, scratch reserve and one prompt slot")
    resident = slots * total
    reserve = slots * largest
    return IncrementalHistoryBudget(
        cap_bytes,
        total,
        generated,
        reserved * total,
        (slots - reserved) * total,
        cap_bytes - resident - reserve,
        reserve,
        resident,
    )


class IncrementalHistoryLayer(BoundedHistoryLayer):
    """Own a contiguous exact cache; grow by chunks, then replace one slot."""

    def __init__(
        self,
        keys,
        values,
        *,
        generated_capacity,
        sink_tokens,
        next_logical_position,
        config,
        transaction_observer=None,
    ):
        import torch

        if (
            keys.ndim != 4
            or keys.shape != values.shape
            or keys.shape[0] != 1
            or keys.dtype != values.dtype
            or keys.device != values.device
            or keys.shape[-2] < 1
            or generated_capacity < 1
            or not 0 <= sink_tokens < generated_capacity
            or next_logical_position < keys.shape[-2]
        ):
            raise ValueError("invalid incremental exact cache geometry")
        self.prompt_length = int(keys.shape[-2])
        self.generated_capacity = generated_capacity
        self.sink_tokens = sink_tokens
        self._capacity = self.prompt_length + generated_capacity
        self._allocated = self.prompt_length
        self._length = self.prompt_length
        self._updates = 0
        self._next_logical_position = next_logical_position
        self._generated_positions = []
        self._key_storage, self._value_storage = keys, values
        self.keys, self.values = keys, values
        self.config = config
        self._scores = torch.zeros(generated_capacity, dtype=torch.float32, device=keys.device)
        self._next_victim = None
        self._victim_queue = []
        self._transaction_observer = transaction_observer
        self.growth_events = []
        self.eviction_count = 0
        self.important_survivor_count = 0

    def _grow(self, required):
        if required <= self._allocated:
            return
        new_length = min(
            self._capacity,
            self.prompt_length
            + math.ceil((required - self.prompt_length) / self.config.chunk_tokens)
            * self.config.chunk_tokens,
        )
        shape = (*self.keys.shape[:-2], new_length, self.keys.shape[-1])
        new_key, new_value = self.keys.new_empty(shape), self.values.new_empty(shape)
        new_key[..., : self._length, :].copy_(self.keys)
        new_value[..., : self._length, :].copy_(self.values)
        if self._transaction_observer is not None:
            self._transaction_observer(((new_key, new_value),))
        self.growth_events.append({"old_positions": self._allocated, "new_positions": new_length})
        self._key_storage, self._value_storage = new_key, new_value
        self._allocated = new_length

    def update(self, key_states, value_states, cache_kwargs=None):
        del cache_kwargs
        expected = (*self.keys.shape[:-2], 1, self.keys.shape[-1])
        if (
            tuple(key_states.shape) != expected
            or tuple(value_states.shape) != expected
            or key_states.dtype != self.keys.dtype
            or value_states.dtype != self.values.dtype
            or key_states.device != self.keys.device
            or value_states.device != self.values.device
        ):
            raise ValueError("incremental history requires one token with unchanged geometry")
        if self._updates < self.generated_capacity:
            slot = self._updates
            self._generated_positions.append(self._next_logical_position)
        elif self.config.policy == "sink_recent":
            slot = self.sink_tokens + (self._updates - self.sink_tokens) % (
                self.generated_capacity - self.sink_tokens
            )
        else:
            if self._victim_queue:
                self._next_victim = self._victim_queue.pop(0)
            if self._next_victim is None:
                raise RuntimeError(
                    "importance eviction needs the preceding causal attention update"
                )
            slot = self._next_victim
            old = self._generated_positions[slot]
            if (
                slot < self.sink_tokens
                or old >= self._next_logical_position - self.config.recent_tokens + 1
            ):
                raise RuntimeError("importance eviction attempted a protected sink/recent token")
        if self._updates >= self.generated_capacity:
            self._generated_positions[slot] = self._next_logical_position
            self.eviction_count += 1
        self._scores[slot] = 0
        self._grow(self.prompt_length + min(self._updates + 1, self.generated_capacity))
        target = self.prompt_length + slot
        self._key_storage[..., target : target + 1, :].copy_(key_states)
        self._value_storage[..., target : target + 1, :].copy_(value_states)
        self._updates += 1
        self._next_logical_position += 1
        self._length = self.prompt_length + min(self._updates, self.generated_capacity)
        self.keys = self._key_storage[..., : self._length, :]
        self.values = self._value_storage[..., : self._length, :]
        self._next_victim = None
        return self.keys, self.values

    def observe_attention(self, probabilities):
        """Update scores from this actual decode step; return GPU victim index."""
        import torch

        if probabilities.shape[-2:] != (1, self._length):
            raise RuntimeError("incremental attention does not align with physical slots")
        count = len(self._generated_positions)
        mass = probabilities.detach().float().mean(dim=(0, 1, 2))[self.prompt_length :]
        self._scores[:count].mul_(self.config.importance_decay).add_(mass)
        if count < self.generated_capacity or count <= self.sink_tokens + self.config.recent_tokens:
            return None, torch.isfinite(probabilities).all()
        # Sort eligible slots by original position before argmin: stable oldest
        # tie-breaking, independent of circular physical layout.
        eligible = sorted(
            (
                i
                for i, p in enumerate(self._generated_positions)
                if i >= self.sink_tokens
                and p < self._next_logical_position - self.config.recent_tokens + 1
            ),
            key=lambda i: self._generated_positions[i],
        )
        if not eligible:
            raise RuntimeError("history capacity has no unprotected eviction candidate")
        indices = torch.tensor(eligible, dtype=torch.long, device=self.keys.device)
        return indices[self._scores.index_select(0, indices).argmin()], torch.isfinite(
            probabilities
        ).all()

    def eligible_victims(self):
        return sorted(
            (
                i
                for i, p in enumerate(self._generated_positions)
                if i >= self.sink_tokens
                and p < self._next_logical_position - self.config.recent_tokens + 1
            ),
            key=lambda i: self._generated_positions[i],
        )


def update_incremental_importance(state, attention_weights):
    import torch

    layers = state.past_key_values.layers
    if len(layers) != len(attention_weights):
        raise RuntimeError("history attention capture missing a layer")
    if layers[0].config.importance_refresh_tokens > 1:
        # Uniform physical lengths allow one batched reduction/update. Layer-local
        # original positions remain independent; no query/head renumbering occurs.
        cache = state.past_key_values
        if len({tuple(p.shape) for p in attention_weights}) != 1:
            raise RuntimeError("batched history scoring requires uniform attention geometry")
        probabilities = torch.stack(attention_weights)
        if not bool(torch.isfinite(probabilities).all()):
            raise RuntimeError("nonfinite causal history attention")
        count = len(layers[0]._generated_positions)
        mass = probabilities.float().mean(dim=(1, 2, 3))[:, layers[0].prompt_length :]
        cache._mosaickv_history_scores[:, :count].mul_(layers[0].config.importance_decay).add_(mass)
        if (
            count < layers[0].generated_capacity
            or count <= (layers[0].sink_tokens + layers[0].config.recent_tokens)
            or layers[0]._victim_queue
        ):
            return
        eligible = [x.eligible_victims() for x in layers]
        if not eligible[0] or len({len(e) for e in eligible}) != 1:
            raise RuntimeError("history layers have inconsistent eligible capacities")
        indices = torch.tensor(eligible, dtype=torch.long, device=probabilities.device)
        scores = cache._mosaickv_history_scores.gather(1, indices)
        width = min(layers[0].config.importance_refresh_tokens, len(eligible[0]))
        order = torch.argsort(scores, dim=1, stable=True)[:, :width]
        choices = indices.gather(1, order).cpu().tolist()
        for layer, queue in zip(layers, choices, strict=True):
            layer._victim_queue = queue
        return
    updates = [
        layer.observe_attention(p) for layer, p in zip(layers, attention_weights, strict=True)
    ]
    if not bool(torch.stack([u[1] for u in updates]).all()):
        raise RuntimeError("nonfinite causal history attention")
    choices = [u[0] for u in updates]
    if any(c is not None for c in choices):
        if any(c is None for c in choices):
            raise RuntimeError("history layers reached inconsistent generated capacities")
        # One host transfer for all layer decisions, not one per layer.
        for layer, choice in zip(layers, torch.stack(choices).cpu().tolist(), strict=True):
            layer._next_victim = int(choice)


def install_incremental_history(
    adapter,
    state,
    *,
    generated_capacity,
    sink_tokens,
    cap_bytes,
    config,
    ownership_ledger=None,
    request_ownership_ledger=None,
):
    import torch
    from transformers import DynamicCache

    from mosaickv.adapters.huggingface.bounded_history_cache import assert_bounded_history_support
    from mosaickv.measurements.cache_lifecycle import CacheLifecycle
    from mosaickv.measurements.kv_ownership import KVOwnershipLedger

    assert_bounded_history_support(adapter)
    if (
        state.model_state.get("mosaickv_packed_all_valid") is not True
        or state.model_state.get("mosaickv_validity_masks") is not None
        or state.model_state.get("mosaickv_prototype_slots")
        or state.model_state.get("mosaickv_bounded_history")
    ):
        raise RuntimeError("incremental history requires an unpadded, exact, private prompt")
    original, _ = adapter._legacy_layers(state.past_key_values)
    cache = DynamicCache()
    ledger = ownership_ledger or KVOwnershipLedger()

    def transaction(new):
        ledger.track("active_decode", tuple((x.keys, x.values) for x in cache.layers))
        if ledger.observe(growth_replacement=new) > cap_bytes:
            raise RuntimeError("incremental resident plus growth transaction exceeded hard cap")
        if request_ownership_ledger is not None:
            request_ownership_ledger.track(
                "decode_cache", tuple((x.keys, x.values) for x in cache.layers)
            )
            request_ownership_ledger.observe(growth_replacement=new)

    cache.layers = [
        IncrementalHistoryLayer(
            k,
            v,
            generated_capacity=generated_capacity,
            sink_tokens=sink_tokens,
            next_logical_position=state.next_decode_position,
            config=config,
            transaction_observer=transaction,
        )
        for k, v in original
    ]
    if config.importance_refresh_tokens > 1:
        cache._mosaickv_history_scores = torch.zeros(
            (len(cache.layers), generated_capacity),
            dtype=torch.float32,
            device=original[0][0].device,
        )
        for index, layer in enumerate(cache.layers):
            layer._scores = cache._mosaickv_history_scores[index]
    resident_ceiling = sum(
        (k.numel() * k.element_size() + v.numel() * v.element_size())
        // k.shape[-2]
        * (k.shape[-2] + generated_capacity)
        for k, v in original
    )
    largest_growth_copy = max(
        (k.numel() * k.element_size() + v.numel() * v.element_size())
        // k.shape[-2]
        * (k.shape[-2] + generated_capacity)
        for k, v in original
    )
    if resident_ceiling + largest_growth_copy > cap_bytes:
        raise RuntimeError("incremental history was planned without growth scratch capacity")
    cache._mosaickv_bounded_history = True
    cache._mosaickv_independent_layer_lengths = len({k.shape[-2] for k, v in original}) != 1
    state.past_key_values = cache
    state.model_state.pop("prefixkv_ragged", None)
    state.model_state.update(
        {
            "mosaickv_bounded_history": True,
            "mosaickv_incremental_history": True,
            "mosaickv_history_importance": config.policy == "recent_importance",
            "mosaickv_bounded_history_capacity": state.active_cache_length + generated_capacity,
            "mosaickv_bounded_history_masks": tuple(
                torch.zeros((1, 1, 1, x._capacity), dtype=x.keys.dtype, device=x.keys.device)
                for x in cache.layers
            ),
            "mosaickv_decode_cache_storage": "incremental_generated_history_v1",
            "mosaickv_history_ownership": ledger,
        }
    )
    payload, storage = CacheLifecycle.count(tuple((x.keys, x.values) for x in cache.layers))
    ledger.track("active_decode", tuple((x.keys, x.values) for x in cache.layers))
    ledger.observe()
    return {
        "status": "incremental_exact_prompt_history_variant",
        "lifetime_cap_bytes": cap_bytes,
        "resident_ceiling_bytes": resident_ceiling,
        "growth_scratch_reserve_bytes": largest_growth_copy,
        "initial_payload_bytes": payload,
        "initial_storage_bytes": storage,
        "generated_capacity": generated_capacity,
        "sink_tokens": sink_tokens,
        "history_policy": config.policy,
        "prompt_slots_unchanged": True,
        "rope_keys_transformed": False,
        "benchmark_labels_accessed": False,
        "whole_request_peak_not_certified_by_this_record": True,
    }


def incremental_history_record(state):
    layers = state.past_key_values.layers
    ledger = state.model_state["mosaickv_history_ownership"]
    ledger.track("active_decode", tuple((x.keys, x.values) for x in layers))
    ledger.observe()
    return {
        "evicted_generated_positions_per_layer": [x.eviction_count for x in layers],
        "growth_events_per_layer": [x.growth_events for x in layers],
        "retained_generated_positions_per_layer": [
            list(x.generated_logical_positions) for x in layers
        ],
        "allocated_positions_per_layer": [x._allocated for x in layers],
        "owned_decode_KV_including_growth_peak": ledger.record(),
        "next_decode_position": state.next_decode_position,
    }
