"""Atomic key-begin/value-commit for the isolated joint projection variant.

There is no fake raw V, no history eviction and no model-weight modification.
The existing one-buffer transport consumes K first and V only after a real
joint projection has committed the current value to its exact history slot.
"""

import torch

from mosaickv.evaluation.joint_incoming_projection import Ledger


class ValueLedger(Ledger):
    def __init__(self, layers):
        super().__init__(layers)
        self.value_commits = 0

    def enter(self, index, history, *, ready):
        if self.value_commits != self.calls:
            raise RuntimeError("previous actual value must commit before another layer")
        return super().enter(index, history, ready=ready)

    def value_ready(self, index, history):
        if (
            self.pending is not None
            or self.issuing is not None
            or self.value_commits + 1 != self.calls
            or (history - 1, index) != divmod(self.calls - 1, self.layers)
        ):
            raise RuntimeError("exactly one real value commit after its key consumer required")
        self.value_commits += 1

    def finish(self, updates):
        result = super().finish(updates)
        if self.value_commits != self.calls:
            raise RuntimeError("joint value transaction ended incomplete")
        return dict(
            result,
            value_commits=self.value_commits,
            all_value_projections_original=False,
            fake_raw_value_tensors=0,
        )


def begin_key(layer, key):
    """Original sealed-history update split before the not-yet-computed V.

    Source history seals complete tails at the shared pre-token barrier. Never
    silently replace that barrier or read the uncommitted raw value slot.
    """
    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes

    expected = (*layer.prompt_shape[:2], 1, layer.prompt_shape[-1])
    if (
        not layer.source_pending
        or layer.leased
        or key.shape != expected
        or key.dtype != torch.bfloat16
        or layer.history_length >= layer.maximum_decode_updates
        or getattr(layer, "_joint_pending_value", None) is not None
        or layer.tail_used >= 16
    ):
        raise ValueError("released source-captured bounded one-token key transaction required")
    if not layer.history_capacity:
        capacity = min(16, layer.maximum_decode_updates)
        newer = tuple(
            torch.empty((*expected[:2], capacity, expected[-1]), device=key.device, dtype=key.dtype)
            for _ in range(2)
        )
        layer.observe(newer)
        layer.history, layer.history_capacity = newer, capacity
        layer.history_bytes = backing_bytes(newer)
        layer.resident_bytes += layer.history_bytes
    if layer.tail_used >= layer.history_capacity:
        raise ValueError("the original full-tail sealing barrier must run before key begin")
    slot = layer.tail_used
    layer.history[0][..., slot : slot + 1, :].copy_(key)
    layer._joint_pending_value = slot
    layer.tail_used += 1
    layer.history_length += 1
    layer.leased = layer._new_lease((*expected[:2], layer.get_seq_length(), expected[-1]), key)
    layer.lease_bytes = backing_bytes(layer.leased)
    layer.observe(())
    layer.materialize()  # Only K; V is neither constructed nor read here.
    return layer.leased


def commit_value(layer, value):
    slot = getattr(layer, "_joint_pending_value", None)
    expected = (*layer.prompt_shape[:2], 1, layer.prompt_shape[-1])
    if (
        type(slot) is not int
        or slot != layer.tail_used - 1
        or not layer.source_pending
        or not layer.leased
        or value.shape != expected
        or value.dtype != torch.bfloat16
        or value.device != layer.history[1].device
    ):
        raise ValueError("same-slot real value before history/PV consumption required")
    layer.history[1][..., slot : slot + 1, :].copy_(value)
    layer._joint_pending_value = None
    layer.source_pending = False


def attention_forward(owner, layer):
    """Exact pinned Llama forward order, but split the private cache update."""
    attention = layer.attention
    from transformers.models.llama import modeling_llama as llama

    if (
        type(attention) is not llama.LlamaAttention
        or attention.num_key_value_groups != 1
        or attention.config._attn_implementation != "eager"
        or attention.training
    ):
        raise NotImplementedError("only pinned private full-MHA eager Llama inference is supported")

    def forward(
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_value=None,
        cache_position=None,
        **kwargs,
    ):
        if (
            past_key_value is not owner.cache
            or hidden_states.shape != (1, 1, 4096)
            or torch.is_grad_enabled()
        ):
            raise RuntimeError("private single-token source cache required by joint attention")
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attention.head_dim)
        query = attention.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key = attention.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query, key = llama.apply_rotary_pos_emb(query, key, cos, sin)
        key, value_lease = begin_key(layer, key)
        output, weights = llama.eager_attention_forward(
            attention,
            query,
            key,
            value_lease,
            attention_mask,
            dropout=0.0,
            scaling=attention.scaling,
            **kwargs,
        )
        if layer.source_pending or getattr(layer, "_joint_pending_value", None) is not None:
            raise RuntimeError("attention may not return with an uncommitted value")
        output = output.reshape(*input_shape, -1).contiguous()
        return attention.o_proj(output), weights

    return forward


def after_prompt(layer, component):
    pool = getattr(layer, "prompt_graph_pool", None)
    owner = None if pool is None else pool.joint_projection
    if owner is None or owner.mode != "kv" or component != 1:
        return
    engine = pool.lt_engine
    raw = engine.output[engine.tail].view(1, 1, 32, 128).transpose(1, 2)
    commit_value(layer, raw)
    owner.ledger.value_ready(owner.indices[id(layer)], layer.history_length)
