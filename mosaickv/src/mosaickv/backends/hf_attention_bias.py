"""Isolated single-request HF eager transport for original keys + synthetic V.

Does not modify the existing exact-only installer, model classes, or weights.
Uses the audited bounded-history update and adapter layer-mask dispatch, but
explicitly labels this different representation. No production method enables it.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
from pathlib import Path


def source_contract():
    import transformers
    from transformers.models.llama.modeling_llama import LlamaAttention, eager_attention_forward

    path = Path(inspect.getfile(LlamaAttention))
    return dict(
        transformers_version=transformers.__version__,
        source_file=str(path),
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        attention_api=f"{eager_attention_forward.__module__}.eager_attention_forward",
        cache_api="BoundedHistoryLayer.update",
        mask_api="LayerAttentionMaskOverride.activate",
    )


def bias_cache_storage(adapter, state):
    """Unique backing allocations, including auxiliary decode metadata.

    Unlike a KV-only ledger, this includes the bias and mask, metadata buffers,
    and the state attention mask. FullKV/private snapshots/workspaces are not
    silently charged to the live application cache; GPU peak is separate.
    """
    import torch

    from mosaickv.measurements.cache_lifecycle import CacheLifecycle

    legacy = adapter._legacy_layers(state.past_key_values)[0]
    kv = CacheLifecycle.count(legacy)[1]
    storages = {}

    def walk(item):
        if isinstance(item, torch.Tensor):
            storage = item.untyped_storage()
            storages[(str(item.device), storage.data_ptr())] = storage.nbytes()
        elif isinstance(item, dict):
            for value in item.values():
                walk(value)
        elif isinstance(item, (tuple, list)):
            for value in item:
                walk(value)

    walk(legacy)
    walk(state.model_state)
    walk(state.attention_mask)
    total = sum(storages.values())
    record = dict(
        active_kv_backing_bytes=kv,
        auxiliary_backing_bytes=total - kv,
        total_persistent_backing_bytes=total,
        cap_bytes=state.model_state["attention_compaction_cap_bytes"],
    )
    if total > record["cap_bytes"]:
        raise RuntimeError(f"bias transport exceeds all-persistent-storage cap: {record}")
    return record


def install_attention_bias_cache(
    adapter,
    reference,
    heads,
    *,
    required_logical_positions,
    generated_capacity,
    sink_tokens,
    cap_bytes,
    retention_one=False,
):
    """Construct a separately owned state from a pristine prefill and head fits.

    `heads[layer][kv_head]` is a CompactedHead, potentially constructed on CPU.
    Mandatory entries and every copied post-RoPE key are verified against the
    actual reference, not trusted diagnostic booleans. No ring may be cloned.
    """
    import torch
    from transformers import DynamicCache

    from mosaickv.adapters.huggingface.bounded_history_cache import (
        BoundedHistoryLayer,
        assert_bounded_history_support,
    )
    from mosaickv.adapters.huggingface.types import DecodeState

    assert_bounded_history_support(adapter)
    if (
        reference.active_cache_length != reference.logical_sequence_length
        or reference.next_decode_position != reference.logical_sequence_length
        or reference.model_state.get("mosaickv_bounded_history")
        or reference.model_state.get("mosaickv_validity_masks") is not None
        or type(cap_bytes) is not int
        or cap_bytes < 1
        or type(retention_one) is not bool
        or type(required_logical_positions) is not tuple
        or any(
            type(p) is not int or not 0 <= p < reference.active_cache_length
            for p in required_logical_positions
        )
    ):
        raise ValueError("bias transport needs an unmodified full prefill and valid hard cap")
    original = adapter._legacy_layers(reference.past_key_values)[0]
    if len(heads) != len(original):
        raise ValueError("bias transport requires every layer")
    buffers, biases, positions, layers = [], [], [], []
    common_slots = None
    with torch.inference_mode():
        for i, ((key, value), head_rows) in enumerate(zip(original, heads, strict=True)):
            if key.ndim != 4 or key.shape != value.shape or key.shape[0] != 1:
                raise ValueError("bias transport requires B=1 dense matching K/V tensors")
            if len(head_rows) != key.shape[1]:
                raise ValueError("bias transport requires every KV head")
            ks, vs, bs, ps = [], [], [], []
            for head, row in enumerate(head_rows):
                slots = len(row.logical_positions)
                if common_slots is None:
                    common_slots = slots
                if (
                    slots != common_slots
                    or slots < 1
                    or tuple(row.keys.shape) != (slots, key.shape[-1])
                    or row.values.shape != row.keys.shape
                    or row.log_bias.shape != (slots,)
                    or row.keys.dtype != key.dtype
                    or row.values.dtype != value.dtype
                    or row.log_bias.dtype != torch.float32
                    or any(
                        not torch.isfinite(t).all() for t in (row.keys, row.values, row.log_bias)
                    )
                    or (row.log_bias < 0).any()
                    or type(row.logical_positions) is not tuple
                    or any(type(p) is not int for p in row.logical_positions)
                    or sorted(set(row.logical_positions)) != list(row.logical_positions)
                    or any(
                        not 0 <= p < reference.active_cache_length for p in row.logical_positions
                    )
                    or not set(required_logical_positions).issubset(row.logical_positions)
                ):
                    raise ValueError("invalid compaction head, physical capacity, or protected map")
                k, v, b = (
                    row.keys.to(key.device),
                    row.values.to(value.device),
                    row.log_bias.to(key.device),
                )
                indices = torch.tensor(row.logical_positions, device=key.device)
                if not torch.equal(k, key[0, head, indices]):
                    raise ValueError("compaction must retain original post-RoPE keys bitwise")
                for position in required_logical_positions:
                    slot = row.logical_positions.index(position)
                    if b[slot] != 0 or not torch.equal(v[slot], value[0, head, position]):
                        raise ValueError("compaction changed a mandatory value or its zero bias")
                if retention_one and (
                    row.logical_positions != tuple(range(reference.active_cache_length))
                    or not torch.equal(v, value[0, head])
                    or (b != 0).any()
                ):
                    raise ValueError("retention one must be an identity without fitted values/bias")
                ks.append(k)
                vs.append(v)
                bs.append(b)
                ps.append(row.logical_positions)
            layer = BoundedHistoryLayer(
                torch.stack(ks)[None],
                torch.stack(vs)[None],
                generated_capacity=generated_capacity,
                sink_tokens=sink_tokens,
                next_logical_position=reference.next_decode_position,
            )
            bias = torch.stack(bs).contiguous()
            query_heads = adapter._language_layers()[i].self_attn.config.num_attention_heads
            groups, remainder = divmod(query_heads, key.shape[1])
            if remainder:
                raise ValueError("invalid GQA head grouping")
            mask = torch.zeros(
                (1, query_heads, 1, layer._capacity), dtype=torch.float32, device=key.device
            )
            mask[0, :, 0, :common_slots].copy_(bias.repeat_interleave(groups, dim=0))
            layers.append(layer)
            buffers.append(mask)
            biases.append(bias)
            positions.append(tuple(ps))
        cache = DynamicCache()
        cache.layers = layers
        cache._mosaickv_bounded_history = True
        cache._mosaickv_independent_layer_lengths = False
        metadata = copy.deepcopy(reference.model_state)
        metadata.update(
            mosaickv_packed_all_valid=True,
            mosaickv_cache_all_valid=True,
            mosaickv_bounded_history=True,
            mosaickv_bounded_history_capacity=common_slots + generated_capacity,
            mosaickv_bounded_history_masks=tuple(buffers),
            attention_compaction_log_bias=tuple(biases),
            attention_compaction_logical_positions=tuple(positions),
            attention_compaction_cap_bytes=cap_bytes,
            attention_compaction_representation="original_keys_synthetic_values_mass_bias_v1",
            attention_compaction_requires_model_validation=True,
        )
        state = DecodeState(
            cache,
            torch.ones(
                (1, common_slots),
                dtype=reference.attention_mask.dtype,
                device=reference.attention_mask.device,
            ),
            common_slots,
            reference.logical_sequence_length,
            reference.next_decode_position,
            reference.modality_map,
            metadata,
        )
    bias_cache_storage(adapter, state)
    return state
