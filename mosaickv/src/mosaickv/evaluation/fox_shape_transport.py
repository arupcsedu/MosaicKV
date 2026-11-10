"""Evaluation-only shape-matched FullKV boundary transport.

The FP16 native operation and weights are unchanged. A masked dummy row makes
the prefix prefill use the original prompt's GEMM shapes; the dummy cache slot
is then removed. During single-token forwards only language Linear and eager
attention query dimensions are padded to that same shape. Fake rows never
enter the KV cache or logical sequence. This expensive correctness diagnostic
is not an online compression policy or a silent repair of historical results.
"""

from __future__ import annotations

import importlib
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from unittest.mock import patch


def padded_prefix(prefix, *, placeholder_id=0):
    import torch

    from mosaickv.adapters.huggingface.types import PreparedInputs

    n = prefix.logical_sequence_length
    if type(placeholder_id) is not int or placeholder_id < 0:
        raise ValueError("explicit nonnegative dummy token ID required")
    ids = prefix.model_inputs.get("input_ids")
    mask = prefix.model_inputs.get("attention_mask")
    if ids is None or mask is None or tuple(ids.shape) != (1, n) or mask.shape != ids.shape:
        raise NotImplementedError("batch-one explicit expanded IDs/mask required")
    if not bool((mask == 1).all()):
        raise NotImplementedError("only the original all-valid prompt prefix is registered")
    inputs = dict(prefix.model_inputs)
    inputs["input_ids"] = torch.cat((ids, ids.new_full((1, 1), placeholder_id)), dim=-1)
    inputs["attention_mask"] = torch.cat((mask, mask.new_zeros((1, 1))), dim=-1)
    if "position_ids" in inputs:
        positions = inputs["position_ids"]
        if tuple(positions.shape) != (1, n):
            raise NotImplementedError("unsupported explicit position-ID geometry")
        inputs["position_ids"] = torch.cat((positions, positions.new_full((1, 1), n)), dim=-1)
    if "token_type_ids" in inputs:
        types = inputs["token_type_ids"]
        if tuple(types.shape) != (1, n):
            raise NotImplementedError("unsupported token-type geometry")
        inputs["token_type_ids"] = torch.cat((types, types.new_zeros((1, 1))), dim=-1)
    spans = prefix.modality_map
    if (
        not spans
        or spans[-1].end != n
        or getattr(spans[-1].modality, "value", spans[-1].modality) != "text"
    ):
        raise NotImplementedError("the dummy extends only a validated text tail")
    return PreparedInputs(inputs, (*spans[:-1], replace(spans[-1], end=n + 1)), n + 1)


def remove_dummy_cache_slot(adapter, state, prefix):
    """Recreate a private cache of the original kind; remove exactly one slot."""
    from mosaickv.adapters.huggingface.types import CacheLayerSnapshot, DecodeState

    n = prefix.logical_sequence_length
    if any(
        value != n + 1
        for value in (
            state.active_cache_length,
            state.logical_sequence_length,
            state.next_decode_position,
        )
    ):
        raise ValueError("dummy prefill did not create exactly one extra position")
    snapshot = adapter.borrow_past_key_values(state.past_key_values)
    if snapshot.independent_layer_lengths:
        raise NotImplementedError("this FullKV control does not support ragged layers")
    layers = []
    for layer in snapshot.layers:
        if (
            layer.sequence_dimension != 2
            or layer.key.ndim != 4
            or layer.key.shape[2] != n + 1
            or layer.value.shape[2] != n + 1
        ):
            raise NotImplementedError("explicit B,H,N,D cache required")
        layers.append(
            CacheLayerSnapshot(layer.key[:, :, :n, :].clone(), layer.value[:, :, :n, :].clone(), 2)
        )
    cache = adapter.inject_past_key_values(
        replace(snapshot, layers=tuple(layers), active_sequence_length=n)
    )
    model_state = dict(state.model_state)
    model_state["mosaickv_cache_all_valid"] = bool(
        (prefix.model_inputs["attention_mask"] == 1).all()
    )
    return DecodeState(
        cache,
        prefix.model_inputs["attention_mask"].clone(),
        n,
        n,
        n,
        prefix.modality_map,
        model_state,
    )


def padded_linear(original, inputs, rows):
    """Keep identical native FP16 arithmetic and one meaningful output row."""
    if inputs.ndim != 3 or inputs.shape[0] != 1:
        raise NotImplementedError("shape transport requires batch-one language projections")
    if inputs.shape[1] != 1 or rows == 1:
        return original(inputs)
    buffer = inputs.new_zeros((1, rows, inputs.shape[-1]))
    buffer[:, -1:, :].copy_(inputs)
    return original(buffer)[:, -1:, :].contiguous()


def padded_attention(original, module, query, key, value, mask, *args, rows, stride, **kwargs):
    import torch

    if query.shape[-2] != 1 or rows == 1:
        return original(module, query, key, value, mask, *args, **kwargs)
    if query.ndim != 4 or query.shape[0] != 1 or key.shape[0] != 1:
        raise NotImplementedError("batch-one B,H,N,D attention required")
    if mask is not None and (mask.ndim != 4 or mask.shape[-2] != 1):
        raise NotImplementedError("one-row additive decode mask required")
    padded = torch.empty_strided(
        (1, query.shape[1], rows, query.shape[-1]), stride, dtype=query.dtype, device=query.device
    )
    padded.zero_()
    padded[:, :, -1:, :].copy_(query)
    if mask is None:
        expanded_mask = query.new_zeros((1, 1, rows, key.shape[-2]))
    else:
        expanded_mask = (
            mask[..., : key.shape[-2]].expand(*mask.shape[:-2], rows, key.shape[-2]).contiguous()
        )
    output, weights = original(module, padded, key, value, expanded_mask, *args, **kwargs)
    return output[:, -1:, ...].contiguous(), weights[..., -1:, :].contiguous()


@contextmanager
def shape_matched_transport(adapter, rows):
    """Serial owner-scoped override; restore every function even on exceptions."""
    import torch

    if type(rows) is not int or rows < 2 or rows > 4096:
        raise ValueError("registered shape-matched prompt length must be 2..4096")
    layers = tuple(adapter._language_layers())
    if not layers or any(
        layer.training or layer.self_attn.config._attn_implementation != "eager" for layer in layers
    ):
        raise NotImplementedError("eval-mode eager decoder required")
    modules = {importlib.import_module(type(layer.self_attn).__module__) for layer in layers}
    if any(module.__name__ != "transformers.models.mistral.modeling_mistral" for module in modules):
        raise NotImplementedError("only the registered Mistral GQA decoder is audited")
    owners = {id(layer.self_attn) for layer in layers}
    strides, counts = (
        {},
        dict(linear_padded_calls=0, attention_padded_calls=0, observed_prefill_layers=0),
    )
    with ExitStack() as stack:
        for layer in layers:
            for module in layer.modules():
                if not isinstance(module, torch.nn.Linear):
                    continue
                original = module.forward

                def linear(inputs, *, _original=original):
                    if inputs.ndim == 3 and inputs.shape[1] == 1:
                        counts["linear_padded_calls"] += 1
                    return padded_linear(_original, inputs, rows)

                stack.enter_context(patch.object(module, "forward", linear))
        for modeling in modules:
            original = modeling.eager_attention_forward

            def attention(module, query, key, value, mask, *args, _original=original, **kwargs):
                if id(module) not in owners:
                    return _original(module, query, key, value, mask, *args, **kwargs)
                if query.shape[-2] == rows:
                    strides[id(module)] = tuple(query.stride())
                    counts["observed_prefill_layers"] = len(strides)
                    return _original(module, query, key, value, mask, *args, **kwargs)
                if query.shape[-2] != 1 or id(module) not in strides:
                    raise ValueError("native full-shape prefill must precede held-token decoding")
                counts["attention_padded_calls"] += 1
                return padded_attention(
                    _original,
                    module,
                    query,
                    key,
                    value,
                    mask,
                    *args,
                    rows=rows,
                    stride=strides[id(module)],
                    **kwargs,
                )

            stack.enter_context(patch.object(modeling, "eager_attention_forward", attention))
        yield counts
