"""Evaluation-only shape-matched FullKV boundary transport.

The FP16 native operation and weights are unchanged. A masked dummy row makes
the prefix prefill use the original prompt's GEMM shapes; the dummy cache slot
is then removed. During single-token forwards only language Linear and eager
attention query dimensions are padded to that same shape. A separately enabled
normalization phase can also pad native language RMSNorm calls, only while
processing the held final-prompt token. The default historical path is unchanged.
Held-token operator padding never enters the KV cache or logical sequence. The
temporary masked prefix-cache slot is removed before that forward. This expensive
correctness diagnostic is not an online compression policy or a repair of
historical results.
"""

from __future__ import annotations

import importlib
import math
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


def padded_normalization(original, inputs, rows):
    """Run the unchanged native norm on a full-row buffer, retaining one row.

    This preserves its equation, precision, weights and epsilon, not the native
    singleton reduction's launch geometry or necessarily its bitwise result.
    Unlike ``padded_linear``, this helper must never accept a prefill silently:
    its caller explicitly authorizes exactly one held-token boundary phase.
    """
    import torch

    if type(rows) is not int or not 2 <= rows <= 4096:
        raise ValueError("registered normalization row count must be 2..4096")
    if (
        not isinstance(inputs, torch.Tensor)
        or inputs.ndim != 3
        or tuple(inputs.shape[:2]) != (1, 1)
        or not inputs.is_floating_point()
    ):
        raise NotImplementedError("normalization phase requires one batch-one token")
    buffer = inputs.new_zeros((1, rows, inputs.shape[-1]))
    buffer[:, -1:, :].copy_(inputs)
    output = original(buffer)
    if (
        not isinstance(output, torch.Tensor)
        or output.shape != buffer.shape
        or output.dtype != inputs.dtype
        or output.device != inputs.device
    ):
        raise ValueError("native normalization changed output geometry/precision/device")
    # A one-row slice can already be contiguous while retaining the entire
    # padded allocation. Return only its own row-sized backing storage.
    return output[:, -1:, :].clone(memory_format=torch.contiguous_format)


class ShapeTransportOperations(dict):
    """JSON-compatible counters with an explicitly scoped, single-use phase."""

    def __init__(self, *, native_norm_boundary=False, norm_names=()):
        super().__init__(linear_padded_calls=0, attention_padded_calls=0, observed_prefill_layers=0)
        self._norm_enabled = native_norm_boundary
        self._norm_names = tuple(norm_names)
        self._norm_calls = dict.fromkeys(self._norm_names, 0)
        self._norm_active = False
        self._closed = False
        if native_norm_boundary:
            self.update(
                normalization_phase_entries=0,
                normalization_phase_completions=0,
                normalization_padded_calls=0,
                normalization_expected_calls=len(self._norm_names),
                normalization_input_calls=0,
                normalization_post_attention_calls=0,
                normalization_final_calls=0,
            )

    @contextmanager
    def boundary_normalization(self):
        """Authorize only one held-token forward, never prefill/continuation."""
        if not self._norm_enabled or self._closed:
            raise ValueError("open opt-in boundary normalization transport required")
        if self._norm_active or self["normalization_phase_entries"]:
            raise ValueError("the held-token normalization phase is single-use, not nested")
        self._norm_active = True
        self["normalization_phase_entries"] += 1
        try:
            yield
            if any(count != 1 for count in self._norm_calls.values()):
                raise ValueError("one native normalization call per language norm is required")
            self["normalization_phase_completions"] += 1
        finally:
            self._norm_active = False


def _language_normalizations(adapter, layers, modeling):
    """Resolve every decoder norm and the unique final norm, never vision norms."""
    model = getattr(adapter, "model", None)
    if model is None:
        raise NotImplementedError("an explicit language decoder owner is required")
    owners = [
        module
        for module in model.modules()
        if getattr(module, "layers", None) is not None and tuple(module.layers) == layers
    ]
    if len(owners) != 1:
        raise NotImplementedError("language decoder parent is not uniquely exposed")
    norms = []
    for index, layer in enumerate(layers):
        norms.extend(
            (f"layer.{index}.{name}", getattr(layer, name, None))
            for name in ("input_layernorm", "post_attention_layernorm")
        )
    norms.append(("final_norm", getattr(owners[0], "norm", None)))
    if len({id(norm) for _, norm in norms}) != len(norms):
        raise NotImplementedError("distinct owned language normalization modules required")
    for _, norm in norms:
        if (
            type(norm) is not modeling.MistralRMSNorm
            or norm.training
            or getattr(norm.forward, "__func__", None) is not modeling.MistralRMSNorm.forward
            or norm.weight.ndim != 1
            or norm.weight.numel() == 0
            or not math.isfinite(norm.variance_epsilon)
            or norm.variance_epsilon <= 0
        ):
            raise NotImplementedError("unchanged eval-mode native Mistral RMSNorm required")
    return norms


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
def shape_matched_transport(adapter, rows, *, native_norm_boundary=False):
    """Serial owner-scoped override; restore every function even on exceptions."""
    import torch

    if type(rows) is not int or rows < 2 or rows > 4096:
        raise ValueError("registered shape-matched prompt length must be 2..4096")
    if type(native_norm_boundary) is not bool:
        raise ValueError("native_norm_boundary requires an explicit boolean")
    layers = tuple(adapter._language_layers())
    if not layers or any(
        layer.training or layer.self_attn.config._attn_implementation != "eager" for layer in layers
    ):
        raise NotImplementedError("eval-mode eager decoder required")
    modules = {importlib.import_module(type(layer.self_attn).__module__) for layer in layers}
    if any(module.__name__ != "transformers.models.mistral.modeling_mistral" for module in modules):
        raise NotImplementedError("only the registered Mistral GQA decoder is audited")
    owners = {id(layer.self_attn) for layer in layers}
    norms = (
        _language_normalizations(adapter, layers, next(iter(modules)))
        if native_norm_boundary
        else ()
    )
    strides = {}
    counts = ShapeTransportOperations(
        native_norm_boundary=native_norm_boundary, norm_names=[name for name, _ in norms]
    )
    with ExitStack() as stack:
        # Closed handles cannot later activate a phase after all overrides have
        # been restored, including when the model forward raises an exception.
        stack.callback(setattr, counts, "_closed", True)
        for name, norm in norms:
            original = norm.forward

            def normalization(inputs, *, _original=original, _name=name, _norm=norm):
                if not counts._norm_active:
                    return _original(inputs)
                if not isinstance(inputs, torch.Tensor) or inputs.ndim != 3:
                    raise NotImplementedError("normalization phase requires a native B,N,H tensor")
                if inputs.dtype != _norm.weight.dtype or inputs.device != _norm.weight.device:
                    raise ValueError("native normalization input/weight precision/device mismatch")
                if inputs.shape[-1] != _norm.weight.numel():
                    raise ValueError("native normalization hidden dimension changed")
                if counts._norm_calls[_name]:
                    raise ValueError("a second forward entered the held-token normalization phase")
                output = padded_normalization(_original, inputs, rows)
                counts["normalization_padded_calls"] += 1
                counts._norm_calls[_name] += 1
                group = (
                    "final"
                    if _name == "final_norm"
                    else "post_attention"
                    if _name.endswith("post_attention_layernorm")
                    else "input"
                )
                counts[f"normalization_{group}_calls"] += 1
                return output

            stack.enter_context(patch.object(norm, "forward", normalization))
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
