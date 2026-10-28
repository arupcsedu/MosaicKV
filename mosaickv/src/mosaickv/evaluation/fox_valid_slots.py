"""Evaluation-only physical-hole canonicalization, not an online default.

Execute the installed eager function unchanged over valid slots. Return its
probabilities on the original physical axis. Precision and stored KV do not
change. This is a separately measured backend variant, never a waiver of old
native-layout failures.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager


def valid_slot_forward(original, module, query, key, value, mask, *args, **kwargs):
    import torch

    if query.shape[-2] != 1 or mask is None:
        return original(module, query, key, value, mask, *args, **kwargs)
    if mask.ndim != 4 or mask.shape[0] != 1 or mask.shape[-2] != 1:
        raise NotImplementedError("valid-slot diagnostic requires batch-one decode masks")
    active_mask = mask[..., : key.shape[-2]]
    if not bool(torch.isneginf(active_mask).any()):
        # Strong identity: unchanged objects and installed operation.
        return original(module, query, key, value, mask, *args, **kwargs)
    if not bool(((active_mask == 0) | torch.isneginf(active_mask)).all()):
        raise NotImplementedError("valid-slot diagnostic does not support attention biases")
    keep = active_mask[0, :, 0, :] == 0
    query_heads, kv_heads = query.shape[1], key.shape[1]
    if query_heads % kv_heads or keep.shape[0] not in {1, query_heads}:
        raise NotImplementedError("unsupported query/KV ownership")
    keep = keep.expand(query_heads, -1)
    groups = query_heads // kv_heads
    grouped = keep.reshape(kv_heads, groups, -1)
    if not bool((grouped == grouped[:, :1]).all()):
        raise NotImplementedError("GQA query heads disagree on their owned KV slots")
    kv_keep = grouped[:, 0]
    counts = kv_keep.sum(-1)
    if not bool((counts == counts[0]).all()):
        raise NotImplementedError("unequal per-head capacities require a separate ragged kernel")
    if int(counts[0]) == 0:
        raise ValueError("all attention slots invalid")
    indices = torch.stack([row.nonzero(as_tuple=False).flatten() for row in kv_keep])
    gather = indices[None, :, :, None].expand(1, kv_heads, -1, key.shape[-1])
    compact_key = key.gather(-2, gather).contiguous()
    compact_value = value.gather(-2, gather).contiguous()
    query_indices = indices.repeat_interleave(groups, dim=0)[None, :, None, :]
    compact_mask = (
        active_mask.expand(1, query_heads, 1, key.shape[-2]).gather(-1, query_indices).contiguous()
    )
    output, compact_probabilities = original(
        module, query, compact_key, compact_value, compact_mask, *args, **kwargs
    )
    probabilities = compact_probabilities.new_zeros(
        (*compact_probabilities.shape[:-1], key.shape[-2])
    )
    probabilities.scatter_(-1, query_indices, compact_probabilities)
    return output, probabilities


@contextmanager
def valid_slot_diagnostic(adapter):
    """Owner-scoped installed-function wrapper; no global precision changes."""
    import importlib
    from unittest.mock import patch

    layers = tuple(adapter._language_layers())
    if any(layer.training for layer in layers):
        raise ValueError("evaluation-only valid-slot diagnostic")
    if any(layer.self_attn.config._attn_implementation != "eager" for layer in layers):
        raise NotImplementedError("valid-slot diagnostic requires eager attention")
    modules = {importlib.import_module(type(layer.self_attn).__module__) for layer in layers}
    if any(
        m.__name__.split(".")[-1] not in {"modeling_llama", "modeling_mistral"} for m in modules
    ):
        raise NotImplementedError("diagnostic supports only installed Llama/Mistral eager")
    owners = {id(layer.self_attn) for layer in layers}
    with ExitStack() as stack:
        for modeling in modules:
            original = modeling.eager_attention_forward

            def wrapped(
                module, query, key, value, attention_mask, *args, _original=original, **kwargs
            ):
                if id(module) not in owners:
                    return _original(module, query, key, value, attention_mask, *args, **kwargs)
                return valid_slot_forward(
                    _original, module, query, key, value, attention_mask, *args, **kwargs
                )

            stack.enter_context(patch.object(modeling, "eager_attention_forward", wrapped))
        yield dict(
            policy="diagnostic_valid_slot_canonicalization",
            weights_and_stored_KV="unchanged_fp16",
            all_valid_path="identical_objects_and_installed_function",
            online_default=False,
        )
