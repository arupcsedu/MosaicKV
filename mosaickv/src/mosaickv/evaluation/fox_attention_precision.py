"""Evaluation-only FP32 eager-attention arithmetic experiment.

Model weights, hidden-output dtype, and stored K/V are unchanged. QK, softmax,
and probability/value multiplication use FP32 temporaries. This is a different
arithmetic policy, NOT a silent fix or a waiver of an old numerical failure.
It is never enabled by the compressor or a default production configuration.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import math
import threading
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

_LOCK = threading.RLock()
_ACTIVE = False
_MODULES = {
    "transformers.models.llama.modeling_llama",
    "transformers.models.mistral.modeling_mistral",
}


def fp32_eager_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """Mirror installed HF eager semantics, changing only intermediate precision."""
    import torch

    del kwargs
    if module.training or dropout != 0:
        raise ValueError("precision diagnosis supports inference without dropout only")
    if not math.isfinite(scaling) or scaling <= 0:
        raise ValueError("invalid attention scaling")
    if query.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("unsupported attention query dtype")
    if key.dtype != query.dtype or value.dtype != key.dtype:
        raise ValueError("Q/K/V dtype mismatch")
    if any(x.ndim != 4 for x in (query, key, value)):
        raise ValueError("expected B,H,N,D tensors")
    groups = int(module.num_key_value_groups)
    if (
        groups < 1
        or query.shape[1] != key.shape[1] * groups
        or key.shape[:3] != value.shape[:3]
        or query.shape[0] != key.shape[0]
        or query.shape[-1] != key.shape[-1]
    ):
        raise ValueError("incompatible grouped-query attention geometry")
    k = key.float().repeat_interleave(groups, dim=1)
    v = value.float().repeat_interleave(groups, dim=1)
    scores = torch.matmul(query.float(), k.transpose(2, 3)) * scaling
    if attention_mask is not None:
        if not attention_mask.is_floating_point() or attention_mask.shape[-1] < key.shape[2]:
            raise ValueError("expected a complete additive floating-point mask")
        scores = scores + attention_mask[..., : key.shape[2]].float()
    probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32)
    output = torch.matmul(probabilities, v).transpose(1, 2).contiguous()
    return output.to(query.dtype), probabilities.to(query.dtype)


@contextmanager
def attention_precision(adapter, policy):
    """Scoped, serial, model-specific override; restore functions/TF32 on errors."""
    import torch

    global _ACTIVE
    if policy not in {"native_fp16", "fp32_attention"}:
        raise ValueError("unregistered attention precision policy")
    with _LOCK:
        if _ACTIVE:
            raise RuntimeError("precision experiments cannot be nested")
        attentions = tuple(layer.self_attn for layer in adapter._language_layers())
        if not attentions:
            raise ValueError("no decoder attention modules")
        modules = {}
        for attention in attentions:
            name = type(attention).__module__
            if name not in _MODULES or attention.config._attn_implementation != "eager":
                raise NotImplementedError("only audited Llama/Mistral eager attention is supported")
            modules[name] = importlib.import_module(name)
        functions = {name: module.eager_attention_forward for name, module in modules.items()}
        metadata = dict(
            policy=policy,
            weights_and_cache_dtype_unchanged=True,
            changes_default_runtime=False,
            process_global_tf32_scope="serial evaluation context only",
            source_functions={
                name: hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()
                for name, fn in functions.items()
            },
        )
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        identities = {id(module) for module in attentions}
        _ACTIVE = True
        try:
            # Explicitly shared by native and candidate diagnostic conditions.
            torch.backends.cuda.matmul.allow_tf32 = False
            with ExitStack() as stack:
                if policy == "fp32_attention":
                    for name, module in modules.items():
                        original = functions[name]

                        def wrapper(owner, *args, _original=original, **kwargs):
                            fn = fp32_eager_attention if id(owner) in identities else _original
                            return fn(owner, *args, **kwargs)

                        stack.enter_context(
                            patch.object(module, "eager_attention_forward", wrapper)
                        )
                yield metadata
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
            _ACTIVE = False
