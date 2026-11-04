"""Experimental prompt-only tiled reader; NOT an enabled HF attention backend.

Exact BF16 decompression, but changed reduction order may change attention bits.
The diagnostic must reject model integration unless the unchanged bitwise gate
passes. No CUDA imports or kernels execute merely by importing this module.
"""

import torch


def bits_equal(a, b):
    """Unlike float torch.equal, distinguish signed zero and retain NaN payloads."""
    return (
        a.dtype == b.dtype
        and a.shape == b.shape
        and a.device == b.device
        and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
    )


def validate(query, key, value):
    from mosaickv.evaluation.lossless_bf16 import PackedBF16

    if (
        not isinstance(key, PackedBF16)
        or not isinstance(value, PackedBF16)
        or len(key.shape) != 4
        or key.shape[0] != 1
        or key.shape != value.shape
        or key.shape[-1] != 128
        or not 1 <= key.shape[1] <= 32
        or not 1 <= key.shape[-2] <= 4096
        or query.shape != (1, key.shape[1], 1, 128)
        or query.dtype != torch.bfloat16
        or not query.is_contiguous()
        or any(t.device != query.device for b in (key, value) for t in b.buffers)
    ):
        raise ValueError("audited one-query BF16 MHA, same-device lossless prompt packets required")


def reference(query, key, value, *, scale):
    from mosaickv.evaluation.exact_storage import decoded

    validate(query, key, value)
    scores = torch.matmul(query, decoded(key).transpose(2, 3)) * scale
    weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    return torch.matmul(weights, decoded(value)), weights


def run(query, key, value, *, scale):
    validate(query, key, value)
    if query.device.type != "cuda":
        raise NotImplementedError("fused prototype requires actual CUDA; reference is CPU-capable")
    if type(scale) is not float or not 0 < scale <= 1:
        raise ValueError("explicit finite positive attention scaling required")
    from mosaickv.evaluation.fused_lossless_attention_kernel import execute

    return execute(query, key, value, scale)
