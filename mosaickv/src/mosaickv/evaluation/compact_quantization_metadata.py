"""Explicit lossy header variant; codes, row membership and values stay distinct.

This is a common quantized-format option, not a special baseline exemption.
Never clip metadata or silently fall back after overflow/underflow.
"""

from dataclasses import replace

import torch

from mosaickv.evaluation.precision_tier import QuantizedBody


def compact(body):
    if (
        not isinstance(body, QuantizedBody)
        or body.codebook is not None
        or body.minimum.dtype != torch.float32
        or body.scale.dtype != torch.float32
    ):
        raise ValueError("registered affine FP32 quantization metadata required")
    minimum, scale = (t.to(torch.float16).contiguous() for t in (body.minimum, body.scale))
    if not (torch.isfinite(minimum).all() and torch.isfinite(scale).all() and (scale > 0).all()):
        raise ValueError("FP16 metadata would overflow or lose positive scale; unsupported")
    return replace(body, minimum=minimum, scale=scale)
