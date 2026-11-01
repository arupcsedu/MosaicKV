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


def compact_format(packet):
    """Same option for every applicable compared format; never changes selection."""
    from mosaickv.evaluation.head_local_precision import HeadLocalFormat
    from mosaickv.evaluation.precision_tier import PrecisionFormat
    from mosaickv.evaluation.shared_source_format import SharedSourceFormat
    from mosaickv.evaluation.text_source_storage import TextSourceFormat

    if isinstance(packet, (TextSourceFormat, HeadLocalFormat)):
        return replace(packet, base=compact_format(packet.base))
    if isinstance(packet, PrecisionFormat):
        return replace(packet, pair=tuple(compact(b) for b in packet.pair))
    if isinstance(packet, SharedSourceFormat) and packet.input_overlay is None:
        return replace(packet, source=compact(packet.source), transport_patches=())
    raise NotImplementedError("unaudited format cannot use compact affine metadata")
