"""Lossless BF16 minima ONLY; retain original FP32 scales and every correction.

Unlike the old lossy FP16-header ablation, accept conversion only after a bitwise
FP32 round trip. Ranges originated in BF16 input minima; scales need not be BF16
representable and are NEVER changed. This is a common representation option.
"""

from dataclasses import replace

import torch

from mosaickv.evaluation.precision_tier import QuantizedBody


def compact(body):
    if (
        not isinstance(body, QuantizedBody)
        or body.codebook is not None
        or body.minimum.dtype not in (torch.float32, torch.bfloat16)
        or body.scale.dtype != torch.float32
        or not bool(torch.isfinite(body.minimum).all())
        or not bool(torch.isfinite(body.scale).all())
        or not bool((body.scale > 0).all())
    ):
        raise ValueError("finite affine metadata and unchanged positive FP32 scale required")
    if body.minimum.dtype == torch.bfloat16:
        return body
    minimum = body.minimum.bfloat16().contiguous()
    if not torch.equal(minimum.float().view(torch.int32), body.minimum.view(torch.int32)):
        raise ValueError("minimum is not exactly BF16 representable; no lossy fallback permitted")
    return replace(body, minimum=minimum)


def compact_format(packet):
    from mosaickv.evaluation.exact_storage import ExactFormat
    from mosaickv.evaluation.head_local_precision import HeadLocalFormat
    from mosaickv.evaluation.precision_tier import PrecisionFormat
    from mosaickv.evaluation.shared_source_format import SharedSourceFormat
    from mosaickv.evaluation.text_source_storage import TextSourceFormat

    if isinstance(packet, (TextSourceFormat, HeadLocalFormat)):
        return replace(packet, base=compact_format(packet.base))
    if isinstance(packet, PrecisionFormat):
        return replace(packet, pair=tuple(compact(body) for body in packet.pair))
    if (
        isinstance(packet, SharedSourceFormat)
        and packet.input_overlay is None
        and packet.channel_exceptions is None
    ):
        # KEEP the original transport patches, exact bodies and head tiling.
        return replace(packet, source=compact(packet.source))
    if isinstance(packet, ExactFormat) and packet.kind == "kv":
        return packet
    raise NotImplementedError("unaudited packet cannot use lossless minimum compaction")


@torch.inference_mode()
def prove_and_compact(adapter, packets):
    converted, records = [], []
    for i, (packet, block) in enumerate(zip(packets, adapter._language_layers(), strict=True)):
        smaller = compact_format(packet)
        original, actual = packet.materialize(block.self_attn), smaller.materialize(block.self_attn)
        if not all(
            torch.equal(a.view(torch.int16), b.view(torch.int16))
            for a, b in zip(original, actual, strict=True)
        ):
            raise RuntimeError("lossless minimum compaction changed a decoded K/V bit")
        converted.append(smaller)
        records.append(
            dict(
                layer=i,
                decoded_KV_bitwise=True,
                original_component_bytes=packet.component_bytes,
                compact_component_bytes=smaller.component_bytes,
                original_FP32_scale_and_transport_corrections_unchanged=True,
            )
        )
    return tuple(converted), records
