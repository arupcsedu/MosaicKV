"""Lossless header encoding, not the negative FP16 scale-rounding variant.

BF16 source extrema can be stored in BF16 without changing their FP32 value.
Scale remains FP32. Apply equally to every compatible quantized method.
"""

import math
from dataclasses import replace

import torch

from mosaickv.evaluation.precision_tier import QuantizedBody


def encode_minimum(body):
    if (
        not isinstance(body, QuantizedBody)
        or body.codebook is not None
        or body.minimum.dtype != torch.float32
        or body.scale.dtype != torch.float32
        or not torch.isfinite(body.minimum).all()
        or not torch.isfinite(body.scale).all()
        or not (body.scale > 0).all()
    ):
        raise ValueError("finite affine FP32 headers with positive scale required")
    minimum = body.minimum.to(torch.bfloat16).contiguous()
    if not torch.equal(
        minimum.float().view(torch.int32), body.minimum.contiguous().view(torch.int32)
    ):
        raise ValueError("minimum is not exactly BF16 representable; do not round it")
    return replace(body, minimum=minimum)


def encode_format(packet):
    from mosaickv.evaluation.exact_storage import ExactFormat
    from mosaickv.evaluation.head_local_precision import HeadLocalFormat
    from mosaickv.evaluation.precision_tier import PrecisionFormat
    from mosaickv.evaluation.shared_source_format import SharedSourceFormat
    from mosaickv.evaluation.text_source_storage import TextSourceFormat

    if isinstance(packet, (TextSourceFormat, HeadLocalFormat)):
        return replace(packet, base=encode_format(packet.base))
    if isinstance(packet, PrecisionFormat):
        return replace(packet, pair=tuple(encode_minimum(b) for b in packet.pair))
    if isinstance(packet, SharedSourceFormat):
        return replace(packet, source=encode_minimum(packet.source))
    if isinstance(packet, ExactFormat) and packet.kind == "kv":
        return packet
    raise NotImplementedError("unaudited format cannot use lossless affine minimum encoding")


def eligible_parent(control):
    """Fail-closed, fixed two-input development gate; never a paper claim."""
    try:
        if control["complete"] is not True or control["failures"]:
            return False
        summaries = control["teacher_metric_summary"]
        if len(summaries) != 2:
            return False
        tasks = {s["provenance"]["dataset"]["dataset_id"] for s in summaries}
        if tasks != {"PrefixKV/LLaVA-Description", "PrefixKV/MM-Vet"}:
            return False
        for item in summaries:
            rows = {r["variant"]: r for r in item["rows"]}
            source, anchor = rows["source3_fp32_tiled_text"], rows["shadow"]
            if not (source["evaluated_token_count"] == anchor["evaluated_token_count"] > 0):
                return False
            for row in (source, anchor):
                if (
                    not math.isfinite(row["PPL"])
                    or row["PPL"] <= 0
                    or not 0 <= row["fullkv_reference_ROUGE_L"] <= 1
                ):
                    return False
            if (
                source["PPL"] > anchor["PPL"]
                or source["fullkv_reference_ROUGE_L"] < anchor["fullkv_reference_ROUGE_L"]
            ):
                return False
        return True
    except (KeyError, TypeError, ValueError):
        return False
