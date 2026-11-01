"""Change hardware tiling only; never recode, recalibrate or change cache bytes."""

from dataclasses import replace

import torch

from mosaickv.evaluation.shared_source_format import SharedSourceFormat
from mosaickv.evaluation.text_source_storage import TextSourceFormat


def retile(packet, head_group):
    if type(head_group) is not int or head_group not in (1, 2, 4):
        raise ValueError("explicit audited head group required")
    if isinstance(packet, TextSourceFormat):
        return replace(packet, base=retile(packet.base, head_group))
    if not isinstance(packet, SharedSourceFormat) or packet.projection_backend != "fused":
        raise NotImplementedError("fused shared-source packet required")
    if packet.source.minimum.dtype != torch.float32 or packet.source.scale.dtype != torch.float32:
        raise ValueError("this gate preserves original FP32 metadata")
    if packet.shape[1] % head_group:
        raise ValueError("head group must divide KV head count")
    return replace(packet, source_head_group=head_group)
