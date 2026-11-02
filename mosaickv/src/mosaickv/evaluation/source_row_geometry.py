"""Optional hardware geometry, never a different source/cache representation."""

from dataclasses import replace

import torch

from mosaickv.evaluation.shared_source_format import SharedSourceFormat
from mosaickv.evaluation.text_source_storage import TextSourceFormat

GEOMETRIES = ((16, 4), (32, 4), (64, 4), (32, 2))


def geometry(packet, rows, heads):
    if type(rows) is not int or type(heads) is not int or (rows, heads) not in GEOMETRIES:
        raise ValueError("one of the four preregistered row/head geometries required")
    if isinstance(packet, TextSourceFormat):
        return replace(packet, base=geometry(packet.base, rows, heads))
    if not isinstance(packet, SharedSourceFormat) or packet.projection_backend != "fused":
        raise NotImplementedError("fused shared-source geometry only")
    if (
        packet.shape[-1] != 128
        or packet.shape[1] % heads
        or packet.source.minimum.dtype not in (torch.float32, torch.bfloat16)
        or packet.source.scale.dtype != torch.float32
    ):
        raise ValueError("audited complete full-MHA source metadata required")
    return replace(packet, source_head_group=heads, source_row_tile=rows)


def eligible(packet):
    if isinstance(packet, TextSourceFormat):
        return eligible(packet.base)
    return isinstance(packet, SharedSourceFormat) and packet.projection_backend == "fused"


def apply(packets, spec):
    if not isinstance(spec, (tuple, list)) or len(spec) != 2:
        raise ValueError("explicit row/head geometry required")
    rows, heads = spec
    # Validate even when a baseline has no eligible source projection.
    if type(rows) is not int or type(heads) is not int or (rows, heads) not in GEOMETRIES:
        raise ValueError("unsupported hardware geometry")
    return tuple(geometry(p, rows, heads) if eligible(p) else p for p in packets)
