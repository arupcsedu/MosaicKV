"""Optional hardware geometry, never a different source/cache representation."""

from dataclasses import replace

import torch

from mosaickv.evaluation.shared_source_format import SharedSourceFormat
from mosaickv.evaluation.text_source_storage import TextSourceFormat

GEOMETRIES = ((16, 4), (32, 4), (64, 4), (32, 2))


def geometry(packet, rows, heads, *, num_warps=None):
    if type(rows) is not int or type(heads) is not int or (rows, heads) not in GEOMETRIES:
        raise ValueError("one of the four preregistered row/head geometries required")
    if num_warps is not None and (type(num_warps) is not int or num_warps not in (4, 8, 16)):
        raise ValueError("explicit supported warp count required")
    if isinstance(packet, TextSourceFormat):
        return replace(packet, base=geometry(packet.base, rows, heads, num_warps=num_warps))
    if not isinstance(packet, SharedSourceFormat) or packet.projection_backend != "fused":
        raise NotImplementedError("fused shared-source geometry only")
    if (
        packet.shape[-1] != 128
        or packet.shape[1] % heads
        or packet.source.minimum.dtype not in (torch.float32, torch.bfloat16)
        or packet.source.scale.dtype != torch.float32
    ):
        raise ValueError("audited complete full-MHA source metadata required")
    return replace(
        packet, source_head_group=heads, source_row_tile=rows, source_num_warps=num_warps
    )


def eligible(packet):
    if isinstance(packet, TextSourceFormat):
        return eligible(packet.base)
    return isinstance(packet, SharedSourceFormat) and packet.projection_backend == "fused"


def apply(packets, spec, *, num_warps=None):
    if not isinstance(spec, (tuple, list)) or len(spec) != 2:
        raise ValueError("explicit row/head geometry required")
    rows, heads = spec
    # Validate even when a baseline has no eligible source projection.
    if type(rows) is not int or type(heads) is not int or (rows, heads) not in GEOMETRIES:
        raise ValueError("unsupported hardware geometry")
    if num_warps is not None and (type(num_warps) is not int or num_warps not in (4, 8, 16)):
        raise ValueError("explicit supported warp count required")
    return tuple(
        geometry(p, rows, heads, num_warps=num_warps) if eligible(p) else p for p in packets
    )


def paired_resource_controls(packets, adapter, spec):
    """Do not introduce hidden paired-projection spills during construction."""
    from mosaickv.evaluation.shared_source_kernel import project

    result = []
    for layer, (packet, block) in enumerate(zip(packets, adapter._language_layers(), strict=True)):
        baseline = geometry(packet, 16, 4)
        proposed = geometry(packet, *spec)
        targets = tuple(
            torch.empty(
                packet.shape, device=next(block.self_attn.parameters()).device, dtype=torch.bfloat16
            )
            for _ in range(2)
        )
        resources = []
        for index, p in enumerate((baseline, proposed)):
            template = p.base if isinstance(p, TextSourceFormat) else p
            if index == 0:
                template = replace(template, source_head_group=1, source_row_tile=16)
            resources.append(
                project(template, block.self_attn, targets, component=-1, inspect_kernel=True)
            )
        if (
            any(type(r.get("spills")) is not int for r in resources)
            or resources[1]["spills"] > resources[0]["spills"]
        ):
            raise RuntimeError("unverified or increased paired projection spills")
        result.append(dict(layer=layer, baseline=resources[0], proposed=resources[1]))
        del targets
    return result
