"""Bounded source workspace + frozen native BF16 BMM into existing head leases."""

import torch


def source_packet(packet):
    while hasattr(packet, "base"):
        packet = packet.base
    if (
        getattr(packet, "projection_backend", None) == "native_bmm_256"
        and getattr(packet, "channel_exceptions", None) is not None
    ):
        raise NotImplementedError("coherent columns are verified only for fused projection")
    return packet if getattr(packet, "projection_backend", None) == "native_bmm_256" else None


def new_workspace(packet):
    n, width = packet.source.shape[2:]
    return torch.empty(
        (1, min(256, n), width), device=packet.source.codes.device, dtype=torch.bfloat16
    )


@torch.inference_mode()
def project_component(packet, attention, component, target, workspace):
    if type(component) is not int or component not in (0, 1):
        raise ValueError("explicit K or V component required")
    _, h, n, d = packet.shape
    width = h * d
    projection = (attention.k_proj, attention.v_proj)[component]
    if (
        workspace.shape != (1, min(256, n), width)
        or workspace.dtype != torch.bfloat16
        or workspace.device != target.device
        or workspace.device != packet.source.codes.device
        or target.dtype != torch.bfloat16
        or target.shape[:2] != (1, h)
        or target.shape[-1] != d
        or target.shape[-2] < n
        or d != 128
        or target.stride(-1) != 1
        or target.stride(-2) != d
        or projection.bias is not None
        or projection.weight.shape != (width, width)
        or projection.weight.dtype != torch.bfloat16
        or projection.weight.device != target.device
        or not projection.weight.is_contiguous()
    ):
        raise ValueError("audited BF16 full-MHA source, weights and bounded strided lease required")
    if target.device.type == "cuda":
        from mosaickv.evaluation.shared_source_kernel import decode_chunk, rotate_native
    else:
        from mosaickv.evaluation.precision_tier import unpack_cpu

        decoded = unpack_cpu(packet.source).squeeze(1)
    weights = projection.weight.view(h, d, width).transpose(1, 2)
    for first in range(0, n, 256):
        take = min(256, n - first)
        x = workspace[:, :take]
        if target.device.type == "cuda":
            decode_chunk(packet, x, first)
        else:
            x.copy_(decoded[:, first : first + take])
        if packet.input_overlay is not None:
            from mosaickv.evaluation.integrated_source_input import overlay_input

            overlay_input(packet.input_overlay, x, first)
        output = target[0, :, first : first + take, :]
        torch.bmm(x.expand(h, take, width), weights, out=output)
        if component == 0:
            if target.device.type == "cuda":
                rotate_native(target, packet.cos, packet.sin, first, take)
            else:
                k = output.clone()
                cos = packet.cos[:, first : first + take]
                sin = packet.sin[:, first : first + take]
                output.copy_(k * cos + torch.cat((-k[..., 64:], k[..., :64]), -1) * sin)
