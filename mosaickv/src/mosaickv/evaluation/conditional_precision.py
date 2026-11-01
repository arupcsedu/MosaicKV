"""Fit low-bit ranges only to rows not restored exactly later.

Dense/sparse quantization hygiene, not a novel general quantizer. No changed
memberships, exact protections, positions, bit width or runtime decoder.
"""

import torch

from mosaickv.evaluation.head_local_precision import HeadLocalFormat
from mosaickv.evaluation.precision_tier import pack
from mosaickv.evaluation.text_source_storage import TextSourceFormat


def exact_rows(packet):
    outer = packet if isinstance(packet, TextSourceFormat) else None
    local = outer.base if outer is not None else packet
    if not isinstance(local, HeadLocalFormat):
        raise ValueError("audited head-local exact-row metadata required")
    _, h, n, _ = local.shape
    rows = torch.zeros(h, n, dtype=torch.bool, device=local.base.pair[0].codes.device)
    rows[:, local.base.exact_indices.long()] = True
    if local.flat_tokens.numel():
        rows.flatten()[local.flat_tokens.long()] = True
    if outer is not None:
        slots = outer.projection.slots
        heads = torch.arange(h, device=rows.device)[:, None].expand_as(slots)
        valid = slots >= 0
        rows[heads[valid], slots[valid].long()] = True
    return rows


@torch.inference_mode()
def pack_unprotected(values, rows, *, bits, axis):
    if (
        values.ndim != 4
        or values.shape[0] != 1
        or rows.shape != values.shape[1:3]
        or rows.dtype != torch.bool
        or rows.device != values.device
        or axis not in (2, 3)
        or values.dtype != torch.bfloat16
        or not torch.isfinite(values).all()
    ):
        raise ValueError("finite BF16 values and exact same-device per-head row mask required")
    x = values.float().movedim(axis, -1).contiguous()
    mask = rows[None, :, :, None].expand_as(values).movedim(axis, -1).contiguous()
    length = x.shape[-1]
    pad = (-length) % 128
    if pad:
        x = torch.cat((x, x[..., -1:].expand(*x.shape[:-1], pad)), -1)
        mask = torch.cat((mask, mask[..., -1:].expand(*mask.shape[:-1], pad)), -1)
    shaped = x.reshape(*x.shape[:-1], -1, 128)
    protected = mask.reshape_as(shaped)
    minimum = shaped.masked_fill(protected, float("inf")).amin(-1)
    # All-protected groups are overwritten completely. Finite representatives
    # avoid NaNs and never affect an unprotected cache scalar.
    minimum = torch.where(torch.isfinite(minimum), minimum, torch.zeros_like(minimum))
    prepared = torch.where(protected, minimum[..., None], shaped).flatten(-2)[..., :length]
    prepared = prepared.movedim(-1, axis).contiguous().bfloat16()
    # All range preparation is complete. Do not overlap its full FP32 source
    # and expanded masks with pack()'s independent quantization workspace.
    del x, mask, shaped, protected, minimum
    return pack(prepared, bits=bits, axis=axis)
