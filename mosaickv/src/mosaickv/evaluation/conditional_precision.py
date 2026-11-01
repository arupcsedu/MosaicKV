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


def _token_group_preparation(values, rows):
    """Same BF16 extrema, without transposing/padding the full source or mask.

    Repeating the last element cannot change a finite nonzero minimum. Empty
    protected groups still use positive zero. Ambiguous zero extrema return
    None so the unchanged FP32 reference handles signed-zero ties.
    """
    n = values.shape[2]
    full, tail = divmod(n, 128)
    pieces = []
    if full:
        window = values[:, :, : full * 128].unfold(2, 128, 128)
        mask = rows[:, : full * 128].unfold(1, 128, 128)[None, :, :, None, :]
        pieces.append(window.masked_fill(mask, float("inf")).amin(-1).movedim(2, -1))
    if tail:
        window = values[:, :, full * 128 :].movedim(2, -1)
        mask = rows[:, full * 128 :][None, :, None, :]
        pieces.append(window.masked_fill(mask, float("inf")).amin(-1)[..., None])
    minimum = torch.cat(pieces, -1) if len(pieces) > 1 else pieces[0]
    if bool(((minimum == 0) & torch.isfinite(minimum)).any()):
        return None
    minimum = torch.where(torch.isfinite(minimum), minimum, torch.zeros_like(minimum))
    # Exactly one private BF16 output. No expanded full mask, padded source,
    # or padded torch.where output overlaps this allocation.
    prepared = torch.empty_like(values, memory_format=torch.contiguous_format)
    if full:
        torch.where(
            rows[:, : full * 128].unfold(1, 128, 128)[None, :, :, None, :],
            minimum[..., :full].movedim(-1, 2)[..., None],
            values[:, :, : full * 128].unfold(2, 128, 128),
            out=prepared[:, :, : full * 128].unfold(2, 128, 128),
        )
    if tail:
        torch.where(
            rows[:, full * 128 :][None, :, None, :],
            minimum[..., -1, None],
            values[:, :, full * 128 :].movedim(2, -1),
            out=prepared[:, :, full * 128 :].movedim(2, -1),
        )
    return prepared


def _compact_float_reference(values, rows, *, axis):
    """Preserve the original padded FP32 reduction, including signed-zero ties.

    Only copies/masks change: one private transposed FP32 tensor, a broadcast
    row mask, in-place replacement and one direct contiguous BF16 conversion.
    """
    x = values.movedim(axis, -1).to(dtype=torch.float32, memory_format=torch.contiguous_format)
    length = x.shape[-1]
    pad = (-length) % 128
    if pad:
        x = torch.cat((x, x[..., -1:].expand(*x.shape[:-1], pad)), -1)
    shaped = x.reshape(*x.shape[:-1], -1, 128)
    if axis == 2:
        mask = rows[None, :, None, :]
        if pad:
            mask = torch.cat((mask, mask[..., -1:].expand(*mask.shape[:-1], pad)), -1)
        protected = mask.reshape(*mask.shape[:-1], -1, 128)
    else:
        protected = rows[None, :, :, None, None]
    minimum = shaped.masked_fill(protected, float("inf")).amin(-1)
    minimum = torch.where(torch.isfinite(minimum), minimum, torch.zeros_like(minimum))
    torch.where(protected, minimum[..., None], shaped, out=shaped)
    return (
        shaped.flatten(-2)[..., :length]
        .movedim(-1, axis)
        .to(dtype=torch.bfloat16, memory_format=torch.contiguous_format)
    )


@torch.inference_mode()
def pack_unprotected(values, rows, *, bits, axis, bf16_prepare=True, compact_ranges=True):
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
    if type(bf16_prepare) is not bool:
        raise ValueError("explicit BF16 range-preparation flag required")
    if type(compact_ranges) is not bool:
        raise ValueError("explicit compact range workspace flag required")
    if bf16_prepare:
        words = values.view(torch.int16)
        if bool((((words & 0x7F80) == 0) & ((words & 0x7F) != 0)).any()):
            # Do not depend on device-specific subnormal reduction behavior.
            return pack_unprotected(
                values,
                rows,
                bits=bits,
                axis=axis,
                bf16_prepare=False,
                compact_ranges=compact_ranges,
            )
        if compact_ranges and axis == 2:
            prepared = _token_group_preparation(values, rows)
            if prepared is None:
                return pack_unprotected(
                    values,
                    rows,
                    bits=bits,
                    axis=axis,
                    bf16_prepare=False,
                    compact_ranges=compact_ranges,
                )
            return pack(prepared, bits=bits, axis=axis)
        if compact_ranges and axis == 3:
            # Each entire feature group belongs to just one token. Protected
            # groups become +0; every unprotected input scalar is unchanged.
            return pack(values.masked_fill(rows[None, :, :, None], 0), bits=bits, axis=axis)
    elif compact_ranges:
        prepared = _compact_float_reference(values, rows, axis=axis)
        return pack(prepared, bits=bits, axis=axis)
    # Min and masked substitution perform no arithmetic: a nonzero extremum
    # has exactly the same representation in BF16 and FP32. Quantization below
    # still uses the ORIGINAL FP32 pack(), including scale and rounding.
    x = (values if bf16_prepare else values.float()).movedim(axis, -1).contiguous()
    mask = rows[None, :, :, None].expand_as(values).movedim(axis, -1).contiguous()
    length = x.shape[-1]
    pad = (-length) % 128
    if pad:
        x = torch.cat((x, x[..., -1:].expand(*x.shape[:-1], pad)), -1)
        mask = torch.cat((mask, mask[..., -1:].expand(*mask.shape[:-1], pad)), -1)
    shaped = x.reshape(*x.shape[:-1], -1, 128)
    protected = mask.reshape_as(shaped)
    minimum = shaped.masked_fill(protected, float("inf")).amin(-1)
    if bf16_prepare and bool(((minimum == 0) & torch.isfinite(minimum)).any()):
        # Signed-zero reduction tie order can depend on the reduction dtype.
        # Release the trial workspace and use the unchanged FP32 reference for
        # the entire tensor whenever an unprotected zero minimum is ambiguous.
        del x, mask, shaped, protected, minimum
        return pack_unprotected(values, rows, bits=bits, axis=axis, bf16_prepare=False)
    # All-protected groups are overwritten completely. Finite representatives
    # avoid NaNs and never affect an unprotected cache scalar.
    minimum = torch.where(torch.isfinite(minimum), minimum, torch.zeros_like(minimum))
    prepared = torch.where(protected, minimum[..., None], shaped).flatten(-2)[..., :length]
    prepared = prepared.movedim(-1, axis).contiguous().bfloat16()
    # All range preparation is complete. Do not overlap its full FP32 source
    # and expanded masks with pack()'s independent quantization workspace.
    del x, mask, shaped, protected, minimum
    return pack(prepared, bits=bits, axis=axis)
