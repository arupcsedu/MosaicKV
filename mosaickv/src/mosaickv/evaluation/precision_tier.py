"""Explicit lossy prompt-precision diagnostic, not mosaickv_exact or official KIVI.

Original BF16 K/V and positions are never overwritten. Quantize keys along
time, values along features; preserve the first four and last eight selected
positions exactly. Original HF attention and lossless generated history remain.
"""

from dataclasses import dataclass
from typing import Any

import torch

from mosaickv.evaluation.lossless_prompt_cache import backing_bytes


@dataclass(frozen=True)
class QuantizedBody:
    shape: tuple[int, ...]
    bits: int
    axis: int
    group: int
    codes: Any
    minimum: Any
    scale: Any
    codebook: Any = None

    def tensors(self) -> tuple[Any, ...]:
        return (
            self.codes,
            self.minimum,
            self.scale,
            *(() if self.codebook is None else (self.codebook,)),
        )


def pack(
    values: Any, *, bits: int, axis: int, group: int = 128, compact_workspace: bool = True
) -> QuantizedBody:
    if (
        type(bits) is not int
        or bits not in (2, 3, 4, 8)
        or axis not in (2, 3)
        or type(group) is not int
        or group != 128
        or values.dtype != torch.bfloat16
        or values.ndim != 4
        or values.shape[0] != 1
        or not all(values.shape)
        or not bool(torch.isfinite(values).all())
    ):
        raise ValueError("finite nonempty batch-one BF16, 2/3/4/8 bits and group128 required")
    if type(compact_workspace) is not bool:
        raise ValueError("explicit compact quantizer workspace flag required")
    if compact_workspace and torch.is_grad_enabled() and values.requires_grad:
        return pack(values, bits=bits, axis=axis, group=group, compact_workspace=False)
    transformed = (values if compact_workspace else values.float()).movedim(axis, -1).contiguous()
    length = transformed.shape[-1]
    pad = (-length) % group
    if pad:
        # Repeat the last real value: padding cannot introduce artificial extrema.
        transformed = torch.cat(
            (transformed, transformed[..., -1:].expand(*transformed.shape[:-1], pad)), -1
        )
    if compact_workspace:
        # Copy/padding commute exactly with BF16->FP32 conversion. Avoid two
        # overlapping full FP32 copies while preserving the original layout.
        transformed = transformed.float()
    shaped = transformed.reshape(*transformed.shape[:-1], -1, group)
    minimum, maximum = shaped.amin(-1), shaped.amax(-1)
    scale = (maximum - minimum) / ((1 << bits) - 1)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    # Only the packed integer arithmetic changes, never the FP32 quantization
    # or stored headers. Up to four bits need at most a 32-bit word; eight-bit
    # values already ARE their serialized bytes and need no word accumulator.
    integer_dtype = torch.uint8 if bits == 8 else torch.int32
    if compact_workspace:
        # PRIVATE workspace: retain separate FP32 ATen operations/rounding.
        # Never mutate caller BF16 inputs, headers, model weights or original KV.
        shaped.sub_(minimum[..., None])
        shaped.div_(scale[..., None])
        shaped.round_()
        shaped.clamp_(0, (1 << bits) - 1)
        if values.device.type == "cuda" and bool(
            (torch.isfinite(scale) & (scale >= torch.finfo(torch.float32).tiny)).all()
        ):
            from mosaickv.evaluation.quantizer_workspace_kernel import serialize

            codes = serialize(shaped, bits=bits, length=length, value_count=values.numel())
            return QuantizedBody(
                tuple(values.shape),
                bits,
                axis,
                group,
                codes,
                minimum.contiguous(),
                scale.contiguous(),
            )
        quantized = shaped.to(integer_dtype)
    else:
        quantized = (
            ((shaped - minimum[..., None]) / scale[..., None])
            .round()
            .clamp(0, (1 << bits) - 1)
            .to(integer_dtype)
        )
    # Integer serialization no longer needs the full FP32 input or its views.
    # Headers, rounding order and quantized integers remain exactly unchanged.
    del transformed, shaped, maximum
    # Group padding affects extrema only, never the physical code payload.
    quantized = quantized.flatten(-2)[..., :length].contiguous()
    if bits == 8:
        codes = quantized.flatten()
    else:
        chunks = quantized.reshape(-1, 8)
        shifts = torch.arange(8, device=values.device, dtype=torch.int32) * bits
        # Four-bit all-15 words wrap modulo 2**32. Masking each shifted byte
        # reproduces the old nonnegative int64 serialization exactly.
        words = (chunks << shifts).sum(-1, dtype=torch.int32)
        codes = torch.stack(
            tuple(((words >> (8 * i)) & 255).to(torch.uint8) for i in range(bits)), -1
        ).flatten()
    return QuantizedBody(
        tuple(values.shape), bits, axis, group, codes, minimum.contiguous(), scale.contiguous()
    )


def unpack_cpu(body: QuantizedBody) -> Any:
    if body.codes.device.type != "cpu":
        raise ValueError("CPU oracle only")
    bytes_ = body.codes.reshape(-1, body.bits).long()
    words = sum(bytes_[:, i] << (8 * i) for i in range(body.bits))
    shifts = torch.arange(8, dtype=torch.long) * body.bits
    quantized = ((words[:, None] >> shifts) & ((1 << body.bits) - 1)).float()
    layout = list(body.shape)
    length = layout.pop(body.axis)
    layout.append(length)
    quantized = quantized.reshape(layout)
    if body.codebook is not None:
        if body.bits != 2 or body.axis != 3 or body.shape[-1] != 128:
            raise ValueError("two-bit per-value-row codebook required")
        centers = body.codebook.reshape(*body.shape[:-1], 4)
        return centers.gather(-1, quantized.long()).contiguous()
    scale = body.scale.repeat_interleave(body.group, -1)[..., :length]
    minimum = body.minimum.repeat_interleave(body.group, -1)[..., :length]
    result = quantized * scale + minimum
    return result.movedim(-1, body.axis).contiguous().to(torch.bfloat16)


@dataclass(frozen=True)
class PrecisionFormat:
    shape: tuple[int, ...]
    pair: tuple[QuantizedBody, ...]
    exact_indices: Any
    exact_map: Any
    exact_values: tuple[Any, ...]
    kind: str = "evicastkv_prompt_precision_variant"
    encoded_exact: bool = False
    corrections: tuple[Any, ...] = ()

    def private(self) -> tuple[Any, ...]:
        exact = (
            tuple(t for b in self.exact_values for t in b.buffers)
            if self.encoded_exact
            else self.exact_values
        )
        return (
            *(t for b in self.pair for t in b.tensors()),
            self.exact_indices,
            self.exact_map,
            *exact,
            *(t for patch in self.corrections for t in patch.tensors),
        )

    def shared(self) -> tuple[Any, ...]:
        return ()

    def tensors(self) -> tuple[Any, ...]:
        return self.private()

    @property
    def component_bytes(self) -> int:
        return backing_bytes(self.tensors())

    def materialize(self, attention: Any = None) -> tuple[Any, ...]:
        del attention
        output = tuple(
            torch.empty(self.shape, dtype=torch.bfloat16, device=b.codes.device) for b in self.pair
        )
        self.fill(None, output, direct=True)
        return output

    def fill(self, attention: Any, destinations: tuple[Any, ...], *, direct: bool) -> None:
        del attention, direct
        for b, exact, target in zip(self.pair, self.exact_values, destinations, strict=True):
            if b.codes.device.type == "cuda":
                from mosaickv.evaluation.precision_tier_kernel import decode

                decode(
                    b,
                    target,
                    self.exact_map,
                    b.codes if self.encoded_exact else exact,
                    overlay=not self.encoded_exact,
                )
                if self.encoded_exact:
                    from mosaickv.evaluation.protected_exact import scatter_into

                    scatter_into(exact, self.exact_indices, target)
            else:
                target[..., : self.shape[-2], :].copy_(unpack_cpu(b))
                if self.encoded_exact:
                    from mosaickv.evaluation.protected_exact import scatter_into

                    scatter_into(exact, self.exact_indices, target)
                else:
                    target[..., self.exact_indices, :] = exact
        if self.corrections:
            for patch, target in zip(self.corrections, destinations, strict=True):
                prefix = target[..., : self.shape[-2], :]
                if target.device.type == "cuda":
                    from mosaickv.evaluation.dual_projection_kernel import apply_patch

                    apply_patch(patch, prefix)
                else:
                    index = patch.indices.long()
                    n, d = self.shape[-2:]
                    target[0, index // (n * d), (index // d) % n, index % d] = patch.values


def construct(
    pair: tuple[Any, ...],
    *,
    bits: int = 4,
    protected: tuple[int, ...] = (),
    encoded_exact: bool = False,
) -> PrecisionFormat:
    if len(pair) != 2 or pair[0].shape != pair[1].shape:
        raise ValueError("same-shape key/value pair required")
    if type(encoded_exact) is not bool:
        raise ValueError("encoded_exact must be an explicit boolean")
    length = pair[0].shape[-2]
    if any(type(i) is not int or not 0 <= i < length for i in protected):
        raise ValueError("protected original selected positions must be in bounds")
    positions = sorted(
        set(protected) | set(range(min(4, length))) | set(range(max(0, length - 8), length))
    )
    indices = torch.tensor(positions, device=pair[0].device, dtype=torch.int32)
    mapping = torch.full((length,), -1, device=pair[0].device, dtype=torch.int32)
    mapping[indices.long()] = torch.arange(len(positions), device=pair[0].device, dtype=torch.int32)
    exact = tuple(t.index_select(2, indices.long()).clone() for t in pair)
    if encoded_exact:
        from mosaickv.evaluation.exact_storage import encode

        exact = tuple(encode(t) for t in exact)
        mapping = torch.empty(0, device=pair[0].device, dtype=torch.int32)
    return PrecisionFormat(
        tuple(pair[0].shape),
        (pack(pair[0], bits=bits, axis=2), pack(pair[1], bits=bits, axis=3)),
        indices,
        mapping,
        exact,
        encoded_exact=encoded_exact,
    )
