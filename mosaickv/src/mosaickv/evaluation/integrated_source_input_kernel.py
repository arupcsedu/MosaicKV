"""Lossless BF16 input scatter directly into the charged bounded source chunk."""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["COUNT", "FIRST", "TAKE"])
def _scatter(
    SM,
    CODES,
    BASE,
    OFFSETS,
    LITERALS,
    POSITIONS,
    OUT,
    COUNT,
    FIRST,
    TAKE,
    WIDTH: tl.constexpr,
    RAW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    block = tl.program_id(0)
    local = tl.arange(0, BLOCK)
    index = block * BLOCK + local
    valid = index < COUNT
    if RAW:
        bits = tl.load(SM + index, valid, other=0).to(tl.uint16)
    else:
        byte, shift = local * 3 // 8, local * 3 % 8
        first = tl.load(CODES + block * 384 + byte).to(tl.int32)
        second = tl.load(CODES + block * 384 + byte + 1, byte + 1 < 384, other=0).to(tl.int32)
        code = ((first >> shift) | (second << (8 - shift))) & 7
        escaped = (code == 7) & valid
        rank = tl.cumsum(escaped.to(tl.int32)) - 1
        offset = tl.load(OFFSETS + block)
        literal = tl.load(LITERALS + offset + rank, escaped, other=0).to(tl.int32)
        base = tl.load(BASE + block).to(tl.int32)
        exponent = tl.where(escaped, literal, base + code)
        sm = tl.load(SM + index, valid, other=0).to(tl.int32)
        bits = ((sm & 127) | ((sm & 128) << 8) | (exponent << 7)).to(tl.uint16)
    position = tl.load(POSITIONS + index // WIDTH, valid, other=0)
    active = valid & (position >= FIRST) & (position < FIRST + TAKE)
    tl.store(OUT + (position - FIRST) * WIDTH + index % WIDTH, bits, active)


def scatter(body, positions, target, first):
    import torch

    count = body.shape[-2] * body.shape[-1]
    if (
        body.shape[:2] != (1, 1)
        or positions.numel() != body.shape[-2]
        or not target.is_contiguous()
        or target.ndim != 3
        or target.shape[0] != 1
        or target.shape[-1] != body.shape[-1]
        or target.dtype != torch.bfloat16
        or type(first) is not int
        or first < 0
    ):
        raise ValueError("compatible original input rows and contiguous chunk required")
    buffers = (body.buffers[0].view(torch.uint16),) * 5 if body.raw else body.buffers
    _scatter[(triton.cdiv(count, 1024),)](
        *buffers,
        positions,
        target.view(torch.uint16),
        count,
        first,
        target.shape[1],
        WIDTH=body.shape[-1],
        RAW=body.raw,
        BLOCK=1024,
    )


def gpu_controls():
    import torch

    from mosaickv.evaluation.lossless_bf16 import PackedBF16
    from mosaickv.evaluation.lossless_bf16_kernel import encode

    torch.manual_seed(20261101)
    values = torch.randn(1, 1, 4, 256, device="cuda").bfloat16()
    positions = torch.tensor([1, 129, 260, 601], device="cuda", dtype=torch.int32)
    records = []
    for raw in (False, True):
        body = (
            PackedBF16(tuple(values.shape), True, (values.clone(),))
            if raw
            else encode(values, raw_fallback=False)
        )
        for first in (0, 128, 256, 512):
            backing = torch.full((1, 134, 256), -71, device="cuda", dtype=torch.bfloat16)
            target = backing[:, 3:131]
            scatter(body, positions, target, first)
            valid = (positions >= first) & (positions < first + 128)
            expected = torch.full_like(target, -71)
            expected[0, positions[valid].long() - first] = values[0, 0, valid]
            if (
                not torch.equal(target.view(torch.int16), expected.view(torch.int16))
                or not (backing[:, :3] == -71).all()
                or not (backing[:, 131:] == -71).all()
            ):
                raise RuntimeError("exact source-input scatter changed input bits or sentinels")
            records.append(dict(raw=raw, first=first, bitwise=True, sentinels_unchanged=True))
    return records
