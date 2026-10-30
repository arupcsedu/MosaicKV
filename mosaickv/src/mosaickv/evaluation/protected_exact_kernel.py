"""Integer-only lossless scatter; all growing cache geometry remains dynamic."""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["COUNT", "P", "HEAD_OUT"])
def _scatter(
    SM,
    CODES,
    BASE,
    OFFSETS,
    LITERALS,
    POSITIONS,
    OUT,
    COUNT,
    P,
    HEAD_OUT,
    D: tl.constexpr,
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
    head, position = index // (P * D), (index // D) % P
    actual = tl.load(POSITIONS + position, valid, other=0)
    tl.store(OUT + head * HEAD_OUT + actual * D + index % D, bits, valid)


def scatter(body, positions, target):
    import torch

    buffers = (body.buffers[0].view(torch.uint16),) * 5 if body.raw else body.buffers
    count = body.shape[1] * body.shape[-2] * body.shape[-1]
    _scatter[(triton.cdiv(count, 1024),)](
        *buffers,
        positions,
        target.view(torch.uint16),
        count,
        body.shape[-2],
        target.stride(1),
        D=body.shape[-1],
        RAW=body.raw,
        BLOCK=1024,
    )
