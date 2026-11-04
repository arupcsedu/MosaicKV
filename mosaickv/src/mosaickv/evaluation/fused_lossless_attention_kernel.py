"""Two-pass lossless packed-cache attention without full K/V materialization.

The eager FP32 softmax remains unchanged. The reduction kernels are experimental
and are not assumed numerically equivalent to cuBLAS. No atomic reductions.
"""

import triton
import triton.language as tl


@triton.jit
def _unpack(SM, CODES, BASE, OFFSETS, LITERALS, block, COUNT: tl.constexpr, RAW: tl.constexpr):
    local = tl.arange(0, 1024)
    index = block * 1024 + local
    if RAW:
        return tl.load(SM + index, mask=index < COUNT, other=0)
    bit = local * 3
    byte, shift = bit // 8, bit % 8
    first = tl.load(CODES + block * 384 + byte)
    second = tl.load(CODES + block * 384 + byte + 1, mask=byte + 1 < 384, other=0)
    code = ((first.to(tl.int32) >> shift) | (second.to(tl.int32) << (8 - shift))) & 7
    escaped = (code == 7) & (index < COUNT)
    rank = tl.cumsum(escaped.to(tl.int32)) - 1
    offset = tl.load(OFFSETS + block)
    literal = tl.load(LITERALS + offset + rank, mask=escaped, other=0).to(tl.int32)
    base = tl.load(BASE + block).to(tl.int32)
    exponent = tl.where(escaped, literal, base + code)
    sm = tl.load(SM + index, mask=index < COUNT, other=0).to(tl.int32)
    bits = ((sm & 127) | ((sm & 128) << 8) | (exponent << 7)).to(tl.uint16)
    return bits.to(tl.bfloat16, bitcast=True)


@triton.jit
def _qk(
    Q,
    SM,
    CODES,
    BASE,
    OFFSETS,
    LITERALS,
    SCORES,
    COUNT: tl.constexpr,
    N: tl.constexpr,
    RAW: tl.constexpr,
):
    block = tl.program_id(0)
    token = block * 8 + tl.arange(0, 8)
    dim = tl.arange(0, 128)
    keys = tl.reshape(_unpack(SM, CODES, BASE, OFFSETS, LITERALS, block, COUNT, RAW), (8, 128))
    query = tl.load(
        Q + (token // N)[:, None] * 128 + dim[None, :], mask=token[:, None] < COUNT // 128, other=0
    )
    dot = tl.sum(keys.to(tl.float32) * query.to(tl.float32), axis=1).to(tl.bfloat16)
    tl.store(SCORES + token, dot, mask=token < COUNT // 128)


@triton.jit
def _pv(
    W,
    SM,
    CODES,
    BASE,
    OFFSETS,
    LITERALS,
    OUTPUT,
    COUNT: tl.constexpr,
    N: tl.constexpr,
    RAW: tl.constexpr,
):
    head = tl.program_id(0)
    accumulator = tl.full((128,), 0, tl.float32)
    for block in range(head * N // 8, tl.cdiv((head + 1) * N, 8)):
        token = block * 8 + tl.arange(0, 8)
        values = tl.reshape(
            _unpack(SM, CODES, BASE, OFFSETS, LITERALS, block, COUNT, RAW), (8, 128)
        )
        weights = tl.load(W + token, mask=(token >= head * N) & (token < (head + 1) * N), other=0)
        accumulator += tl.sum(values.to(tl.float32) * weights[:, None].to(tl.float32), axis=0)
    tl.store(OUTPUT + head * 128 + tl.arange(0, 128), accumulator.to(tl.bfloat16))


def execute(query, key, value, scale):
    import torch

    h, n = key.shape[1:3]
    count = h * n * 128
    scores = torch.empty((1, h, 1, n), device=query.device, dtype=torch.bfloat16)

    def arguments(body):
        return (body.buffers[0],) * 5 if body.raw else body.buffers

    _qk[(triton.cdiv(count, 1024),)](
        query, *arguments(key), scores, count, n, key.raw, enable_fp_fusion=False
    )
    weights = torch.softmax(scores * scale, dim=-1, dtype=torch.float32).to(query.dtype)
    output = torch.empty_like(query)
    _pv[(h,)](weights, *arguments(value), output, count, n, value.raw, enable_fp_fusion=False)
    return output, weights
