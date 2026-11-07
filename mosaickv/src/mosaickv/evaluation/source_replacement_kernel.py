"""Construct compact code owners, validating every omitted integer code."""

import torch
import triton
import triton.language as tl


@triton.jit
def _check(CODES, MAP, BAD, N: tl.constexpr, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < COUNT
    bit = i * 3
    a = tl.load(CODES + bit // 8, valid, other=0).to(tl.int32)
    b = tl.load(CODES + bit // 8 + 1, valid & (bit % 8 > 5), other=0).to(tl.int32)
    code = ((a >> (bit % 8)) | (b << (8 - bit % 8))) & 7
    slot = tl.load(MAP + i % N, valid, other=0)
    bad = tl.sum((valid & (slot < 0) & (code != 0)).to(tl.int32), 0)
    tl.atomic_max(BAD, (bad > 0).to(tl.int32))


@triton.jit
def _pack(
    CODES,
    KEEP,
    OUT,
    N: tl.constexpr,
    P: tl.constexpr,
    COUNT: tl.constexpr,
    EMPTY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    group = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    word = tl.full((BLOCK,), 0, tl.int32)
    for j in tl.static_range(8):
        compact = group * 8 + j
        row = tl.load(KEEP + compact % P, (compact < COUNT) & (not EMPTY), other=0)
        bit = (compact // P * N + row) * 3
        valid = (compact < COUNT) & (not EMPTY)
        a = tl.load(CODES + bit // 8, valid, other=0).to(tl.int32)
        b = tl.load(CODES + bit // 8 + 1, valid & (bit % 8 > 5), other=0).to(tl.int32)
        code = ((a >> (bit % 8)) | (b << (8 - bit % 8))) & 7
        word = word | (code << (j * 3))
    for j in tl.static_range(3):
        tl.store(OUT + group * 3 + j, (word >> (8 * j)) & 255, group * 8 < COUNT)


def construct(body, keep, mapping, physical):
    n, width = body.shape[2:]
    bad = torch.zeros(1, device=body.codes.device, dtype=torch.int32)
    _check[(triton.cdiv(n * width, 512),)](
        body.codes, mapping, bad, N=n, COUNT=n * width, BLOCK=512
    )
    if int(bad):
        raise ValueError("cannot retire nonzero source codes")
    out = torch.empty(physical * width * 3 // 8, device=body.codes.device, dtype=torch.uint8)
    _pack[(triton.cdiv(physical * width // 8, 256),)](
        body.codes,
        keep,
        out,
        N=n,
        P=physical,
        COUNT=physical * width,
        EMPTY=keep.numel() == 0,
        BLOCK=256,
    )
    return out
