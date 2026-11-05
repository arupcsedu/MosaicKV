"""Masked fixed-capacity writes used ONLY before canonical sparse commit."""

import torch
import triton
import triton.language as tl


@triton.jit
def _apply(
    IDS,
    VALUES,
    COUNT,
    OUTPUT,
    HEAD: tl.constexpr,
    N: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    rank = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    count = tl.minimum(tl.load(COUNT), CAPACITY)
    valid = rank < count
    index = tl.load(IDS + rank, valid, other=0).to(tl.int32)
    bits = tl.load(VALUES + rank, valid, other=0)
    offset = index // (N * 128) * HEAD + index // 128 % N * 128 + index % 128
    tl.store(OUTPUT + offset, bits, valid & (index >= 0))


def apply(patch, target, *, prompt_length):
    if target.stride(-1) != 1 or target.stride(-2) != 128:
        raise ValueError("original strided D=128 constructor target required")
    compiled = _apply[(triton.cdiv(patch.indices.numel(), 256),)](
        patch.indices,
        patch.values.view(torch.int16),
        patch.count,
        target.view(torch.int16),
        HEAD=target.stride(-3),
        N=prompt_length,
        CAPACITY=patch.indices.numel(),
        BLOCK=256,
    )
    if compiled.metadata.global_scratch_size != 0:
        raise RuntimeError("temporary correction may not add opaque global scratch")
