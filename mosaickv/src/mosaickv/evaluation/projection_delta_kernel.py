"""Integer-only fixed-code projection correction into a strided attention lease."""

from typing import Any

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["COUNT", "KEPT", "HEAD_STRIDE"])
def _apply(
    CODES,
    OFFSETS,
    LITERALS,
    OUTPUT,
    COUNT,
    KEPT,
    HEAD_STRIDE,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    block = tl.program_id(0)
    local = tl.arange(0, BLOCK)
    index = block * BLOCK + local
    valid = index < COUNT
    packed = tl.load(CODES + index // 4, mask=valid, other=0).to(tl.int32)
    code = (packed >> (2 * (index % 4))) & 3
    escaped = valid & (code == 3)
    rank = tl.cumsum(escaped.to(tl.int32)) - 1
    offset = tl.load(OFFSETS + block)
    literal = tl.load(LITERALS + offset + rank, mask=escaped, other=0).to(tl.int32)
    head = index // (KEPT * DIM)
    physical = head * HEAD_STRIDE + index % (KEPT * DIM)
    original = tl.load(OUTPUT + physical, mask=valid, other=0).to(tl.int32)
    delta = tl.where(code == 1, -1, tl.where(code == 2, 1, 0))
    output = tl.where(escaped, literal, original + delta).to(tl.int16)
    tl.store(OUTPUT + physical, output, mask=valid)


def apply(patch: Any, destination: Any) -> None:
    import torch

    _apply[(triton.cdiv(destination.numel(), 1024),)](
        *patch.buffers,
        destination.view(torch.int16),
        destination.numel(),
        destination.shape[-2],
        destination.stride(-3),
        DIM=destination.shape[-1],
        BLOCK=1024,
    )
