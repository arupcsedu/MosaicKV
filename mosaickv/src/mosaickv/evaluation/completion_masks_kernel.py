"""One fused mask allocation, no arange/boolean/workspace tensors on CUDA."""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["LENGTH", "PROMPT"])
def _fill(
    OUT,
    COUNTS,
    OLD,
    BIAS,
    LENGTH,
    PROMPT,
    H: tl.constexpr,
    OLD_H: tl.constexpr,
    SH,
    SN,
    MINIMUM: tl.constexpr,
    USE_OLD: tl.constexpr,
    USE_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    h, p = i // LENGTH, i % LENGTH
    valid = i < H * LENGTH
    count = tl.load(COUNTS + h, valid, other=0).to(tl.int32)
    value = tl.where((p < count) | (p >= PROMPT), 0.0, MINIMUM)
    if USE_OLD:
        old_head = h if OLD_H == H else tl.full((BLOCK,), 0, tl.int32)
        value += tl.load(OLD + old_head * SH + p * SN, valid, other=0).to(tl.float32)
    if USE_BIAS:
        value += tl.load(BIAS + h * PROMPT + p, valid & (p < PROMPT), other=0)
    tl.store(OUT + i, value, valid)


def fill(mask, counts, prompt, old, bias=None):
    import torch

    length = mask.shape[-1]
    _fill[(triton.cdiv(counts.numel() * length, 256),)](
        mask,
        counts,
        counts if old is None else old,
        counts if bias is None else bias,
        length,
        prompt,
        H=counts.numel(),
        OLD_H=1 if old is None else old.shape[1],
        SH=0 if old is None else old.stride(1),
        SN=0 if old is None else old.stride(-1),
        MINIMUM=torch.finfo(mask.dtype).min,
        USE_OLD=old is not None,
        USE_BIAS=bias is not None,
        BLOCK=256,
    )
