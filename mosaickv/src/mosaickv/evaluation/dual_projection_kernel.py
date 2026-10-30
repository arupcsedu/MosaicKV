"""Experimental frozen-weight MHA K/V projection into a strided cache lease.

FP32 accumulation is NOT assumed bit-equivalent to cuBLAS. The caller must
measure, store and apply exact original-bit corrections before use.
"""

from typing import Any

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["ROWS", "MAP_STRIDE", "HEAD_OUT"])
def _project(
    X,
    WK,
    WV,
    UNION,
    SLOTS,
    COS,
    SIN,
    OK,
    OV,
    ROWS,
    WIDTH: tl.constexpr,
    MAP_STRIDE,
    HEAD_OUT,
    DIM: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    head = tl.program_id(1)
    col = tl.arange(0, DIM)
    reduction = tl.arange(0, BK)
    slot = tl.load(SLOTS + head * MAP_STRIDE + row, row < ROWS, other=-1).to(tl.int32)
    if tl.sum((slot >= 0).to(tl.int32), 0) > 0:
        ak = tl.full((BM, DIM), 0, tl.float32)
        av = tl.full((BM, DIM), 0, tl.float32)
        for start in range(tl.cdiv(WIDTH, BK)):
            k = start * BK + reduction
            x = tl.load(
                X + row[:, None] * WIDTH + k[None, :],
                (row[:, None] < ROWS) & (k[None, :] < WIDTH),
                other=0,
            )
            wk = tl.load(
                WK + (head * DIM + col[None, :]) * WIDTH + k[:, None], k[:, None] < WIDTH, other=0
            )
            wv = tl.load(
                WV + (head * DIM + col[None, :]) * WIDTH + k[:, None], k[:, None] < WIDTH, other=0
            )
            ak = tl.dot(x, wk, ak)
            av = tl.dot(x, wv, av)
        key = ak.to(tl.bfloat16).to(tl.float32)
        rotated = tl.gather(key, tl.broadcast_to(((col + DIM // 2) % DIM)[None, :], (BM, DIM)), 1)
        rotated = tl.where(col[None, :] < DIM // 2, -rotated, rotated)
        logical = tl.load(UNION + row, row < ROWS, other=0).to(tl.int32)
        cos = tl.load(COS + logical[:, None] * DIM + col[None, :], row[:, None] < ROWS, other=0).to(
            tl.float32
        )
        sin = tl.load(SIN + logical[:, None] * DIM + col[None, :], row[:, None] < ROWS, other=0).to(
            tl.float32
        )
        first = (key * cos).to(tl.bfloat16).to(tl.float32)
        second = (rotated * sin).to(tl.bfloat16).to(tl.float32)
        pointer = head * HEAD_OUT + slot[:, None] * DIM + col[None, :]
        mask = (row[:, None] < ROWS) & (slot[:, None] >= 0)
        tl.store(OK + pointer, (first + second).to(tl.bfloat16), mask)
        tl.store(OV + pointer, av.to(tl.bfloat16), mask)


@triton.jit(do_not_specialize=["COUNT", "HEAD_OUT"])
def _patch(
    INDICES,
    VALUES,
    OUTPUT,
    COUNT,
    KEPT: tl.constexpr,
    HEAD_OUT,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    index = tl.load(INDICES + offset, offset < COUNT, other=0).to(tl.int32)
    value = tl.load(VALUES + offset, offset < COUNT, other=0)
    head = index // (KEPT * DIM)
    position = index // DIM % KEPT
    tl.store(OUTPUT + head * HEAD_OUT + position * DIM + index % DIM, value, offset < COUNT)


def project(source: Any, attention: Any, archive: Any, slots: Any, targets: Any) -> None:
    import torch

    heads, kept = archive.selected.shape
    dim = attention.head_dim
    width = source.shape[-1]
    if (
        source.dtype != torch.bfloat16
        or source.device.type != "cuda"
        or width != heads * dim
        or dim != 128
        or attention.k_proj.bias is not None
        or attention.v_proj.bias is not None
        or any(
            t.shape != (1, heads, kept, dim) or t.stride(-2) != dim or t.stride(-1) != 1
            for t in targets
        )
        or targets[0].stride(-3) != targets[1].stride(-3)
    ):
        raise NotImplementedError("audited BF16 bias-free MHA D=128 strided lease required")
    if (
        not source.is_contiguous()
        or not attention.k_proj.weight.is_contiguous()
        or not attention.v_proj.weight.is_contiguous()
    ):
        raise ValueError("contiguous source and unmodified original weights required")
    _project[(triton.cdiv(archive.union.numel(), 16), heads)](
        source,
        attention.k_proj.weight,
        attention.v_proj.weight,
        archive.union,
        slots,
        archive.cos,
        archive.sin,
        *targets,
        ROWS=archive.union.numel(),
        WIDTH=width,
        MAP_STRIDE=slots.stride(0),
        HEAD_OUT=targets[0].stride(-3),
        DIM=dim,
        BM=16,
        BK=64,
        enable_fp_fusion=False,
        num_warps=4,
    )


def apply_patch(patch: Any, target: Any) -> None:
    import torch

    count = patch.indices.numel()
    if count:
        _patch[(triton.cdiv(count, 256),)](
            patch.indices,
            patch.values.view(torch.int16),
            target.view(torch.int16),
            COUNT=count,
            KEPT=target.shape[-2],
            HEAD_OUT=target.stride(-3),
            DIM=target.shape[-1],
            BLOCK=256,
        )
