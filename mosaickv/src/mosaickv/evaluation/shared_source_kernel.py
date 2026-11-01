"""Direct encoded-input K/V projection into the existing charged layer lease.

Derived from this repository's encoded_projection_kernel; no decoded full
source or separate K/V temporary is allocated in GPU global memory.
"""

import triton
import triton.language as tl


@triton.jit
def _read(CODES, MINIMUM, SCALE, row, start, ROWS, WIDTH: tl.constexpr, BITS: tl.constexpr):
    col = start + tl.arange(0, 64)
    index = col[None, :] * ROWS + row[:, None]
    valid = row[:, None] < ROWS
    bit = index * BITS
    byte, shift = bit // 8, bit % 8
    a = tl.load(CODES + byte, valid, other=0).to(tl.int32)
    b = tl.load(CODES + byte + 1, valid & (shift + BITS > 8), other=0).to(tl.int32)
    code = ((a >> shift) | (b << (8 - shift))) & ((1 << BITS) - 1)
    group = col[None, :] * tl.cdiv(ROWS, 128) + row[:, None] // 128
    minimum = tl.load(MINIMUM + group, valid, other=0)
    scale = tl.load(SCALE + group, valid, other=1)
    return (code.to(tl.float32) * scale + minimum).to(tl.bfloat16)


@triton.jit(do_not_specialize=["ROWS", "HEAD_OUT"])
def _project(
    CODES,
    MINIMUM,
    SCALE,
    WK,
    WV,
    COS,
    SIN,
    OK,
    OV,
    ROWS,
    HEAD_OUT,
    WIDTH: tl.constexpr,
    DIM: tl.constexpr,
    BITS: tl.constexpr,
):
    row = tl.program_id(0) * 16 + tl.arange(0, 16)
    head = tl.program_id(1)
    col = tl.arange(0, DIM)
    reduction = tl.arange(0, 64)
    ak = tl.full((16, DIM), 0, tl.float32)
    av = tl.full((16, DIM), 0, tl.float32)
    for start in range(WIDTH // 64):
        x = _read(CODES, MINIMUM, SCALE, row, start * 64, ROWS, WIDTH, BITS)
        k = start * 64 + reduction
        wk = tl.load(WK + (head * DIM + col[None, :]) * WIDTH + k[:, None])
        wv = tl.load(WV + (head * DIM + col[None, :]) * WIDTH + k[:, None])
        ak = tl.dot(x, wk, ak)
        av = tl.dot(x, wv, av)
    key = ak.to(tl.bfloat16).to(tl.float32)
    rotated = tl.gather(key, tl.broadcast_to(((col + DIM // 2) % DIM)[None, :], (16, DIM)), 1)
    rotated = tl.where(col[None, :] < DIM // 2, -rotated, rotated)
    valid = row[:, None] < ROWS
    cos = tl.load(COS + row[:, None] * DIM + col[None, :], valid, other=0).to(tl.float32)
    sin = tl.load(SIN + row[:, None] * DIM + col[None, :], valid, other=0).to(tl.float32)
    first = (key * cos).to(tl.bfloat16).to(tl.float32)
    second = (rotated * sin).to(tl.bfloat16).to(tl.float32)
    pointer = head * HEAD_OUT + row[:, None] * DIM + col[None, :]
    tl.store(OK + pointer, first + second, valid)
    tl.store(OV + pointer, av.to(tl.bfloat16), valid)


def project(packet, attention, targets):
    import torch

    h, n, d = packet.shape[1:]
    body = packet.source
    width = body.shape[-1]
    if body.bits != 3 or body.axis != 2 or body.group != 128 or width != h * d or d != 128:
        raise ValueError("audited three-bit per-channel full-MHA source required")
    if any(t.stride(-1) != 1 or t.stride(-2) != d for t in targets) or targets[0].stride(
        1
    ) != targets[1].stride(1):
        raise ValueError("compatible existing strided K/V leases required")
    if (
        attention.k_proj.bias is not None
        or attention.v_proj.bias is not None
        or not attention.k_proj.weight.is_contiguous()
        or not attention.v_proj.weight.is_contiguous()
        or any(
            weight.shape != (h * d, width)
            or weight.dtype != torch.bfloat16
            or weight.device != body.codes.device
            for weight in (attention.k_proj.weight, attention.v_proj.weight)
        )
    ):
        raise ValueError("unchanged contiguous bias-free native projection weights required")
    _project[(triton.cdiv(n, 16), h)](
        body.codes,
        body.minimum,
        body.scale,
        attention.k_proj.weight,
        attention.v_proj.weight,
        packet.cos,
        packet.sin,
        *targets,
        n,
        targets[0].stride(1),
        WIDTH=width,
        DIM=d,
        BITS=3,
        enable_fp_fusion=False,
    )
