"""One source-history launch; same original fixed16-row dot/RoPE operations."""

import triton
import triton.language as tl

from mosaickv.evaluation.encoded_projection_kernel import _read


@triton.jit(do_not_specialize=["HEAD_OUT"])
def _project(TABLE, RAW, WK, WV, OK, OV, HEAD_OUT, WIDTH: tl.constexpr, DIM: tl.constexpr):
    chunk, head = tl.program_id(0), tl.program_id(1)
    row, col, reduction = tl.arange(0, 16), tl.arange(0, DIM), tl.arange(0, 64)
    address = TABLE + chunk * 8
    sm_address = tl.load(address)
    codes = tl.load(address + 1).to(tl.pointer_type(tl.uint8))
    bases = tl.load(address + 2).to(tl.pointer_type(tl.uint8))
    offsets = tl.load(address + 3).to(tl.pointer_type(tl.int32))
    literals = tl.load(address + 4).to(tl.pointer_type(tl.uint8))
    check = tl.load(address + 5).to(tl.pointer_type(tl.int16))
    cos_ptr = tl.load(address + 6).to(tl.pointer_type(tl.bfloat16))
    sin_ptr = tl.load(address + 7).to(tl.pointer_type(tl.bfloat16))
    raw = tl.load(RAW + chunk) != 0
    ak, av = tl.full((16, DIM), 0, tl.float32), tl.full((16, DIM), 0, tl.float32)
    for start in range(WIDTH // 64):
        if raw:
            words = _read(
                sm_address.to(tl.pointer_type(tl.uint16)),
                codes,
                bases,
                offsets,
                literals,
                check,
                row,
                start * 64,
                16,
                WIDTH,
                True,
            )
        else:
            words = _read(
                sm_address.to(tl.pointer_type(tl.uint8)),
                codes,
                bases,
                offsets,
                literals,
                check,
                row,
                start * 64,
                16,
                WIDTH,
                False,
            )
        x = words.to(tl.bfloat16, bitcast=True)
        k = start * 64 + reduction
        wk = tl.load(WK + (head * DIM + col[None, :]) * WIDTH + k[:, None])
        wv = tl.load(WV + (head * DIM + col[None, :]) * WIDTH + k[:, None])
        ak, av = tl.dot(x, wk, ak), tl.dot(x, wv, av)
    key = ak.to(tl.bfloat16).to(tl.float32)
    rotated = tl.gather(key, tl.broadcast_to(((col + DIM // 2) % DIM)[None, :], (16, DIM)), 1)
    rotated = tl.where(col[None, :] < DIM // 2, -rotated, rotated)
    cos = tl.load(cos_ptr + row[:, None] * DIM + col[None, :]).to(tl.float32)
    sin = tl.load(sin_ptr + row[:, None] * DIM + col[None, :]).to(tl.float32)
    first = (key * cos).to(tl.bfloat16).to(tl.float32)
    second = (rotated * sin).to(tl.bfloat16).to(tl.float32)
    pointer = head * HEAD_OUT + (chunk * 16 + row[:, None]) * DIM + col[None, :]
    tl.store(OK + pointer, (first + second).to(tl.bfloat16))
    tl.store(OV + pointer, av.to(tl.bfloat16))


def fill(table, attention, targets):
    import torch

    from mosaickv.evaluation.dual_projection_kernel import apply_patch

    heads = attention.k_proj.weight.shape[0] // 128
    if attention.head_dim != 128 or any(
        p.bias is not None
        or p.weight.dtype != torch.bfloat16
        or p.weight.shape != (heads * 128, heads * 128)
        or not p.weight.is_contiguous()
        for p in (attention.k_proj, attention.v_proj)
    ):
        raise NotImplementedError("validated bias-free BF16 full-MHA square projections only")
    if any(
        t.shape != (1, heads, table.count * 16, 128) or t.stride(-2) != 128 or t.stride(-1) != 1
        for t in targets
    ) or targets[0].stride(-3) != targets[1].stride(-3):
        raise ValueError("matching original MHA strided history leases required")
    _project[(table.count, heads)](
        table.pointers,
        table.raw,
        attention.k_proj.weight,
        attention.v_proj.weight,
        *targets,
        HEAD_OUT=targets[0].stride(-3),
        WIDTH=heads * 128,
        DIM=128,
        num_warps=4,
        enable_fp_fusion=False,
    )
    for patch, target in zip(table.patches, targets, strict=True):
        apply_patch(patch, target)
