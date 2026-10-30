"""Direct strided-lease decoding; original BF16 eager attention follows."""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["N", "SH", "PROTECTED"])
def _decode(
    CODES,
    MINIMUM,
    SCALE,
    EXACT_IDX,
    EXACT,
    OUT,
    N,
    D: tl.constexpr,
    H: tl.constexpr,
    GROUP: tl.constexpr,
    BITS: tl.constexpr,
    AXIS: tl.constexpr,
    SH,
    SN: tl.constexpr,
    PROTECTED,
    BLOCK: tl.constexpr,
    OVERLAY: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < H * N * D
    d, n, h = i % D, (i // D) % N, i // (N * D)
    if AXIS == 2:
        transformed = (h * D + d) * N + n
        group = (h * D + d) * tl.cdiv(N, GROUP) + n // GROUP
    else:
        transformed = (h * N + n) * D + d
        group = (h * N + n) * tl.cdiv(D, GROUP) + d // GROUP
    word, shift = transformed // 8, (transformed % 8) * BITS
    packed = tl.full((BLOCK,), 0, tl.int32)
    for j in tl.static_range(BITS):
        packed |= tl.load(CODES + word * BITS + j, mask=valid, other=0).to(tl.int32) << (j * 8)
    q = (packed >> shift) & ((1 << BITS) - 1)
    scale = tl.load(SCALE + group, mask=valid, other=1)
    minimum = tl.load(MINIMUM + group, mask=valid, other=0)
    value = q.to(tl.float32) * scale + minimum
    # Exact overlays included in the same launch; no dense temporary allocation.
    if OVERLAY:
        slot = tl.load(EXACT_IDX + n, mask=valid, other=-1)
        exact = tl.load(EXACT + (h * PROTECTED + slot) * D + d, mask=valid & (slot >= 0), other=0)
        value = tl.where(slot >= 0, exact.to(tl.float32), value)
    tl.store(OUT + h * SH + n * SN + d, value, mask=valid)


def decode(body, destination, indices, exact, *, overlay=True):
    n, d, h = body.shape[-2], body.shape[-1], body.shape[1]
    if (
        destination.shape[:2] != body.shape[:2]
        or destination.shape[-1] != d
        or destination.shape[-2] < n
        or destination.stride(-1) != 1
    ):
        raise ValueError("compatible feature-contiguous full-layer lease required")
    _decode[(triton.cdiv(h * n * d, 512),)](
        body.codes,
        body.minimum,
        body.scale,
        indices,
        exact,
        destination,
        N=n,
        D=d,
        H=h,
        GROUP=body.group,
        BITS=body.bits,
        AXIS=body.axis,
        SH=destination.stride(1),
        SN=destination.stride(2),
        PROTECTED=exact.shape[-2] if overlay else 0,
        BLOCK=512,
        OVERLAY=overlay,
        enable_fp_fusion=False,
    )
