"""Opt-in BF16 staged-rounding RoPE/gather; audited by original-cache bit checks."""

from typing import Any

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["COUNT", "KEPT"])
def _rope_gather(
    KEY,
    COS,
    SIN,
    POSITIONS,
    OUTPUT,
    COUNT,
    KEPT,
    HEAD_STRIDE: tl.constexpr,
    SEQ_STRIDE: tl.constexpr,
    POS_STRIDE: tl.constexpr,
    COS_STRIDE: tl.constexpr,
    SIN_STRIDE: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offset < COUNT
    dim = offset % DIM
    head = offset // (KEPT * DIM)
    slot = offset // DIM % KEPT
    position = tl.load(POSITIONS + head * POS_STRIDE + slot, mask=valid, other=0)
    base = head * HEAD_STRIDE + position * SEQ_STRIDE
    key = tl.load(KEY + base + dim, mask=valid, other=0).to(tl.float32)
    other = tl.load(KEY + base + (dim + DIM // 2) % DIM, mask=valid, other=0).to(tl.float32)
    rotated = tl.where(dim < DIM // 2, -other, other)
    cos = tl.load(COS + position * COS_STRIDE + dim, mask=valid, other=0).to(tl.float32)
    sin = tl.load(SIN + position * SIN_STRIDE + dim, mask=valid, other=0).to(tl.float32)
    # Match the two separate PyTorch BF16 multiplies followed by BF16 addition.
    first = (key * cos).to(tl.bfloat16).to(tl.float32)
    second = (rotated * sin).to(tl.bfloat16).to(tl.float32)
    tl.store(OUTPUT + offset, first + second, mask=valid)


def rope_gather(key: Any, cos: Any, sin: Any, positions: Any) -> Any:
    import torch

    if (
        key.device.type != "cuda"
        or key.dtype != torch.bfloat16
        or cos.dtype != torch.bfloat16
        or sin.dtype != torch.bfloat16
        or key.shape[0] != 1
        or key.shape[-1] % 2
        or key.stride(-1) != 1
        or cos.stride(-1) != 1
        or sin.stride(-1) != 1
        or positions.ndim != 2
        or positions.shape[0] != key.shape[1]
        or positions.shape[1] < 1
        or positions.stride(-1) != 1
        or positions.dtype != torch.int64
        or any(t.device != key.device for t in (cos, sin, positions))
        or cos.shape != sin.shape
        or cos.shape != (1, key.shape[-2], key.shape[-1])
    ):
        raise ValueError("batch-one CUDA BF16 MHA with contiguous head dimensions required")
    heads, kept = positions.shape
    result = torch.empty((1, heads, kept, key.shape[-1]), device=key.device, dtype=key.dtype)
    _rope_gather[(triton.cdiv(result.numel(), 256),)](
        key,
        cos,
        sin,
        positions,
        result,
        result.numel(),
        kept,
        HEAD_STRIDE=key.stride(1),
        SEQ_STRIDE=key.stride(2),
        POS_STRIDE=positions.stride(0),
        COS_STRIDE=cos.stride(1),
        SIN_STRIDE=sin.stride(1),
        DIM=key.shape[-1],
        BLOCK=256,
        enable_fp_fusion=False,
    )
    return result


def gpu_control() -> list[Any]:
    """Bounded random/zero, irregular positions and stride-zero index control."""
    import torch
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    torch.manual_seed(20261030)
    results = []
    for shared in (False, True):
        key = torch.randn((1, 17, 3, 16), device="cuda", dtype=torch.bfloat16).transpose(1, 2)
        key[0, 0, 0, 0] = 0.0
        key[0, 0, 0, 1] = -0.0
        phase = torch.randn((1, 17, 16), device="cuda")
        cos, sin = phase.cos().bfloat16(), phase.sin().bfloat16()
        positions = torch.tensor(
            [[0, 3, 7, 9, 16], [1, 5, 8, 12, 15], [2, 4, 6, 10, 14]], device="cuda"
        )
        if shared:
            positions = positions[:1].expand(3, -1)
        _, rotated = apply_rotary_pos_emb(key[:, :1], key, cos, sin)
        reference = rotated.gather(2, positions[None, :, :, None].expand(1, 3, 5, 16))
        actual = rope_gather(key, cos, sin, positions)
        equal = torch.equal(actual.view(torch.int16), reference.view(torch.int16))
        results.append(
            dict(
                stride_zero_positions=shared,
                bitwise_equal=equal,
                maximum_absolute_difference=float((actual.float() - reference.float()).abs().max()),
            )
        )
    if not all(r["bitwise_equal"] for r in results):
        raise RuntimeError(f"fused BF16 RoPE control failed: {results}")
    return results
