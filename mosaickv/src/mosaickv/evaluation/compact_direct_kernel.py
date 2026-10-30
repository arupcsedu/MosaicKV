"""Compact projection -> original-position RoPE/gather and lossless direct scatter."""

from typing import Any

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["COUNT", "KEPT", "HEAD_OUT"])
def _gather(
    INPUT,
    COS,
    SIN,
    SELECTED,
    MAP,
    OUTPUT,
    COUNT,
    KEPT,
    HEAD_OUT,
    HEAD_IN: tl.constexpr,
    SEQ_IN: tl.constexpr,
    SELECT_STRIDE: tl.constexpr,
    DIM: tl.constexpr,
    KEY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offset < COUNT
    dimension = offset % DIM
    head = offset // (KEPT * DIM)
    slot = offset // DIM % KEPT
    logical = tl.load(SELECTED + head * SELECT_STRIDE + slot, mask=valid, other=0)
    row = tl.load(MAP + logical, mask=valid, other=-1)
    present = valid & (row >= 0)
    base = head * HEAD_IN + tl.maximum(row, 0) * SEQ_IN
    value = tl.load(INPUT + base + dimension, mask=present, other=0)
    if KEY:
        other = tl.load(INPUT + base + (dimension + DIM // 2) % DIM, mask=present, other=0).to(
            tl.float32
        )
        rotated = tl.where(dimension < DIM // 2, -other, other)
        cos = tl.load(COS + logical * DIM + dimension, mask=present, other=0).to(tl.float32)
        sin = tl.load(SIN + logical * DIM + dimension, mask=present, other=0).to(tl.float32)
        first = (value.to(tl.float32) * cos).to(tl.bfloat16).to(tl.float32)
        second = (rotated * sin).to(tl.bfloat16).to(tl.float32)
        value = (first + second).to(tl.bfloat16)
    value = tl.where(present, value, 0)
    tl.store(OUTPUT + head * HEAD_OUT + slot * DIM + dimension, value, mask=valid)


def gather(source: Any, archive: Any, mapping: Any, target: Any, *, key: bool) -> None:
    heads, kept = archive.selected.shape
    if (
        source.dtype != target.dtype
        or source.shape[1] != heads
        or tuple(target.shape) != (1, heads, kept, source.shape[-1])
        or target.stride(-1) != 1
        or target.stride(-2) != source.shape[-1]
        or target.stride(-3) < kept * source.shape[-1]
    ):
        raise ValueError("compatible nonoverlapping compact-projection destination required")
    _gather[(triton.cdiv(target.numel(), 256),)](
        source,
        archive.cos,
        archive.sin,
        archive.selected,
        mapping,
        target,
        target.numel(),
        kept,
        target.stride(-3),
        HEAD_IN=source.stride(1),
        SEQ_IN=source.stride(2),
        SELECT_STRIDE=archive.selected.stride(0),
        DIM=source.shape[-1],
        KEY=key,
        BLOCK=256,
        enable_fp_fusion=False,
    )


@triton.jit(do_not_specialize=["COUNT", "HEAD_OUT"])
def _scatter(
    SM,
    CODES,
    BASE,
    OFFSETS,
    LITERALS,
    HEADS,
    POSITIONS,
    OUTPUT,
    COUNT,
    HEAD_OUT,
    DIM: tl.constexpr,
    RAW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    block = tl.program_id(0)
    local = tl.arange(0, BLOCK)
    index = block * BLOCK + local
    valid = index < COUNT
    if RAW:
        bits = tl.load(SM + index, mask=valid, other=0).to(tl.uint16)
    else:
        bit = local * 3
        byte = bit // 8
        shift = bit % 8
        first = tl.load(CODES + block * (BLOCK * 3 // 8) + byte)
        second = tl.load(
            CODES + block * (BLOCK * 3 // 8) + byte + 1, mask=byte + 1 < BLOCK * 3 // 8, other=0
        )
        code = ((first.to(tl.int32) >> shift) | (second.to(tl.int32) << (8 - shift))) & 7
        escaped = (code == 7) & valid
        rank = tl.cumsum(escaped.to(tl.int32)) - 1
        offset = tl.load(OFFSETS + block)
        literal = tl.load(LITERALS + offset + rank, mask=escaped, other=0).to(tl.int32)
        exponent = tl.where(escaped, literal, tl.load(BASE + block).to(tl.int32) + code)
        sm = tl.load(SM + index, mask=valid, other=0).to(tl.int32)
        bits = ((sm & 127) | ((sm & 128) << 8) | (exponent << 7)).to(tl.uint16)
    slot = index // DIM
    head = tl.load(HEADS + slot, mask=valid, other=0)
    position = tl.load(POSITIONS + slot, mask=valid, other=0)
    tl.store(OUTPUT + head * HEAD_OUT + position * DIM + index % DIM, bits, mask=valid)


def scatter(body: Any, coordinates: Any, target: Any) -> None:
    import torch

    count = coordinates.shape[-1] * target.shape[-1]
    if not count:
        return
    if body.raw:
        raw = body.buffers[0].view(torch.uint16)
        buffers = (raw,) * 5
    else:
        buffers = body.buffers
    _scatter[(triton.cdiv(count, 1024),)](
        *buffers,
        coordinates[0],
        coordinates[1],
        target.view(torch.uint16),
        count,
        target.stride(-3),
        DIM=target.shape[-1],
        RAW=body.raw,
        BLOCK=1024,
    )


def gpu_controls() -> list[dict[str, Any]]:
    from types import SimpleNamespace

    import torch
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    from mosaickv.evaluation.lossless_bf16 import PackedBF16
    from mosaickv.evaluation.lossless_bf16_kernel import encode

    torch.manual_seed(20261030)
    union = torch.tensor([0, 2, 4, 6, 8, 10, 12, 14, 16], device="cuda")
    mapping = torch.full((17,), -1, device="cuda", dtype=torch.int32)
    mapping[union] = torch.arange(9, device="cuda", dtype=torch.int32)
    positions = torch.tensor([[0, 1, 4, 7, 16], [2, 3, 6, 9, 14], [4, 5, 8, 11, 12]], device="cuda")
    phase = torch.randn(1, 17, 16, device="cuda")
    archive = SimpleNamespace(
        selected=positions, cos=phase.cos().bfloat16(), sin=phase.sin().bfloat16()
    )
    source = torch.randn(1, 9, 3, 16, device="cuda").bfloat16().transpose(1, 2)
    canvas = torch.zeros(1, 3, 17, 16, device="cuda", dtype=torch.bfloat16)
    canvas.index_copy_(2, union, source)
    _, rotated = apply_rotary_pos_emb(canvas[:, :1], canvas, archive.cos, archive.sin)
    results = []
    for key in (True, False):
        reference = (rotated if key else canvas).gather(
            2, positions[None, :, :, None].expand(1, 3, 5, 16)
        )
        present = mapping[positions] >= 0
        reference.masked_fill_(~present[None, :, :, None], 0)
        backing = torch.full((1, 3, 11, 16), -71, device="cuda", dtype=torch.bfloat16)
        target = backing[..., 3:8, :]
        gather(source, archive, mapping, target, key=key)
        same = torch.equal(target.view(torch.int16), reference.view(torch.int16))
        sentinel = bool((backing[..., :3, :] == -71).all() and (backing[..., 8:, :] == -71).all())
        if not same or not sentinel:
            raise RuntimeError("compact direct RoPE/gather violated bits or strided sentinels")
        results.append(dict(key=key, bitwise_equal=same, sentinels_unchanged=sentinel))
    coordinates = torch.tensor([[0, 1, 2, 0, 2], [0, 1, 2, 4, 4]], device="cuda", dtype=torch.int32)
    values = torch.randn(1, 1, 5, 16, device="cuda").bfloat16()
    for raw in (False, True):
        body = (
            PackedBF16(tuple(values.shape), True, (values.clone(),))
            if raw
            else encode(values, raw_fallback=False)
        )
        backing = torch.full((1, 3, 11, 16), -71, device="cuda", dtype=torch.bfloat16)
        expected = backing.clone()
        window = backing[..., 3:8, :]
        expected[..., 3:8, :][0, coordinates[0], coordinates[1], :] = values.reshape(5, 16)
        scatter(body, coordinates, window)
        same = torch.equal(backing.view(torch.int16), expected.view(torch.int16))
        if not same:
            raise RuntimeError("compact direct lossless scatter corrupted surrounding cache")
        results.append(dict(scatter_raw=raw, bitwise_equal=same, sentinels_unchanged=True))
    return results
