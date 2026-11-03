"""Original bit-exact scatter and disjoint corrections in one launch.

No floating arithmetic, new GPU metadata, output cache, or tensor workspace.
The original constructor's checked non-overlap is mandatory; absent proof fails.
"""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["COUNT", "PATCHES", "N", "HEAD_OUT"])
def _restore(
    SM,
    CODES,
    BASE,
    OFFSETS,
    LITERALS,
    POSITIONS,
    INDICES,
    VALUES,
    OUT,
    COUNT,
    PATCHES,
    N,
    HEAD_OUT,
    D: tl.constexpr,
    RAW: tl.constexpr,
):
    block = tl.program_id(0)
    scatter_blocks = tl.cdiv(COUNT, 1024)
    if block < scatter_blocks:
        local = tl.arange(0, 1024)
        index = block * 1024 + local
        valid = index < COUNT
        if RAW:
            bits = tl.load(SM + index, valid, other=0).to(tl.uint16)
        else:
            byte, shift = local * 3 // 8, local * 3 % 8
            first = tl.load(CODES + block * 384 + byte).to(tl.int32)
            second = tl.load(CODES + block * 384 + byte + 1, byte + 1 < 384, other=0).to(tl.int32)
            code = ((first >> shift) | (second << (8 - shift))) & 7
            escaped = (code == 7) & valid
            rank = tl.cumsum(escaped.to(tl.int32)) - 1
            offset = tl.load(OFFSETS + block)
            literal = tl.load(LITERALS + offset + rank, escaped, other=0).to(tl.int32)
            base = tl.load(BASE + block).to(tl.int32)
            exponent = tl.where(escaped, literal, base + code)
            sm = tl.load(SM + index, valid, other=0).to(tl.int32)
            bits = ((sm & 127) | ((sm & 128) << 8) | (exponent << 7)).to(tl.uint16)
        position = tl.load(POSITIONS + index // D, valid, other=0)
        tl.store(OUT + (position // N) * HEAD_OUT + (position % N) * D + index % D, bits, valid)
    else:
        patch_offset = (block - scatter_blocks) * 256 + tl.arange(0, 256)
        patch_index = tl.load(INDICES + patch_offset, patch_offset < PATCHES, other=0).to(tl.int32)
        patch_bits = tl.load(VALUES + patch_offset, patch_offset < PATCHES, other=0)
        patch_head = patch_index // (N * D)
        patch_position = patch_index // D % N
        tl.store(
            OUT + patch_head * HEAD_OUT + patch_position * D + patch_index % D,
            patch_bits,
            patch_offset < PATCHES,
        )


def restore(packet, component, target):
    import torch

    from mosaickv.evaluation.prompt_source_subphases import phase
    from mosaickv.evaluation.source_hotpath import record_restore

    if not packet.transport_patches or packet.protected_patches_disjoint is not True:
        raise ValueError("constructor-verified disjoint source patches required")
    if type(component) is not int or component not in (0, 1):
        raise ValueError("explicit key/value component required")
    body, patch = packet.exact[component], packet.transport_patches[component]
    n, d = packet.shape[-2:]
    if (
        target.device.type != "cuda"
        or target.dtype != torch.bfloat16
        or target.stride(-1) != 1
        or target.stride(-2) != d
        or patch.indices.dtype != torch.int32
        or patch.values.dtype != torch.bfloat16
    ):
        raise ValueError("original CUDA BF16 lease and int32 patch indices required")
    with phase("indexing_views"):
        buffers = (body.buffers[0].view(torch.uint16),) * 5 if body.raw else body.buffers
        count = body.shape[-2] * d
        patches = patch.indices.numel()
        values, output = patch.values.view(torch.uint16), target.view(torch.uint16)
        grid = (triton.cdiv(count, 1024) + triton.cdiv(patches, 256),)
    with phase("protected_restore_dispatch"):
        compiled = _restore[grid](
            *buffers,
            packet.row_positions,
            patch.indices,
            values,
            output,
            count,
            patches,
            n,
            target.stride(1),
            D=d,
            RAW=body.raw,
            num_warps=4,
        )
    if getattr(compiled.metadata, "global_scratch_size", None) != 0:
        raise RuntimeError("fused restore cannot allocate opaque global scratch")
    record_restore(patches)
