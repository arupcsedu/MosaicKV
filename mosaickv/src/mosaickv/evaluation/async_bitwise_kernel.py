"""Integer-only bit comparison with <= one byte per 4096 source scalars."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["COUNT", "ROWS", "AH", "AR", "AC", "BH", "BR", "BC"])
def _flags(
    A, B, FLAGS, COUNT, ROWS, AH, AR, AC, BH, BR, BC, DIM: tl.constexpr, BLOCK: tl.constexpr
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < COUNT
    head, row, col = index // (ROWS * DIM), index // DIM % ROWS, index % DIM
    left = tl.load(A + head * AH + row * AR + col * AC, valid, other=0).to(tl.int16, bitcast=True)
    right = tl.load(B + head * BH + row * BR + col * BC, valid, other=0).to(tl.int16, bitcast=True)
    equal = tl.sum(((left != right) & valid).to(tl.int32), 0) == 0
    tl.store(FLAGS + tl.program_id(0), equal)


def match_flags(reference, candidate):
    count = reference.numel()
    output = torch.empty((triton.cdiv(count, 4096),), device=reference.device, dtype=torch.bool)
    _flags[(output.numel(),)](
        reference,
        candidate,
        output,
        count,
        reference.shape[-2],
        *reference.stride()[1:],
        *candidate.stride()[1:],
        DIM=reference.shape[-1],
        BLOCK=4096,
        num_warps=4,
    )
    return output
