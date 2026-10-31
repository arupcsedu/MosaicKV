"""In-place uint16 cycle rotation: one register row per disjoint head cycle."""

import triton
import triton.language as tl


@triton.jit
def _cycle(K, V, HEADS, OFFSETS, ROWS, SHK, SHV, SNK, SNV, D: tl.constexpr):
    cycle = tl.program_id(0)
    head = tl.load(HEADS + cycle).to(tl.int32)
    begin = tl.load(OFFSETS + cycle)
    end = tl.load(OFFSETS + cycle + 1)
    first = tl.load(ROWS + begin).to(tl.int32)
    col = tl.arange(0, D)
    k = tl.load(K + head * SHK + first * SNK + col)
    v = tl.load(V + head * SHV + first * SNV + col)
    offset = begin + 1
    while offset < end:
        row = tl.load(ROWS + offset).to(tl.int32)
        next_k = tl.load(K + head * SHK + row * SNK + col)
        next_v = tl.load(V + head * SHV + row * SNV + col)
        tl.store(K + head * SHK + row * SNK + col, k)
        tl.store(V + head * SHV + row * SNV + col, v)
        k, v = next_k, next_v
        offset += 1
    tl.store(K + head * SHK + first * SNK + col, k)
    tl.store(V + head * SHV + first * SNV + col, v)


def apply(plan, targets):
    import torch

    if any(t.shape[-1] != 128 or t.device != plan.heads.device for t in targets):
        raise NotImplementedError("audited width128 same-device cache cycle required")
    _cycle[(plan.heads.numel(),)](
        targets[0].view(torch.uint16),
        targets[1].view(torch.uint16),
        plan.heads,
        plan.offsets,
        plan.rows,
        targets[0].stride(1),
        targets[1].stride(1),
        targets[0].stride(2),
        targets[1].stride(2),
        D=128,
        num_warps=4,
    )
