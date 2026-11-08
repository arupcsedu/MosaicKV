"""Single-write exact/prototype/repair fill; no auxiliary CUDA allocations."""

import triton
import triton.language as tl


@triton.jit
def update(
    destination,
    exact_ids,
    exact_values,
    positions,
    member_keys,
    prototype_value,
    promotion,
    promoted,
    counter,
    HEAD_OFFSET: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    EXACT: tl.constexpr,
    MEMBERS: tl.constexpr,
    DIM: tl.constexpr,
    KEY: tl.constexpr,
    DELTA: tl.constexpr,
    BLOCK: tl.constexpr,
):
    lane = tl.arange(0, BLOCK)
    row, column = lane // DIM, lane % DIM
    active = row < EXACT + MEMBERS
    exact = row < EXACT
    member = tl.maximum(row - EXACT, 0)
    a = tl.load(exact_ids + row, active & exact, other=0)
    b = tl.load(positions + member, active & ~exact, other=0)
    position = tl.where(exact, a, b)
    pointer = destination + HEAD_OFFSET + position * ROW_STRIDE + column
    value = tl.load(exact_values + row * DIM + column, active & exact, other=0)
    if KEY:
        proto = tl.load(member_keys + member * DIM + column, active & ~exact, other=0)
    else:
        proto = tl.load(prototype_value + column, active & ~exact, other=0)
    if DELTA:
        parent = tl.load(pointer, active & ~exact, other=0)
        # Match native BF16 add's FP32 intermediate and BF16 rounding BEFORE
        # the final selection. No reassociation or approximate arithmetic.
        proto = (parent.to(tl.float32) + proto.to(tl.float32)).to(tl.bfloat16)
    restored = tl.load(promotion + member * DIM + column, active & ~exact, other=0)
    flag = tl.load(promoted)
    selected = tl.where(exact, value, tl.where(flag, restored, proto))
    tl.store(pointer, selected, active)
    old = tl.load(counter)
    tl.store(counter, old + 1)
