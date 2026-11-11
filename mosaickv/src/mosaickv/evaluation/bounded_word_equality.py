"""Bitwise BF16 comparison without a full-size contiguous verification copy.

CUDA uses two bounded reduction kernels and the caller's already charged head
scratch.  The CPU fallback is a fixture implementation, not CUDA evidence.
"""

from functools import cache

import torch


def _validate(first, second, scratch):
    if (
        not isinstance(first, torch.Tensor)
        or not isinstance(second, torch.Tensor)
        or first.dtype != torch.bfloat16
        or second.dtype != torch.bfloat16
        or first.shape != second.shape
        or first.ndim != 4
        or first.numel() < 1
        or first.numel() >= 2**31
        or first.device != second.device
        or first.stride(-1) != 1
        or second.stride(-1) != 1
        or any(stride < 0 for stride in (*first.stride(), *second.stride()))
    ):
        raise ValueError("same-device positive BF16 BHND tensors with unit width stride required")
    blocks = (first.numel() + 255) // 256
    if (
        not isinstance(scratch, torch.Tensor)
        or scratch.dtype != torch.bfloat16
        or scratch.device != first.device
        or not scratch.is_contiguous()
        or scratch.numel() % 2
        or scratch.numel() // 2 < blocks + 1
    ):
        raise ValueError("bounded contiguous same-device BF16 reduction scratch required")
    scratch_storage = scratch.untyped_storage()
    start, stop = scratch_storage.data_ptr(), scratch_storage.data_ptr() + scratch_storage.nbytes()
    for tensor in (first, second):
        storage = tensor.untyped_storage()
        other_start, other_stop = storage.data_ptr(), storage.data_ptr() + storage.nbytes()
        if start < other_stop and other_start < stop:
            raise ValueError("comparison scratch must not alias either immutable input")
    return blocks


@cache
def _cuda_kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def partial(
        A,
        B,
        COUNTS,
        H: tl.constexpr,
        N: tl.constexpr,
        D: tl.constexpr,
        TOTAL: tl.constexpr,
        A0: tl.constexpr,
        A1: tl.constexpr,
        A2: tl.constexpr,
        B0: tl.constexpr,
        B1: tl.constexpr,
        B2: tl.constexpr,
        TILE: tl.constexpr,
    ):
        offset = tl.program_id(0) * TILE + tl.arange(0, TILE)
        width = offset % D
        token = offset // D % N
        head = offset // (D * N) % H
        batch = offset // (D * N * H)
        first_word = tl.load(
            A + batch * A0 + head * A1 + token * A2 + width, offset < TOTAL, other=0
        ).to(tl.uint16, bitcast=True)
        second_word = tl.load(
            B + batch * B0 + head * B1 + token * B2 + width, offset < TOTAL, other=0
        ).to(tl.uint16, bitcast=True)
        different = (offset < TOTAL) & (first_word != second_word)
        tl.store(COUNTS + tl.program_id(0), tl.sum(different.to(tl.int32), 0))

    @triton.jit
    def finish(COUNTS, BLOCKS: tl.constexpr, TILE: tl.constexpr):
        index = tl.arange(0, TILE)
        counts = tl.load(COUNTS + index, index < BLOCKS, other=0)
        tl.store(COUNTS + BLOCKS, tl.sum(counts, 0))

    return partial, finish


@torch.no_grad()
def equal_words(first, second, *, scratch):
    """Compare original words, including signed zero and NaN payload bits.

    The supplied contiguous BF16 scratch is overwritten, never allocated here.
    BHND storage can be head-major or a transposed N-major projection.  Input
    views and their backing are not modified or made contiguous.
    """
    blocks = _validate(first, second, scratch)
    if first.device.type == "cpu":
        return torch.equal(first.view(torch.int16), second.view(torch.int16))
    if first.device.type != "cuda":
        raise NotImplementedError("CPU fixtures or allocated CUDA required")
    import triton

    partial, finish = _cuda_kernels()

    counts = scratch.view(torch.int32).view(-1)
    partial[(blocks,)](
        first,
        second,
        counts,
        *first.shape[1:],
        first.numel(),
        *first.stride()[:3],
        *second.stride()[:3],
        256,
    )
    finish[(1,)](counts, blocks, triton.next_power_of_2(blocks))
    return int(counts[blocks].item()) == 0


@torch.no_grad()
def selftest(*, device="cuda"):
    """Fixture allocation is allowed here; the comparator itself never clones."""
    from mosaickv.evaluation.bounded_component_attention import ComponentWorkspace

    device = torch.device(device)
    workspace = ComponentWorkspace(device=device)
    cases = []
    for length in (7, 600, 604, 1024):
        first = torch.arange(32 * length * 128, device=device).to(torch.bfloat16)
        first = first.view(1, 32, length, 128)
        second = first.transpose(1, 2).contiguous().transpose(1, 2)
        equal = equal_words(first, second, scratch=workspace.head)
        second.view(torch.int16)[0, 17, length // 2, 7] ^= 1
        different = not equal_words(first, second, scratch=workspace.head)
        cases.append(
            dict(
                length=length,
                different_strides=True,
                identical_words_pass=equal,
                one_word_difference_detected=different,
            )
        )
    words = torch.tensor([0, -32768, 32704, 32705], device=device, dtype=torch.int16)
    first = words.view(torch.bfloat16).view(1, 1, 1, 4)
    second = first.clone()
    special_equal = equal_words(first, second, scratch=workspace.head)
    second.view(torch.int16)[0, 0, 0, 0] = -32768
    signed_zero = not equal_words(first, second, scratch=workspace.head)
    second.view(torch.int16).copy_(words.view(1, 1, 1, 4))
    second.view(torch.int16)[0, 0, 0, 2] = 32705
    nan_payload = not equal_words(first, second, scratch=workspace.head)
    return dict(
        all_passed=all(
            row["identical_words_pass"] and row["one_word_difference_detected"] for row in cases
        )
        and special_equal
        and signed_zero
        and nan_payload,
        cases=cases,
        signed_zero_difference_detected=signed_zero,
        nan_payload_difference_detected=nan_payload,
        identical_nan_payloads_pass=special_equal,
        comparator_full_component_copy_allocations=0,
        workspace_bytes=workspace.backing_bytes,
        fixture_allocations_not_serving_memory=True,
    )
