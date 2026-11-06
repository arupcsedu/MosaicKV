"""Exact closeness-mask operator and default-off private constructor scope.

Retain canonical dynamic torch.nonzero and exact BF16 repair words. Only fuse
the FP32 isclose predicate and output-finiteness scan; never change tolerances.
The only runtime scope is an explicitly registered consumed-input diagnostic.
"""

import random
import statistics
import time
from contextlib import contextmanager
from contextvars import ContextVar

import torch
import triton
import triton.language as tl

LIMIT = 4 * 1024 * 1024
_ABSENT = object()
_scope = ContextVar("evicastkv_fused_transport_predicate", default=_ABSENT)


@triton.jit
def _mask(reference, actual, bad, finite, N: tl.constexpr, BLOCK: tl.constexpr):
    block = tl.program_id(0)
    index = block * BLOCK + tl.arange(0, BLOCK)
    live = index < N
    x = tl.load(reference + index, live, other=0).to(tl.float32)
    y = tl.load(actual + index, live, other=0).to(tl.float32)
    fx = (x == x) & (tl.abs(x) != float("inf"))
    fy = (y == y) & (tl.abs(y) != float("inf"))
    # Match separate FP32 ATen operations, not an FMA or BF16 subtraction.
    error = tl.abs(x - y)
    tolerance = 0.015625 + 0.01 * tl.abs(y)
    close = (x == y) | (fx & fy & (error <= tolerance))
    tl.store(bad + index, ~close, live)
    tl.store(finite + block, tl.sum((live & ~fy).to(tl.int32), 0) == 0)


def validate(reference, actual):
    if (
        reference.device != actual.device
        or reference.shape != actual.shape
        or reference.dtype != torch.bfloat16
        or actual.dtype != reference.dtype
        or not reference.is_contiguous()
        or not actual.is_contiguous()
        or not 0 < reference.numel() <= LIMIT
    ):
        raise ValueError("same-device contiguous bounded nonempty BF16 pair required")
    if reference.device.type != "cuda":
        raise NotImplementedError("CUDA-only diagnostic operator; no CPU imitation")


def fused(reference, actual, *, out=None):
    validate(reference, actual)
    count = reference.numel()
    bad = out
    if bad is None:
        bad = torch.empty(reference.shape, device=reference.device, dtype=torch.bool)
    elif (
        bad.dtype != torch.bool
        or bad.device != reference.device
        or bad.numel() != count
        or not bad.is_contiguous()
        or bad.untyped_storage().data_ptr()
        in (reference.untyped_storage().data_ptr(), actual.untyped_storage().data_ptr())
    ):
        raise ValueError("independent contiguous same-device Boolean correction mask required")
    flags = torch.empty(triton.cdiv(count, 1024), device=reference.device, dtype=torch.bool)
    _mask[(flags.numel(),)](
        reference, actual, bad, flags, count, 1024, enable_fp_fusion=False, num_warps=4
    )
    return bad, flags.all()


class Transaction:
    def __init__(self):
        self.calls, self.maximum_scalars, self.closed = 0, 0, False

    def report(self):
        return dict(
            enabled=True,
            calls=self.calls,
            maximum_scalars=self.maximum_scalars,
            hard_bound_scalars=LIMIT,
            closed_before_decode=self.closed,
            original_canonical_nonzero_and_BF16_words=True,
            retained_tensor_owners=0,
            unchanged_finite_protected_and_close_commit=True,
        )


def active():
    return isinstance(_scope.get(), Transaction)


def fill_mask(reference, actual, mask, *, quantum):
    state = _scope.get()
    if not isinstance(state, Transaction) or state.closed:
        raise RuntimeError("private active fused predicate transaction required")
    if quantum != reference.numel() or mask is None:
        raise ValueError("unchanged bounded full-span coalesced correction mask required")
    bad, finite = fused(reference, actual, out=mask)
    if bad is not mask:
        raise RuntimeError("fused predicate must use the original coalesced mask owner")
    state.calls += 1
    state.maximum_scalars = max(state.maximum_scalars, reference.numel())
    return finite


@contextmanager
def use(enabled):
    if type(enabled) is not bool or _scope.get() is not _ABSENT:
        raise ValueError("explicit nonnested fused predicate transaction required")
    state = Transaction() if enabled else None
    token = _scope.set(state)
    try:
        yield state
    finally:
        if state is not None:
            state.closed = True
        _scope.reset(token)


def original(reference, actual):
    return (
        ~torch.isclose(reference.float(), actual.float(), rtol=0.01, atol=2**-6),
        torch.isfinite(actual).all(),
    )


def stage(function, reference, actual):
    bad, finite = function(reference, actual)
    indices = bad.flatten().nonzero().flatten()
    return (
        indices.to(torch.int32),
        reference.flatten().index_select(0, indices).clone(),
        finite,
    )


@torch.inference_mode()
def controls():
    records = []
    # Every BF16 bit pattern, including signed zero, subnormal, Inf and NaN.
    words = torch.arange(65536, dtype=torch.int32, device="cuda").to(torch.int16)
    values = words.view(torch.bfloat16)
    references = (
        values,
        values,
        values,
        values,
        torch.tensor([0, -0.0, 1, 2**-6, float("inf"), float("nan")], device="cuda").bfloat16(),
    )
    actuals = (
        values.clone(),
        values.roll(1),
        values.roll(128),
        (values.float() * 1.01 + 2**-6).bfloat16(),
        torch.tensor([-0.0, 0, 1, 0, -float("inf"), float("nan")], device="cuda").bfloat16(),
    )
    for index, (reference, actual) in enumerate(zip(references, actuals, strict=True)):
        expected, got = original(reference, actual), fused(reference, actual)
        if not all(torch.equal(x, y) for x, y in zip(expected, got, strict=True)):
            raise RuntimeError(f"fused predicate differs from pinned ATen fixture {index}")
        left, right = stage(original, reference, actual), stage(fused, reference, actual)
        if not all(
            torch.equal(x.view(torch.uint8), y.view(torch.uint8))
            for x, y in zip(left, right, strict=True)
        ):
            raise RuntimeError("canonical repair indices, BF16 words or finite flag differ")
        records.append(
            dict(fixture=index, scalars=reference.numel(), mask_and_words_identical=True)
        )
    generator = torch.Generator(device="cuda").manual_seed(20261106)
    for size in (1, 17, 1023, 1024, 1025, LIMIT - 1, LIMIT):
        reference = torch.randn(size, device="cuda", generator=generator).bfloat16()
        actual = reference.clone()
        actual[::127] += 0.25
        expected, got = original(reference, actual), fused(reference, actual)
        if not all(torch.equal(x, y) for x, y in zip(expected, got, strict=True)):
            raise RuntimeError("tail/bound predicate differs from pinned ATen")
        records.append(dict(scalars=size, mask_and_finite_identical=True))
    return records


@torch.inference_mode()
def study():
    checked = controls()  # Any mismatch aborts BEFORE performance measurement.
    records = []
    generator = torch.Generator(device="cuda").manual_seed(20261106)
    rng = random.Random(20261106)
    for length in (600, 604):
        reference = torch.randn((1, 32, length, 128), device="cuda", generator=generator).bfloat16()
        actual = reference.clone()
        actual.flatten()[::4096] += 0.25
        for queued_projection in (False, True):
            matrices = (
                (
                    torch.randn((length, 4096), device="cuda", generator=generator).bfloat16(),
                    torch.randn((4096, 4096), device="cuda", generator=generator).bfloat16(),
                )
                if queued_projection
                else None
            )
            trials = []
            for trial in range(-5, 20):
                order = [("original", original), ("fused", fused)]
                rng.shuffle(order)
                for place, (name, function) in enumerate(order):
                    torch.cuda.synchronize()
                    allocated = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                    before = time.perf_counter()
                    start.record()
                    projected = torch.mm(*matrices) if matrices is not None else None
                    result = stage(function, reference, actual)
                    end.record()
                    torch.cuda.synchronize()
                    wall = time.perf_counter() - before
                    trials.append(
                        dict(
                            trial=trial,
                            warmup=trial < 0,
                            method=name,
                            order=place,
                            wall_seconds=wall,
                            gpu_seconds=start.elapsed_time(end) / 1000,
                            torch_allocator_peak_increment_bytes=(
                                torch.cuda.max_memory_allocated() - allocated
                            ),
                            allocator_boundary_NOT_vendor_or_whole_request_peak=True,
                        )
                    )
                    del result, projected
            means = {
                name: statistics.mean(
                    t["wall_seconds"] for t in trials if t["method"] == name and not t["warmup"]
                )
                for name in ("original", "fused")
            }
            records.append(
                dict(
                    source_length=length,
                    queued_projection=queued_projection,
                    means=means,
                    fused_over_original=means["fused"] / means["original"],
                    trials=trials,
                    synthetic_operator_only=True,
                )
            )
    return dict(
        complete=True,
        failures=[],
        controls=checked,
        records=records,
        operator_nonregression_all_shapes=all(r["fused_over_original"] <= 1 for r in records)
        and any(r["fused_over_original"] <= 0.96 for r in records),
        no_runtime_integration=True,
        no_model_latency_or_quality_acceptance=True,
        independent_quality_N=0,
    )
