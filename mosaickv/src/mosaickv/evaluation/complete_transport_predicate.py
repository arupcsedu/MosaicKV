"""Default-off union of BOTH frozen transport predicates, with exact rounding.

ATen v2.8.0 TensorCompare.cpp:isclose computes abs(self-other),
atol+abs(rtol*other), including BF16 rounding at each tensor operation. The
legacy FP32 and actual BF16 predicates are asymmetric; neither may be dropped.
No backend approximation, tolerance change or static nonzero is introduced.
"""

from contextlib import contextmanager
from unittest.mock import patch

import torch
import triton
import triton.language as tl


@triton.jit
def _predicate(Reference, Actual, Mask, Flags, n, VERIFY: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = i < n
    x = tl.load(Reference + i, live, 0).to(tl.float32)
    y = tl.load(Actual + i, live, 0).to(tl.float32)
    equal = x == y
    actual_error = tl.abs((y - x).to(tl.bfloat16).to(tl.float32))
    allowed_error = (
        (tl.abs((0.01 * x).to(tl.bfloat16).to(tl.float32)) + 0.015625)
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    actual_finite = (actual_error == actual_error) & (actual_error != float("inf"))
    required_close = equal | (actual_finite & (actual_error <= allowed_error))
    if VERIFY:
        flag = ~live | required_close
    else:
        error = tl.abs(x - y)
        tolerance = 0.015625 + tl.abs(0.01 * y)
        finite_error = (error == error) & (error != float("inf"))
        legacy_close = equal | (finite_error & (error <= tolerance))
        tl.store(Mask + i, ~(legacy_close & required_close), live)
        flag = ~live | ((y == y) & (tl.abs(y) != float("inf")))
    tl.store(Flags + tl.program_id(0), tl.sum((~flag).to(tl.int32), 0) == 0)


def validate(reference, actual):
    if (
        reference.dtype != torch.bfloat16
        or actual.dtype != reference.dtype
        or reference.shape != actual.shape
        or reference.device != actual.device
        or not reference.is_contiguous()
        or not actual.is_contiguous()
        or not 0 < reference.numel() <= 4 * 1024 * 1024
    ):
        raise ValueError("same-device contiguous bounded BF16 predicate pair required")


def fused(reference, actual, *, verify=False, out=None):
    validate(reference, actual)
    if reference.device.type != "cuda":
        raise NotImplementedError("real CUDA operator required; use oracle for CPU")
    n = reference.numel()
    if out is None and not verify:
        out = torch.empty_like(reference, dtype=torch.bool)
    if out is not None and (
        out.dtype != torch.bool
        or out.device != reference.device
        or not out.is_contiguous()
        or out.numel() != n
        or out.untyped_storage().data_ptr()
        in (reference.untyped_storage().data_ptr(), actual.untyped_storage().data_ptr())
    ):
        raise ValueError("complete independent Boolean mask required")
    flags = torch.empty(triton.cdiv(n, 1024), dtype=torch.bool, device=reference.device)
    compiled = _predicate[(flags.numel(),)](
        reference,
        actual,
        flags if verify else out,
        flags,
        n,
        VERIFY=verify,
        BLOCK=1024,
        enable_fp_fusion=False,
        num_warps=4,
    )
    if compiled.metadata.global_scratch_size or compiled.n_spills:
        raise RuntimeError("complete predicate introduced undeclared workspace")
    return out, flags.all()


def oracle(reference, actual):
    return (
        ~torch.isclose(reference.float(), actual.float(), rtol=0.01, atol=2**-6)
        | ~torch.isclose(actual, reference, rtol=0.01, atol=2**-6, equal_nan=False),
        torch.isfinite(actual).all(),
    )


@torch.inference_mode()
def gpu_controls():
    from mosaickv.evaluation import shared_source_format as source
    from mosaickv.evaluation.deferred_transport_checks import use as validation

    rows = []
    words = torch.arange(65536, dtype=torch.int32, device="cuda").short().view(torch.bfloat16)
    pairs = [(words, words.roll(n)) for n in (0, 1, 128, 8191)]
    pairs += [(words, (words.float() * 1.01 + 2**-6).bfloat16())]
    gen = torch.Generator(device="cuda").manual_seed(20261108)
    for count in (1, 17, 1023, 1024, 1025, 4 * 1024 * 1024 - 1, 4 * 1024 * 1024):
        x = torch.randn(count, generator=gen, device="cuda").bfloat16()
        y = x.clone()
        y[::127] += 0.25
        pairs.append((x, y))
    for number, (x, y) in enumerate(pairs):
        expected, actual = oracle(x, y), fused(x, y)
        if not all(torch.equal(a, b) for a, b in zip(expected, actual, strict=True)):
            raise RuntimeError(
                f"complete predicate union/finite GPU control failed at fixture {number}"
            )
        _, close = fused(x, y, verify=True)
        if not torch.equal(close, torch.isclose(y, x, rtol=0.01, atol=2**-6).all()):
            raise RuntimeError(f"complete verification GPU control failed at fixture {number}")
        rows.append(dict(fixture=number, union_finite_and_verification_bitwise=True))
    x = torch.randn(1, 32, 603, 128, generator=gen, device="cuda").bfloat16()
    y = x.clone()
    y.flatten()[::127] += 0.25
    empty = torch.empty(0, device="cuda", dtype=torch.int32)
    with validation(True, coalesce_nonzero=True, full_predicate_span=True):
        old = source.transport_corrections((x,), (y,), empty)[0]
    with use() as report, validation(True, coalesce_nonzero=True, full_predicate_span=True):
        new = source.transport_corrections((x,), (y,), empty, chunk_scalars=524288)[0]
        source.assert_transport_close(x, x.clone())
    if not all(torch.equal(a, b) for a, b in zip(old.tensors, new.tensors, strict=True)):
        raise RuntimeError("complete predicate changed canonical patch indices/values")
    rows.append(dict(canonical_patch_words_identical=True, report=report))
    for kind, error in (
        ("finite", ValueError),
        ("protected", RuntimeError),
        ("close", AssertionError),
    ):
        a = torch.ones(1, 1, 8, 128, dtype=torch.bfloat16, device="cuda")
        b = a.clone()
        b.flatten()[-1] = float("nan") if kind == "finite" else 2.0
        try:
            with use(), validation(True, coalesce_nonzero=True, full_predicate_span=True):
                if kind == "close":
                    source.assert_transport_close(b, a)
                else:
                    source.transport_corrections(
                        (a,), (b,), torch.tensor([7], dtype=torch.int32, device="cuda")
                    )
        except error:
            rows.append(dict(failure=kind, rejected_before_commit=True))
        else:
            raise RuntimeError("failed GPU predicate was not rejected: " + kind)
    return rows


@contextmanager
def use():
    from mosaickv.evaluation import shared_source_format as source
    from mosaickv.evaluation import static_transport_patches as static
    from mosaickv.evaluation.deferred_transport_checks import (
        correction_mask,
        enqueue,
        predicate_span,
    )
    from mosaickv.evaluation.exact_storage import SparseBitPatch

    original_verify = source.assert_transport_close
    report = dict(
        enabled=True,
        common_to_every_method=True,
        complete_FP32_and_BF16_union=True,
        correction_calls=0,
        verification_calls=0,
        unsupported_operator_fallbacks=0,
        no_threshold_change=True,
    )

    def corrections(reference, actual, protected_rows, *, chunk_scalars=None):
        if static.enabled():
            raise ValueError("registered dynamic full-span correction transaction required")
        if chunk_scalars is not None and (
            type(chunk_scalars) is not int or chunk_scalars < 1
        ):
            raise ValueError("positive explicit correction chunk hint required")
        patches = []
        for expected, output in zip(reference, actual, strict=True):
            validate(expected, output)
            n = expected.numel()
            if predicate_span(n, chunk_scalars, kind="corrections") != n:
                raise ValueError("registered full-span validation required")
            mask = correction_mask(n, expected.device)
            if mask is None:
                raise ValueError("original coalesced mask required")
            mask, finite = fused(expected, output, out=mask)
            if not enqueue(finite, "finite") and not bool(finite):
                raise ValueError("finite compatible BF16 construction pair required")
            ids = mask.flatten().nonzero().flatten()
            protected = torch.isin(ids // expected.shape[-1], protected_rows.long()).any()
            if not enqueue(protected, "protected", expected=False) and bool(protected):
                raise RuntimeError("protected K/V mismatch may not be hidden by a numerical patch")
            patches.append(
                SparseBitPatch(ids.to(torch.int32), expected.flatten().index_select(0, ids).clone())
            )
            report["correction_calls"] += 1
        return tuple(patches)

    def verify(output, expected, *, chunk_scalars=524288):
        if type(chunk_scalars) is not int or chunk_scalars < 1:
            raise ValueError("positive explicit verification chunk required")
        validate(expected, output)
        predicate_span(output.numel(), chunk_scalars, kind="verification")
        _, close = fused(expected, output, verify=True)
        report["verification_calls"] += 1
        if not enqueue(close, "close") and not bool(close):
            original_verify(output, expected, chunk_scalars=chunk_scalars)

    with (
        patch.object(source, "transport_corrections", corrections),
        patch.object(source, "assert_transport_close", verify),
    ):
        yield report
