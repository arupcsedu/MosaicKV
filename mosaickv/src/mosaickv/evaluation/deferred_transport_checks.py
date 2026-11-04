"""Request-private validation barrier; identical predicates, no cache math.

Only scalar flags survive until commit. No source KV, activations, labels or
reference logits are retained. Failure aborts before decoding. Default off.
"""

from contextlib import contextmanager
from contextvars import ContextVar

import torch

from mosaickv.evaluation.lossless_prompt_cache import backing_bytes

_state = ContextVar("evicastkv_transport_validation_transaction", default=None)


class Validation:
    def __init__(self, *, coalesce_nonzero=False):
        self.flags, self.kinds, self.expected = [], [], []
        self.device, self.closed, self.flushes = None, False, 0
        self.counts, self.peak_bytes = {}, 0
        self.coalesce_nonzero = coalesce_nonzero
        self.mask_calls, self.mask_peak_bytes, self.mask_bound_fallbacks = 0, 0, 0

    def correction_mask(self, size, device):
        # Constructor-only scratch, never persistent cache or decode workspace.
        # Above the recorded hard bound retain the original chunked operation.
        if not self.coalesce_nonzero:
            return None
        if size > 4 * 1024 * 1024:
            self.mask_bound_fallbacks += 1
            return None
        self.mask_calls += 1
        self.mask_peak_bytes = max(self.mask_peak_bytes, size)
        return torch.empty(size, device=device, dtype=torch.bool)

    def append(self, flag, kind, expected):
        if self.closed or len(self.flags) >= 256:
            raise RuntimeError("closed or exhausted bounded validation transaction")
        if (
            kind not in ("finite", "protected", "close")
            or type(expected) is not bool
            or flag.dtype != torch.bool
            or flag.numel() != 1
            or (self.device is not None and flag.device != self.device)
        ):
            raise ValueError("one-device scalar unchanged Boolean predicate required")
        self.device = flag.device
        self.flags.append(flag)
        self.kinds.append(kind)
        self.expected.append(expected)
        self.counts[kind] = self.counts.get(kind, 0) + 1

    def flush(self):
        if self.closed:
            return
        self.closed = True
        if not self.flags:
            return
        self.flushes += 1
        values = torch.stack(self.flags).reshape(-1)
        expected = torch.tensor(self.expected, dtype=torch.bool, device=self.device)
        comparison = values == expected
        okay = comparison.all()
        self.peak_bytes = backing_bytes((*self.flags, values, expected, comparison, okay))
        if not bool(okay):
            failures = comparison.cpu().tolist()
            first = next(k for k, passed in zip(self.kinds, failures, strict=True) if not passed)
            if first == "finite":
                raise ValueError("finite compatible BF16 construction pair required (commit)")
            if first == "protected":
                raise RuntimeError(
                    "protected K/V mismatch may not be hidden by a numerical patch (commit)"
                )
            raise AssertionError(
                "transport output not close under unchanged rtol=0.01, atol=2**-6 (commit)"
            )

    def report(self):
        return dict(
            enabled=True,
            counts=dict(self.counts),
            commit_flushes=self.flushes,
            validation_flag_peak_tensor_bytes=self.peak_bytes,
            no_source_or_reference_tensors_retained=True,
            all_checks_before_decode=self.closed,
            coalesced_correction_nonzero=self.coalesce_nonzero,
            correction_mask_calls=self.mask_calls,
            correction_mask_peak_tensor_bytes=self.mask_peak_bytes,
            correction_mask_hard_bound_bytes=4 * 1024 * 1024,
            correction_mask_bound_fallbacks=self.mask_bound_fallbacks,
            correction_mask_persistent_bytes=0,
        )


def enqueue(flag, kind, *, expected=True):
    state = _state.get()
    if state is None:
        return False
    state.append(flag, kind, expected)
    return True


def correction_mask(size, device):
    state = _state.get()
    return None if state is None else state.correction_mask(size, device)


@contextmanager
def use(enabled, *, coalesce_nonzero=False):
    if (
        type(enabled) is not bool
        or type(coalesce_nonzero) is not bool
        or (coalesce_nonzero and not enabled)
        or _state.get() is not None
    ):
        raise ValueError("explicit nonnested private validation transaction required")
    state = Validation(coalesce_nonzero=coalesce_nonzero) if enabled else None
    token = _state.set(state)
    try:
        yield state
        if state is not None:
            state.flush()
    finally:
        _state.reset(token)
        if state is not None:
            state.flags.clear()


def gpu_controls(*, coalesce_nonzero=False):
    """Real same-predicate/patch and failure controls, never task evidence."""
    from mosaickv.evaluation.shared_source_format import (
        assert_transport_close,
        transport_corrections,
    )

    cases = []
    generator = torch.Generator(device="cuda").manual_seed(20261103)
    for rows in (17, 65, 632):
        reference = torch.randn((1, 2, rows, 128), device="cuda", generator=generator).bfloat16()
        actual = reference.clone()
        actual.flatten()[::127] += 0.25
        protected = torch.empty(0, device="cuda", dtype=torch.int32)
        chunk = 128 if rows < 100 else 524288
        old = transport_corrections((reference,), (actual,), protected, chunk_scalars=chunk)[0]
        with use(True, coalesce_nonzero=coalesce_nonzero) as state:
            new = transport_corrections((reference,), (actual,), protected, chunk_scalars=chunk)[0]
            assert_transport_close(reference, reference.clone(), chunk_scalars=chunk)
        if not all(
            torch.equal(a.view(torch.uint8), b.view(torch.uint8))
            for a, b in zip(old.tensors, new.tensors, strict=True)
        ):
            raise RuntimeError("deferred GPU patch words differ")
        cases.append(dict(rows=rows, patch_words_identical=True, validation=state.report()))
    x = torch.ones((1, 1, 8, 128), dtype=torch.bfloat16, device="cuda")
    for kind, error in (
        ("finite", ValueError),
        ("protected", RuntimeError),
        ("close", AssertionError),
    ):
        y = x.clone()
        y.flatten()[0] = float("nan") if kind == "finite" else 2.0
        try:
            with use(True, coalesce_nonzero=coalesce_nonzero):
                if kind == "close":
                    assert_transport_close(y, x)
                else:
                    transport_corrections(
                        (x,), (y,), torch.tensor([0], device="cuda", dtype=torch.int32)
                    )
        except error:
            cases.append(dict(failed_predicate=kind, rejected_before_commit=True))
        else:
            raise RuntimeError("GPU negative validation control did not abort")
    return cases


def gpu_correction_timing():
    """Paired operator diagnostic, not a substitute for request-level timing."""
    import random
    from time import perf_counter

    from mosaickv.evaluation.shared_source_format import transport_corrections

    generator = torch.Generator(device="cuda").manual_seed(20261104)
    reference = torch.randn((1, 32, 632, 128), device="cuda", generator=generator).bfloat16()
    actual = reference.clone()
    actual.flatten()[::127] += 0.25
    protected = torch.empty(0, device="cuda", dtype=torch.int32)
    raw, rng = [], random.Random(20261104)
    frozen = None
    for trial in range(-5, 20):
        order = [False, True]
        rng.shuffle(order)
        for coalesced in order:
            torch.cuda.synchronize()
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start = perf_counter()
            begin.record()
            with use(True, coalesce_nonzero=coalesced) as state:
                patch = transport_corrections(
                    (reference,), (actual,), protected, chunk_scalars=524288
                )[0]
            end.record()
            torch.cuda.synchronize()
            elapsed = perf_counter() - start
            if frozen is None:
                frozen = tuple(t.clone() for t in patch.tensors)
            if not all(
                torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                for a, b in zip(frozen, patch.tensors, strict=True)
            ):
                raise RuntimeError("operator correction changed patch words")
            raw.append(
                dict(
                    trial=trial,
                    coalesced=coalesced,
                    wall_seconds=elapsed,
                    stream_milliseconds=begin.elapsed_time(end),
                    validation=state.report(),
                    patch_words_identical=True,
                )
            )
    return dict(
        rows=raw,
        shape=list(reference.shape),
        warmups=5,
        repetitions=20,
        operator_only_not_request_latency=True,
        independent_quality_N=0,
    )
