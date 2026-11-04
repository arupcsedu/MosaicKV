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
    def __init__(self):
        self.flags, self.kinds, self.expected = [], [], []
        self.device, self.closed, self.flushes = None, False, 0
        self.counts, self.peak_bytes = {}, 0

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
        )


def enqueue(flag, kind, *, expected=True):
    state = _state.get()
    if state is None:
        return False
    state.append(flag, kind, expected)
    return True


@contextmanager
def use(enabled):
    if type(enabled) is not bool or _state.get() is not None:
        raise ValueError("explicit nonnested private validation transaction required")
    state = Validation() if enabled else None
    token = _state.set(state)
    try:
        yield state
        if state is not None:
            state.flush()
    finally:
        _state.reset(token)
        if state is not None:
            state.flags.clear()


def gpu_controls():
    """Real same-predicate/patch and failure controls, never task evidence."""
    from mosaickv.evaluation.shared_source_format import (
        assert_transport_close,
        transport_corrections,
    )

    cases = []
    generator = torch.Generator(device="cuda").manual_seed(20261103)
    for rows in (17, 65):
        reference = torch.randn((1, 2, rows, 128), device="cuda", generator=generator).bfloat16()
        actual = reference.clone()
        actual.flatten()[::127] += 0.25
        protected = torch.empty(0, device="cuda", dtype=torch.int32)
        old = transport_corrections((reference,), (actual,), protected, chunk_scalars=128)[0]
        with use(True) as state:
            new = transport_corrections((reference,), (actual,), protected, chunk_scalars=128)[0]
            assert_transport_close(reference, reference.clone(), chunk_scalars=128)
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
            with use(True):
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
