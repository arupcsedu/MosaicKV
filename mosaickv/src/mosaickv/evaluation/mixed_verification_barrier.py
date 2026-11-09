"""Default-off epoch commit of unchanged bitwise mixed-history checks.

All three K/V verification passes still execute. Retain only their Boolean
scalars and synchronize once before consolidation returns/decoding resumes.
No tolerance, source, projection or numerical patch changes are permitted.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from unittest.mock import patch

import torch

_epoch = ContextVar("evicastkv_mixed_verification_epoch", default=None)
_policy = ContextVar("evicastkv_mixed_verification_policy", default=None)


def flush(cache, flags):
    if not flags:
        return
    # Each original verify contributes three complete K/V checks, not a
    # cheaper substitute or spot check. All fail closed before next decoding.
    if (
        len(flags) > 192
        or len(flags) % 6
        or any(t.dtype != torch.bool or t.numel() != 1 for t in flags)
    ):
        raise RuntimeError("incomplete bounded original mixed-history predicates")
    values = torch.stack(flags)
    passed = values.all()
    cache.observe((*flags, values, passed))
    if not bool(passed):
        raise RuntimeError("original mixed-history bitwise verification failed before commit")


@contextmanager
def transaction(cache):
    from mosaickv.evaluation.native_history_reader import current

    report = _policy.get()
    state = current()
    if report is None or _epoch.get() is not None:
        raise RuntimeError("active nonnested mixed verification policy required")
    if (
        state is None
        or state.closed
        or state.pool.cache is not cache
        or not 1 <= len(cache.layers) <= 32
        or any(layer.leased or layer.source_pending for layer in cache.layers)
    ):
        raise RuntimeError("private released cache epoch required before verification deferral")
    flags = []
    token = _epoch.set((cache, flags))
    try:
        yield
        flush(cache, flags)
        if flags:
            report["commit_flushes"] += 1
            report["complete_KV_verification_flags"] += len(flags)
            report["maximum_epoch_flags"] = max(report["maximum_epoch_flags"], len(flags))
    finally:
        had_flags = bool(flags)
        flags.clear()
        _epoch.reset(token)
        if had_flags:
            cache.observe(())


@contextmanager
def use():
    from mosaickv.evaluation import consolidated_history as combined
    from mosaickv.evaluation import history_verification_workspace as verification
    from mosaickv.evaluation.mixed_history_projection import MixedHistory

    if _policy.get() is not None:
        raise RuntimeError("nonnested mixed verification policy required")
    original_consolidate, original_verify = combined.consolidate, verification.verify
    report = dict(
        variant="same_predicate_mixed_epoch_commit",
        commit_flushes=0,
        complete_KV_verification_flags=0,
        maximum_epoch_flags=0,
        synchronous_delegations=0,
        all_original_three_passes_preserved=True,
        no_reference_KV_retained=True,
        tolerance_unchanged=True,
    )
    policy_token = _policy.set(report)

    def verify(layer, candidate, formats, shape, bit_flags, scratch=None):
        epoch = _epoch.get()
        if epoch is None or bit_flags is not None or not isinstance(candidate, MixedHistory):
            return original_verify(layer, candidate, formats, shape, bit_flags, scratch)
        cache, flags = epoch
        before = len(flags)
        result = original_verify(layer, candidate, formats, shape, flags, scratch)
        if len(flags) - before != 6 or len(flags) > 192:
            raise RuntimeError("all original three K/V checks required in this epoch")
        cache.observe(tuple(flags))
        return result

    def consolidate(cache):
        from mosaickv.evaluation.native_history_reader import current

        state = current()
        if (
            state is None
            or state.closed
            or state.pool is None
            or state.pool.cache is not cache
            or not 1 <= len(cache.layers) <= 32
            or any(layer.leased or layer.source_pending for layer in cache.layers)
        ):
            # The unchanged constructor still performs synchronous checks.
            # Never turn an ineligible no-op into a failed model request.
            report["synchronous_delegations"] += 1
            return original_consolidate(cache)
        with transaction(cache):
            return original_consolidate(cache)

    try:
        with (
            patch.object(verification, "verify", verify),
            patch.object(combined, "consolidate", consolidate),
        ):
            yield report
    finally:
        _policy.reset(policy_token)


def gpu_controls():
    from types import SimpleNamespace
    from unittest.mock import patch

    from mosaickv.evaluation import consolidated_history as combined
    from mosaickv.evaluation import history_verification_workspace as verification
    from mosaickv.evaluation import native_history_reader as native
    from mosaickv.evaluation.mixed_history_projection import MixedHistory

    shape = (1, 32, 2, 128)
    pair = tuple(torch.zeros(shape, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    scratch = (tuple(t.clone() for t in pair), tuple(t.clone() for t in pair))
    cache = SimpleNamespace(
        layers=[SimpleNamespace(leased=(), source_pending=False)], observe=lambda _: None
    )
    state = SimpleNamespace(closed=False, pool=SimpleNamespace(cache=cache))
    phase = torch.ones(1, 2, 128, device="cuda", dtype=torch.bfloat16)
    candidate = MixedHistory(None, (), phase, phase, (), None, shape, ())
    rows = []
    for bad in (False, True):
        counts = dict(fills=0)

        def fill(_packet, _attention, targets, *, direct, counts=counts, bad=bad):
            counts["fills"] += 1
            for t in targets:
                t.zero_()
            if bad:
                targets[0].flatten()[0] = -0.0  # equal float, different original BF16 bit

        formats = (
            SimpleNamespace(
                shape=shape, fill=lambda _a, targets, *, direct: [t.zero_() for t in targets]
            ),
        )

        def consolidate(_cache, formats=formats):
            verification.verify(
                SimpleNamespace(attention=None), candidate, formats, shape, None, scratch
            )
            return True

        with (
            patch.object(native, "current", lambda: state),
            patch.object(combined, "consolidate", consolidate),
            patch.object(MixedHistory, "fill", fill),
        ):
            try:
                with use() as report:
                    combined.consolidate(cache)
            except RuntimeError as exc:
                if not bad or "bitwise verification failed" not in str(exc):
                    raise
            else:
                if bad:
                    raise RuntimeError("changed original signed-zero word was accepted")
        if counts["fills"] != 3:
            raise RuntimeError("an original verification pass was skipped")
        rows.append(
            dict(
                signed_zero_corruption=bad,
                all_three_passes_executed=True,
                rejected_before_commit=bad,
                report=report,
            )
        )
    return rows
