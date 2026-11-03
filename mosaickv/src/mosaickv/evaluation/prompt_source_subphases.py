"""Scoped operation attribution, never uninstrumented latency evidence.

The disabled path allocates neither tensors nor timing records. All arithmetic,
kernel arguments, checks, correction order and storage ownership remain in the
original callers. Event intervals include host gaps, not just GPU-active time.
"""

from contextlib import ExitStack, contextmanager, nullcontext
from contextvars import ContextVar
from functools import wraps
from math import isfinite

_timer = ContextVar("prompt_source_subphase_timer", default=None)
_off = nullcontext()
NAMES = frozenset(
    (
        "validation",
        "indexing_views",
        "projection_dispatch",
        "protected_restore_dispatch",
        "correction_dispatch",
        "temporary_buffers",
    )
)


def phase(name):
    timer = _timer.get()
    if timer is None:
        return _off
    if name not in NAMES:
        raise ValueError("registered prompt-source subphase required")
    from mosaickv.evaluation.reconstruction_roles import _role

    if _role.get() != "prompt_source":
        return _off
    return timer.phase("prompt_detail." + name)


@contextmanager
def use(timer):
    if _timer.get() is not None or not callable(getattr(timer, "phase", None)):
        raise ValueError("one private nonnested prompt-source diagnostic required")
    token = _timer.set(timer)
    try:
        from unittest.mock import patch

        import torch

        def allocation(function):
            @wraps(function)
            def wrapped(*args, **kwargs):
                with phase("temporary_buffers"):
                    return function(*args, **kwargs)

            return wrapped

        with ExitStack() as stack:
            # Explicit Python allocation entrypoints, not a general ATen or
            # allocator profiler. Do not infer GPU peak from a zero count.
            for owner, names in (
                (
                    torch,
                    (
                        "empty",
                        "empty_like",
                        "zeros",
                        "zeros_like",
                        "ones",
                        "ones_like",
                        "full",
                        "full_like",
                    ),
                ),
                (torch.Tensor, ("new_empty", "new_zeros", "new_ones", "new_full")),
            ):
                for name in names:
                    stack.enter_context(patch.object(owner, name, allocation(getattr(owner, name))))
            yield
    finally:
        _timer.reset(token)


def totals(phases, *, require_source=True):
    """Exclusive leaves only; do not add parent reconstruction intervals."""
    result = {
        name: dict(calls=0, exclusive_wall_seconds=0.0, stream_elapsed_ms=0.0)
        for name in sorted(NAMES)
    }
    by_window = {w: {name: 0 for name in sorted(NAMES)} for w in ("early", "middle", "late")}
    for key, value in phases.items():
        window, _, suffix = key.partition("/")
        if window not in by_window or not suffix.startswith("prompt_detail."):
            continue
        name = suffix.removeprefix("prompt_detail.")
        if name not in result:
            raise ValueError("unknown subphase in producer")
        if (
            type(value["calls"]) is not int
            or value["calls"] <= 0
            or any(
                type(value[key]) not in (int, float) or not isfinite(value[key]) or value[key] < 0
                for key in ("exclusive_wall_seconds", "stream_elapsed_ms")
            )
        ):
            raise ValueError("finite nonnegative operation measurements required")
        target = result[name]
        target["calls"] += value["calls"]
        target["exclusive_wall_seconds"] += value["exclusive_wall_seconds"]
        target["stream_elapsed_ms"] += value["stream_elapsed_ms"]
        by_window[window][name] += value["calls"]
    required = NAMES - {"temporary_buffers"}
    if require_source and any(by_window[w][name] <= 0 for w in by_window for name in required):
        raise ValueError("every actual decode window needs complete prompt subphases")
    return dict(
        operations=result,
        window_calls=by_window,
        # Explicit empty-like calls only, not a general allocator audit.
        tracked_temporary_allocation_calls=result["temporary_buffers"]["calls"],
        allocation_inventory_is_not_peak_evidence=True,
        instrumentation_not_latency_evidence=True,
        stream_intervals_include_host_gaps=True,
    )
