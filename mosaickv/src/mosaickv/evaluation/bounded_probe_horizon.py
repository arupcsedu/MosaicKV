"""Exact engagement counts for a separately scoped, one-update probe.

The published storage path still requires all 127 updates by default. This
scope changes only expected operation counts, not operators or tolerances.
It is entered only in the separate probe thread and cannot nest.
"""

from contextlib import contextmanager
from contextvars import ContextVar

_updates = ContextVar("evicastkv_bounded_probe_updates", default=None)


def expected_updates():
    value = _updates.get()
    return 127 if value is None else value


@contextmanager
def two_prediction_probe():
    if _updates.get() is not None:
        raise RuntimeError("bounded model probe scopes cannot nest")
    token = _updates.set(1)
    try:
        yield
    finally:
        _updates.reset(token)
