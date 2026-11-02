"""Request-local integer-reader reuse; no new tensor or cache representation."""

from contextlib import contextmanager
from contextvars import ContextVar

POLICIES = ("original", "reuse_headers", "carry_literals", "reuse_both")
_policy = ContextVar("evicastkv_reconstruction_reader_policy", default="original")


def validate(policy):
    if type(policy) is not str or policy not in POLICIES:
        raise ValueError("explicit audited reconstruction-reader policy required")
    return policy


def current():
    return _policy.get()


def reuse_headers():
    return current() in ("reuse_headers", "reuse_both")


def carry_literals():
    return current() in ("carry_literals", "reuse_both")


@contextmanager
def use(policy):
    token = _policy.set(validate(policy))
    try:
        yield
    finally:
        _policy.reset(token)
