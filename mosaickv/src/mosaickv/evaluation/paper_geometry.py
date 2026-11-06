"""Explicit request-local geometry extension; legacy diagnostics stay frozen.

Only the registered standard COCO prompt geometry is added. Neither prompt
padding nor a changed chat template is used to enter the old two-input path.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

_length = ContextVar("evicastkv_paper_prompt_length", default=None)


def enabled():
    return _length.get() is not None


def current_length():
    return _length.get()


def frozen_rows_allowed(rows):
    return enabled() and _length.get() == 603 and tuple(sorted(rows)) == (91, 92, 128)


def inherit_worker_geometry(function):
    """Carry only the declared integer across a thread boundary, not KV owners.

    ContextVars are NOT inherited by ThreadPoolExecutor. Other constructor
    scopes deliberately remain governed by the original worker implementation.
    """

    @wraps(function)
    def call(owner, *args, **kwargs):
        length = owner.paper_prompt_length
        if length is None:
            return function(owner, *args, **kwargs)
        with registered_geometry(length):
            return function(owner, *args, **kwargs)

    return call


def source_shape_allowed(shape):
    return tuple(shape) in ((1, 32, 600, 128), (1, 32, 604, 128)) or (
        enabled() and tuple(shape) == (1, 32, _length.get(), 128)
    )


def odd_length_allowed(length):
    return enabled() and length == _length.get()


@contextmanager
def registered_geometry(length):
    if type(length) is not int or length != 603 or enabled():
        raise ValueError("only the preregistered, isolated 603-token COCO geometry is enabled")
    token = _length.set(length)
    try:
        yield
    finally:
        _length.reset(token)
