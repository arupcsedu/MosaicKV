"""Explicit request-local geometry extension; legacy diagnostics stay frozen.

Only the registered standard COCO prompt geometry is added. Neither prompt
padding nor a changed chat template is used to enter the old two-input path.
"""

from contextlib import contextmanager
from contextvars import ContextVar

_length = ContextVar("evicastkv_paper_prompt_length", default=None)


def enabled():
    return _length.get() is not None


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
