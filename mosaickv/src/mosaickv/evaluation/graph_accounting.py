"""Private accounting execution policies, never a different byte boundary."""

from contextlib import contextmanager
from contextvars import ContextVar

POLICIES = ("original", "single_pass", "event_sum")
_policy = ContextVar("evicastkv_graph_accounting", default=None)


def validate(value):
    if type(value) is not str or value not in POLICIES:
        raise ValueError("registered original/single_pass/event_sum graph accounting required")


@contextmanager
def use(value):
    validate(value)
    if _policy.get() is not None:
        raise RuntimeError("private nonnested graph accounting scope required")
    token = _policy.set(value)
    try:
        yield
    finally:
        _policy.reset(token)


def current():
    return _policy.get() or "original"


def layer_class():
    from mosaickv.evaluation.prompt_graph_reader import PromptGraphLayer, TrackedPromptGraphLayer

    return TrackedPromptGraphLayer if current() == "event_sum" else PromptGraphLayer
