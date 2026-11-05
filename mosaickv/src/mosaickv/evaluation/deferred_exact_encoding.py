"""Default-off lossless constructor scheduling; no new cache representation.

Intermediate raw bodies own their storage. Surviving bodies are encoded at a
retirement or byte-decision barrier. Weak references never extend GPU lifetime.
"""

import weakref
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import fields, is_dataclass, replace

import torch

from mosaickv.evaluation.lossless_bf16 import PackedBF16

MAX_WORDS = 4 * 2**20  # At most 8 MiB per temporary body; never a full KV alias.
_state = ContextVar("evicastkv_deferred_exact_encoding", default=None)


class DeferredBF16(PackedBF16):
    """Constructor-only marker. It must never reach the decode interface."""


@contextmanager
def use(active):
    if type(active) is not bool or _state.get() is not None:
        raise ValueError("explicit isolated nonnested deferred-encoding scope required")
    state = dict(created=0, encoded=0, barriers=0, cache={}) if active else None
    token = _state.set(state)
    try:
        yield state
    finally:
        _state.reset(token)


def pending(tensor):
    if _state.get() is None or tensor.numel() > MAX_WORDS:
        return None
    if tensor.dtype != torch.bfloat16 or tensor.numel() == 0:
        raise ValueError("nonempty BF16 intermediate required")
    # Clone: a view would retain an original prefill K/V owner after retirement.
    body = DeferredBF16(tuple(tensor.shape), True, (tensor.contiguous().clone(),))
    _state.get()["created"] += 1
    return body


def finalize(value):
    state = _state.get()
    if state is None:
        return value
    state["barriers"] += 1
    memo = {}

    def visit(item):
        if id(item) in memo:
            return memo[id(item)]
        if isinstance(item, DeferredBF16):
            saved = state["cache"].get(id(item))
            encoded = saved[1]() if saved and saved[0]() is item else None
            if encoded is None:
                from mosaickv.evaluation.exact_storage import encode

                token = _state.set(None)
                try:
                    encoded = encode(item.buffers[0])
                finally:
                    _state.reset(token)
                state["encoded"] += 1
                state["cache"][id(item)] = (weakref.ref(item), weakref.ref(encoded))
            memo[id(item)] = encoded
            return encoded
        if isinstance(item, torch.Tensor):
            return item
        if is_dataclass(item) and not isinstance(item, type):
            changes = {}
            for field in fields(item):
                old = getattr(item, field.name)
                new = visit(old)
                if new is not old:
                    changes[field.name] = new
            result = replace(item, **changes) if changes else item
        elif isinstance(item, (tuple, list)):
            values = [visit(x) for x in item]
            result = (
                item
                if all(a is b for a, b in zip(item, values, strict=True))
                else type(item)(values)
            )
        elif isinstance(item, dict):
            values = {key: visit(x) for key, x in item.items()}
            result = item if all(values[key] is x for key, x in item.items()) else values
        else:
            result = item
        memo[id(item)] = result
        return result

    return visit(value)


def report(state):
    return dict(
        enabled=state is not None,
        maximum_intermediate_body_bytes=MAX_WORDS * 2,
        created=0 if state is None else state["created"],
        encoded_at_barriers=0 if state is None else state["encoded"],
        barriers=0 if state is None else state["barriers"],
        no_values_offloaded=True,
        no_changes_to_final_codec=True,
        whole_request_peak_certified=False,
    )
