"""Non-overlapping projection-role attribution on private sequential caches."""

from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from functools import wraps
from unittest.mock import patch

_layer = ContextVar("evicastkv_reconstruction_layer", default=None)
_role = ContextVar("evicastkv_reconstruction_role", default="history_construction")


def packet_role(packet, layer):
    if layer is None:
        return "history_construction"
    if packet is layer.representation:
        return "prompt_source"
    if any(packet is p for p in layer.source_sealed):
        return "generated_history"
    raise ValueError("unowned packet cannot receive a reconstruction role")


@contextmanager
def instrument(timer):
    from mosaickv.evaluation import encoded_projection_kernel as encoded
    from mosaickv.evaluation import sequential_source_history as history
    from mosaickv.evaluation import shared_source_kernel as source

    original_method = history.SequentialSourceHistoryLayer.materialize_component
    original_fill = history.fill_component

    @wraps(original_method)
    def materialize(self, *args, **kwargs):
        token = _layer.set(self)
        try:
            return original_method(self, *args, **kwargs)
        finally:
            _layer.reset(token)

    @wraps(original_fill)
    def fill(packet, *args, **kwargs):
        role = packet_role(packet, _layer.get())
        token = _role.set(role)
        try:
            with timer.phase(role + ".reconstruction"):
                return original_fill(packet, *args, **kwargs)
        finally:
            _role.reset(token)

    def projected(function, kind):
        @wraps(function)
        def wrapper(*args, **kwargs):
            role = _role.get()
            if kind == "encoded" and role == "prompt_source":
                role = "text_overlay"
            with timer.phase(role + ".projection"):
                return function(*args, **kwargs)

        return wrapper

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(history.SequentialSourceHistoryLayer, "materialize_component", materialize)
        )
        stack.enter_context(patch.object(history, "fill_component", fill))
        stack.enter_context(patch.object(source, "project", projected(source.project, "source")))
        stack.enter_context(patch.object(encoded, "project", projected(encoded.project, "encoded")))
        yield


def choose_order(summaries):
    """Use only consumed-input aggregate operation timings, never answers."""
    groups = [s for s in summaries if s.get("method") == "mosaickv_exact"]
    if len(groups) != 2 or {s["input_index"] for s in groups} != {0, 1}:
        raise ValueError("both consumed source-role summaries required")
    costs = {}
    for role in ("prompt_source", "text_overlay", "generated_history"):
        values = [
            s["phases"].get(role + ".projection", {}).get("mean_inclusive_cuda_stream_ms")
            for s in groups
        ]
        if any(type(x) not in (int, float) or not 0 < x < float("inf") for x in values):
            raise ValueError("complete finite role projection timings required")
        costs[role] = sum(values) / len(values)
    primary = (
        "reuse_headers"
        if costs["prompt_source"] >= max(costs["text_overlay"], costs["generated_history"])
        else "carry_literals"
    )
    return [primary, "carry_literals" if primary == "reuse_headers" else "reuse_headers"], costs
