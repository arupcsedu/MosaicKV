"""Consumed-input exact controls and bounded helper timing, not request latency."""

import random
from contextlib import ExitStack, contextmanager
from functools import wraps
from unittest.mock import patch

from mosaickv.evaluation.reconstruction_policy import use


@contextmanager
def resources(policy):
    from mosaickv.evaluation import encoded_projection_kernel as encoded
    from mosaickv.evaluation import shared_source_kernel as source

    records = {}

    def wrapped(function, kind):
        @wraps(function)
        def call(*args, **kwargs):
            component = kwargs.get("component", -1)
            if kind == "source":
                p, attention = args[:2]
                signature = (
                    kind,
                    p.shape,
                    component,
                    str(p.source.minimum.dtype),
                    p.source_head_group,
                    p.source_row_tile,
                )
            else:
                body, _, attention, archive = args[:4]
                signature = (
                    kind,
                    body.shape,
                    tuple(archive.selected.shape),
                    component,
                    body.raw,
                    kwargs.get("packed_counts", False),
                )
            key = str((signature, attention.k_proj.weight.data_ptr()))
            inspect = dict(kwargs, inspect_kernel=True)
            if key not in records:
                with use("original"):
                    old = function(*args, **inspect)
                records[key] = dict(signature=str(signature), baseline=old, candidate=None)
            actual = function(*args, **inspect)
            records[key]["candidate"] = actual
            return actual if kwargs.get("inspect_kernel", False) else None

        return call

    with ExitStack() as stack:
        stack.enter_context(patch.object(source, "project", wrapped(source.project, "source")))
        stack.enter_context(patch.object(encoded, "project", wrapped(encoded.project, "encoded")))
        with use(policy):
            yield records


def spills_pass(records):
    return bool(records) and all(
        type(r.get("baseline", {}).get("spills")) is int
        and type((r.get("candidate") or {}).get("spills")) is int
        and r["candidate"]["spills"] <= r["baseline"]["spills"]
        for r in records.values()
    )


def helper_trials(cache, policy, seed):
    import torch

    from mosaickv.evaluation.sequential_source_history import SequentialSourceHistoryLayer

    if len(cache.layers) != 32 or any(
        not isinstance(layer, SequentialSourceHistoryLayer) or layer.leased
        for layer in cache.layers
    ):
        raise ValueError("released complete private source-history cache required")
    shape = (1, 32, max(layer.get_seq_length() for layer in cache.layers), 128)
    buffers = tuple(torch.empty(shape, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    # Compare BOTH components at every layer, including sealed history and tail.
    for layer in cache.layers:
        for component in (0, 1):
            for label, buffer in zip(("original", policy), buffers, strict=True):
                layer.leased = (buffer[..., : layer.get_seq_length(), :],) * 2
                try:
                    with use(label):
                        layer.materialize_component(component)
                finally:
                    layer.leased = ()
            n = layer.get_seq_length()
            if not torch.equal(
                buffers[0][..., :n, :].view(torch.int16), buffers[1][..., :n, :].view(torch.int16)
            ):
                raise RuntimeError("reader changed a complete live K/V reconstruction bit")
    rng = random.Random(seed)
    rows = []
    for trial in range(-5, 20):
        labels = ["original", policy]
        rng.shuffle(labels)
        for label in labels:
            begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            torch.cuda.synchronize()
            begin.record()
            with use(label):
                for layer in cache.layers:
                    layer.leased = (buffers[0][..., : layer.get_seq_length(), :],) * 2
                    try:
                        layer.materialize_component(0)
                        layer.materialize_component(1)
                    finally:
                        layer.leased = ()
            end.record()
            torch.cuda.synchronize()
            rows.append(
                dict(
                    trial=trial,
                    warmup=trial < 0,
                    policy=label,
                    cuda_stream_ms=begin.elapsed_time(end),
                )
            )
    return rows
