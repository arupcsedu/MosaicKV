"""Nested construction diagnostics, never uninstrumented performance evidence.

No tensor outputs are retained. CUDA events measure current-stream elapsed time
(including host launch gaps), not isolated kernel-active time. Only one final
synchronization resolves all events. CPU exclusive time subtracts child scopes.
"""

import time
from contextlib import ExitStack, contextmanager
from functools import wraps
from unittest.mock import patch


class PhaseTimer:
    def __init__(self, *, clock=None, event_factory=None):
        self.clock = time.monotonic if clock is None else clock
        self.event_factory = event_factory
        self.stack, self.records = [], []

    @contextmanager
    def phase(self, name):
        if not isinstance(name, str) or not name:
            raise ValueError("named phase required")
        start = None if self.event_factory is None else self.event_factory()
        end = None if self.event_factory is None else self.event_factory()
        if start is not None:
            start.record()
        frame = dict(name=name, begin=self.clock(), child_wall=0.0, depth=len(self.stack))
        self.stack.append(frame)
        try:
            yield
        finally:
            if end is not None:
                end.record()
            wall = self.clock() - frame["begin"]
            self.stack.pop()
            if self.stack:
                self.stack[-1]["child_wall"] += wall
            self.records.append(
                dict(
                    phase=name,
                    depth=frame["depth"],
                    wall_seconds=wall,
                    exclusive_wall_seconds=max(0.0, wall - frame["child_wall"]),
                    events=None if start is None else (start, end),
                )
            )

    def wrap(self, name, function):
        @wraps(function)
        def measured(*args, **kwargs):
            with self.phase(name):
                return function(*args, **kwargs)

        return measured

    def resolve(self):
        if self.stack:
            raise RuntimeError("all phases must close before event resolution")
        return [
            dict(
                **{k: v for k, v in row.items() if k != "events"},
                stream_elapsed_ms=None
                if row["events"] is None
                else row["events"][0].elapsed_time(row["events"][1]),
            )
            for row in self.records
        ]


@contextmanager
def instrument(timer):
    from mosaickv.evaluation import (
        conditional_precision,
        exact_minimum_metadata,
        exact_storage,
        head_local_precision,
        layer_code_delta,
        request_precision,
        sequential_kv_lease,
        shared_source_format,
        shared_source_kernel,
        source_retile,
        text_source_storage,
    )

    targets = (
        (shared_source_format, "construct"),
        (shared_source_format, "fit_transport_components"),
        (shared_source_format, "transport_corrections"),
        (shared_source_format, "assert_transport_close"),
        (shared_source_format, "pack_unprotected"),
        (shared_source_format, "encode"),
        (shared_source_format, "decoded"),
        (shared_source_format.SharedSourceFormat, "materialize"),
        (shared_source_kernel, "decode_source_for_construction"),
        (sequential_kv_lease, "fill_component"),
        (conditional_precision, "pack"),
        (text_source_storage, "optimize"),
        (text_source_storage, "strip_text_overlays"),
        (text_source_storage, "encode"),
        (text_source_storage.TextSourceFormat, "materialize"),
        (text_source_storage.TextProjection, "fill"),
        (head_local_precision, "encode"),
        (exact_storage, "encode"),
        (exact_storage, "decoded"),
        (request_precision, "ranked_positions"),
        (source_retile, "retile"),
        (exact_minimum_metadata, "compact_format"),
        (layer_code_delta, "convert"),
    )
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                text_source_storage,
                "_construction_phase",
                lambda name: timer.phase("text_source." + name),
            )
        )
        for owner, key in targets:
            prefix = getattr(owner, "__name__", type(owner).__name__)
            stack.enter_context(
                patch.object(owner, key, timer.wrap(prefix + "." + key, getattr(owner, key)))
            )
        yield


def phase_totals(trace):
    totals = {}
    for row in trace:
        entry = totals.setdefault(
            row["phase"],
            dict(
                calls=0,
                inclusive_wall_seconds=0.0,
                exclusive_wall_seconds=0.0,
                stream_elapsed_ms=0.0,
            ),
        )
        entry["calls"] += 1
        entry["inclusive_wall_seconds"] += row["wall_seconds"]
        entry["exclusive_wall_seconds"] += row["exclusive_wall_seconds"]
        entry["stream_elapsed_ms"] += row["stream_elapsed_ms"] or 0.0
    return totals
