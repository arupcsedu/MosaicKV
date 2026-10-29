"""Opt-in, request-local instrumentation; never input to selection.

Synchronized profiling contaminates latency and is diagnostic-only. Nested
spans expose inclusive and exclusive wall time; do not add both. No tensors,
benchmark references, or model outputs are retained by this collector.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


@dataclass
class ExecutionAudit:
    torch_module: Any = None
    phases: list[dict[str, Any]] = field(default_factory=list)
    routing: list[dict[str, Any]] = field(default_factory=list)
    exact_representations: list[dict[str, Any]] = field(default_factory=list)
    _stack: list[int] = field(default_factory=list, repr=False)

    def synchronize(self) -> None:
        if self.torch_module is not None:
            self.torch_module.cuda.synchronize()

    def payload(self) -> dict[str, Any]:
        if self._stack:
            raise RuntimeError("cannot publish an unfinished execution audit")
        return {
            "schema_version": 1,
            "measurement_type": "diagnostic_profile",
            "profiler_contaminated": True,
            "cuda_synchronized_boundaries": self.torch_module is not None,
            "nested_wall_times_must_not_be_double_counted": True,
            "selection_uses_audit_records": False,
            "phases": self.phases,
            "routing": self.routing,
            "exact_representations": self.exact_representations,
        }


_ACTIVE: ContextVar[ExecutionAudit | None] = ContextVar("evicastkv_execution_audit", default=None)


def record_exact_representation(exact: Any) -> None:
    """Observe existing counts only; never enumerate/materialize deferred views."""
    audit = _ACTIVE.get()
    if audit is not None:
        audit.exact_representations.append(
            {
                "sequence_type": type(exact.key_blocks).__name__,
                "selected_blocks": len(exact.blocks),
                "materialized_block_views": exact.materialized_block_view_count,
                "selected_payload_bytes": exact.active_bytes,
                "payload_bytes_are_not_parent_backing_storage": True,
            }
        )


@contextmanager
def collect_execution_audit(*, torch_module: Any = None) -> Iterator[ExecutionAudit]:
    """Collect one isolated request; reset reliably after success or failure."""
    audit = ExecutionAudit(torch_module=torch_module)
    token = _ACTIVE.set(audit)
    try:
        yield audit
    finally:
        _ACTIVE.reset(token)


@contextmanager
def audit_phase(name: str) -> Iterator[None]:
    audit = _ACTIVE.get()
    if audit is None:
        yield
        return
    audit.synchronize()
    start_event = end_event = None
    if audit.torch_module is not None:
        start_event = audit.torch_module.cuda.Event(enable_timing=True)
        end_event = audit.torch_module.cuda.Event(enable_timing=True)
        start_event.record()
    parent = audit._stack[-1] if audit._stack else None
    index = len(audit.phases)
    row: dict[str, Any] = {
        "index": index,
        "parent": parent,
        "depth": len(audit._stack),
        "operation": name,
        "status": "running",
    }
    audit.phases.append(row)
    audit._stack.append(index)
    started = time.perf_counter()
    try:
        yield
        row["status"] = "completed"
    except BaseException as exc:
        row.update(status="failed", failure_type=type(exc).__name__)
        raise
    finally:
        if end_event is not None:
            end_event.record()
        audit.synchronize()
        row["inclusive_wall_seconds"] = time.perf_counter() - started
        child_time = sum(r["inclusive_wall_seconds"] for r in audit.phases if r["parent"] == index)
        row["exclusive_wall_seconds"] = max(0.0, row["inclusive_wall_seconds"] - child_time)
        row["cuda_event_elapsed_seconds"] = (
            float(start_event.elapsed_time(end_event)) / 1000 if start_event is not None else None
        )
        row["cuda_event_is_not_a_kernel_only_profile"] = True
        if audit._stack.pop() != index:
            raise RuntimeError("execution-audit phase nesting was corrupted")


def audit_operation(name: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        @wraps(function)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            if _ACTIVE.get() is None:
                return function(*args, **kwargs)
            with audit_phase(name):
                return function(*args, **kwargs)

        return wrapped

    return decorate


def record_routing_scores(
    ordered: Sequence[tuple[int, int]],
    scores: Mapping[tuple[int, int], float],
    selected: Sequence[tuple[int, int]],
    *,
    fraction: float,
    threshold: float,
    label: str,
) -> None:
    audit = _ACTIVE.get()
    if audit is None:
        return
    audit.routing.append(
        {
            "label": label,
            "fraction": fraction,
            "threshold": threshold,
            "head_count": len(ordered),
            "ordering": "descending_score_then_layer_then_kv_head",
            "selected": [list(identity) for identity in selected],
            "all_scores": [
                {
                    "layer": identity[0],
                    "kv_head": identity[1],
                    "score": float(scores[identity]),
                    "rank": rank,
                    "selected": identity in selected,
                }
                for rank, identity in enumerate(ordered, start=1)
            ],
        }
    )
