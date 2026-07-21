"""CUDA-event timers with synchronization at every timing boundary."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Self


@dataclass(slots=True)
class SynchronizationAudit:
    """Count explicit synchronizations made by a measurement trial."""

    calls: int = 0


class CudaEventTimer:
    """One interval measured by torch.cuda.Event on a selected device."""

    def __init__(self, torch_module: Any, device: Any, audit: SynchronizationAudit) -> None:
        self._torch = torch_module
        self._device = device
        self._audit = audit
        self._start_event: Any | None = None
        self._end_event: Any | None = None
        self._elapsed_seconds: float | None = None

    def _synchronize(self) -> None:
        self._torch.cuda.synchronize(self._device)
        self._audit.calls += 1

    def start(self) -> None:
        if self._start_event is not None:
            raise RuntimeError("CUDA timer has already started")
        self._synchronize()
        self._start_event = self._torch.cuda.Event(enable_timing=True)
        self._end_event = self._torch.cuda.Event(enable_timing=True)
        self._start_event.record()

    def stop(self) -> float:
        if self._start_event is None or self._end_event is None:
            raise RuntimeError("CUDA timer was not started")
        if self._elapsed_seconds is not None:
            raise RuntimeError("CUDA timer has already stopped")
        self._end_event.record()
        self._synchronize()
        milliseconds = float(self._start_event.elapsed_time(self._end_event))
        if milliseconds < 0:
            raise RuntimeError("CUDA event returned a negative elapsed time")
        self._elapsed_seconds = milliseconds / 1000.0
        return self._elapsed_seconds


class ModuleCudaTimer(AbstractContextManager["ModuleCudaTimer"]):
    """Accumulate synchronized CUDA intervals for repeated module invocations."""

    def __init__(
        self,
        module: Any | None,
        torch_module: Any,
        device: Any,
        audit: SynchronizationAudit,
    ) -> None:
        self._module = module
        self._torch = torch_module
        self._device = device
        self._audit = audit
        self._handles: list[Any] = []
        self._active: list[CudaEventTimer] = []
        self._intervals: list[float] = []

    def __enter__(self) -> Self:
        if self._module is None:
            return self

        def before(_module: Any, _inputs: tuple[Any, ...]) -> None:
            timer = CudaEventTimer(self._torch, self._device, self._audit)
            timer.start()
            self._active.append(timer)

        def after(_module: Any, _inputs: tuple[Any, ...], _output: Any) -> None:
            if not self._active:
                raise RuntimeError("module timing hook ended without a start boundary")
            self._intervals.append(self._active.pop().stop())

        self._handles.append(self._module.register_forward_pre_hook(before))
        self._handles.append(self._module.register_forward_hook(after))
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        if exc_type is None and self._active:
            raise RuntimeError("module timing hook has an unmatched start boundary")

    @property
    def invocation_count(self) -> int:
        return len(self._intervals)

    @property
    def total_seconds(self) -> float | None:
        return sum(self._intervals) if self._intervals else None


__all__ = ["CudaEventTimer", "ModuleCudaTimer", "SynchronizationAudit"]
