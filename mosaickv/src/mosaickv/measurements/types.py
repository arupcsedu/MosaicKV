"""Schemas for raw FullKV measurements and derived statistics."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, cast

from mosaickv.types import JsonObject


def _nonnegative(value: float | int | None, name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class GpuDeviceState:
    """One nvidia-smi device observation without inferred values."""

    index: str
    uuid: str
    name: str
    power_state: str
    graphics_clock_mhz: str
    sm_clock_mhz: str
    memory_clock_mhz: str
    power_draw_w: str
    power_limit_w: str
    driver_version: str
    visible_to_process: bool


@dataclass(frozen=True, slots=True)
class GpuProcessState:
    """One compute process reported by nvidia-smi."""

    gpu_uuid: str
    pid: int
    process_name: str
    used_gpu_memory_mib: str
    is_current_process: bool


@dataclass(frozen=True, slots=True)
class GpuEnvironmentSnapshot:
    """GPU clocks, power, software, and concurrent-process state."""

    captured_at_utc: str
    cuda_visible_devices: str
    torch_version: str
    cuda_runtime: str
    devices: tuple[GpuDeviceState, ...]
    processes: tuple[GpuProcessState, ...]
    background_processes: tuple[GpuProcessState, ...]
    visible_process_concurrency: int
    query_error: str | None = None


@dataclass(frozen=True, slots=True)
class PhaseTimings:
    """Synchronized CUDA timings for one generation trial, in seconds."""

    image_video_encoder: float | None
    projector: float | None
    language_model_prefill: float
    compression: float
    ttft: float
    per_token_decode: tuple[float, ...]
    repair: float
    total_latency: float
    host_total_latency: float

    def __post_init__(self) -> None:
        for name in (
            "image_video_encoder",
            "projector",
            "language_model_prefill",
            "compression",
            "ttft",
            "repair",
            "total_latency",
            "host_total_latency",
        ):
            _nonnegative(getattr(self, name), name)
        if any(not math.isfinite(value) or value < 0 for value in self.per_token_decode):
            raise ValueError("per_token_decode values must be finite and nonnegative")

    @property
    def decode_total(self) -> float:
        return sum(self.per_token_decode)


@dataclass(frozen=True, slots=True)
class MemoryMeasurements:
    """Peak allocator and explicit KV/residual payload bytes."""

    max_memory_allocated: int
    max_memory_reserved: int
    active_kv_bytes: int
    cpu_residual_bytes: int

    def __post_init__(self) -> None:
        for name in (
            "max_memory_allocated",
            "max_memory_reserved",
            "active_kv_bytes",
            "cpu_residual_bytes",
        ):
            _nonnegative(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class FullKVTrialMeasurement:
    """Immutable raw observation for one sample and one measured trial."""

    run_id: str
    sample_id: str
    trial_index: int
    model_id: str
    model_revision: str
    dataset_id: str
    dataset_revision: str
    manifest_path: str
    status: str
    error: str | None
    answer: str | None
    generated_token_ids: tuple[int, ...]
    timings: PhaseTimings | None
    memory: MemoryMeasurements | None
    active_cache_length: int | None
    logical_sequence_length: int | None
    synchronization_calls: int
    phase_event_counts: dict[str, int]
    gpu_before: GpuEnvironmentSnapshot
    gpu_after: GpuEnvironmentSnapshot
    method: str = "fullkv"
    backend: str = "huggingface"
    retention_ratio: float = 1.0
    measurement_type: str = "reference_measured"

    def __post_init__(self) -> None:
        for name in (
            "run_id",
            "sample_id",
            "model_id",
            "model_revision",
            "dataset_id",
            "dataset_revision",
            "manifest_path",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        if self.trial_index < 0:
            raise ValueError("trial_index must be nonnegative")
        if self.status not in {"completed", "failed"}:
            raise ValueError("status must be completed or failed")
        if self.status == "completed":
            if (
                self.error is not None
                or self.answer is None
                or self.timings is None
                or self.memory is None
            ):
                raise ValueError("completed trials require answer/timings/memory and no error")
        elif self.error is None or not self.error.strip():
            raise ValueError("failed trials require an error")
        if self.retention_ratio != 1.0:
            raise ValueError("FullKV retention_ratio must be exactly 1.0")
        if self.synchronization_calls < 0:
            raise ValueError("synchronization_calls must be nonnegative")

    def to_json_object(self) -> JsonObject:
        return cast("JsonObject", asdict(self))


@dataclass(frozen=True, slots=True)
class SummaryStatistics:
    """Distribution summary with a bootstrap confidence interval for the mean."""

    count: int
    median: float
    p5: float
    p95: float
    mean: float
    standard_deviation: float
    bootstrap_mean_ci_low: float
    bootstrap_mean_ci_high: float
    confidence_level: float
    bootstrap_samples: int


@dataclass(frozen=True, slots=True)
class FullKVAggregate:
    """Derived summaries that retain links to all underlying trial rows."""

    run_id: str
    warmups: int
    repeated_trials: int
    completed_trials: int
    failed_trials: int
    deterministic_token_match: bool | None
    trial_keys: tuple[str, ...]
    metrics: dict[str, SummaryStatistics]
    per_token_decode: tuple[SummaryStatistics, ...]
    schema_version: int = 1

    def to_json_object(self) -> JsonObject:
        return cast("JsonObject", asdict(self))


@dataclass(frozen=True, slots=True)
class FullKVTrialOutput:
    """Internal generation output paired with one raw measurement."""

    answer: str
    token_ids: Any
    measurement: FullKVTrialMeasurement


__all__ = [
    "FullKVAggregate",
    "FullKVTrialMeasurement",
    "FullKVTrialOutput",
    "GpuDeviceState",
    "GpuEnvironmentSnapshot",
    "GpuProcessState",
    "MemoryMeasurements",
    "PhaseTimings",
    "SummaryStatistics",
]
