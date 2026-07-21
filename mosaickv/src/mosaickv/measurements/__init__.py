"""CUDA timing, memory accounting, telemetry, and repeated-trial statistics."""

from mosaickv.measurements.memory import (
    active_kv_bytes,
    cache_tensors,
    cpu_residual_bytes,
    tensor_payload_bytes,
)
from mosaickv.measurements.statistics import aggregate_trials, percentile, summarize
from mosaickv.measurements.storage import write_aggregate_json, write_json_object, write_trial_jsonl
from mosaickv.measurements.telemetry import capture_gpu_environment
from mosaickv.measurements.timing import CudaEventTimer, ModuleCudaTimer, SynchronizationAudit
from mosaickv.measurements.types import (
    FullKVAggregate,
    FullKVTrialMeasurement,
    FullKVTrialOutput,
    GpuDeviceState,
    GpuEnvironmentSnapshot,
    GpuProcessState,
    MemoryMeasurements,
    PhaseTimings,
    SummaryStatistics,
)

__all__ = [
    "CudaEventTimer",
    "FullKVAggregate",
    "FullKVTrialMeasurement",
    "FullKVTrialOutput",
    "GpuDeviceState",
    "GpuEnvironmentSnapshot",
    "GpuProcessState",
    "MemoryMeasurements",
    "ModuleCudaTimer",
    "PhaseTimings",
    "SummaryStatistics",
    "SynchronizationAudit",
    "active_kv_bytes",
    "aggregate_trials",
    "cache_tensors",
    "capture_gpu_environment",
    "cpu_residual_bytes",
    "percentile",
    "summarize",
    "tensor_payload_bytes",
    "write_aggregate_json",
    "write_json_object",
    "write_trial_jsonl",
]
