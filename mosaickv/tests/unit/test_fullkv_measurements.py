from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mosaickv.evaluation.model import EvaluationRequest
from mosaickv.fullkv import FullKV, FullKVBenchmarkConfig, FullKVEvaluationModel
from mosaickv.measurements.memory import (
    active_kv_bytes,
    cpu_residual_bytes,
    tensor_payload_bytes,
)
from mosaickv.measurements.statistics import aggregate_trials, percentile, summarize
from mosaickv.measurements.storage import (
    write_aggregate_json,
    write_json_object,
    write_trial_jsonl,
)
from mosaickv.measurements.telemetry import capture_gpu_environment
from mosaickv.measurements.timing import CudaEventTimer, ModuleCudaTimer, SynchronizationAudit
from mosaickv.measurements.types import (
    FullKVTrialMeasurement,
    FullKVTrialOutput,
    GpuEnvironmentSnapshot,
    MemoryMeasurements,
    PhaseTimings,
)


@dataclass
class FakeDevice:
    type: str


class FakeTensor:
    def __init__(self, numel: int, element_size: int, device: str = "cuda") -> None:
        self._numel = numel
        self._element_size = element_size
        self.device = FakeDevice(device)

    def numel(self) -> int:
        return self._numel

    def element_size(self) -> int:
        return self._element_size


class FakeEvent:
    def __init__(self, cuda: FakeCuda) -> None:
        self.cuda = cuda
        self.timestamp: int | None = None

    def record(self) -> None:
        self.cuda.operations.append("record")
        self.timestamp = self.cuda.clock
        self.cuda.clock += 2

    def elapsed_time(self, other: FakeEvent) -> float:
        assert self.timestamp is not None
        assert other.timestamp is not None
        return float(other.timestamp - self.timestamp)


class FakeCuda:
    def __init__(self) -> None:
        self.operations: list[str] = []
        self.clock = 1

    def synchronize(self, _device: object) -> None:
        self.operations.append("synchronize")

    def Event(self, *, enable_timing: bool) -> FakeEvent:
        assert enable_timing
        return FakeEvent(self)


class FakeModule:
    def __init__(self) -> None:
        self.pre_hooks: list[Any] = []
        self.post_hooks: list[Any] = []

    def register_forward_pre_hook(self, hook: Any) -> Any:
        self.pre_hooks.append(hook)
        return SimpleNamespace(remove=lambda: self.pre_hooks.remove(hook))

    def register_forward_hook(self, hook: Any) -> Any:
        self.post_hooks.append(hook)
        return SimpleNamespace(remove=lambda: self.post_hooks.remove(hook))

    def forward(self) -> None:
        for hook in tuple(self.pre_hooks):
            hook(self, ())
        for hook in tuple(self.post_hooks):
            hook(self, (), None)


def _snapshot() -> GpuEnvironmentSnapshot:
    return GpuEnvironmentSnapshot(
        captured_at_utc="2026-01-01T00:00:00+00:00",
        cuda_visible_devices="0",
        torch_version="test",
        cuda_runtime="test",
        devices=(),
        processes=(),
        background_processes=(),
        visible_process_concurrency=1,
    )


def _trial(index: int, total: float, tokens: tuple[int, ...] = (1, 2)) -> FullKVTrialMeasurement:
    return FullKVTrialMeasurement(
        run_id="run",
        sample_id="sample",
        trial_index=index,
        model_id="model",
        model_revision="a" * 40,
        dataset_id="dataset",
        dataset_revision="b" * 40,
        manifest_path="/tmp/manifest.json",
        status="completed",
        error=None,
        answer="answer",
        generated_token_ids=tokens,
        timings=PhaseTimings(0.1, 0.2, 0.3, 0.0, 0.6, (0.4,), 0.0, total, total),
        memory=MemoryMeasurements(100, 120, 64, 0),
        active_cache_length=2,
        logical_sequence_length=2,
        synchronization_calls=17,
        phase_event_counts={"total_latency": 1},
        gpu_before=_snapshot(),
        gpu_after=_snapshot(),
    )


def test_active_kv_bytes_is_exact_numel_times_element_size() -> None:
    cache = (
        (FakeTensor(12, 2), FakeTensor(12, 2)),
        (FakeTensor(8, 4), FakeTensor(8, 4)),
    )
    expected = 12 * 2 + 12 * 2 + 8 * 4 + 8 * 4
    assert active_kv_bytes(cache) == expected
    assert tensor_payload_bytes(cache[0][0]) == 24


def test_cpu_residual_bytes_counts_unique_cpu_tensors_only() -> None:
    cpu = FakeTensor(7, 4, "cpu")
    value = {"first": cpu, "duplicate": [cpu], "gpu": FakeTensor(99, 2, "cuda")}
    assert cpu_residual_bytes(value) == 28
    assert cpu_residual_bytes(None) == 0


def test_cuda_event_timer_synchronizes_both_boundaries() -> None:
    cuda = FakeCuda()
    torch = SimpleNamespace(cuda=cuda)
    audit = SynchronizationAudit()
    timer = CudaEventTimer(torch, FakeDevice("cuda"), audit)
    timer.start()
    assert timer.stop() == pytest.approx(0.002)
    assert cuda.operations == ["synchronize", "record", "record", "synchronize"]
    assert audit.calls == 2


def test_module_cuda_timer_uses_event_timer_for_every_invocation() -> None:
    cuda = FakeCuda()
    module = FakeModule()
    audit = SynchronizationAudit()
    with ModuleCudaTimer(module, SimpleNamespace(cuda=cuda), FakeDevice("cuda"), audit) as timer:
        module.forward()
        module.forward()
    assert timer.invocation_count == 2
    assert timer.total_seconds == pytest.approx(0.004)
    assert audit.calls == 4
    assert module.pre_hooks == []
    assert module.post_hooks == []


def test_gpu_snapshot_reports_clocks_power_and_background_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mosaickv.measurements import telemetry

    device_row = "0, GPU-test, NVIDIA Test GPU, P2, 1200, 1100, 1500, 88.5, 400, 595.1\n"
    process_rows = (
        f"GPU-test, {os.getpid()}, current-python, 512\nGPU-test, 424242, background-python, 1024\n"
    )

    def fake_query(arguments: list[str]) -> tuple[str, str | None]:
        return (
            (process_rows, None) if "--query-compute-apps" in arguments[0] else (device_row, None)
        )

    monkeypatch.setattr(telemetry, "_run_nvidia_smi", fake_query)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    torch = SimpleNamespace(__version__="2.test", version=SimpleNamespace(cuda="13.test"))
    snapshot = capture_gpu_environment(torch)
    assert snapshot.devices[0].power_state == "P2"
    assert snapshot.devices[0].sm_clock_mhz == "1100"
    assert snapshot.visible_process_concurrency == 2
    assert [process.pid for process in snapshot.background_processes] == [424242]


def test_statistics_and_bootstrap_are_deterministic() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    first = summarize(values, bootstrap_samples=100, confidence_level=0.9, seed=7)
    second = summarize(values, bootstrap_samples=100, confidence_level=0.9, seed=7)
    assert first == second
    assert first.median == 2.5
    assert first.p5 == pytest.approx(1.15)
    assert first.p95 == pytest.approx(3.85)
    assert percentile(values, 0.5) == 2.5


def test_aggregate_reports_per_token_data_and_determinism() -> None:
    aggregate = aggregate_trials(
        "run",
        (_trial(0, 1.0), _trial(1, 2.0)),
        warmups=1,
        repeated_trials=2,
        bootstrap_samples=10,
        confidence_level=0.95,
        seed=0,
    )
    assert aggregate.completed_trials == 2
    assert aggregate.failed_trials == 0
    assert aggregate.deterministic_token_match is True
    assert aggregate.metrics["total_latency_seconds"].median == 1.5
    assert len(aggregate.per_token_decode) == 1

    mismatch = aggregate_trials(
        "run",
        (_trial(0, 1.0), _trial(1, 1.0, (1, 3))),
        warmups=0,
        repeated_trials=2,
        bootstrap_samples=10,
        confidence_level=0.95,
        seed=0,
    )
    assert mismatch.deterministic_token_match is False


def test_raw_trials_and_aggregate_are_immutable(tmp_path: Path) -> None:
    trials = (_trial(0, 1.0), _trial(1, 2.0))
    aggregate = aggregate_trials(
        "run",
        trials,
        warmups=0,
        repeated_trials=2,
        bootstrap_samples=10,
        confidence_level=0.95,
        seed=0,
    )
    raw = write_trial_jsonl(trials, tmp_path / "raw.jsonl")
    derived = write_aggregate_json(aggregate, tmp_path / "aggregate.json")
    rows = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines()]
    assert [row["trial_index"] for row in rows] == [0, 1]
    assert json.loads(derived.read_text(encoding="utf-8"))["completed_trials"] == 2
    log = write_json_object({"status": "completed"}, tmp_path / "log.json")
    assert json.loads(log.read_text(encoding="utf-8"))["status"] == "completed"
    with pytest.raises(FileExistsError, match="overwrite"):
        write_trial_jsonl(trials, raw)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"warmups": -1}, "warmups"),
        ({"repeated_trials": 0}, "repeated_trials"),
        ({"max_new_tokens": 0}, "max_new_tokens"),
        ({"confidence_level": 1.0}, "confidence_level"),
    ],
)
def test_benchmark_config_rejects_invalid_values(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        FullKVBenchmarkConfig(**kwargs)  # type: ignore[arg-type]


def test_fullkv_rejects_non_greedy_reference_generation() -> None:
    with pytest.raises(ValueError, match="temperature=0"):
        FullKV._validate_generation(temperature=0.1, do_sample=False)
    with pytest.raises(ValueError, match="do_sample=False"):
        FullKV._validate_generation(temperature=0.0, do_sample=True)
    FullKV._assert_device_resident_cache(((FakeTensor(1, 2), FakeTensor(1, 2)),))
    with pytest.raises(RuntimeError, match="offloading is forbidden"):
        FullKV._assert_device_resident_cache(((FakeTensor(1, 2, "cpu"), FakeTensor(1, 2)),))


def test_fullkv_evaluation_wrapper_maps_raw_metrics() -> None:
    measurement = _trial(0, 1.0)

    class FakeAdapter:
        capabilities = SimpleNamespace(video=True)

        def prepare_inputs(self, messages: object) -> object:
            assert messages == ()
            return "prepared"

    class FakeReference:
        model_id = "model"
        adapter = FakeAdapter()

        def run_trial(self, prepared: object, **kwargs: object) -> FullKVTrialOutput:
            assert prepared == "prepared"
            assert kwargs["max_new_tokens"] == 2
            return FullKVTrialOutput("answer", None, measurement)

    model = FullKVEvaluationModel(
        FakeReference(),  # type: ignore[arg-type]
        dataset_id="dataset",
        dataset_revision="revision",
        manifest_path="/tmp/manifest.json",
        max_new_tokens=2,
    )
    generated = model.generate(EvaluationRequest("run", "sample", "task", (), {}))
    assert generated.answer == "answer"
    assert generated.metrics.ttft == 0.6
    assert generated.metrics.active_kv_bytes == 64
    assert generated.metrics.repair_count == 0
    assert model.raw_measurements == (measurement,)

    with pytest.raises(ValueError, match="differs"):
        model.generate(EvaluationRequest("run", "sample", "task", (), {"max_new_tokens": 3}))
