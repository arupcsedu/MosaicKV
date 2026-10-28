"""Bounded observational telemetry, separate from live-KV accounting."""

from __future__ import annotations

import os
import threading
import time
from contextlib import ExitStack, contextmanager
from functools import wraps
from unittest.mock import patch


class ProcessSampler:
    """Sample this PID only; unavailable NVML is reported, never replaced by KV."""

    def __init__(self, gpu_uuid, interval=0.1):
        self.gpu_uuid = gpu_uuid
        self.interval = interval
        self.samples = []
        self.errors = []
        self._stop = threading.Event()
        self._thread = None
        self._nvml = None
        self._handle = None

    def sample(self):
        rss = None
        with open("/proc/self/status") as stream:
            for line in stream:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) * 1024
                    break
        used = None
        if self._nvml is not None:
            try:
                processes = self._nvml.nvmlDeviceGetComputeRunningProcesses(self._handle)
                own = [p.usedGpuMemory for p in processes if p.pid == os.getpid()]
                if own:
                    used = int(own[0])
            except Exception as error:
                self.errors.append(f"{type(error).__name__}: {error}")
        self.samples.append(
            dict(monotonic=time.perf_counter(), cpu_rss_bytes=rss, process_gpu_bytes=used)
        )

    def __enter__(self):
        try:
            import pynvml

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByUUID(self.gpu_uuid)
            self._nvml = pynvml
        except Exception as error:
            self.errors.append(f"{type(error).__name__}: {error}")
        self.sample()

        def loop():
            while not self._stop.wait(self.interval):
                self.sample()

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *unused):
        self._stop.set()
        self._thread.join(timeout=2)
        self.sample()
        if self._nvml is not None:
            self._nvml.nvmlShutdown()

    def record(self):
        gpu = [s["process_gpu_bytes"] for s in self.samples if s["process_gpu_bytes"] is not None]
        cpu = [s["cpu_rss_bytes"] for s in self.samples if s["cpu_rss_bytes"] is not None]
        return dict(
            interval_seconds=self.interval,
            sample_count=len(self.samples),
            sampled_process_gpu_peak_bytes=max(gpu) if gpu else None,
            sampled_cpu_rss_peak_bytes=max(cpu) if cpu else None,
            process_peak_scope="sampled_not_continuous_not_allocator_not_KV",
            errors=sorted(set(self.errors)),
            samples=self.samples,
        )


@contextmanager
def phase_ranges():
    """Observational ranges around actual functions; nested totals are not additive."""
    import torch

    import mosaickv.backends.hf_runtime as runtime

    functions = (
        "forecast_with_rollout",
        "_prompt_token_attention_by_head",
        "_similarity_shortlist",
        "_dense_physical_cache_budget",
        "_gather_dense_heads",
        "build_evidence_graph",
        "compute_block_utilities",
        "_selection",
        "_exact_construction",
        "pack_runtime_cache",
        "_state_from_packed",
    )
    with ExitStack() as stack:
        for name in functions:
            function = getattr(runtime, name)

            def wrap(original, label):
                @wraps(original)
                def call(*args, **kwargs):
                    with torch.profiler.record_function("evicastkv/" + label):
                        return original(*args, **kwargs)

                return call

            stack.enter_context(patch.object(runtime, name, wrap(function, name)))
        yield
