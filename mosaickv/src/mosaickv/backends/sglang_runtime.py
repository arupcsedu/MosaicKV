"""Version-pinned SGLang FullKV measurements and native safety gate.

SGLang is imported only by the launched server process.  This keeps CPU-only
diagnostics and unit tests independent of the optional serving environment.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Protocol, cast

from mosaickv.backends.vllm_runtime import (
    _resolve_model_source,
    prepare_vllm_prompt,
)
from mosaickv.config import RunConfig
from mosaickv.evaluation.model import EvaluationRequest, GenerationMetrics, ModelGeneration
from mosaickv.types import JsonObject, Precision

_AUDITED_SGLANG_VERSION = "0.5.10.post1"
_SUPPORTED_MODELS = {
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
}


class SGLangRuntimeError(RuntimeError):
    """Raised when a measured SGLang request cannot preserve its contract."""


class NativeMosaicKVUnsupported(SGLangRuntimeError):
    """Raised before server launch when the native mutation contract is absent."""


@dataclass(frozen=True, slots=True)
class NativeIntegrationCapability:
    """Machine-readable verdict for the installed SGLang integration seam."""

    sglang_version: str
    supported: bool
    feature: str
    reason_code: str
    blocker_document: str
    inspected_symbols: tuple[str, ...]

    def to_json_object(self) -> JsonObject:
        return cast("JsonObject", asdict(self))


def native_integration_capability(sglang_version: str) -> NativeIntegrationCapability:
    """Return a fail-closed whole-block capability for a SGLang release."""

    return NativeIntegrationCapability(
        sglang_version=sglang_version,
        supported=False,
        feature="whole_block_selection_with_original_logical_and_mrope_positions",
        reason_code=(
            "audited_0_5_10_post1_missing_atomic_sparse_request_cache_hook"
            if sglang_version == _AUDITED_SGLANG_VERSION
            else "unaudited_sglang_version"
        ),
        blocker_document="docs/sglang_native_blocker.md",
        inspected_symbols=(
            "sglang.srt.entrypoints.engine.Engine.generate",
            "sglang.srt.mem_cache.memory_pool.ReqToTokenPool.req_to_token",
            "sglang.srt.mem_cache.memory_pool.MHATokenToKVPool.set_kv_buffer",
            "sglang.srt.mem_cache.radix_cache.RadixCache.cache_finished_req",
            "sglang.srt.model_executor.forward_batch_info.ForwardBatch.mrope_positions",
            "sglang.srt.layers.attention.base_attn_backend.AttentionBackend",
        ),
    )


def require_native_mosaickv_support(*, enabled: bool, sglang_version: str) -> None:
    """Reject an unavailable native path before a server or weights are loaded."""

    if not enabled:
        return
    capability = native_integration_capability(sglang_version)
    if not capability.supported:
        raise NativeMosaicKVUnsupported(
            "native MosaicKV is unsupported for SGLang "
            f"{sglang_version}: {capability.reason_code}; see "
            f"{capability.blocker_document}. No simulated MosaicKV row was emitted."
        )


@dataclass(frozen=True, slots=True)
class SGLangRuntimeOptions:
    """Server and measurement controls not represented by :class:`RunConfig`."""

    tensor_parallel_size: int = 1
    mem_fraction_static: float = 0.8
    context_length: int | None = 4096
    cache_probe_repeats: int = 2
    local_files_only: bool = False
    enable_mosaickv: bool = False
    port: int = 0
    startup_timeout_seconds: float = 1800.0
    page_size: int = 1

    def __post_init__(self) -> None:
        if self.tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be >= 1")
        if not math.isfinite(self.mem_fraction_static) or not (0 < self.mem_fraction_static < 1):
            raise ValueError("mem_fraction_static must be finite and in (0, 1)")
        if self.context_length is not None and self.context_length < 2:
            raise ValueError("context_length must be >= 2 or null")
        if self.cache_probe_repeats < 1:
            raise ValueError("cache_probe_repeats must be >= 1")
        if not 0 <= self.port <= 65535:
            raise ValueError("port must be in [0, 65535]")
        if not math.isfinite(self.startup_timeout_seconds) or self.startup_timeout_seconds <= 0:
            raise ValueError("startup_timeout_seconds must be finite and > 0")
        if self.page_size < 1:
            raise ValueError("page_size must be >= 1")


@dataclass(frozen=True, slots=True)
class KVCacheGeometry:
    """Exact logical KV footprint for the served decoder at TP=1."""

    layers: int
    kv_heads: int
    head_dim: int
    dtype_bytes: int

    def __post_init__(self) -> None:
        if min(self.layers, self.kv_heads, self.head_dim, self.dtype_bytes) < 1:
            raise ValueError("KV cache geometry values must all be >= 1")

    @property
    def bytes_per_position(self) -> int:
        """K plus V bytes for one logical decoder position."""

        return 2 * self.layers * self.kv_heads * self.head_dim * self.dtype_bytes

    def active_bytes(self, positions: int) -> int:
        if positions < 0:
            raise ValueError("positions must be >= 0")
        return positions * self.bytes_per_position

    def to_json_object(self) -> JsonObject:
        return {
            "layers": self.layers,
            "kv_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "dtype_bytes": self.dtype_bytes,
            "bytes_per_position": self.bytes_per_position,
        }


def _text_config(config: Any) -> Any:
    return getattr(config, "text_config", config)


def kv_cache_geometry(model_config: Any, precision: Precision) -> KVCacheGeometry:
    """Read cache geometry from the exact checkpoint configuration."""

    config = _text_config(model_config)
    layers = int(config.num_hidden_layers)
    attention_heads = int(config.num_attention_heads)
    kv_heads = int(getattr(config, "num_key_value_heads", attention_heads))
    head_dim = int(getattr(config, "head_dim", int(config.hidden_size) // attention_heads))
    dtype_bytes = {
        Precision.FP16: 2,
        Precision.BF16: 2,
        Precision.FP32: 4,
    }.get(precision)
    if dtype_bytes is None:
        raise SGLangRuntimeError("SGLang FullKV supports fp16, bf16, or fp32")
    return KVCacheGeometry(layers, kv_heads, head_dim, dtype_bytes)


@dataclass(frozen=True, slots=True)
class PreparedSGLangPrompt:
    """Canonical rendered prompt plus JSON-safe unchanged media payloads."""

    request_payload: JsonObject
    rendered_text: str
    prompt_sha256: str
    media_sha256: str
    prompt_token_ids: tuple[int, ...]


def _json_media(payload: Any, modality: str) -> str:
    if isinstance(payload, bytes):
        return base64.b64encode(payload).decode("ascii")
    if isinstance(payload, str):
        path = Path(payload)
        if path.is_file():
            return path.resolve().as_uri()
        return payload
    if modality == "image" and hasattr(payload, "save"):
        buffer = io.BytesIO()
        payload.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")
    raise SGLangRuntimeError(
        f"SGLang HTTP {modality} payloads must be bytes, paths/URLs, or PIL images"
    )


def prepare_sglang_prompt(
    processor: Any,
    model_id: str,
    messages: Sequence[Any],
) -> PreparedSGLangPrompt:
    """Reuse the HF/vLLM chat template boundary and serialize media for HTTP."""

    common = prepare_vllm_prompt(processor, model_id, messages)
    request: JsonObject = {"text": common.rendered_text}
    multi_modal = common.engine_prompt.get("multi_modal_data", {})
    modalities: list[str] = []
    if not isinstance(multi_modal, Mapping):
        raise SGLangRuntimeError("multimodal prompt payload must be a mapping")
    for modality in ("image", "video"):
        raw = multi_modal.get(modality)
        if raw is None:
            continue
        values = list(raw) if isinstance(raw, (list, tuple)) else [raw]
        serialized = [_json_media(value, modality) for value in values]
        request[f"{modality}_data"] = cast(
            "Any", serialized[0] if len(serialized) == 1 else serialized
        )
        modalities.extend([modality] * len(serialized))
    if modalities:
        request["modalities"] = cast("Any", modalities)
    tokenizer = getattr(processor, "tokenizer", processor)
    encoded = tokenizer(common.rendered_text, add_special_tokens=False)
    input_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return PreparedSGLangPrompt(
        request_payload=request,
        rendered_text=common.rendered_text,
        prompt_sha256=common.prompt_sha256,
        media_sha256=common.media_sha256,
        prompt_token_ids=tuple(int(value) for value in input_ids),
    )


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def build_server_command(
    config: RunConfig,
    options: SGLangRuntimeOptions,
    *,
    model_source: str,
    port: int,
) -> tuple[str, ...]:
    """Build the exact correctness-first server command recorded in manifests."""

    dtype = {
        Precision.FP16: "float16",
        Precision.BF16: "bfloat16",
        Precision.FP32: "float32",
    }.get(config.model.precision)
    if dtype is None:
        raise SGLangRuntimeError("SGLang FullKV supports fp16, bf16, or fp32")
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_source,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--dtype",
        dtype,
        "--tp-size",
        str(options.tensor_parallel_size),
        "--mem-fraction-static",
        str(options.mem_fraction_static),
        "--page-size",
        str(options.page_size),
        "--stream-interval",
        "1",
        "--random-seed",
        str(config.execution.seed),
        "--attention-backend",
        config.execution.attention_implementation,
        "--model-impl",
        "sglang",
        "--disable-fast-image-processor",
        "--enable-multimodal",
        "--enable-deterministic-inference",
        "--enable-metrics",
        "--enable-cache-report",
        "--disable-overlap-schedule",
        "--disable-cuda-graph",
        "--skip-server-warmup",
    ]
    if options.context_length is not None:
        command.extend(("--context-length", str(options.context_length)))
    if model_source == config.model.id:
        command.extend(("--revision", config.model.revision))
    return tuple(command)


def _descendant_pids(root: int) -> set[int]:
    parents: dict[int, int] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text(encoding="utf-8").split()
            parents[int(entry.name)] = int(fields[3])
        except (IndexError, OSError, ValueError):
            continue
    selected = {root}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in selected and pid not in selected:
                selected.add(pid)
                changed = True
    return selected


def _query_gpu_bytes(root_pid: int) -> tuple[int | None, str | None]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None, None
    if completed.returncode != 0:
        return None, None
    pids = _descendant_pids(root_pid)
    total_mib = 0
    observed = False
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            pid, used_mib = int(fields[0]), int(fields[1])
        except ValueError:
            continue
        if pid in pids:
            total_mib += used_mib
            observed = True
    return (total_mib * 1024 * 1024 if observed else None), "nvidia-smi process memory"


def _executable_version(executable: str) -> str | None:
    """Return the first version line for a runtime tool without failing launch."""

    try:
        completed = subprocess.run(
            [executable, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    lines = completed.stdout.splitlines()
    return lines[0].strip() if lines else None


class _GPUMemorySampler:
    def __init__(self, root_pid: int, interval_seconds: float = 0.02) -> None:
        self.root_pid = root_pid
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.baseline_bytes: int | None = None
        self.peak_bytes: int | None = None
        self.source: str | None = None

    def start(self) -> None:
        self.baseline_bytes, self.source = _query_gpu_bytes(self.root_pid)
        self.peak_bytes = self.baseline_bytes

        def sample() -> None:
            while not self._stop.wait(self.interval_seconds):
                value, source = _query_gpu_bytes(self.root_pid)
                if source is not None:
                    self.source = source
                if value is not None and (self.peak_bytes is None or value > self.peak_bytes):
                    self.peak_bytes = value

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        value, source = _query_gpu_bytes(self.root_pid)
        if source is not None:
            self.source = source
        if value is not None and (self.peak_bytes is None or value > self.peak_bytes):
            self.peak_bytes = value


def _metric_values(payload: str, name: str) -> tuple[float, ...]:
    values: list[float] = []
    prefix = name + "{"
    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        metric_name, _, raw_value = line.rpartition(" ")
        if metric_name == name or metric_name.startswith(prefix):
            try:
                values.append(float(raw_value))
            except ValueError:
                continue
    return tuple(values)


def _metric_sum(payload: str, name: str) -> float | None:
    values = _metric_values(payload, name)
    return sum(values) if values else None


def _metric_last(payload: str, name: str) -> float | None:
    values = _metric_values(payload, name)
    return values[-1] if values else None


def _float_or_none(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


@dataclass(frozen=True, slots=True)
class SGLangTrialMeasurement:
    """One raw, non-aggregated HTTP streaming observation."""

    request_id: str
    answer: str
    token_ids: tuple[int, ...]
    prompt_tokens: int
    generated_tokens: int
    active_cache_positions: int
    active_kv_bytes: int
    ttft_seconds: float | None
    request_latency_seconds: float
    decode_seconds: float | None
    inter_token_latencies_seconds: tuple[float, ...]
    token_timestamps_seconds: tuple[float, ...]
    throughput_tokens_per_second: float
    decode_throughput_tokens_per_second: float | None
    cached_tokens: int
    prefix_cache_hit_rate: float
    server_e2e_latency_seconds: float | None
    server_prefill_seconds: float | None
    server_queue_seconds: float | None
    prometheus_cache_hit_rate: float | None
    prometheus_cached_tokens_delta: float | None
    prometheus_generation_throughput: float | None
    prometheus_token_usage: float | None
    gpu_memory_source: str | None
    gpu_memory_baseline_bytes: int | None
    gpu_memory_peak_bytes: int | None
    gpu_memory_peak_delta_bytes: int | None
    finish_reason: str | None

    def to_json_object(self) -> JsonObject:
        return cast("JsonObject", asdict(self))


class SGLangTrialRunner(Protocol):
    """Injectable server boundary used by the FullKV wrapper."""

    sglang_version: str
    engine_metadata: JsonObject
    cache_geometry: KVCacheGeometry

    def run(
        self,
        prompt: JsonObject,
        generation: Mapping[str, object],
        request_id: str,
    ) -> SGLangTrialMeasurement: ...

    def close(self) -> None: ...


class SGLangHTTPTrialRunner:
    """Persistent correctness-first SGLang HTTP server and streaming client."""

    def __init__(
        self,
        config: RunConfig,
        options: SGLangRuntimeOptions,
        *,
        model_source: str,
        cache_geometry: KVCacheGeometry,
        log_directory: str | Path,
    ) -> None:
        try:
            import importlib.metadata

            version = importlib.metadata.version("sglang")
        except importlib.metadata.PackageNotFoundError as error:
            raise SGLangRuntimeError("the pinned SGLang environment is not installed") from error
        if version != _AUDITED_SGLANG_VERSION:
            raise SGLangRuntimeError(
                f"SGLang {version} is unaudited; expected exactly {_AUDITED_SGLANG_VERSION}"
            )
        self.sglang_version = version
        self.cache_geometry = cache_geometry
        self._port = options.port or _free_local_port()
        self._base_url = f"http://127.0.0.1:{self._port}"
        self._command = build_server_command(
            config, options, model_source=model_source, port=self._port
        )
        log_root = Path(log_directory)
        log_root.mkdir(parents=True, exist_ok=True)
        self._log_path = log_root / f"server-{self._port}.log"
        self._log_handle = self._log_path.open("w", encoding="utf-8")
        environment = dict(os.environ)
        environment.setdefault("TOKENIZERS_PARALLELISM", "false")
        # SGLang JIT-compiles several attention/position kernels after the
        # server has started.  It invokes the pinned ``ninja`` executable by
        # name, so direct calls to ``<venv>/bin/python`` must still expose the
        # interpreter's sibling executables.  Do this for the child only; it
        # neither activates nor mutates the environment.
        interpreter_bin = str(Path(sys.executable).parent)
        environment["PATH"] = os.pathsep.join(
            part for part in (interpreter_bin, environment.get("PATH", "")) if part
        )
        self._process = subprocess.Popen(
            self._command,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
        )
        try:
            import httpx
        except ImportError as error:
            self.close()
            raise SGLangRuntimeError("SGLang measurement requires httpx") from error
        self._client = httpx.Client(base_url=self._base_url, timeout=None)
        try:
            self._wait_until_ready(options.startup_timeout_seconds)
            server_info = self._get_json("/server_info")
        except BaseException:
            self.close()
            raise
        internal_states = server_info.get("internal_states", [])
        memory_usage: JsonObject | None = None
        if (
            isinstance(internal_states, list)
            and internal_states
            and isinstance(internal_states[0], Mapping)
            and isinstance(internal_states[0].get("memory_usage"), Mapping)
        ):
            raw_memory = cast("Mapping[str, Any]", internal_states[0]["memory_usage"])
            memory_usage = cast("JsonObject", dict(raw_memory))
        self.engine_metadata: JsonObject = {
            "sglang_version": version,
            "execution_mode": "http_server_correctness_debug",
            "model_source": model_source,
            "model_revision": config.model.revision,
            "server_command": list(self._command),
            "server_log": str(self._log_path.resolve()),
            "server_pid": self._process.pid,
            "host": "127.0.0.1",
            "port": self._port,
            "attention_backend": config.execution.attention_implementation,
            "deterministic_inference": True,
            "overlap_schedule": False,
            "cuda_graph": False,
            "server_warmup": "skipped",
            "page_size": options.page_size,
            "tensor_parallel_size": options.tensor_parallel_size,
            "mem_fraction_static": options.mem_fraction_static,
            "context_length": options.context_length,
            "radix_cache": True,
            "image_processor": "checkpoint_slow_processor",
            "cxx_compiler": _executable_version("g++"),
            "cc_environment": os.environ.get("CC"),
            "cxx_environment": os.environ.get("CXX"),
            "tvm_ffi_cache_dir": os.environ.get("TVM_FFI_CACHE_DIR"),
            "memory_pool": memory_usage,
            "kv_cache_geometry": cache_geometry.to_json_object(),
        }

    def _log_tail(self) -> str:
        self._log_handle.flush()
        try:
            lines = self._log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return "unavailable"
        return "\n".join(lines[-80:])

    def _wait_until_ready(self, timeout_seconds: float) -> None:
        deadline = time.monotonic() + timeout_seconds
        last_error = "server has not responded"
        while time.monotonic() < deadline:
            return_code = self._process.poll()
            if return_code is not None:
                raise SGLangRuntimeError(
                    f"SGLang server exited with code {return_code}:\n{self._log_tail()}"
                )
            try:
                response = self._client.get("/health", timeout=2.0)
                if response.status_code == 200:
                    return
                last_error = f"health status {response.status_code}"
            except Exception as error:
                last_error = f"{type(error).__name__}: {error}"
            time.sleep(0.25)
        raise SGLangRuntimeError(
            f"SGLang startup timed out after {timeout_seconds}s ({last_error}):\n{self._log_tail()}"
        )

    def _get_json(self, path: str) -> JsonObject:
        response = self._client.get(path, timeout=30.0)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise SGLangRuntimeError(f"{path} did not return a JSON object")
        return cast("JsonObject", dict(payload))

    def _metrics(self) -> str:
        response = self._client.get("/metrics", timeout=30.0)
        response.raise_for_status()
        return response.text

    def run(
        self,
        prompt: JsonObject,
        generation: Mapping[str, object],
        request_id: str,
    ) -> SGLangTrialMeasurement:
        max_new_tokens = generation.get("max_new_tokens")
        seed = generation.get("seed")
        temperature = generation.get("temperature")
        top_p = generation.get("top_p")
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
            raise SGLangRuntimeError("generation.max_new_tokens must be an integer")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise SGLangRuntimeError("generation.seed must be an integer")
        if temperature != 0 and temperature != 0.0:
            raise SGLangRuntimeError("SGLang FullKV measurements require temperature=0")
        if top_p != 1 and top_p != 1.0:
            raise SGLangRuntimeError("SGLang FullKV measurements require top_p=1")
        payload = dict(prompt)
        payload.update(
            {
                "rid": request_id,
                "stream": True,
                "sampling_params": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "sampling_seed": seed,
                    # The explicit HF loop is pure greedy decoding with no
                    # repetition/presence/frequency transform and always runs
                    # the configured fixed output length, including past EOS.
                    "repetition_penalty": 1.0,
                    "presence_penalty": 0.0,
                    "frequency_penalty": 0.0,
                    "ignore_eos": True,
                },
            }
        )
        before_metrics = self._metrics()
        memory = _GPUMemorySampler(self._process.pid)
        memory.start()
        started = time.perf_counter()
        token_ids: tuple[int, ...] = ()
        timestamps: list[float] = []
        last_output: Mapping[str, Any] | None = None
        try:
            with self._client.stream("POST", "/generate", json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    if not isinstance(event, Mapping):
                        raise SGLangRuntimeError("stream event must be a JSON object")
                    if "error" in event:
                        raise SGLangRuntimeError(f"SGLang request failed: {event['error']}")
                    raw_ids = event.get("output_ids")
                    if not isinstance(raw_ids, list):
                        raise SGLangRuntimeError("SGLang stream event omitted output_ids")
                    current_ids = tuple(int(value) for value in raw_ids)
                    if current_ids[: len(token_ids)] != token_ids:
                        raise SGLangRuntimeError("SGLang cumulative output IDs changed in flight")
                    now = time.perf_counter() - started
                    timestamps.extend([now] * (len(current_ids) - len(token_ids)))
                    token_ids = current_ids
                    last_output = event
        finally:
            memory.stop()
        request_latency = time.perf_counter() - started
        after_metrics = self._metrics()
        if last_output is None:
            raise SGLangRuntimeError("SGLang returned no stream events")
        meta = last_output.get("meta_info")
        if not isinstance(meta, Mapping):
            raise SGLangRuntimeError("SGLang final event omitted meta_info")
        answer = last_output.get("text")
        if not isinstance(answer, str):
            raise SGLangRuntimeError("SGLang final event omitted decoded text")
        prompt_tokens = int(meta.get("prompt_tokens", 0))
        completion_tokens = int(meta.get("completion_tokens", len(token_ids)))
        if completion_tokens != len(token_ids):
            raise SGLangRuntimeError(
                "SGLang completion token count differs from streamed output IDs"
            )
        cached_tokens = int(meta.get("cached_tokens", 0))
        active_positions = prompt_tokens + max(0, completion_tokens - 1)
        active_bytes = self.cache_geometry.active_bytes(active_positions)
        ttft = timestamps[0] if timestamps else None
        itls = tuple(later - earlier for earlier, later in pairwise(timestamps))
        decode = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0
        decode_throughput = (
            (len(timestamps) - 1) / decode if len(timestamps) > 1 and decode > 0 else None
        )
        server_prefill = None
        forward_entry = _float_or_none(meta.get("forward_entry_time"))
        prefill_finished = _float_or_none(meta.get("prefill_finished_time"))
        if forward_entry is not None and prefill_finished is not None:
            server_prefill = max(0.0, prefill_finished - forward_entry)
        before_cached = _metric_sum(before_metrics, "sglang:cached_tokens_total")
        after_cached = _metric_sum(after_metrics, "sglang:cached_tokens_total")
        cached_delta = (
            after_cached - before_cached
            if before_cached is not None and after_cached is not None
            else None
        )
        peak_delta = (
            max(0, memory.peak_bytes - memory.baseline_bytes)
            if memory.peak_bytes is not None and memory.baseline_bytes is not None
            else None
        )
        return SGLangTrialMeasurement(
            request_id=request_id,
            answer=answer,
            token_ids=token_ids,
            prompt_tokens=prompt_tokens,
            generated_tokens=completion_tokens,
            active_cache_positions=active_positions,
            active_kv_bytes=active_bytes,
            ttft_seconds=ttft,
            request_latency_seconds=request_latency,
            decode_seconds=decode,
            inter_token_latencies_seconds=itls,
            token_timestamps_seconds=tuple(timestamps),
            throughput_tokens_per_second=(
                completion_tokens / request_latency if request_latency > 0 else 0.0
            ),
            decode_throughput_tokens_per_second=decode_throughput,
            cached_tokens=cached_tokens,
            prefix_cache_hit_rate=(cached_tokens / prompt_tokens if prompt_tokens else 0.0),
            server_e2e_latency_seconds=_float_or_none(meta.get("e2e_latency")),
            server_prefill_seconds=server_prefill,
            server_queue_seconds=_float_or_none(meta.get("queue_time")),
            prometheus_cache_hit_rate=_metric_last(after_metrics, "sglang:cache_hit_rate"),
            prometheus_cached_tokens_delta=cached_delta,
            prometheus_generation_throughput=_metric_last(after_metrics, "sglang:gen_throughput"),
            prometheus_token_usage=_metric_last(after_metrics, "sglang:token_usage"),
            gpu_memory_source=memory.source,
            gpu_memory_baseline_bytes=memory.baseline_bytes,
            gpu_memory_peak_bytes=memory.peak_bytes,
            gpu_memory_peak_delta_bytes=peak_delta,
            finish_reason=(
                str(meta["finish_reason"]) if meta.get("finish_reason") is not None else None
            ),
        )

    def close(self) -> None:
        client = getattr(self, "_client", None)
        if client is not None:
            client.close()
        process = getattr(self, "_process", None)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        log_handle = getattr(self, "_log_handle", None)
        if log_handle is not None and not log_handle.closed:
            log_handle.close()


def _atomic_json(path: Path, payload: JsonObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = handle.name
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
        raise


def _safe_trace_name(sample_id: str) -> str:
    readable = "".join(
        character if character.isalnum() or character in "-_" else "_" for character in sample_id
    )
    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:12]
    return f"{readable[:80]}-{digest}.json"


class SGLangFullKVModel:
    """Backend-neutral evaluation adapter for measured SGLang FullKV."""

    backend = "sglang"
    method = "full_kv"
    retention_ratio = 1.0
    supports_video = True

    def __init__(
        self,
        *,
        model_id: str,
        processor: Any,
        runner: SGLangTrialRunner,
        trace_directory: str | Path,
        generation: Mapping[str, object],
        cache_probe_repeats: int = 2,
    ) -> None:
        if model_id not in _SUPPORTED_MODELS:
            known = ", ".join(sorted(_SUPPORTED_MODELS))
            raise LookupError(f"unsupported SGLang multimodal model {model_id!r}; known: {known}")
        if cache_probe_repeats < 1:
            raise ValueError("cache_probe_repeats must be >= 1")
        self.model_id = model_id
        self.processor = processor
        self.runner = runner
        self.trace_directory = Path(trace_directory)
        self.generation = dict(generation)
        self.cache_probe_repeats = cache_probe_repeats
        self._isolation_anchor: (
            tuple[
                PreparedSGLangPrompt,
                dict[str, object],
                tuple[int, ...],
                Path,
                str,
                str,
            ]
            | None
        ) = None
        self._observed_input_fingerprints: set[tuple[str, str]] = set()

    @classmethod
    def from_config(
        cls,
        config: RunConfig,
        options: SGLangRuntimeOptions,
        *,
        trace_directory: str | Path,
    ) -> SGLangFullKVModel:
        if config.execution.backend.value != "sglang":
            raise ValueError("SGLangFullKVModel requires execution.backend='sglang'")
        if options.enable_mosaickv:
            require_native_mosaickv_support(enabled=True, sglang_version=_AUDITED_SGLANG_VERSION)
        if not config.method.is_full_cache or config.cache.retention_ratio != 1.0:
            raise SGLangRuntimeError(
                "Stage A emits only SGLang FullKV rows at retention_ratio=1.0; "
                "native MosaicKV is not simulated"
            )
        if options.tensor_parallel_size != 1:
            raise SGLangRuntimeError(
                "the audited active-KV byte contract currently supports tp_size=1 only"
            )
        if options.page_size != 1:
            raise SGLangRuntimeError(
                "the audited FullKV byte contract requires page_size=1 to avoid padding"
            )
        if config.model.id not in _SUPPORTED_MODELS:
            known = ", ".join(sorted(_SUPPORTED_MODELS))
            raise LookupError(f"unsupported SGLang multimodal model; known: {known}")
        if config.execution.attention_implementation not in {"triton", "fa3"}:
            raise SGLangRuntimeError(
                "the deterministic SGLang wrapper requires a Radix-compatible deterministic "
                "attention backend: triton or fa3"
            )
        if not config.execution.deterministic_algorithms:
            raise SGLangRuntimeError("SGLang FullKV requires deterministic_algorithms=true")
        if config.generation.do_sample or config.generation.temperature != 0:
            raise SGLangRuntimeError("SGLang FullKV requires greedy temperature-0 generation")
        try:
            from transformers import AutoConfig, AutoProcessor
        except ImportError as error:
            raise SGLangRuntimeError("SGLang environment must provide transformers") from error
        model_source = _resolve_model_source(
            config.model.id,
            config.model.revision,
            local_files_only=options.local_files_only,
        )
        load_kwargs: dict[str, Any] = {
            "local_files_only": options.local_files_only,
        }
        if model_source == config.model.id:
            load_kwargs["revision"] = config.model.revision
        processor_kwargs = dict(load_kwargs)
        # Transformers 5 changed Qwen-VL to fast-by-default. The HF reference
        # was established with the checkpoint's slow processor.
        processor_kwargs["use_fast"] = False
        processor = AutoProcessor.from_pretrained(model_source, **processor_kwargs)
        model_config = AutoConfig.from_pretrained(model_source, **load_kwargs)
        geometry = kv_cache_geometry(model_config, config.model.precision)
        runner = SGLangHTTPTrialRunner(
            config,
            options,
            model_source=model_source,
            cache_geometry=geometry,
            log_directory=Path(trace_directory) / "server_logs",
        )
        return cls(
            model_id=config.model.id,
            processor=processor,
            runner=runner,
            trace_directory=trace_directory,
            generation={
                "max_new_tokens": config.generation.max_new_tokens,
                "temperature": config.generation.temperature,
                "top_p": config.generation.top_p,
                "seed": config.execution.seed,
            },
            cache_probe_repeats=options.cache_probe_repeats,
        )

    def generate(self, request: EvaluationRequest) -> ModelGeneration:
        prepared = prepare_sglang_prompt(self.processor, self.model_id, request.messages)
        generation = dict(self.generation)
        generation.update(request.generation_kwargs)
        trials = tuple(
            self.runner.run(
                prepared.request_payload,
                generation,
                f"{request.run_id}-{request.sample_id}-trial-{index}",
            )
            for index in range(self.cache_probe_repeats)
        )
        reference_tokens = trials[0].token_ids
        for index, trial in enumerate(trials[1:], start=1):
            if trial.token_ids != reference_tokens:
                raise SGLangRuntimeError(
                    "deterministic SGLang FullKV cache probe changed generated tokens at "
                    f"trial {index}; first={reference_tokens}, current={trial.token_ids}"
                )
        first = trials[0]
        trace: JsonObject = {
            "schema_version": 1,
            "measurement_type": "sglang_fullkv",
            "backend": "sglang",
            "method": "full_kv",
            "native_mosaickv": False,
            "run_id": request.run_id,
            "sample_id": request.sample_id,
            "model": self.model_id,
            "prompt_sha256": prepared.prompt_sha256,
            "media_sha256": prepared.media_sha256,
            "prompt_token_ids": list(prepared.prompt_token_ids),
            "generation": cast("JsonObject", generation),
            "effective_sampling_parameters": {
                "max_new_tokens": cast("int", generation["max_new_tokens"]),
                "temperature": 0.0,
                "top_p": 1.0,
                "sampling_seed": cast("int", generation["seed"]),
                "repetition_penalty": 1.0,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "ignore_eos": True,
            },
            "engine": self.runner.engine_metadata,
            "trials": [trial.to_json_object() for trial in trials],
            "cache_measurement": {
                "radix_prefix_cache": "Generate response meta_info.cached_tokens",
                "prefix_cache_gauge": "Prometheus sglang:cache_hit_rate",
                "cached_token_counter": "Prometheus sglang:cached_tokens_total delta",
                "encoder_cache": (
                    "repeat-request timing probe only; SGLang 0.5.10.post1 exposes no "
                    "per-request multimodal encoder-cache hit counter"
                ),
                "active_kv_bytes": (
                    "(prompt_tokens + max(completion_tokens - 1, 0)) * "
                    "2 * layers * kv_heads * head_dim * dtype_bytes"
                ),
            },
            "request_isolation": {
                "single_request_per_http_call": True,
                "session_params": None,
                "conversation_id": None,
                "globally_unique_request_ids": True,
            },
        }
        trace_path = self.trace_directory / request.run_id / _safe_trace_name(request.sample_id)
        _atomic_json(trace_path, trace)
        self._observed_input_fingerprints.add((prepared.prompt_sha256, prepared.media_sha256))
        if self._isolation_anchor is None:
            self._isolation_anchor = (
                prepared,
                generation,
                reference_tokens,
                trace_path,
                request.run_id,
                request.sample_id,
            )
        return ModelGeneration(
            answer=first.answer,
            metrics=GenerationMetrics(
                ttft=first.ttft_seconds,
                prefill_time=first.server_prefill_seconds,
                compression_time=0.0,
                decode_time=first.decode_seconds,
                end_to_end_time=first.request_latency_seconds,
                generated_tokens=first.generated_tokens,
                active_kv_bytes=first.active_kv_bytes,
                residual_kv_bytes=0,
                peak_gpu_memory=first.gpu_memory_peak_bytes,
                repair_count=0,
                repaired_bytes=0,
            ),
            effective_method="full_kv",
        )

    def verify_request_isolation(self) -> bool:
        """Re-run the first input after other inputs to detect request-state leakage.

        This A-B-A probe is separate from benchmark measurements. It checks that
        the first request's deterministic token IDs remain unchanged after at
        least one distinct prompt/media fingerprint has traversed the same
        persistent server and Radix cache.
        """

        anchor = self._isolation_anchor
        if anchor is None or len(self._observed_input_fingerprints) < 2:
            return False
        prepared, generation, expected_tokens, trace_path, run_id, sample_id = anchor
        request_id = f"{run_id}-{sample_id}-isolation-recheck"
        trial = self.runner.run(prepared.request_payload, generation, request_id)
        tokens_match = trial.token_ids == expected_tokens
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        if not isinstance(trace, dict) or not isinstance(trace.get("request_isolation"), dict):
            raise SGLangRuntimeError("anchor trace is invalid during request-isolation probe")
        isolation = cast("dict[str, object]", trace["request_isolation"])
        isolation["post_intervening_request_probe"] = {
            "performed": True,
            "intervening_distinct_input_fingerprints": (len(self._observed_input_fingerprints) - 1),
            "request_id": request_id,
            "token_ids_match_anchor": tokens_match,
            "anchor_token_ids": list(expected_tokens),
            "probe_token_ids": list(trial.token_ids),
            "probe_measurement": trial.to_json_object(),
        }
        _atomic_json(trace_path, cast("JsonObject", trace))
        if not tokens_match:
            raise SGLangRuntimeError(
                "SGLang request-isolation A-B-A probe changed the anchor token IDs; "
                "a cache from an intervening request may have leaked"
            )
        return True

    def close(self) -> None:
        self.runner.close()


def sglang_trial_summary(trials: Sequence[SGLangTrialMeasurement]) -> JsonObject:
    """Summarize measured trials while retaining raw observations in traces."""

    if not trials:
        raise ValueError("at least one trial is required")
    latencies = [trial.request_latency_seconds for trial in trials]
    ttfts = [trial.ttft_seconds for trial in trials if trial.ttft_seconds is not None]
    throughputs = [trial.throughput_tokens_per_second for trial in trials]
    return {
        "trials": len(trials),
        "request_latency_median_seconds": statistics.median(latencies),
        "ttft_median_seconds": statistics.median(ttfts) if ttfts else None,
        "throughput_median_tokens_per_second": statistics.median(throughputs),
    }


__all__ = [
    "KVCacheGeometry",
    "NativeIntegrationCapability",
    "NativeMosaicKVUnsupported",
    "PreparedSGLangPrompt",
    "SGLangFullKVModel",
    "SGLangHTTPTrialRunner",
    "SGLangRuntimeError",
    "SGLangRuntimeOptions",
    "SGLangTrialMeasurement",
    "build_server_command",
    "kv_cache_geometry",
    "native_integration_capability",
    "prepare_sglang_prompt",
    "require_native_mosaickv_support",
    "sglang_trial_summary",
]
