"""Version-pinned vLLM FullKV measurement runtime and native safety gate.

The module deliberately imports vLLM only while constructing a real engine so
CPU-only diagnostics and unit tests do not require the optional environment.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import math
import os
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

from mosaickv.adapters.huggingface.qwen2_5_vl import (
    QWEN25_VL_MAX_PIXELS,
    QWEN25_VL_MIN_PIXELS,
)
from mosaickv.backends.native_blocks import native_audit_path
from mosaickv.backends.native_selector_plan import (
    NativeSelectorPlanCatalog,
    load_native_selector_plan_catalog,
)
from mosaickv.config import RunConfig
from mosaickv.evaluation.messages import MultimodalMessage
from mosaickv.evaluation.model import EvaluationRequest, GenerationMetrics, ModelGeneration
from mosaickv.generation import QualityDecode, decode_quality_output, terminal_token_ids
from mosaickv.types import JsonObject, JsonValue, Precision

_AUDITED_VLLM_VERSION = "0.10.2"
_SUPPORTED_MODELS: dict[str, bool] = {
    "Qwen/Qwen2.5-VL-3B-Instruct": True,
    "Qwen/Qwen2.5-VL-7B-Instruct": True,
    "llava-hf/llava-1.5-7b-hf": False,
    "llava-hf/llava-onevision-qwen2-0.5b-ov-hf": True,
}


class VLLMRuntimeError(RuntimeError):
    """Raised when a vLLM measurement cannot preserve the runtime contract."""


class NativeMosaicKVUnsupported(VLLMRuntimeError):
    """Raised before model loading when the native mutation contract is absent."""


@dataclass(frozen=True, slots=True)
class NativeIntegrationCapability:
    """Machine-readable verdict for the installed native integration seam."""

    vllm_version: str
    supported: bool
    feature: str
    reason_code: str
    blocker_document: str
    inspected_symbols: tuple[str, ...]

    def to_json_object(self) -> JsonObject:
        return cast("JsonObject", asdict(self))


def native_integration_capability(vllm_version: str) -> NativeIntegrationCapability:
    """Return the version-pinned native whole-block capability."""

    return NativeIntegrationCapability(
        vllm_version=vllm_version,
        supported=vllm_version == _AUDITED_VLLM_VERSION,
        feature="request_wide_whole_block_compaction_with_logical_positions",
        reason_code=(
            "repository_plugin_transaction_v1"
            if vllm_version == _AUDITED_VLLM_VERSION
            else "unaudited_vllm_version"
        ),
        blocker_document="docs/vllm_native_blocker.md",
        inspected_symbols=(
            "vllm.engine.async_llm_engine.AsyncLLMEngine.generate",
            "vllm.v1.worker.gpu_model_runner.GPUModelRunner._prepare_inputs",
            "vllm.v1.core.kv_cache_manager.KVCacheManager",
            "vllm.v1.core.kv_cache_utils.KVCacheBlock",
            "vllm.v1.attention.backends.utils.CommonAttentionMetadata",
            "vllm.v1.attention.backends.flash_attn.FlashAttentionMetadata",
            "mosaickv.backends.vllm_native_plugin.register",
        ),
    )


def require_native_mosaickv_support(
    *,
    enabled: bool,
    vllm_version: str,
    enforce_eager: bool,
    attention_backend: str,
) -> None:
    """Validate Stage B and reject unsupported execution before loading weights."""

    if not enabled:
        return
    if not enforce_eager or attention_backend != "eager":
        raise NativeMosaicKVUnsupported(
            "--enable-mosaickv requires --attention-backend eager, enforce_eager=True, "
            "and CUDA graphs disabled"
        )
    capability = native_integration_capability(vllm_version)
    if not capability.supported:
        raise NativeMosaicKVUnsupported(
            "native MosaicKV is unsupported for vLLM "
            f"{vllm_version}: {capability.reason_code}; see {capability.blocker_document}. "
            "No simulated MosaicKV row was emitted."
        )


@dataclass(frozen=True, slots=True)
class VLLMRuntimeOptions:
    """vLLM engine and measurement controls not represented in RunConfig."""

    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int | None = None
    cache_probe_repeats: int = 2
    local_files_only: bool = False
    enable_prefix_caching: bool = False
    enable_mosaickv: bool = False
    native_selector_plan: str | None = None
    native_selector_plan_sha256: str | None = None
    native_selector_task: str | None = None

    def __post_init__(self) -> None:
        if self.tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be >= 1")
        if not math.isfinite(self.gpu_memory_utilization) or not (
            0 < self.gpu_memory_utilization <= 1
        ):
            raise ValueError("gpu_memory_utilization must be finite and in (0, 1]")
        if self.max_model_len is not None and self.max_model_len < 2:
            raise ValueError("max_model_len must be >= 2 or null")
        if self.cache_probe_repeats < 1:
            raise ValueError("cache_probe_repeats must be >= 1")
        plan_values = (
            self.native_selector_plan,
            self.native_selector_plan_sha256,
            self.native_selector_task,
        )
        if any(value is not None for value in plan_values) != all(
            value is not None for value in plan_values
        ):
            raise ValueError(
                "native selector plan, SHA-256, and task must be supplied together"
            )
        if self.native_selector_plan is not None and not self.enable_mosaickv:
            raise ValueError("native selector plans require enable_mosaickv=True")


@dataclass(frozen=True, slots=True)
class VLLMKVCacheGeometry:
    """Exact TP=1 decoder KV bytes per active token position."""

    layers: int
    kv_heads: int
    head_dim: int
    dtype_bytes: int

    @property
    def bytes_per_position(self) -> int:
        return 2 * self.layers * self.kv_heads * self.head_dim * self.dtype_bytes

    def active_bytes(self, positions: int) -> int:
        if positions < 0:
            raise ValueError("positions must be >= 0")
        return positions * self.bytes_per_position

    def to_json_object(self) -> JsonObject:
        return cast("JsonObject", asdict(self))


def _vllm_kv_cache_geometry(model_config: Any, precision: Precision) -> VLLMKVCacheGeometry:
    text_config = getattr(model_config, "text_config", model_config)
    layers = int(text_config.num_hidden_layers)
    heads = int(text_config.num_attention_heads)
    kv_heads = int(getattr(text_config, "num_key_value_heads", heads))
    hidden = int(text_config.hidden_size)
    if hidden % heads:
        raise VLLMRuntimeError("model hidden size is not divisible by attention heads")
    dtype_bytes = {
        Precision.FP16: 2,
        Precision.BF16: 2,
        Precision.FP32: 4,
    }.get(precision)
    if dtype_bytes is None:
        raise VLLMRuntimeError("native vLLM supports fp16, bf16, or fp32 KV geometry")
    return VLLMKVCacheGeometry(layers, kv_heads, hidden // heads, dtype_bytes)


@dataclass(frozen=True, slots=True)
class PreparedVLLMPrompt:
    """Rendered prompt and unchanged media payloads accepted by vLLM."""

    engine_prompt: dict[str, Any]
    rendered_text: str
    prompt_sha256: str
    media_sha256: str


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        if name not in value:
            raise ValueError(f"missing required field {name!r}")
        return value[name]
    if not hasattr(value, name):
        raise ValueError(f"missing required field {name!r}")
    return getattr(value, name)


def _optional_field(value: Any, name: str) -> Any | None:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _media_digest(items: Sequence[tuple[str, Any]]) -> str:
    digest = hashlib.sha256()
    for kind, payload in items:
        digest.update(kind.encode("utf-8"))
        digest.update(b"\0")
        if isinstance(payload, bytes):
            digest.update(payload)
        elif isinstance(payload, str):
            path = Path(payload)
            if path.is_file():
                digest.update(path.read_bytes())
            else:
                digest.update(payload.encode("utf-8"))
        elif hasattr(payload, "tobytes"):
            digest.update(str(getattr(payload, "mode", "")).encode("utf-8"))
            digest.update(str(getattr(payload, "size", "")).encode("utf-8"))
            digest.update(cast("bytes", payload.tobytes()))
        else:
            digest.update(repr(payload).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _normalize_messages(
    messages: Sequence[MultimodalMessage],
) -> tuple[list[dict[str, Any]], list[tuple[str, Any]]]:
    if not messages:
        raise ValueError("messages must be non-empty")
    chat: list[dict[str, Any]] = []
    media: list[tuple[str, Any]] = []
    for message in messages:
        role = _field(message, "role")
        content = _field(message, "content")
        if not isinstance(role, str) or not role:
            raise ValueError("each message must have a non-empty role")
        if isinstance(content, str):
            raw_parts: Sequence[Any] = (content,)
        elif isinstance(content, Sequence):
            raw_parts = content
        else:
            raise ValueError("message content must be text or a sequence")
        parts: list[dict[str, Any]] = []
        for raw_part in raw_parts:
            if isinstance(raw_part, str):
                parts.append({"type": "text", "text": raw_part})
                continue
            part_type = str(_field(raw_part, "type"))
            if part_type == "text":
                text = _optional_field(raw_part, "text")
                if text is None:
                    text = _optional_field(raw_part, "value")
                if not isinstance(text, str) or not text:
                    raise ValueError("text parts require non-empty text")
                parts.append({"type": "text", "text": text})
            elif part_type in {"image", "video"}:
                payload = _optional_field(raw_part, part_type)
                if payload is None:
                    payload = _optional_field(raw_part, "payload")
                if payload is None:
                    payload = _optional_field(raw_part, "value")
                if payload is None:
                    raise ValueError(f"{part_type} parts require a payload")
                media.append((part_type, payload))
                parts.append({"type": part_type})
            else:
                raise ValueError(f"unsupported message part type: {part_type!r}")
        chat.append({"role": role, "content": parts})
    return chat, media


def _llava15_fallback_prompt(chat: Sequence[dict[str, Any]]) -> str:
    pieces: list[str] = []
    for message in chat:
        content = "".join(
            "<image>\n" if part["type"] == "image" else str(part["text"])
            for part in message["content"]
        )
        role = message["role"]
        if role == "system":
            pieces.append(content)
        elif role == "user":
            pieces.append(f"USER: {content}")
        elif role == "assistant":
            pieces.append(f"ASSISTANT: {content}")
        else:
            raise ValueError(f"LLaVA-1.5 does not support role {role!r}")
    if not chat or chat[-1]["role"] != "assistant":
        pieces.append("ASSISTANT:")
    return " ".join(pieces)


def prepare_vllm_prompt(
    processor: Any,
    model_id: str,
    messages: Sequence[MultimodalMessage],
) -> PreparedVLLMPrompt:
    """Use the same chat-template inputs and raw media as the HF harness."""

    chat, media = _normalize_messages(messages)
    renderer = getattr(processor, "apply_chat_template", None)
    if not getattr(processor, "chat_template", None):
        tokenizer = getattr(processor, "tokenizer", None)
        tokenizer_renderer = getattr(tokenizer, "apply_chat_template", None)
        if getattr(tokenizer, "chat_template", None) and callable(tokenizer_renderer):
            renderer = tokenizer_renderer
    if callable(renderer):
        rendered = renderer(chat, tokenize=False, add_generation_prompt=True)
    elif model_id == "llava-hf/llava-1.5-7b-hf":
        rendered = _llava15_fallback_prompt(chat)
    else:
        raise VLLMRuntimeError("processor and tokenizer expose no chat template")
    if not isinstance(rendered, str) or not rendered:
        raise VLLMRuntimeError("chat template returned an empty prompt")

    by_modality: dict[str, list[Any]] = {}
    for kind, payload in media:
        by_modality.setdefault(kind, []).append(payload)
    multi_modal_data: dict[str, Any] = {
        kind: values[0] if len(values) == 1 else values for kind, values in by_modality.items()
    }
    engine_prompt: dict[str, Any] = {"prompt": rendered}
    if multi_modal_data:
        engine_prompt["multi_modal_data"] = multi_modal_data
    return PreparedVLLMPrompt(
        engine_prompt=engine_prompt,
        rendered_text=rendered,
        prompt_sha256=hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        media_sha256=_media_digest(media),
    )


def _resolve_model_source(
    model_id: str,
    revision: str,
    *,
    local_files_only: bool,
) -> str:
    """Resolve an offline checkpoint once and give both loaders its snapshot path.

    Transformers probes optional processor files while constructing some
    multimodal processors. Resolving the immutable snapshot first prevents an
    absent optional file from being reported as a network/cache failure, and
    ensures vLLM and Transformers consume the identical local checkpoint.
    """

    if not local_files_only:
        return model_id
    cache_dir = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME")
    if not cache_dir:
        raise VLLMRuntimeError("offline vLLM loading requires HF_HUB_CACHE or HF_HOME")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise VLLMRuntimeError("offline vLLM loading requires huggingface_hub") from error
    try:
        snapshot = snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=True,
        )
    except Exception as error:
        # huggingface_hub>=1 reports an IncompleteSnapshotError when repository
        # documentation (for example README.md or .gitattributes) was never
        # fetched, even if every runtime artifact is present. Accept only the
        # exact immutable snapshot and validate all indexed weight shards plus
        # tokenizer/processor configuration before bypassing that docs-only
        # incompleteness.
        repo_directory = Path(cache_dir) / f"models--{model_id.replace('/', '--')}"
        candidate = repo_directory / "snapshots" / revision
        if not _runtime_complete_snapshot(candidate):
            raise VLLMRuntimeError(
                f"pinned model snapshot is unavailable offline: {model_id}@{revision}"
            ) from error
        snapshot = str(candidate)
    snapshot_path = Path(snapshot).resolve()
    if not snapshot_path.is_dir():
        raise VLLMRuntimeError(f"resolved model snapshot is not a directory: {snapshot_path}")
    return str(snapshot_path)


def _runtime_complete_snapshot(snapshot: Path) -> bool:
    """Validate the runtime subset of an immutable HF cache snapshot."""

    if not snapshot.is_dir():
        return False
    required = ("config.json", "tokenizer_config.json", "preprocessor_config.json")
    if any(not (snapshot / name).is_file() for name in required):
        return False
    tokenizer_present = (snapshot / "tokenizer.json").is_file() or (
        (snapshot / "vocab.json").is_file() and (snapshot / "merges.txt").is_file()
    )
    if not tokenizer_present:
        return False
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = snapshot / index_name
        if not index_path.is_file():
            continue
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = payload["weight_map"]
        except (KeyError, OSError, TypeError, ValueError):
            return False
        if not isinstance(weight_map, dict) or not weight_map:
            return False
        shard_names = {name for name in weight_map.values() if isinstance(name, str)}
        return len(shard_names) > 0 and all(
            (snapshot / shard_name).is_file() for shard_name in shard_names
        )
    return any((snapshot / name).is_file() for name in ("model.safetensors", "pytorch_model.bin"))


@dataclass(frozen=True, slots=True)
class VLLMTrialMeasurement:
    """One raw, non-aggregated vLLM request observation."""

    request_id: str
    answer: str
    token_ids: tuple[int, ...]
    prompt_tokens: int | None
    generated_tokens: int
    ttft_seconds: float | None
    request_latency_seconds: float
    decode_seconds: float | None
    inter_token_latencies_seconds: tuple[float, ...]
    token_timestamps_seconds: tuple[float, ...]
    throughput_tokens_per_second: float
    decode_throughput_tokens_per_second: float | None
    num_cached_tokens: int | None
    prefix_cache_hit_rate: float | None
    mm_cache_queries: int | None
    mm_cache_hits: int | None
    mm_cache_hit_rate: float | None
    engine_prefill_seconds: float | None
    engine_decode_seconds: float | None
    engine_ttft_seconds: float | None
    engine_inter_token_latencies_seconds: tuple[float, ...]
    gpu_memory_source: str | None
    gpu_memory_baseline_bytes: int | None
    gpu_memory_peak_bytes: int | None
    gpu_memory_peak_delta_bytes: int | None
    finish_reason: str | None
    engine_timing_source: str = "unspecified"
    native_audit: JsonObject | None = None

    def to_json_object(self) -> JsonObject:
        return cast("JsonObject", asdict(self))


class VLLMTrialRunner(Protocol):
    """Injectable engine boundary used by the FullKV wrapper."""

    vllm_version: str
    engine_metadata: JsonObject

    def run(
        self,
        prompt: dict[str, Any],
        generation: Mapping[str, object],
        request_id: str,
    ) -> VLLMTrialMeasurement: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class _StatSnapshot:
    ttfts: tuple[float, ...] = ()
    itls: tuple[float, ...] = ()
    prefill_seconds: tuple[float, ...] = ()
    decode_seconds: tuple[float, ...] = ()


class _VLLMStatCollector:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot = _StatSnapshot()

    def reset(self) -> None:
        with self._lock:
            self._snapshot = _StatSnapshot()

    def record_v0(self, stats: Any) -> None:
        """Record the legacy vLLM V0 ``Stats`` payload."""

        with self._lock:
            current = self._snapshot
            self._snapshot = _StatSnapshot(
                ttfts=current.ttfts
                + tuple(float(value) for value in stats.time_to_first_tokens_iter),
                itls=current.itls
                + tuple(float(value) for value in stats.time_per_output_tokens_iter),
                prefill_seconds=current.prefill_seconds
                + tuple(float(value) for value in stats.time_prefill_requests),
                decode_seconds=current.decode_seconds
                + tuple(float(value) for value in stats.time_decode_requests),
            )

    def record_v1(self, iteration_stats: Any) -> None:
        """Record the audited vLLM 0.10.2 V1 ``IterationStats`` payload."""

        finished = tuple(iteration_stats.finished_requests)
        with self._lock:
            current = self._snapshot
            self._snapshot = _StatSnapshot(
                ttfts=current.ttfts
                + tuple(float(value) for value in iteration_stats.time_to_first_tokens_iter),
                itls=current.itls
                + tuple(float(value) for value in iteration_stats.inter_token_latencies_iter),
                prefill_seconds=current.prefill_seconds
                + tuple(float(value.prefill_time) for value in finished),
                decode_seconds=current.decode_seconds
                + tuple(float(value.decode_time) for value in finished),
            )

    def snapshot(self) -> _StatSnapshot:
        with self._lock:
            return self._snapshot


class _VLLMStatLogger:
    """Minimal legacy V0 ``StatLoggerBase``-compatible collector."""

    def __init__(self, collector: _VLLMStatCollector) -> None:
        self.collector = collector

    def log(self, stats: Any) -> None:
        self.collector.record_v0(stats)

    def info(self, type: str, obj: Any) -> None:
        del type, obj


class _VLLMV1StatLogger:
    """Version-gated vLLM 0.10.2 V1 iteration-stat collector."""

    def __init__(
        self,
        vllm_config: Any,
        engine_index: int = 0,
        *,
        collector: _VLLMStatCollector,
    ) -> None:
        del vllm_config
        self.engine_index = engine_index
        self.collector = collector

    def record(
        self,
        scheduler_stats: Any,
        iteration_stats: Any,
        engine_idx: int = 0,
    ) -> None:
        del scheduler_stats, engine_idx
        if iteration_stats is not None:
            self.collector.record_v1(iteration_stats)

    def log_engine_initialized(self) -> None:
        return None

    def log(self) -> None:
        return None


def _vllm_mm_cache_snapshot(engine: Any | None = None) -> tuple[int, int] | None:
    """Read audited vLLM 0.10.2 multimodal processor-cache counters.

    This private boundary is deliberately version-gated by the engine startup
    check.  It is measurement-only and never mutates the cache.
    """

    if engine is not None:
        try:
            cache = engine.processor.mm_processor_cache._cache
            cache_info = cache.stat()
            return int(cache_info.total), int(cache_info.hits)
        except (AttributeError, TypeError, ValueError):
            pass
    try:
        from vllm.multimodal import MULTIMODAL_REGISTRY

        registry = cast("Any", MULTIMODAL_REGISTRY)
        processing_cache = registry._processing_cache
        cache = processing_cache._cache
        cache_info = cache.stat()
        return int(cache_info.total), int(cache_info.hits)
    except (AttributeError, ImportError, TypeError, ValueError):
        return None


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


def _query_process_gpu_bytes() -> tuple[int | None, str | None]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None, None
    if completed.returncode != 0:
        return None, None
    pids = _descendant_pids(os.getpid())
    total_mib = 0
    observed = False
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            pid, used = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        if pid in pids:
            total_mib += used
            observed = True
    return (total_mib * 1024 * 1024 if observed else None), "nvidia-smi process memory"


class _GPUMemorySampler:
    def __init__(self, interval_seconds: float = 0.02) -> None:
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.baseline_bytes: int | None = None
        self.peak_bytes: int | None = None
        self.source: str | None = None

    def start(self) -> None:
        self.baseline_bytes, self.source = _query_process_gpu_bytes()
        self.peak_bytes = self.baseline_bytes

        def sample() -> None:
            while not self._stop.wait(self.interval_seconds):
                value, source = _query_process_gpu_bytes()
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
        value, source = _query_process_gpu_bytes()
        if source is not None:
            self.source = source
        if value is not None and (self.peak_bytes is None or value > self.peak_bytes):
            self.peak_bytes = value


def _one_or_none(values: Sequence[float]) -> float | None:
    return float(values[-1]) if values else None


def _generation_float(generation: Mapping[str, object], name: str) -> float:
    value = generation.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VLLMRuntimeError(f"generation.{name} must be numeric")
    return float(value)


def _generation_int(generation: Mapping[str, object], name: str) -> int:
    value = generation.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise VLLMRuntimeError(f"generation.{name} must be an integer")
    return value


class AsyncVLLMTrialRunner:
    """Persistent public AsyncLLMEngine streaming wrapper for vLLM 0.10.2."""

    def __init__(
        self,
        config: RunConfig,
        options: VLLMRuntimeOptions,
        *,
        model_source: str | None = None,
        native_audit_directory: str | Path | None = None,
    ) -> None:
        self.config = config
        self.options = options
        self._model_source = model_source or config.model.id
        self._native_audit_directory = (
            Path(native_audit_directory) if native_audit_directory is not None else None
        )
        self._collector = _VLLMStatCollector()
        self._measurement_guard = threading.Lock()
        self._active_measurements = 0
        self._concurrency_epoch = 0
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None
        self._engine: Any = None
        self.vllm_version = "not_loaded"
        self.engine_metadata: JsonObject = {}
        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=1800):
            raise VLLMRuntimeError("timed out while constructing the vLLM engine")
        if self._startup_error is not None:
            raise VLLMRuntimeError(f"vLLM engine initialization failed: {self._startup_error}")

    def _thread_main(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._engine = self._build_engine()
        except BaseException as error:
            self._startup_error = error
            self._ready.set()
            return
        self._ready.set()
        self._loop.run_forever()

    def _build_engine(self) -> Any:
        # V1 is required: vLLM 0.10.2 disables multimodal prefix caching in
        # V0.  Set this before importing vLLM so the selected engine and the
        # recorded metadata cannot diverge.
        os.environ["VLLM_USE_V1"] = "1"
        # The Stage-A contract is deterministic greedy decoding. vLLM's
        # FlashInfer sampler is unnecessary for that path, while its startup
        # profile JIT-compiles stochastic top-k/top-p kernels even when they
        # will never be used. Select vLLM's native sampler explicitly; parity
        # tests below remain the authority for output equivalence.
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        use_vit_sdpa = self.config.model.id == "llava-hf/llava-1.5-7b-hf"
        if self.options.enable_mosaickv:
            if self.options.enable_prefix_caching:
                raise VLLMRuntimeError("native MosaicKV requires vLLM prefix caching disabled")
            if self._native_audit_directory is None:
                raise VLLMRuntimeError("native MosaicKV requires an audit directory")
            os.environ["VLLM_PLUGINS"] = "mosaickv_native"
            os.environ["MOSAICKV_NATIVE_VLLM"] = "1"
            os.environ["MOSAICKV_NATIVE_RETENTION_RATIO"] = str(self.config.cache.retention_ratio)
            os.environ["MOSAICKV_NATIVE_BLOCK_SIZE"] = str(self.config.cache.block_size)
            os.environ["MOSAICKV_NATIVE_AUDIT_DIR"] = str(self._native_audit_directory.resolve())
        else:
            # The evaluation process may construct a native engine and a
            # FullKV engine sequentially.  Never let the process-global
            # plugin controls leak into the reference engine.
            if (
                not use_vit_sdpa
                and os.environ.get("VLLM_PLUGINS") == "mosaickv_native"
            ):
                os.environ.pop("VLLM_PLUGINS", None)
            for variable in (
                "MOSAICKV_NATIVE_VLLM",
                "MOSAICKV_NATIVE_RETENTION_RATIO",
                "MOSAICKV_NATIVE_BLOCK_SIZE",
                "MOSAICKV_NATIVE_AUDIT_DIR",
            ):
                os.environ.pop(variable, None)
        if use_vit_sdpa:
            # LLaVA's CLIP encoder is cache-free. Route only its existing
            # vLLM MultiHeadAttention implementation to Torch SDPA; decoder
            # paged attention remains unchanged.
            os.environ["VLLM_PLUGINS"] = "mosaickv_native"
            os.environ["MOSAICKV_VLLM_VIT_SDPA"] = "1"
        else:
            os.environ.pop("MOSAICKV_VLLM_VIT_SDPA", None)
        # FlashInfer JIT compilation is performed in vLLM's spawned engine
        # process and invokes the locked ``ninja`` executable by name. Direct
        # execution of ``<venv>/bin/python`` does not activate the environment,
        # so explicitly expose its sibling executables to every child process.
        interpreter_bin = str(Path(sys.executable).parent)
        os.environ["PATH"] = os.pathsep.join(
            part for part in (interpreter_bin, os.environ.get("PATH", "")) if part
        )
        try:
            import vllm
            from vllm import AsyncEngineArgs, AsyncLLMEngine
        except ImportError as error:
            raise VLLMRuntimeError("vLLM backend requires the common environment") from error
        self.vllm_version = str(vllm.__version__)
        if self.vllm_version != _AUDITED_VLLM_VERSION:
            raise VLLMRuntimeError(
                f"vLLM {self.vllm_version} is unaudited; expected {_AUDITED_VLLM_VERSION}"
            )
        require_native_mosaickv_support(
            enabled=self.options.enable_mosaickv,
            vllm_version=self.vllm_version,
            enforce_eager=True,
            attention_backend=self.config.execution.attention_implementation,
        )
        if self.options.local_files_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        dtype = {
            Precision.FP16: "float16",
            Precision.BF16: "bfloat16",
            Precision.FP32: "float32",
        }.get(self.config.model.precision)
        if dtype is None:
            raise VLLMRuntimeError("vLLM FullKV supports fp16, bf16, or fp32")
        cache_dir = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME")
        kwargs: dict[str, Any] = {
            "model": self._model_source,
            "dtype": dtype,
            "seed": self.config.execution.seed,
            "tensor_parallel_size": self.options.tensor_parallel_size,
            "gpu_memory_utilization": self.options.gpu_memory_utilization,
            "block_size": self.config.cache.block_size,
            # Correctness/reference runs default to an uncached prompt.  The
            # opt-in repeat-request cache probe remains available, but vLLM
            # 0.10.2 changed Qwen2.5-VL greedy tokens between cold and cached
            # requests in the controlled A100 probe.  Never hide that failed
            # parity by enabling prefix caching in a reference row.
            "enable_prefix_caching": self.options.enable_prefix_caching,
            "enforce_eager": True,
            "disable_log_stats": True,
            "enable_log_requests": False,
        }
        if self.config.model.id.startswith("Qwen/Qwen2.5-VL-"):
            kwargs["mm_processor_kwargs"] = {
                "min_pixels": QWEN25_VL_MIN_PIXELS,
                "max_pixels": QWEN25_VL_MAX_PIXELS,
            }
        if self._model_source == self.config.model.id:
            kwargs["revision"] = self.config.model.revision
            kwargs["tokenizer_revision"] = self.config.model.revision
        if cache_dir:
            kwargs["download_dir"] = cache_dir
        if self.options.max_model_len is not None:
            kwargs["max_model_len"] = self.options.max_model_len
        engine_args = AsyncEngineArgs(**kwargs)

        stat_logger_factory = functools.partial(
            _VLLMV1StatLogger,
            collector=self._collector,
        )
        engine = AsyncLLMEngine.from_engine_args(
            engine_args,
            stat_loggers=cast("Any", [stat_logger_factory]),
        )
        self.engine_metadata = {
            "vllm_version": self.vllm_version,
            "execution_mode": "v1_eager",
            "cuda_graph": False,
            "attention_backend": os.environ.get("VLLM_ATTENTION_BACKEND", "auto"),
            "sampling_backend": "vllm_native_greedy",
            "block_size": self.config.cache.block_size,
            "prefix_caching": self.options.enable_prefix_caching,
            "native_mosaickv": self.options.enable_mosaickv,
            "native_patch": (
                "mosaickv.backends.vllm_native_plugin:register"
                if self.options.enable_mosaickv
                else "not_used"
            ),
            "tensor_parallel_size": self.options.tensor_parallel_size,
            "gpu_memory_utilization": self.options.gpu_memory_utilization,
            "max_model_len": self.options.max_model_len,
            "model_source": (
                "local_snapshot"
                if self._model_source != self.config.model.id
                else "model_id_and_revision"
            ),
            "vit_attention_backend": "TORCH_SDPA" if use_vit_sdpa else "vllm_default",
            "mm_processor_kwargs": kwargs.get("mm_processor_kwargs", {}),
        }
        return engine

    async def _run_async(
        self,
        prompt: dict[str, Any],
        generation: Mapping[str, object],
        request_id: str,
    ) -> VLLMTrialMeasurement:
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind

        sampling = SamplingParams(
            temperature=_generation_float(generation, "temperature"),
            top_p=_generation_float(generation, "top_p"),
            max_tokens=_generation_int(generation, "max_new_tokens"),
            seed=_generation_int(generation, "seed"),
            # The owned HF correctness loop executes exactly max_new_tokens
            # and only strips terminal tokens for quality scoring afterwards.
            # Match that timed-output contract instead of letting vLLM stop
            # its cache/timing trajectory at the first EOS token.
            ignore_eos=True,
            output_kind=RequestOutputKind.DELTA,
            detokenize=True,
            skip_special_tokens=True,
            extra_args=(
                {
                    "mosaickv_native": {
                        "retention_ratio": self.config.cache.retention_ratio,
                        "block_size": self.config.cache.block_size,
                        "scope": "request_wide_token_blocks",
                        **(
                            {
                                "selected_blocks": generation["native_selected_blocks"],
                                "mandatory_blocks": generation["native_mandatory_blocks"],
                                "selected_block_count": len(
                                    cast(
                                        "list[int]",
                                        generation["native_selected_blocks"],
                                    )
                                ),
                                "logical_positions": generation["native_logical_positions"],
                                "policy": generation["native_policy"],
                                "method": generation["native_method"],
                                "catalog_sha256": generation["native_catalog_sha256"],
                                "selector_planning_seconds": generation[
                                    "native_selector_planning_seconds"
                                ],
                                "source_trace_sha256": generation[
                                    "native_source_trace_sha256"
                                ],
                            }
                            if "native_selected_blocks" in generation
                            else {}
                        ),
                    }
                }
                if self.options.enable_mosaickv
                else None
            ),
        )
        with self._measurement_guard:
            exclusive_start = self._active_measurements == 0
            self._active_measurements += 1
            if self._active_measurements > 1:
                self._concurrency_epoch += 1
            request_epoch = self._concurrency_epoch
            if exclusive_start:
                self._collector.reset()
        mm_cache_before = _vllm_mm_cache_snapshot(self._engine)
        memory = _GPUMemorySampler()
        memory.start()
        started = time.perf_counter()
        token_times: list[float] = []
        token_ids: list[int] = []
        text_parts: list[str] = []
        prompt_tokens: int | None = None
        cached_tokens: int | None = None
        finish_reason: str | None = None
        output_metrics: Any = None
        try:
            async for output in self._engine.generate(prompt, sampling, request_id):
                now = time.perf_counter()
                if output.prompt_token_ids is not None:
                    prompt_tokens = len(output.prompt_token_ids)
                if output.num_cached_tokens is not None:
                    cached_tokens = int(output.num_cached_tokens)
                output_metrics = output.metrics
                for completion in output.outputs:
                    ids = [int(value) for value in completion.token_ids]
                    token_ids.extend(ids)
                    token_times.extend([now - started] * len(ids))
                    if completion.text:
                        text_parts.append(str(completion.text))
                    if completion.finish_reason is not None:
                        finish_reason = str(completion.finish_reason)
        finally:
            finished = time.perf_counter()
            memory.stop()
            with self._measurement_guard:
                exclusive_measurement = exclusive_start and request_epoch == self._concurrency_epoch
                self._active_measurements -= 1
        latency = max(0.0, finished - started)
        ttft = token_times[0] if token_times else None
        itls = tuple(max(0.0, current - previous) for previous, current in pairwise(token_times))
        decode = max(0.0, token_times[-1] - token_times[0]) if len(token_times) > 1 else None
        generated = len(token_ids)
        stats = self._collector.snapshot() if exclusive_measurement else _StatSnapshot()
        mm_cache_after = _vllm_mm_cache_snapshot(self._engine)
        mm_queries: int | None = None
        mm_hits: int | None = None
        if exclusive_measurement and mm_cache_before is not None and mm_cache_after is not None:
            mm_queries = max(0, mm_cache_after[0] - mm_cache_before[0])
            mm_hits = max(0, mm_cache_after[1] - mm_cache_before[1])
        prefix_rate: float | None = None
        if cached_tokens is not None and prompt_tokens is not None and prompt_tokens > 0:
            prefix_rate = cached_tokens / prompt_tokens
        mm_rate = (
            mm_hits / mm_queries
            if mm_queries is not None and mm_queries > 0 and mm_hits is not None
            else None
        )
        metrics_ttft: float | None = None
        metrics_prefill: float | None = None
        metrics_decode: float | None = None
        if output_metrics is not None:
            arrival = getattr(output_metrics, "arrival_time", None)
            scheduled = getattr(output_metrics, "first_scheduled_time", None)
            first_token = getattr(output_metrics, "first_token_time", None)
            completed = getattr(output_metrics, "finished_time", None)
            if arrival is not None and first_token is not None:
                metrics_ttft = max(0.0, float(first_token) - float(arrival))
            if scheduled is not None and first_token is not None:
                metrics_prefill = max(0.0, float(first_token) - float(scheduled))
            if first_token is not None and completed is not None:
                metrics_decode = max(0.0, float(completed) - float(first_token))
        engine_prefill = _one_or_none(stats.prefill_seconds) or metrics_prefill or ttft
        engine_decode = _one_or_none(stats.decode_seconds) or metrics_decode or decode
        engine_ttft = (
            metrics_ttft if metrics_ttft is not None else _one_or_none(stats.ttfts) or ttft
        )
        native_audit: JsonObject | None = None
        if self.options.enable_mosaickv:
            assert self._native_audit_directory is not None
            audit_path = native_audit_path(self._native_audit_directory, "vllm", request_id)
            if not audit_path.is_file():
                raise VLLMRuntimeError(
                    f"native vLLM request completed without scheduler audit: {audit_path}"
                )
            payload = json.loads(audit_path.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                raise VLLMRuntimeError("native vLLM scheduler audit is not a JSON object")
            native_audit = cast("JsonObject", dict(payload))
        return VLLMTrialMeasurement(
            request_id=request_id,
            answer="".join(text_parts),
            token_ids=tuple(token_ids),
            prompt_tokens=prompt_tokens,
            generated_tokens=generated,
            ttft_seconds=ttft,
            request_latency_seconds=latency,
            decode_seconds=decode,
            inter_token_latencies_seconds=itls,
            token_timestamps_seconds=tuple(token_times),
            throughput_tokens_per_second=(generated / latency if latency > 0 else 0.0),
            decode_throughput_tokens_per_second=(
                (generated - 1) / decode if decode is not None and decode > 0 else None
            ),
            num_cached_tokens=cached_tokens,
            prefix_cache_hit_rate=prefix_rate,
            mm_cache_queries=mm_queries,
            mm_cache_hits=mm_hits,
            mm_cache_hit_rate=mm_rate,
            engine_prefill_seconds=engine_prefill,
            engine_decode_seconds=engine_decode,
            engine_ttft_seconds=engine_ttft,
            engine_inter_token_latencies_seconds=stats.itls,
            gpu_memory_source=memory.source,
            gpu_memory_baseline_bytes=memory.baseline_bytes,
            gpu_memory_peak_bytes=memory.peak_bytes,
            gpu_memory_peak_delta_bytes=(
                max(0, memory.peak_bytes - memory.baseline_bytes)
                if memory.peak_bytes is not None and memory.baseline_bytes is not None
                else None
            ),
            finish_reason=finish_reason,
            engine_timing_source=(
                "vllm_v1_iteration_stats"
                if stats.prefill_seconds or stats.decode_seconds
                else "client_stream_boundaries_concurrent_safe_fallback"
            ),
            native_audit=native_audit,
        )

    def run(
        self,
        prompt: dict[str, Any],
        generation: Mapping[str, object],
        request_id: str,
    ) -> VLLMTrialMeasurement:
        future = asyncio.run_coroutine_threadsafe(
            self._run_async(prompt, generation, request_id), self._loop
        )
        try:
            return future.result(timeout=1800)
        except TimeoutError as error:
            future.cancel()
            raise VLLMRuntimeError(f"vLLM request {request_id!r} timed out") from error

    async def _shutdown_async(self) -> None:
        if self._engine is not None:
            shutdown = getattr(self._engine, "shutdown", None)
            if shutdown is not None:
                shutdown()
            else:
                self._engine.shutdown_background_loop()
                executor = getattr(getattr(self._engine, "engine", None), "model_executor", None)
                if executor is not None and hasattr(executor, "shutdown"):
                    executor.shutdown()
        # vLLM's synchronous shutdown schedules cancellation of its asyncio
        # handlers. Give those cancellations a loop turn, then await every
        # remaining handler before stopping the owned event loop.
        await asyncio.sleep(0)
        current = asyncio.current_task()
        pending = [task for task in asyncio.all_tasks() if task is not current]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    def close(self) -> None:
        if self._thread.is_alive():
            future = asyncio.run_coroutine_threadsafe(self._shutdown_async(), self._loop)
            try:
                future.result(timeout=30)
            except TimeoutError as error:
                raise VLLMRuntimeError("timed out while shutting down vLLM") from error
            finally:
                self._loop.call_soon_threadsafe(self._loop.stop)
                self._thread.join(timeout=30)
        if self._thread.is_alive():
            raise VLLMRuntimeError("vLLM event-loop thread did not stop")


def _atomic_json(path: Path, payload: JsonObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
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
    return f"{readable[:80]}-{hashlib.sha256(sample_id.encode()).hexdigest()[:12]}.json"


class VLLMFullKVModel:
    """Evaluation adapter for vLLM FullKV or native exact block transport."""

    backend = "vllm"
    method = "full_kv"
    retention_ratio = 1.0

    def __init__(
        self,
        *,
        model_id: str,
        processor: Any,
        runner: VLLMTrialRunner,
        trace_directory: str | Path,
        generation: Mapping[str, object],
        cache_probe_repeats: int = 2,
        method: str = "full_kv",
        retention_ratio: float = 1.0,
        block_size: int = 16,
        cache_geometry: VLLMKVCacheGeometry | None = None,
        native_enabled: bool | None = None,
        selector_plan_catalog: NativeSelectorPlanCatalog | None = None,
    ) -> None:
        if model_id not in _SUPPORTED_MODELS:
            known = ", ".join(sorted(_SUPPORTED_MODELS))
            raise LookupError(f"unsupported vLLM multimodal model {model_id!r}; known: {known}")
        if cache_probe_repeats < 1:
            raise ValueError("cache_probe_repeats must be >= 1")
        self.model_id = model_id
        self.supports_video = _SUPPORTED_MODELS[model_id]
        self.processor = processor
        self.runner = runner
        self.trace_directory = Path(trace_directory)
        self.generation = dict(generation)
        self.cache_probe_repeats = cache_probe_repeats
        self.method = method
        self.retention_ratio = retention_ratio
        self.block_size = block_size
        self.cache_geometry = cache_geometry
        self.native_enabled = method != "full_kv" if native_enabled is None else native_enabled
        self.selector_plan_catalog = selector_plan_catalog
        if self.selector_plan_catalog is not None and not self.native_enabled:
            raise ValueError("selector-plan catalog requires native block compaction")
        self._isolation_anchor: (
            tuple[
                PreparedVLLMPrompt,
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
        options: VLLMRuntimeOptions,
        *,
        trace_directory: str | Path,
    ) -> VLLMFullKVModel:
        if config.execution.backend.value != "vllm":
            raise ValueError("VLLMFullKVModel requires execution.backend='vllm'")
        if options.enable_mosaickv:
            require_native_mosaickv_support(
                enabled=True,
                vllm_version=_AUDITED_VLLM_VERSION,
                enforce_eager=True,
                attention_backend=config.execution.attention_implementation,
            )
        native = options.enable_mosaickv
        supported_native_methods = {
            "mosaickv_exact",
            "prompt_attention_topk",
            "random_kv",
            "lookm_reimpl",
            "prefixkv_reimpl",
            "vl_cache_reimpl",
        }
        if native and config.method.value not in supported_native_methods:
            raise VLLMRuntimeError(
                "native vLLM selector plans support only mosaickv_exact, "
                "prompt_attention_topk, random_kv, lookm_reimpl, "
                "prefixkv_reimpl, or vl_cache_reimpl"
            )
        if native and config.method.value != "mosaickv_exact" and (
            options.native_selector_plan is None
        ):
            raise VLLMRuntimeError("native vLLM baselines require a hash-bound selector plan")
        if not native and (not config.method.is_full_cache or config.cache.retention_ratio != 1.0):
            raise VLLMRuntimeError("vLLM FullKV requires retention_ratio=1.0")
        if config.model.id not in _SUPPORTED_MODELS:
            known = ", ".join(sorted(_SUPPORTED_MODELS))
            raise LookupError(f"unsupported vLLM multimodal model; known: {known}")
        if config.execution.attention_implementation != "eager":
            raise VLLMRuntimeError(
                "the controlled vLLM wrapper currently requires attention_backend='eager' "
                "(vLLM enforce_eager execution with CUDA graphs disabled)"
            )
        try:
            from transformers import AutoConfig, AutoProcessor
        except ImportError as error:
            raise VLLMRuntimeError("vLLM environment must provide transformers") from error
        model_source = _resolve_model_source(
            config.model.id,
            config.model.revision,
            local_files_only=options.local_files_only,
        )
        processor_kwargs: dict[str, Any] = {
            "local_files_only": options.local_files_only,
            # Match the audited HF and SGLang Qwen2.5-VL preprocessing path.
            # Transformers 4.55 switched this checkpoint to a fast processor
            # serving-backend optimization.
            "use_fast": False,
        }
        if model_source == config.model.id:
            processor_kwargs["revision"] = config.model.revision
        processor = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
            model_source, **processor_kwargs
        )
        config_kwargs: dict[str, Any] = {
            "local_files_only": options.local_files_only,
        }
        if model_source == config.model.id:
            config_kwargs["revision"] = config.model.revision
        model_config = AutoConfig.from_pretrained(model_source, **config_kwargs)
        geometry = _vllm_kv_cache_geometry(model_config, config.model.precision)
        selector_catalog = (
            None
            if options.native_selector_plan is None
            else load_native_selector_plan_catalog(
                options.native_selector_plan,
                cast("str", options.native_selector_plan_sha256),
                model_id=config.model.id,
                model_revision=config.model.revision,
                task=cast("str", options.native_selector_task),
                dataset_revision=config.dataset.revision,
                method=config.method.value,
            )
        )
        if selector_catalog is not None:
            for entry in selector_catalog.entries:
                projection = entry.projection
                if projection.block_size != config.cache.block_size:
                    raise VLLMRuntimeError(
                        "selector-plan block size differs from resolved run config"
                    )
        runner = AsyncVLLMTrialRunner(
            config,
            options,
            model_source=model_source,
            native_audit_directory=Path(trace_directory) / "native_audit",
        )
        generation = {
            "max_new_tokens": config.generation.max_new_tokens,
            "temperature": config.generation.temperature,
            "top_p": config.generation.top_p,
            "seed": config.execution.seed,
        }
        return cls(
            model_id=config.model.id,
            processor=processor,
            runner=runner,
            trace_directory=trace_directory,
            generation=generation,
            cache_probe_repeats=options.cache_probe_repeats,
            method=config.method.value,
            retention_ratio=config.cache.retention_ratio,
            block_size=config.cache.block_size,
            cache_geometry=geometry,
            native_enabled=native,
            selector_plan_catalog=selector_catalog,
        )

    def generate(self, request: EvaluationRequest) -> ModelGeneration:
        prepared = prepare_vllm_prompt(self.processor, self.model_id, request.messages)
        # lmms-eval task definitions carry their own generation defaults.  They
        # are useful when lmms owns generation, but this backend is paired with
        # the HF runtime under the resolved MosaicKV RunConfig.  Letting task
        # kwargs override that config changes output length across tasks (for
        # example, DocVQA requests 32 tokens) and invalidates the controlled
        # comparison.  The immutable run configuration is therefore the sole
        # authority for every behavior-affecting sampling parameter.
        generation = dict(self.generation)
        ignored_task_generation_keys = sorted(request.generation_kwargs)
        selector_catalog = self.selector_plan_catalog
        selector_entry = (
            None if selector_catalog is None else selector_catalog.entry(request.sample_id)
        )
        if selector_entry is not None:
            assert selector_catalog is not None
            projection = selector_entry.projection
            generation.update(
                {
                    "native_selected_blocks": list(projection.selected_blocks),
                    "native_mandatory_blocks": list(projection.mandatory_blocks),
                    "native_logical_positions": projection.logical_positions,
                    "native_block_size": projection.block_size,
                    "native_retention_ratio": projection.retention_ratio,
                    "native_policy": projection.policy,
                    "native_method": projection.method,
                    "native_catalog_sha256": selector_catalog.catalog_sha256,
                    "native_selector_planning_seconds": selector_entry.selector_planning_seconds,
                    "native_source_trace_sha256": selector_entry.source_trace_sha256,
                }
            )
        effective_retention_ratio = (
            self.retention_ratio
            if selector_entry is None
            else selector_entry.projection.retention_ratio
        )
        trials = tuple(
            self.runner.run(
                prepared.engine_prompt,
                generation,
                f"{request.run_id}-{request.sample_id}-trial-{index}",
            )
            for index in range(self.cache_probe_repeats)
        )
        reference_tokens = trials[0].token_ids
        for index, trial in enumerate(trials[1:], start=1):
            if trial.token_ids != reference_tokens:
                raise VLLMRuntimeError(
                    "deterministic FullKV cache probe changed generated tokens at "
                    f"trial {index}; first={reference_tokens}, current={trial.token_ids}"
                )
        first = trials[0]
        configured_terminal_ids = terminal_token_ids(self.processor)
        quality = (
            decode_quality_output(self.processor, first.token_ids, configured_terminal_ids)
            if configured_terminal_ids
            else QualityDecode(
                text=first.answer,
                timed_token_count=len(first.token_ids),
                scored_token_count=len(first.token_ids),
                quality_completion_tokens=len(first.token_ids),
                terminal_token_id=None,
                terminal_token_index=None,
            )
        )
        native = self.native_enabled
        active_kv_bytes: int | None = None
        logical_prefill_positions: int | None = first.prompt_tokens
        physical_prefill_positions: int | None = None
        compression_seconds = 0.0
        if native:
            if first.native_audit is None or self.cache_geometry is None:
                raise VLLMRuntimeError("native vLLM result omitted audit or KV geometry")
            selection = first.native_audit.get("selection")
            if not isinstance(selection, Mapping):
                raise VLLMRuntimeError("native vLLM audit omitted its selection")
            raw_selected = selection.get("selected_positions")
            raw_selected_blocks = selection.get("selected_blocks")
            raw_compression = first.native_audit.get("compression_seconds")
            if isinstance(raw_selected, bool) or not isinstance(raw_selected, int):
                raise VLLMRuntimeError("native vLLM selected-position count is invalid")
            if not isinstance(raw_selected_blocks, list) or not all(
                isinstance(block, int) and not isinstance(block, bool) and block >= 0
                for block in raw_selected_blocks
            ):
                raise VLLMRuntimeError("native vLLM selected-block list is invalid")
            if isinstance(raw_compression, bool) or not isinstance(raw_compression, (int, float)):
                raise VLLMRuntimeError("native vLLM compression timing is invalid")
            if selector_entry is not None:
                assert selector_catalog is not None
                if (
                    first.native_audit.get("selector_source")
                    != "hash_bound_common_hf_selector_projection"
                    or first.native_audit.get("selector_method") != self.method
                    or first.native_audit.get("selector_catalog_sha256")
                    != selector_catalog.catalog_sha256
                    or first.native_audit.get("source_trace_sha256")
                    != selector_entry.source_trace_sha256
                ):
                    raise VLLMRuntimeError("native vLLM audit differs from the selector plan")
            logical_prefill_positions = raw_selected
            physical_prefill_positions = len(raw_selected_blocks) * self.block_size
            active_kv_bytes = self.cache_geometry.active_bytes(physical_prefill_positions)
            compression_seconds = float(raw_compression) + (
                0.0 if selector_entry is None else selector_entry.selector_planning_seconds
            )
        elif self.cache_geometry is not None and first.prompt_tokens is not None:
            # vLLM allocates whole physical blocks. The paper's primary
            # active-memory field is the live prefill cache before decoding,
            # including backend layout padding but excluding the free pool.
            physical_prefill_positions = (
                (first.prompt_tokens + self.block_size - 1) // self.block_size
            ) * self.block_size
            active_kv_bytes = self.cache_geometry.active_bytes(physical_prefill_positions)
        logical_prefill_kv_bytes = (
            None
            if self.cache_geometry is None or logical_prefill_positions is None
            else self.cache_geometry.active_bytes(logical_prefill_positions)
        )
        packed_padding_kv_bytes = (
            None
            if active_kv_bytes is None or logical_prefill_kv_bytes is None
            else active_kv_bytes - logical_prefill_kv_bytes
        )
        final_active_kv_bytes = (
            None
            if self.cache_geometry is None or logical_prefill_positions is None
            else self.cache_geometry.active_bytes(
                (
                    logical_prefill_positions
                    + max(0, first.generated_tokens - 1)
                    + self.block_size
                    - 1
                )
                // self.block_size
                * self.block_size
            )
        )
        selector_plan_trace: JsonObject | None = None
        if selector_entry is not None:
            assert selector_catalog is not None
            selector_plan_trace = {
                "catalog_path": str(selector_catalog.path),
                "catalog_sha256": selector_catalog.catalog_sha256,
                "source_trace": selector_entry.source_trace,
                "source_trace_sha256": selector_entry.source_trace_sha256,
                "selector_planning_seconds": selector_entry.selector_planning_seconds,
                "projection": selector_entry.projection.to_json_object(),
            }
        trace: JsonObject = {
            "schema_version": 1,
            "measurement_type": (
                "vllm_native_mosaickv"
                if native and self.method == "mosaickv_exact"
                else "vllm_native_baseline" if native else "vllm_fullkv"
            ),
            "backend": "vllm",
            "method": self.method,
            "effective_retention_ratio": effective_retention_ratio,
            "native_mosaickv": native and self.method == "mosaickv_exact",
            "native_block_compaction": native,
            "selector_plan": selector_plan_trace,
            "run_id": request.run_id,
            "sample_id": request.sample_id,
            "model": self.model_id,
            "prompt_sha256": prepared.prompt_sha256,
            "media_sha256": prepared.media_sha256,
            "generation": cast("JsonObject", generation),
            "task_generation_kwargs_policy": {
                "authority": "resolved_run_config",
                "ignored_task_generation_keys": cast(
                    "list[JsonValue]", ignored_task_generation_keys
                ),
            },
            "quality_answer": quality.text,
            "quality_decoding": quality.to_record(),
            "status": "completed",
            "kv_cache_accounting": {
                "boundary": "packed_prefill_active_kv_tensor_storage",
                "logical_prefill_positions": logical_prefill_positions,
                "physical_prefill_positions": physical_prefill_positions,
                "logical_prefill_kv_bytes": logical_prefill_kv_bytes,
                "active_kv_bytes": active_kv_bytes,
                "packed_padding_kv_bytes": packed_padding_kv_bytes,
                "final_active_kv_bytes": final_active_kv_bytes,
                "geometry": (
                    None
                    if self.cache_geometry is None
                    else self.cache_geometry.to_json_object()
                ),
                "includes_live_block_padding": True,
                "excludes_free_block_pool": True,
            },
            "effective_sampling_parameters": {
                "max_new_tokens": cast("int", generation["max_new_tokens"]),
                "temperature": 0.0,
                "top_p": cast("float", generation["top_p"]),
                "seed": cast("int", generation["seed"]),
                "ignore_eos": True,
            },
            "engine": self.runner.engine_metadata,
            "trials": [trial.to_json_object() for trial in trials],
            "native_scope": (
                "hash_bound_common_selector_request_wide_projection"
                if selector_entry is not None
                else "request_wide_whole_token_blocks_transport_gate" if native else "not_used"
            ),
            "cache_measurement": {
                "prefix_cache": "RequestOutput.num_cached_tokens",
                "multimodal_preprocessor_cache": (
                    "AsyncLLM.processor.mm_processor_cache._cache.stat() per-request "
                    "counter deltas (audited vLLM 0.10.2 V1 private read-only API)"
                ),
                "encoder_output_cache": (
                    "repeat-request timing probe only; vLLM 0.10.2 exposes no per-request "
                    "encoder-output-cache hit counter"
                ),
            },
        }
        trace_path = self.trace_directory / request.run_id / _safe_trace_name(request.sample_id)
        _atomic_json(trace_path, trace)
        fingerprint = (prepared.prompt_sha256, prepared.media_sha256)
        self._observed_input_fingerprints.add(fingerprint)
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
            answer=quality.text,
            metrics=GenerationMetrics(
                ttft=(
                    None
                    if first.ttft_seconds is None
                    else first.ttft_seconds
                    + (
                        0.0
                        if selector_entry is None
                        else selector_entry.selector_planning_seconds
                    )
                ),
                prefill_time=first.engine_prefill_seconds,
                compression_time=compression_seconds,
                decode_time=first.decode_seconds,
                end_to_end_time=first.request_latency_seconds
                + (0.0 if selector_entry is None else selector_entry.selector_planning_seconds),
                generated_tokens=first.generated_tokens,
                active_kv_bytes=active_kv_bytes,
                residual_kv_bytes=0,
                peak_gpu_memory=first.gpu_memory_peak_bytes,
                repair_count=0,
                repaired_bytes=0,
            ),
            effective_method=(
                f"{self.method}__vllm_requestwide_projection"
                if selector_entry is not None
                else self.method
            ),
            effective_retention_ratio=effective_retention_ratio,
        )

    def verify_request_isolation(self) -> bool:
        """Run an A-B-A probe against the persistent engine request state."""

        anchor = self._isolation_anchor
        if anchor is None or len(self._observed_input_fingerprints) < 2:
            return False
        prepared, generation, expected_tokens, trace_path, run_id, sample_id = anchor
        request_id = f"{run_id}-{sample_id}-isolation-recheck"
        trial = self.runner.run(prepared.engine_prompt, generation, request_id)
        tokens_match = trial.token_ids == expected_tokens
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        if not isinstance(trace, dict):
            raise VLLMRuntimeError("anchor trace is invalid during request-isolation probe")
        trace["request_isolation"] = {
            "performed": True,
            "intervening_distinct_input_fingerprints": (len(self._observed_input_fingerprints) - 1),
            "request_id": request_id,
            "token_ids_match_anchor": tokens_match,
            "anchor_token_ids": list(expected_tokens),
            "probe_token_ids": list(trial.token_ids),
            "probe_measurement": trial.to_json_object(),
            "prefix_cache_disabled": True,
        }
        _atomic_json(trace_path, cast("JsonObject", trace))
        if not tokens_match:
            raise VLLMRuntimeError(
                "vLLM request-isolation probe changed deterministic token IDs after an "
                "intervening request"
            )
        return True

    def close(self) -> None:
        self.runner.close()


def vllm_trial_summary(trials: Sequence[VLLMTrialMeasurement]) -> JsonObject:
    """Summarize real trials without replacing the raw trace observations."""

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
    "AsyncVLLMTrialRunner",
    "NativeIntegrationCapability",
    "NativeMosaicKVUnsupported",
    "PreparedVLLMPrompt",
    "VLLMFullKVModel",
    "VLLMRuntimeError",
    "VLLMRuntimeOptions",
    "VLLMTrialMeasurement",
    "VLLMTrialRunner",
    "native_integration_capability",
    "prepare_vllm_prompt",
    "require_native_mosaickv_support",
    "vllm_trial_summary",
]
