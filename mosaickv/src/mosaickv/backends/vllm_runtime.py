"""Version-pinned vLLM FullKV measurement runtime and native safety gate.

The module deliberately imports vLLM only while constructing a real engine so
CPU-only diagnostics and unit tests do not require the optional environment.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import statistics
import subprocess
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Protocol, cast

from mosaickv.config import RunConfig
from mosaickv.evaluation.messages import MultimodalMessage
from mosaickv.evaluation.model import EvaluationRequest, GenerationMetrics, ModelGeneration
from mosaickv.types import JsonObject, Precision

_AUDITED_VLLM_VERSION = "0.11.2"
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
    """Return the fail-closed native whole-block capability for a vLLM release."""

    return NativeIntegrationCapability(
        vllm_version=vllm_version,
        supported=False,
        feature="whole_block_selection_with_original_logical_positions",
        reason_code=(
            "audited_0_11_2_missing_sparse_logical_block_table_hook"
            if vllm_version == _AUDITED_VLLM_VERSION
            else "unaudited_vllm_version"
        ),
        blocker_document="docs/vllm_native_blocker.md",
        inspected_symbols=(
            "vllm.v1.worker.gpu_model_runner.GPUModelRunner._prepare_inputs",
            "vllm.v1.worker.block_table.BlockTable.compute_slot_mapping",
            "vllm.v1.core.kv_cache_manager.KVCacheManager.allocate_slots",
            "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend.get_kv_cache_shape",
            "vllm.distributed.kv_transfer.kv_connector.v1.base.KVConnectorBase_V1",
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
    enable_mosaickv: bool = False

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
    prefix_queries: int = 0
    prefix_hits: int = 0
    mm_queries: int = 0
    mm_hits: int = 0
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

    def record(self, scheduler: Any, iteration: Any, mm_cache: Any) -> None:
        with self._lock:
            current = self._snapshot
            prefix_queries = current.prefix_queries
            prefix_hits = current.prefix_hits
            mm_queries = current.mm_queries
            mm_hits = current.mm_hits
            ttfts = list(current.ttfts)
            itls = list(current.itls)
            prefills = list(current.prefill_seconds)
            decodes = list(current.decode_seconds)
            if scheduler is not None:
                prefix = getattr(scheduler, "prefix_cache_stats", None)
                if prefix is not None:
                    prefix_queries += int(getattr(prefix, "queries", 0))
                    prefix_hits += int(getattr(prefix, "hits", 0))
            if mm_cache is not None:
                mm_queries += int(getattr(mm_cache, "queries", 0))
                mm_hits += int(getattr(mm_cache, "hits", 0))
            if iteration is not None:
                ttfts.extend(float(value) for value in iteration.time_to_first_tokens_iter)
                itls.extend(float(value) for value in iteration.inter_token_latencies_iter)
                for finished in iteration.finished_requests:
                    prefills.append(float(finished.prefill_time))
                    decodes.append(float(finished.decode_time))
            self._snapshot = _StatSnapshot(
                prefix_queries,
                prefix_hits,
                mm_queries,
                mm_hits,
                tuple(ttfts),
                tuple(itls),
                tuple(prefills),
                tuple(decodes),
            )

    def snapshot(self) -> _StatSnapshot:
        with self._lock:
            return self._snapshot


class _VLLMStatLogger:
    """vLLM 0.11.2 custom stat-logger interface; explicitly version-pinned."""

    def __init__(self, collector: _VLLMStatCollector) -> None:
        self.collector = collector

    def record(
        self,
        scheduler_stats: Any,
        iteration_stats: Any,
        mm_cache_stats: Any = None,
        engine_idx: int = 0,
    ) -> None:
        del engine_idx
        self.collector.record(scheduler_stats, iteration_stats, mm_cache_stats)

    def log_engine_initialized(self) -> None:
        return None

    def log(self) -> None:
        return None

    def record_sleep_state(self, is_awake: int, level: int) -> None:
        del is_awake, level


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
    """Persistent public AsyncLLMEngine streaming wrapper for vLLM 0.11.2."""

    def __init__(
        self,
        config: RunConfig,
        options: VLLMRuntimeOptions,
        *,
        model_source: str | None = None,
    ) -> None:
        self.config = config
        self.options = options
        self._model_source = model_source or config.model.id
        self._collector = _VLLMStatCollector()
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
        try:
            import vllm
            from vllm import AsyncEngineArgs, AsyncLLMEngine
        except ImportError as error:
            raise VLLMRuntimeError(
                "vLLM backend requires the separately pinned vllm environment"
            ) from error
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
            "enable_prefix_caching": True,
            "enforce_eager": True,
            "disable_log_stats": True,
        }
        if self._model_source == self.config.model.id:
            kwargs["revision"] = self.config.model.revision
            kwargs["tokenizer_revision"] = self.config.model.revision
        if cache_dir:
            kwargs["download_dir"] = cache_dir
        if self.options.max_model_len is not None:
            kwargs["max_model_len"] = self.options.max_model_len
        engine_args = AsyncEngineArgs(**kwargs)

        def stat_logger_factory(_vllm_config: Any, _engine_index: int) -> _VLLMStatLogger:
            return _VLLMStatLogger(self._collector)

        engine = AsyncLLMEngine.from_engine_args(engine_args, stat_loggers=[stat_logger_factory])
        self.engine_metadata = {
            "vllm_version": self.vllm_version,
            "execution_mode": "eager",
            "cuda_graph": False,
            "attention_backend": os.environ.get("VLLM_ATTENTION_BACKEND", "auto"),
            "block_size": self.config.cache.block_size,
            "prefix_caching": True,
            "tensor_parallel_size": self.options.tensor_parallel_size,
            "gpu_memory_utilization": self.options.gpu_memory_utilization,
            "max_model_len": self.options.max_model_len,
            "model_source": (
                "local_snapshot"
                if self._model_source != self.config.model.id
                else "model_id_and_revision"
            ),
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
            output_kind=RequestOutputKind.DELTA,
            detokenize=True,
            skip_special_tokens=True,
        )
        self._collector.reset()
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
        latency = max(0.0, finished - started)
        ttft = token_times[0] if token_times else None
        itls = tuple(max(0.0, current - previous) for previous, current in pairwise(token_times))
        decode = max(0.0, token_times[-1] - token_times[0]) if len(token_times) > 1 else None
        generated = len(token_ids)
        stats = self._collector.snapshot()
        prefix_rate: float | None = None
        if cached_tokens is not None and prompt_tokens is not None and prompt_tokens > 0:
            prefix_rate = cached_tokens / prompt_tokens
        mm_rate = stats.mm_hits / stats.mm_queries if stats.mm_queries else None
        metrics_ttft = getattr(output_metrics, "first_token_latency", None)
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
            mm_cache_queries=stats.mm_queries,
            mm_cache_hits=stats.mm_hits,
            mm_cache_hit_rate=mm_rate,
            engine_prefill_seconds=_one_or_none(stats.prefill_seconds),
            engine_decode_seconds=_one_or_none(stats.decode_seconds),
            engine_ttft_seconds=(
                float(metrics_ttft) if metrics_ttft is not None else _one_or_none(stats.ttfts)
            ),
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
            self._engine.shutdown()
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
    """Backend-neutral evaluation adapter for measured vLLM FullKV requests."""

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
        if not config.method.is_full_cache or config.cache.retention_ratio != 1.0:
            raise VLLMRuntimeError(
                "Stage A emits only vLLM FullKV rows at retention_ratio=1.0; "
                "native MosaicKV is not simulated"
            )
        if config.model.id not in _SUPPORTED_MODELS:
            known = ", ".join(sorted(_SUPPORTED_MODELS))
            raise LookupError(f"unsupported vLLM multimodal model; known: {known}")
        if config.execution.attention_implementation != "eager":
            raise VLLMRuntimeError(
                "the controlled vLLM wrapper currently requires attention_backend='eager' "
                "(vLLM enforce_eager execution with CUDA graphs disabled)"
            )
        try:
            from transformers import AutoProcessor
        except ImportError as error:
            raise VLLMRuntimeError("vLLM environment must provide transformers") from error
        model_source = _resolve_model_source(
            config.model.id,
            config.model.revision,
            local_files_only=options.local_files_only,
        )
        processor_kwargs: dict[str, Any] = {
            "local_files_only": options.local_files_only,
        }
        if model_source == config.model.id:
            processor_kwargs["revision"] = config.model.revision
        processor = AutoProcessor.from_pretrained(model_source, **processor_kwargs)
        runner = AsyncVLLMTrialRunner(config, options, model_source=model_source)
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
        )

    def generate(self, request: EvaluationRequest) -> ModelGeneration:
        prepared = prepare_vllm_prompt(self.processor, self.model_id, request.messages)
        generation = dict(self.generation)
        generation.update(request.generation_kwargs)
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
        trace: JsonObject = {
            "schema_version": 1,
            "measurement_type": "vllm_fullkv",
            "backend": "vllm",
            "method": "full_kv",
            "native_mosaickv": False,
            "run_id": request.run_id,
            "sample_id": request.sample_id,
            "model": self.model_id,
            "prompt_sha256": prepared.prompt_sha256,
            "media_sha256": prepared.media_sha256,
            "generation": cast("JsonObject", generation),
            "engine": self.runner.engine_metadata,
            "trials": [trial.to_json_object() for trial in trials],
            "cache_measurement": {
                "prefix_cache": "RequestOutput.num_cached_tokens",
                "multimodal_preprocessor_cache": "custom StatLogger MultiModalCacheStats",
                "encoder_output_cache": (
                    "repeat-request timing probe only; vLLM 0.11.2 exposes no per-request "
                    "encoder-output-cache hit counter"
                ),
            },
        }
        trace_path = self.trace_directory / request.run_id / _safe_trace_name(request.sample_id)
        _atomic_json(trace_path, trace)
        return ModelGeneration(
            answer=first.answer,
            metrics=GenerationMetrics(
                ttft=first.ttft_seconds,
                prefill_time=first.engine_prefill_seconds,
                compression_time=0.0,
                decode_time=first.decode_seconds,
                end_to_end_time=first.request_latency_seconds,
                generated_tokens=first.generated_tokens,
                active_kv_bytes=None,
                residual_kv_bytes=0,
                peak_gpu_memory=first.gpu_memory_peak_bytes,
                repair_count=0,
                repaired_bytes=0,
            ),
            effective_method="full_kv",
        )

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
