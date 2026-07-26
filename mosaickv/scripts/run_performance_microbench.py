#!/usr/bin/env python3
"""Run the bounded Experiment-2 validation matrix on one persistent GPU model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from mosaickv.adapters.huggingface import audited_model_revision, load_hf_adapter
from mosaickv.backends.hf_runtime import HuggingFaceMosaicKVModel
from mosaickv.backends.sglang_runtime import SGLangFullKVModel, SGLangRuntimeOptions
from mosaickv.backends.vllm_runtime import VLLMFullKVModel, VLLMRuntimeOptions
from mosaickv.config import (
    CacheConfig,
    DatasetConfig,
    ExecutionConfig,
    ForecastingConfig,
    GenerationConfig,
    GraphConfig,
    ModelConfig,
    PrototypeConfig,
    RepairConfig,
    ResidualConfig,
    RunConfig,
    SelectionConfig,
    config_sha256,
)
from mosaickv.evaluation.messages import MediaItem, MediaKind, build_multimodal_messages
from mosaickv.evaluation.model import EvaluationRequest, ModelGeneration
from mosaickv.manifest import (
    ArtifactProvenance,
    InputProvenance,
    RunManifestWriter,
    sha256_bytes,
    sha256_text,
)
from mosaickv.types import (
    Backend,
    BudgetUnit,
    ForecastMode,
    MeasurementType,
    MosaicKVMethod,
    Precision,
    RepairPolicy,
)

MODEL_ALIASES = {
    "qwen3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "llava7b": "llava-hf/llava-1.5-7b-hf",
}


@dataclass(frozen=True, slots=True)
class Point:
    method: str
    retention_ratio: float
    resolution: int
    image_count: int
    output_length: int
    batch_size: int = 1

    @property
    def point_id(self) -> str:
        retention = str(self.retention_ratio).replace(".", "p")
        return (
            f"{self.method}-r{retention}-px{self.resolution}-i{self.image_count}-"
            f"o{self.output_length}-b{self.batch_size}"
        )


def _deduplicate(points: Iterable[Point]) -> tuple[Point, ...]:
    return tuple(dict.fromkeys(points))


def bounded_points(backend: str, model_alias: str) -> tuple[Point, ...]:
    """Return the preregistered small matrix; no full Cartesian product is run."""

    default_resolution = 336 if model_alias == "llava7b" else 448
    if backend in {"hf_sdpa", "hf_flash_attention_2"}:
        return (
            Point("full_kv", 1.0, default_resolution, 1, 64),
            Point("full_kv", 1.0, default_resolution, 1, 256),
        )
    if backend == "vllm":
        if model_alias != "qwen3b":
            raise ValueError("this bounded campaign admits only the parity-verified Qwen-3B vLLM")
        return _deduplicate(
            [Point("full_kv", 1.0, 448, 1, length) for length in (64, 256)]
            + [Point("full_kv", 1.0, resolution, 1, 64) for resolution in (224, 672)]
            + [Point("full_kv", 1.0, 448, count, 64) for count in (2, 4)]
            + [Point("full_kv", 1.0, 448, 1, 64, batch) for batch in (2, 4)]
        )
    if backend == "sglang":
        if model_alias != "qwen3b":
            raise ValueError("this bounded campaign admits only Qwen-3B SGLang FullKV")
        return (
            Point("full_kv", 1.0, 448, 1, 64),
            Point("full_kv", 1.0, 448, 1, 256),
        )
    if backend not in {"hf_eager", "hf_eager_sdpa_decode"}:
        raise ValueError(f"unsupported measured backend: {backend}")
    if model_alias == "llava7b":
        # LLaVA has 32 KV heads, so block-size 16 creates four times as many
        # Python graph/cache descriptors as the grouped-query Qwen models.
        # The point config below uses block-size 64 for every LLaVA row; retain
        # a bounded pair of compression points instead of abandoning the model.
        return (
            Point("full_kv", 1.0, default_resolution, 1, 64),
            Point("full_kv", 1.0, default_resolution, 1, 256),
            Point("mosaickv_exact", 0.2, default_resolution, 1, 64),
            Point("mosaickv_exact", 0.1, default_resolution, 1, 64),
        )
    if model_alias == "qwen3b":
        return _deduplicate(
            [Point("full_kv", 1.0, resolution, 1, 64) for resolution in (224, 448, 672)]
            + [Point("full_kv", 1.0, 448, count, 64) for count in (2, 4)]
            + [Point("full_kv", 1.0, 448, 1, length) for length in (32, 128, 256)]
            + [Point("mosaickv_exact", ratio, 448, 1, 64) for ratio in (0.4, 0.2, 0.1)]
            + [Point("mosaickv_exact", ratio, 448, 1, 256) for ratio in (0.4, 0.2, 0.1)]
        )
    return _deduplicate(
        [Point("full_kv", 1.0, default_resolution, 1, length) for length in (64, 256)]
        + [Point("mosaickv_exact", ratio, default_resolution, 1, 64) for ratio in (0.4, 0.2, 0.1)]
        + [Point("mosaickv_exact", 0.1, default_resolution, 1, 256)]
    )


def _disabled_stages() -> dict[str, Any]:
    return {
        "forecasting": ForecastingConfig(
            enabled=False,
            mode=ForecastMode.PROMPT_WINDOW,
            prompt_window=16,
            draft_steps=0,
        ),
        "graph": GraphConfig(enabled=False),
        "selection": SelectionConfig(enabled=False),
        "prototypes": PrototypeConfig(enabled=False),
        "residual": ResidualConfig(enabled=False, require_pinned_memory=False),
        "repair": RepairConfig(
            enabled=False,
            policy=RepairPolicy.NONE,
            max_blocks_per_step=0,
        ),
    }


def point_config(point: Point, backend: str, model_id: str) -> RunConfig:
    method = MosaicKVMethod(point.method)
    backend_name = {
        "vllm": Backend.VLLM,
        "sglang": Backend.SGLANG,
    }.get(backend, Backend.HUGGINGFACE)
    attention = {
        "hf_sdpa": "sdpa",
        "hf_flash_attention_2": "flash_attention_2",
        "sglang": "torch_native",
    }.get(backend, "eager")
    stages = _disabled_stages()
    if method is MosaicKVMethod.MOSAICKV_EXACT:
        stages.update(
            forecasting=ForecastingConfig(
                enabled=True,
                mode=ForecastMode.HYBRID,
                prompt_window=16,
                draft_steps=1,
                centroid_count=4,
            ),
            graph=GraphConfig(
                enabled=True,
                max_neighbors=4 if model_id.startswith("llava-hf/") else 8,
                similarity_chunk_size=128 if model_id.startswith("llava-hf/") else 256,
            ),
            selection=SelectionConfig(
                enabled=True,
                stop_on_nonpositive_gain=False,
                minimum_stratum_fraction=0.0,
            ),
        )
    return RunConfig(
        model=ModelConfig(model_id, audited_model_revision(model_id), Precision.BF16),
        dataset=DatasetConfig("mosaickv/controlled-visual-microbench", "schema-v1", "validation"),
        execution=ExecutionConfig(
            backend_name,
            attention,
            seed=17,
            deterministic_algorithms=True,
            decode_attention_implementation=("sdpa" if backend == "hf_eager_sdpa_decode" else None),
        ),
        generation=GenerationConfig(max_new_tokens=point.output_length),
        cache=CacheConfig(
            budget_value=2**31 - 1,
            budget_unit=BudgetUnit.BYTES,
            retention_ratio=point.retention_ratio,
            block_size=64 if model_id.startswith("llava-hf/") else 16,
        ),
        method=method,
        **stages,
    )


def _controlled_image(resolution: int, index: int) -> Image.Image:
    image = Image.new("RGB", (resolution, resolution), (244, 246, 250))
    draw = ImageDraw.Draw(image)
    width = max(1, resolution // 96)
    spacing = max(8, resolution // 8)
    color = ((37 * index + 23) % 200, (67 * index + 41) % 200, (97 * index + 59) % 200)
    for coordinate in range(0, resolution, spacing):
        draw.line((coordinate, 0, coordinate, resolution - 1), fill=color, width=width)
        draw.line((0, coordinate, resolution - 1, coordinate), fill=color, width=width)
    draw.rectangle(
        (resolution // 4, resolution // 4, 3 * resolution // 4, 3 * resolution // 4),
        outline=(10, 20, 30),
        width=max(1, 2 * width),
    )
    return image


def _messages(point: Point, batch_member: int) -> tuple[Any, ...]:
    images = tuple(_controlled_image(point.resolution, index) for index in range(point.image_count))
    media = tuple(MediaItem(MediaKind.IMAGE, image) for image in images)
    prompt = (
        "Inspect every image carefully. State the dominant grid color and count the outlined "
        f"central rectangles. Batch member {batch_member}. Answer concisely."
    )
    return build_multimodal_messages(prompt, media)


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not_installed"


def _command_output(command: list[str]) -> str:
    completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30)
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def _job_provenance() -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[2]
    git_sha = _command_output(["git", "-C", str(repo), "rev-parse", "HEAD"])
    diff = subprocess.run(
        ["git", "-C", str(repo), "diff", "--binary", "HEAD"],
        check=False,
        capture_output=True,
        timeout=60,
    ).stdout
    status = _command_output(["git", "-C", str(repo), "status", "--porcelain=v1"])
    gpu = _command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,pstate,clocks.current.sm,clocks.current.memory,power.draw",
            "--format=csv,noheader,nounits",
        ]
    )
    processes = _command_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    import torch

    return {
        "git_sha": git_sha,
        "git_dirty": bool(status and status != "unavailable"),
        "patch_sha": sha256_bytes(diff) if status and status != "unavailable" else "not_applicable",
        "python": sys.version.split()[0],
        "torch": _package_version("torch"),
        "transformers": _package_version("transformers"),
        "vllm": _package_version("vllm"),
        "sglang": _package_version("sglang"),
        "cuda_runtime": str(torch.version.cuda),
        "gpu_snapshot": gpu,
        "background_gpu_processes_at_start": processes,
        "gpu_count": len([line for line in gpu.splitlines() if line.strip()]),
        "fixed_clock_attempt": os.environ.get("MOSAICKV_FIXED_CLOCK_STATUS", "not_attempted"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node": os.environ.get("SLURMD_NODENAME"),
    }


def _trace_path(trace_root: Path, backend: str, run_id: str, sample_id: str) -> Path:
    if backend.startswith("hf_"):
        identity = hashlib.sha256(f"{run_id}\0{sample_id}".encode()).hexdigest()[:16]
        return trace_root / run_id / f"{identity}.json"
    readable = "".join(c if c.isalnum() or c in "-_" else "_" for c in sample_id)
    digest = hashlib.sha256(sample_id.encode()).hexdigest()[:12]
    return trace_root / run_id / f"{readable[:80]}-{digest}.json"


def _median(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return statistics.median(finite) if finite else None


def _hf_observation(
    generation: ModelGeneration,
    trace: dict[str, Any],
) -> dict[str, Any]:
    timing = trace.get("timing_breakdown", {})
    per_token = [float(value) for value in timing.get("per_token_decode", [])]
    return {
        "answer": generation.answer,
        "token_ids": trace.get("generated_token_ids", []),
        "image_encoder_time": timing.get("image_video_encoder"),
        "projector_time": timing.get("projector"),
        "prefill_time": generation.metrics.prefill_time,
        "compression_time": generation.metrics.compression_time,
        "ttft": generation.metrics.ttft,
        "inter_token_latency": _median(per_token),
        "inter_token_latencies": per_token,
        "decode_time": generation.metrics.decode_time,
        "end_to_end_time": generation.metrics.end_to_end_time,
        "generated_tokens": generation.metrics.generated_tokens,
        "active_kv_bytes": generation.metrics.active_kv_bytes,
        "residual_kv_bytes": generation.metrics.residual_kv_bytes,
        "peak_gpu_memory_allocated": trace.get("peak_gpu_memory_allocated"),
        "peak_gpu_memory_reserved": trace.get("peak_gpu_memory_reserved"),
        "residual_transfer_time": timing.get("repair_transfer"),
        "repair_time": timing.get("repair_redecode"),
        "repair_count": generation.metrics.repair_count,
        "repaired_bytes": generation.metrics.repaired_bytes,
        "prefix_cache_hit_rate": None,
        "encoder_cache_hit_rate": None,
    }


def _vllm_observation(
    generation: ModelGeneration,
    trace: dict[str, Any],
    bytes_per_position: int,
) -> dict[str, Any]:
    trial = trace["trials"][0]
    prompt_tokens = trial.get("prompt_tokens")
    active_bytes = None if prompt_tokens is None else int(prompt_tokens) * bytes_per_position
    return {
        "answer": generation.answer,
        "token_ids": trial.get("token_ids", []),
        "image_encoder_time": None,
        "projector_time": None,
        "prefill_time": generation.metrics.prefill_time,
        "compression_time": 0.0,
        "ttft": generation.metrics.ttft,
        "inter_token_latency": _median(trial.get("inter_token_latencies_seconds", [])),
        "inter_token_latencies": trial.get("inter_token_latencies_seconds", []),
        "decode_time": generation.metrics.decode_time,
        "end_to_end_time": generation.metrics.end_to_end_time,
        "generated_tokens": generation.metrics.generated_tokens,
        "active_kv_bytes": active_bytes,
        "residual_kv_bytes": 0,
        "peak_gpu_memory_allocated": generation.metrics.peak_gpu_memory,
        "peak_gpu_memory_reserved": None,
        "residual_transfer_time": 0.0,
        "repair_time": 0.0,
        "repair_count": 0,
        "repaired_bytes": 0,
        "prefix_cache_hit_rate": trial.get("prefix_cache_hit_rate"),
        "encoder_cache_hit_rate": trial.get("mm_cache_hit_rate"),
        "prompt_tokens": prompt_tokens,
    }


def _sglang_observation(
    generation: ModelGeneration,
    trace: dict[str, Any],
) -> dict[str, Any]:
    trial = trace["trials"][0]
    return {
        "answer": generation.answer,
        "token_ids": trial.get("token_ids", []),
        "image_encoder_time": None,
        "projector_time": None,
        "prefill_time": generation.metrics.prefill_time,
        "compression_time": 0.0,
        "ttft": generation.metrics.ttft,
        "inter_token_latency": _median(trial.get("inter_token_latencies_seconds", [])),
        "inter_token_latencies": trial.get("inter_token_latencies_seconds", []),
        "decode_time": generation.metrics.decode_time,
        "end_to_end_time": generation.metrics.end_to_end_time,
        "generated_tokens": generation.metrics.generated_tokens,
        "active_kv_bytes": generation.metrics.active_kv_bytes,
        "residual_kv_bytes": 0,
        "peak_gpu_memory_allocated": generation.metrics.peak_gpu_memory,
        "peak_gpu_memory_reserved": None,
        "residual_transfer_time": 0.0,
        "repair_time": 0.0,
        "repair_count": 0,
        "repaired_bytes": 0,
        "prefix_cache_hit_rate": trial.get("prefix_cache_hit_rate"),
        "encoder_cache_hit_rate": None,
        "prompt_tokens": trial.get("prompt_tokens"),
    }


def _text_config(config: Any) -> Any:
    return getattr(config, "text_config", config)


def _bytes_per_position(model_id: str, revision: str) -> int:
    from transformers import AutoConfig

    config = _text_config(
        AutoConfig.from_pretrained(model_id, revision=revision, local_files_only=True)
    )
    layers = int(config.num_hidden_layers)
    heads = int(config.num_attention_heads)
    kv_heads = int(getattr(config, "num_key_value_heads", heads))
    head_dim = int(getattr(config, "head_dim", int(config.hidden_size) // heads))
    return 2 * layers * kv_heads * head_dim * 2


def _input_provenance(model: Any, backend: str, point: Point) -> tuple[InputProvenance, int | None]:
    prompt_digest = hashlib.sha256()
    media_digest = hashlib.sha256()
    token_digest = hashlib.sha256()
    visual_tokens: int | None = None
    for member in range(point.batch_size):
        messages = _messages(point, member)
        prompt_digest.update(repr(messages).encode())
        for message in messages:
            for part in message.content:
                if part.type == "image":
                    media_digest.update(part.value.tobytes())
        if backend.startswith("hf_"):
            prepared = model.adapter.prepare_inputs(messages)
            ids = prepared.model_inputs["input_ids"].detach().cpu().contiguous()
            token_digest.update(ids.numpy().tobytes())
            count = sum(
                span.end - span.start
                for span in prepared.modality_map
                if span.modality.value != "text"
            )
            visual_tokens = count if visual_tokens is None else visual_tokens + count
        elif backend == "vllm":
            from mosaickv.backends.vllm_runtime import prepare_vllm_prompt

            prepared = prepare_vllm_prompt(model.processor, model.model_id, messages)
            tokenizer = getattr(model.processor, "tokenizer", model.processor)
            ids = tokenizer(prepared.rendered_text, add_special_tokens=False)["input_ids"]
            token_digest.update(json.dumps(ids, separators=(",", ":")).encode())
        else:
            from mosaickv.backends.sglang_runtime import prepare_sglang_prompt

            prepared = prepare_sglang_prompt(model.processor, model.model_id, messages)
            token_digest.update(
                json.dumps(prepared.prompt_token_ids, separators=(",", ":")).encode()
            )
    return (
        InputProvenance(
            prompt_set_sha=prompt_digest.hexdigest(),
            media_set_sha=media_digest.hexdigest(),
            preprocessing_sha=sha256_text(
                f"controlled-visual-microbench-v1\0{model.model_id}\0{point.resolution}"
            ),
            tokenization_sha=token_digest.hexdigest(),
        ),
        visual_tokens,
    )


def _run_batch(
    model: Any,
    backend: str,
    point: Point,
    *,
    run_id: str,
    trial_label: str,
    trace_root: Path,
    bytes_per_position: int,
) -> tuple[list[dict[str, Any]], float]:
    def one(member: int) -> dict[str, Any]:
        sample_id = f"{point.point_id}-{trial_label}-member-{member}"
        request = EvaluationRequest(
            run_id=run_id,
            sample_id=sample_id,
            task="controlled_visual_microbench",
            messages=_messages(point, member),
            generation_kwargs={
                "max_new_tokens": point.output_length,
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": 17,
            },
        )
        generation = model.generate(request)
        path = _trace_path(trace_root, backend, run_id, sample_id)
        trace = json.loads(path.read_text(encoding="utf-8"))
        if backend.startswith("hf_"):
            observation = _hf_observation(generation, trace)
        elif backend == "vllm":
            observation = _vllm_observation(generation, trace, bytes_per_position)
        else:
            observation = _sglang_observation(generation, trace)
        observation["trace_path"] = str(path)
        return observation

    started = time.perf_counter()
    if point.batch_size == 1:
        observations = [one(0)]
    elif backend == "vllm":
        with ThreadPoolExecutor(max_workers=point.batch_size) as executor:
            observations = list(executor.map(one, range(point.batch_size)))
    else:
        raise ValueError("HF adapters support batch_size=1 only; no sequential proxy is allowed")
    return observations, time.perf_counter() - started


def _aggregate_observations(
    observations: list[dict[str, Any]],
    batch_wall_seconds: float,
) -> dict[str, Any]:
    generated = sum(int(item["generated_tokens"] or 0) for item in observations)
    active = [item["active_kv_bytes"] for item in observations]
    residual = [item["residual_kv_bytes"] for item in observations]
    result: dict[str, Any] = {
        "answer": [item["answer"] for item in observations],
        "token_ids": [item["token_ids"] for item in observations],
        "inter_token_latencies": [item["inter_token_latencies"] for item in observations],
        "generated_tokens": generated,
        "active_kv_bytes": sum(int(value) for value in active)
        if all(v is not None for v in active)
        else None,
        "residual_kv_bytes": sum(int(value) for value in residual)
        if all(v is not None for v in residual)
        else None,
        "end_to_end_time": batch_wall_seconds,
        "tokens_per_second": generated / batch_wall_seconds if batch_wall_seconds > 0 else None,
        "requests_per_second": len(observations) / batch_wall_seconds
        if batch_wall_seconds > 0
        else None,
        "repair_count": sum(int(item["repair_count"] or 0) for item in observations),
        "repaired_bytes": sum(int(item["repaired_bytes"] or 0) for item in observations),
        "trace_paths": [item["trace_path"] for item in observations],
    }
    for name in (
        "image_encoder_time",
        "projector_time",
        "prefill_time",
        "compression_time",
        "ttft",
        "inter_token_latency",
        "decode_time",
        "peak_gpu_memory_allocated",
        "peak_gpu_memory_reserved",
        "residual_transfer_time",
        "repair_time",
        "prefix_cache_hit_rate",
        "encoder_cache_hit_rate",
    ):
        result[name] = _median([item.get(name) for item in observations])
    result["repair_frequency"] = result["repair_count"] / generated if generated > 0 else 0.0
    return result


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> int:
    cache_root = Path(os.environ.get("MOSAICKV_CACHE_ROOT", "")).resolve()
    home = Path.home().resolve()
    if not str(cache_root) or cache_root == home or home in cache_root.parents:
        raise RuntimeError("MOSAICKV_CACHE_ROOT must be set outside the home directory")
    output_root = Path(args.output_root).resolve()
    if home in output_root.parents or output_root == home:
        raise RuntimeError("performance outputs must be outside the home directory")
    output_root.mkdir(parents=True, exist_ok=False)
    trace_root = output_root / "traces"
    points_root = output_root / "points"
    points_root.mkdir()
    provenance = _job_provenance()
    model_id = MODEL_ALIASES[args.model]
    points = bounded_points(args.backend, args.model)
    if args.point_id:
        requested = set(args.point_id)
        available = {point.point_id: point for point in points}
        unknown = sorted(requested - available.keys())
        if unknown:
            raise ValueError(
                "requested performance point IDs are not in the bounded matrix: "
                + ", ".join(unknown)
            )
        points = tuple(point for point in points if point.point_id in requested)
    if args.method_override is not None:
        points = tuple(replace(point, method=args.method_override) for point in points)
    base_config = point_config(points[0], args.backend, model_id)
    import torch

    torch.use_deterministic_algorithms(True)
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.backend.startswith("hf_"):
        model_kwargs: dict[str, Any] = {
            "attn_implementation": base_config.execution.attention_implementation,
            "torch_dtype": torch.bfloat16,
            "local_files_only": True,
            "low_cpu_mem_usage": True,
            "device_map": {"": "cuda:0"},
            "cache_dir": os.environ["HF_HUB_CACHE"],
        }
        adapter = load_hf_adapter(
            model_id,
            revision=base_config.model.revision,
            model_kwargs=model_kwargs,
            processor_kwargs={
                "local_files_only": True,
                "cache_dir": os.environ["HF_HUB_CACHE"],
            },
        )
        model: Any = None
        bytes_per_position = 0
    elif args.backend == "vllm":
        adapter = None
        model = VLLMFullKVModel.from_config(
            base_config,
            VLLMRuntimeOptions(
                local_files_only=True,
                cache_probe_repeats=1,
                gpu_memory_utilization=0.85,
                max_model_len=4096,
            ),
            trace_directory=trace_root,
        )
        bytes_per_position = _bytes_per_position(model_id, base_config.model.revision)
    else:
        adapter = None
        model = SGLangFullKVModel.from_config(
            base_config,
            SGLangRuntimeOptions(
                local_files_only=True,
                cache_probe_repeats=1,
                mem_fraction_static=0.8,
                context_length=4096,
            ),
            trace_directory=trace_root,
        )
        bytes_per_position = 0

    failures = 0
    try:
        for point in points:
            config = point_config(point, args.backend, model_id)
            if args.backend.startswith("hf_"):
                if adapter is None:
                    raise RuntimeError("HF performance run did not construct an adapter")
                model = HuggingFaceMosaicKVModel(adapter, config, trace_directory=trace_root)
            point_root = points_root / point.point_id
            point_root.mkdir()
            raw_path = point_root / "trials.jsonl"
            log_path = point_root / "point-log.json"
            manifest_path = point_root / "manifest.json"
            input_provenance, visual_tokens = _input_provenance(model, args.backend, point)
            run_id = f"{args.campaign}-{args.backend}-{args.model}-{point.point_id}"
            for warmup in range(args.warmups):
                _run_batch(
                    model,
                    args.backend,
                    point,
                    run_id=run_id,
                    trial_label=f"warmup-{warmup:02d}",
                    trace_root=trace_root,
                    bytes_per_position=bytes_per_position,
                )
            rows: list[dict[str, Any]] = []
            for trial_index in range(args.trials):
                row: dict[str, Any] = {
                    "schema_version": 1,
                    "campaign": args.campaign,
                    "point_id": point.point_id,
                    "trial_index": trial_index,
                    "warmups": args.warmups,
                    "timed_repetitions": args.trials,
                    "status": "started",
                    "model": model_id,
                    "model_revision": config.model.revision,
                    "precision": config.model.precision.value,
                    "backend": args.backend,
                    "attention_implementation": config.execution.attention_implementation,
                    "decode_attention_implementation": (
                        config.execution.decode_attention_implementation
                        or config.execution.attention_implementation
                    ),
                    "method": point.method,
                    "retention_ratio": point.retention_ratio,
                    "resolution": point.resolution,
                    "image_count": point.image_count,
                    "visual_token_count": visual_tokens,
                    "output_length": point.output_length,
                    "batch_size": point.batch_size,
                    "seed": config.execution.seed,
                    "measurement_type": (
                        MeasurementType.REFERENCE.value
                        if config.method.is_full_cache
                        else MeasurementType.METHOD.value
                    ),
                    "config_sha": config_sha256(config),
                    "dataset": config.dataset.id,
                    "dataset_revision": config.dataset.revision,
                    "manifest_path": str(manifest_path),
                    "native_mosaickv": False,
                    **provenance,
                }
                try:
                    observations, wall = _run_batch(
                        model,
                        args.backend,
                        point,
                        run_id=run_id,
                        trial_label=f"trial-{trial_index:03d}",
                        trace_root=trace_root,
                        bytes_per_position=bytes_per_position,
                    )
                    row.update(_aggregate_observations(observations, wall))
                    row["status"] = "completed"
                    row["error"] = None
                except Exception as error:
                    failures += 1
                    row.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
                rows.append(row)
                with raw_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            log = {
                "campaign": args.campaign,
                "point": asdict(point),
                "completed_trials": sum(row["status"] == "completed" for row in rows),
                "failed_trials": sum(row["status"] == "failed" for row in rows),
                "visual_token_count": visual_tokens,
                "paper_eligible": not provenance["git_dirty"],
            }
            _atomic_json(log_path, log)
            RunManifestWriter().write(
                manifest_path,
                config,
                (
                    MeasurementType.REFERENCE
                    if config.method.is_full_cache
                    else MeasurementType.METHOD
                ),
                input_provenance,
                ArtifactProvenance(
                    sha256_bytes(raw_path.read_bytes()),
                    "not_applicable",
                    sha256_bytes(log_path.read_bytes()),
                ),
                run_id=run_id,
                execution_metadata={
                    "campaign": args.campaign,
                    "warmups": args.warmups,
                    "timed_repetitions": args.trials,
                    "batch_size": point.batch_size,
                    "fixed_clock_attempt": str(provenance["fixed_clock_attempt"]),
                },
            )
            print(json.dumps(log, sort_keys=True), flush=True)
    finally:
        if args.backend in {"vllm", "sglang"} and model is not None:
            model.close()
    _atomic_json(
        output_root / "job-summary.json",
        {
            "campaign": args.campaign,
            "backend": args.backend,
            "model": model_id,
            "points": len(points),
            "warmups": args.warmups,
            "trials": args.trials,
            "protocol": (
                "experiment_2"
                if args.warmups >= 5 and args.trials >= 20
                else "debug_only_not_for_tables"
            ),
            "failed_trials": failures,
            "provenance": provenance,
        },
    )
    return 1 if failures else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=(
            "hf_eager",
            "hf_eager_sdpa_decode",
            "hf_sdpa",
            "hf_flash_attention_2",
            "vllm",
            "sglang",
        ),
        required=True,
    )
    parser.add_argument("--model", choices=tuple(MODEL_ALIASES), required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--point-id",
        action="append",
        default=[],
        help="run only the named bounded point; repeat to select multiple points",
    )
    parser.add_argument(
        "--method-override",
        choices=("mosaickv_exact", "prompt_attention_topk"),
        help=(
            "replace the selected bounded point method while preserving its byte budget; "
            "intended for paired latency diagnostics"
        ),
    )
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument(
        "--allow-debug-protocol",
        action="store_true",
        help="allow fewer repetitions for a diagnostic run that is ineligible for tables",
    )
    args = parser.parse_args()
    if (args.warmups < 5 or args.trials < 20) and not args.allow_debug_protocol:
        parser.error("Experiment-2 requires at least 5 warmups and 20 timed repetitions")
    if args.warmups < 0 or args.trials < 1:
        parser.error("warmups must be nonnegative and trials must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
