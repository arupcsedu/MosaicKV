#!/usr/bin/env python3
"""Run no-download or pinned-checkpoint HF adapter correctness gates."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import asdict
from hashlib import sha256
from importlib import metadata
from pathlib import Path
from typing import Any

import torch
import transformers

from mosaickv.adapters.huggingface import (
    InternVLVideo,
    Llava15Adapter,
    LlavaOneVisionAdapter,
    Qwen25VLAdapter,
    compare_cache_reinjection,
    compare_mosaickv_retention_one,
    compare_with_generate,
    load_hf_adapter,
)
from mosaickv.config import ForecastingConfig
from mosaickv.evaluation.messages import MediaItem, MediaKind, build_multimodal_messages
from mosaickv.evaluation.oracle_queries import collect_evaluation_only_true_future_queries
from mosaickv.forecasting import forecast_from_prefill
from mosaickv.fullkv import FullKV, FullKVBenchmarkConfig, FullKVBenchmarkRunner, FullKVSample
from mosaickv.measurements.memory import cache_tensors
from mosaickv.types import ForecastMode

REVISIONS = {
    "llava-hf/llava-1.5-7b-hf": "b234b804b114d9e37bb655e11cbbb5f5e971b7a9",
    "Qwen/Qwen2.5-VL-3B-Instruct": "66285546d2b821cf421d4f5eb2576359d3770cd3",
    "Qwen/Qwen2.5-VL-7B-Instruct": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
    "llava-hf/llava-onevision-qwen2-0.5b-ov-hf": "74dd0bf867a4cda7950c17663794267c60cf4b40",
    "OpenGVLab/InternVL2_5-4B": "2cf4a8158bbc40d35015e7c63b527890de4d27b3",
}


class TextOnlyProcessor:
    """Deterministic processor for randomly initialized architecture gates."""

    chat_template = "adapter-validation-template-v1"

    def apply_chat_template(self, *_args: Any, **_kwargs: Any) -> str:
        return "tiny prompt"

    def __call__(self, **_kwargs: Any) -> dict[str, Any]:
        return {
            "input_ids": torch.tensor([[1, 3, 4, 5]], dtype=torch.long),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
        }

    def batch_decode(self, token_ids: Any, **_kwargs: Any) -> list[str]:
        values = token_ids.detach().cpu().reshape(-1).tolist()
        return [" ".join(str(int(value)) for value in values)]


def tiny_adapters() -> list[tuple[str, Any]]:
    """Build small real Transformers architectures without model downloads."""

    from transformers import (
        LlavaConfig,
        LlavaForConditionalGeneration,
        LlavaOnevisionConfig,
        LlavaOnevisionForConditionalGeneration,
        Qwen2_5_VLConfig,
        Qwen2_5_VLForConditionalGeneration,
    )

    llava_config = LlavaConfig(
        vision_config={
            "model_type": "clip_vision_model",
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "image_size": 8,
            "patch_size": 4,
            "projection_dim": 16,
        },
        text_config={
            "model_type": "llama",
            "vocab_size": 64,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "max_position_embeddings": 64,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
        },
        image_token_index=63,
        attn_implementation="eager",
    )
    qwen_config = Qwen2_5_VLConfig(
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "patch_size": 2,
            "spatial_merge_size": 1,
            "temporal_patch_size": 1,
            "window_size": 4,
            "out_hidden_size": 16,
            "fullatt_block_indexes": [0],
        },
        text_config={
            "vocab_size": 64,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "max_position_embeddings": 64,
            "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2]},
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
        },
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=62,
        vision_end_token_id=63,
        attn_implementation="eager",
    )
    onevision_config = LlavaOnevisionConfig(
        vision_config={
            "model_type": "siglip_vision_model",
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "image_size": 8,
            "patch_size": 4,
            "vision_use_head": False,
        },
        text_config={
            "model_type": "qwen2",
            "vocab_size": 64,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "max_position_embeddings": 64,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
        },
        image_token_index=62,
        video_token_index=63,
        image_grid_pinpoints=[[8, 8]],
        attn_implementation="eager",
    )
    processor = TextOnlyProcessor()
    models = [
        (
            "tiny-random/LLaVA-1.5",
            LlavaForConditionalGeneration(llava_config).eval(),
            Llava15Adapter,
        ),
        (
            "tiny-random/Qwen2.5-VL",
            Qwen2_5_VLForConditionalGeneration(qwen_config).eval(),
            Qwen25VLAdapter,
        ),
        (
            "tiny-random/LLaVA-OneVision",
            LlavaOnevisionForConditionalGeneration(onevision_config).eval(),
            LlavaOneVisionAdapter,
        ),
    ]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return [
        (model_id, adapter_type(model.to(device), processor, device=device))
        for model_id, model, adapter_type in models
    ]


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not_installed"


def git_source() -> dict[str, Any]:
    root = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )

    def git_bytes(*arguments: str) -> bytes:
        return subprocess.run(["git", *arguments], cwd=root, check=True, capture_output=True).stdout

    commit = git_bytes("rev-parse", "HEAD").decode().strip()
    status = git_bytes("status", "--porcelain=v1", "-z")
    dirty = bool(status)
    if not dirty:
        return {"git_sha": commit, "git_dirty": False, "patch_sha": "not_applicable"}
    digest = sha256(status)
    digest.update(git_bytes("diff", "--binary", "HEAD"))
    digest.update(git_bytes("diff", "--binary", "--cached", "HEAD"))
    untracked = git_bytes("ls-files", "--others", "--exclude-standard", "-z")
    for relative_bytes in sorted(item for item in untracked.split(b"\0") if item):
        relative = relative_bytes.decode()
        digest.update(relative_bytes)
        path = root / relative
        if path.is_file() and not path.is_symlink():
            digest.update(path.read_bytes())
    return {"git_sha": commit, "git_dirty": True, "patch_sha": digest.hexdigest()}


def driver_version() -> str:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
    )
    return ",".join(sorted(set(result.stdout.split()))) if result.returncode == 0 else "not_used"


def provenance(config: dict[str, Any]) -> dict[str, Any]:
    config_bytes = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return {
        **git_source(),
        "config_sha": sha256(config_bytes).hexdigest(),
        "cuda": torch.version.cuda,
        "driver": driver_version(),
        "pytorch": torch.__version__,
        "transformers": transformers.__version__,
        "accelerate": package_version("accelerate"),
        "flash_attn": package_version("flash-attn"),
        "vllm": package_version("vllm"),
        "sglang": package_version("sglang"),
        "lmms_eval": package_version("lmms-eval"),
        "datasets": package_version("datasets"),
        "gpu_type": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "not_used",
        "gpu_count": torch.cuda.device_count(),
        "backend": "huggingface",
        "attention_implementation": "eager",
        "seed": 0,
        "measurement_type": "validation_smoke",
    }


def validate(adapter: Any, prepared: Any, *, tolerance: float) -> dict[str, Any]:
    reference = compare_with_generate(adapter, prepared, max_new_tokens=16)
    reinjected = compare_cache_reinjection(adapter, prepared, max_new_tokens=16)
    mosaic_reinjected = compare_mosaickv_retention_one(
        adapter, prepared, max_new_tokens=16, block_size=3
    )
    if reference.token_agreement != 1.0 or reference.maximum_logit_difference > tolerance:
        raise RuntimeError(f"generate parity failed: {reference}")
    if reinjected.token_agreement != 1.0 or reinjected.maximum_logit_difference > tolerance:
        raise RuntimeError(f"cache reinjection parity failed: {reinjected}")
    if (
        mosaic_reinjected.token_agreement != 1.0
        or mosaic_reinjected.maximum_logit_difference > tolerance
    ):
        raise RuntimeError(f"MosaicKV state retention-1 reinjection failed: {mosaic_reinjected}")
    forecasting = validate_forecasting(adapter, prepared)
    return {
        "generate_reference": asdict(reference),
        "retention_1_reinjection": asdict(reinjected),
        "mosaickv_state_retention_1": asdict(mosaic_reinjected),
        "future_query_forecasting": forecasting,
    }


def validate_forecasting(adapter: Any, prepared: Any) -> dict[str, Any]:
    """Validate isolated hybrid drafting and evaluation-only oracle agreement."""

    prefill = adapter.prefill(prepared)
    cache_before = adapter.extract_past_key_values(prefill.state.past_key_values)
    config = ForecastingConfig(
        mode=ForecastMode.HYBRID,
        prompt_window=3,
        draft_steps=3,
        centroid_count=2,
        low_memory_centroids=True,
    )
    first = forecast_from_prefill(adapter, prefill, config)
    second = forecast_from_prefill(adapter, prefill, config)
    oracle = collect_evaluation_only_true_future_queries(adapter, prefill, future_steps=3)
    cache_after = adapter.extract_past_key_values(prefill.state.past_key_values)
    for before_layer, after_layer in zip(cache_before.layers, cache_after.layers, strict=True):
        if not torch.equal(before_layer.key, after_layer.key) or not torch.equal(
            before_layer.value, after_layer.value
        ):
            raise RuntimeError("forecast validation changed the original prefill cache")
    head_count = 0
    for layer_index, (first_layer, second_layer) in enumerate(
        zip(first.layers, second.layers, strict=True)
    ):
        oracle_layer = oracle.query_layers[layer_index]
        for left, right in zip(first_layer, second_layer, strict=True):
            if not torch.equal(left.normalized_centroids, right.normalized_centroids):
                raise RuntimeError("hybrid forecast centroids are not reproducible")
            if not torch.equal(left.forecast_weights, right.forecast_weights):
                raise RuntimeError("hybrid forecast weights are not reproducible")
            if left.draft_query_samples is None:
                raise RuntimeError("hybrid forecast did not retain draft query samples")
            provenance = left.provenance
            expected = (
                oracle_layer[
                    :,
                    provenance.query_head_start : provenance.query_head_end,
                    :,
                    :,
                ]
                .detach()
                .float()
                .reshape(-1, oracle_layer.shape[-1])
            )
            if not torch.equal(left.draft_query_samples, expected):
                raise RuntimeError("draft queries differ from deterministic FullKV oracle queries")
            head_count += 1
    return {
        "mode": first.provenance.mode.value,
        "prompt_window": first.provenance.actual_prompt_window,
        "draft_steps": first.provenance.completed_draft_steps,
        "kv_head_forecasts": head_count,
        "cache_unchanged": first.provenance.draft_cache_isolated,
        "reproducible": True,
        "oracle_source": oracle.source,
        "timing_backend": first.timing.timing_backend,
        "forecast_overhead_seconds": {
            "cache_clone": first.timing.cache_clone,
            "draft_decode": first.timing.draft_decode,
            "query_preparation": first.timing.query_preparation,
            "prompt_statistics": first.timing.prompt_statistics,
            "centroid_construction": first.timing.centroid_construction,
            "total": first.timing.total,
        },
    }


def validate_fullkv(model_id: str, adapter: Any, prepared: Any) -> dict[str, Any]:
    """Exercise synchronized FullKV timing and exact byte accounting on CUDA."""

    if not torch.cuda.is_available():
        raise RuntimeError("FullKV validation requires CUDA")
    revision = sha256(f"{model_id}:tiny-fullkv-v1".encode()).hexdigest()
    reference = FullKV(adapter, model_id=model_id, model_revision=revision)
    runner = FullKVBenchmarkRunner(
        reference,
        FullKVBenchmarkConfig(
            warmups=0,
            repeated_trials=2,
            max_new_tokens=16,
            bootstrap_samples=100,
            seed=0,
        ),
    )
    output = runner.run(
        (FullKVSample("tiny-text", prepared),),
        run_id=f"fullkv-{sha256(model_id.encode()).hexdigest()[:12]}",
        dataset_id="mosaickv/tiny-text-adapter-validation",
        dataset_revision=sha256(b"tiny prompt").hexdigest(),
        manifest_path="/tmp/mosaickv-fullkv-tiny-manifest.json",
    )
    if output.aggregate.deterministic_token_match is not True:
        raise RuntimeError("FullKV repeated greedy trials did not produce identical tokens")

    manual = adapter.prefill(prepared, capture_queries=False)
    state = manual.state
    token = manual.next_token_id
    for _index in range(15):
        step = adapter.decode_one_token(token, state, capture_queries=False)
        token = step.next_token_id
        state = step.state
    expected_bytes = sum(
        int(tensor.numel()) * int(tensor.element_size())
        for tensor in cache_tensors(state.past_key_values)
    )
    for trial in output.trials:
        if trial.memory is None or trial.timings is None:
            raise RuntimeError(f"FullKV trial failed: {trial.error}")
        if trial.memory.active_kv_bytes != expected_bytes:
            raise RuntimeError(
                f"FullKV KV bytes {trial.memory.active_kv_bytes} != manual {expected_bytes}"
            )
        if trial.memory.cpu_residual_bytes != 0:
            raise RuntimeError("FullKV created CPU residual state")
        if trial.timings.compression != 0.0 or trial.timings.repair != 0.0:
            raise RuntimeError("FullKV unexpectedly executed compression or repair")
        if trial.synchronization_calls != 37:
            raise RuntimeError(
                "FullKV timing boundary synchronization mismatch: "
                f"expected 37, observed {trial.synchronization_calls}"
            )
    return {
        "aggregate": output.aggregate.to_json_object(),
        "expected_active_kv_bytes": expected_bytes,
        "trials": [trial.to_json_object() for trial in output.trials],
    }


def run_tiny() -> dict[str, Any]:
    torch.manual_seed(0)
    tolerance = 1e-6
    config = {
        "mode": "tiny_random_architectures",
        "models": ["LLaVA-1.5", "Qwen2.5-VL", "LLaVA-OneVision"],
        "generation": {"do_sample": False, "max_new_tokens": 16},
        "retention_ratio": 1.0,
        "precision": "fp32",
        "logit_absolute_tolerance": tolerance,
    }
    results: dict[str, Any] = {}
    for model_id, adapter in tiny_adapters():
        prepared = adapter.prepare_inputs([{"role": "user", "content": "tiny prompt"}])
        results[model_id] = {
            **validate(adapter, prepared, tolerance=tolerance),
            "fullkv": validate_fullkv(model_id, adapter, prepared),
        }
    return {
        **provenance(config),
        "model_id": "tiny-random/architecture-suite",
        "model_revision": sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "dataset": "mosaickv/tiny-text-adapter-validation",
        "dataset_revision": sha256(b"tiny prompt").hexdigest(),
        "precision": "fp32",
        "output_length": 16,
        "cache_budget": {"retention_ratio": 1.0, "unit": "retained_slots"},
        "results": results,
    }


def checkpoint_messages(model_id: str, internvl_pixels: Path | None) -> tuple[Any, bytes]:
    if model_id.startswith("OpenGVLab/"):
        if internvl_pixels is None:
            raise ValueError("InternVL requires --internvl-pixel-values")
        saved = torch.load(internvl_pixels, map_location="cpu", weights_only=True)
        if isinstance(saved, dict):
            pixels = saved["pixel_values"]
            counts = saved.get("num_patches_list")
            payload = pixels if counts is None else InternVLVideo(pixels, tuple(map(int, counts)))
            kind = MediaKind.IMAGE if counts is None else MediaKind.VIDEO
        else:
            payload, kind = saved, MediaKind.IMAGE
        media_bytes = internvl_pixels.read_bytes()
    else:
        from PIL import Image

        payload = Image.new("RGB", (32, 32), color=(23, 47, 89))
        kind = MediaKind.IMAGE
        media_bytes = bytes((23, 47, 89)) * (32 * 32)
    messages = build_multimodal_messages(
        "Describe the visual input briefly.", (MediaItem(kind, payload),)
    )
    return messages, media_bytes


def run_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    model_id = str(args.model_id)
    if model_id not in REVISIONS:
        raise ValueError(f"model is not audited: {model_id}")
    if not torch.cuda.is_available():
        raise RuntimeError("checkpoint validation requires CUDA")
    revision = str(args.revision or REVISIONS[model_id])
    if revision != REVISIONS[model_id]:
        raise ValueError("checkpoint gate requires the audited immutable revision")
    tolerance = float(args.logit_atol)
    cache_root = Path(args.cache_root)
    hub_cache = Path(os.environ.get("HF_HUB_CACHE", cache_root / "huggingface" / "hub"))
    adapter = load_hf_adapter(
        model_id,
        revision=revision,
        model_kwargs={
            "cache_dir": str(hub_cache),
            "device_map": "auto",
            "torch_dtype": "auto",
            "local_files_only": not args.allow_download,
        },
        processor_kwargs={
            "cache_dir": str(hub_cache),
            "local_files_only": not args.allow_download,
        },
    )
    messages, media_bytes = checkpoint_messages(model_id, args.internvl_pixel_values)
    prepared = adapter.prepare_inputs(messages)
    config = {
        "mode": "pinned_checkpoint",
        "model_id": model_id,
        "revision": revision,
        "generation": {"do_sample": False, "max_new_tokens": 16},
        "retention_ratio": 1.0,
        "attention_implementation": "eager",
        "precision": str(adapter.model.dtype),
        "logit_absolute_tolerance": tolerance,
    }
    return {
        **provenance(config),
        "model_id": model_id,
        "model_revision": revision,
        "dataset": "mosaickv/adapter-validation-input",
        "dataset_revision": sha256(b"adapter-validation-input-v1").hexdigest(),
        "precision": str(adapter.model.dtype),
        "prompt_set_sha": sha256(b"Describe the visual input briefly.").hexdigest(),
        "media_set_sha": sha256(media_bytes).hexdigest(),
        "tokenization_sha": sha256(
            prepared.model_inputs["input_ids"].detach().cpu().numpy().tobytes()
        ).hexdigest(),
        "output_length": 16,
        "cache_budget": {"retention_ratio": 1.0, "unit": "retained_slots"},
        "results": validate(adapter, prepared, tolerance=tolerance),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id")
    parser.add_argument("--revision")
    parser.add_argument(
        "--cache-root",
        default=os.environ.get("MOSAICKV_CACHE_ROOT", "/scratch/djy8hg/cache/mosaickv"),
    )
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--internvl-pixel-values", type=Path)
    parser.add_argument("--logit-atol", type=float, default=1e-4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_checkpoint(args) if args.model_id else run_tiny()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
