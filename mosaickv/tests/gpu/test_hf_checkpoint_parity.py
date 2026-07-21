from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict
from hashlib import sha256
from importlib import metadata
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from mosaickv.adapters.huggingface import (  # noqa: E402
    InternVLVideo,
    compare_cache_reinjection,
    compare_with_generate,
    load_hf_adapter,
)
from mosaickv.evaluation.messages import (  # noqa: E402
    MediaItem,
    MediaKind,
    build_multimodal_messages,
)

_REVISIONS = {
    "llava-hf/llava-1.5-7b-hf": "b234b804b114d9e37bb655e11cbbb5f5e971b7a9",
    "Qwen/Qwen2.5-VL-3B-Instruct": "66285546d2b821cf421d4f5eb2576359d3770cd3",
    "Qwen/Qwen2.5-VL-7B-Instruct": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
    "llava-hf/llava-onevision-qwen2-0.5b-ov-hf": "74dd0bf867a4cda7950c17663794267c60cf4b40",
    "OpenGVLab/InternVL2_5-4B": "2cf4a8158bbc40d35015e7c63b527890de4d27b3",
}


def _git_sha() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def _version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "not_installed"


def _driver_version() -> str:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    return ",".join(sorted(set(completed.stdout.split())))


def _native_messages() -> tuple[Any, ...]:
    from PIL import Image

    image = Image.new("RGB", (32, 32), color=(23, 47, 89))
    return build_multimodal_messages(
        "Describe the visual input briefly.",
        (MediaItem(MediaKind.IMAGE, image),),
    )


def _internvl_messages() -> tuple[Any, ...]:
    path_value = os.environ.get("MOSAICKV_INTERNVL_PIXEL_VALUES")
    if not path_value:
        raise RuntimeError(
            "InternVL validation requires MOSAICKV_INTERNVL_PIXEL_VALUES from the "
            "checkpoint's pinned public preprocessing"
        )
    path = Path(path_value)
    pixel_values = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(pixel_values, dict):
        pixels = pixel_values["pixel_values"]
        patch_counts = pixel_values.get("num_patches_list")
        payload: Any
        if patch_counts is None:
            payload = pixels
            kind = MediaKind.IMAGE
        else:
            payload = InternVLVideo(pixels, tuple(int(item) for item in patch_counts))
            kind = MediaKind.VIDEO
    else:
        payload = pixel_values
        kind = MediaKind.IMAGE
    return build_multimodal_messages(
        "Describe the visual input briefly.",
        (MediaItem(kind, payload),),
    )


@pytest.mark.gpu
@pytest.mark.hf
def test_pinned_checkpoint_16_token_parity() -> None:
    """The authoritative, measured adapter acceptance gate."""

    model_id = os.environ.get("MOSAICKV_HF_MODEL_ID")
    if not model_id:
        pytest.skip("set MOSAICKV_HF_MODEL_ID to run a pinned checkpoint gate")
    if model_id not in _REVISIONS:
        pytest.fail(f"MOSAICKV_HF_MODEL_ID is not audited: {model_id}")
    if not torch.cuda.is_available():
        pytest.fail("pinned checkpoint parity requires a visible CUDA device")
    revision = os.environ.get("MOSAICKV_HF_REVISION", _REVISIONS[model_id])
    if revision != _REVISIONS[model_id]:
        pytest.fail("checkpoint parity must use the audited immutable revision")
    cache_root = Path(os.environ.get("MOSAICKV_CACHE_ROOT", "/scratch/djy8hg/cache/mosaickv"))
    hub_cache = Path(os.environ.get("HF_HUB_CACHE", cache_root / "huggingface" / "hub"))
    allow_download = os.environ.get("MOSAICKV_ALLOW_MODEL_DOWNLOAD") == "1"
    model_kwargs: dict[str, Any] = {
        "cache_dir": str(hub_cache),
        "device_map": "auto",
        "torch_dtype": "auto",
        "local_files_only": not allow_download,
    }
    adapter = load_hf_adapter(
        model_id,
        revision=revision,
        model_kwargs=model_kwargs,
        processor_kwargs={
            "cache_dir": str(hub_cache),
            "local_files_only": not allow_download,
        },
    )
    messages = _internvl_messages() if model_id.startswith("OpenGVLab/") else _native_messages()
    prepared = adapter.prepare_inputs(messages)
    reference = compare_with_generate(adapter, prepared, max_new_tokens=16)
    reinjection = compare_cache_reinjection(adapter, prepared, max_new_tokens=16)
    tolerance = float(os.environ.get("MOSAICKV_FULL_CACHE_LOGIT_ATOL", "1e-4"))
    input_ids_bytes = prepared.model_inputs["input_ids"].detach().cpu().numpy().tobytes()
    if model_id.startswith("OpenGVLab/"):
        internvl_media_path = Path(os.environ["MOSAICKV_INTERNVL_PIXEL_VALUES"])
        media_bytes = internvl_media_path.read_bytes()
    else:
        media_bytes = bytes((23, 47, 89)) * (32 * 32)
    inline_config = {
        "model_id": model_id,
        "model_revision": revision,
        "backend": "huggingface",
        "attention_implementation": "eager",
        "generation": {"do_sample": False, "max_new_tokens": 16},
        "retention_ratio": 1.0,
        "seed": 0,
        "logit_absolute_tolerance": tolerance,
    }
    config_bytes = json.dumps(inline_config, sort_keys=True, separators=(",", ":")).encode()
    record = {
        "git_sha": _git_sha(),
        "config_sha": sha256(config_bytes).hexdigest(),
        "model_id": model_id,
        "model_revision": revision,
        "dataset": "mosaickv/adapter-validation-input",
        "dataset_revision": sha256(b"adapter-validation-input-v1").hexdigest(),
        "backend": "huggingface",
        "attention_implementation": "eager",
        "precision": str(adapter.model.dtype),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "accelerate": _version("accelerate"),
        "flash_attn": _version("flash-attn"),
        "vllm": _version("vllm"),
        "sglang": _version("sglang"),
        "lmms_eval": _version("lmms-eval"),
        "datasets": _version("datasets"),
        "cuda": torch.version.cuda,
        "driver": _driver_version(),
        "gpu_type": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
        "seed": 0,
        "measurement_type": "validation_smoke",
        "prompt_set_sha": sha256(b"Describe the visual input briefly.").hexdigest(),
        "media_set_sha": sha256(media_bytes).hexdigest(),
        "tokenization_sha": sha256(input_ids_bytes).hexdigest(),
        "generation_parameters": inline_config["generation"],
        "output_length": 16,
        "cache_budget": {"retention_ratio": 1.0, "unit": "retained_slots"},
        "logit_absolute_tolerance": tolerance,
        "generate_reference": asdict(reference),
        "retention_1_reinjection": asdict(reinjection),
    }
    print(json.dumps(record, sort_keys=True))
    assert reference.generated_tokens == 16
    assert reference.token_agreement == 1.0
    assert reference.maximum_logit_difference <= tolerance
    assert reinjection.generated_tokens == 16
    assert reinjection.token_agreement == 1.0
    assert reinjection.maximum_logit_difference <= tolerance
