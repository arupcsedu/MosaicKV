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
    "llava-hf/llava-v1.6-mistral-7b-hf": "2424fdd47412fccc66d91719126b420e9fbd7065",
    "Qwen/Qwen2.5-VL-3B-Instruct": "66285546d2b821cf421d4f5eb2576359d3770cd3",
    "Qwen/Qwen2.5-VL-7B-Instruct": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
    "llava-hf/llava-onevision-qwen2-0.5b-ov-hf": "74dd0bf867a4cda7950c17663794267c60cf4b40",
    "llava-hf/llava-onevision-qwen2-7b-ov-hf": "0d50680527681998e456c7b78950205bedd8a068",
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
        # The pinned OneVision-7B config mixes a BF16 nested text dtype with
        # FP16 stored tensors. Explicit BF16 converts every module uniformly;
        # `auto` creates mixed lm_head/hidden dtypes and eager FP16 produced
        # non-finite logits on the A100 gate.
        "torch_dtype": (
            torch.bfloat16 if model_id == "llava-hf/llava-onevision-qwen2-7b-ov-hf" else "auto"
        ),
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

    if os.environ.get("MOSAICKV_VERIFY_THREE_TIER") == "1":
        _verify_three_tier_runtime(adapter, messages, model_id, revision, record)


def _verify_three_tier_runtime(
    adapter: Any,
    messages: tuple[Any, ...],
    model_id: str,
    revision: str,
    provenance: dict[str, Any],
) -> None:
    """Exercise identity, prototypes, and a forced repair on one real checkpoint."""

    import tempfile

    from mosaickv.backends import HuggingFaceMosaicKVModel
    from mosaickv.config import (
        CacheConfig,
        DatasetConfig,
        ExecutionConfig,
        ForecastingConfig,
        GenerationConfig,
        ModelConfig,
        RepairConfig,
        ResidualConfig,
        RunConfig,
    )
    from mosaickv.evaluation.model import EvaluationRequest
    from mosaickv.types import (
        Backend,
        BudgetUnit,
        ForecastMode,
        MosaicKVMethod,
        Precision,
        RepairPolicy,
    )

    precision = Precision.BF16 if adapter.model.dtype == torch.bfloat16 else Precision.FP16
    dataset_revision = sha256(b"adapter-three-tier-validation-v1").hexdigest()

    def config(method: MosaicKVMethod, retention_ratio: float) -> RunConfig:
        repair_enabled = method is MosaicKVMethod.MOSAICKV_FULL and retention_ratio < 1.0
        return RunConfig(
            model=ModelConfig(model_id, revision, precision),
            dataset=DatasetConfig(
                "mosaickv/adapter-three-tier-validation",
                dataset_revision,
                "validation",
            ),
            execution=ExecutionConfig(Backend.HUGGINGFACE, "eager", 0, True),
            generation=GenerationConfig(max_new_tokens=4),
            cache=CacheConfig(2_147_483_647, BudgetUnit.BYTES, retention_ratio, 16),
            method=method,
            forecasting=ForecastingConfig(
                enabled=method is not MosaicKVMethod.FULLKV,
                mode=ForecastMode.PROMPT_WINDOW,
                prompt_window=8,
                draft_steps=0,
                centroid_count=4,
            ),
            residual=ResidualConfig(
                enabled=method is MosaicKVMethod.MOSAICKV_FULL,
                require_pinned_memory=repair_enabled,
            ),
            repair=RepairConfig(
                enabled=repair_enabled,
                policy=RepairPolicy.ENTROPY if repair_enabled else RepairPolicy.NONE,
                entropy_threshold=0.0,
                max_blocks_per_step=1 if repair_enabled else 0,
            ),
        )

    request = EvaluationRequest(
        run_id="adapter-three-tier-validation",
        sample_id="single-image",
        task="synthetic_smoke",
        messages=messages,
        generation_kwargs={},
    )
    scratch_tmp = Path(os.environ["TMPDIR"])
    with tempfile.TemporaryDirectory(prefix="hf-three-tier-", dir=scratch_tmp) as raw_directory:
        directory = Path(raw_directory)
        runs: dict[str, tuple[Any, dict[str, Any]]] = {}
        for method, ratio in (
            (MosaicKVMethod.FULLKV, 1.0),
            (MosaicKVMethod.MOSAICKV_PROTO, 1.0),
            (MosaicKVMethod.MOSAICKV_FULL, 1.0),
            (MosaicKVMethod.MOSAICKV_PROTO, 0.5),
            (MosaicKVMethod.MOSAICKV_FULL, 0.5),
        ):
            run_request = EvaluationRequest(
                run_id=f"{method.value}-{ratio}",
                sample_id=request.sample_id,
                task=request.task,
                messages=request.messages,
                generation_kwargs=request.generation_kwargs,
            )
            output = HuggingFaceMosaicKVModel(
                adapter,
                config(method, ratio),
                trace_directory=directory,
            ).generate(run_request)
            trace_path = next((directory / run_request.run_id).glob("*.json"))
            runs[f"{method.value}:{ratio}"] = (
                output,
                json.loads(trace_path.read_text(encoding="utf-8")),
            )
        reference_trace = runs["fullkv:1.0"][1]
        for identity in ("mosaickv_proto:1.0", "mosaickv_full:1.0"):
            output, trace = runs[identity]
            assert trace["generated_token_ids"] == reference_trace["generated_token_ids"]
            assert output.metrics.compression_time == 0.0
            assert trace["tier_mode"] == "retention_one"
        for identity in ("mosaickv_proto:0.5", "mosaickv_full:0.5"):
            output, trace = runs[identity]
            assert output.metrics.generated_tokens == 4
            assert trace["tier_mode"] == "three_tier"
            assert trace["prototype_safety"]["safe"] is True
            assert trace["prototypes"]
            assert all(event["re_decode_count"] <= 1 for event in trace["repair_events"])
        full_trace = runs["mosaickv_full:0.5"][1]
        triggered = [event for event in full_trace["repair_events"] if event["triggered"]]
        assert triggered, "forced entropy policy did not exercise residual restoration"
        assert all(event["restored_block_ids"] for event in triggered)
        assert all(event["restored_bytes"] > 0 for event in triggered)

        provenance["three_tier_runtime"] = {
            "model_id": model_id,
            "model_revision": revision,
            "retention_one_token_identity": True,
            "compressed_tier_modes": {
                identity: runs[identity][1]["tier_mode"]
                for identity in ("mosaickv_proto:0.5", "mosaickv_full:0.5")
            },
            "prototype_phase_transform": full_trace["prototype_safety"][
                "phase_transform_provenance"
            ],
            "triggered_repairs": len(triggered),
            "maximum_redecode_count": max(
                event["re_decode_count"] for event in full_trace["repair_events"]
            ),
        }
        print(json.dumps(provenance, sort_keys=True))
