"""Runtime Hugging Face adapter resolution without importing torch eagerly."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mosaickv.adapters.huggingface.base import (
    HuggingFaceMultimodalAdapter,
    validate_hf_revision,
)
from mosaickv.adapters.huggingface.internvl import InternVL25Adapter
from mosaickv.adapters.huggingface.llava import Llava15Adapter
from mosaickv.adapters.huggingface.llava_onevision import LlavaOneVisionAdapter
from mosaickv.adapters.huggingface.qwen2_5_vl import Qwen25VLAdapter

RuntimeAdapterType = type[HuggingFaceMultimodalAdapter]

_MODEL_IDS: dict[str, RuntimeAdapterType] = {
    "llava-hf/llava-1.5-7b-hf": Llava15Adapter,
    "Qwen/Qwen2.5-VL-3B-Instruct": Qwen25VLAdapter,
    "Qwen/Qwen2.5-VL-7B-Instruct": Qwen25VLAdapter,
    "llava-hf/llava-onevision-qwen2-0.5b-ov-hf": LlavaOneVisionAdapter,
    "OpenGVLab/InternVL2_5-4B": InternVL25Adapter,
}

_AUDITED_REVISIONS: dict[str, str] = {
    "llava-hf/llava-1.5-7b-hf": "b234b804b114d9e37bb655e11cbbb5f5e971b7a9",
    "Qwen/Qwen2.5-VL-3B-Instruct": "66285546d2b821cf421d4f5eb2576359d3770cd3",
    "Qwen/Qwen2.5-VL-7B-Instruct": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
    "llava-hf/llava-onevision-qwen2-0.5b-ov-hf": ("74dd0bf867a4cda7950c17663794267c60cf4b40"),
    "OpenGVLab/InternVL2_5-4B": "2cf4a8158bbc40d35015e7c63b527890de4d27b3",
}

_ARCHITECTURES: dict[str, RuntimeAdapterType] = {
    architecture: adapter_type
    for adapter_type in set(_MODEL_IDS.values())
    for architecture in adapter_type.capabilities.architectures
}


def _resolve_model_source(
    model_id: str,
    revision: str,
    *,
    local_files_only: bool,
) -> str:
    if not local_files_only:
        return model_id
    cache_dir = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME")
    if not cache_dir:
        raise RuntimeError("offline HF loading requires HF_HUB_CACHE or HF_HOME")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("offline HF loading requires huggingface_hub") from exc
    try:
        snapshot = snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"pinned model snapshot is unavailable offline: {model_id}@{revision}"
        ) from exc
    snapshot_path = Path(snapshot).resolve()
    if not snapshot_path.is_dir():
        raise RuntimeError(f"resolved model snapshot is not a directory: {snapshot_path}")
    return str(snapshot_path)


def runtime_adapter_class(model_id: str) -> RuntimeAdapterType:
    """Resolve only explicitly audited model IDs and fail closed otherwise."""

    try:
        return _MODEL_IDS[model_id]
    except KeyError as exc:
        known = ", ".join(sorted(_MODEL_IDS))
        raise LookupError(f"no runtime HF adapter for {model_id!r}; known models: {known}") from exc


def audited_model_revision(model_id: str) -> str:
    """Return the immutable revision audited by the repository capability matrix."""

    runtime_adapter_class(model_id)
    return _AUDITED_REVISIONS[model_id]


def adapter_for_loaded_model(model: Any, processor: Any) -> HuggingFaceMultimodalAdapter:
    """Wrap a loaded model based on its exact public architecture class."""

    architecture = type(model).__name__
    try:
        adapter_type = _ARCHITECTURES[architecture]
    except KeyError as exc:
        known = ", ".join(sorted(_ARCHITECTURES))
        raise LookupError(
            f"no runtime HF adapter for architecture {architecture!r}; known: {known}"
        ) from exc
    return adapter_type(model, processor)


def load_hf_adapter(
    model_id: str,
    *,
    revision: str,
    model_kwargs: Mapping[str, Any] | None = None,
    processor_kwargs: Mapping[str, Any] | None = None,
) -> HuggingFaceMultimodalAdapter:
    """Load one audited, immutable checkpoint through its runtime adapter."""

    adapter_type = runtime_adapter_class(model_id)
    validate_hf_revision(revision)
    selected_model_kwargs = dict(model_kwargs or {})
    selected_processor_kwargs = dict(processor_kwargs or {})
    local_files_only = bool(
        selected_model_kwargs.get("local_files_only", False)
        or selected_processor_kwargs.get("local_files_only", False)
    )
    model_source = _resolve_model_source(
        model_id,
        revision,
        local_files_only=local_files_only,
    )
    return adapter_type.from_pretrained(  # type: ignore[attr-defined,no-any-return]
        model_source,
        revision=revision,
        model_kwargs=selected_model_kwargs,
        processor_kwargs=selected_processor_kwargs,
    )


__all__ = [
    "adapter_for_loaded_model",
    "audited_model_revision",
    "load_hf_adapter",
    "runtime_adapter_class",
]
