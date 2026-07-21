"""Model-adapter interfaces and metadata registry."""

from mosaickv.adapters.base import ModelAdapter, ModelCapabilities, StaticModelAdapter
from mosaickv.adapters.huggingface import (
    AdapterCapabilities,
    InternVL25Adapter,
    Llava15Adapter,
    LlavaOneVisionAdapter,
    Qwen25VLAdapter,
    load_hf_adapter,
)
from mosaickv.adapters.registry import AdapterRegistry, default_registry

__all__ = [
    "AdapterCapabilities",
    "AdapterRegistry",
    "InternVL25Adapter",
    "Llava15Adapter",
    "LlavaOneVisionAdapter",
    "ModelAdapter",
    "ModelCapabilities",
    "Qwen25VLAdapter",
    "StaticModelAdapter",
    "default_registry",
    "load_hf_adapter",
]
