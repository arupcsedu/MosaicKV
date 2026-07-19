"""Model-adapter interfaces and metadata registry."""

from mosaickv.adapters.base import ModelAdapter, ModelCapabilities, StaticModelAdapter
from mosaickv.adapters.registry import AdapterRegistry, default_registry

__all__ = [
    "AdapterRegistry",
    "ModelAdapter",
    "ModelCapabilities",
    "StaticModelAdapter",
    "default_registry",
]
