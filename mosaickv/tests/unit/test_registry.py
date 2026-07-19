from __future__ import annotations

from typing import cast

import pytest

from mosaickv.adapters import AdapterRegistry, default_registry
from mosaickv.types import JsonObject


def test_default_registry_inspects_without_runtime_load() -> None:
    registry = default_registry()
    adapter = registry.resolve("Qwen/Qwen2.5-VL-3B-Instruct")
    report = adapter.inspect("Qwen/Qwen2.5-VL-3B-Instruct")
    assert report["runtime_load_verified"] is False
    assert report["loads_model_weights"] is False
    capabilities = cast("JsonObject", report["capabilities"])
    assert capabilities["video"] is True


def test_unknown_model_fails_closed() -> None:
    with pytest.raises(LookupError, match="no adapter matches"):
        default_registry().resolve("unknown/model")


def test_duplicate_adapter_name_is_rejected() -> None:
    adapter = default_registry().resolve("Qwen/Qwen2.5-VL-3B-Instruct")
    registry = AdapterRegistry([adapter])
    with pytest.raises(ValueError, match="already registered"):
        registry.register(adapter)
