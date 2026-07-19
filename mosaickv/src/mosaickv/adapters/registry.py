"""Deterministic model-adapter registry."""

from __future__ import annotations

from collections.abc import Iterable

from mosaickv.adapters.base import ModelAdapter, ModelCapabilities, StaticModelAdapter


class AdapterRegistry:
    """Registry that fails closed on duplicate or ambiguous adapter matches."""

    def __init__(self, adapters: Iterable[ModelAdapter] = ()) -> None:
        self._adapters: dict[str, ModelAdapter] = {}
        for adapter in adapters:
            self.register(adapter)

    def register(self, adapter: ModelAdapter) -> None:
        if adapter.name in self._adapters:
            raise ValueError(f"adapter {adapter.name!r} is already registered")
        self._adapters[adapter.name] = adapter

    def resolve(self, model_id: str) -> ModelAdapter:
        matches = [adapter for adapter in self._adapters.values() if adapter.matches(model_id)]
        if not matches:
            known = ", ".join(sorted(self._adapters))
            raise LookupError(f"no adapter matches model {model_id!r}; known adapters: {known}")
        if len(matches) > 1:
            names = ", ".join(sorted(adapter.name for adapter in matches))
            raise LookupError(f"ambiguous adapter match for model {model_id!r}: {names}")
        return matches[0]

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))


def default_registry() -> AdapterRegistry:
    """Return adapters for the five models in the audited capability matrix."""

    return AdapterRegistry(
        [
            StaticModelAdapter(
                name="llava_1_5_hf",
                pattern=r"llava-hf/llava-1\.5-7b-hf",
                audited_revision="b234b804b114d9e37bb655e11cbbb5f5e971b7a9",
                architecture="LlavaForConditionalGeneration",
                cache_layout="32 x (K,V)[B,32,S,128]",
                capabilities=ModelCapabilities(True, True, False, True, True, True, True, True),
            ),
            StaticModelAdapter(
                name="qwen2_5_vl_3b",
                pattern=r"Qwen/Qwen2\.5-VL-3B-Instruct",
                audited_revision="66285546d2b821cf421d4f5eb2576359d3770cd3",
                architecture="Qwen2_5_VLForConditionalGeneration",
                cache_layout="36 x (K,V)[B,2,S,128]",
                capabilities=ModelCapabilities(True, True, True, True, True, True, True, True),
            ),
            StaticModelAdapter(
                name="qwen2_5_vl_7b",
                pattern=r"Qwen/Qwen2\.5-VL-7B-Instruct",
                audited_revision="cc594898137f460bfe9f0759e9844b3ce807cfb5",
                architecture="Qwen2_5_VLForConditionalGeneration",
                cache_layout="28 x (K,V)[B,4,S,128]",
                capabilities=ModelCapabilities(True, True, True, True, True, True, True, True),
            ),
            StaticModelAdapter(
                name="internvl2_5_4b",
                pattern=r"OpenGVLab/InternVL2_5-4B",
                audited_revision="2cf4a8158bbc40d35015e7c63b527890de4d27b3",
                architecture="InternVLChatModel(Qwen2ForCausalLM)",
                cache_layout="36 x (K,V)[B,2,S,128]",
                capabilities=ModelCapabilities(True, True, True, True, True, True, True, True),
            ),
            StaticModelAdapter(
                name="llava_onevision_0_5b",
                pattern=r"llava-hf/llava-onevision-qwen2-0\.5b-ov-hf",
                audited_revision="74dd0bf867a4cda7950c17663794267c60cf4b40",
                architecture="LlavaOnevisionForConditionalGeneration",
                cache_layout="24 x (K,V)[B,2,S,64]",
                capabilities=ModelCapabilities(True, True, True, True, True, False, True, True),
            ),
        ]
    )


__all__ = ["AdapterRegistry", "default_registry"]
