"""Eager Hugging Face adapter for LLaVA-OneVision."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Self, cast

from mosaickv.adapters.huggingface.base import HuggingFaceMultimodalAdapter, validate_hf_revision
from mosaickv.adapters.huggingface.types import (
    AdapterCapabilities,
    CachedKeyState,
    QueryVectorState,
)


class LlavaOneVisionAdapter(HuggingFaceMultimodalAdapter):
    """Adapter for the native Transformers LLaVA-OneVision wrapper.

    The checkpoint's language model uses ``Qwen2Attention``.  Its forward
    applies RoPE before ``Cache.update``, so K is post-RoPE and q_proj hooks
    observe pre-RoPE query projections.
    """

    capabilities = AdapterCapabilities(
        model_family="llava_onevision",
        architectures=("LlavaOnevisionForConditionalGeneration",),
        attention_implementations=("eager",),
        image=True,
        multi_image=True,
        video=True,
        cache_classes=("Cache", "DynamicCache"),
        cache_sequence_dimension=-2,
        cached_key_state=CachedKeyState.POST_ROPE,
        query_vector_state=QueryVectorState.Q_PROJ_PRE_ROPE,
        supports_prototype_merge=False,
        supports_residual_repair=False,
        notes=("eager attention is the only correctness-gated implementation",),
    )

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        revision: str,
        model_kwargs: Mapping[str, Any] | None = None,
        processor_kwargs: Mapping[str, Any] | None = None,
    ) -> Self:
        """Load a pinned LLaVA-OneVision checkpoint explicitly in eager mode."""

        validate_hf_revision(revision)
        from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

        selected_model_kwargs = dict(model_kwargs or {})
        attention = selected_model_kwargs.setdefault("attn_implementation", "eager")
        if attention != "eager":
            raise ValueError("LlavaOneVisionAdapter only supports attn_implementation='eager'")
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_id, revision=revision, **selected_model_kwargs
        )
        processor = AutoProcessor.from_pretrained(
            model_id, revision=revision, **dict(processor_kwargs or {})
        )
        return cls(model, processor)

    def _language_layers(self) -> Sequence[Any]:
        return cast("Sequence[Any]", self.model.model.language_model.layers)

    def _image_token_id(self) -> int | None:
        return int(self.model.config.image_token_id)

    def _video_token_id(self) -> int | None:
        return int(self.model.config.video_token_id)

    def _processor_kwargs(
        self, prompt: str, images: Sequence[Any], videos: Sequence[Any]
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"text": [prompt], "return_tensors": "pt", "padding": False}
        if images:
            kwargs["images"] = list(images)
        if videos:
            kwargs["videos"] = list(videos)
        return kwargs


__all__ = ["LlavaOneVisionAdapter"]
