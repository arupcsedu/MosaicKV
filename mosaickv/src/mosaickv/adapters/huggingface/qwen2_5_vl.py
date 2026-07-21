"""Eager Hugging Face adapter for Qwen2.5-VL."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Self, cast

from mosaickv.adapters.huggingface.base import HuggingFaceMultimodalAdapter, validate_hf_revision
from mosaickv.adapters.huggingface.types import (
    AdapterCapabilities,
    CachedKeyState,
    DecodeState,
    QueryVectorState,
)


class Qwen25VLAdapter(HuggingFaceMultimodalAdapter):
    """Adapter shared by Qwen2.5-VL 3B and 7B checkpoints.

    ``Qwen2_5_VLAttention.forward`` calls
    ``apply_multimodal_rotary_pos_emb`` before ``Cache.update``.  Cache keys are
    post-M-RoPE.  The wrapper's mutable ``rope_deltas`` is copied into each
    decode state and restored before every token step.
    """

    capabilities = AdapterCapabilities(
        model_family="qwen2_5_vl",
        architectures=("Qwen2_5_VLForConditionalGeneration",),
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
        notes=(
            "M-RoPE deltas are session state and are restored before decode",
            "eager attention is the only correctness-gated implementation",
        ),
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
        """Load a pinned Qwen2.5-VL checkpoint explicitly in eager mode."""

        validate_hf_revision(revision)
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        selected_model_kwargs = dict(model_kwargs or {})
        attention = selected_model_kwargs.setdefault("attn_implementation", "eager")
        if attention != "eager":
            raise ValueError("Qwen25VLAdapter only supports attn_implementation='eager'")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, revision=revision, **selected_model_kwargs
        )
        selected_processor_kwargs = dict(processor_kwargs or {})
        # Transformers 5 switched Qwen-VL to a fast image processor by
        # default. Pin the checkpoint-compatible slow path so HF, vLLM, and
        # SGLang do not silently compare different media preprocessing.
        selected_processor_kwargs.setdefault("use_fast", False)
        processor = AutoProcessor.from_pretrained(
            model_id, revision=revision, **selected_processor_kwargs
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

    def _capture_model_state(self, output: Any) -> dict[str, Any]:
        rope_deltas = getattr(output, "rope_deltas", None)
        if rope_deltas is None:
            rope_deltas = getattr(self.model.model, "rope_deltas", None)
        if rope_deltas is None:
            raise RuntimeError("Qwen2.5-VL prefill did not produce rope_deltas")
        return {"rope_deltas": rope_deltas.detach().clone()}

    def _restore_model_state(self, state: DecodeState) -> None:
        rope_deltas = state.model_state.get("rope_deltas")
        if rope_deltas is None:
            raise RuntimeError("Qwen2.5-VL decode state is missing rope_deltas")
        self.model.model.rope_deltas = rope_deltas


__all__ = ["Qwen25VLAdapter"]
