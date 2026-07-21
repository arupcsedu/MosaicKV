"""Eager Hugging Face adapter for LLaVA-1.5."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Self

from mosaickv.adapters.huggingface.base import HuggingFaceMultimodalAdapter, validate_hf_revision
from mosaickv.adapters.huggingface.types import (
    AdapterCapabilities,
    CachedKeyState,
    QueryVectorState,
)


class Llava15Adapter(HuggingFaceMultimodalAdapter):
    """Adapter for ``llava-hf/llava-1.5-7b-hf``.

    Transformers' ``LlamaAttention.forward`` applies rotary embeddings before
    ``Cache.update``.  The cached keys are therefore post-RoPE, while the hook
    on ``q_proj`` captures pre-RoPE query projections.
    """

    capabilities = AdapterCapabilities(
        model_family="llava_1_5",
        architectures=("LlavaForConditionalGeneration",),
        attention_implementations=("eager",),
        image=True,
        multi_image=True,
        video=False,
        cache_classes=("Cache", "DynamicCache", "legacy tuple"),
        cache_sequence_dimension=-2,
        cached_key_state=CachedKeyState.POST_ROPE,
        query_vector_state=QueryVectorState.Q_PROJ_PRE_ROPE,
        supports_prototype_merge=False,
        supports_residual_repair=False,
        notes=(
            "eager attention is the only correctness-gated implementation",
            "prototype merge and residual repair remain disabled until correctness gates pass",
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
        """Load a pinned LLaVA checkpoint explicitly in eager mode."""

        validate_hf_revision(revision)
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        selected_model_kwargs = dict(model_kwargs or {})
        attention = selected_model_kwargs.setdefault("attn_implementation", "eager")
        if attention != "eager":
            raise ValueError("Llava15Adapter only supports attn_implementation='eager'")
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, revision=revision, **selected_model_kwargs
        )
        processor = AutoProcessor.from_pretrained(
            model_id, revision=revision, **dict(processor_kwargs or {})
        )
        return cls(model, processor)

    def _language_layers(self) -> Sequence[Any]:
        return self._standard_language_layers()

    def _image_token_id(self) -> int | None:
        value = getattr(self.model.config, "image_token_id", None)
        if value is None:
            value = getattr(self.model.config, "image_token_index", None)
        return int(value) if value is not None else None

    def _render_prompt(self, chat: list[dict[str, Any]]) -> str:
        template = getattr(self.processor, "chat_template", None)
        if template:
            return super()._render_prompt(chat)
        # The pinned LLaVA-1.5 model card specifies USER/ASSISTANT prompts.
        pieces: list[str] = []
        for message in chat:
            role = message["role"]
            content = "".join(
                "<image>\n" if part["type"] == "image" else str(part["text"])
                for part in message["content"]
            )
            if role == "system":
                pieces.append(content)
            elif role == "user":
                pieces.append(f"USER: {content}")
            elif role == "assistant":
                pieces.append(f"ASSISTANT: {content}")
            else:
                raise ValueError(f"LLaVA-1.5 does not support role {role!r}")
        if not chat or chat[-1]["role"] != "assistant":
            pieces.append("ASSISTANT:")
        return " ".join(pieces)

    def _processor_kwargs(
        self, prompt: str, images: Sequence[Any], videos: Sequence[Any]
    ) -> dict[str, Any]:
        if videos:
            raise ValueError("LLaVA-1.5 processor does not accept videos")
        kwargs: dict[str, Any] = {"text": [prompt], "return_tensors": "pt"}
        if images:
            kwargs["images"] = list(images)
        return kwargs


__all__ = ["Llava15Adapter"]
