"""Hugging Face adapter for the LLaVA-v1.6 Mistral checkpoint.

This adapter exists to validate the paper-faithful ``vl_cache_reimpl`` on a
model family used by the VL-Cache paper. It deliberately exposes only exact
cache selection; prototype and residual-repair support require a separate
checkpoint-specific RoPE audit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Self

from mosaickv.adapters.huggingface.base import (
    HuggingFaceMultimodalAdapter,
    validate_hf_revision,
)
from mosaickv.adapters.huggingface.types import (
    AdapterCapabilities,
    CachedKeyState,
    QueryVectorState,
)


class LlavaNextMistralAdapter(HuggingFaceMultimodalAdapter):
    """Exact-cache adapter for ``llava-hf/llava-v1.6-mistral-7b-hf``.

    Transformers' Mistral attention applies rotary embeddings before updating
    ``past_key_values``. Keys are therefore post-RoPE. This is sufficient for
    FullKV and exact VL-Cache selection, but it is not evidence that prototype
    averaging is phase-safe.
    """

    capabilities = AdapterCapabilities(
        model_family="llava_next_mistral",
        architectures=("LlavaNextForConditionalGeneration",),
        attention_implementations=("eager",),
        image=True,
        multi_image=False,
        video=False,
        cache_classes=("Cache", "DynamicCache", "legacy tuple"),
        cache_sequence_dimension=-2,
        cached_key_state=CachedKeyState.POST_ROPE,
        query_vector_state=QueryVectorState.Q_PROJ_PRE_ROPE,
        supports_prototype_merge=False,
        supports_residual_repair=False,
        notes=(
            "validated initially only for eager FullKV and exact VL-Cache trend reproduction",
            "prototype merging and residual repair are fail-closed pending a RoPE audit",
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
        """Load an immutable LLaVA-NeXT checkpoint."""

        validate_hf_revision(revision)
        from transformers import AutoProcessor, LlavaNextForConditionalGeneration

        selected_model_kwargs = dict(model_kwargs or {})
        attention = selected_model_kwargs.setdefault("attn_implementation", "eager")
        if attention not in cls.capabilities.attention_implementations:
            raise ValueError(f"unsupported LLaVA-NeXT attention implementation: {attention}")
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            revision=revision,
            **selected_model_kwargs,
        )
        processor = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
            model_id,
            revision=revision,
            **dict(processor_kwargs or {}),
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
        if getattr(self.processor, "chat_template", None):
            return super()._render_prompt(chat)
        if len(chat) != 1 or chat[0]["role"] != "user":
            raise ValueError("the audited LLaVA-v1.6 prompt path requires one user message")
        content = "".join(
            "<image>\n" if part["type"] == "image" else str(part["text"])
            for part in chat[0]["content"]
        )
        return f"[INST] {content} [/INST]"

    def _processor_kwargs(
        self,
        prompt: str,
        images: Sequence[Any],
        videos: Sequence[Any],
    ) -> dict[str, Any]:
        if videos:
            raise ValueError("LLaVA-v1.6 Mistral does not accept video inputs")
        if len(images) != 1:
            raise ValueError("the audited LLaVA-v1.6 path requires exactly one image")
        return {
            "text": [prompt],
            "images": list(images),
            "return_tensors": "pt",
            "padding": False,
        }


__all__ = ["LlavaNextMistralAdapter"]
