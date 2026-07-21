"""Optional eager adapter for the pinned InternVL2.5 remote-code API."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Self, cast

from mosaickv.adapters.huggingface.base import (
    HuggingFaceMultimodalAdapter,
    _torch,
    validate_hf_revision,
)
from mosaickv.adapters.huggingface.types import (
    AdapterCapabilities,
    CachedKeyState,
    DecodeState,
    Modality,
    ModalitySpan,
    QueryVectorState,
)


@dataclass(frozen=True, slots=True)
class InternVLVideo:
    """Preprocessed InternVL video frames and patch counts per sampled frame."""

    pixel_values: Any
    num_patches_list: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.num_patches_list or any(count <= 0 for count in self.num_patches_list):
            raise ValueError("num_patches_list must contain positive frame patch counts")


class InternVL25Adapter(HuggingFaceMultimodalAdapter):
    """Cache adapter for ``InternVLChatModel(Qwen2ForCausalLM)``.

    The public checkpoint API does not ship an ``AutoProcessor``.  Image media
    must therefore already be transformed to InternVL pixel tensors of shape
    ``[num_patches, C, H, W]``.  Videos use :class:`InternVLVideo`, matching the
    model card's frame-to-image preprocessing contract.  Prefill uses the
    remote multimodal wrapper; token decode calls its public ``language_model``
    because ``InternVLChatModel.forward`` always extracts image features.
    """

    capabilities = AdapterCapabilities(
        model_family="internvl2_5",
        architectures=("InternVLChatModel",),
        attention_implementations=("eager",),
        image=True,
        multi_image=True,
        video=True,
        cache_classes=("Cache", "DynamicCache", "legacy tuple"),
        cache_sequence_dimension=-2,
        cached_key_state=CachedKeyState.POST_ROPE,
        query_vector_state=QueryVectorState.Q_PROJ_PRE_ROPE,
        supports_prototype_merge=False,
        supports_residual_repair=False,
        notes=(
            "optional adapter: preprocessing remains checkpoint-specific public remote code",
            "load with trust_remote_code=True at an immutable revision and use_flash_attn=False",
            "Qwen2Attention applies RoPE before Cache.update",
        ),
    )

    _placeholder_pattern = re.compile(r"<(image|video)>")

    def __init__(self, model: Any, processor: Any, *, device: Any | None = None) -> None:
        self._pending_visual_modalities: tuple[Modality, ...] = ()
        super().__init__(model, processor, device=device)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        revision: str,
        model_kwargs: Mapping[str, Any] | None = None,
        processor_kwargs: Mapping[str, Any] | None = None,
    ) -> Self:
        """Load pinned checkpoint remote code with FlashAttention disabled."""

        validate_hf_revision(revision)
        from transformers import AutoModel, AutoTokenizer

        selected_model_kwargs = dict(model_kwargs or {})
        selected_model_kwargs["trust_remote_code"] = True
        use_flash = selected_model_kwargs.setdefault("use_flash_attn", False)
        if use_flash:
            raise ValueError("InternVL25Adapter only supports use_flash_attn=False")
        model = AutoModel.from_pretrained(model_id, revision=revision, **selected_model_kwargs)
        tokenizer_kwargs = dict(processor_kwargs or {})
        tokenizer_kwargs["trust_remote_code"] = True
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, **tokenizer_kwargs)
        return cls(model, tokenizer)

    def _language_layers(self) -> Sequence[Any]:
        return cast("Sequence[Any]", self.model.language_model.model.layers)

    def _image_token_id(self) -> int | None:
        token_id = getattr(self.model, "img_context_token_id", None)
        return None if token_id is None else int(token_id)

    def _video_token_id(self) -> int | None:
        # InternVL represents each sampled frame with the same IMG_CONTEXT token.
        return None

    def _render_prompt(self, chat: list[dict[str, Any]]) -> str:
        template = copy.deepcopy(self.model.conv_template)
        system_parts: list[str] = []
        for message in chat:
            content = "".join(
                str(part["text"]) if part["type"] == "text" else f"<{part['type']}>\n"
                for part in message["content"]
            )
            role = message["role"]
            if role == "system":
                system_parts.append(content)
            elif role == "user":
                template.append_message(template.roles[0], content)
            elif role == "assistant":
                template.append_message(template.roles[1], content)
            else:
                raise ValueError(f"InternVL does not support role {role!r}")
        if system_parts:
            template.system_message = "\n".join(system_parts)
        if not chat or chat[-1]["role"] != "assistant":
            template.append_message(template.roles[1], None)
        return str(template.get_prompt())

    def _processor_kwargs(
        self, prompt: str, images: Sequence[Any], videos: Sequence[Any]
    ) -> dict[str, Any]:
        del prompt, images, videos
        raise AssertionError("InternVL uses its tokenizer and preprocessed media directly")

    def _process_prompt(
        self, prompt: str, images: Sequence[Any], videos: Sequence[Any]
    ) -> Mapping[str, Any]:
        torch = _torch()
        image_queue = list(images)
        video_queue = list(videos)
        pixel_batches: list[Any] = []
        visual_modalities: list[Modality] = []
        image_context_token = "<IMG_CONTEXT>"
        image_start_token = "<img>"
        image_end_token = "</img>"

        def image_tokens(num_patches: int, modality: Modality) -> str:
            if num_patches <= 0:
                raise ValueError("InternVL media must contain at least one patch")
            visual_modalities.append(modality)
            contexts = image_context_token * int(self.model.num_image_token) * num_patches
            return image_start_token + contexts + image_end_token

        def replacement(match: re.Match[str]) -> str:
            kind = match.group(1)
            if kind == "image":
                if not image_queue:
                    raise ValueError("InternVL prompt has more image markers than payloads")
                pixels = image_queue.pop(0)
                self._validate_pixel_tensor(pixels)
                pixel_batches.append(pixels)
                return image_tokens(int(pixels.shape[0]), Modality.IMAGE)
            if not video_queue:
                raise ValueError("InternVL prompt has more video markers than payloads")
            video = video_queue.pop(0)
            if not isinstance(video, InternVLVideo):
                raise TypeError("InternVL videos must be InternVLVideo instances")
            self._validate_pixel_tensor(video.pixel_values)
            if sum(video.num_patches_list) != int(video.pixel_values.shape[0]):
                raise ValueError("InternVLVideo patch counts do not match pixel_values")
            pixel_batches.append(video.pixel_values)
            frames = [
                f"Frame{index}: {image_tokens(count, Modality.VIDEO)}"
                for index, count in enumerate(video.num_patches_list, start=1)
            ]
            return "\n".join(frames)

        expanded_prompt = self._placeholder_pattern.sub(replacement, prompt)
        if image_queue or video_queue:
            raise ValueError("InternVL received media without corresponding prompt markers")
        self._pending_visual_modalities = tuple(visual_modalities)
        self.model.img_context_token_id = self.processor.convert_tokens_to_ids(image_context_token)
        tokenized = dict(self.processor(expanded_prompt, return_tensors="pt"))
        if pixel_batches:
            pixel_values = torch.cat(pixel_batches, dim=0)
            tokenized["pixel_values"] = pixel_values
            tokenized["image_flags"] = torch.ones(
                (pixel_values.shape[0], 1), dtype=torch.long, device=pixel_values.device
            )
        return tokenized

    @staticmethod
    def _validate_pixel_tensor(value: Any) -> None:
        if getattr(value, "ndim", None) != 4:
            raise ValueError("InternVL pixel tensors must have shape [num_patches, C, H, W]")

    def _modality_spans(self, input_ids: Any) -> tuple[ModalitySpan, ...]:
        context_id = self._image_token_id()
        if context_id is None:
            raise RuntimeError("InternVL image context token ID was not initialized")
        ids = [int(value) for value in input_ids[0].detach().cpu().tolist()]
        visual_index = 0
        modalities: list[Modality] = []
        inside_context = False
        for token_id in ids:
            if token_id == context_id:
                if not inside_context:
                    if visual_index >= len(self._pending_visual_modalities):
                        raise RuntimeError("more IMG_CONTEXT runs than prepared media patches")
                    current = self._pending_visual_modalities[visual_index]
                    visual_index += 1
                    inside_context = True
                modalities.append(current)
            else:
                inside_context = False
                modalities.append(Modality.TEXT)
        if visual_index != len(self._pending_visual_modalities):
            raise RuntimeError("fewer IMG_CONTEXT runs than prepared media patches")
        spans: list[ModalitySpan] = []
        start = 0
        for index in range(1, len(modalities) + 1):
            if index == len(modalities) or modalities[index] != modalities[start]:
                spans.append(ModalitySpan(start, index, modalities[start]))
                start = index
        return tuple(spans)

    def _prefill_forward(self, forward_inputs: dict[str, Any]) -> Any:
        selected = dict(forward_inputs)
        selected.pop("cache_position", None)
        selected.pop("logits_to_keep", None)
        if "pixel_values" in selected:
            return self.model(**selected)
        selected.pop("image_flags", None)
        return self.model.language_model(**selected)

    def _decode_forward(
        self, token_id: Any, state: DecodeState, attention_mask: Any, cache_position: Any
    ) -> Any:
        return self.model.language_model(
            input_ids=token_id,
            attention_mask=attention_mask,
            past_key_values=state.past_key_values,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )


__all__ = ["InternVL25Adapter", "InternVLVideo"]
