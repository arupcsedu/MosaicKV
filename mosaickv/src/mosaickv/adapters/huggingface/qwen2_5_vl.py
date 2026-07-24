"""Eager Hugging Face adapter for Qwen2.5-VL."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike, fspath
from typing import Any, Self

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
    # Installed lmms-eval 0.7.2 Qwen2.5-VL image and frame-count defaults.
    # Eager attention cannot fit its 1,605,632-pixel video-frame ceiling on an
    # 80 GiB A100, so video frames use Qwen's documented 128-token minimum.
    _MIN_PIXELS = 256 * 28 * 28
    _MAX_PIXELS = 1_605_632
    _MAX_VIDEO_FRAMES = 32
    _VIDEO_FRAME_PIXELS = 128 * 28 * 28

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
        selected_processor_kwargs.setdefault("min_pixels", cls._MIN_PIXELS)
        selected_processor_kwargs.setdefault("max_pixels", cls._MAX_PIXELS)
        processor = AutoProcessor.from_pretrained(
            model_id, revision=revision, **selected_processor_kwargs
        )
        return cls(model, processor)

    def _language_layers(self) -> Sequence[Any]:
        return self._standard_language_layers()

    def _prefill_forward(self, forward_inputs: dict[str, Any]) -> Any:
        # Transformers 4.49's monolithic Qwen2.5-VL wrapper always computes
        # full logits and does not yet accept the later logits_to_keep option.
        selected = dict(forward_inputs)
        selected.pop("logits_to_keep", None)
        return self.model(**selected)

    def _decode_forward(
        self,
        token_id: Any,
        state: DecodeState,
        attention_mask: Any,
        cache_position: Any,
        *,
        output_attentions: bool = False,
    ) -> Any:
        return self.model(
            input_ids=token_id,
            attention_mask=attention_mask,
            past_key_values=state.past_key_values,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
            output_attentions=output_attentions,
        )

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
            path_videos = [isinstance(video, str | PathLike) for video in videos]
            if any(path_videos) and not all(path_videos):
                raise ValueError("Qwen2.5-VL does not accept mixed video paths and decoded videos")
            if all(path_videos):
                from qwen_vl_utils import fetch_video  # type: ignore[import-untyped]

                decoded: list[Any] = []
                for video in videos:
                    frames, _sample_fps = fetch_video(
                        {
                            "type": "video",
                            "video": fspath(video),
                            "min_pixels": self._VIDEO_FRAME_PIXELS,
                            "max_pixels": self._VIDEO_FRAME_PIXELS,
                        },
                        return_video_sample_fps=True,
                    )
                    if int(frames.shape[0]) > self._MAX_VIDEO_FRAMES:
                        import numpy as np

                        indices = np.linspace(
                            0,
                            int(frames.shape[0]) - 1,
                            self._MAX_VIDEO_FRAMES,
                            dtype=int,
                        )
                        frames = frames[np.unique(indices)]
                    decoded.append(frames)
                kwargs["videos"] = decoded
            else:
                kwargs["videos"] = list(videos)
        return kwargs

    def _capture_model_state(self, output: Any) -> dict[str, Any]:
        rope_deltas = getattr(output, "rope_deltas", None)
        if rope_deltas is None:
            rope_deltas = getattr(self.model, "rope_deltas", None)
        if rope_deltas is None:
            rope_deltas = getattr(self.model.model, "rope_deltas", None)
        if rope_deltas is None:
            raise RuntimeError("Qwen2.5-VL prefill did not produce rope_deltas")
        return {"rope_deltas": rope_deltas.detach().clone()}

    def _restore_model_state(self, state: DecodeState) -> None:
        rope_deltas = state.model_state.get("rope_deltas")
        if rope_deltas is None:
            raise RuntimeError("Qwen2.5-VL decode state is missing rope_deltas")
        if hasattr(self.model, "rope_deltas"):
            self.model.rope_deltas = rope_deltas
        else:
            self.model.model.rope_deltas = rope_deltas


__all__ = ["Qwen25VLAdapter"]
