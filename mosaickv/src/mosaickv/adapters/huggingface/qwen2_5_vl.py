"""Hugging Face adapter for Qwen2.5-VL."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib.metadata import version
from os import PathLike, fspath
from typing import Any, Self

from mosaickv.adapters.huggingface.base import HuggingFaceMultimodalAdapter, validate_hf_revision
from mosaickv.adapters.huggingface.types import (
    AdapterCapabilities,
    CachedKeyState,
    DecodeState,
    Modality,
    ModalitySpan,
    PreparedInputs,
    QueryVectorState,
)
from mosaickv.prototypes.rope import RotaryPrototypeKeyTransform

# This is the controlled image-preprocessing profile used by the HF quality
# harness and both serving-backend measurement wrappers.  Keep these public:
# vLLM accepts them through ``mm_processor_kwargs`` and the audited SGLang
# launcher applies them before constructing its tokenizer manager.
QWEN25_VL_MIN_PIXELS = 256 * 28 * 28
QWEN25_VL_MAX_PIXELS = 1_605_632


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
        attention_implementations=("eager", "sdpa", "flash_attention_2"),
        image=True,
        multi_image=True,
        video=True,
        cache_classes=("Cache", "DynamicCache"),
        cache_sequence_dimension=-2,
        cached_key_state=CachedKeyState.POST_ROPE,
        query_vector_state=QueryVectorState.Q_PROJ_PRE_ROPE,
        supports_prototype_merge=True,
        supports_residual_repair=True,
        notes=(
            "M-RoPE deltas are session state and are restored before decode",
            "prototype keys are de-rotated with exact prefill M-RoPE phases and "
            "re-rotated at anchors",
            "prototype and repair decode require eager attention; SDPA/FlashAttention-2 "
            "remain exact-only",
        ),
    )
    # Installed lmms-eval 0.7.2 Qwen2.5-VL image and frame-count defaults.
    # Eager attention cannot fit its 1,605,632-pixel video-frame ceiling on an
    # 80 GiB A100, so video frames use Qwen's documented 128-token minimum.
    _MIN_PIXELS = QWEN25_VL_MIN_PIXELS
    _MAX_PIXELS = QWEN25_VL_MAX_PIXELS
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
        """Load a pinned Qwen2.5-VL checkpoint with an audited attention backend."""

        validate_hf_revision(revision)
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        selected_model_kwargs = dict(model_kwargs or {})
        attention = selected_model_kwargs.setdefault("attn_implementation", "eager")
        if attention not in cls.capabilities.attention_implementations:
            raise ValueError(f"unsupported Qwen2.5-VL attention implementation: {attention}")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, revision=revision, **selected_model_kwargs
        )
        selected_processor_kwargs = dict(processor_kwargs or {})
        # Transformers 4.55.2 switched Qwen-VL to a fast image processor by
        # default. Pin the checkpoint-compatible slow path so HF, vLLM, and
        # SGLang do not silently compare different media preprocessing.
        selected_processor_kwargs.setdefault("use_fast", False)
        selected_processor_kwargs.setdefault("min_pixels", cls._MIN_PIXELS)
        selected_processor_kwargs.setdefault("max_pixels", cls._MAX_PIXELS)
        processor = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
            model_id, revision=revision, **selected_processor_kwargs
        )
        return cls(model, processor)

    def _language_layers(self) -> Sequence[Any]:
        return self._standard_language_layers()

    @staticmethod
    def _grid_rows(value: Any | None, name: str) -> tuple[tuple[int, int, int], ...]:
        if value is None:
            return ()
        raw = value.detach().cpu().tolist() if hasattr(value, "detach") else value
        rows = tuple(tuple(int(item) for item in row) for row in raw)
        if any(len(row) != 3 or any(item <= 0 for item in row) for row in rows):
            raise RuntimeError(f"Qwen2.5-VL {name} must contain positive [T,H,W] rows")
        return tuple((row[0], row[1], row[2]) for row in rows)

    @classmethod
    def enrich_modality_spans(
        cls,
        spans: Sequence[ModalitySpan],
        *,
        image_grid_thw: Any | None,
        video_grid_thw: Any | None,
        spatial_merge_size: int,
    ) -> tuple[ModalitySpan, ...]:
        """Attach exact Qwen visual-grid provenance to expanded token runs.

        The processor-provided ``*_grid_thw`` tensors are the authoritative
        mapping between visual patches and language tokens.  A mismatch is a
        correctness error: silently guessing a raster shape would create
        fabricated spatial graph edges.
        """

        if spatial_merge_size <= 0:
            raise ValueError("Qwen2.5-VL spatial_merge_size must be positive")
        image_specs = list(cls._grid_rows(image_grid_thw, "image_grid_thw"))
        video_specs = list(cls._grid_rows(video_grid_thw, "video_grid_thw"))
        image_index = 0
        video_index = 0
        enriched: list[ModalitySpan] = []
        for span in spans:
            if span.modality is Modality.TEXT:
                enriched.append(span)
                continue
            specs = image_specs if span.modality is Modality.IMAGE else video_specs
            source_name = "image_grid_thw" if span.modality is Modality.IMAGE else "video_grid_thw"
            if not specs:
                raise RuntimeError(
                    f"Qwen2.5-VL expanded {span.modality.value} tokens have no {source_name} row"
                )
            temporal, raw_rows, raw_columns = specs.pop(0)
            if raw_rows % spatial_merge_size or raw_columns % spatial_merge_size:
                raise RuntimeError(
                    f"Qwen2.5-VL {source_name} is not divisible by spatial_merge_size"
                )
            rows = raw_rows // spatial_merge_size
            columns = raw_columns // spatial_merge_size
            tokens_per_slice = rows * columns
            expected_tokens = temporal * tokens_per_slice
            if span.end - span.start != expected_tokens:
                raise RuntimeError(
                    f"Qwen2.5-VL {span.modality.value} token run has length "
                    f"{span.end - span.start}, but {source_name} requires {expected_tokens}"
                )
            if span.modality is Modality.IMAGE:
                if temporal != 1:
                    raise RuntimeError(
                        "Qwen2.5-VL image_grid_thw temporal dimension must equal one"
                    )
                enriched.append(
                    ModalitySpan(
                        span.start,
                        span.end,
                        Modality.IMAGE,
                        image_index=image_index,
                        grid_shape=(rows, columns),
                    )
                )
                image_index += 1
                continue
            for frame_index in range(temporal):
                start = span.start + frame_index * tokens_per_slice
                enriched.append(
                    ModalitySpan(
                        start,
                        start + tokens_per_slice,
                        Modality.VIDEO,
                        image_index=video_index,
                        frame_index=frame_index,
                        grid_shape=(rows, columns),
                    )
                )
            video_index += 1
        if image_specs or video_specs:
            raise RuntimeError(
                "Qwen2.5-VL processor returned visual grids without matching token runs"
            )
        return tuple(sorted(enriched, key=lambda item: item.start))

    def prepare_inputs(self, messages: Sequence[Any], media: Sequence[Any] = ()) -> PreparedInputs:
        prepared = super().prepare_inputs(messages, media)
        vision_config = getattr(self.model.config, "vision_config", None)
        merge_size = int(getattr(vision_config, "spatial_merge_size", 0))
        prepared.modality_map = self.enrich_modality_spans(
            prepared.modality_map,
            image_grid_thw=prepared.model_inputs.get("image_grid_thw"),
            video_grid_thw=prepared.model_inputs.get("video_grid_thw"),
            spatial_merge_size=merge_size,
        )
        return prepared

    def build_prototype_key_transform(
        self, prepared: PreparedInputs
    ) -> RotaryPrototypeKeyTransform:
        """Capture Qwen's exact prefill M-RoPE phase for safe prototype keys.

        This mirrors the installed Transformers ``get_rope_index``,
        ``Qwen2_5_VLRotaryEmbedding``, and
        ``apply_multimodal_rotary_pos_emb`` dataflow.  No position is inferred
        from packed cache indices.
        """

        import torch

        installed_transformers = version("transformers")
        if installed_transformers != "4.55.2":
            raise RuntimeError(
                "Qwen2.5-VL prototype M-RoPE support is audited only for "
                f"Transformers 4.55.2; found {installed_transformers}"
            )
        root = self.model.model
        required = ("get_rope_index", "language_model")
        if any(not hasattr(root, name) for name in required):
            raise RuntimeError("installed Qwen2.5-VL model lacks the audited M-RoPE API")
        model_inputs = prepared.model_inputs
        position_ids, _rope_deltas = root.get_rope_index(
            input_ids=model_inputs.get("input_ids"),
            image_grid_thw=model_inputs.get("image_grid_thw"),
            video_grid_thw=model_inputs.get("video_grid_thw"),
            second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
            attention_mask=model_inputs.get("attention_mask"),
        )
        if tuple(position_ids.shape[:2]) != (3, 1) or int(position_ids.shape[-1]) != (
            prepared.logical_sequence_length
        ):
            raise RuntimeError("Qwen2.5-VL get_rope_index returned an unexpected prefill shape")
        language_model = root.language_model
        rotary = getattr(language_model, "rotary_emb", None)
        rope_scaling = getattr(language_model.config, "rope_scaling", None)
        if rotary is None or not isinstance(rope_scaling, Mapping):
            raise RuntimeError("installed Qwen2.5-VL language model lacks audited M-RoPE metadata")
        raw_sections = rope_scaling.get("mrope_section")
        if not isinstance(raw_sections, list | tuple) or len(raw_sections) != 3:
            raise RuntimeError("Qwen2.5-VL rope_scaling.mrope_section must contain three values")
        sections = [int(value) * 2 for value in raw_sections]
        if any(value <= 0 for value in sections):
            raise RuntimeError("Qwen2.5-VL M-RoPE sections must be positive")
        reference = torch.empty((), dtype=self.model.dtype, device=self.device)
        cos, sin = rotary(reference, position_ids)

        def select_mrope_phase(value: Any) -> Any:
            chunks = value.split(sections, dim=-1)
            if len(chunks) != len(sections):
                raise RuntimeError("Qwen2.5-VL M-RoPE sections do not cover the head dimension")
            return torch.cat(
                [chunk[index % 3] for index, chunk in enumerate(chunks)], dim=-1
            ).unsqueeze(1)

        selected_cos = select_mrope_phase(cos)
        selected_sin = select_mrope_phase(sin)
        return RotaryPrototypeKeyTransform(
            selected_cos,
            selected_sin,
            provenance=(
                "transformers_4_55_2_qwen2_5_vl_get_rope_index_"
                "rotary_emb_apply_multimodal_rotary_pos_emb_v1"
            ),
        )

    def _decode_forward(
        self,
        token_id: Any,
        state: DecodeState,
        attention_mask: Any,
        cache_position: Any,
        *,
        output_attentions: bool = False,
    ) -> Any:
        # Transformers 4.55.2 constructs the one-token Qwen M-RoPE positions
        # with ``expand`` and then updates that overlapping view in place.  On
        # current PyTorch this raises before attention is entered.  Generation
        # normally supplies the equivalent position_ids from
        # ``prepare_inputs_for_generation``; our explicit decode loop must do
        # the same instead of relying on the broken internal fallback.
        torch = __import__("torch")
        rope_deltas = state.model_state.get("rope_deltas")
        if rope_deltas is None:
            raise RuntimeError("Qwen2.5-VL decode state is missing rope_deltas")
        batch_size, sequence_length = token_id.shape
        if sequence_length != 1:
            raise ValueError("Qwen2.5-VL decode expects exactly one token")
        required_length = state.next_decode_position + 1
        position_buffer = state.model_state.get("mosaickv_qwen_decode_position_ids_buffer")
        position_capacity = int(
            state.model_state.get("mosaickv_qwen_decode_position_ids_capacity", 0)
        )
        if (
            position_buffer is None
            or position_capacity < required_length
            or position_buffer.device != token_id.device
            or int(position_buffer.shape[1]) != batch_size
        ):
            position_capacity = max(required_length, max(512, state.next_decode_position * 2))
            base = torch.arange(position_capacity, dtype=torch.long, device=token_id.device).view(
                1, 1, -1
            )
            delta = rope_deltas.to(device=token_id.device, dtype=torch.long)
            if tuple(delta.shape) != (batch_size, 1):
                raise RuntimeError(
                    "Qwen2.5-VL rope_deltas must have shape [batch, 1] during decode; "
                    f"received {tuple(delta.shape)}"
                )
            position_buffer = base.expand(3, batch_size, -1) + delta.view(1, batch_size, 1)
            state.model_state["mosaickv_qwen_decode_position_ids_buffer"] = position_buffer
            state.model_state["mosaickv_qwen_decode_position_ids_capacity"] = position_capacity
        position_ids = position_buffer[..., state.next_decode_position : required_length]
        model_attention_mask = attention_mask
        if attention_mask is None and state.model_state.get("mosaickv_cache_all_valid") is True:
            # Qwen2.5-VL's text model accepts a per-attention-type mapping as
            # an already prepared mask. For a one-token query every active KV
            # slot precedes the logical decode position, so both masks are
            # exactly None. Supplying the mapping avoids rebuilding an
            # all-zero [batch, 1, 1, KV] causal tensor on every token.
            model_attention_mask = {
                "full_attention": None,
                "sliding_attention": None,
            }
        return self.model(
            input_ids=token_id,
            attention_mask=model_attention_mask,
            position_ids=position_ids,
            past_key_values=state.past_key_values,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
            output_attentions=output_attentions,
            logits_to_keep=1,
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

    def _capture_decode_model_state(self, output: Any) -> dict[str, Any]:
        # Explicit decode supplies position_ids, so the audited Transformers
        # forward does not update rope_deltas. Preserve the prefill-owned copy
        # instead of launching a redundant device clone on every token.
        del output
        return {}

    def _restore_model_state(self, state: DecodeState) -> None:
        rope_deltas = state.model_state.get("rope_deltas")
        if rope_deltas is None:
            raise RuntimeError("Qwen2.5-VL decode state is missing rope_deltas")
        if hasattr(self.model, "rope_deltas"):
            self.model.rope_deltas = rope_deltas
        else:
            self.model.model.rope_deltas = rope_deltas


__all__ = ["QWEN25_VL_MAX_PIXELS", "QWEN25_VL_MIN_PIXELS", "Qwen25VLAdapter"]
