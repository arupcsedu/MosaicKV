"""Eager Hugging Face prefill/decode adapter implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager, ExitStack
from types import TracebackType
from typing import Any, Self, cast

from mosaickv.adapters.huggingface.types import (
    AdapterCapabilities,
    AdapterProfilingModules,
    CacheLayerLayout,
    CacheLayerSnapshot,
    CacheLayout,
    CacheSnapshot,
    DecodeOutput,
    DecodeState,
    GreedyDecodeOutput,
    Modality,
    ModalitySpan,
    PrefillOutput,
    PreparedInputs,
    QueryVectors,
)


def _torch() -> Any:
    """Import torch only when a runtime adapter is actually used."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise RuntimeError("Hugging Face adapters require the MosaicKV hf environment") from exc
    return torch


def validate_hf_revision(revision: str) -> None:
    """Require an immutable 40-character Hugging Face git commit."""

    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError("revision must be a lowercase 40-character commit SHA")


class QueryCapture(AbstractContextManager["QueryCapture"]):
    """Capture q_proj results with hooks while leaving model weights untouched."""

    def __init__(self, projections: Sequence[tuple[Any, int, int]]) -> None:
        self._projections = tuple(projections)
        self._handles: list[Any] = []
        self._layers: list[Any | None] = [None] * len(self._projections)

    def __enter__(self) -> Self:
        if self._handles:
            raise RuntimeError("query capture context cannot be entered twice")
        for layer_index, (projection, num_heads, head_dim) in enumerate(self._projections):

            def hook(
                _module: Any,
                _inputs: tuple[Any, ...],
                output: Any,
                *,
                index: int = layer_index,
                heads: int = num_heads,
                dimension: int = head_dim,
            ) -> None:
                batch, sequence, width = output.shape
                if width != heads * dimension:
                    raise RuntimeError(
                        f"q_proj width {width} does not equal num_heads*head_dim "
                        f"({heads}*{dimension})"
                    )
                query = output.view(batch, sequence, heads, dimension).transpose(1, 2)
                self._layers[index] = query.detach()

            self._handles.append(projection.register_forward_hook(hook))
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def result(self) -> QueryVectors:
        if any(layer is None for layer in self._layers):
            missing = [index for index, layer in enumerate(self._layers) if layer is None]
            raise RuntimeError(f"q_proj hooks did not run for layers: {missing}")
        return QueryVectors(tuple(self._layers))


class AttentionCapture(AbstractContextManager["AttentionCapture"]):
    """Capture eager self-attention probabilities without changing model weights."""

    def __init__(self, attentions: Sequence[Any], *, query_window: int | None = None) -> None:
        self._attentions = tuple(attentions)
        self._query_window = query_window
        self._handles: list[Any] = []
        self._layers: list[Any | None] = [None] * len(self._attentions)
        if query_window is not None and query_window < 1:
            raise ValueError("attention capture query_window must be positive")

    def __enter__(self) -> Self:
        if self._handles:
            raise RuntimeError("attention capture context cannot be entered twice")
        for layer_index, attention in enumerate(self._attentions):

            def hook(
                _module: Any,
                _inputs: tuple[Any, ...],
                output: Any,
                *,
                index: int = layer_index,
            ) -> None:
                if not isinstance(output, tuple) or len(output) < 2 or output[1] is None:
                    raise RuntimeError(
                        f"eager attention layer {index} did not expose attention probabilities"
                    )
                probabilities = output[1]
                if self._query_window is not None:
                    probabilities = probabilities[..., -self._query_window :, :]
                # Clone so a narrow query slice does not retain the full QxK backing storage.
                self._layers[index] = probabilities.detach().clone()

            self._handles.append(attention.register_forward_hook(hook))
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def result(self) -> tuple[Any, ...]:
        if any(layer is None for layer in self._layers):
            missing = [index for index, layer in enumerate(self._layers) if layer is None]
            raise RuntimeError(f"attention hooks did not run for layers: {missing}")
        return tuple(self._layers)


class LayerAttentionMaskOverride(AbstractContextManager["LayerAttentionMaskOverride"]):
    """Replace the shared HF mask with checked layer/head-specific packed-cache masks."""

    def __init__(self, attentions: Sequence[Any], masks: Sequence[Any]) -> None:
        self._attentions = tuple(attentions)
        self._masks = tuple(masks)
        self._handles: list[Any] = []
        if len(self._attentions) != len(self._masks):
            raise ValueError("packed attention masks must cover every decoder layer")

    def __enter__(self) -> Self:
        for layer_index, (attention, mask) in enumerate(
            zip(self._attentions, self._masks, strict=True)
        ):

            def hook(
                _module: Any,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                *,
                index: int = layer_index,
                replacement: Any = mask,
            ) -> tuple[tuple[Any, ...], dict[str, Any]]:
                if "attention_mask" not in kwargs:
                    raise RuntimeError(
                        f"decoder attention layer {index} did not receive attention_mask"
                    )
                kwargs["attention_mask"] = replacement
                return args, kwargs

            self._handles.append(attention.register_forward_pre_hook(hook, with_kwargs=True))
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


class HuggingFaceMultimodalAdapter(ABC):
    """Common explicit cache-aware generation loop for native HF VLMs."""

    capabilities: AdapterCapabilities

    def __init__(self, model: Any, processor: Any, *, device: Any | None = None) -> None:
        self.model = model
        self.processor = processor
        self.device = device if device is not None else self._infer_input_device()
        self._validate_architecture()
        self._validate_eager_attention()
        if getattr(model, "training", False):
            model.eval()

    @property
    def supports_prototype_merge(self) -> bool:
        return self.capabilities.supports_prototype_merge

    @property
    def supports_residual_repair(self) -> bool:
        return self.capabilities.supports_residual_repair

    @abstractmethod
    def _language_layers(self) -> Sequence[Any]:
        """Return decoder layers in execution order."""

    def _standard_language_layers(self) -> Sequence[Any]:
        """Locate decoder layers across supported Transformers wrapper layouts."""

        pending = [self.model]
        visited: set[int] = set()
        while pending:
            candidate = pending.pop(0)
            if candidate is None or id(candidate) in visited:
                continue
            visited.add(id(candidate))
            layers = getattr(candidate, "layers", None)
            if layers is not None:
                return cast("Sequence[Any]", layers)
            pending.extend(
                (
                    getattr(candidate, "language_model", None),
                    getattr(candidate, "model", None),
                )
            )
        raise RuntimeError("adapter cannot locate language-model decoder layers")

    @abstractmethod
    def _image_token_id(self) -> int | None:
        """Return the expanded image-context token ID, if present."""

    def _video_token_id(self) -> int | None:
        return None

    @abstractmethod
    def _processor_kwargs(
        self, prompt: str, images: Sequence[Any], videos: Sequence[Any]
    ) -> dict[str, Any]:
        """Build the model-specific processor call."""

    def _infer_input_device(self) -> Any:
        try:
            return next(self.model.parameters()).device
        except (AttributeError, StopIteration) as exc:
            raise ValueError("cannot infer model input device; pass device explicitly") from exc

    def _validate_architecture(self) -> None:
        architecture = type(self.model).__name__
        if architecture not in self.capabilities.architectures:
            expected = ", ".join(self.capabilities.architectures)
            raise TypeError(f"expected one of [{expected}], received {architecture}")

    def _validate_eager_attention(self) -> None:
        implementations: set[str] = set()
        for config in self._attention_configs():
            value = getattr(config, "_attn_implementation", None)
            if value is not None:
                implementations.add(str(value))
        unsupported = implementations.difference(self.capabilities.attention_implementations)
        if unsupported:
            found = ", ".join(sorted(unsupported))
            raise ValueError(
                f"adapter is eager-only but model uses {found}; reload with "
                "attn_implementation='eager'"
            )

    def _attention_configs(self) -> tuple[Any, ...]:
        configs: list[Any] = []
        config = getattr(self.model, "config", None)
        if config is not None:
            configs.append(config)
            text_config = getattr(config, "text_config", None)
            if text_config is not None:
                configs.append(text_config)
            llm_config = getattr(config, "llm_config", None)
            if llm_config is not None:
                configs.append(llm_config)
        language_model = getattr(self.model, "language_model", None)
        lm_config = getattr(language_model, "config", None)
        if lm_config is not None:
            configs.append(lm_config)
        return tuple(configs)

    def capture_query_vectors(self) -> QueryCapture:
        """Return a context that captures every language-layer q_proj output."""

        projections: list[tuple[Any, int, int]] = []
        for index, layer in enumerate(self._language_layers()):
            attention = getattr(layer, "self_attn", None)
            projection = getattr(attention, "q_proj", None)
            num_heads = getattr(attention, "num_heads", None)
            attention_config = getattr(attention, "config", None)
            if num_heads is None:
                num_heads = getattr(attention_config, "num_attention_heads", None)
            head_dim = getattr(attention, "head_dim", None)
            if head_dim is None and attention_config is not None and num_heads is not None:
                head_dim = int(attention_config.hidden_size) // int(num_heads)
            if projection is None or num_heads is None or head_dim is None:
                raise RuntimeError(f"decoder layer {index} has no auditable q_proj geometry")
            projections.append((projection, int(num_heads), int(head_dim)))
        return QueryCapture(projections)

    def capture_attention_weights(self, *, query_window: int | None = None) -> AttentionCapture:
        """Return a context that captures every eager language attention map."""

        attentions = tuple(getattr(layer, "self_attn", None) for layer in self._language_layers())
        if any(attention is None for attention in attentions):
            raise RuntimeError("a decoder layer has no self_attn module")
        return AttentionCapture(attentions, query_window=query_window)

    def get_profiling_modules(self) -> AdapterProfilingModules:
        """Return audited modules used for synchronized phase measurements."""

        root = getattr(self.model, "model", self.model)
        vision = getattr(root, "vision_tower", None)
        if vision is None:
            vision = getattr(root, "visual", None)
        if vision is None:
            vision = getattr(self.model, "visual", None)
        if vision is None:
            vision = getattr(self.model, "vision_model", None)

        projector = getattr(root, "multi_modal_projector", None)
        if projector is None and vision is not None:
            projector = getattr(vision, "merger", None)
        if projector is None:
            projector = getattr(self.model, "mlp1", None)

        language_model = getattr(root, "language_model", None)
        if language_model is None:
            language_model = getattr(self.model, "language_model", None)
        if language_model is None and hasattr(root, "layers"):
            language_model = root
        if language_model is None:
            raise RuntimeError("adapter cannot locate the language-model module for profiling")
        return AdapterProfilingModules(
            vision_encoder=vision,
            projector=projector,
            language_model=language_model,
            vision_includes_projector=(
                vision is not None
                and projector is not None
                and getattr(vision, "merger", None) is projector
            ),
        )

    def prepare_inputs(self, messages: Sequence[Any], media: Sequence[Any] = ()) -> PreparedInputs:
        """Render messages, preprocess media, and map expanded modality tokens."""

        chat, images, videos = self._normalize_messages(messages, media)
        if videos and not self.capabilities.video:
            raise ValueError(f"{self.capabilities.model_family} does not support video inputs")
        prompt = self._render_prompt(chat)
        processed = self._process_prompt(prompt, images, videos)
        model_inputs = dict(processed)
        model_inputs = self._move_model_inputs(model_inputs)
        input_ids = model_inputs.get("input_ids")
        if input_ids is None or getattr(input_ids, "ndim", None) != 2:
            raise RuntimeError("processor must return rank-2 input_ids")
        if int(input_ids.shape[0]) != 1:
            raise ValueError("MosaicKV HF adapters currently require batch size 1")
        logical_length = int(input_ids.shape[-1])
        modality_map = self._modality_spans(input_ids)
        return PreparedInputs(model_inputs, modality_map, logical_length)

    def _process_prompt(
        self, prompt: str, images: Sequence[Any], videos: Sequence[Any]
    ) -> Mapping[str, Any]:
        return cast(
            "Mapping[str, Any]",
            self.processor(**self._processor_kwargs(prompt, images, videos)),
        )

    def _render_prompt(self, chat: list[dict[str, Any]]) -> str:
        renderer = getattr(self.processor, "apply_chat_template", None)
        if not getattr(self.processor, "chat_template", None):
            tokenizer = getattr(self.processor, "tokenizer", None)
            tokenizer_renderer = getattr(tokenizer, "apply_chat_template", None)
            if getattr(tokenizer, "chat_template", None) and callable(tokenizer_renderer):
                renderer = tokenizer_renderer
        if not callable(renderer):
            raise RuntimeError("processor and tokenizer have no apply_chat_template implementation")
        prompt = renderer(chat, tokenize=False, add_generation_prompt=True)
        if not isinstance(prompt, str) or not prompt:
            raise RuntimeError("chat template did not return a non-empty prompt")
        return prompt

    def _normalize_messages(
        self, messages: Sequence[Any], media: Sequence[Any]
    ) -> tuple[list[dict[str, Any]], list[Any], list[Any]]:
        if not messages:
            raise ValueError("messages must be non-empty")
        queued = [self._normalize_media(item) for item in media]
        chat: list[dict[str, Any]] = []
        images: list[Any] = []
        videos: list[Any] = []
        visual_parts = 0

        for message in messages:
            role = self._field(message, "role")
            raw_content = self._field(message, "content")
            if not isinstance(role, str) or not role:
                raise ValueError("each message must have a non-empty role")
            raw_parts: Sequence[Any]
            if isinstance(raw_content, str):
                raw_parts = (raw_content,)
            elif isinstance(raw_content, Sequence):
                raw_parts = raw_content
            else:
                raise ValueError("message content must be text or a sequence of parts")
            parts: list[dict[str, Any]] = []
            for raw_part in raw_parts:
                if isinstance(raw_part, str):
                    parts.append({"type": "text", "text": raw_part})
                    continue
                part_type = self._field(raw_part, "type")
                if part_type == "text":
                    text = self._optional_field(raw_part, "text")
                    if text is None:
                        text = self._optional_field(raw_part, "value")
                    if not isinstance(text, str) or not text:
                        raise ValueError("text message parts must contain non-empty text")
                    parts.append({"type": "text", "text": text})
                elif part_type in {"image", "video"}:
                    payload = self._visual_payload(raw_part, str(part_type))
                    if payload is None:
                        payload = self._take_media(queued, str(part_type))
                    self._append_media(str(part_type), payload, images, videos)
                    parts.append({"type": str(part_type)})
                    visual_parts += 1
                else:
                    raise ValueError(f"unsupported message part type: {part_type!r}")
            chat.append({"role": role, "content": parts})

        if queued:
            if visual_parts:
                kinds = ", ".join(kind for kind, _payload in queued)
                raise ValueError(f"unreferenced media remains after filling placeholders: {kinds}")
            user = next((item for item in chat if item["role"] == "user"), None)
            if user is None:
                raise ValueError("standalone media requires a user message")
            inserted: list[dict[str, str]] = []
            for kind, payload in queued:
                self._append_media(kind, payload, images, videos)
                inserted.append({"type": kind})
            user["content"] = inserted + user["content"]
            queued.clear()
        return chat, images, videos

    @staticmethod
    def _field(value: Any, name: str) -> Any:
        if isinstance(value, Mapping):
            if name not in value:
                raise ValueError(f"missing required field {name!r}")
            return value[name]
        if not hasattr(value, name):
            raise ValueError(f"missing required field {name!r}")
        return getattr(value, name)

    @staticmethod
    def _optional_field(value: Any, name: str) -> Any | None:
        if isinstance(value, Mapping):
            return value.get(name)
        return getattr(value, name, None)

    def _normalize_media(self, item: Any) -> tuple[str, Any]:
        kind = self._optional_field(item, "kind")
        payload = self._optional_field(item, "payload")
        if kind is None and isinstance(item, tuple) and len(item) == 2:
            kind, payload = item
        if kind is None:
            if self.capabilities.image and not self.capabilities.video:
                return "image", item
            raise ValueError("media items must identify kind as image or video")
        normalized_kind = str(getattr(kind, "value", kind))
        if normalized_kind not in {"image", "video"}:
            raise ValueError(f"unsupported media kind: {normalized_kind!r}")
        if payload is None:
            raise ValueError("media payload cannot be None")
        return normalized_kind, payload

    def _visual_payload(self, part: Any, kind: str) -> Any | None:
        for name in (kind, "payload", "value"):
            payload = self._optional_field(part, name)
            if payload is not None:
                return payload
        return None

    @staticmethod
    def _take_media(queued: list[tuple[str, Any]], kind: str) -> Any:
        for index, (candidate_kind, payload) in enumerate(queued):
            if candidate_kind == kind:
                queued.pop(index)
                return payload
        raise ValueError(f"message contains an unfilled {kind} placeholder")

    @staticmethod
    def _append_media(kind: str, payload: Any, images: list[Any], videos: list[Any]) -> None:
        if kind == "image":
            images.append(payload)
        else:
            videos.append(payload)

    def _move_model_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        moved: dict[str, Any] = {}
        model_dtype = getattr(self.model, "dtype", None)
        for name, value in inputs.items():
            mover = getattr(value, "to", None)
            if mover is None:
                moved[name] = value
                continue
            value = mover(self.device)
            if name.startswith("pixel_values") and model_dtype is not None:
                value = value.to(dtype=model_dtype)
            moved[name] = value
        return moved

    def _modality_spans(self, input_ids: Any) -> tuple[ModalitySpan, ...]:
        image_id = self._image_token_id()
        video_id = self._video_token_id()
        ids = [int(value) for value in input_ids[0].detach().cpu().tolist()]
        modalities: list[Modality] = []
        for token_id in ids:
            if image_id is not None and token_id == image_id:
                modalities.append(Modality.IMAGE)
            elif video_id is not None and token_id == video_id:
                modalities.append(Modality.VIDEO)
            else:
                modalities.append(Modality.TEXT)
        spans: list[ModalitySpan] = []
        start = 0
        for index in range(1, len(modalities) + 1):
            if index == len(modalities) or modalities[index] != modalities[start]:
                spans.append(ModalitySpan(start, index, modalities[start]))
                start = index
        return tuple(spans)

    def get_modality_map(self, prepared: PreparedInputs) -> tuple[ModalitySpan, ...]:
        return prepared.modality_map

    def get_logical_sequence_length(self, value: PreparedInputs | DecodeState) -> int:
        return value.logical_sequence_length

    def prefill(
        self,
        prepared: PreparedInputs,
        *,
        capture_queries: bool = True,
        capture_attentions: bool = False,
        attention_query_window: int | None = None,
    ) -> PrefillOutput:
        """Run one full prefill without calling GenerationMixin.generate."""

        torch = _torch()
        input_ids = prepared.model_inputs["input_ids"]
        cache_position = torch.arange(
            prepared.logical_sequence_length, device=input_ids.device, dtype=torch.long
        )
        forward_inputs = dict(prepared.model_inputs)
        forward_inputs.update(
            use_cache=True,
            return_dict=True,
            cache_position=cache_position,
            logits_to_keep=1,
        )
        with torch.inference_mode(), ExitStack() as stack:
            query_capture = (
                stack.enter_context(self.capture_query_vectors()) if capture_queries else None
            )
            attention_capture = (
                stack.enter_context(
                    self.capture_attention_weights(query_window=attention_query_window)
                )
                if capture_attentions
                else None
            )
            output = self._prefill_forward(forward_inputs)
        query_vectors = QueryVectors(()) if query_capture is None else query_capture.result()
        attention_weights = () if attention_capture is None else attention_capture.result()
        logits = output.logits[:, -1, :]
        cache = self.extract_past_key_values(output, clone=False)
        active_length = self._cache_length(cache)
        if active_length != prepared.logical_sequence_length:
            raise RuntimeError(
                "full-cache prefill length mismatch: "
                f"active={active_length}, logical={prepared.logical_sequence_length}"
            )
        attention_mask = prepared.model_inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        state = DecodeState(
            past_key_values=cache,
            attention_mask=attention_mask,
            active_cache_length=active_length,
            logical_sequence_length=prepared.logical_sequence_length,
            next_decode_position=prepared.logical_sequence_length,
            modality_map=prepared.modality_map,
            model_state=self._capture_model_state(output),
        )
        return PrefillOutput(
            logits,
            self._greedy_token(logits),
            state,
            query_vectors,
            attention_weights,
        )

    def _prefill_forward(self, forward_inputs: dict[str, Any]) -> Any:
        return self.model(**forward_inputs)

    def _capture_model_state(self, output: Any) -> dict[str, Any]:
        del output
        return {}

    def _restore_model_state(self, state: DecodeState) -> None:
        del state

    def decode_one_token(
        self,
        token_id: Any,
        state: DecodeState,
        *,
        capture_queries: bool = True,
        capture_attentions: bool = False,
    ) -> DecodeOutput:
        """Decode one token and expose the cache before the next token is chosen."""

        packed_validity = state.model_state.get("mosaickv_validity_masks")
        if state.active_cache_length != state.logical_sequence_length and packed_validity is None:
            raise NotImplementedError(
                "compressed-cache decoding requires an explicit logical-position attention mask; "
                "only the retention-ratio-1.0 path is enabled"
            )
        torch = _torch()
        if getattr(token_id, "ndim", None) == 1:
            token_id = token_id.unsqueeze(-1)
        if tuple(token_id.shape) != (1, 1):
            raise ValueError("decode_one_token expects token_id shape [1, 1]")
        one = torch.ones(
            (1, 1), dtype=state.attention_mask.dtype, device=state.attention_mask.device
        )
        attention_mask = torch.cat((state.attention_mask, one), dim=-1)
        cache_position = torch.tensor(
            [state.next_decode_position], dtype=torch.long, device=token_id.device
        )
        self._restore_model_state(state)
        layer_masks = (
            self._packed_decode_masks(state, token_id) if packed_validity is not None else ()
        )
        with torch.inference_mode(), ExitStack() as stack:
            query_capture = (
                stack.enter_context(self.capture_query_vectors()) if capture_queries else None
            )
            attention_capture = (
                stack.enter_context(self.capture_attention_weights())
                if capture_attentions
                else None
            )
            if layer_masks:
                stack.enter_context(
                    LayerAttentionMaskOverride(
                        tuple(layer.self_attn for layer in self._language_layers()),
                        layer_masks,
                    )
                )
            output = self._decode_forward(token_id, state, attention_mask, cache_position)
        query_vectors = QueryVectors(()) if query_capture is None else query_capture.result()
        attention_weights = () if attention_capture is None else attention_capture.result()
        logits = output.logits[:, -1, :]
        cache = self.extract_past_key_values(output, clone=False)
        active_length = self._cache_length(cache)
        expected_active = state.active_cache_length + 1
        if active_length != expected_active:
            raise RuntimeError(f"decode cache grew to {active_length}; expected {expected_active}")
        model_state = {**state.model_state, **self._capture_model_state(output)}
        if packed_validity is not None:
            model_state["mosaickv_validity_masks"] = tuple(
                torch.cat(
                    (
                        validity,
                        torch.ones(
                            (int(validity.shape[0]), 1),
                            dtype=torch.bool,
                            device=validity.device,
                        ),
                    ),
                    dim=-1,
                )
                for validity in packed_validity
            )
        updated = DecodeState(
            past_key_values=cache,
            attention_mask=attention_mask,
            active_cache_length=active_length,
            logical_sequence_length=state.logical_sequence_length + 1,
            next_decode_position=state.next_decode_position + 1,
            modality_map=state.modality_map,
            model_state=model_state,
        )
        return DecodeOutput(
            logits,
            self._greedy_token(logits),
            updated,
            query_vectors,
            attention_weights,
        )

    def _packed_decode_masks(self, state: DecodeState, token_id: Any) -> tuple[Any, ...]:
        """Build additive eager masks for one packed-cache token decode."""

        torch = _torch()
        validity_masks = state.model_state.get("mosaickv_validity_masks")
        if not isinstance(validity_masks, tuple) or len(validity_masks) != len(
            self._language_layers()
        ):
            raise RuntimeError("packed cache validity metadata is incomplete")
        result: list[Any] = []
        for layer_index, (layer, validity) in enumerate(
            zip(self._language_layers(), validity_masks, strict=True)
        ):
            if getattr(validity, "dtype", None) != torch.bool or getattr(validity, "ndim", 0) != 2:
                raise RuntimeError(f"packed validity mask {layer_index} must be rank-2 bool")
            if int(validity.shape[-1]) != state.active_cache_length:
                raise RuntimeError(f"packed validity mask {layer_index} has the wrong cache length")
            attention = layer.self_attn
            kv_heads = int(validity.shape[0])
            query_heads = int(getattr(attention, "num_heads", attention.config.num_attention_heads))
            if query_heads % kv_heads:
                raise RuntimeError("query heads are not divisible by packed KV heads")
            valid = validity.repeat_interleave(query_heads // kv_heads, dim=0)
            valid = torch.cat(
                (
                    valid,
                    torch.ones((query_heads, 1), dtype=torch.bool, device=valid.device),
                ),
                dim=-1,
            )
            dtype = next(attention.parameters()).dtype
            additive = torch.zeros(valid.shape, dtype=dtype, device=token_id.device)
            additive.masked_fill_(~valid.to(device=token_id.device), torch.finfo(dtype).min)
            if not bool(torch.isfinite(additive).all()):
                raise RuntimeError("packed attention mask contains NaN or infinity")
            result.append(additive.unsqueeze(0).unsqueeze(-2))
        return tuple(result)

    def _decode_forward(
        self, token_id: Any, state: DecodeState, attention_mask: Any, cache_position: Any
    ) -> Any:
        return self.model(
            input_ids=token_id,
            attention_mask=attention_mask,
            past_key_values=state.past_key_values,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )

    @staticmethod
    def _greedy_token(logits: Any) -> Any:
        return logits.argmax(dim=-1, keepdim=True)

    def greedy_decode(
        self,
        prepared: PreparedInputs,
        *,
        max_new_tokens: int,
        reinject_after_prefill: bool = False,
    ) -> GreedyDecodeOutput:
        """Run deterministic greedy decoding through the explicit adapter loop."""

        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        torch = _torch()
        prefill = self.prefill(prepared)
        state = prefill.state
        if reinject_after_prefill:
            snapshot = self.extract_past_key_values(state.past_key_values)
            state.past_key_values = self.inject_past_key_values(snapshot)
        tokens = [prefill.next_token_id]
        logits = [prefill.logits]
        token = prefill.next_token_id
        for _index in range(max_new_tokens - 1):
            step = self.decode_one_token(token, state)
            state = step.state
            token = step.next_token_id
            tokens.append(token)
            logits.append(step.logits)
        return GreedyDecodeOutput(torch.cat(tokens, dim=-1), tuple(logits), state)

    def extract_past_key_values(self, value: Any, *, clone: bool = True) -> Any:
        """Get a cache from model output, or make a typed snapshot of a cache."""

        cache = getattr(value, "past_key_values", value)
        if not clone:
            if cache is None:
                raise RuntimeError("forward output did not contain past_key_values")
            return cache
        legacy, source_kind = self._legacy_layers(cache)
        layers: list[CacheLayerSnapshot] = []
        active_length: int | None = None
        for index, (key, item_value) in enumerate(legacy):
            if getattr(key, "ndim", 0) < 2 or getattr(item_value, "ndim", 0) < 2:
                raise RuntimeError(f"cache layer {index} is not a K/V tensor pair")
            sequence_dimension = self._cache_sequence_dimension(key)
            key_length = int(key.shape[sequence_dimension])
            value_sequence_dimension = self._cache_sequence_dimension(item_value)
            if value_sequence_dimension != sequence_dimension:
                raise RuntimeError(f"cache layer {index} uses different K and V sequence axes")
            value_length = int(item_value.shape[value_sequence_dimension])
            if key_length != value_length:
                raise RuntimeError(f"cache layer {index} has different K and V lengths")
            if active_length is None:
                active_length = key_length
            elif key_length != active_length:
                raise RuntimeError("cache layers have inconsistent active lengths")
            layers.append(
                CacheLayerSnapshot(
                    key.detach().clone(), item_value.detach().clone(), sequence_dimension
                )
            )
        if active_length is None:
            raise RuntimeError("cache contains no layers")
        return CacheSnapshot(
            tuple(layers),
            type(cache),
            source_kind,
            active_length,
            self.capabilities.cached_key_state,
        )

    @staticmethod
    def _legacy_layers(cache: Any) -> tuple[tuple[tuple[Any, Any], ...], str]:
        if cache is None:
            raise RuntimeError("past_key_values is None")
        converter = getattr(cache, "to_legacy_cache", None)
        if converter is not None:
            return tuple(converter()), "cache_object"
        if isinstance(cache, tuple):
            return tuple(cache), "tuple"
        if isinstance(cache, list):
            return tuple(cache), "list"
        layers = getattr(cache, "layers", None)
        if layers is not None:
            return tuple((layer.keys, layer.values) for layer in layers), "cache_layers"
        raise TypeError(f"unsupported past_key_values type: {type(cache).__qualname__}")

    def inject_past_key_values(self, snapshot: CacheSnapshot) -> Any:
        """Recreate the original cache kind from an extracted snapshot."""

        legacy = tuple((layer.key, layer.value) for layer in snapshot.layers)
        if snapshot.source_kind == "tuple":
            return legacy
        if snapshot.source_kind == "list":
            return list(legacy)
        factory = getattr(snapshot.source_class, "from_legacy_cache", None)
        if factory is not None:
            try:
                return factory(legacy)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"{snapshot.source_class.__qualname__}.from_legacy_cache rejected snapshot"
                ) from exc
        raise TypeError(
            f"cache class {snapshot.source_class.__qualname__} cannot be reconstructed safely"
        )

    def get_cache_layout(self, cache: Any) -> CacheLayout:
        """Return observed shapes/dtypes/devices without assuming a shared layout."""

        if isinstance(cache, CacheSnapshot):
            legacy = tuple((layer.key, layer.value) for layer in cache.layers)
            cache_class = cache.source_class
        else:
            legacy, _source_kind = self._legacy_layers(cache)
            cache_class = type(cache)
        layouts: list[CacheLayerLayout] = []
        active_length: int | None = None
        for index, (key, value) in enumerate(legacy):
            sequence_dimension = self._cache_sequence_dimension(key)
            if self._cache_sequence_dimension(value) != sequence_dimension:
                raise RuntimeError(f"cache layer {index} uses different K and V sequence axes")
            length = int(key.shape[sequence_dimension])
            active_length = length if active_length is None else active_length
            if length != active_length:
                raise RuntimeError("cache layers have inconsistent active lengths")
            layouts.append(
                CacheLayerLayout(
                    layer=index,
                    key_shape=tuple(int(item) for item in key.shape),
                    value_shape=tuple(int(item) for item in value.shape),
                    key_dtype=str(key.dtype),
                    value_dtype=str(value.dtype),
                    key_device=str(key.device),
                    value_device=str(value.device),
                    sequence_dimension=sequence_dimension,
                )
            )
        if active_length is None:
            raise RuntimeError("cache contains no layers")
        return CacheLayout(
            cache_class=f"{cache_class.__module__}.{cache_class.__qualname__}",
            active_sequence_length=active_length,
            cached_key_state=self.capabilities.cached_key_state,
            layers=tuple(layouts),
        )

    def _cache_length(self, cache: Any) -> int:
        getter = getattr(cache, "get_seq_length", None)
        if getter is not None:
            return int(getter())
        legacy, _source_kind = self._legacy_layers(cache)
        if not legacy:
            return 0
        key = legacy[0][0]
        return int(key.shape[self._cache_sequence_dimension(key)])

    def _cache_sequence_dimension(self, tensor: Any) -> int:
        configured = self.capabilities.cache_sequence_dimension
        dimension = configured if configured >= 0 else int(tensor.ndim) + configured
        if dimension < 0 or dimension >= int(tensor.ndim):
            raise RuntimeError(
                "configured cache sequence dimension "
                f"{configured} is invalid for rank {tensor.ndim}"
            )
        return dimension


__all__ = [
    "AttentionCapture",
    "HuggingFaceMultimodalAdapter",
    "LayerAttentionMaskOverride",
    "QueryCapture",
    "validate_hf_revision",
]
