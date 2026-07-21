"""Runtime types for Hugging Face multimodal cache adapters.

This module intentionally does not import PyTorch.  The core MosaicKV package
must remain importable in the mock/CPU environment where torch is optional.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from mosaickv.cache_state import Modality, ModalitySpan


class CachedKeyState(StrEnum):
    """Whether keys in the returned cache have already received RoPE."""

    UNKNOWN = "unknown"
    PRE_ROPE = "pre_rope"
    POST_ROPE = "post_rope"
    NOT_APPLICABLE = "not_applicable"


class QueryVectorState(StrEnum):
    """Representation captured from each decoder attention layer."""

    Q_PROJ_PRE_ROPE = "q_proj_output_pre_rope"


@dataclass(frozen=True, slots=True)
class AdapterCapabilities:
    """Runtime adapter behavior supported by a particular model family."""

    model_family: str
    architectures: tuple[str, ...]
    attention_implementations: tuple[str, ...]
    image: bool
    multi_image: bool
    video: bool
    cache_classes: tuple[str, ...]
    cache_sequence_dimension: int
    cached_key_state: CachedKeyState
    query_vector_state: QueryVectorState
    supports_prototype_merge: bool
    supports_residual_repair: bool
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AdapterProfilingModules:
    """Model modules that define non-overlapping prefill phase boundaries."""

    vision_encoder: Any | None
    projector: Any | None
    language_model: Any
    vision_includes_projector: bool = False


@dataclass(slots=True)
class PreparedInputs:
    """Processor output and its language-sequence metadata."""

    model_inputs: dict[str, Any]
    modality_map: tuple[ModalitySpan, ...]
    logical_sequence_length: int

    def __post_init__(self) -> None:
        if self.logical_sequence_length <= 0:
            raise ValueError("logical_sequence_length must be positive")


@dataclass(frozen=True, slots=True)
class CacheLayerSnapshot:
    """One cache layer without imposing a model-independent head layout."""

    key: Any
    value: Any
    sequence_dimension: int


@dataclass(frozen=True, slots=True)
class CacheSnapshot:
    """A cache copy plus enough type information for faithful reinjection."""

    layers: tuple[CacheLayerSnapshot, ...]
    source_class: type[Any]
    source_kind: str
    active_sequence_length: int
    cached_key_state: CachedKeyState

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("cache snapshot must contain at least one layer")
        if self.active_sequence_length <= 0:
            raise ValueError("active_sequence_length must be positive")


@dataclass(frozen=True, slots=True)
class CacheLayerLayout:
    """Observed runtime tensor metadata for one K/V layer."""

    layer: int
    key_shape: tuple[int, ...]
    value_shape: tuple[int, ...]
    key_dtype: str
    value_dtype: str
    key_device: str
    value_device: str
    sequence_dimension: int


@dataclass(frozen=True, slots=True)
class CacheLayout:
    """Observed cache class and every per-layer tensor layout."""

    cache_class: str
    active_sequence_length: int
    cached_key_state: CachedKeyState
    layers: tuple[CacheLayerLayout, ...]


@dataclass(frozen=True, slots=True)
class QueryVectors:
    """Captured query projections in [batch, heads, sequence, head_dim]."""

    layers: tuple[Any, ...]
    state: QueryVectorState = QueryVectorState.Q_PROJ_PRE_ROPE


@dataclass(slots=True)
class DecodeState:
    """Mutable decode state with physical and logical positions separated."""

    past_key_values: Any
    attention_mask: Any
    active_cache_length: int
    logical_sequence_length: int
    next_decode_position: int
    modality_map: tuple[ModalitySpan, ...]
    model_state: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.active_cache_length <= 0:
            raise ValueError("active_cache_length must be positive")
        if self.logical_sequence_length <= 0:
            raise ValueError("logical_sequence_length must be positive")
        if self.next_decode_position < self.logical_sequence_length:
            raise ValueError("next_decode_position cannot precede logical_sequence_length")


@dataclass(frozen=True, slots=True)
class PrefillOutput:
    """Logits, greedy next token, cache state, and captured prefill queries."""

    logits: Any
    next_token_id: Any
    state: DecodeState
    query_vectors: QueryVectors
    attention_weights: tuple[Any, ...] = ()


@dataclass(frozen=True, slots=True)
class DecodeOutput:
    """One explicit decoding step and the updated cache state."""

    logits: Any
    next_token_id: Any
    state: DecodeState
    query_vectors: QueryVectors
    attention_weights: tuple[Any, ...] = ()


@dataclass(frozen=True, slots=True)
class GreedyDecodeOutput:
    """Output of the adapter's explicit prefill/token decode loop."""

    token_ids: Any
    step_logits: tuple[Any, ...]
    state: DecodeState


@dataclass(frozen=True, slots=True)
class ParityReport:
    """Measured correctness result from a real model execution."""

    comparison: str
    generated_tokens: int
    token_agreement: float
    maximum_logit_difference: float
    reference_token_ids: tuple[int, ...]
    candidate_token_ids: tuple[int, ...]
    measurement_type: str = "validation_smoke"


__all__ = [
    "AdapterCapabilities",
    "AdapterProfilingModules",
    "CacheLayerLayout",
    "CacheLayerSnapshot",
    "CacheLayout",
    "CacheSnapshot",
    "CachedKeyState",
    "DecodeOutput",
    "DecodeState",
    "GreedyDecodeOutput",
    "Modality",
    "ModalitySpan",
    "ParityReport",
    "PrefillOutput",
    "PreparedInputs",
    "QueryVectorState",
    "QueryVectors",
]
