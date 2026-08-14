"""Strict, dependency-light configuration schema for MosaicKV runs."""

from __future__ import annotations

import hashlib
import json
import math
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TypeVar, cast

import yaml

from mosaickv.types import (
    AblationVariant,
    Backend,
    BudgetUnit,
    ForecastCovariance,
    ForecastMode,
    JsonObject,
    LookMMergeStrategy,
    MosaicKVMethod,
    OutputLengthPolicy,
    Precision,
    PrefixKVProfileMode,
    RepairCandidateRanking,
    RepairPolicy,
    ResidualStorageDType,
)

_SYNTHETIC_ACCOUNTING_SHA = "60915ac01ad35843c4de56620864145d5b81b973a3bfec03fb322d5e4f6ff695"


class ConfigurationError(ValueError):
    """Raised when configuration input violates the strict schema."""


EnumT = TypeVar("EnumT", bound=StrEnum)


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{path} must be a table/object, got {type(value).__name__}")
    if not all(isinstance(key, str) for key in value):
        raise ConfigurationError(f"{path} keys must all be strings")
    return cast("Mapping[str, object]", value)


def _reject_unknown(data: Mapping[str, object], allowed: set[str], path: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigurationError(f"{path} contains unknown field(s): {', '.join(unknown)}")


def _required_str(data: Mapping[str, object], key: str, path: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{path}.{key} must be a non-empty string")
    return value


def _optional_str(data: Mapping[str, object], key: str, default: str, path: str) -> str:
    value = data.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{path}.{key} must be a non-empty string")
    return value


def _int(data: Mapping[str, object], key: str, default: int, path: str) -> int:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{path}.{key} must be an integer")
    return value


def _float(data: Mapping[str, object], key: str, default: float, path: str) -> float:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{path}.{key} must be a number")
    return float(value)


def _bool(data: Mapping[str, object], key: str, default: bool, path: str) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise ConfigurationError(f"{path}.{key} must be a boolean")
    return value


def _optional_int(
    data: Mapping[str, object], key: str, default: int | None, path: str
) -> int | None:
    value = data.get(key, default)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{path}.{key} must be an integer or null")
    return value


def _string_tuple(
    data: Mapping[str, object], key: str, default: tuple[str, ...], path: str
) -> tuple[str, ...]:
    value = data.get(key, default)
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ConfigurationError(f"{path}.{key} must be an array of non-empty strings")
    return tuple(cast("str", item) for item in value)


def _enum(enum_type: type[EnumT], value: object, path: str) -> EnumT:
    if not isinstance(value, str):
        raise ConfigurationError(f"{path} must be a string")
    try:
        return enum_type(value)
    except ValueError as error:
        choices = ", ".join(member.value for member in enum_type)
        raise ConfigurationError(f"{path} must be one of: {choices}; got {value!r}") from error


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Immutable model identity and numerical precision."""

    id: str
    revision: str
    precision: Precision

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ConfigurationError("model.id must be a non-empty string")
        if not self.revision.strip():
            raise ConfigurationError("model.revision must be a non-empty immutable revision")

    @classmethod
    def from_mapping(cls, value: object) -> ModelConfig:
        data = _mapping(value, "model")
        _reject_unknown(data, {"id", "revision", "precision"}, "model")
        return cls(
            id=_required_str(data, "id", "model"),
            revision=_required_str(data, "revision", "model"),
            precision=_enum(Precision, data.get("precision"), "model.precision"),
        )


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Immutable dataset identity."""

    id: str
    revision: str
    split: str

    def __post_init__(self) -> None:
        for name, value in (("id", self.id), ("revision", self.revision), ("split", self.split)):
            if not value.strip():
                raise ConfigurationError(f"dataset.{name} must be a non-empty string")

    @classmethod
    def from_mapping(cls, value: object) -> DatasetConfig:
        data = _mapping(value, "dataset")
        _reject_unknown(data, {"id", "revision", "split"}, "dataset")
        return cls(
            id=_required_str(data, "id", "dataset"),
            revision=_required_str(data, "revision", "dataset"),
            split=_required_str(data, "split", "dataset"),
        )


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    """Backend and determinism controls."""

    backend: Backend
    attention_implementation: str
    seed: int = 0
    deterministic_algorithms: bool = True
    decode_attention_implementation: str | None = None

    def __post_init__(self) -> None:
        if not self.attention_implementation.strip():
            raise ConfigurationError("execution.attention_implementation must be non-empty")
        if (
            self.decode_attention_implementation is not None
            and not self.decode_attention_implementation.strip()
        ):
            raise ConfigurationError(
                "execution.decode_attention_implementation must be non-empty or null"
            )
        if self.seed < 0:
            raise ConfigurationError("execution.seed must be >= 0")

    @classmethod
    def from_mapping(cls, value: object) -> ExecutionConfig:
        data = _mapping(value, "execution")
        allowed = {
            "backend",
            "attention_implementation",
            "decode_attention_implementation",
            "seed",
            "deterministic_algorithms",
        }
        _reject_unknown(data, allowed, "execution")
        return cls(
            backend=_enum(Backend, data.get("backend"), "execution.backend"),
            attention_implementation=_required_str(data, "attention_implementation", "execution"),
            seed=_int(data, "seed", 0, "execution"),
            deterministic_algorithms=_bool(data, "deterministic_algorithms", True, "execution"),
            decode_attention_implementation=(
                None
                if data.get("decode_attention_implementation") is None
                else _optional_str(
                    data,
                    "decode_attention_implementation",
                    "eager",
                    "execution",
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Generation controls that must be matched across comparisons."""

    max_new_tokens: int = 32
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    output_length_policy: OutputLengthPolicy = OutputLengthPolicy.FIXED_MAX_NEW_TOKENS

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1:
            raise ConfigurationError("generation.max_new_tokens must be >= 1")
        if self.temperature < 0:
            raise ConfigurationError("generation.temperature must be >= 0")
        if self.do_sample and self.temperature <= 0:
            raise ConfigurationError(
                "generation.temperature must be > 0 when generation.do_sample is true"
            )
        if not 0 < self.top_p <= 1:
            raise ConfigurationError("generation.top_p must be in the interval (0, 1]")

    @classmethod
    def from_mapping(cls, value: object) -> GenerationConfig:
        data = _mapping(value, "generation")
        allowed = {"max_new_tokens", "do_sample", "temperature", "top_p", "output_length_policy"}
        _reject_unknown(data, allowed, "generation")
        return cls(
            max_new_tokens=_int(data, "max_new_tokens", 32, "generation"),
            do_sample=_bool(data, "do_sample", False, "generation"),
            temperature=_float(data, "temperature", 0.0, "generation"),
            top_p=_float(data, "top_p", 1.0, "generation"),
            output_length_policy=_enum(
                OutputLengthPolicy,
                data.get("output_length_policy", OutputLengthPolicy.FIXED_MAX_NEW_TOKENS.value),
                "generation.output_length_policy",
            ),
        )


@dataclass(frozen=True, slots=True)
class FidelityDiagnosticsConfig:
    """Evaluation-only paired FullKV diagnostics for a fixed sample ledger."""

    enabled: bool = False
    sample_ids: tuple[str, ...] = ()
    sample_ledger_sha: str | None = None
    forecast_oracle_steps: int = 0
    capture_full_trace: bool = False
    capture_selection_evidence: bool = False

    def __post_init__(self) -> None:
        if len(self.sample_ids) != len(set(self.sample_ids)):
            raise ConfigurationError("diagnostics.sample_ids cannot contain duplicates")
        if any(not sample_id.strip() for sample_id in self.sample_ids):
            raise ConfigurationError("diagnostics.sample_ids must be non-empty strings")
        if self.forecast_oracle_steps < 0:
            raise ConfigurationError("diagnostics.forecast_oracle_steps must be >= 0")
        if not self.enabled and self.forecast_oracle_steps:
            raise ConfigurationError(
                "diagnostics.forecast_oracle_steps requires diagnostics.enabled=true"
            )
        if self.enabled and not self.sample_ids:
            raise ConfigurationError(
                "diagnostics.sample_ids must be non-empty when diagnostics are enabled"
            )
        if not self.enabled and (self.sample_ids or self.sample_ledger_sha is not None):
            raise ConfigurationError(
                "diagnostics sample provenance requires diagnostics.enabled=true"
            )
        if self.enabled:
            value = self.sample_ledger_sha
            if (
                value is None
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ConfigurationError(
                    "diagnostics.sample_ledger_sha must be a lowercase SHA-256 digest"
                )

    @classmethod
    def from_mapping(cls, value: object) -> FidelityDiagnosticsConfig:
        data = _mapping(value, "diagnostics")
        _reject_unknown(
            data,
            {
                "enabled",
                "sample_ids",
                "sample_ledger_sha",
                "forecast_oracle_steps",
                "capture_full_trace",
                "capture_selection_evidence",
            },
            "diagnostics",
        )
        raw_ids = data.get("sample_ids", ())
        if not isinstance(raw_ids, list | tuple) or not all(
            isinstance(sample_id, str) for sample_id in raw_ids
        ):
            raise ConfigurationError("diagnostics.sample_ids must be a string array")
        raw_sha = data.get("sample_ledger_sha")
        if raw_sha is not None and not isinstance(raw_sha, str):
            raise ConfigurationError("diagnostics.sample_ledger_sha must be a string or null")
        return cls(
            enabled=_bool(data, "enabled", False, "diagnostics"),
            sample_ids=tuple(cast("list[str] | tuple[str, ...]", raw_ids)),
            sample_ledger_sha=raw_sha,
            forecast_oracle_steps=_int(data, "forecast_oracle_steps", 0, "diagnostics"),
            capture_full_trace=_bool(data, "capture_full_trace", False, "diagnostics"),
            capture_selection_evidence=_bool(
                data,
                "capture_selection_evidence",
                False,
                "diagnostics",
            ),
        )


@dataclass(frozen=True, slots=True)
class AblationConfig:
    """Explicit component-ablation identity and oracle safety gate."""

    variant: AblationVariant = AblationVariant.NONE
    evaluation_only: bool = False

    def __post_init__(self) -> None:
        is_oracle = self.variant in {
            AblationVariant.ORACLE_FUTURE_QUERY,
            AblationVariant.ORACLE_REPAIR,
        }
        if is_oracle != self.evaluation_only:
            raise ConfigurationError(
                "oracle ablations require ablation.evaluation_only=true, and that flag "
                "is reserved for oracle ablations"
            )

    @classmethod
    def from_mapping(cls, value: object) -> AblationConfig:
        data = _mapping(value, "ablation")
        _reject_unknown(data, {"variant", "evaluation_only"}, "ablation")
        return cls(
            variant=_enum(
                AblationVariant,
                data.get("variant", AblationVariant.NONE.value),
                "ablation.variant",
            ),
            evaluation_only=_bool(data, "evaluation_only", False, "ablation"),
        )


@dataclass(frozen=True, slots=True)
class CacheConfig:
    """Cache budget and block layout."""

    budget_value: int
    budget_unit: BudgetUnit
    retention_ratio: float
    block_size: int = 16
    accounting_spec_sha: str = _SYNTHETIC_ACCOUNTING_SHA
    physical_budget_block_size: int | None = None
    physical_budget_mandatory_policy: str = "method"
    exact_gather_mode: str = "batched_layer"
    # The layer-batched mixed-tier scatter is retained for controlled
    # experiments, but A100 measurements showed that its extra full-cache
    # copy and indexed scatter are slower than contiguous per-head packing.
    packing_mode: str = "per_head_reference"
    # Optional hard cap over active device K/V plus encoded CPU residual K/V.
    # The ordinary selection budget remains the active-cache constraint; this
    # second boundary is required for claims about total compressed storage.
    total_storage_budget_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.budget_value < 1:
            raise ConfigurationError("cache.budget_value must be >= 1")
        if not 0 < self.retention_ratio <= 1:
            raise ConfigurationError("cache.retention_ratio must be in the interval (0, 1]")
        if self.block_size < 1:
            raise ConfigurationError("cache.block_size must be >= 1")
        if self.physical_budget_block_size is not None and self.physical_budget_block_size < 1:
            raise ConfigurationError("cache.physical_budget_block_size must be >= 1 or null")
        if self.physical_budget_mandatory_policy not in {"method", "last_token"}:
            raise ConfigurationError(
                "cache.physical_budget_mandatory_policy must be 'method' or 'last_token'"
            )
        if self.exact_gather_mode not in {
            "per_head_reference",
            "batched_layer",
            "source_views",
        }:
            raise ConfigurationError(
                "cache.exact_gather_mode must be 'per_head_reference', 'batched_layer', "
                "or 'source_views'"
            )
        if self.packing_mode not in {
            "per_head_reference",
            "batched_layer",
            "preallocated_layer",
        }:
            raise ConfigurationError(
                "cache.packing_mode must be 'per_head_reference', 'batched_layer', "
                "or 'preallocated_layer'"
            )
        if self.total_storage_budget_bytes is not None and self.total_storage_budget_bytes < 1:
            raise ConfigurationError("cache.total_storage_budget_bytes must be >= 1 or null")
        if len(self.accounting_spec_sha) != 64 or any(
            char not in "0123456789abcdef" for char in self.accounting_spec_sha
        ):
            raise ConfigurationError("cache.accounting_spec_sha must be a lowercase SHA-256 digest")

    @classmethod
    def from_mapping(cls, value: object) -> CacheConfig:
        data = _mapping(value, "cache")
        allowed = {
            "budget_value",
            "budget_unit",
            "retention_ratio",
            "block_size",
            "accounting_spec_sha",
            "physical_budget_block_size",
            "physical_budget_mandatory_policy",
            "exact_gather_mode",
            "packing_mode",
            "total_storage_budget_bytes",
        }
        _reject_unknown(data, allowed, "cache")
        return cls(
            budget_value=_int(data, "budget_value", 0, "cache"),
            budget_unit=_enum(BudgetUnit, data.get("budget_unit"), "cache.budget_unit"),
            retention_ratio=_float(data, "retention_ratio", 0.0, "cache"),
            block_size=_int(data, "block_size", 16, "cache"),
            accounting_spec_sha=_optional_str(
                data, "accounting_spec_sha", _SYNTHETIC_ACCOUNTING_SHA, "cache"
            ),
            physical_budget_block_size=_optional_int(
                data, "physical_budget_block_size", None, "cache"
            ),
            physical_budget_mandatory_policy=_optional_str(
                data, "physical_budget_mandatory_policy", "method", "cache"
            ),
            exact_gather_mode=_optional_str(
                data,
                "exact_gather_mode",
                "batched_layer",
                "cache",
            ),
            packing_mode=_optional_str(
                data,
                "packing_mode",
                "per_head_reference",
                "cache",
            ),
            total_storage_budget_bytes=_optional_int(
                data,
                "total_storage_budget_bytes",
                None,
                "cache",
            ),
        )


@dataclass(frozen=True, slots=True)
class ForecastingConfig:
    """Future-query forecasting controls."""

    enabled: bool = True
    mode: ForecastMode = ForecastMode.HYBRID
    prompt_window: int = 16
    draft_steps: int = 4
    centroid_count: int = 4
    covariance: ForecastCovariance = ForecastCovariance.DIAGONAL
    low_memory_centroids: bool = True
    centroid_iterations: int = 4
    adaptive_head_routing: bool = False
    head_shift_threshold: float = 0.15
    stable_head_prompt_weight: float = 1.0
    shifted_head_prompt_weight: float = 0.25
    uncertainty_exact_head_fraction: float = 0.0
    centroid_execution_mode: str = "batched_layer"
    conditional_draft_enabled: bool = False
    conditional_draft_uncertainty_threshold: float = 0.15

    def __post_init__(self) -> None:
        if self.prompt_window < 0:
            raise ConfigurationError("forecasting.prompt_window must be >= 0")
        if self.draft_steps < 0:
            raise ConfigurationError("forecasting.draft_steps must be >= 0")
        if self.centroid_count < 1:
            raise ConfigurationError("forecasting.centroid_count must be >= 1")
        if self.centroid_iterations < 1:
            raise ConfigurationError("forecasting.centroid_iterations must be >= 1")
        if not 0 <= self.head_shift_threshold <= 1:
            raise ConfigurationError("forecasting.head_shift_threshold must be in [0, 1]")
        for name in ("stable_head_prompt_weight", "shifted_head_prompt_weight"):
            if not 0 <= getattr(self, name) <= 1:
                raise ConfigurationError(f"forecasting.{name} must be in [0, 1]")
        if not 0 <= self.uncertainty_exact_head_fraction <= 1:
            raise ConfigurationError(
                "forecasting.uncertainty_exact_head_fraction must be in [0, 1]"
            )
        if self.uncertainty_exact_head_fraction and not self.adaptive_head_routing:
            raise ConfigurationError(
                "forecasting.uncertainty_exact_head_fraction requires adaptive_head_routing"
            )
        if self.centroid_execution_mode not in {"reference_per_head", "batched_layer"}:
            raise ConfigurationError(
                "forecasting.centroid_execution_mode must be 'reference_per_head' or "
                "'batched_layer'"
            )
        if not 0 <= self.conditional_draft_uncertainty_threshold <= 1:
            raise ConfigurationError(
                "forecasting.conditional_draft_uncertainty_threshold must be in [0, 1]"
            )
        if not self.enabled:
            return
        if self.mode is ForecastMode.PROMPT_WINDOW and (
            self.prompt_window == 0 or self.draft_steps != 0
        ):
            raise ConfigurationError(
                "prompt_window mode requires prompt_window > 0 and draft_steps = 0"
            )
        if self.mode is ForecastMode.DRAFT_ROLLOUT and (
            self.draft_steps == 0 or self.prompt_window != 0
        ):
            raise ConfigurationError(
                "draft_rollout mode requires draft_steps > 0 and prompt_window = 0"
            )
        if self.mode is ForecastMode.HYBRID and (self.prompt_window == 0 or self.draft_steps == 0):
            raise ConfigurationError(
                "hybrid forecasting requires prompt_window > 0 and draft_steps > 0"
            )
        if self.conditional_draft_enabled and self.mode is not ForecastMode.HYBRID:
            raise ConfigurationError(
                "forecasting.conditional_draft_enabled requires hybrid forecasting"
            )

    @classmethod
    def from_mapping(cls, value: object) -> ForecastingConfig:
        data = _mapping(value, "forecasting")
        allowed = {
            "enabled",
            "mode",
            "prompt_window",
            "draft_steps",
            "centroid_count",
            "covariance",
            "low_memory_centroids",
            "centroid_iterations",
            "adaptive_head_routing",
            "head_shift_threshold",
            "stable_head_prompt_weight",
            "shifted_head_prompt_weight",
            "uncertainty_exact_head_fraction",
            "centroid_execution_mode",
            "conditional_draft_enabled",
            "conditional_draft_uncertainty_threshold",
        }
        _reject_unknown(data, allowed, "forecasting")
        return cls(
            enabled=_bool(data, "enabled", True, "forecasting"),
            mode=_enum(ForecastMode, data.get("mode", "hybrid"), "forecasting.mode"),
            prompt_window=_int(data, "prompt_window", 16, "forecasting"),
            draft_steps=_int(data, "draft_steps", 4, "forecasting"),
            centroid_count=_int(data, "centroid_count", 4, "forecasting"),
            covariance=_enum(
                ForecastCovariance,
                data.get("covariance", "diagonal"),
                "forecasting.covariance",
            ),
            low_memory_centroids=_bool(data, "low_memory_centroids", True, "forecasting"),
            centroid_iterations=_int(data, "centroid_iterations", 4, "forecasting"),
            adaptive_head_routing=_bool(data, "adaptive_head_routing", False, "forecasting"),
            head_shift_threshold=_float(data, "head_shift_threshold", 0.15, "forecasting"),
            stable_head_prompt_weight=_float(data, "stable_head_prompt_weight", 1.0, "forecasting"),
            shifted_head_prompt_weight=_float(
                data, "shifted_head_prompt_weight", 0.25, "forecasting"
            ),
            uncertainty_exact_head_fraction=_float(
                data, "uncertainty_exact_head_fraction", 0.0, "forecasting"
            ),
            centroid_execution_mode=_optional_str(
                data,
                "centroid_execution_mode",
                "batched_layer",
                "forecasting",
            ),
            conditional_draft_enabled=_bool(
                data,
                "conditional_draft_enabled",
                False,
                "forecasting",
            ),
            conditional_draft_uncertainty_threshold=_float(
                data,
                "conditional_draft_uncertainty_threshold",
                0.15,
                "forecasting",
            ),
        )


@dataclass(frozen=True, slots=True)
class GraphConfig:
    """Sparse cross-modal evidence graph controls."""

    enabled: bool = True
    max_neighbors: int = 8
    min_edge_weight: float = 0.0
    similarity_chunk_size: int = 256
    require_same_layer: bool = True
    require_same_kv_head: bool = True
    allowed_modality_pairs: tuple[str, ...] = (
        "text:text",
        "text:image",
        "text:video",
        "image:text",
        "image:image",
        "image:video",
        "video:text",
        "video:image",
        "video:video",
    )
    semantic_weight: float = 1.0
    attention_weight: float = 1.0
    spatial_weight: float = 1.0
    layout_weight: float = 1.0
    temporal_weight: float = 1.0
    same_region_weight: float = 1.0
    cross_modal_weight: float = 1.0
    fallback_weight: float = 0.25
    semantic_max_position_span: int | None = None
    attention_max_position_span: int | None = None
    fallback_max_position_span: int | None = 128
    spatial_max_distance: float = 0.125
    temporal_window: int = 1
    attention_host_transfer_mode: str = "deferred_layer"
    edge_accumulation_mode: str = "preselected"
    # When positive, prompt attention chooses a bounded candidate set for the
    # expensive semantic/co-activation similarity sources. Every cache block
    # remains a graph node and still receives structural/fallback edges.
    similarity_shortlist_multiplier: float = 0.0
    similarity_shortlist_min_nodes: int = 0
    # Pool learned K/V descriptors only for the prompt-ranked shortlist while
    # retaining every source block as a lightweight structural/fallback node.
    # This is an explicit approximation because excluded nodes no longer
    # contribute learned semantic/value descriptors.
    candidate_descriptor_pooling: bool = False
    # ``shortlist_rerank`` keeps the complete source-block table for exact
    # accounting and prompt-attention selection, but runs learned/structural
    # graph sources only on the prompt-ranked candidate union.  Positional
    # prototype compatibility remains analytic in the tier planner and is not
    # materialized as O(source_nodes * degree) fallback edges.
    candidate_graph_mode: str = "full"
    candidate_neighbor_hops: int = 0
    screening_group_blocks: int = 1
    uncertain_head_only: bool = False
    uncertain_head_fraction: float = 1.0
    uncertain_head_threshold: float = 0.0
    # Hard latency guard for candidate reranking. Mandatory blocks are always
    # admitted separately and therefore do not consume this non-mandatory
    # fraction. ``1.0`` preserves the prior behavior.
    candidate_nonmandatory_fraction: float = 1.0

    def __post_init__(self) -> None:
        if self.max_neighbors < 1:
            raise ConfigurationError("graph.max_neighbors must be >= 1")
        if not 0 <= self.min_edge_weight <= 1:
            raise ConfigurationError("graph.min_edge_weight must be in the interval [0, 1]")
        if self.similarity_chunk_size < 1:
            raise ConfigurationError("graph.similarity_chunk_size must be >= 1")
        valid_modalities = {"text", "image", "video"}
        if not self.allowed_modality_pairs:
            raise ConfigurationError("graph.allowed_modality_pairs cannot be empty")
        for pair in self.allowed_modality_pairs:
            if not isinstance(pair, str):
                raise ConfigurationError("graph.allowed_modality_pairs entries must be strings")
            parts = pair.split(":")
            if len(parts) != 2 or any(part not in valid_modalities for part in parts):
                raise ConfigurationError(
                    "graph.allowed_modality_pairs entries must use '<source>:<target>' "
                    "with text, image, or video"
                )
        if len(set(self.allowed_modality_pairs)) != len(self.allowed_modality_pairs):
            raise ConfigurationError("graph.allowed_modality_pairs cannot contain duplicates")
        weight_fields = (
            "semantic_weight",
            "attention_weight",
            "spatial_weight",
            "layout_weight",
            "temporal_weight",
            "same_region_weight",
            "cross_modal_weight",
            "fallback_weight",
        )
        for name in weight_fields:
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ConfigurationError(f"graph.{name} must be in the interval [0, 1]")
        span_fields = (
            "semantic_max_position_span",
            "attention_max_position_span",
            "fallback_max_position_span",
        )
        for name in span_fields:
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ConfigurationError(f"graph.{name} must be >= 0 or null")
        if not 0 < self.spatial_max_distance <= math.sqrt(2):
            raise ConfigurationError(
                "graph.spatial_max_distance must be in the interval (0, sqrt(2)]"
            )
        if self.temporal_window < 1:
            raise ConfigurationError("graph.temporal_window must be >= 1")
        if self.attention_host_transfer_mode not in {
            "deferred_layer",
            "deferred_layer_vectorized",
            "per_group_reference",
        }:
            raise ConfigurationError(
                "graph.attention_host_transfer_mode must be 'deferred_layer', "
                "'deferred_layer_vectorized', or 'per_group_reference'"
            )
        if self.edge_accumulation_mode not in {"dictionary", "preselected"}:
            raise ConfigurationError(
                "graph.edge_accumulation_mode must be 'dictionary' or 'preselected'"
            )
        if (
            not math.isfinite(self.similarity_shortlist_multiplier)
            or self.similarity_shortlist_multiplier < 0
        ):
            raise ConfigurationError(
                "graph.similarity_shortlist_multiplier must be finite and >= 0"
            )
        if self.similarity_shortlist_min_nodes < 0:
            raise ConfigurationError("graph.similarity_shortlist_min_nodes must be >= 0")
        if self.candidate_descriptor_pooling and self.similarity_shortlist_multiplier <= 0:
            raise ConfigurationError(
                "graph.candidate_descriptor_pooling requires a positive "
                "similarity_shortlist_multiplier"
            )
        if self.candidate_graph_mode not in {"full", "shortlist_rerank"}:
            raise ConfigurationError(
                "graph.candidate_graph_mode must be 'full' or 'shortlist_rerank'"
            )
        if self.candidate_graph_mode == "shortlist_rerank" and (
            self.similarity_shortlist_multiplier <= 0 or not self.candidate_descriptor_pooling
        ):
            raise ConfigurationError(
                "graph shortlist_rerank mode requires positive similarity_shortlist_multiplier "
                "and candidate_descriptor_pooling=true"
            )
        if self.candidate_neighbor_hops < 0:
            raise ConfigurationError("graph.candidate_neighbor_hops must be >= 0")
        if self.screening_group_blocks < 1:
            raise ConfigurationError("graph.screening_group_blocks must be >= 1")
        if not 0 < self.uncertain_head_fraction <= 1:
            raise ConfigurationError("graph.uncertain_head_fraction must be in (0, 1]")
        if not 0 <= self.uncertain_head_threshold <= 1:
            raise ConfigurationError("graph.uncertain_head_threshold must be in [0, 1]")
        if not 0 < self.candidate_nonmandatory_fraction <= 1:
            raise ConfigurationError(
                "graph.candidate_nonmandatory_fraction must be in (0, 1]"
            )
        if self.uncertain_head_only and self.candidate_graph_mode != "shortlist_rerank":
            raise ConfigurationError(
                "graph.uncertain_head_only requires candidate_graph_mode='shortlist_rerank'"
            )

    @classmethod
    def from_mapping(cls, value: object) -> GraphConfig:
        data = _mapping(value, "graph")
        allowed = {
            "enabled",
            "max_neighbors",
            "min_edge_weight",
            "similarity_chunk_size",
            "require_same_layer",
            "require_same_kv_head",
            "allowed_modality_pairs",
            "semantic_weight",
            "attention_weight",
            "spatial_weight",
            "layout_weight",
            "temporal_weight",
            "same_region_weight",
            "cross_modal_weight",
            "fallback_weight",
            "semantic_max_position_span",
            "attention_max_position_span",
            "fallback_max_position_span",
            "spatial_max_distance",
            "temporal_window",
            "attention_host_transfer_mode",
            "edge_accumulation_mode",
            "similarity_shortlist_multiplier",
            "similarity_shortlist_min_nodes",
            "candidate_descriptor_pooling",
            "candidate_graph_mode",
            "candidate_neighbor_hops",
            "screening_group_blocks",
            "uncertain_head_only",
            "uncertain_head_fraction",
            "uncertain_head_threshold",
            "candidate_nonmandatory_fraction",
        }
        _reject_unknown(data, allowed, "graph")
        defaults = cls()
        return cls(
            enabled=_bool(data, "enabled", True, "graph"),
            max_neighbors=_int(data, "max_neighbors", 8, "graph"),
            min_edge_weight=_float(data, "min_edge_weight", 0.0, "graph"),
            similarity_chunk_size=_int(data, "similarity_chunk_size", 256, "graph"),
            require_same_layer=_bool(data, "require_same_layer", True, "graph"),
            require_same_kv_head=_bool(data, "require_same_kv_head", True, "graph"),
            allowed_modality_pairs=_string_tuple(
                data,
                "allowed_modality_pairs",
                defaults.allowed_modality_pairs,
                "graph",
            ),
            semantic_weight=_float(data, "semantic_weight", 1.0, "graph"),
            attention_weight=_float(data, "attention_weight", 1.0, "graph"),
            spatial_weight=_float(data, "spatial_weight", 1.0, "graph"),
            layout_weight=_float(data, "layout_weight", 1.0, "graph"),
            temporal_weight=_float(data, "temporal_weight", 1.0, "graph"),
            same_region_weight=_float(data, "same_region_weight", 1.0, "graph"),
            cross_modal_weight=_float(data, "cross_modal_weight", 1.0, "graph"),
            fallback_weight=_float(data, "fallback_weight", 0.25, "graph"),
            semantic_max_position_span=_optional_int(
                data, "semantic_max_position_span", None, "graph"
            ),
            attention_max_position_span=_optional_int(
                data, "attention_max_position_span", None, "graph"
            ),
            fallback_max_position_span=_optional_int(
                data, "fallback_max_position_span", 128, "graph"
            ),
            spatial_max_distance=_float(data, "spatial_max_distance", 0.125, "graph"),
            temporal_window=_int(data, "temporal_window", 1, "graph"),
            attention_host_transfer_mode=_optional_str(
                data,
                "attention_host_transfer_mode",
                "deferred_layer",
                "graph",
            ),
            edge_accumulation_mode=_optional_str(
                data,
                "edge_accumulation_mode",
                "preselected",
                "graph",
            ),
            similarity_shortlist_multiplier=_float(
                data,
                "similarity_shortlist_multiplier",
                0.0,
                "graph",
            ),
            similarity_shortlist_min_nodes=_int(
                data,
                "similarity_shortlist_min_nodes",
                0,
                "graph",
            ),
            candidate_descriptor_pooling=_bool(
                data,
                "candidate_descriptor_pooling",
                False,
                "graph",
            ),
            candidate_graph_mode=_optional_str(
                data,
                "candidate_graph_mode",
                "full",
                "graph",
            ),
            candidate_neighbor_hops=_int(
                data,
                "candidate_neighbor_hops",
                0,
                "graph",
            ),
            screening_group_blocks=_int(
                data,
                "screening_group_blocks",
                1,
                "graph",
            ),
            uncertain_head_only=_bool(
                data,
                "uncertain_head_only",
                False,
                "graph",
            ),
            uncertain_head_fraction=_float(
                data,
                "uncertain_head_fraction",
                1.0,
                "graph",
            ),
            uncertain_head_threshold=_float(
                data,
                "uncertain_head_threshold",
                0.0,
                "graph",
            ),
            candidate_nonmandatory_fraction=_float(
                data,
                "candidate_nonmandatory_fraction",
                1.0,
                "graph",
            ),
        )


@dataclass(frozen=True, slots=True)
class UtilityConfig:
    """Signed coefficients in the preregistered local utility equation.

    The equation contains explicit minus signs for value contribution and
    uniqueness.  Negative ``lambda_v``/``lambda_o`` values therefore express
    preservation rewards without changing the registered equation.  Keeping
    these coefficients signed also prevents the accidental behavior where
    high-contribution, novel evidence is preferentially discarded.
    """

    lambda_q: float = 1.0
    lambda_v: float = -0.25
    lambda_o: float = -0.25
    normalization: str = "layer_head_l1"
    execution_mode: str = "vectorized_edges"

    def __post_init__(self) -> None:
        if not math.isfinite(self.lambda_q) or self.lambda_q < 0:
            raise ConfigurationError("utility.lambda_q must be finite and >= 0")
        for name in ("lambda_v", "lambda_o"):
            if not math.isfinite(getattr(self, name)):
                raise ConfigurationError(f"utility.{name} must be finite")
        if self.normalization != "layer_head_l1":
            raise ConfigurationError("utility.normalization must be 'layer_head_l1'")
        if self.execution_mode not in {"reference_sparse", "vectorized_edges"}:
            raise ConfigurationError(
                "utility.execution_mode must be 'reference_sparse' or 'vectorized_edges'"
            )

    @classmethod
    def from_mapping(cls, value: object) -> UtilityConfig:
        data = _mapping(value, "utility")
        _reject_unknown(
            data,
            {"lambda_q", "lambda_v", "lambda_o", "normalization", "execution_mode"},
            "utility",
        )
        return cls(
            lambda_q=_float(data, "lambda_q", 1.0, "utility"),
            lambda_v=_float(data, "lambda_v", -0.25, "utility"),
            lambda_o=_float(data, "lambda_o", -0.25, "utility"),
            normalization=_optional_str(data, "normalization", "layer_head_l1", "utility"),
            execution_mode=_optional_str(
                data,
                "execution_mode",
                "vectorized_edges",
                "utility",
            ),
        )


@dataclass(frozen=True, slots=True)
class SelectionConfig:
    """Budgeted submodular selection controls."""

    enabled: bool = True
    algorithm: str = "lazy_greedy"
    lambda_g: float = -0.25
    lambda_m: float = -0.25
    stop_on_nonpositive_gain: bool = True
    minimum_stratum_fraction: float = 0.25
    exhaustive_max_nodes: int = 20
    prompt_anchor_fraction: float = 0.0
    evidence_preservation_fraction: float = 0.0
    coverage_correction_rounds: int = 2
    stable_head_prompt_routing: bool = False
    stable_head_uniform_fill: bool = False
    routed_head_only_planning: bool = False
    # MosaicKV may deliberately materialize less than the shared comparison
    # budget when evaluating a quality-matched memory point. The shared hard
    # cap remains recorded separately; actual active bytes remain the primary
    # comparison quantity. Baselines do not consume this MosaicKV planner knob.
    physical_capacity_fraction: float = 1.0

    def __post_init__(self) -> None:
        if self.algorithm not in {"lazy_greedy", "vectorized_coverage"}:
            raise ConfigurationError(
                "selection.algorithm must be 'lazy_greedy' or 'vectorized_coverage'"
            )
        if self.coverage_correction_rounds < 0 or self.coverage_correction_rounds > 8:
            raise ConfigurationError(
                "selection.coverage_correction_rounds must be between 0 and 8"
            )
        for name in ("lambda_g", "lambda_m"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ConfigurationError(f"selection.{name} must be finite")
        if not 0 <= self.minimum_stratum_fraction <= 1:
            raise ConfigurationError("selection.minimum_stratum_fraction must be in [0, 1]")
        if self.exhaustive_max_nodes < 1:
            raise ConfigurationError("selection.exhaustive_max_nodes must be >= 1")
        for name in ("prompt_anchor_fraction", "evidence_preservation_fraction"):
            if not 0 <= getattr(self, name) <= 1:
                raise ConfigurationError(f"selection.{name} must be in [0, 1]")
        if self.prompt_anchor_fraction + self.evidence_preservation_fraction > 1:
            raise ConfigurationError(
                "selection prompt/evidence reservation fractions cannot sum above 1"
            )
        if self.stable_head_uniform_fill and not self.stable_head_prompt_routing:
            raise ConfigurationError(
                "selection.stable_head_uniform_fill requires stable_head_prompt_routing"
            )
        if self.routed_head_only_planning and not self.stable_head_prompt_routing:
            raise ConfigurationError(
                "selection.routed_head_only_planning requires stable_head_prompt_routing"
            )
        if (
            not math.isfinite(self.physical_capacity_fraction)
            or not 0 < self.physical_capacity_fraction <= 1
        ):
            raise ConfigurationError(
                "selection.physical_capacity_fraction must be finite and in (0, 1]"
            )

    @classmethod
    def from_mapping(cls, value: object) -> SelectionConfig:
        data = _mapping(value, "selection")
        allowed = {
            "enabled",
            "algorithm",
            "lambda_g",
            "lambda_m",
            "stop_on_nonpositive_gain",
            "minimum_stratum_fraction",
            "exhaustive_max_nodes",
            "prompt_anchor_fraction",
            "evidence_preservation_fraction",
            "coverage_correction_rounds",
            "stable_head_prompt_routing",
            "stable_head_uniform_fill",
            "routed_head_only_planning",
            "physical_capacity_fraction",
        }
        _reject_unknown(data, allowed, "selection")
        return cls(
            enabled=_bool(data, "enabled", True, "selection"),
            algorithm=_optional_str(data, "algorithm", "lazy_greedy", "selection"),
            lambda_g=_float(data, "lambda_g", -0.25, "selection"),
            lambda_m=_float(data, "lambda_m", -0.25, "selection"),
            stop_on_nonpositive_gain=_bool(data, "stop_on_nonpositive_gain", True, "selection"),
            minimum_stratum_fraction=_float(data, "minimum_stratum_fraction", 0.25, "selection"),
            exhaustive_max_nodes=_int(data, "exhaustive_max_nodes", 20, "selection"),
            prompt_anchor_fraction=_float(data, "prompt_anchor_fraction", 0.0, "selection"),
            evidence_preservation_fraction=_float(
                data, "evidence_preservation_fraction", 0.0, "selection"
            ),
            coverage_correction_rounds=_int(
                data,
                "coverage_correction_rounds",
                2,
                "selection",
            ),
            stable_head_prompt_routing=_bool(
                data, "stable_head_prompt_routing", False, "selection"
            ),
            stable_head_uniform_fill=_bool(
                data, "stable_head_uniform_fill", False, "selection"
            ),
            routed_head_only_planning=_bool(
                data, "routed_head_only_planning", False, "selection"
            ),
            physical_capacity_fraction=_float(
                data,
                "physical_capacity_fraction",
                1.0,
                "selection",
            ),
        )


@dataclass(frozen=True, slots=True)
class PrototypeConfig:
    """Prototype-tier controls."""

    enabled: bool = True
    group_size: int = 4
    max_position_span: int | None = 128
    min_anchor_weight: float = 0.0
    max_key_dispersion: float | None = None
    max_value_dispersion: float | None = None
    max_member_key_error: float | None = None
    max_member_value_error: float | None = None
    forecast_weight: float = 0.0
    attention_log_bias_scale: float = 0.0
    pooling_mode: str = "graph_weighted_block_mean"
    attention_bias_basis: str = "source_position_multiplicity"
    anchor_position_strategy: str = "selected_anchor_start"
    adaptive_grouping: bool = False
    exact_head_routes: tuple[str, ...] = ()
    # Unselected blocks in these heads are dropped instead of prototyped. The
    # already-selected exact subset remains active, unlike ``exact_head_routes``
    # which requires every source block in the route to stay exact.
    selection_only_head_routes: tuple[str, ...] = ()
    maximum_prototypes_per_head: int | None = None
    uncertain_head_only: bool = False
    uncertain_head_fraction: float = 1.0
    uncertain_head_threshold: float = 0.0
    source_modalities: tuple[str, ...] = ("text", "image", "video")
    head_group_sizes: tuple[str, ...] = ()
    head_position_spans: tuple[str, ...] = ()
    source_gather_mode: str = "batched_layer"
    budget_planning_mode: str = "one_pass_anchor_cover"
    allowed_modality_pairs: tuple[str, ...] = (
        "text:text",
        "image:image",
        "video:video",
    )

    def __post_init__(self) -> None:
        if self.group_size < 1:
            raise ConfigurationError("prototypes.group_size must be >= 1")
        if self.max_position_span is not None and self.max_position_span < 0:
            raise ConfigurationError("prototypes.max_position_span must be >= 0 or null")
        if not 0 <= self.min_anchor_weight <= 1:
            raise ConfigurationError("prototypes.min_anchor_weight must be in [0, 1]")
        for name in (
            "max_key_dispersion",
            "max_value_dispersion",
            "max_member_key_error",
            "max_member_value_error",
        ):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ConfigurationError(f"prototypes.{name} must be finite and >= 0 or null")
        if not 0 <= self.forecast_weight <= 1:
            raise ConfigurationError("prototypes.forecast_weight must be in [0, 1]")
        if not math.isfinite(self.attention_log_bias_scale) or not (
            0 <= self.attention_log_bias_scale <= 1
        ):
            raise ConfigurationError(
                "prototypes.attention_log_bias_scale must be finite and in [0, 1]"
            )
        if self.pooling_mode not in {
            "graph_weighted_block_mean",
            "forecast_attention_token",
            "exact_medoid_key_forecast_value",
        }:
            raise ConfigurationError(
                "prototypes.pooling_mode must be 'graph_weighted_block_mean' or "
                "'forecast_attention_token' or 'exact_medoid_key_forecast_value'"
            )
        if self.attention_bias_basis not in {
            "source_position_multiplicity",
            "forecast_attention_entropy",
        }:
            raise ConfigurationError(
                "prototypes.attention_bias_basis must be "
                "'source_position_multiplicity' or 'forecast_attention_entropy'"
            )
        if self.anchor_position_strategy not in {
            "selected_anchor_start",
            "forecast_weighted_member_medoid",
        }:
            raise ConfigurationError(
                "prototypes.anchor_position_strategy must be "
                "'selected_anchor_start' or 'forecast_weighted_member_medoid'"
            )
        if self.source_gather_mode not in {"per_head_reference", "batched_layer"}:
            raise ConfigurationError(
                "prototypes.source_gather_mode must be 'per_head_reference' or 'batched_layer'"
            )
        if self.budget_planning_mode not in {
            "one_pass_anchor_cover",
            "bounded_search_reference",
        }:
            raise ConfigurationError(
                "prototypes.budget_planning_mode must be 'one_pass_anchor_cover' or "
                "'bounded_search_reference'"
            )
        if (
            self.pooling_mode in {
                "forecast_attention_token",
                "exact_medoid_key_forecast_value",
            }
            and self.attention_bias_basis != "forecast_attention_entropy"
        ):
            raise ConfigurationError(
                "forecast-attention token pooling requires the forecast-attention "
                "entropy bias basis"
            )
        if (
            self.attention_bias_basis == "forecast_attention_entropy"
            and self.pooling_mode
            not in {"forecast_attention_token", "exact_medoid_key_forecast_value"}
        ):
            raise ConfigurationError(
                "forecast-attention entropy bias requires forecast-attention token pooling"
            )
        if (
            self.pooling_mode == "exact_medoid_key_forecast_value"
            and self.anchor_position_strategy != "forecast_weighted_member_medoid"
        ):
            raise ConfigurationError(
                "exact-medoid key pooling requires "
                "anchor_position_strategy='forecast_weighted_member_medoid'"
            )
        self._validate_head_routes(self.head_group_sizes, "head_group_sizes", minimum=1)
        self._validate_head_routes(self.head_position_spans, "head_position_spans", minimum=0)
        exact_identities: set[tuple[int, int]] = set()
        for route in self.exact_head_routes:
            parts = route.split(":")
            if len(parts) != 2:
                raise ConfigurationError(
                    "prototypes.exact_head_routes entries must use '<layer>:<kv_head>'"
                )
            try:
                identity = int(parts[0]), int(parts[1])
            except ValueError as error:
                raise ConfigurationError(
                    "prototypes.exact_head_routes entries must contain integers"
                ) from error
            if min(identity) < 0 or identity in exact_identities:
                raise ConfigurationError(
                    "prototypes.exact_head_routes require unique nonnegative identities"
                )
            exact_identities.add(identity)
        selection_only_identities: set[tuple[int, int]] = set()
        for route in self.selection_only_head_routes:
            parts = route.split(":")
            if len(parts) != 2:
                raise ConfigurationError(
                    "prototypes.selection_only_head_routes entries must use "
                    "'<layer>:<kv_head>'"
                )
            try:
                identity = int(parts[0]), int(parts[1])
            except ValueError as error:
                raise ConfigurationError(
                    "prototypes.selection_only_head_routes entries must contain integers"
                ) from error
            if min(identity) < 0 or identity in selection_only_identities:
                raise ConfigurationError(
                    "prototypes.selection_only_head_routes require unique nonnegative "
                    "identities"
                )
            selection_only_identities.add(identity)
        overlap = exact_identities & selection_only_identities
        if overlap:
            raise ConfigurationError(
                "prototype exact and selection-only head routes must be disjoint: "
                f"{sorted(overlap)}"
            )
        if self.maximum_prototypes_per_head is not None and (
            self.maximum_prototypes_per_head < 1
        ):
            raise ConfigurationError(
                "prototypes.maximum_prototypes_per_head must be >= 1 or null"
            )
        if not 0 < self.uncertain_head_fraction <= 1:
            raise ConfigurationError(
                "prototypes.uncertain_head_fraction must be in (0, 1]"
            )
        if not 0 <= self.uncertain_head_threshold <= 1:
            raise ConfigurationError(
                "prototypes.uncertain_head_threshold must be in [0, 1]"
            )
        valid_modalities = {"text", "image", "video"}
        if not self.source_modalities or any(
            modality not in valid_modalities for modality in self.source_modalities
        ):
            raise ConfigurationError(
                "prototypes.source_modalities must contain text, image, or video"
            )
        if len(set(self.source_modalities)) != len(self.source_modalities):
            raise ConfigurationError("prototypes.source_modalities cannot contain duplicates")
        if not self.allowed_modality_pairs:
            raise ConfigurationError("prototypes.allowed_modality_pairs cannot be empty")
        for pair in self.allowed_modality_pairs:
            if not isinstance(pair, str):
                raise ConfigurationError(
                    "prototypes.allowed_modality_pairs entries must be strings"
                )
            parts = pair.split(":")
            if len(parts) != 2 or any(part not in valid_modalities for part in parts):
                raise ConfigurationError(
                    "prototypes.allowed_modality_pairs entries must use "
                    "'<source>:<anchor>' with text, image, or video"
                )
        if len(set(self.allowed_modality_pairs)) != len(self.allowed_modality_pairs):
            raise ConfigurationError("prototypes.allowed_modality_pairs cannot contain duplicates")

    @classmethod
    def from_mapping(cls, value: object) -> PrototypeConfig:
        data = _mapping(value, "prototypes")
        allowed = {
            "enabled",
            "group_size",
            "max_position_span",
            "min_anchor_weight",
            "allowed_modality_pairs",
            "max_key_dispersion",
            "max_value_dispersion",
            "max_member_key_error",
            "max_member_value_error",
            "forecast_weight",
            "attention_log_bias_scale",
            "pooling_mode",
            "attention_bias_basis",
            "anchor_position_strategy",
            "adaptive_grouping",
            "exact_head_routes",
            "selection_only_head_routes",
            "maximum_prototypes_per_head",
            "uncertain_head_only",
            "uncertain_head_fraction",
            "uncertain_head_threshold",
            "source_modalities",
            "head_group_sizes",
            "head_position_spans",
            "source_gather_mode",
            "budget_planning_mode",
        }
        _reject_unknown(data, allowed, "prototypes")
        defaults = cls()
        return cls(
            enabled=_bool(data, "enabled", True, "prototypes"),
            group_size=_int(data, "group_size", 4, "prototypes"),
            max_position_span=_optional_int(data, "max_position_span", 128, "prototypes"),
            min_anchor_weight=_float(data, "min_anchor_weight", 0.0, "prototypes"),
            max_key_dispersion=(
                None
                if data.get("max_key_dispersion") is None
                else _float(data, "max_key_dispersion", 0.0, "prototypes")
            ),
            max_value_dispersion=(
                None
                if data.get("max_value_dispersion") is None
                else _float(data, "max_value_dispersion", 0.0, "prototypes")
            ),
            max_member_key_error=(
                None
                if data.get("max_member_key_error") is None
                else _float(data, "max_member_key_error", 0.0, "prototypes")
            ),
            max_member_value_error=(
                None
                if data.get("max_member_value_error") is None
                else _float(data, "max_member_value_error", 0.0, "prototypes")
            ),
            forecast_weight=_float(data, "forecast_weight", 0.0, "prototypes"),
            attention_log_bias_scale=_float(
                data,
                "attention_log_bias_scale",
                0.0,
                "prototypes",
            ),
            pooling_mode=_optional_str(
                data,
                "pooling_mode",
                "graph_weighted_block_mean",
                "prototypes",
            ),
            attention_bias_basis=_optional_str(
                data,
                "attention_bias_basis",
                "source_position_multiplicity",
                "prototypes",
            ),
            anchor_position_strategy=_optional_str(
                data,
                "anchor_position_strategy",
                "selected_anchor_start",
                "prototypes",
            ),
            adaptive_grouping=_bool(data, "adaptive_grouping", False, "prototypes"),
            exact_head_routes=_string_tuple(data, "exact_head_routes", (), "prototypes"),
            selection_only_head_routes=_string_tuple(
                data,
                "selection_only_head_routes",
                (),
                "prototypes",
            ),
            maximum_prototypes_per_head=_optional_int(
                data,
                "maximum_prototypes_per_head",
                None,
                "prototypes",
            ),
            uncertain_head_only=_bool(
                data,
                "uncertain_head_only",
                False,
                "prototypes",
            ),
            uncertain_head_fraction=_float(
                data,
                "uncertain_head_fraction",
                1.0,
                "prototypes",
            ),
            uncertain_head_threshold=_float(
                data,
                "uncertain_head_threshold",
                0.0,
                "prototypes",
            ),
            source_modalities=_string_tuple(
                data,
                "source_modalities",
                defaults.source_modalities,
                "prototypes",
            ),
            head_group_sizes=_string_tuple(data, "head_group_sizes", (), "prototypes"),
            head_position_spans=_string_tuple(data, "head_position_spans", (), "prototypes"),
            source_gather_mode=_optional_str(
                data,
                "source_gather_mode",
                "batched_layer",
                "prototypes",
            ),
            budget_planning_mode=_optional_str(
                data,
                "budget_planning_mode",
                "one_pass_anchor_cover",
                "prototypes",
            ),
            allowed_modality_pairs=_string_tuple(
                data,
                "allowed_modality_pairs",
                defaults.allowed_modality_pairs,
                "prototypes",
            ),
        )

    @staticmethod
    def _validate_head_routes(routes: tuple[str, ...], name: str, *, minimum: int) -> None:
        identities: set[tuple[int, int]] = set()
        for route in routes:
            parts = route.split(":")
            if len(parts) != 3:
                raise ConfigurationError(
                    f"prototypes.{name} entries must use '<layer>:<kv_head>:<value>'"
                )
            try:
                layer, head, value = (int(part) for part in parts)
            except ValueError as error:
                raise ConfigurationError(
                    f"prototypes.{name} entries must contain integers"
                ) from error
            if layer < 0 or head < 0 or value < minimum:
                raise ConfigurationError(
                    f"prototypes.{name} entries require nonnegative layer/head and "
                    f"value >= {minimum}"
                )
            if (layer, head) in identities:
                raise ConfigurationError(
                    f"prototypes.{name} cannot repeat layer/head {(layer, head)}"
                )
            identities.add((layer, head))

    @staticmethod
    def _head_route(routes: tuple[str, ...], layer: int, head: int) -> int | None:
        prefix = f"{layer}:{head}:"
        for route in routes:
            if route.startswith(prefix):
                return int(route.rsplit(":", maxsplit=1)[1])
        return None

    def group_size_for(self, layer: int, head: int) -> int:
        return self._head_route(self.head_group_sizes, layer, head) or self.group_size

    def position_span_for(self, layer: int, head: int) -> int | None:
        routed = self._head_route(self.head_position_spans, layer, head)
        return self.max_position_span if routed is None else routed

    def requires_exact_head(self, layer: int, head: int) -> bool:
        return f"{layer}:{head}" in self.exact_head_routes

    def uses_prototype_head(self, layer: int, head: int) -> bool:
        identity = f"{layer}:{head}"
        return identity not in self.exact_head_routes and identity not in (
            self.selection_only_head_routes
        )

    def uses_prototype_block(self, layer: int, head: int, modality: object) -> bool:
        value = getattr(modality, "value", modality)
        return self.uses_prototype_head(layer, head) and str(value) in self.source_modalities


@dataclass(frozen=True, slots=True)
class ResidualConfig:
    """Residual-tier controls."""

    enabled: bool = True
    rank: int = 8
    storage_dtype: ResidualStorageDType = ResidualStorageDType.LOSSLESS
    require_pinned_memory: bool = True
    index_mode: str = "compact"
    # These limits are applied before CPU allocation. ``None`` preserves the
    # original paper-faithful behavior in which every prototyped source block
    # remains repairable. Setting either limit creates an explicitly bounded
    # residual tier whose stored members are chosen by the declared policy.
    max_bytes: int | None = None
    max_source_blocks: int | None = None
    selection_policy: str = "risk_weighted_reconstruction_error"
    # Optional resident rescue payloads avoid a host transfer after a repair
    # trigger. They are outside the attention-active cache and are reported as
    # separate GPU-resident KV bytes; experiments must include them in total
    # GPU-memory comparisons.
    gpu_rescue_max_blocks: int = 0
    gpu_rescue_max_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ConfigurationError("residual.rank must be >= 1")
        if self.index_mode not in {"materialized", "compact"}:
            raise ConfigurationError("residual.index_mode must be 'materialized' or 'compact'")
        if self.max_bytes is not None and self.max_bytes < 0:
            raise ConfigurationError("residual.max_bytes must be >= 0 or null")
        if self.max_source_blocks is not None and self.max_source_blocks < 0:
            raise ConfigurationError("residual.max_source_blocks must be >= 0 or null")
        if self.gpu_rescue_max_blocks < 0:
            raise ConfigurationError("residual.gpu_rescue_max_blocks must be >= 0")
        if self.gpu_rescue_max_bytes is not None and self.gpu_rescue_max_bytes <= 0:
            raise ConfigurationError("residual.gpu_rescue_max_bytes must be > 0 or null")
        if self.selection_policy not in {
            "source_order",
            "risk_weighted_reconstruction_error",
            "prototype_coverage_then_risk",
        }:
            raise ConfigurationError(
                "residual.selection_policy must be 'source_order' or "
                "'risk_weighted_reconstruction_error' or "
                "'prototype_coverage_then_risk'"
            )

    @classmethod
    def from_mapping(cls, value: object) -> ResidualConfig:
        data = _mapping(value, "residual")
        _reject_unknown(
            data,
            {
                "enabled",
                "rank",
                "storage_dtype",
                "require_pinned_memory",
                "index_mode",
                "max_bytes",
                "max_source_blocks",
                "selection_policy",
                "gpu_rescue_max_blocks",
                "gpu_rescue_max_bytes",
            },
            "residual",
        )
        return cls(
            enabled=_bool(data, "enabled", True, "residual"),
            rank=_int(data, "rank", 8, "residual"),
            storage_dtype=_enum(
                ResidualStorageDType,
                data.get("storage_dtype", ResidualStorageDType.LOSSLESS.value),
                "residual.storage_dtype",
            ),
            require_pinned_memory=_bool(data, "require_pinned_memory", True, "residual"),
            index_mode=_optional_str(data, "index_mode", "compact", "residual"),
            max_bytes=_optional_int(data, "max_bytes", None, "residual"),
            max_source_blocks=_optional_int(
                data,
                "max_source_blocks",
                None,
                "residual",
            ),
            selection_policy=_optional_str(
                data,
                "selection_policy",
                "risk_weighted_reconstruction_error",
                "residual",
            ),
            gpu_rescue_max_blocks=_int(
                data,
                "gpu_rescue_max_blocks",
                0,
                "residual",
            ),
            gpu_rescue_max_bytes=_optional_int(
                data,
                "gpu_rescue_max_bytes",
                None,
                "residual",
            ),
        )


@dataclass(frozen=True, slots=True)
class LookMConfig:
    """Paper-facing LOOK-M ratios and KV-pair merge strategy."""

    enabled: bool = False
    recent_ratio: float = 0.1
    important_ratio: float = 0.1
    merge_strategy: LookMMergeStrategy = LookMMergeStrategy.PIVOTAL
    text_prior: bool = True
    official_repository_sha: str = "ecf0f51a9c416c2d85e47faf2638502f01a6d748"

    def __post_init__(self) -> None:
        for name in ("recent_ratio", "important_ratio"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ConfigurationError(f"lookm.{name} must be finite and in [0, 1]")
        if self.recent_ratio + self.important_ratio <= 0:
            raise ConfigurationError("lookm recent + important ratios must be positive")
        if self.recent_ratio + self.important_ratio > 1:
            raise ConfigurationError("lookm recent + important ratios cannot exceed 1")
        if len(self.official_repository_sha) != 40 or any(
            character not in "0123456789abcdef" for character in self.official_repository_sha
        ):
            raise ConfigurationError(
                "lookm.official_repository_sha must be a lowercase 40-character git SHA"
            )

    @classmethod
    def from_mapping(cls, value: object) -> LookMConfig:
        data = _mapping(value, "lookm")
        _reject_unknown(
            data,
            {
                "enabled",
                "recent_ratio",
                "important_ratio",
                "merge_strategy",
                "text_prior",
                "official_repository_sha",
            },
            "lookm",
        )
        return cls(
            enabled=_bool(data, "enabled", False, "lookm"),
            recent_ratio=_float(data, "recent_ratio", 0.1, "lookm"),
            important_ratio=_float(data, "important_ratio", 0.1, "lookm"),
            merge_strategy=_enum(
                LookMMergeStrategy,
                data.get("merge_strategy", LookMMergeStrategy.PIVOTAL.value),
                "lookm.merge_strategy",
            ),
            text_prior=_bool(data, "text_prior", True, "lookm"),
            official_repository_sha=_optional_str(
                data,
                "official_repository_sha",
                "ecf0f51a9c416c2d85e47faf2638502f01a6d748",
                "lookm",
            ),
        )


@dataclass(frozen=True, slots=True)
class PrefixKVConfig:
    """Paper-facing PrefixKV profile and fixed-distance eviction controls."""

    enabled: bool = False
    profile_mode: PrefixKVProfileMode = PrefixKVProfileMode.OFFLINE_PROFILE
    profile_path: str | None = None
    start_size: int = 1
    protect_size: int = 1
    eviction_distance: int = -25
    official_repository_sha: str = "597f1ab032704951550f93bcc8a23f1454b80aa4"

    def __post_init__(self) -> None:
        if self.start_size < 0:
            raise ConfigurationError("prefixkv.start_size must be >= 0")
        if self.protect_size < 1:
            raise ConfigurationError("prefixkv.protect_size must be >= 1")
        if self.eviction_distance == 0:
            raise ConfigurationError("prefixkv.eviction_distance cannot be zero")
        if self.profile_path is not None and not self.profile_path.strip():
            raise ConfigurationError("prefixkv.profile_path must be non-empty or null")
        if (
            self.enabled
            and self.profile_mode is PrefixKVProfileMode.OFFLINE_PROFILE
            and self.profile_path is None
        ):
            raise ConfigurationError(
                "prefixkv.profile_path is required for profile_mode='offline_profile'"
            )
        if self.profile_mode is PrefixKVProfileMode.FIXED_GLOBAL and self.profile_path is not None:
            raise ConfigurationError(
                "prefixkv.profile_path must be null for profile_mode='fixed_global'"
            )
        if len(self.official_repository_sha) != 40 or any(
            character not in "0123456789abcdef" for character in self.official_repository_sha
        ):
            raise ConfigurationError(
                "prefixkv.official_repository_sha must be a lowercase 40-character git SHA"
            )

    @classmethod
    def from_mapping(cls, value: object) -> PrefixKVConfig:
        data = _mapping(value, "prefixkv")
        _reject_unknown(
            data,
            {
                "enabled",
                "profile_mode",
                "profile_path",
                "start_size",
                "protect_size",
                "eviction_distance",
                "official_repository_sha",
            },
            "prefixkv",
        )
        raw_path = data.get("profile_path")
        if raw_path is not None and not isinstance(raw_path, str):
            raise ConfigurationError("prefixkv.profile_path must be a string or null")
        return cls(
            enabled=_bool(data, "enabled", False, "prefixkv"),
            profile_mode=_enum(
                PrefixKVProfileMode,
                data.get("profile_mode", PrefixKVProfileMode.OFFLINE_PROFILE.value),
                "prefixkv.profile_mode",
            ),
            profile_path=raw_path,
            start_size=_int(data, "start_size", 1, "prefixkv"),
            protect_size=_int(data, "protect_size", 1, "prefixkv"),
            eviction_distance=_int(data, "eviction_distance", -25, "prefixkv"),
            official_repository_sha=_optional_str(
                data,
                "official_repository_sha",
                "597f1ab032704951550f93bcc8a23f1454b80aa4",
                "prefixkv",
            ),
        )


@dataclass(frozen=True, slots=True)
class VLCacheConfig:
    """ICLR VL-Cache paper parameters plus explicit ambiguity controls.

    The paper algorithm is prompt-adaptive and does not require an offline
    profile.  Calibration provenance is optional and is used only when these
    ambiguity controls are tuned rather than left at the paper defaults.
    """

    enabled: bool = False
    sparsity_threshold: float = 0.01
    min_layer_retention: float = 0.01
    max_layer_retention: float = 1.0
    recent_window_fraction: float = 0.1
    max_post_vision_queries: int | None = None
    calibration_dataset_id: str | None = None
    calibration_dataset_revision: str | None = None
    calibration_split: str | None = None
    calibration_sample_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not math.isfinite(self.sparsity_threshold) or not 0 < self.sparsity_threshold < 1:
            raise ConfigurationError("vl_cache.sparsity_threshold must be finite and in (0, 1)")
        for name in ("min_layer_retention", "max_layer_retention"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ConfigurationError(f"vl_cache.{name} must be finite and in (0, 1]")
        if self.min_layer_retention > self.max_layer_retention:
            raise ConfigurationError(
                "vl_cache.min_layer_retention cannot exceed max_layer_retention"
            )
        if not math.isfinite(self.recent_window_fraction) or not (
            0 <= self.recent_window_fraction <= 1
        ):
            raise ConfigurationError("vl_cache.recent_window_fraction must be finite and in [0, 1]")
        if self.max_post_vision_queries is not None and self.max_post_vision_queries < 1:
            raise ConfigurationError("vl_cache.max_post_vision_queries must be >= 1 or null")
        if len(self.calibration_sample_ids) != len(set(self.calibration_sample_ids)):
            raise ConfigurationError("vl_cache.calibration_sample_ids cannot contain duplicates")
        calibration_metadata = (
            self.calibration_dataset_id,
            self.calibration_dataset_revision,
            self.calibration_split,
        )
        has_metadata = any(value is not None for value in calibration_metadata)
        if self.calibration_sample_ids and not all(
            isinstance(value, str) and value.strip() for value in calibration_metadata
        ):
            raise ConfigurationError(
                "vl_cache calibration sample IDs require dataset ID, revision, and split"
            )
        if has_metadata and not self.calibration_sample_ids:
            raise ConfigurationError(
                "vl_cache calibration metadata requires non-empty calibration_sample_ids"
            )

    @classmethod
    def from_mapping(cls, value: object) -> VLCacheConfig:
        data = _mapping(value, "vl_cache")
        _reject_unknown(
            data,
            {
                "enabled",
                "sparsity_threshold",
                "min_layer_retention",
                "max_layer_retention",
                "recent_window_fraction",
                "max_post_vision_queries",
                "calibration_dataset_id",
                "calibration_dataset_revision",
                "calibration_split",
                "calibration_sample_ids",
            },
            "vl_cache",
        )
        metadata: dict[str, str | None] = {}
        for key in (
            "calibration_dataset_id",
            "calibration_dataset_revision",
            "calibration_split",
        ):
            raw = data.get(key)
            if raw is not None and (not isinstance(raw, str) or not raw.strip()):
                raise ConfigurationError(f"vl_cache.{key} must be a non-empty string or null")
            metadata[key] = raw
        return cls(
            enabled=_bool(data, "enabled", False, "vl_cache"),
            sparsity_threshold=_float(data, "sparsity_threshold", 0.01, "vl_cache"),
            min_layer_retention=_float(data, "min_layer_retention", 0.01, "vl_cache"),
            max_layer_retention=_float(data, "max_layer_retention", 1.0, "vl_cache"),
            recent_window_fraction=_float(data, "recent_window_fraction", 0.1, "vl_cache"),
            max_post_vision_queries=_optional_int(
                data, "max_post_vision_queries", None, "vl_cache"
            ),
            calibration_dataset_id=metadata["calibration_dataset_id"],
            calibration_dataset_revision=metadata["calibration_dataset_revision"],
            calibration_split=metadata["calibration_split"],
            calibration_sample_ids=_string_tuple(data, "calibration_sample_ids", (), "vl_cache"),
        )


@dataclass(frozen=True, slots=True)
class RepairConfig:
    """Uncertainty-guided residual repair controls."""

    enabled: bool = True
    policy: RepairPolicy = RepairPolicy.ENTROPY_OR_PROTOTYPE_RISK
    entropy_threshold: float = 0.5
    prototype_risk_threshold: float = 0.25
    prototype_risk_min_entropy: float = 0.0
    decision_margin_threshold: float = 0.15
    draft_kl_threshold: float | None = None
    counterfactual_draft_steps: int = 0
    draft_require_token_mismatch: bool = False
    counterfactual_route_heads: bool = False
    normalize_prototype_risk: bool = False
    decision_high_risk_threshold: float | None = None
    max_blocks_per_step: int = 2
    candidate_ranking: RepairCandidateRanking = (
        RepairCandidateRanking.RISK_WEIGHTED_RECONSTRUCTION_ERROR
    )
    preserve_partial_prototype: bool = True
    evaluation_only: bool = False

    def __post_init__(self) -> None:
        if not 0 <= self.entropy_threshold <= 1:
            raise ConfigurationError("repair.entropy_threshold must be in the interval [0, 1]")
        if not math.isfinite(self.prototype_risk_threshold) or self.prototype_risk_threshold < 0:
            raise ConfigurationError("repair.prototype_risk_threshold must be finite and >= 0")
        if not 0 <= self.prototype_risk_min_entropy <= 1:
            raise ConfigurationError(
                "repair.prototype_risk_min_entropy must be in the interval [0, 1]"
            )
        if not 0 <= self.decision_margin_threshold <= 1:
            raise ConfigurationError("repair.decision_margin_threshold must be in [0, 1]")
        if self.draft_kl_threshold is not None and (
            not math.isfinite(self.draft_kl_threshold) or self.draft_kl_threshold < 0
        ):
            raise ConfigurationError("repair.draft_kl_threshold must be finite and >= 0 or null")
        if self.draft_kl_threshold is not None and self.policy is not RepairPolicy.DECISION_AWARE:
            raise ConfigurationError("repair.draft_kl_threshold requires policy='decision_aware'")
        if self.counterfactual_draft_steps < 0:
            raise ConfigurationError("repair.counterfactual_draft_steps must be >= 0")
        if self.draft_require_token_mismatch and self.draft_kl_threshold is None:
            raise ConfigurationError(
                "repair.draft_require_token_mismatch requires draft_kl_threshold"
            )
        if self.counterfactual_route_heads and self.counterfactual_draft_steps < 1:
            raise ConfigurationError(
                "repair.counterfactual_route_heads requires counterfactual_draft_steps >= 1"
            )
        if self.decision_high_risk_threshold is not None and (
            not math.isfinite(self.decision_high_risk_threshold)
            or self.decision_high_risk_threshold < 0
        ):
            raise ConfigurationError(
                "repair.decision_high_risk_threshold must be finite and >= 0 or null"
            )
        if self.max_blocks_per_step < 0:
            raise ConfigurationError("repair.max_blocks_per_step must be >= 0")
        if self.enabled and self.policy is not RepairPolicy.NONE and self.max_blocks_per_step < 1:
            raise ConfigurationError(
                "repair.max_blocks_per_step must be >= 1 for an enabled repair policy"
            )
        if self.policy is RepairPolicy.ORACLE and not self.evaluation_only:
            raise ConfigurationError("repair.policy='oracle' requires repair.evaluation_only=true")

    @classmethod
    def from_mapping(cls, value: object) -> RepairConfig:
        data = _mapping(value, "repair")
        allowed = {
            "enabled",
            "policy",
            "entropy_threshold",
            "prototype_risk_threshold",
            "prototype_risk_min_entropy",
            "decision_margin_threshold",
            "draft_kl_threshold",
            "counterfactual_draft_steps",
            "draft_require_token_mismatch",
            "counterfactual_route_heads",
            "normalize_prototype_risk",
            "decision_high_risk_threshold",
            "max_blocks_per_step",
            "candidate_ranking",
            "preserve_partial_prototype",
            "evaluation_only",
        }
        _reject_unknown(data, allowed, "repair")
        return cls(
            enabled=_bool(data, "enabled", True, "repair"),
            policy=_enum(
                RepairPolicy,
                data.get("policy", RepairPolicy.ENTROPY_OR_PROTOTYPE_RISK.value),
                "repair.policy",
            ),
            entropy_threshold=_float(data, "entropy_threshold", 0.5, "repair"),
            prototype_risk_threshold=_float(data, "prototype_risk_threshold", 0.25, "repair"),
            prototype_risk_min_entropy=_float(data, "prototype_risk_min_entropy", 0.0, "repair"),
            decision_margin_threshold=_float(data, "decision_margin_threshold", 0.15, "repair"),
            draft_kl_threshold=(
                None
                if data.get("draft_kl_threshold") is None
                else _float(data, "draft_kl_threshold", 0.0, "repair")
            ),
            counterfactual_draft_steps=_int(
                data, "counterfactual_draft_steps", 0, "repair"
            ),
            draft_require_token_mismatch=_bool(
                data, "draft_require_token_mismatch", False, "repair"
            ),
            counterfactual_route_heads=_bool(
                data, "counterfactual_route_heads", False, "repair"
            ),
            normalize_prototype_risk=_bool(data, "normalize_prototype_risk", False, "repair"),
            decision_high_risk_threshold=(
                None
                if data.get("decision_high_risk_threshold") is None
                else _float(data, "decision_high_risk_threshold", 0.0, "repair")
            ),
            max_blocks_per_step=_int(data, "max_blocks_per_step", 2, "repair"),
            candidate_ranking=_enum(
                RepairCandidateRanking,
                data.get(
                    "candidate_ranking",
                    RepairCandidateRanking.RISK_WEIGHTED_RECONSTRUCTION_ERROR.value,
                ),
                "repair.candidate_ranking",
            ),
            preserve_partial_prototype=_bool(data, "preserve_partial_prototype", True, "repair"),
            evaluation_only=_bool(data, "evaluation_only", False, "repair"),
        )


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Fully resolved MosaicKV run configuration."""

    model: ModelConfig
    dataset: DatasetConfig
    execution: ExecutionConfig
    generation: GenerationConfig
    cache: CacheConfig
    diagnostics: FidelityDiagnosticsConfig = field(default_factory=FidelityDiagnosticsConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    method: MosaicKVMethod = MosaicKVMethod.FULLKV
    forecasting: ForecastingConfig = field(default_factory=ForecastingConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    utility: UtilityConfig = field(default_factory=UtilityConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    prototypes: PrototypeConfig = field(default_factory=PrototypeConfig)
    residual: ResidualConfig = field(default_factory=ResidualConfig)
    repair: RepairConfig = field(default_factory=RepairConfig)
    lookm: LookMConfig = field(default_factory=LookMConfig)
    prefixkv: PrefixKVConfig = field(default_factory=PrefixKVConfig)
    vl_cache: VLCacheConfig = field(default_factory=VLCacheConfig)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ConfigurationError("schema_version must equal 1")
        if self.ablation.variant is not AblationVariant.NONE and not self.method.is_mosaickv:
            raise ConfigurationError("component ablations require a MosaicKV method")
        if (
            self.ablation.variant is AblationVariant.NO_FUTURE_QUERY_FORECAST
            and self.forecasting.enabled
        ):
            raise ConfigurationError("no_future_query_forecast requires forecasting.enabled=false")
        if (
            self.ablation.variant is not AblationVariant.NO_FUTURE_QUERY_FORECAST
            and self.method.is_mosaickv
            and not self.forecasting.enabled
        ):
            raise ConfigurationError(
                "MosaicKV requires forecasting unless the declared ablation removes it"
            )
        if self.repair.draft_kl_threshold is not None and (
            not self.method.is_mosaickv
            or not self.forecasting.enabled
            or (
                self.forecasting.draft_steps < 1
                and self.repair.counterfactual_draft_steps < 1
            )
        ):
            raise ConfigurationError(
                "repair.draft_kl_threshold requires either forecasting.draft_steps >= 1 "
                "or repair.counterfactual_draft_steps >= 1"
            )
        if (
            self.method.is_mosaickv
            and self.graph.similarity_shortlist_multiplier > 0
            and self.forecasting.prompt_window < 1
        ):
            raise ConfigurationError(
                "graph prompt shortlisting requires forecasting.prompt_window >= 1"
            )
        if self.selection.routed_head_only_planning and (
            not self.method.is_mosaickv
            or not self.graph.enabled
            or self.graph.candidate_graph_mode != "shortlist_rerank"
            or not self.graph.candidate_descriptor_pooling
            or not self.prototypes.uncertain_head_only
        ):
            raise ConfigurationError(
                "routed-head-only planning requires a MosaicKV method, enabled "
                "shortlist_rerank graph with candidate descriptor pooling, and "
                "prototypes.uncertain_head_only=true"
            )
        if (self.residual.gpu_rescue_max_blocks or self.residual.gpu_rescue_max_bytes) and (
            self.method is not MosaicKVMethod.MOSAICKV_FULL
            or not self.residual.enabled
            or not self.repair.enabled
            or self.repair.policy is RepairPolicy.NONE
        ):
            raise ConfigurationError(
                "residual GPU rescue requires mosaickv_full with residual repair enabled"
            )
        variant = self.ablation.variant
        if variant is AblationVariant.NO_EVIDENCE_GRAPH and self.graph.enabled:
            raise ConfigurationError("no_evidence_graph requires graph.enabled=false")
        if variant is AblationVariant.SEMANTIC_ONLY_GRAPH:
            nonsemantic = (
                self.graph.attention_weight,
                self.graph.spatial_weight,
                self.graph.layout_weight,
                self.graph.temporal_weight,
                self.graph.same_region_weight,
                self.graph.cross_modal_weight,
                self.graph.fallback_weight,
            )
            if not self.graph.enabled or self.graph.semantic_weight <= 0 or any(nonsemantic):
                raise ConfigurationError(
                    "semantic_only_graph requires an enabled positive semantic source and "
                    "all other graph-source weights equal to zero"
                )
        if variant is AblationVariant.NO_VALUE_AWARE_CONTRIBUTION and self.utility.lambda_v != 0:
            raise ConfigurationError("no_value_aware_contribution requires utility.lambda_v=0")
        if variant is AblationVariant.ATTENTION_ONLY_UTILITY and (
            self.utility.lambda_v != 0
            or self.utility.lambda_o != 0
            or self.selection.lambda_g != 0
            or self.selection.lambda_m != 0
        ):
            raise ConfigurationError(
                "attention_only_utility requires value, uniqueness, graph-coverage, and "
                "modality-coverage coefficients equal to zero"
            )
        if variant is AblationVariant.NO_PROTOTYPE_MERGING and (
            self.prototypes.enabled or self.residual.enabled or self.repair.enabled
        ):
            raise ConfigurationError(
                "no_prototype_merging requires prototype, residual, and repair stages disabled"
            )
        if variant is AblationVariant.NO_RESIDUAL_REPAIR and (
            not self.prototypes.enabled
            or self.residual.enabled
            or self.repair.enabled
            or self.repair.policy is not RepairPolicy.NONE
        ):
            raise ConfigurationError(
                "no_residual_repair requires prototypes enabled and residual/repair disabled"
            )
        if variant is AblationVariant.ORACLE_FUTURE_QUERY:
            if self.diagnostics.forecast_oracle_steps < 1:
                raise ConfigurationError(
                    "oracle future-query ablation requires diagnostics.forecast_oracle_steps > 0"
                )
            if (
                self.forecasting.mode is not ForecastMode.DRAFT_ROLLOUT
                or self.forecasting.prompt_window != 0
                or self.forecasting.draft_steps != self.diagnostics.forecast_oracle_steps
            ):
                raise ConfigurationError(
                    "oracle future-query ablation uses draft_rollout geometry with draft_steps "
                    "equal to diagnostics.forecast_oracle_steps and prompt_window=0"
                )
        if variant is AblationVariant.ORACLE_REPAIR and (
            not self.repair.enabled
            or self.repair.policy is not RepairPolicy.ORACLE
            or not self.repair.evaluation_only
        ):
            raise ConfigurationError(
                "oracle repair ablation requires enabled evaluation-only oracle repair"
            )
        if self.method is MosaicKVMethod.FULL_KV and self.cache.retention_ratio != 1.0:
            raise ConfigurationError("method='full_kv' requires cache.retention_ratio=1.0")
        if (
            self.method is MosaicKVMethod.PROMPT_ATTENTION_TOPK
            and self.forecasting.prompt_window < 1
        ):
            raise ConfigurationError(
                "method='prompt_attention_topk' requires forecasting.prompt_window >= 1"
            )
        if self.method.is_simple_baseline:
            enabled = []
            if self.forecasting.enabled:
                enabled.append("forecasting")
            if self.graph.enabled:
                enabled.append("graph")
            if self.selection.enabled:
                enabled.append("selection")
            if self.prototypes.enabled:
                enabled.append("prototypes")
            if self.residual.enabled:
                enabled.append("residual")
            if self.repair.enabled or self.repair.policy is not RepairPolicy.NONE:
                enabled.append("repair")
            if enabled:
                raise ConfigurationError(
                    f"method={self.method.value!r} requires disabled MosaicKV stages: "
                    + ", ".join(enabled)
                )
        if self.method.is_lookm_reimplementation:
            if not self.lookm.enabled:
                raise ConfigurationError("method='lookm_reimpl' requires lookm.enabled=true")
            if not self.lookm.text_prior:
                raise ConfigurationError(
                    "method='lookm_reimpl' is the paper text-prior variant and requires "
                    "lookm.text_prior=true"
                )
            if self.cache.block_size != 1:
                raise ConfigurationError(
                    "method='lookm_reimpl' requires cache.block_size=1 for token/head selection"
                )
            expected_ratio = self.lookm.recent_ratio + self.lookm.important_ratio
            if not math.isclose(
                self.cache.retention_ratio,
                expected_ratio,
                rel_tol=0,
                abs_tol=1e-12,
            ):
                raise ConfigurationError(
                    "lookm recent + important ratios must equal cache.retention_ratio"
                )
            if self.cache.budget_unit not in {BudgetUnit.BLOCKS, BudgetUnit.BYTES}:
                raise ConfigurationError(
                    "method='lookm_reimpl' requires cache.budget_unit='blocks' or 'bytes'"
                )
            enabled = []
            if self.forecasting.enabled:
                enabled.append("forecasting")
            if self.graph.enabled:
                enabled.append("graph")
            if self.selection.enabled:
                enabled.append("selection")
            if self.prototypes.enabled:
                enabled.append("prototypes")
            if self.residual.enabled:
                enabled.append("residual")
            if self.repair.enabled or self.repair.policy is not RepairPolicy.NONE:
                enabled.append("repair")
            if enabled:
                raise ConfigurationError(
                    "method='lookm_reimpl' requires disabled MosaicKV stages: " + ", ".join(enabled)
                )
        elif self.lookm.enabled:
            raise ConfigurationError("lookm.enabled=true is reserved for method='lookm_reimpl'")
        if self.method.is_prefixkv_reimplementation:
            if not self.prefixkv.enabled:
                raise ConfigurationError("method='prefixkv_reimpl' requires prefixkv.enabled=true")
            if self.cache.block_size != 1:
                raise ConfigurationError(
                    "method='prefixkv_reimpl' requires cache.block_size=1 for token selection"
                )
            if self.cache.budget_unit not in {BudgetUnit.BLOCKS, BudgetUnit.BYTES}:
                raise ConfigurationError(
                    "method='prefixkv_reimpl' requires cache.budget_unit='blocks' or 'bytes'"
                )
            enabled = []
            if self.forecasting.enabled:
                enabled.append("forecasting")
            if self.graph.enabled:
                enabled.append("graph")
            if self.selection.enabled:
                enabled.append("selection")
            if self.prototypes.enabled:
                enabled.append("prototypes")
            if self.residual.enabled:
                enabled.append("residual")
            if self.repair.enabled or self.repair.policy is not RepairPolicy.NONE:
                enabled.append("repair")
            if enabled:
                raise ConfigurationError(
                    "method='prefixkv_reimpl' requires disabled MosaicKV stages: "
                    + ", ".join(enabled)
                )
        elif self.prefixkv.enabled:
            raise ConfigurationError(
                "prefixkv.enabled=true is reserved for method='prefixkv_reimpl'"
            )
        if self.method.is_vl_cache_reimplementation:
            if not self.vl_cache.enabled:
                raise ConfigurationError("method='vl_cache_reimpl' requires vl_cache.enabled=true")
            if self.cache.block_size != 1:
                raise ConfigurationError(
                    "method='vl_cache_reimpl' requires cache.block_size=1 for token/head selection"
                )
            if self.cache.budget_unit not in {BudgetUnit.BLOCKS, BudgetUnit.BYTES}:
                raise ConfigurationError(
                    "method='vl_cache_reimpl' requires cache.budget_unit='blocks' or 'bytes'"
                )
            enabled = []
            if self.forecasting.enabled:
                enabled.append("forecasting")
            if self.graph.enabled:
                enabled.append("graph")
            if self.selection.enabled:
                enabled.append("selection")
            if self.prototypes.enabled:
                enabled.append("prototypes")
            if self.residual.enabled:
                enabled.append("residual")
            if self.repair.enabled or self.repair.policy is not RepairPolicy.NONE:
                enabled.append("repair")
            if enabled:
                raise ConfigurationError(
                    "method='vl_cache_reimpl' requires disabled MosaicKV stages: "
                    + ", ".join(enabled)
                )
        elif self.vl_cache.enabled:
            raise ConfigurationError(
                "vl_cache.enabled=true is reserved for method='vl_cache_reimpl'"
            )

    @classmethod
    def from_mapping(cls, value: object) -> RunConfig:
        data = _mapping(value, "config")
        allowed = {
            "schema_version",
            "model",
            "dataset",
            "execution",
            "generation",
            "cache",
            "diagnostics",
            "ablation",
            "method",
            "forecasting",
            "graph",
            "utility",
            "selection",
            "prototypes",
            "residual",
            "repair",
            "lookm",
            "prefixkv",
            "vl_cache",
        }
        _reject_unknown(data, allowed, "config")
        required = {"model", "dataset", "execution", "generation", "cache"}
        missing = sorted(required - set(data))
        if missing:
            raise ConfigurationError(f"config is missing required section(s): {', '.join(missing)}")
        return cls(
            model=ModelConfig.from_mapping(data["model"]),
            dataset=DatasetConfig.from_mapping(data["dataset"]),
            execution=ExecutionConfig.from_mapping(data["execution"]),
            generation=GenerationConfig.from_mapping(data["generation"]),
            cache=CacheConfig.from_mapping(data["cache"]),
            diagnostics=FidelityDiagnosticsConfig.from_mapping(data.get("diagnostics", {})),
            ablation=AblationConfig.from_mapping(data.get("ablation", {})),
            method=_enum(
                MosaicKVMethod,
                data.get("method", MosaicKVMethod.FULLKV.value),
                "method",
            ),
            forecasting=ForecastingConfig.from_mapping(data.get("forecasting", {})),
            graph=GraphConfig.from_mapping(data.get("graph", {})),
            utility=UtilityConfig.from_mapping(data.get("utility", {})),
            selection=SelectionConfig.from_mapping(data.get("selection", {})),
            prototypes=PrototypeConfig.from_mapping(data.get("prototypes", {})),
            residual=ResidualConfig.from_mapping(data.get("residual", {})),
            repair=RepairConfig.from_mapping(data.get("repair", {})),
            lookm=LookMConfig.from_mapping(data.get("lookm", {})),
            prefixkv=PrefixKVConfig.from_mapping(data.get("prefixkv", {})),
            vl_cache=VLCacheConfig.from_mapping(data.get("vl_cache", {})),
            schema_version=_int(data, "schema_version", 1, "config"),
        )


def load_config(path: str | Path) -> RunConfig:
    """Load strict JSON, TOML, or YAML without environment interpolation."""

    config_path = Path(path)
    try:
        raw = config_path.read_bytes()
    except OSError as error:
        raise ConfigurationError(f"cannot read configuration {config_path}: {error}") from error

    try:
        if config_path.suffix == ".json":
            parsed = json.loads(raw)
        elif config_path.suffix == ".toml":
            parsed = tomllib.loads(raw.decode("utf-8"))
        elif config_path.suffix in {".yaml", ".yml"}:
            parsed = yaml.safe_load(raw.decode("utf-8"))
        else:
            raise ConfigurationError(
                f"unsupported configuration format {config_path.suffix!r}; "
                "use .json, .toml, .yaml, or .yml"
            )
    except (
        json.JSONDecodeError,
        tomllib.TOMLDecodeError,
        UnicodeDecodeError,
        yaml.YAMLError,
    ) as error:
        raise ConfigurationError(
            f"invalid configuration syntax in {config_path}: {error}"
        ) from error
    return RunConfig.from_mapping(parsed)


def canonical_config(config: RunConfig) -> JsonObject:
    """Return the machine-independent canonical configuration object."""

    payload = cast("JsonObject", asdict(config))
    # Preserve hashes of pre-Experiment-3 configurations. The explicit field
    # enters provenance only when an ablation is actually requested.
    if config.ablation.variant is AblationVariant.NONE:
        payload.pop("ablation", None)
    return payload


def canonical_config_json(config: RunConfig) -> str:
    """Serialize the resolved configuration with deterministic ordering."""

    return json.dumps(
        canonical_config(config), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def config_sha256(config: RunConfig) -> str:
    """Hash the canonical resolved configuration."""

    return hashlib.sha256(canonical_config_json(config).encode("utf-8")).hexdigest()


def synthetic_smoke_config(seed: int = 0) -> RunConfig:
    """Build a validated configuration for non-measured CPU smoke tests."""

    return RunConfig(
        model=ModelConfig("synthetic/smoke", "schema-v1", Precision.FP32),
        dataset=DatasetConfig("synthetic/smoke", "schema-v1", "validation"),
        execution=ExecutionConfig(Backend.SYNTHETIC, "numpy", seed, True),
        generation=GenerationConfig(),
        cache=CacheConfig(64, BudgetUnit.RETAINED_SLOTS, 1.0, 16),
    )


def synthetic_evaluation_config(seed: int = 0) -> RunConfig:
    """Build provenance for the packaged CPU-only evaluation fixture."""

    return RunConfig(
        model=ModelConfig("mosaickv/synthetic-color-model", "schema-v1", Precision.FP32),
        dataset=DatasetConfig("mosaickv/synthetic-ci", "schema-v1", "test"),
        execution=ExecutionConfig(Backend.SYNTHETIC, "python", seed, True),
        generation=GenerationConfig(max_new_tokens=1),
        cache=CacheConfig(1, BudgetUnit.RETAINED_SLOTS, 1.0, 1),
        forecasting=ForecastingConfig(enabled=False),
        graph=GraphConfig(enabled=False),
        utility=UtilityConfig(),
        selection=SelectionConfig(enabled=False),
        prototypes=PrototypeConfig(enabled=False),
        residual=ResidualConfig(enabled=False),
        repair=RepairConfig(enabled=False),
    )


__all__ = [
    "AblationConfig",
    "CacheConfig",
    "ConfigurationError",
    "DatasetConfig",
    "ExecutionConfig",
    "ForecastingConfig",
    "GenerationConfig",
    "GraphConfig",
    "LookMConfig",
    "ModelConfig",
    "PrefixKVConfig",
    "PrototypeConfig",
    "RepairConfig",
    "ResidualConfig",
    "RunConfig",
    "SelectionConfig",
    "UtilityConfig",
    "VLCacheConfig",
    "canonical_config",
    "canonical_config_json",
    "config_sha256",
    "load_config",
    "synthetic_evaluation_config",
    "synthetic_smoke_config",
]
