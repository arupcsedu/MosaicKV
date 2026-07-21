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

    def __post_init__(self) -> None:
        if not self.attention_implementation.strip():
            raise ConfigurationError("execution.attention_implementation must be non-empty")
        if self.seed < 0:
            raise ConfigurationError("execution.seed must be >= 0")

    @classmethod
    def from_mapping(cls, value: object) -> ExecutionConfig:
        data = _mapping(value, "execution")
        allowed = {"backend", "attention_implementation", "seed", "deterministic_algorithms"}
        _reject_unknown(data, allowed, "execution")
        return cls(
            backend=_enum(Backend, data.get("backend"), "execution.backend"),
            attention_implementation=_required_str(data, "attention_implementation", "execution"),
            seed=_int(data, "seed", 0, "execution"),
            deterministic_algorithms=_bool(data, "deterministic_algorithms", True, "execution"),
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
class CacheConfig:
    """Cache budget and block layout."""

    budget_value: int
    budget_unit: BudgetUnit
    retention_ratio: float
    block_size: int = 16
    accounting_spec_sha: str = _SYNTHETIC_ACCOUNTING_SHA

    def __post_init__(self) -> None:
        if self.budget_value < 1:
            raise ConfigurationError("cache.budget_value must be >= 1")
        if not 0 < self.retention_ratio <= 1:
            raise ConfigurationError("cache.retention_ratio must be in the interval (0, 1]")
        if self.block_size < 1:
            raise ConfigurationError("cache.block_size must be >= 1")
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

    def __post_init__(self) -> None:
        if self.prompt_window < 0:
            raise ConfigurationError("forecasting.prompt_window must be >= 0")
        if self.draft_steps < 0:
            raise ConfigurationError("forecasting.draft_steps must be >= 0")
        if self.centroid_count < 1:
            raise ConfigurationError("forecasting.centroid_count must be >= 1")
        if self.centroid_iterations < 1:
            raise ConfigurationError("forecasting.centroid_iterations must be >= 1")
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
        )


@dataclass(frozen=True, slots=True)
class UtilityConfig:
    """Weights in the preregistered per-block local utility equation."""

    lambda_q: float = 1.0
    lambda_v: float = 0.25
    lambda_o: float = 0.25

    def __post_init__(self) -> None:
        for name in ("lambda_q", "lambda_v", "lambda_o"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ConfigurationError(f"utility.{name} must be finite and >= 0")

    @classmethod
    def from_mapping(cls, value: object) -> UtilityConfig:
        data = _mapping(value, "utility")
        _reject_unknown(data, {"lambda_q", "lambda_v", "lambda_o"}, "utility")
        return cls(
            lambda_q=_float(data, "lambda_q", 1.0, "utility"),
            lambda_v=_float(data, "lambda_v", 0.25, "utility"),
            lambda_o=_float(data, "lambda_o", 0.25, "utility"),
        )


@dataclass(frozen=True, slots=True)
class SelectionConfig:
    """Budgeted submodular selection controls."""

    enabled: bool = True
    algorithm: str = "lazy_greedy"
    lambda_g: float = -0.25
    lambda_m: float = -0.25
    stop_on_nonpositive_gain: bool = True
    exhaustive_max_nodes: int = 20

    def __post_init__(self) -> None:
        if self.algorithm != "lazy_greedy":
            raise ConfigurationError("selection.algorithm must be 'lazy_greedy'")
        for name in ("lambda_g", "lambda_m"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ConfigurationError(f"selection.{name} must be finite")
        if self.exhaustive_max_nodes < 1:
            raise ConfigurationError("selection.exhaustive_max_nodes must be >= 1")

    @classmethod
    def from_mapping(cls, value: object) -> SelectionConfig:
        data = _mapping(value, "selection")
        allowed = {
            "enabled",
            "algorithm",
            "lambda_g",
            "lambda_m",
            "stop_on_nonpositive_gain",
            "exhaustive_max_nodes",
        }
        _reject_unknown(data, allowed, "selection")
        return cls(
            enabled=_bool(data, "enabled", True, "selection"),
            algorithm=_optional_str(data, "algorithm", "lazy_greedy", "selection"),
            lambda_g=_float(data, "lambda_g", -0.25, "selection"),
            lambda_m=_float(data, "lambda_m", -0.25, "selection"),
            stop_on_nonpositive_gain=_bool(data, "stop_on_nonpositive_gain", True, "selection"),
            exhaustive_max_nodes=_int(data, "exhaustive_max_nodes", 20, "selection"),
        )


@dataclass(frozen=True, slots=True)
class PrototypeConfig:
    """Prototype-tier controls."""

    enabled: bool = True
    group_size: int = 4
    max_position_span: int | None = 128
    min_anchor_weight: float = 0.0
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
        valid_modalities = {"text", "image", "video"}
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
        }
        _reject_unknown(data, allowed, "prototypes")
        defaults = cls()
        return cls(
            enabled=_bool(data, "enabled", True, "prototypes"),
            group_size=_int(data, "group_size", 4, "prototypes"),
            max_position_span=_optional_int(data, "max_position_span", 128, "prototypes"),
            min_anchor_weight=_float(data, "min_anchor_weight", 0.0, "prototypes"),
            allowed_modality_pairs=_string_tuple(
                data,
                "allowed_modality_pairs",
                defaults.allowed_modality_pairs,
                "prototypes",
            ),
        )


@dataclass(frozen=True, slots=True)
class ResidualConfig:
    """Residual-tier controls."""

    enabled: bool = True
    rank: int = 8
    storage_dtype: ResidualStorageDType = ResidualStorageDType.LOSSLESS
    require_pinned_memory: bool = True

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ConfigurationError("residual.rank must be >= 1")

    @classmethod
    def from_mapping(cls, value: object) -> ResidualConfig:
        data = _mapping(value, "residual")
        _reject_unknown(
            data,
            {"enabled", "rank", "storage_dtype", "require_pinned_memory"},
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
    max_blocks_per_step: int = 2
    evaluation_only: bool = False

    def __post_init__(self) -> None:
        if not 0 <= self.entropy_threshold <= 1:
            raise ConfigurationError("repair.entropy_threshold must be in the interval [0, 1]")
        if not math.isfinite(self.prototype_risk_threshold) or self.prototype_risk_threshold < 0:
            raise ConfigurationError("repair.prototype_risk_threshold must be finite and >= 0")
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
            "max_blocks_per_step",
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
            max_blocks_per_step=_int(data, "max_blocks_per_step", 2, "repair"),
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
            if self.cache.budget_unit is not BudgetUnit.BLOCKS:
                raise ConfigurationError(
                    "method='lookm_reimpl' requires cache.budget_unit='blocks'"
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
            if self.cache.budget_unit is not BudgetUnit.BLOCKS:
                raise ConfigurationError(
                    "method='vl_cache_reimpl' requires cache.budget_unit='blocks'"
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

    return cast("JsonObject", asdict(config))


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
