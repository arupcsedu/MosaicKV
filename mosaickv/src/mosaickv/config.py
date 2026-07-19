"""Strict, dependency-light configuration schema for MosaicKV runs."""

from __future__ import annotations

import hashlib
import json
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TypeVar, cast

from mosaickv.types import Backend, BudgetUnit, JsonObject, OutputLengthPolicy, Precision

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
    horizon: int = 4
    candidates: int = 4

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ConfigurationError("forecasting.horizon must be >= 1")
        if self.candidates < 1:
            raise ConfigurationError("forecasting.candidates must be >= 1")

    @classmethod
    def from_mapping(cls, value: object) -> ForecastingConfig:
        data = _mapping(value, "forecasting")
        _reject_unknown(data, {"enabled", "horizon", "candidates"}, "forecasting")
        return cls(
            enabled=_bool(data, "enabled", True, "forecasting"),
            horizon=_int(data, "horizon", 4, "forecasting"),
            candidates=_int(data, "candidates", 4, "forecasting"),
        )


@dataclass(frozen=True, slots=True)
class GraphConfig:
    """Sparse cross-modal evidence graph controls."""

    enabled: bool = True
    max_neighbors: int = 8
    min_edge_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.max_neighbors < 1:
            raise ConfigurationError("graph.max_neighbors must be >= 1")
        if not 0 <= self.min_edge_weight <= 1:
            raise ConfigurationError("graph.min_edge_weight must be in the interval [0, 1]")

    @classmethod
    def from_mapping(cls, value: object) -> GraphConfig:
        data = _mapping(value, "graph")
        _reject_unknown(data, {"enabled", "max_neighbors", "min_edge_weight"}, "graph")
        return cls(
            enabled=_bool(data, "enabled", True, "graph"),
            max_neighbors=_int(data, "max_neighbors", 8, "graph"),
            min_edge_weight=_float(data, "min_edge_weight", 0.0, "graph"),
        )


@dataclass(frozen=True, slots=True)
class SelectionConfig:
    """Budgeted submodular selection controls."""

    enabled: bool = True
    algorithm: str = "lazy_greedy"

    def __post_init__(self) -> None:
        if self.algorithm != "lazy_greedy":
            raise ConfigurationError("selection.algorithm must be 'lazy_greedy'")

    @classmethod
    def from_mapping(cls, value: object) -> SelectionConfig:
        data = _mapping(value, "selection")
        _reject_unknown(data, {"enabled", "algorithm"}, "selection")
        return cls(
            enabled=_bool(data, "enabled", True, "selection"),
            algorithm=_optional_str(data, "algorithm", "lazy_greedy", "selection"),
        )


@dataclass(frozen=True, slots=True)
class PrototypeConfig:
    """Prototype-tier controls."""

    enabled: bool = True
    group_size: int = 4

    def __post_init__(self) -> None:
        if self.group_size < 1:
            raise ConfigurationError("prototypes.group_size must be >= 1")

    @classmethod
    def from_mapping(cls, value: object) -> PrototypeConfig:
        data = _mapping(value, "prototypes")
        _reject_unknown(data, {"enabled", "group_size"}, "prototypes")
        return cls(
            enabled=_bool(data, "enabled", True, "prototypes"),
            group_size=_int(data, "group_size", 4, "prototypes"),
        )


@dataclass(frozen=True, slots=True)
class ResidualConfig:
    """Residual-tier controls."""

    enabled: bool = True
    rank: int = 8

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ConfigurationError("residual.rank must be >= 1")

    @classmethod
    def from_mapping(cls, value: object) -> ResidualConfig:
        data = _mapping(value, "residual")
        _reject_unknown(data, {"enabled", "rank"}, "residual")
        return cls(
            enabled=_bool(data, "enabled", True, "residual"),
            rank=_int(data, "rank", 8, "residual"),
        )


@dataclass(frozen=True, slots=True)
class RepairConfig:
    """Uncertainty-guided residual repair controls."""

    enabled: bool = True
    uncertainty_threshold: float = 0.5
    max_blocks_per_step: int = 2

    def __post_init__(self) -> None:
        if not 0 <= self.uncertainty_threshold <= 1:
            raise ConfigurationError("repair.uncertainty_threshold must be in the interval [0, 1]")
        if self.max_blocks_per_step < 0:
            raise ConfigurationError("repair.max_blocks_per_step must be >= 0")

    @classmethod
    def from_mapping(cls, value: object) -> RepairConfig:
        data = _mapping(value, "repair")
        allowed = {"enabled", "uncertainty_threshold", "max_blocks_per_step"}
        _reject_unknown(data, allowed, "repair")
        return cls(
            enabled=_bool(data, "enabled", True, "repair"),
            uncertainty_threshold=_float(data, "uncertainty_threshold", 0.5, "repair"),
            max_blocks_per_step=_int(data, "max_blocks_per_step", 2, "repair"),
        )


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Fully resolved MosaicKV run configuration."""

    model: ModelConfig
    dataset: DatasetConfig
    execution: ExecutionConfig
    generation: GenerationConfig
    cache: CacheConfig
    forecasting: ForecastingConfig = field(default_factory=ForecastingConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    prototypes: PrototypeConfig = field(default_factory=PrototypeConfig)
    residual: ResidualConfig = field(default_factory=ResidualConfig)
    repair: RepairConfig = field(default_factory=RepairConfig)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ConfigurationError("schema_version must equal 1")

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
            "forecasting",
            "graph",
            "selection",
            "prototypes",
            "residual",
            "repair",
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
            forecasting=ForecastingConfig.from_mapping(data.get("forecasting", {})),
            graph=GraphConfig.from_mapping(data.get("graph", {})),
            selection=SelectionConfig.from_mapping(data.get("selection", {})),
            prototypes=PrototypeConfig.from_mapping(data.get("prototypes", {})),
            residual=ResidualConfig.from_mapping(data.get("residual", {})),
            repair=RepairConfig.from_mapping(data.get("repair", {})),
            schema_version=_int(data, "schema_version", 1, "config"),
        )


def load_config(path: str | Path) -> RunConfig:
    """Load a strict JSON or TOML configuration without environment interpolation."""

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
        else:
            raise ConfigurationError(
                f"unsupported configuration format {config_path.suffix!r}; use .json or .toml"
            )
    except (json.JSONDecodeError, tomllib.TOMLDecodeError, UnicodeDecodeError) as error:
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
    "ModelConfig",
    "PrototypeConfig",
    "RepairConfig",
    "ResidualConfig",
    "RunConfig",
    "SelectionConfig",
    "canonical_config",
    "canonical_config_json",
    "config_sha256",
    "load_config",
    "synthetic_evaluation_config",
    "synthetic_smoke_config",
]
