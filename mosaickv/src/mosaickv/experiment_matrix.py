"""Versioned, fail-closed experiment-matrix expansion.

The matrix describes planned experiment axes.  Only combinations placed in a
``sweeps`` entry are runnable; ``blocked`` entries document deliberately
unrunnable parts of the requested research matrix.  Expansion always produces
one strict, immutable :class:`~mosaickv.config.RunConfig` per Slurm array task.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, cast

import yaml

from mosaickv.config import (
    CacheConfig,
    DatasetConfig,
    ExecutionConfig,
    ForecastingConfig,
    GenerationConfig,
    GraphConfig,
    LookMConfig,
    ModelConfig,
    PrefixKVConfig,
    PrototypeConfig,
    RepairConfig,
    ResidualConfig,
    RunConfig,
    SelectionConfig,
    UtilityConfig,
    VLCacheConfig,
    canonical_config_json,
    config_sha256,
    load_config,
)
from mosaickv.types import (
    Backend,
    BudgetUnit,
    ForecastMode,
    MosaicKVMethod,
    Precision,
    PrefixKVProfileMode,
    RepairPolicy,
)

MATRIX_SCHEMA_VERSION = 1
CAPABILITY_CATALOG_VERSION = "common-env-2026-07-21-v1"
ACCOUNTING_SPEC_SHA = "60915ac01ad35843c4de56620864145d5b81b973a3bfec03fb322d5e4f6ff695"

REQUIRED_MODELS = frozenset(
    {
        "llava_1_5_7b",
        "qwen2_5_vl_3b",
        "qwen2_5_vl_7b",
        "llava_onevision_7b_optional",
    }
)
REQUIRED_BACKENDS = frozenset({"hf_eager", "hf_sdpa", "hf_flash_attention_2", "vllm", "sglang"})
REQUIRED_METHODS = frozenset(
    {
        "full_kv",
        "random_kv",
        "prompt_attention_topk",
        "lookm_reimpl",
        "prefixkv_reimpl",
        "vl_cache_reimpl",
        "mosaickv_exact",
        "mosaickv_proto",
        "mosaickv_full",
    }
)
REQUIRED_TASKS = frozenset({"mmstar", "mmvet", "textvqa", "docvqa", "chartqa", "videomme"})
REQUIRED_RETENTION_RATIOS = frozenset({1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05})
REQUIRED_OUTPUT_LENGTHS = frozenset({32, 64, 128})


class ExperimentMatrixError(ValueError):
    """Raised when a matrix cannot be expanded into canonical jobs."""


@dataclass(frozen=True, slots=True)
class ModelCapability:
    key: str
    model_id: str
    revision: str
    video: bool
    hf_eager_adapter: bool
    note: str


@dataclass(frozen=True, slots=True)
class BackendCapability:
    key: str
    backend: Backend
    attention_implementation: str
    runnable: bool
    reason: str


@dataclass(frozen=True, slots=True)
class TaskCapability:
    key: str
    task_name: str
    dataset_id: str
    revision: str
    split: str
    video: bool


MODEL_CATALOG: dict[str, ModelCapability] = {
    "llava_1_5_7b": ModelCapability(
        "llava_1_5_7b",
        "llava-hf/llava-1.5-7b-hf",
        "b234b804b114d9e37bb655e11cbbb5f5e971b7a9",
        False,
        True,
        "registered eager HF adapter; image-only",
    ),
    "qwen2_5_vl_3b": ModelCapability(
        "qwen2_5_vl_3b",
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "66285546d2b821cf421d4f5eb2576359d3770cd3",
        True,
        True,
        "registered eager HF adapter",
    ),
    "qwen2_5_vl_7b": ModelCapability(
        "qwen2_5_vl_7b",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "cc594898137f460bfe9f0759e9844b3ce807cfb5",
        True,
        True,
        "registered eager HF adapter",
    ),
    "llava_onevision_7b_optional": ModelCapability(
        "llava_onevision_7b_optional",
        "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        "0d50680527681998e456c7b78950205bedd8a068",
        True,
        False,
        "optional checkpoint is pinned but no exact 7B runtime adapter is registered",
    ),
}

BACKEND_CATALOG: dict[str, BackendCapability] = {
    "hf_eager": BackendCapability(
        "hf_eager", Backend.HUGGINGFACE, "eager", True, "correctness-first HF path"
    ),
    "hf_sdpa": BackendCapability(
        "hf_sdpa",
        Backend.HUGGINGFACE,
        "sdpa",
        False,
        "HF adapters explicitly accept eager attention only",
    ),
    "hf_flash_attention_2": BackendCapability(
        "hf_flash_attention_2",
        Backend.HUGGINGFACE,
        "flash_attention_2",
        False,
        "FlashAttention-2 is not correctness-gated by the HF adapters",
    ),
    "vllm": BackendCapability(
        "vllm",
        Backend.VLLM,
        "eager",
        False,
        "common-lock Qwen2.5-VL model loading failed and native cache mutation is unavailable",
    ),
    "sglang": BackendCapability(
        "sglang",
        Backend.SGLANG,
        "triton",
        False,
        "common-lock model-serving parity and native cache mutation are not validated",
    ),
}

TASK_CATALOG: dict[str, TaskCapability] = {
    "mmstar": TaskCapability(
        "mmstar",
        "mmstar",
        "Lin-Chen/MMStar",
        "bc98d668301da7b14f648724866e57302778ab27",
        "test",
        False,
    ),
    "mmvet": TaskCapability(
        "mmvet",
        "mmvet",
        "lmms-lab/MMVet",
        "b310d12d0d9e765db953da28a7ef7ff43620dc83",
        "test",
        False,
    ),
    "textvqa": TaskCapability(
        "textvqa",
        "textvqa",
        "lmms-lab/textvqa",
        "9c0699cd19768ac5ab97568f6b3cbac4c0062884",
        "validation",
        False,
    ),
    "docvqa": TaskCapability(
        "docvqa",
        "docvqa",
        "lmms-lab/DocVQA",
        "539088ef8a8ada01ac8e2e6d4e372586748a265e",
        "validation",
        False,
    ),
    "chartqa": TaskCapability(
        "chartqa",
        "chartqa",
        "lmms-lab/ChartQA",
        "9e63b7df1592a1c2158e735cc1725454aef0d6d9",
        "test",
        False,
    ),
    "videomme": TaskCapability(
        "videomme",
        "video_mme",
        "lmms-lab/Video-MME",
        "ead1408f75b618502df9a1d8e0950166bf0a2a0b",
        "test",
        True,
    ),
}

_PUBLISHED_REIMPLEMENTATIONS = frozenset({"lookm_reimpl", "prefixkv_reimpl", "vl_cache_reimpl"})
_PROTOTYPE_METHODS = frozenset({"mosaickv_proto", "mosaickv_full"})
_OVERRIDABLE_SECTIONS = frozenset(
    {"forecasting", "graph", "utility", "selection", "prototypes", "residual", "repair"}
)
_SCOPE_DIMENSIONS = frozenset({"models", "backends", "methods", "tasks"})


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ExperimentMatrixError(f"{path} must be an object with string keys")
    return cast("Mapping[str, object]", value)


def _reject_unknown(data: Mapping[str, object], allowed: set[str], path: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ExperimentMatrixError(f"{path} contains unknown field(s): {', '.join(unknown)}")


def _required_string(data: Mapping[str, object], key: str, path: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ExperimentMatrixError(f"{path}.{key} must be a non-empty string")
    return value


def _string_tuple(value: object, path: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ExperimentMatrixError(f"{path} must be a non-empty string array")
    result = tuple(value)
    if not result or not all(isinstance(item, str) and item.strip() for item in result):
        raise ExperimentMatrixError(f"{path} must be a non-empty string array")
    strings = cast("tuple[str, ...]", result)
    if len(set(strings)) != len(strings):
        raise ExperimentMatrixError(f"{path} must not contain duplicates")
    return strings


def _int_tuple(value: object, path: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ExperimentMatrixError(f"{path} must be a non-empty integer array")
    result = tuple(value)
    if not result or any(isinstance(item, bool) or not isinstance(item, int) for item in result):
        raise ExperimentMatrixError(f"{path} must be a non-empty integer array")
    integers = cast("tuple[int, ...]", result)
    if len(set(integers)) != len(integers):
        raise ExperimentMatrixError(f"{path} must not contain duplicates")
    return integers


def _float_tuple(value: object, path: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ExperimentMatrixError(f"{path} must be a non-empty numeric array")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ExperimentMatrixError(f"{path} must be a non-empty numeric array")
        result.append(float(item))
    if not result or len(set(result)) != len(result):
        raise ExperimentMatrixError(f"{path} must be non-empty and contain no duplicates")
    return tuple(result)


def _positive_int(data: Mapping[str, object], key: str, path: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ExperimentMatrixError(f"{path}.{key} must be an integer >= 1")
    return value


def _optional_positive_int(value: object, path: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ExperimentMatrixError(f"{path} must be null or an integer >= 1")
    return value


def _bool(data: Mapping[str, object], key: str, path: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise ExperimentMatrixError(f"{path}.{key} must be boolean")
    return value


@dataclass(frozen=True, slots=True)
class AxisSpec:
    models: tuple[str, ...]
    backends: tuple[str, ...]
    methods: tuple[str, ...]
    tasks: tuple[str, ...]
    retention_ratios: tuple[float, ...]
    output_lengths: tuple[int, ...]
    selection_seeds: tuple[int, ...]

    @classmethod
    def from_mapping(cls, value: object) -> AxisSpec:
        data = _mapping(value, "axes")
        allowed = {
            "models",
            "backends",
            "methods",
            "tasks",
            "retention_ratios",
            "output_lengths",
            "selection_seeds",
        }
        _reject_unknown(data, allowed, "axes")
        axes = cls(
            models=_string_tuple(data.get("models"), "axes.models"),
            backends=_string_tuple(data.get("backends"), "axes.backends"),
            methods=_string_tuple(data.get("methods"), "axes.methods"),
            tasks=_string_tuple(data.get("tasks"), "axes.tasks"),
            retention_ratios=_float_tuple(data.get("retention_ratios"), "axes.retention_ratios"),
            output_lengths=_int_tuple(data.get("output_lengths"), "axes.output_lengths"),
            selection_seeds=_int_tuple(data.get("selection_seeds"), "axes.selection_seeds"),
        )
        axes.validate()
        return axes

    def validate(self) -> None:
        requirements: tuple[tuple[str, frozenset[Any], frozenset[Any]], ...] = (
            ("models", frozenset(self.models), REQUIRED_MODELS),
            ("backends", frozenset(self.backends), REQUIRED_BACKENDS),
            ("methods", frozenset(self.methods), REQUIRED_METHODS),
            ("tasks", frozenset(self.tasks), REQUIRED_TASKS),
            ("retention_ratios", frozenset(self.retention_ratios), REQUIRED_RETENTION_RATIOS),
            ("output_lengths", frozenset(self.output_lengths), REQUIRED_OUTPUT_LENGTHS),
        )
        for name, actual, expected in requirements:
            if actual != expected:
                missing = sorted(expected - actual)
                extra = sorted(actual - expected)
                raise ExperimentMatrixError(
                    f"axes.{name} must equal the version-1 vocabulary; "
                    f"missing={missing}, extra={extra}"
                )
        if len(self.selection_seeds) != 3 or any(seed < 0 for seed in self.selection_seeds):
            raise ExperimentMatrixError(
                "axes.selection_seeds must contain exactly three distinct nonnegative seeds"
            )


@dataclass(frozen=True, slots=True)
class GenerationPolicy:
    deterministic: bool
    do_sample: bool
    temperature: float
    top_p: float
    output_length_policy: str

    @classmethod
    def from_mapping(cls, value: object) -> GenerationPolicy:
        data = _mapping(value, "generation")
        allowed = {
            "deterministic",
            "do_sample",
            "temperature",
            "top_p",
            "output_length_policy",
        }
        _reject_unknown(data, allowed, "generation")
        temperature = data.get("temperature")
        top_p = data.get("top_p")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise ExperimentMatrixError("generation.temperature must be numeric")
        if isinstance(top_p, bool) or not isinstance(top_p, (int, float)):
            raise ExperimentMatrixError("generation.top_p must be numeric")
        policy = cls(
            deterministic=_bool(data, "deterministic", "generation"),
            do_sample=_bool(data, "do_sample", "generation"),
            temperature=float(temperature),
            top_p=float(top_p),
            output_length_policy=_required_string(data, "output_length_policy", "generation"),
        )
        if (
            not policy.deterministic
            or policy.do_sample
            or policy.temperature != 0.0
            or policy.top_p != 1.0
            or policy.output_length_policy != "fixed_max_new_tokens"
        ):
            raise ExperimentMatrixError(
                "generation must be deterministic greedy decoding with fixed output length"
            )
        return policy


@dataclass(frozen=True, slots=True)
class PerformancePolicy:
    warmups: int
    timed_repetitions: int

    @classmethod
    def from_mapping(cls, value: object) -> PerformancePolicy:
        data = _mapping(value, "performance")
        _reject_unknown(data, {"warmups", "timed_repetitions"}, "performance")
        policy = cls(
            warmups=_positive_int(data, "warmups", "performance"),
            timed_repetitions=_positive_int(data, "timed_repetitions", "performance"),
        )
        if policy.warmups < 5:
            raise ExperimentMatrixError("performance.warmups must be >= 5")
        if policy.timed_repetitions < 20:
            raise ExperimentMatrixError("performance.timed_repetitions must be >= 20")
        return policy


@dataclass(frozen=True, slots=True)
class ComparisonPolicy:
    primary_budget: str
    nominal_retention_only_for_scheduling: bool
    require_realized_byte_match_before_aggregation: bool

    @classmethod
    def from_mapping(cls, value: object) -> ComparisonPolicy:
        data = _mapping(value, "comparison")
        allowed = {
            "primary_budget",
            "nominal_retention_only_for_scheduling",
            "require_realized_byte_match_before_aggregation",
        }
        _reject_unknown(data, allowed, "comparison")
        policy = cls(
            primary_budget=_required_string(data, "primary_budget", "comparison"),
            nominal_retention_only_for_scheduling=_bool(
                data, "nominal_retention_only_for_scheduling", "comparison"
            ),
            require_realized_byte_match_before_aggregation=_bool(
                data, "require_realized_byte_match_before_aggregation", "comparison"
            ),
        )
        if (
            policy.primary_budget != "active_kv_bytes"
            or not policy.nominal_retention_only_for_scheduling
            or not policy.require_realized_byte_match_before_aggregation
        ):
            raise ExperimentMatrixError(
                "comparison must use realized active_kv_bytes and require byte matching"
            )
        return policy


@dataclass(frozen=True, slots=True)
class RunPolicy:
    output_root: str
    cache_root: str
    subset_size: int | None
    measurement_mode: str
    resume: bool

    @classmethod
    def from_mapping(cls, value: object) -> RunPolicy:
        data = _mapping(value, "run")
        allowed = {"output_root", "cache_root", "subset_size", "measurement_mode", "resume"}
        _reject_unknown(data, allowed, "run")
        policy = cls(
            output_root=_required_string(data, "output_root", "run"),
            cache_root=_required_string(data, "cache_root", "run"),
            subset_size=_optional_positive_int(data.get("subset_size"), "run.subset_size"),
            measurement_mode=_required_string(data, "measurement_mode", "run"),
            resume=_bool(data, "resume", "run"),
        )
        for field_name, raw_path in (
            ("output_root", policy.output_root),
            ("cache_root", policy.cache_root),
        ):
            path = Path(raw_path)
            if not path.is_absolute():
                raise ExperimentMatrixError(f"run.{field_name} must be absolute")
            home = Path.home().resolve()
            resolved = path.resolve()
            if resolved == home or home in resolved.parents:
                raise ExperimentMatrixError(f"run.{field_name} must be outside the home directory")
        if policy.cache_root != "/scratch/djy8hg/cache/mosaickv":
            raise ExperimentMatrixError(
                "run.cache_root must use the shared /scratch/djy8hg/cache/mosaickv policy"
            )
        if not policy.resume:
            raise ExperimentMatrixError("run.resume must be true")
        return policy


@dataclass(frozen=True, slots=True)
class SweepSpec:
    name: str
    models: tuple[str, ...]
    backends: tuple[str, ...]
    methods: tuple[str, ...]
    tasks: tuple[str, ...]
    retention_ratios: tuple[float, ...]
    output_lengths: tuple[int, ...]
    selection_seeds: tuple[int, ...]
    subset_size: int | None
    variant: str
    overrides: dict[str, Any]

    @classmethod
    def from_mapping(cls, value: object, index: int) -> SweepSpec:
        path = f"sweeps[{index}]"
        data = _mapping(value, path)
        allowed = {
            "name",
            "models",
            "backends",
            "methods",
            "tasks",
            "retention_ratios",
            "output_lengths",
            "selection_seeds",
            "subset_size",
            "variant",
            "overrides",
        }
        _reject_unknown(data, allowed, path)
        name = _required_string(data, "name", path)
        variant_raw = data.get("variant", name)
        if not isinstance(variant_raw, str) or not variant_raw.strip():
            raise ExperimentMatrixError(f"{path}.variant must be a non-empty string")
        overrides_raw = data.get("overrides", {})
        overrides = dict(_mapping(overrides_raw, f"{path}.overrides"))
        unknown_sections = sorted(set(overrides) - _OVERRIDABLE_SECTIONS)
        if unknown_sections:
            raise ExperimentMatrixError(
                f"{path}.overrides may not change run identity sections: {unknown_sections}"
            )
        return cls(
            name=name,
            models=_string_tuple(data.get("models"), f"{path}.models"),
            backends=_string_tuple(data.get("backends"), f"{path}.backends"),
            methods=_string_tuple(data.get("methods"), f"{path}.methods"),
            tasks=_string_tuple(data.get("tasks"), f"{path}.tasks"),
            retention_ratios=_float_tuple(data.get("retention_ratios"), f"{path}.retention_ratios"),
            output_lengths=_int_tuple(data.get("output_lengths"), f"{path}.output_lengths"),
            selection_seeds=_int_tuple(data.get("selection_seeds"), f"{path}.selection_seeds"),
            subset_size=_optional_positive_int(data.get("subset_size"), f"{path}.subset_size"),
            variant=variant_raw,
            overrides=overrides,
        )


@dataclass(frozen=True, slots=True)
class BlockedSpec:
    scope: dict[str, tuple[str, ...]]
    reason: str

    @classmethod
    def from_mapping(cls, value: object, index: int) -> BlockedSpec:
        path = f"blocked[{index}]"
        data = _mapping(value, path)
        _reject_unknown(data, {"scope", "reason"}, path)
        scope_data = _mapping(data.get("scope"), f"{path}.scope")
        _reject_unknown(scope_data, set(_SCOPE_DIMENSIONS), f"{path}.scope")
        if not scope_data:
            raise ExperimentMatrixError(f"{path}.scope must not be empty")
        return cls(
            scope={
                key: _string_tuple(item, f"{path}.scope.{key}") for key, item in scope_data.items()
            },
            reason=_required_string(data, "reason", path),
        )


@dataclass(frozen=True, slots=True)
class ExperimentMatrix:
    matrix_schema_version: int
    matrix_revision: int
    capability_catalog_version: str
    experiment_id: str
    description: str
    enabled: bool
    axes: AxisSpec
    generation: GenerationPolicy
    performance: PerformancePolicy
    comparison: ComparisonPolicy
    run: RunPolicy
    sweeps: tuple[SweepSpec, ...]
    blocked: tuple[BlockedSpec, ...]

    @classmethod
    def from_mapping(cls, value: object) -> ExperimentMatrix:
        data = _mapping(value, "matrix")
        allowed = {
            "matrix_schema_version",
            "matrix_revision",
            "capability_catalog_version",
            "experiment_id",
            "description",
            "enabled",
            "axes",
            "generation",
            "performance",
            "comparison",
            "run",
            "sweeps",
            "blocked",
        }
        _reject_unknown(data, allowed, "matrix")
        schema = data.get("matrix_schema_version")
        revision = data.get("matrix_revision")
        if isinstance(schema, bool) or not isinstance(schema, int):
            raise ExperimentMatrixError("matrix_schema_version must be an integer")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise ExperimentMatrixError("matrix_revision must be an integer >= 1")
        sweeps_raw = data.get("sweeps")
        blocked_raw = data.get("blocked")
        if not isinstance(sweeps_raw, Sequence) or isinstance(sweeps_raw, (str, bytes)):
            raise ExperimentMatrixError("sweeps must be an array")
        if not isinstance(blocked_raw, Sequence) or isinstance(blocked_raw, (str, bytes)):
            raise ExperimentMatrixError("blocked must be an array")
        matrix = cls(
            matrix_schema_version=schema,
            matrix_revision=revision,
            capability_catalog_version=_required_string(
                data, "capability_catalog_version", "matrix"
            ),
            experiment_id=_required_string(data, "experiment_id", "matrix"),
            description=_required_string(data, "description", "matrix"),
            enabled=_bool(data, "enabled", "matrix"),
            axes=AxisSpec.from_mapping(data.get("axes")),
            generation=GenerationPolicy.from_mapping(data.get("generation")),
            performance=PerformancePolicy.from_mapping(data.get("performance")),
            comparison=ComparisonPolicy.from_mapping(data.get("comparison")),
            run=RunPolicy.from_mapping(data.get("run")),
            sweeps=tuple(
                SweepSpec.from_mapping(item, index) for index, item in enumerate(sweeps_raw)
            ),
            blocked=tuple(
                BlockedSpec.from_mapping(item, index) for index, item in enumerate(blocked_raw)
            ),
        )
        matrix.validate()
        return matrix

    def validate(self) -> None:
        if self.matrix_schema_version != MATRIX_SCHEMA_VERSION:
            raise ExperimentMatrixError(f"matrix_schema_version must equal {MATRIX_SCHEMA_VERSION}")
        if self.capability_catalog_version != CAPABILITY_CATALOG_VERSION:
            raise ExperimentMatrixError(
                "capability_catalog_version does not match the installed matrix validator"
            )
        if not self.experiment_id.replace("_", "").replace("-", "").isalnum():
            raise ExperimentMatrixError(
                "experiment_id may contain only letters, numbers, underscores, and hyphens"
            )
        if self.enabled and not self.sweeps:
            raise ExperimentMatrixError("an enabled matrix must contain at least one sweep")
        if not self.enabled and self.sweeps:
            raise ExperimentMatrixError("a disabled matrix must not contain runnable sweeps")
        if not self.blocked:
            raise ExperimentMatrixError("blocked must document at least one excluded scope")
        sweep_names = [sweep.name for sweep in self.sweeps]
        if len(sweep_names) != len(set(sweep_names)):
            raise ExperimentMatrixError("sweep names must be unique")
        root_axes: dict[str, set[Any]] = {
            "models": set(self.axes.models),
            "backends": set(self.axes.backends),
            "methods": set(self.axes.methods),
            "tasks": set(self.axes.tasks),
            "retention_ratios": set(self.axes.retention_ratios),
            "output_lengths": set(self.axes.output_lengths),
            "selection_seeds": set(self.axes.selection_seeds),
        }
        for sweep in self.sweeps:
            for field_name in root_axes:
                values = set(cast("Sequence[Any]", getattr(sweep, field_name)))
                unknown = sorted(values - root_axes[field_name])
                if unknown:
                    raise ExperimentMatrixError(
                        f"sweep {sweep.name!r} references values outside axes.{field_name}: "
                        f"{unknown}"
                    )
        for blocked in self.blocked:
            for field_name, scope_values in blocked.scope.items():
                unknown = sorted(
                    set(scope_values) - cast("set[str]", root_axes[field_name]) - {"*"}
                )
                if unknown:
                    raise ExperimentMatrixError(
                        f"blocked scope references unknown axes.{field_name}: {unknown}"
                    )

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class ExpandedRun:
    config: RunConfig
    sweep_name: str
    variant: str
    model_key: str
    backend_key: str
    task_key: str
    task_name: str
    selection_seed: int
    subset_size: int | None


def load_experiment_matrix(path: str | Path) -> ExperimentMatrix:
    matrix_path = Path(path)
    try:
        payload = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ExperimentMatrixError(
            f"cannot load experiment matrix {matrix_path}: {error}"
        ) from error
    return ExperimentMatrix.from_mapping(payload)


def unsupported_reason(
    model_key: str,
    backend_key: str,
    method_name: str,
    task_key: str,
    retention_ratio: float,
) -> str | None:
    """Return the first fail-closed capability violation for one planned run."""

    model = MODEL_CATALOG[model_key]
    backend = BACKEND_CATALOG[backend_key]
    task = TASK_CATALOG[task_key]
    if not model.hf_eager_adapter:
        return model.note
    if not backend.runnable:
        return backend.reason
    if backend_key != "hf_eager":
        return "the current method implementations own only the HF eager cache interface"
    if method_name in _PROTOTYPE_METHODS:
        return "all registered HF adapters currently disable prototype merge and residual repair"
    if method_name == "prefixkv_reimpl":
        return (
            "paper-result PrefixKV requires a pinned offline layer profile and disjoint "
            "calibration IDs; this matrix provides neither artifact"
        )
    if method_name == "vl_cache_reimpl":
        return (
            "VL-Cache requires pinned disjoint calibration provenance and a checkpoint-specific "
            "retention-1 parity gate before quality rows"
        )
    if method_name in _PUBLISHED_REIMPLEMENTATIONS and model_key != "llava_1_5_7b":
        return (
            f"{method_name} is restricted to the audited LLaVA-1.5 experiment matrix; "
            "generalized model rows require a separate result label"
        )
    if task.video and not model.video:
        return f"{model.model_id} does not support video input"
    if method_name == "full_kv" and retention_ratio != 1.0:
        return "full_kv requires retention_ratio=1.0"
    return None


def _base_config(
    *,
    model_key: str,
    backend_key: str,
    method_name: str,
    task_key: str,
    retention_ratio: float,
    output_length: int,
    seed: int,
) -> RunConfig:
    model = MODEL_CATALOG[model_key]
    backend = BACKEND_CATALOG[backend_key]
    task = TASK_CATALOG[task_key]
    method = MosaicKVMethod(method_name)
    is_mosaickv = method.is_mosaickv
    prototype_enabled = (
        method is MosaicKVMethod.MOSAICKV_PROTO or method is MosaicKVMethod.MOSAICKV_FULL
    )
    residual_enabled = method is MosaicKVMethod.MOSAICKV_FULL
    repair_enabled = residual_enabled
    paper_reimplementation = method.is_published_reimplementation
    block_size = 1 if paper_reimplementation else 16
    forecasting = (
        ForecastingConfig(
            enabled=True,
            mode=ForecastMode.HYBRID,
            prompt_window=16,
            draft_steps=4,
            centroid_count=4,
        )
        if is_mosaickv
        else ForecastingConfig(
            enabled=False,
            mode=ForecastMode.PROMPT_WINDOW,
            prompt_window=16,
            draft_steps=0,
            centroid_count=4,
        )
    )
    return RunConfig(
        model=ModelConfig(model.model_id, model.revision, Precision.BF16),
        dataset=DatasetConfig(task.dataset_id, task.revision, task.split),
        execution=ExecutionConfig(backend.backend, backend.attention_implementation, seed, True),
        generation=GenerationConfig(max_new_tokens=output_length),
        cache=CacheConfig(
            budget_value=2_147_483_647,
            budget_unit=BudgetUnit.BLOCKS,
            retention_ratio=retention_ratio,
            block_size=block_size,
            accounting_spec_sha=ACCOUNTING_SPEC_SHA,
        ),
        method=method,
        forecasting=forecasting,
        graph=GraphConfig(enabled=is_mosaickv),
        utility=UtilityConfig(),
        selection=SelectionConfig(enabled=is_mosaickv),
        prototypes=PrototypeConfig(enabled=prototype_enabled),
        residual=ResidualConfig(
            enabled=residual_enabled,
            require_pinned_memory=residual_enabled,
        ),
        repair=RepairConfig(
            enabled=repair_enabled,
            policy=(
                RepairPolicy.ENTROPY_OR_PROTOTYPE_RISK if repair_enabled else RepairPolicy.NONE
            ),
            max_blocks_per_step=2 if repair_enabled else 0,
        ),
        lookm=LookMConfig(
            enabled=method.is_lookm_reimplementation,
            recent_ratio=retention_ratio / 2 if method.is_lookm_reimplementation else 0.1,
            important_ratio=retention_ratio / 2 if method.is_lookm_reimplementation else 0.1,
        ),
        prefixkv=PrefixKVConfig(
            enabled=method.is_prefixkv_reimplementation,
            profile_mode=PrefixKVProfileMode.FIXED_GLOBAL,
        ),
        vl_cache=VLCacheConfig(enabled=method.is_vl_cache_reimplementation),
    )


def _deep_override(target: dict[str, Any], changes: Mapping[str, object], path: str) -> None:
    for key, value in changes.items():
        if key not in target:
            raise ExperimentMatrixError(f"{path} references unknown config field {key!r}")
        if isinstance(value, Mapping):
            current = target[key]
            if not isinstance(current, dict):
                raise ExperimentMatrixError(f"{path}.{key} cannot be merged into a scalar")
            _deep_override(current, cast("Mapping[str, object]", value), f"{path}.{key}")
        else:
            target[key] = value


def _apply_overrides(config: RunConfig, overrides: Mapping[str, object]) -> RunConfig:
    if not overrides:
        return config
    payload = cast("dict[str, Any]", json.loads(canonical_config_json(config)))
    _deep_override(payload, overrides, "overrides")
    return RunConfig.from_mapping(payload)


def expand_experiment_matrix(matrix: ExperimentMatrix) -> tuple[ExpandedRun, ...]:
    """Validate capabilities and deterministically expand every runnable sweep."""

    if not matrix.enabled:
        return ()
    expanded: list[ExpandedRun] = []
    seen_configs: dict[str, str] = {}
    for sweep in matrix.sweeps:
        subset_size = sweep.subset_size if sweep.subset_size is not None else matrix.run.subset_size
        combinations = product(
            sweep.models,
            sweep.backends,
            sweep.methods,
            sweep.tasks,
            sweep.retention_ratios,
            sweep.output_lengths,
            sweep.selection_seeds,
        )
        for model_key, backend_key, method_name, task_key, ratio, length, seed in combinations:
            reason = unsupported_reason(model_key, backend_key, method_name, task_key, ratio)
            identity = (
                f"model={model_key}, backend={backend_key}, method={method_name}, "
                f"task={task_key}, retention={ratio}, output_length={length}, seed={seed}"
            )
            if reason is not None:
                raise ExperimentMatrixError(
                    f"unsupported combination in sweep {sweep.name!r}: {identity}: {reason}"
                )
            config = _apply_overrides(
                _base_config(
                    model_key=model_key,
                    backend_key=backend_key,
                    method_name=method_name,
                    task_key=task_key,
                    retention_ratio=ratio,
                    output_length=length,
                    seed=seed,
                ),
                sweep.overrides,
            )
            digest = config_sha256(config)
            if digest in seen_configs:
                raise ExperimentMatrixError(
                    f"duplicate resolved config in sweep {sweep.name!r}: {identity}; "
                    f"first emitted by {seen_configs[digest]!r}"
                )
            seen_configs[digest] = sweep.name
            expanded.append(
                ExpandedRun(
                    config=config,
                    sweep_name=sweep.name,
                    variant=sweep.variant,
                    model_key=model_key,
                    backend_key=backend_key,
                    task_key=task_key,
                    task_name=TASK_CATALOG[task_key].task_name,
                    selection_seed=seed,
                    subset_size=subset_size,
                )
            )
    return tuple(expanded)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_once(path: Path, payload: bytes, *, mode: int = 0o444) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable matrix artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
        raise


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _config_yaml_bytes(config: RunConfig) -> bytes:
    payload = json.loads(canonical_config_json(config))
    rendered = cast("str", yaml.safe_dump(payload, sort_keys=True))
    return rendered.encode()


def _default_expansion_root(matrix: ExperimentMatrix) -> Path:
    return Path(f"/scratch/djy8hg/runs/mosaickv/matrices/{matrix.experiment_id}/{matrix.sha256}")


def materialize_experiment_matrix(
    matrix_path: str | Path,
    *,
    output_directory: str | Path | None = None,
    resume: bool = False,
) -> Path:
    """Write an immutable config directory and return its JSONL array index."""

    source = Path(matrix_path).resolve()
    matrix = load_experiment_matrix(source)
    runs = expand_experiment_matrix(matrix)
    destination = (
        Path(output_directory).resolve()
        if output_directory is not None
        else _default_expansion_root(matrix).resolve()
    )
    manifest_path = destination / "matrix_manifest.json"
    index_path = destination / "jobs.jsonl"
    if destination.exists():
        if not resume:
            raise FileExistsError(
                f"matrix expansion already exists; pass --resume to verify it: {destination}"
            )
        if not manifest_path.is_file() or not index_path.is_file():
            raise ExperimentMatrixError("existing expansion is incomplete and cannot be resumed")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("matrix_sha") != matrix.sha256:
            raise ExperimentMatrixError("existing expansion was produced by a different matrix")
        verify_expanded_index(index_path)
        return index_path
    destination.mkdir(parents=True, exist_ok=False)
    source_bytes = source.read_bytes()
    _write_once(destination / "source_matrix.yaml", source_bytes)
    rows: list[dict[str, Any]] = []
    for array_index, run in enumerate(runs):
        digest = config_sha256(run.config)
        config_path = destination / "configs" / f"{array_index:06d}-{digest}.yaml"
        _write_once(config_path, _config_yaml_bytes(run.config))
        run_id = f"{matrix.experiment_id}-{matrix.sha256[:10]}-{array_index:06d}-{digest[:12]}"
        rows.append(
            {
                "array_index": array_index,
                "backend": run.backend_key,
                "command": "evaluate",
                "config_path": str(config_path.resolve()),
                "config_sha": digest,
                "experiment_id": matrix.experiment_id,
                "matrix_revision": matrix.matrix_revision,
                "matrix_sha": matrix.sha256,
                "measurement_mode": matrix.run.measurement_mode,
                "method": run.config.method.value,
                "model": run.config.model.id,
                "model_key": run.model_key,
                "output_root": matrix.run.output_root,
                "retention_ratio": run.config.cache.retention_ratio,
                "run_id": run_id,
                "selection_seed": run.selection_seed,
                "subset_size": run.subset_size,
                "sweep": run.sweep_name,
                "task": run.task_name,
                "task_key": run.task_key,
                "timed_repetitions": matrix.performance.timed_repetitions,
                "trace_directory": str(Path(matrix.run.output_root) / "traces"),
                "variant": run.variant,
                "warmups": matrix.performance.warmups,
            }
        )
    index_payload = b"".join(_json_bytes(row) for row in rows)
    _write_once(index_path, index_payload)
    manifest = {
        "capability_catalog_version": matrix.capability_catalog_version,
        "experiment_id": matrix.experiment_id,
        "index_path": str(index_path.resolve()),
        "index_sha256": _sha256_bytes(index_payload),
        "job_count": len(rows),
        "matrix_revision": matrix.matrix_revision,
        "matrix_schema_version": matrix.matrix_schema_version,
        "matrix_sha": matrix.sha256,
        "source_matrix_sha256": _sha256_bytes(source_bytes),
    }
    _write_once(manifest_path, _json_bytes(manifest))
    verify_expanded_index(index_path)
    return index_path


def _validate_expanded_row(row: Mapping[str, Any], expected_index: int) -> tuple[str, Path]:
    if row.get("array_index") != expected_index:
        raise ExperimentMatrixError("array indices must be contiguous and zero-based")
    run_id = str(row.get("run_id", ""))
    if not run_id:
        raise ExperimentMatrixError("run IDs must be non-empty")
    config_path = Path(str(row.get("config_path", ""))).resolve()
    if not config_path.is_file():
        raise ExperimentMatrixError(f"expanded config does not exist: {config_path}")
    permissions = stat.S_IMODE(config_path.stat().st_mode)
    if permissions & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise ExperimentMatrixError(f"expanded config is writable: {config_path}")
    config = load_config(config_path)
    digest = config_sha256(config)
    if row.get("config_sha") != digest or digest not in config_path.name:
        raise ExperimentMatrixError(f"config SHA mismatch: {config_path}")
    if row.get("method") != config.method.value or row.get("model") != config.model.id:
        raise ExperimentMatrixError(f"index/config identity mismatch: {config_path}")
    backend_key = str(row.get("backend", ""))
    if backend_key not in BACKEND_CATALOG:
        raise ExperimentMatrixError(f"index contains unknown backend {backend_key!r}")
    backend = BACKEND_CATALOG[backend_key]
    if (
        config.execution.backend is not backend.backend
        or config.execution.attention_implementation != backend.attention_implementation
    ):
        raise ExperimentMatrixError(f"index/config backend mismatch: {config_path}")
    model_key = str(row.get("model_key", ""))
    task_key = str(row.get("task_key", ""))
    if model_key not in MODEL_CATALOG or task_key not in TASK_CATALOG:
        raise ExperimentMatrixError("index contains an unknown model or task key")
    model = MODEL_CATALOG[model_key]
    task = TASK_CATALOG[task_key]
    if config.model.id != model.model_id or config.model.revision != model.revision:
        raise ExperimentMatrixError(f"index/config model revision mismatch: {config_path}")
    if (
        config.dataset.id != task.dataset_id
        or config.dataset.revision != task.revision
        or config.dataset.split != task.split
        or row.get("task") != task.task_name
    ):
        raise ExperimentMatrixError(f"index/config dataset revision mismatch: {config_path}")
    reason = unsupported_reason(
        model_key,
        backend_key,
        config.method.value,
        task_key,
        config.cache.retention_ratio,
    )
    if reason is not None:
        raise ExperimentMatrixError(f"expanded index contains unsupported job: {reason}")
    if int(row.get("warmups", 0)) < 5 or int(row.get("timed_repetitions", 0)) < 20:
        raise ExperimentMatrixError("expanded performance controls violate the minimum")
    output_root = Path(str(row.get("output_root", ""))).resolve()
    home = Path.home().resolve()
    if output_root == home or home in output_root.parents:
        raise ExperimentMatrixError("expanded output root must be outside home")
    return run_id, config_path


def verify_expanded_index(index_path: str | Path, *, array_index: int | None = None) -> int:
    """Verify immutable configs and array metadata before Slurm submission.

    The pre-submission path validates every config.  An array worker may pass
    its index to validate only that config after the manifest has authenticated
    the complete JSONL index.
    """

    path = Path(index_path).resolve()
    if not path.is_file():
        raise ExperimentMatrixError(f"expanded index does not exist: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ExperimentMatrixError(f"invalid index JSON at line {line_number}") from error
        if not isinstance(row, dict):
            raise ExperimentMatrixError(f"index line {line_number} must be an object")
        rows.append(row)
    manifest_path = path.parent / "matrix_manifest.json"
    if array_index is not None and not manifest_path.is_file():
        raise ExperimentMatrixError("array-task validation requires matrix_manifest.json")
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("index_sha256") != _sha256_bytes(path.read_bytes()):
            raise ExperimentMatrixError("expanded index SHA does not match its manifest")
        if manifest.get("job_count") != len(rows):
            raise ExperimentMatrixError("expanded index count does not match its manifest")
    if array_index is not None:
        if not 0 <= array_index < len(rows):
            raise ExperimentMatrixError(
                f"array index {array_index} is outside [0, {len(rows) - 1}]"
            )
        _validate_expanded_row(rows[array_index], array_index)
        return len(rows)
    run_ids: set[str] = set()
    config_paths: set[Path] = set()
    for expected_index, row in enumerate(rows):
        run_id, config_path = _validate_expanded_row(row, expected_index)
        if run_id in run_ids or config_path in config_paths:
            raise ExperimentMatrixError("run IDs and config paths must be unique")
        run_ids.add(run_id)
        config_paths.add(config_path)
    return len(rows)


__all__ = [
    "BACKEND_CATALOG",
    "CAPABILITY_CATALOG_VERSION",
    "MATRIX_SCHEMA_VERSION",
    "MODEL_CATALOG",
    "TASK_CATALOG",
    "ExpandedRun",
    "ExperimentMatrix",
    "ExperimentMatrixError",
    "expand_experiment_matrix",
    "load_experiment_matrix",
    "materialize_experiment_matrix",
    "unsupported_reason",
    "verify_expanded_index",
]
