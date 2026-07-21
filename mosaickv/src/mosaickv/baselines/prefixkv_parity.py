"""Controlled official PrefixKV versus ``prefixkv_reimpl`` comparisons."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from mosaickv.types import JsonObject


class PrefixKVParityError(ValueError):
    """Raised when a PrefixKV parity artifact is malformed."""


def _digest(value: str, name: str, length: int = 64) -> None:
    if len(value) != length or any(character not in "0123456789abcdef" for character in value):
        raise PrefixKVParityError(f"{name} must be a lowercase {length * 4}-bit digest")


@dataclass(frozen=True, slots=True)
class PrefixKVParityControls:
    """Comparison controls that must match before numerical deltas are valid."""

    model_id: str
    model_revision: str
    tokenizer_revision: str
    dataset_id: str
    dataset_revision: str
    calibration_sample_set_sha256: str
    evaluation_sample_set_sha256: str
    prompt_payload_sha256: str
    media_payload_sha256: str
    profile_sha256: str
    environment_sha256: str
    hardware_sha256: str
    measurement_protocol_sha256: str
    cache_budget_value: int
    cache_budget_unit: str
    block_size: int
    retention_ratio: float
    official_forget_ratio: float
    generation_parameters: JsonObject
    output_length_policy: str
    model_precision: str
    backend: str
    backend_configuration: JsonObject
    attention_implementation: str
    seed: int

    def __post_init__(self) -> None:
        for name in (
            "model_id",
            "model_revision",
            "tokenizer_revision",
            "dataset_id",
            "dataset_revision",
            "output_length_policy",
            "model_precision",
            "backend",
            "cache_budget_unit",
            "attention_implementation",
        ):
            if not str(getattr(self, name)).strip():
                raise PrefixKVParityError(f"controls.{name} must be non-empty")
        for name in (
            "calibration_sample_set_sha256",
            "evaluation_sample_set_sha256",
            "prompt_payload_sha256",
            "media_payload_sha256",
            "profile_sha256",
            "environment_sha256",
            "hardware_sha256",
            "measurement_protocol_sha256",
        ):
            _digest(str(getattr(self, name)), f"controls.{name}")
        if not 0 < self.retention_ratio <= 1:
            raise PrefixKVParityError("controls.retention_ratio must be in (0, 1]")
        if not math.isclose(
            self.official_forget_ratio,
            1 - self.retention_ratio,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise PrefixKVParityError("official_forget_ratio must equal one minus retention_ratio")
        if self.seed < 0:
            raise PrefixKVParityError("controls.seed must be nonnegative")
        if self.cache_budget_value < 1:
            raise PrefixKVParityError("controls.cache_budget_value must be positive")
        if self.cache_budget_unit not in {"blocks", "bytes"}:
            raise PrefixKVParityError("controls.cache_budget_unit must be blocks or bytes")
        if self.block_size != 1:
            raise PrefixKVParityError("PrefixKV parity requires controls.block_size=1")

    @classmethod
    def from_json_object(cls, payload: JsonObject) -> PrefixKVParityControls:
        return cls(**cast("dict[str, Any]", payload))


@dataclass(frozen=True, slots=True)
class PrefixKVSampleObservation:
    """Requested layer, byte, quality, answer, and latency observations."""

    sample_id: str
    per_layer_cache_sizes: tuple[int, ...]
    total_retained_bytes: int
    actual_active_kv_bytes: int
    generated_answer: str
    generated_token_ids: tuple[int, ...]
    latency_seconds: float
    perplexity: float | None = None
    rouge_l_f1: float | None = None

    def __post_init__(self) -> None:
        if not self.sample_id.strip() or not self.per_layer_cache_sizes:
            raise PrefixKVParityError("sample ID and per-layer cache sizes must be non-empty")
        if any(value < 1 for value in self.per_layer_cache_sizes):
            raise PrefixKVParityError("per-layer cache sizes must be positive")
        if self.total_retained_bytes < 1 or self.actual_active_kv_bytes < 1:
            raise PrefixKVParityError("retained and actual active KV bytes must be positive")
        if self.actual_active_kv_bytes < self.total_retained_bytes:
            raise PrefixKVParityError(
                "actual_active_kv_bytes cannot be smaller than retained payload bytes"
            )
        if not self.generated_token_ids or any(token < 0 for token in self.generated_token_ids):
            raise PrefixKVParityError("generated token IDs must be non-empty and nonnegative")
        if not math.isfinite(self.latency_seconds) or self.latency_seconds < 0:
            raise PrefixKVParityError("latency_seconds must be finite and nonnegative")
        for name in ("perplexity", "rouge_l_f1"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise PrefixKVParityError(f"{name} must be finite and nonnegative or null")

    @classmethod
    def from_json_object(cls, payload: JsonObject) -> PrefixKVSampleObservation:
        data = cast("dict[str, Any]", payload)
        return cls(
            sample_id=str(data["sample_id"]),
            per_layer_cache_sizes=tuple(int(value) for value in data["per_layer_cache_sizes"]),
            total_retained_bytes=int(data["total_retained_bytes"]),
            actual_active_kv_bytes=int(data["actual_active_kv_bytes"]),
            generated_answer=str(data["generated_answer"]),
            generated_token_ids=tuple(int(value) for value in data["generated_token_ids"]),
            latency_seconds=float(data["latency_seconds"]),
            perplexity=(None if data.get("perplexity") is None else float(data["perplexity"])),
            rouge_l_f1=(None if data.get("rouge_l_f1") is None else float(data["rouge_l_f1"])),
        )


@dataclass(frozen=True, slots=True)
class PrefixKVParityArtifact:
    """Official or reimplementation measurements with immutable provenance."""

    implementation: str
    official_repository_sha: str
    executable_git_sha: str
    config_sha256: str
    manifest_path: str
    measurement_type: str
    controls: PrefixKVParityControls
    samples: tuple[PrefixKVSampleObservation, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.implementation not in {
            "official_prefixkv",
            "prefixkv_reimpl",
            "generalized_prefixkv_reimpl",
        }:
            raise PrefixKVParityError("unsupported PrefixKV implementation label")
        _digest(self.official_repository_sha, "official_repository_sha", 40)
        _digest(self.config_sha256, "config_sha256")
        if not self.executable_git_sha.strip() or not self.manifest_path.strip():
            raise PrefixKVParityError("executable_git_sha and manifest_path must be non-empty")
        expected_measurement = (
            "baseline_official_measured"
            if self.implementation == "official_prefixkv"
            else "baseline_reimpl_measured"
        )
        if self.measurement_type != expected_measurement:
            raise PrefixKVParityError(
                f"{self.implementation} requires measurement_type={expected_measurement}"
            )
        if not self.samples:
            raise PrefixKVParityError("a parity artifact must contain samples")
        ids = tuple(sample.sample_id for sample in self.samples)
        if ids != tuple(sorted(set(ids))):
            raise PrefixKVParityError("parity sample IDs must be sorted and unique")
        if self.schema_version != 1:
            raise PrefixKVParityError("schema_version must equal 1")

    def to_json_object(self) -> JsonObject:
        return cast("JsonObject", asdict(self))

    @classmethod
    def from_json_object(cls, payload: JsonObject) -> PrefixKVParityArtifact:
        data = cast("dict[str, Any]", payload)
        return cls(
            implementation=str(data["implementation"]),
            official_repository_sha=str(data["official_repository_sha"]),
            executable_git_sha=str(data["executable_git_sha"]),
            config_sha256=str(data["config_sha256"]),
            manifest_path=str(data["manifest_path"]),
            measurement_type=str(data["measurement_type"]),
            controls=PrefixKVParityControls.from_json_object(cast("JsonObject", data["controls"])),
            samples=tuple(
                PrefixKVSampleObservation.from_json_object(cast("JsonObject", sample))
                for sample in data["samples"]
            ),
            schema_version=int(data.get("schema_version", 1)),
        )


def prefixkv_control_mismatches(
    official: PrefixKVParityControls, reimplementation: PrefixKVParityControls
) -> tuple[str, ...]:
    """Return exact control fields that differ."""

    first = asdict(official)
    second = asdict(reimplementation)
    return tuple(sorted(key for key in first if first[key] != second[key]))


def _token_agreement(first: tuple[int, ...], second: tuple[int, ...]) -> float:
    length = max(len(first), len(second))
    return 1.0 if length == 0 else sum(a == b for a, b in zip(first, second, strict=False)) / length


def compare_prefixkv_artifacts(
    official: PrefixKVParityArtifact,
    reimplementation: PrefixKVParityArtifact,
) -> JsonObject:
    """Compare artifacts only after every controlled input has matched."""

    if official.implementation != "official_prefixkv":
        raise PrefixKVParityError("first artifact must be official_prefixkv")
    if reimplementation.implementation != "prefixkv_reimpl":
        raise PrefixKVParityError(
            "official parity is only defined for LLaVA prefixkv_reimpl, not generalized results"
        )
    mismatches = prefixkv_control_mismatches(official.controls, reimplementation.controls)
    if mismatches:
        return {
            "status": "not_comparable",
            "reason": "controlled inputs differ",
            "control_mismatches": list(mismatches),
            "samples": [],
        }
    official_by_id = {sample.sample_id: sample for sample in official.samples}
    reimpl_by_id = {sample.sample_id: sample for sample in reimplementation.samples}
    if set(official_by_id) != set(reimpl_by_id):
        return {
            "status": "not_comparable",
            "reason": "sample IDs differ",
            "control_mismatches": ["sample_ids"],
            "samples": [],
        }
    rows: list[JsonObject] = []
    for sample_id in sorted(official_by_id):
        first = official_by_id[sample_id]
        second = reimpl_by_id[sample_id]
        rows.append(
            {
                "sample_id": sample_id,
                "per_layer_cache_sizes_match": (
                    first.per_layer_cache_sizes == second.per_layer_cache_sizes
                ),
                "official_per_layer_cache_sizes": list(first.per_layer_cache_sizes),
                "reimpl_per_layer_cache_sizes": list(second.per_layer_cache_sizes),
                "retained_byte_delta": second.total_retained_bytes - first.total_retained_bytes,
                "actual_active_kv_byte_delta": second.actual_active_kv_bytes
                - first.actual_active_kv_bytes,
                "perplexity_delta": (
                    None
                    if first.perplexity is None or second.perplexity is None
                    else second.perplexity - first.perplexity
                ),
                "rouge_l_f1_delta": (
                    None
                    if first.rouge_l_f1 is None or second.rouge_l_f1 is None
                    else second.rouge_l_f1 - first.rouge_l_f1
                ),
                "answers_exact_match": first.generated_answer == second.generated_answer,
                "token_agreement": _token_agreement(
                    first.generated_token_ids, second.generated_token_ids
                ),
                "official_latency_seconds": first.latency_seconds,
                "reimpl_latency_seconds": second.latency_seconds,
                "latency_delta_seconds": second.latency_seconds - first.latency_seconds,
            }
        )
    return cast(
        "JsonObject",
        {
            "status": "comparable",
            "control_mismatches": [],
            "official_repository_sha": official.official_repository_sha,
            "samples": rows,
        },
    )


def load_prefixkv_parity_artifact(path: str) -> PrefixKVParityArtifact:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PrefixKVParityError(f"cannot read parity artifact {path}: {error}") from error
    if not isinstance(payload, dict):
        raise PrefixKVParityError("parity artifact root must be a JSON object")
    return PrefixKVParityArtifact.from_json_object(cast("JsonObject", payload))


__all__ = [
    "PrefixKVParityArtifact",
    "PrefixKVParityControls",
    "PrefixKVParityError",
    "PrefixKVSampleObservation",
    "compare_prefixkv_artifacts",
    "load_prefixkv_parity_artifact",
    "prefixkv_control_mismatches",
]
