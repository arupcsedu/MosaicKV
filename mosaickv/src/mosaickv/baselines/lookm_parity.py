"""Controlled-comparison schema for official LOOK-M and ``lookm_reimpl``.

This module deliberately refuses to aggregate observations when any comparison
control differs.  A report with such a mismatch is marked ``not_comparable``;
it is not converted into a numerical baseline result.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any, cast

from mosaickv.types import JsonObject, JsonValue


class LookMParityError(ValueError):
    """Raised when a parity artifact is malformed or not comparable."""


@dataclass(frozen=True, slots=True)
class LookMParityControls:
    """Inputs that must be identical before observations may be compared."""

    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    dataset_id: str
    dataset_revision: str
    sample_set_sha256: str
    prompt_payload_sha256: str
    media_payload_sha256: str
    environment_sha256: str
    hardware_sha256: str
    measurement_protocol_sha256: str
    cache_budget_value: int
    cache_budget_unit: str
    block_size: int
    retention_ratio: float
    recent_ratio: float
    important_ratio: float
    merge_strategy: str
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
            "tokenizer_id",
            "tokenizer_revision",
            "dataset_id",
            "dataset_revision",
            "cache_budget_unit",
            "merge_strategy",
            "output_length_policy",
            "model_precision",
            "backend",
            "attention_implementation",
        ):
            if not str(getattr(self, name)).strip():
                raise LookMParityError(f"controls.{name} must be non-empty")
        for name in (
            "sample_set_sha256",
            "prompt_payload_sha256",
            "media_payload_sha256",
            "environment_sha256",
            "hardware_sha256",
            "measurement_protocol_sha256",
        ):
            digest = str(getattr(self, name))
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise LookMParityError(f"controls.{name} must be a lowercase SHA-256 digest")
        for name in ("retention_ratio", "recent_ratio", "important_ratio"):
            ratio_value = float(getattr(self, name))
            if not math.isfinite(ratio_value) or not 0 <= ratio_value <= 1:
                raise LookMParityError(f"controls.{name} must be finite and in [0, 1]")
        if not 0 < self.retention_ratio <= 1:
            raise LookMParityError("controls.retention_ratio must be in (0, 1]")
        if not math.isclose(
            self.retention_ratio,
            self.recent_ratio + self.important_ratio,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise LookMParityError(
                "controls.retention_ratio must equal recent_ratio + important_ratio"
            )
        if self.seed < 0:
            raise LookMParityError("controls.seed must be nonnegative")
        if self.cache_budget_value < 1:
            raise LookMParityError("controls.cache_budget_value must be positive")
        if self.cache_budget_unit not in {"blocks", "bytes", "retained_slots"}:
            raise LookMParityError("controls.cache_budget_unit is unsupported")
        if self.block_size != 1:
            raise LookMParityError("LOOK-M parity requires controls.block_size=1")

    @classmethod
    def from_json_object(cls, payload: JsonObject) -> LookMParityControls:
        """Validate controls loaded from an artifact."""

        return cls(**cast("dict[str, Any]", payload))


@dataclass(frozen=True, slots=True)
class LookMSelectedPositions:
    """Selected physical cache indices for one layer and KV head."""

    layer: int
    kv_head: int
    positions: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.layer < 0 or self.kv_head < 0:
            raise LookMParityError("selected layer and KV head must be nonnegative")
        if self.positions != tuple(sorted(set(self.positions))) or any(
            position < 0 for position in self.positions
        ):
            raise LookMParityError("selected positions must be sorted, unique, and nonnegative")

    @classmethod
    def from_json_object(cls, payload: JsonObject) -> LookMSelectedPositions:
        """Validate one layer/head selection record."""

        data = cast("dict[str, Any]", payload)
        return cls(
            layer=int(data["layer"]),
            kv_head=int(data["kv_head"]),
            positions=tuple(int(position) for position in data["positions"]),
        )


@dataclass(frozen=True, slots=True)
class LookMSampleObservation:
    """The five requested parity observations for one controlled sample."""

    sample_id: str
    selected_positions: tuple[LookMSelectedPositions, ...]
    active_kv_bytes: int
    generated_token_ids: tuple[int, ...]
    task_score: float
    latency_seconds: float

    def __post_init__(self) -> None:
        if not self.sample_id.strip():
            raise LookMParityError("sample_id must be non-empty")
        identities = tuple((item.layer, item.kv_head) for item in self.selected_positions)
        if identities != tuple(sorted(set(identities))):
            raise LookMParityError("selected layer/head entries must be sorted and unique")
        if self.active_kv_bytes < 0:
            raise LookMParityError("active_kv_bytes must be nonnegative")
        if not self.generated_token_ids or any(token < 0 for token in self.generated_token_ids):
            raise LookMParityError("generated_token_ids must be non-empty and nonnegative")
        if not math.isfinite(self.task_score):
            raise LookMParityError("task_score must be finite")
        if not math.isfinite(self.latency_seconds) or self.latency_seconds < 0:
            raise LookMParityError("latency_seconds must be finite and nonnegative")

    @classmethod
    def from_json_object(cls, payload: JsonObject) -> LookMSampleObservation:
        """Validate one per-sample observation."""

        data = cast("dict[str, Any]", payload)
        selections = cast("list[JsonObject]", data["selected_positions"])
        return cls(
            sample_id=str(data["sample_id"]),
            selected_positions=tuple(
                LookMSelectedPositions.from_json_object(item) for item in selections
            ),
            active_kv_bytes=int(data["active_kv_bytes"]),
            generated_token_ids=tuple(
                int(token) for token in cast("list[Any]", data["generated_token_ids"])
            ),
            task_score=float(data["task_score"]),
            latency_seconds=float(data["latency_seconds"]),
        )


@dataclass(frozen=True, slots=True)
class LookMParityArtifact:
    """Official or local observation bundle with immutable provenance."""

    implementation: str
    official_repository_sha: str
    executable_git_sha: str
    config_sha256: str
    manifest_path: str
    measurement_type: str
    controls: LookMParityControls
    samples: tuple[LookMSampleObservation, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.implementation not in {"official_lookm", "lookm_reimpl"}:
            raise LookMParityError(
                "implementation must be 'official_lookm' or 'lookm_reimpl'"
            )
        if len(self.official_repository_sha) != 40 or any(
            character not in "0123456789abcdef"
            for character in self.official_repository_sha
        ):
            raise LookMParityError("official_repository_sha must be a lowercase git SHA")
        if not self.executable_git_sha.strip() or not self.manifest_path.strip():
            raise LookMParityError("executable_git_sha and manifest_path must be non-empty")
        if len(self.config_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.config_sha256
        ):
            raise LookMParityError("config_sha256 must be a lowercase SHA-256 digest")
        if self.measurement_type not in {
            "baseline_official_measured",
            "baseline_reimpl_measured",
        }:
            raise LookMParityError("measurement_type must identify official or reimplementation")
        if (
            self.implementation == "official_lookm"
            and self.measurement_type != "baseline_official_measured"
        ):
            raise LookMParityError(
                "official_lookm must use measurement_type=baseline_official_measured"
            )
        if (
            self.implementation == "lookm_reimpl"
            and self.measurement_type != "baseline_reimpl_measured"
        ):
            raise LookMParityError(
                "lookm_reimpl must use measurement_type=baseline_reimpl_measured"
            )
        if not self.samples:
            raise LookMParityError("a parity artifact must contain at least one sample")
        sample_ids = tuple(sample.sample_id for sample in self.samples)
        if sample_ids != tuple(sorted(set(sample_ids))):
            raise LookMParityError("samples must be sorted by unique sample_id")
        if self.schema_version != 1:
            raise LookMParityError("schema_version must equal 1")

    @classmethod
    def from_json_object(cls, payload: JsonObject) -> LookMParityArtifact:
        """Validate an artifact loaded from JSON."""

        data = cast("dict[str, Any]", payload)
        return cls(
            implementation=str(data["implementation"]),
            official_repository_sha=str(data["official_repository_sha"]),
            executable_git_sha=str(data["executable_git_sha"]),
            config_sha256=str(data["config_sha256"]),
            manifest_path=str(data["manifest_path"]),
            measurement_type=str(data["measurement_type"]),
            controls=LookMParityControls.from_json_object(
                cast("JsonObject", data["controls"])
            ),
            samples=tuple(
                LookMSampleObservation.from_json_object(item)
                for item in cast("list[JsonObject]", data["samples"])
            ),
            schema_version=int(data.get("schema_version", 1)),
        )


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def comparison_control_mismatches(
    official: LookMParityArtifact,
    reimplementation: LookMParityArtifact,
) -> tuple[str, ...]:
    """Return exact control fields that prevent a direct comparison."""

    mismatches: list[str] = []
    official_controls = asdict(official.controls)
    reimplementation_controls = asdict(reimplementation.controls)
    for field_name in official_controls:
        if _canonical(official_controls[field_name]) != _canonical(
            reimplementation_controls[field_name]
        ):
            mismatches.append(f"controls.{field_name}")
    if official.official_repository_sha != reimplementation.official_repository_sha:
        mismatches.append("official_repository_sha")
    official_samples = tuple(sample.sample_id for sample in official.samples)
    reimplementation_samples = tuple(sample.sample_id for sample in reimplementation.samples)
    if official_samples != reimplementation_samples:
        mismatches.append("sample_ids")
    return tuple(mismatches)


def _token_agreement(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    denominator = max(len(left), len(right))
    matches = sum(first == second for first, second in zip(left, right, strict=False))
    return matches / denominator


def _selected_positions_json(
    selections: tuple[LookMSelectedPositions, ...],
) -> list[JsonValue]:
    return [
        {
            "layer": selection.layer,
            "kv_head": selection.kv_head,
            "positions": list(selection.positions),
        }
        for selection in selections
    ]


def compare_lookm_artifacts(
    official: LookMParityArtifact,
    reimplementation: LookMParityArtifact,
) -> JsonObject:
    """Compare requested metrics only after every control has matched."""

    if official.implementation != "official_lookm":
        raise LookMParityError("the first artifact must be official_lookm")
    if reimplementation.implementation != "lookm_reimpl":
        raise LookMParityError("the second artifact must be lookm_reimpl")
    mismatches = comparison_control_mismatches(official, reimplementation)
    if mismatches:
        raise LookMParityError("comparison controls differ: " + ", ".join(mismatches))
    rows: list[JsonObject] = []
    for left, right in zip(official.samples, reimplementation.samples, strict=True):
        left_positions = _selected_positions_json(left.selected_positions)
        right_positions = _selected_positions_json(right.selected_positions)
        rows.append(
            {
                "sample_id": left.sample_id,
                "selected_positions_exact_match": left_positions == right_positions,
                "official_selected_positions": left_positions,
                "reimpl_selected_positions": right_positions,
                "official_active_kv_bytes": left.active_kv_bytes,
                "reimpl_active_kv_bytes": right.active_kv_bytes,
                "active_kv_bytes_delta": right.active_kv_bytes - left.active_kv_bytes,
                "official_generated_token_ids": list(left.generated_token_ids),
                "reimpl_generated_token_ids": list(right.generated_token_ids),
                "generated_tokens_exact_match": (
                    left.generated_token_ids == right.generated_token_ids
                ),
                "token_agreement": _token_agreement(
                    left.generated_token_ids, right.generated_token_ids
                ),
                "official_task_score": left.task_score,
                "reimpl_task_score": right.task_score,
                "task_score_delta": right.task_score - left.task_score,
                "official_latency_seconds": left.latency_seconds,
                "reimpl_latency_seconds": right.latency_seconds,
                "latency_seconds_delta": right.latency_seconds - left.latency_seconds,
            }
        )
    return cast(
        "JsonObject",
        {
            "schema_version": 1,
            "status": "comparable",
            "official_implementation": "official_lookm",
            "reimplementation": "lookm_reimpl",
            "official_repository_sha": official.official_repository_sha,
            "measurement_type": "controlled_official_vs_reimplementation_parity",
            "control_mismatches": [],
            "samples": rows,
        },
    )


def build_lookm_parity_report(
    official: LookMParityArtifact,
    reimplementation: LookMParityArtifact,
) -> JsonObject:
    """Build a comparable report or an explicit non-result on mismatched controls."""

    mismatches = comparison_control_mismatches(official, reimplementation)
    if mismatches:
        return {
            "schema_version": 1,
            "status": "not_comparable",
            "official_implementation": official.implementation,
            "reimplementation": reimplementation.implementation,
            "official_repository_sha": official.official_repository_sha,
            "measurement_type": "comparison_validation_failure",
            "control_mismatches": list(mismatches),
            "samples": [],
        }
    return compare_lookm_artifacts(official, reimplementation)


__all__ = [
    "LookMParityArtifact",
    "LookMParityControls",
    "LookMParityError",
    "LookMSampleObservation",
    "LookMSelectedPositions",
    "build_lookm_parity_report",
    "compare_lookm_artifacts",
    "comparison_control_mismatches",
]
