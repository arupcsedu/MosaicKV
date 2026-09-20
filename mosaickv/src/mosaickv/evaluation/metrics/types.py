"""Strict schemas for sample contributions and final metric observations."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import cast

from mosaickv.types import JsonObject


class MetricDirection(StrEnum):
    """Optimization direction used by tables and Pareto analyses."""

    HIGHER = "higher"
    LOWER = "lower"


class MetricScope(StrEnum):
    """Whether a value belongs to one sample or a complete corpus."""

    SAMPLE = "sample"
    CORPUS = "corpus"


@dataclass(frozen=True, slots=True)
class MetricSample:
    """Sufficient statistics retained for one sample.

    ``value`` is used by mean-reduced metrics.  ``numerator`` and
    ``denominator`` preserve additive sufficient statistics such as total NLL
    and evaluated target-token count.  Predictions and references are retained
    for corpus reducers such as CIDEr, which cannot be reconstructed from a
    mean of per-sample values.
    """

    sample_id: str
    prediction: str
    references: tuple[str, ...]
    value: float | None = None
    numerator: float | None = None
    denominator: int | None = None
    metadata_json: str = "{}"

    def __post_init__(self) -> None:
        if not self.sample_id.strip():
            raise ValueError("sample_id must be non-empty")
        if not self.references or any(not value.strip() for value in self.references):
            raise ValueError("references must contain non-empty strings")
        for name in ("value", "numerator"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not math.isfinite(value)):
                raise ValueError(f"{name} must be finite when present")
        if self.denominator is not None and (
            isinstance(self.denominator, bool) or self.denominator <= 0
        ):
            raise ValueError("denominator must be a positive integer when present")
        try:
            metadata = json.loads(self.metadata_json)
        except json.JSONDecodeError as error:
            raise ValueError("metadata_json must be valid JSON") from error
        if not isinstance(metadata, dict):
            raise ValueError("metadata_json must encode an object")


@dataclass(frozen=True, slots=True)
class MetricObservation:
    """One immutable sample- or corpus-level metric result."""

    run_id: str
    task: str
    model: str
    backend: str
    method: str
    retention_ratio: float
    metric_name: str
    metric_value: float
    metric_direction: MetricDirection
    metric_scope: MetricScope
    metric_implementation: str
    metric_version: str
    metric_unit: str
    measurement_type: str
    manifest_path: str
    sample_id: str | None = None
    metadata_json: str = "{}"
    evaluator_commit: str = "not_used"
    reference_artifact_sha: str = "not_used"
    evaluated_token_count: int | None = None
    numerator: float | None = None
    denominator: int | None = None
    sample_count: int = 1
    schema_version: int = 1

    def __post_init__(self) -> None:
        required = (
            "run_id",
            "task",
            "model",
            "backend",
            "method",
            "metric_name",
            "metric_implementation",
            "metric_version",
            "metric_unit",
            "measurement_type",
            "manifest_path",
            "evaluator_commit",
            "reference_artifact_sha",
        )
        for name in required:
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        if not 0 < self.retention_ratio <= 1:
            raise ValueError("retention_ratio must be in (0, 1]")
        for name in ("metric_value", "numerator"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not math.isfinite(value)):
                raise ValueError(f"{name} must be finite when present")
        if self.metric_scope is MetricScope.SAMPLE and not (self.sample_id or "").strip():
            raise ValueError("sample-scoped observations require sample_id")
        if self.metric_scope is MetricScope.CORPUS and self.sample_id is not None:
            raise ValueError("corpus-scoped observations cannot have sample_id")
        try:
            metadata = json.loads(self.metadata_json)
        except json.JSONDecodeError as error:
            raise ValueError("metadata_json must be valid JSON") from error
        if not isinstance(metadata, dict):
            raise ValueError("metadata_json must encode an object")
        if self.denominator is not None and self.denominator <= 0:
            raise ValueError("denominator must be positive when present")
        if self.evaluated_token_count is not None and (
            isinstance(self.evaluated_token_count, bool)
            or not isinstance(self.evaluated_token_count, int)
            or self.evaluated_token_count < 0
        ):
            raise ValueError("evaluated_token_count must be a non-negative integer when present")
        if self.sample_count < 1:
            raise ValueError("sample_count must be positive")
        if self.schema_version != 1:
            raise ValueError("schema_version must be 1")

    def to_json_object(self) -> JsonObject:
        payload = asdict(self)
        payload["metric_direction"] = self.metric_direction.value
        payload["metric_scope"] = self.metric_scope.value
        return cast("JsonObject", payload)

    @classmethod
    def from_json_object(cls, payload: JsonObject) -> MetricObservation:
        data = dict(payload)
        legacy_reference_sha = data.pop("reference_artifact_sha256", None)
        if "reference_artifact_sha" not in data and legacy_reference_sha is not None:
            data["reference_artifact_sha"] = legacy_reference_sha
        elif (
            legacy_reference_sha is not None
            and data.get("reference_artifact_sha") != legacy_reference_sha
        ):
            raise ValueError(
                "conflicting reference_artifact_sha and legacy reference_artifact_sha256"
            )
        if "evaluated_token_count" not in data and data.get("denominator") is not None:
            data["evaluated_token_count"] = data["denominator"]
        data["metric_direction"] = MetricDirection(str(data["metric_direction"]))
        data["metric_scope"] = MetricScope(str(data["metric_scope"]))
        return cls(**data)  # type: ignore[arg-type]

    @property
    def reference_artifact_sha256(self) -> str:
        """Compatibility accessor for code written against schema-v0 naming."""

        return self.reference_artifact_sha


METRIC_COLUMNS = tuple(MetricObservation.__dataclass_fields__)


__all__ = [
    "METRIC_COLUMNS",
    "MetricDirection",
    "MetricObservation",
    "MetricSample",
    "MetricScope",
]
