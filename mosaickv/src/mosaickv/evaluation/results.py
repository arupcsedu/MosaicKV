"""Strict per-sample evaluation result schema."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import cast

from mosaickv.evaluation.model import GenerationMetrics
from mosaickv.types import JsonObject


class ResultStatus(StrEnum):
    """Terminal states preserved in raw evaluation output."""

    COMPLETED = "completed"
    FAILED = "failed"


def _validate_optional_nonnegative(value: float | int | None, name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"{name} must be finite and >= 0 when present")


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Schema-v1 raw observation for one run/sample pair."""

    run_id: str
    sample_id: str
    task: str
    status: ResultStatus
    error: str | None
    model: str
    backend: str
    method: str
    retention_ratio: float
    answer: str | None
    reference: str
    task_score: float | None
    ttft: float | None
    prefill_time: float | None
    compression_time: float | None
    decode_time: float | None
    end_to_end_time: float | None
    generated_tokens: int | None
    active_kv_bytes: int | None
    residual_kv_bytes: int | None
    peak_gpu_memory: int | None
    repair_count: int | None
    repaired_bytes: int | None
    token_agreement: float | None
    logit_kl: float | None
    attention_output_error: float | None
    manifest_path: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        for name in ("run_id", "sample_id", "task", "model", "backend", "method"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        if not self.manifest_path.strip():
            raise ValueError("manifest_path must be non-empty")
        if not 0 < self.retention_ratio <= 1:
            raise ValueError("retention_ratio must be in the interval (0, 1]")
        if self.status == ResultStatus.COMPLETED:
            if self.answer is None:
                raise ValueError("completed results require an answer")
            if self.error is not None:
                raise ValueError("completed results cannot contain an error")
        elif self.error is None or not self.error.strip():
            raise ValueError("failed results require a non-empty error")
        if self.task_score is not None and (
            isinstance(self.task_score, bool) or not math.isfinite(self.task_score)
        ):
            raise ValueError("task_score must be a finite number when present")
        for name in (
            "generated_tokens",
            "active_kv_bytes",
            "residual_kv_bytes",
            "peak_gpu_memory",
            "repair_count",
            "repaired_bytes",
        ):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
                raise ValueError(f"{name} must be an integer when present")
        for name in (
            "ttft",
            "prefill_time",
            "compression_time",
            "decode_time",
            "end_to_end_time",
            "logit_kl",
            "attention_output_error",
        ):
            _validate_optional_nonnegative(getattr(self, name), name)
        if self.token_agreement is not None and not 0 <= self.token_agreement <= 1:
            raise ValueError("token_agreement must be in the interval [0, 1]")
        if self.schema_version != 1:
            raise ValueError("schema_version must be 1")

    @classmethod
    def from_generation(
        cls,
        *,
        run_id: str,
        sample_id: str,
        task: str,
        model: str,
        backend: str,
        method: str,
        retention_ratio: float,
        answer: str,
        reference: str,
        task_score: float,
        metrics: GenerationMetrics,
        manifest_path: str,
    ) -> EvaluationResult:
        """Build a completed row from a model generation."""

        return cls(
            run_id=run_id,
            sample_id=sample_id,
            task=task,
            status=ResultStatus.COMPLETED,
            error=None,
            model=model,
            backend=backend,
            method=method,
            retention_ratio=retention_ratio,
            answer=answer,
            reference=reference,
            task_score=task_score,
            manifest_path=manifest_path,
            **asdict(metrics),
        )

    @classmethod
    def failed(
        cls,
        *,
        run_id: str,
        sample_id: str,
        task: str,
        model: str,
        backend: str,
        method: str,
        retention_ratio: float,
        reference: str,
        error: str,
        manifest_path: str,
        metrics: GenerationMetrics | None = None,
        answer: str | None = None,
    ) -> EvaluationResult:
        """Build a failure row while preserving any measurements already made."""

        return cls(
            run_id=run_id,
            sample_id=sample_id,
            task=task,
            status=ResultStatus.FAILED,
            error=error,
            model=model,
            backend=backend,
            method=method,
            retention_ratio=retention_ratio,
            answer=answer,
            reference=reference,
            task_score=None,
            manifest_path=manifest_path,
            **asdict(metrics or GenerationMetrics()),
        )

    def to_json_object(self) -> JsonObject:
        """Return a JSON-safe object with enum values normalized."""

        payload = asdict(self)
        payload["status"] = self.status.value
        return cast("JsonObject", payload)

    @classmethod
    def from_json_object(cls, payload: JsonObject) -> EvaluationResult:
        """Validate and construct a result loaded from JSON."""

        data = dict(payload)
        data["status"] = ResultStatus(str(data["status"]))
        return cls(**data)  # type: ignore[arg-type]


RESULT_COLUMNS = tuple(EvaluationResult.__dataclass_fields__)


__all__ = ["RESULT_COLUMNS", "EvaluationResult", "ResultStatus"]
