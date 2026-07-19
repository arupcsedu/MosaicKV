"""Local model protocol and MosaicKV telemetry returned per generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from mosaickv.evaluation.messages import MultimodalMessage


@dataclass(frozen=True, slots=True)
class GenerationMetrics:
    """Optional timings, cache accounting, and fidelity measurements.

    Timings use seconds, memory fields use bytes, and unavailable observations
    remain ``None``. Callers must not synthesize missing measurements.
    """

    ttft: float | None = None
    prefill_time: float | None = None
    compression_time: float | None = None
    decode_time: float | None = None
    end_to_end_time: float | None = None
    generated_tokens: int | None = None
    active_kv_bytes: int | None = None
    residual_kv_bytes: int | None = None
    peak_gpu_memory: int | None = None
    repair_count: int | None = None
    repaired_bytes: int | None = None
    token_agreement: float | None = None
    logit_kl: float | None = None
    attention_output_error: float | None = None


@dataclass(frozen=True, slots=True)
class EvaluationRequest:
    """One backend-neutral generation request."""

    run_id: str
    sample_id: str
    task: str
    messages: tuple[MultimodalMessage, ...]
    generation_kwargs: dict[str, object]


@dataclass(frozen=True, slots=True)
class ModelGeneration:
    """Text plus measurements emitted by a local model implementation."""

    answer: str
    metrics: GenerationMetrics = GenerationMetrics()


class LocalEvaluationModel(Protocol):
    """Contract implemented by a local full-cache or MosaicKV model."""

    model_id: str
    backend: str
    method: str
    retention_ratio: float
    supports_video: bool

    def generate(self, request: EvaluationRequest) -> ModelGeneration:
        """Generate exactly one answer and return available measurements."""


__all__ = [
    "EvaluationRequest",
    "GenerationMetrics",
    "LocalEvaluationModel",
    "ModelGeneration",
]
