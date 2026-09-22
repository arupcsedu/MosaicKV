"""Paper-native PrefixKV PPL and FullKV-reference ROUGE protocol support."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, cast

from mosaickv.evaluation.metrics.registry import default_metric_registry
from mosaickv.evaluation.metrics.scorers import rouge_l_f1
from mosaickv.evaluation.metrics.types import (
    MetricObservation,
    MetricSample,
    MetricScope,
)
from mosaickv.evaluation.model import EvaluationRequest, LocalEvaluationModel
from mosaickv.evaluation.results import EvaluationResult, ResultStatus
from mosaickv.types import JsonObject, ResultCategory, ResultOrigin


@dataclass(frozen=True, slots=True)
class TeacherForcedNll:
    """Additive teacher-forced loss for one target sequence."""

    negative_log_likelihood_sum: float
    evaluated_tokens: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.negative_log_likelihood_sum, bool)
            or not math.isfinite(self.negative_log_likelihood_sum)
            or self.negative_log_likelihood_sum < 0
        ):
            raise ValueError("negative_log_likelihood_sum must be finite and nonnegative")
        if isinstance(self.evaluated_tokens, bool) or self.evaluated_tokens < 1:
            raise ValueError("evaluated_tokens must be positive")


class TeacherForcedEvaluationModel(LocalEvaluationModel, Protocol):
    """Backend contract required for PrefixKV's PPL protocol."""

    def teacher_forced_nll(self, request: EvaluationRequest, target: str) -> TeacherForcedNll:
        """Score target tokens through the same cache path used by decoding."""


@dataclass(frozen=True, slots=True)
class FullKVReferenceArtifact:
    """Frozen deterministic reference outputs used by PrefixKV ROUGE."""

    run_id: str
    model_id: str
    model_revision: str
    tokenizer_revision: str
    generation_config_sha256: str
    sample_answers: dict[str, str]
    schema_version: int = 1

    def __post_init__(self) -> None:
        for name in (
            "run_id",
            "model_id",
            "model_revision",
            "tokenizer_revision",
            "generation_config_sha256",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        if not self.sample_answers or any(
            not sample_id.strip() or not answer.strip()
            for sample_id, answer in self.sample_answers.items()
        ):
            raise ValueError("sample_answers must contain non-empty IDs and answers")
        if self.schema_version != 1:
            raise ValueError("schema_version must be 1")

    def to_json_object(self) -> JsonObject:
        return cast("JsonObject", asdict(self))


def _canonical_bytes(payload: JsonObject) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def artifact_sha256(artifact: FullKVReferenceArtifact) -> str:
    return hashlib.sha256(_canonical_bytes(artifact.to_json_object())).hexdigest()


def target_artifact_sha256(targets: Mapping[str, str]) -> str:
    """Hash the exact human-answer mapping used for teacher forcing."""

    if not targets or any(not key.strip() or not value.strip() for key, value in targets.items()):
        raise ValueError("teacher-forcing targets require non-empty IDs and answers")
    payload = cast("JsonObject", {"targets": dict(sorted(targets.items()))})
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def build_fullkv_reference_artifact(
    rows: tuple[EvaluationResult, ...],
    *,
    model_revision: str,
    tokenizer_revision: str,
    generation_config_sha256: str,
) -> FullKVReferenceArtifact:
    if not rows:
        raise ValueError("reference artifact requires at least one FullKV row")
    run_ids = {row.run_id for row in rows}
    models = {row.model for row in rows}
    if len(run_ids) != 1 or len(models) != 1:
        raise ValueError("reference rows must share one run and model")
    answers: dict[str, str] = {}
    for row in rows:
        if row.status is not ResultStatus.COMPLETED or row.answer is None:
            raise ValueError("reference artifact cannot contain failed or missing answers")
        if row.retention_ratio != 1.0 or row.method not in {"full_kv", "full_cache"}:
            raise ValueError("reference artifact accepts only retention-1 FullKV rows")
        if row.sample_id in answers:
            raise ValueError(f"duplicate reference sample {row.sample_id!r}")
        answers[row.sample_id] = row.answer
    return FullKVReferenceArtifact(
        run_id=next(iter(run_ids)),
        model_id=next(iter(models)),
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
        generation_config_sha256=generation_config_sha256,
        sample_answers=dict(sorted(answers.items())),
    )


def write_fullkv_reference_artifact(artifact: FullKVReferenceArtifact, output: str | Path) -> Path:
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite FullKV reference artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb", dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as handle:
            temporary_name = handle.name
            handle.write(_canonical_bytes(artifact.to_json_object()) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise
    return destination


def load_fullkv_reference_artifact(path: str | Path) -> FullKVReferenceArtifact:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("FullKV reference artifact must be an object")
    return FullKVReferenceArtifact(**payload)


def perplexity_sample(sample_id: str, target: str, loss: TeacherForcedNll) -> MetricSample:
    return MetricSample(
        sample_id=sample_id,
        prediction="",
        references=(target,),
        numerator=loss.negative_log_likelihood_sum,
        denominator=loss.evaluated_tokens,
    )


def rouge_samples(
    rows: tuple[EvaluationResult, ...], artifact: FullKVReferenceArtifact
) -> tuple[MetricSample, ...]:
    samples: list[MetricSample] = []
    for row in rows:
        if row.status is not ResultStatus.COMPLETED or row.answer is None:
            continue
        try:
            reference = artifact.sample_answers[row.sample_id]
        except KeyError as error:
            raise ValueError(f"reference artifact lacks sample {row.sample_id!r}") from error
        samples.append(
            MetricSample(
                sample_id=row.sample_id,
                prediction=row.answer,
                references=(reference,),
                value=rouge_l_f1(row.answer, (reference,)),
            )
        )
    if not samples:
        raise ValueError("no completed rows available for ROUGE")
    return tuple(samples)


def prefixkv_metric_observations(
    *,
    run_id: str,
    task: str,
    model: str,
    backend: str,
    method: str,
    retention_ratio: float,
    manifest_path: str,
    rouge: tuple[MetricSample, ...],
    perplexity: tuple[MetricSample, ...],
    reference_artifact_sha256: str,
    target_artifact_sha256: str,
    evaluator_commit: str,
) -> tuple[MetricObservation, ...]:
    """Materialize sample ROUGE and corpus PPL without conflating them."""

    registry = default_metric_registry()
    rouge_definition = registry.resolve("rouge_l_f1")
    ppl_definition = registry.resolve("teacher_forced_ppl")
    if any(sample.numerator is None or sample.denominator is None for sample in perplexity):
        raise ValueError("perplexity samples require NLL numerator and token denominator")
    observations: list[MetricObservation] = []
    for sample in rouge:
        if sample.value is None:
            raise ValueError("ROUGE samples require a per-sample value")
        observations.append(
            MetricObservation(
                run_id=run_id,
                sample_id=sample.sample_id,
                task=task,
                model=model,
                backend=backend,
                method=method,
                retention_ratio=retention_ratio,
                metric_name=rouge_definition.name,
                metric_value=sample.value,
                metric_direction=rouge_definition.direction,
                metric_scope=MetricScope.SAMPLE,
                metric_implementation=rouge_definition.implementation,
                metric_version=rouge_definition.version,
                metric_unit=rouge_definition.unit,
                measurement_type="baseline_paper_native_measured",
                manifest_path=manifest_path,
                evaluator_commit=evaluator_commit,
                reference_artifact_sha=reference_artifact_sha256,
                result_category=ResultCategory.PAPER_REPRODUCTION,
                result_origin=ResultOrigin.LOCALLY_REPRODUCED,
            )
        )
    ppl_sample_observations: list[MetricObservation] = []
    total_nll_values: list[float] = []
    total_tokens = 0
    for sample in perplexity:
        if sample.numerator is None or sample.denominator is None:
            raise ValueError("perplexity sample lost its sufficient statistics")
        total_nll_values.append(sample.numerator)
        total_tokens += sample.denominator
        ppl_sample_observations.append(
            MetricObservation(
                run_id=run_id,
                sample_id=sample.sample_id,
                task=task,
                model=model,
                backend=backend,
                method=method,
                retention_ratio=retention_ratio,
                metric_name=ppl_definition.name,
                metric_value=math.exp(sample.numerator / sample.denominator),
                metric_direction=ppl_definition.direction,
                metric_scope=MetricScope.SAMPLE,
                metric_implementation=ppl_definition.implementation,
                metric_version=ppl_definition.version,
                metric_unit=ppl_definition.unit,
                measurement_type="baseline_paper_native_sufficient_statistic",
                manifest_path=manifest_path,
                evaluator_commit=evaluator_commit,
                reference_artifact_sha=target_artifact_sha256,
                evaluated_token_count=sample.denominator,
                numerator=sample.numerator,
                denominator=sample.denominator,
                result_category=ResultCategory.PAPER_REPRODUCTION,
                result_origin=ResultOrigin.LOCALLY_REPRODUCED,
            )
        )
    observations.extend(ppl_sample_observations)
    total_nll = math.fsum(total_nll_values)
    observations.append(
        MetricObservation(
            run_id=run_id,
            sample_id=None,
            task=task,
            model=model,
            backend=backend,
            method=method,
            retention_ratio=retention_ratio,
            metric_name=ppl_definition.name,
            metric_value=ppl_definition.aggregate(perplexity),
            metric_direction=ppl_definition.direction,
            metric_scope=MetricScope.CORPUS,
            metric_implementation=ppl_definition.implementation,
            metric_version=ppl_definition.version,
            metric_unit=ppl_definition.unit,
            measurement_type="baseline_paper_native_measured",
            manifest_path=manifest_path,
            evaluator_commit=evaluator_commit,
            reference_artifact_sha=target_artifact_sha256,
            evaluated_token_count=total_tokens,
            numerator=total_nll,
            denominator=total_tokens,
            sample_count=len(perplexity),
            result_category=ResultCategory.PAPER_REPRODUCTION,
            result_origin=ResultOrigin.LOCALLY_REPRODUCED,
        )
    )
    return tuple(observations)


def evaluate_prefixkv_native_metrics(
    model: TeacherForcedEvaluationModel,
    requests: Sequence[EvaluationRequest],
    targets: Mapping[str, str],
    generated_rows: tuple[EvaluationResult, ...],
    fullkv_reference: FullKVReferenceArtifact,
    *,
    manifest_path: str,
    evaluator_commit: str,
) -> tuple[MetricObservation, ...]:
    """Execute PrefixKV's PPL/ROUGE protocol on one paired sample cohort."""

    if not requests or not generated_rows:
        raise ValueError("PrefixKV native evaluation requires requests and generated rows")
    request_by_id = {request.sample_id: request for request in requests}
    if len(request_by_id) != len(requests):
        raise ValueError("PrefixKV requests contain duplicate sample IDs")
    row_ids = {row.sample_id for row in generated_rows}
    if row_ids != set(request_by_id) or row_ids != set(targets):
        raise ValueError("requests, targets, and generated rows must use identical sample IDs")
    if not row_ids <= set(fullkv_reference.sample_answers):
        raise ValueError("FullKV reference artifact does not cover the evaluation cohort")
    identities = {
        (
            row.run_id,
            row.task,
            row.model,
            row.backend,
            row.method,
            row.retention_ratio,
        )
        for row in generated_rows
    }
    if len(identities) != 1:
        raise ValueError("generated rows must share one run/task/model/backend/method/ratio")
    run_id, task, model_id, backend, method, retention_ratio = next(iter(identities))
    if model_id != model.model_id or backend != model.backend or method != model.method:
        raise ValueError("generated rows do not match the teacher-forced model identity")
    ppl_samples = tuple(
        perplexity_sample(
            sample_id,
            targets[sample_id],
            model.teacher_forced_nll(request_by_id[sample_id], targets[sample_id]),
        )
        for sample_id in sorted(row_ids)
    )
    return prefixkv_metric_observations(
        run_id=run_id,
        task=task,
        model=model_id,
        backend=backend,
        method=method,
        retention_ratio=retention_ratio,
        manifest_path=manifest_path,
        rouge=rouge_samples(generated_rows, fullkv_reference),
        perplexity=ppl_samples,
        reference_artifact_sha256=artifact_sha256(fullkv_reference),
        target_artifact_sha256=target_artifact_sha256(targets),
        evaluator_commit=evaluator_commit,
    )


__all__ = [
    "FullKVReferenceArtifact",
    "TeacherForcedEvaluationModel",
    "TeacherForcedNll",
    "artifact_sha256",
    "build_fullkv_reference_artifact",
    "evaluate_prefixkv_native_metrics",
    "load_fullkv_reference_artifact",
    "perplexity_sample",
    "prefixkv_metric_observations",
    "rouge_samples",
    "target_artifact_sha256",
    "write_fullkv_reference_artifact",
]
