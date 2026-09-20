"""Executable, fail-closed runners for paper-native baseline protocols.

PrefixKV's LLaVA-Description/MM-Vet inputs and LOOK-M's MileBench inputs do
not have audited lmms-eval 0.7.2 routes.  This module consumes a small,
content-addressed materialized ledger instead of silently substituting a
different benchmark or scorer.  Public VL-Cache tasks continue to execute
through :mod:`mosaickv.evaluation.lmms_adapter`.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import cast

from mosaickv.evaluation.messages import MediaItem, MediaKind, build_multimodal_messages
from mosaickv.evaluation.metrics.registry import default_metric_registry
from mosaickv.evaluation.metrics.storage import (
    JsonlMetricStore,
    write_metric_parquet_tables,
)
from mosaickv.evaluation.metrics.types import (
    MetricDirection,
    MetricObservation,
    MetricScope,
)
from mosaickv.evaluation.milebench import (
    MileBenchMetric,
    MileBenchQuestionType,
    MileBenchSampleScore,
    aggregate_milebench_samples,
    route_milebench_dataset,
    score_milebench_sample,
)
from mosaickv.evaluation.model import (
    EvaluationRequest,
    GenerationMetrics,
    LocalEvaluationModel,
)
from mosaickv.evaluation.prefixkv_metrics import (
    TeacherForcedEvaluationModel,
    artifact_sha256,
    build_fullkv_reference_artifact,
    evaluate_prefixkv_native_metrics,
    load_fullkv_reference_artifact,
    write_fullkv_reference_artifact,
)
from mosaickv.evaluation.published_protocols import (
    PublishedBaselineProtocol,
    PublishedProtocolSuite,
)
from mosaickv.evaluation.results import EvaluationResult, ResultStatus
from mosaickv.evaluation.storage import JsonlResultStore, write_parquet_aggregate
from mosaickv.evaluation.tasks import TaskSpec, deterministic_sample_ids
from mosaickv.manifest import sha256_bytes
from mosaickv.types import JsonObject

_PREFIXKV_TASKS = frozenset({"llava_description", "mmvet_prefixkv"})
_PREFIXKV_METHODS = frozenset(
    {
        "full_kv",
        "prefixkv_reimpl",
        "elastic_cache_reimpl",
        "pyramidkv_reimpl",
        "h2o_reimpl",
        "mosaickv_exact",
    }
)
_MILEBENCH_METHODS = frozenset(
    {
        "full_kv",
        "lookm_reimpl",
        "snapkv_reimpl",
        "roco_reimpl",
        "h2o_reimpl",
        "mosaickv_exact",
    }
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class MaterializedMedia:
    """One immutable media object referenced by a paper-native ledger."""

    kind: MediaKind
    path: Path
    sha256: str

    def __post_init__(self) -> None:
        if len(self.sha256) != 64 or any(char not in "0123456789abcdef" for char in self.sha256):
            raise ValueError("media sha256 must be a lowercase SHA-256 digest")
        resolved = self.path.resolve(strict=True)
        home = Path.home().resolve()
        if resolved == home or home in resolved.parents:
            raise ValueError(
                f"paper-native media cannot be read from the home directory: {resolved}"
            )
        if not resolved.is_file():
            raise ValueError(f"paper-native media is not a regular file: {resolved}")
        observed = _file_sha256(resolved)
        if observed != self.sha256:
            raise ValueError(
                f"media SHA mismatch for {resolved}: expected {self.sha256}, observed {observed}"
            )
        object.__setattr__(self, "path", resolved)


@dataclass(frozen=True, slots=True)
class MaterializedSample:
    """One prompt, reference set, media tuple, and scorer metadata."""

    sample_id: str
    prompt: str
    references: tuple[str, ...]
    media: tuple[MaterializedMedia, ...]
    teacher_forcing_target: str | None = None
    milebench_dataset: str | None = None
    milebench_question_type: MileBenchQuestionType | None = None
    choices: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.sample_id.strip() or not self.prompt.strip():
            raise ValueError("materialized sample ID and prompt must be non-empty")
        if not self.references or any(not reference.strip() for reference in self.references):
            raise ValueError("materialized sample references must be non-empty")
        if not self.media:
            raise ValueError("paper-native multimodal samples require media")
        if self.teacher_forcing_target is not None and not self.teacher_forcing_target.strip():
            raise ValueError("teacher_forcing_target must be non-empty when provided")
        milebench_fields = (
            self.milebench_dataset,
            self.milebench_question_type,
        )
        if any(value is not None for value in milebench_fields) and not all(
            value is not None for value in milebench_fields
        ):
            raise ValueError("MileBench dataset and question type must be provided together")
        if self.milebench_dataset is not None:
            route = route_milebench_dataset(
                self.milebench_dataset,
                cast("MileBenchQuestionType", self.milebench_question_type),
            )
            if route.question_type is MileBenchQuestionType.MULTI_CHOICE and not self.choices:
                raise ValueError("multiple-choice MileBench samples require choices")
            if route.question_type is MileBenchQuestionType.OPEN_ENDED and self.choices:
                raise ValueError("open-ended MileBench samples cannot contain choices")

    def canonical_reference(self) -> str:
        if len(self.references) == 1:
            return self.references[0]
        return json.dumps(self.references, ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class PaperNativeLedger:
    """Hash-bound materialized inputs for one exact benchmark route."""

    task: str
    dataset_id: str
    dataset_revision: str
    split: str
    samples: tuple[MaterializedSample, ...]
    sha256: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        for name in ("task", "dataset_id", "dataset_revision", "split"):
            if not getattr(self, name).strip():
                raise ValueError(f"paper-native ledger {name} must be non-empty")
        if self.schema_version != 1:
            raise ValueError("paper-native ledger schema_version must be 1")
        if re.fullmatch(r"[0-9a-f]{40,64}", self.dataset_revision) is None:
            raise ValueError("paper-native dataset_revision must be an immutable 40-64 hex SHA")
        if not self.samples:
            raise ValueError("paper-native ledger cannot be empty")
        ids = [sample.sample_id for sample in self.samples]
        if len(ids) != len(set(ids)):
            raise ValueError("paper-native ledger contains duplicate sample IDs")


@dataclass(frozen=True, slots=True)
class CacheOperatingPoint:
    """Nominal retention and measured active K/V storage for one run."""

    method: str
    nominal_retention_ratio: float
    sample_count: int
    mean_active_kv_bytes: float
    minimum_active_kv_bytes: int
    maximum_active_kv_bytes: int

    def to_json_object(self) -> JsonObject:
        return cast("JsonObject", asdict(self))


def _required_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _load_media(value: object, ledger_root: Path, path: str) -> MaterializedMedia:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    raw_path = Path(_required_string(value.get("path"), f"{path}.path"))
    resolved = raw_path if raw_path.is_absolute() else ledger_root / raw_path
    try:
        kind = MediaKind(_required_string(value.get("kind"), f"{path}.kind"))
    except ValueError as error:
        raise ValueError(f"{path}.kind must be image or video") from error
    return MaterializedMedia(
        kind=kind,
        path=resolved,
        sha256=_required_string(value.get("sha256"), f"{path}.sha256"),
    )


def _load_sample(value: object, ledger_root: Path, path: str) -> MaterializedSample:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    references = value.get("references")
    media = value.get("media")
    choices = value.get("choices", [])
    if not isinstance(references, list) or not all(isinstance(item, str) for item in references):
        raise ValueError(f"{path}.references must be a string array")
    if not isinstance(media, list):
        raise ValueError(f"{path}.media must be an array")
    if not isinstance(choices, list) or not all(isinstance(item, str) for item in choices):
        raise ValueError(f"{path}.choices must be a string array")
    question_type_value = value.get("milebench_question_type")
    question_type = (
        None
        if question_type_value is None
        else MileBenchQuestionType(_required_string(question_type_value, f"{path}.question_type"))
    )
    target = value.get("teacher_forcing_target")
    dataset = value.get("milebench_dataset")
    if target is not None and not isinstance(target, str):
        raise ValueError(f"{path}.teacher_forcing_target must be a string or null")
    if dataset is not None and not isinstance(dataset, str):
        raise ValueError(f"{path}.milebench_dataset must be a string or null")
    return MaterializedSample(
        sample_id=_required_string(value.get("sample_id"), f"{path}.sample_id"),
        prompt=_required_string(value.get("prompt"), f"{path}.prompt"),
        references=tuple(cast("list[str]", references)),
        media=tuple(
            _load_media(item, ledger_root, f"{path}.media[{index}]")
            for index, item in enumerate(media)
        ),
        teacher_forcing_target=target,
        milebench_dataset=dataset,
        milebench_question_type=question_type,
        choices=tuple(cast("list[str]", choices)),
    )


def load_paper_native_ledger(
    path: str | Path,
    *,
    expected_sha256: str,
    task: TaskSpec,
    dataset_revision: str,
) -> PaperNativeLedger:
    """Load and fully validate an immutable materialized sample ledger."""

    source = Path(path).resolve(strict=True)
    raw = source.read_bytes()
    observed = sha256_bytes(raw)
    if observed != expected_sha256:
        raise ValueError(
            f"paper-native ledger SHA mismatch: expected {expected_sha256}, observed {observed}"
        )
    payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise ValueError("paper-native ledger must be a JSON object")
    schema_version = payload.get("schema_version")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise ValueError("paper-native ledger schema_version must be an integer")
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list):
        raise ValueError("paper-native ledger samples must be an array")
    ledger = PaperNativeLedger(
        task=_required_string(payload.get("task"), "task"),
        dataset_id=_required_string(payload.get("dataset_id"), "dataset_id"),
        dataset_revision=_required_string(payload.get("dataset_revision"), "dataset_revision"),
        split=_required_string(payload.get("split"), "split"),
        samples=tuple(
            _load_sample(item, source.parent, f"samples[{index}]")
            for index, item in enumerate(raw_samples)
        ),
        sha256=observed,
        schema_version=schema_version,
    )
    expected = (task.name, task.dataset_id, dataset_revision, task.split)
    actual = (ledger.task, ledger.dataset_id, ledger.dataset_revision, ledger.split)
    if actual != expected:
        raise ValueError(
            f"paper-native ledger identity mismatch: expected {expected!r}, got {actual!r}"
        )
    if task.name in _PREFIXKV_TASKS:
        if any(sample.teacher_forcing_target is None for sample in ledger.samples):
            raise ValueError(
                "PrefixKV native ledgers require teacher_forcing_target for every sample"
            )
        if any(sample.milebench_dataset is not None for sample in ledger.samples):
            raise ValueError("PrefixKV native ledgers cannot contain MileBench routing metadata")
    elif task.name == "milebench":
        if any(sample.milebench_dataset is None for sample in ledger.samples):
            raise ValueError("MileBench ledgers require official routing metadata for every sample")
        if any(len(sample.references) != 1 for sample in ledger.samples):
            raise ValueError("MileBench ledgers require exactly one official reference per sample")
        if any(sample.teacher_forcing_target is not None for sample in ledger.samples):
            raise ValueError("MileBench ledgers cannot contain teacher-forcing targets")
    else:
        raise ValueError(f"task {task.name!r} is not a materialized paper-native route")
    return ledger


def _media_payload(item: MaterializedMedia) -> object:
    if item.kind is MediaKind.VIDEO:
        return str(item.path)
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - pinned common environment includes Pillow.
        raise RuntimeError("Pillow is required for paper-native image ledgers") from error
    with Image.open(item.path) as image:
        return image.convert("RGB").copy()


def _validate_method(task: str, method: str) -> None:
    allowed = _PREFIXKV_METHODS if task in _PREFIXKV_TASKS else _MILEBENCH_METHODS
    if method not in allowed:
        raise ValueError(
            f"method {method!r} is not enabled for paper-native task {task!r}; "
            f"allowed methods: {', '.join(sorted(allowed))}"
        )


def cache_operating_point(rows: Sequence[EvaluationResult]) -> CacheOperatingPoint:
    """Require and summarize measured active bytes without token-ratio substitution."""

    completed = [row for row in rows if row.status is ResultStatus.COMPLETED]
    if not completed:
        raise ValueError("cache operating point requires completed rows")
    identities = {(row.method, row.retention_ratio) for row in completed}
    if len(identities) != 1:
        raise ValueError("cache operating point cannot mix methods or nominal retention ratios")
    if any(row.active_kv_bytes is None for row in completed):
        missing = [row.sample_id for row in completed if row.active_kv_bytes is None]
        raise ValueError(f"paper-native rows lack measured active_kv_bytes: {missing[:5]!r}")
    active = [cast("int", row.active_kv_bytes) for row in completed]
    method, ratio = next(iter(identities))
    return CacheOperatingPoint(
        method=method,
        nominal_retention_ratio=ratio,
        sample_count=len(completed),
        mean_active_kv_bytes=math.fsum(active) / len(active),
        minimum_active_kv_bytes=min(active),
        maximum_active_kv_bytes=max(active),
    )


def validate_byte_matched_operating_points(
    left: Sequence[EvaluationResult],
    right: Sequence[EvaluationResult],
    *,
    maximum_relative_difference: float = 0.01,
) -> tuple[CacheOperatingPoint, CacheOperatingPoint]:
    """Reject a claimed byte-matched comparison when actual active bytes differ."""

    if not 0 <= maximum_relative_difference < 1:
        raise ValueError("maximum_relative_difference must lie in [0, 1)")
    left_by_id = {row.sample_id: row for row in left if row.status is ResultStatus.COMPLETED}
    right_by_id = {row.sample_id: row for row in right if row.status is ResultStatus.COMPLETED}
    if left_by_id.keys() != right_by_id.keys() or not left_by_id:
        raise ValueError("byte-matched methods require identical non-empty sample IDs")
    for sample_id in sorted(left_by_id):
        left_bytes = left_by_id[sample_id].active_kv_bytes
        right_bytes = right_by_id[sample_id].active_kv_bytes
        if left_bytes is None or right_bytes is None:
            raise ValueError("byte-matched comparison requires measured active_kv_bytes")
        denominator = max(left_bytes, right_bytes)
        difference = abs(left_bytes - right_bytes) / denominator if denominator else 0.0
        if difference > maximum_relative_difference:
            raise ValueError(
                f"active-byte mismatch for {sample_id!r}: {left_bytes} versus {right_bytes} "
                f"({difference:.3%} > {maximum_relative_difference:.3%})"
            )
    return cache_operating_point(tuple(left_by_id.values())), cache_operating_point(
        tuple(right_by_id.values())
    )


def _milebench_observations(
    rows: Sequence[EvaluationResult],
    samples: Mapping[str, MaterializedSample],
    *,
    manifest_path: str,
    evaluator_commit: str,
    ledger_sha: str,
) -> tuple[MetricObservation, ...]:
    observations: list[MetricObservation] = []
    scores: list[MileBenchSampleScore] = []
    for row in rows:
        if row.status is not ResultStatus.COMPLETED or row.task_score is None:
            continue
        sample = samples[row.sample_id]
        dataset = cast("str", sample.milebench_dataset)
        question_type = cast("MileBenchQuestionType", sample.milebench_question_type)
        route = route_milebench_dataset(dataset, question_type)
        definition = default_metric_registry().resolve(route.metric.value)
        metadata = json.dumps(
            {"dataset": dataset, "question_type": question_type.value},
            sort_keys=True,
            separators=(",", ":"),
        )
        observations.append(
            MetricObservation(
                run_id=row.run_id,
                sample_id=row.sample_id,
                task=row.task,
                model=row.model,
                backend=row.backend,
                method=row.method,
                retention_ratio=row.retention_ratio,
                metric_name=definition.name,
                metric_value=row.task_score,
                metric_direction=definition.direction,
                metric_scope=MetricScope.SAMPLE,
                metric_implementation="lookm_official_score.py@ecf0f51",
                metric_version="milebench-2024-pinned-v1",
                metric_unit=definition.unit,
                measurement_type="baseline_paper_native_measured",
                manifest_path=manifest_path,
                metadata_json=metadata,
                evaluator_commit=evaluator_commit,
                reference_artifact_sha=ledger_sha,
            )
        )
        scores.append(
            MileBenchSampleScore(
                sample_id=row.sample_id,
                dataset=dataset,
                question_type=question_type,
                metric=route.metric,
                value=row.task_score,
            )
        )
    if not scores:
        raise ValueError("MileBench produced no completed sample scores")
    first = next(row for row in rows if row.status is ResultStatus.COMPLETED)
    for hierarchy_key, value in aggregate_milebench_samples(scores).items():
        metric_suffix = hierarchy_key.rsplit("/", maxsplit=1)[-1]
        metric = MileBenchMetric(metric_suffix)
        observations.append(
            MetricObservation(
                run_id=first.run_id,
                sample_id=None,
                task=first.task,
                model=first.model,
                backend=first.backend,
                method=first.method,
                retention_ratio=first.retention_ratio,
                metric_name=f"milebench/{hierarchy_key}",
                metric_value=value,
                metric_direction=MetricDirection.HIGHER,
                metric_scope=MetricScope.CORPUS,
                metric_implementation="lookm_official_equal_dataset_hierarchy@ecf0f51",
                metric_version="milebench-2024-pinned-v1",
                metric_unit=default_metric_registry().resolve(metric.value).unit,
                measurement_type="baseline_paper_native_measured",
                manifest_path=manifest_path,
                metadata_json=json.dumps(
                    {"hierarchy_key": hierarchy_key, "base_metric": metric.value},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                evaluator_commit=evaluator_commit,
                reference_artifact_sha=ledger_sha,
                sample_count=len(scores),
            )
        )
    return tuple(observations)


def _write_metric_outputs(
    rows: Sequence[MetricObservation],
    *,
    metric_output: str | Path,
    metric_observations_parquet_output: str | Path,
    metric_aggregates_parquet_output: str | Path,
) -> tuple[str, str, str]:
    store = JsonlMetricStore(metric_output)
    for row in rows:
        store.append(row)
    observation_path, aggregate_path = write_metric_parquet_tables(
        rows,
        observations_output=metric_observations_parquet_output,
        aggregates_output=metric_aggregates_parquet_output,
    )
    return (
        str(Path(metric_output).resolve()),
        str(observation_path.resolve()),
        str(aggregate_path.resolve()),
    )


def run_materialized_paper_native_evaluation(
    *,
    run_id: str,
    task: TaskSpec,
    ledger: PaperNativeLedger,
    model: LocalEvaluationModel,
    raw_output: str | Path,
    parquet_output: str | Path,
    manifest_path: str,
    metric_output: str | Path,
    metric_observations_parquet_output: str | Path,
    metric_aggregates_parquet_output: str | Path,
    seed: int,
    subset_size: int | None,
    evaluator_commit: str,
    model_revision: str,
    tokenizer_revision: str,
    generation_config_sha256: str,
    fullkv_reference_input: str | Path | None = None,
    fullkv_reference_sha256: str | None = None,
    fullkv_reference_output: str | Path | None = None,
) -> JsonObject:
    """Run PrefixKV/MileBench native metrics and preserve all failed samples."""

    if ledger.task != task.name:
        raise ValueError("ledger/task mismatch")
    _validate_method(task.name, model.method)
    selected_ids = deterministic_sample_ids(
        tuple(sample.sample_id for sample in ledger.samples),
        seed=seed,
        subset_size=subset_size,
    )
    sample_by_id = {sample.sample_id: sample for sample in ledger.samples}
    store = JsonlResultStore(raw_output)
    prior = store.completed_sample_ids(run_id)
    requests: dict[str, EvaluationRequest] = {}
    for sample_id in selected_ids:
        sample = sample_by_id[sample_id]
        request = EvaluationRequest(
            run_id=run_id,
            sample_id=sample_id,
            task=task.name,
            messages=build_multimodal_messages(
                sample.prompt,
                tuple(MediaItem(item.kind, _media_payload(item)) for item in sample.media),
            ),
            generation_kwargs={},
        )
        requests[sample_id] = request
        if sample_id in prior:
            continue
        started = time.perf_counter()
        try:
            generation = model.generate(request)
            metrics = generation.metrics
            if metrics.end_to_end_time is None:
                metrics = replace(
                    metrics,
                    end_to_end_time=max(0.0, time.perf_counter() - started),
                )
            task_score: float | None = None
            if task.name == "milebench":
                route = route_milebench_dataset(
                    cast("str", sample.milebench_dataset),
                    cast("MileBenchQuestionType", sample.milebench_question_type),
                )
                task_score = score_milebench_sample(
                    route,
                    generation.answer,
                    sample.references[0],
                    choices=sample.choices,
                )
            row = EvaluationResult.from_generation(
                run_id=run_id,
                sample_id=sample_id,
                task=task.name,
                model=model.model_id,
                backend=model.backend,
                method=generation.effective_method or model.method,
                retention_ratio=(
                    model.retention_ratio
                    if generation.effective_retention_ratio is None
                    else generation.effective_retention_ratio
                ),
                answer=generation.answer,
                reference=sample.canonical_reference(),
                task_score=task_score,
                metrics=metrics,
                manifest_path=manifest_path,
            )
        except Exception as error:
            row = EvaluationResult.failed(
                run_id=run_id,
                sample_id=sample_id,
                task=task.name,
                model=model.model_id,
                backend=model.backend,
                method=model.method,
                retention_ratio=model.retention_ratio,
                reference=sample.canonical_reference(),
                error=f"{type(error).__name__}: {error}",
                manifest_path=manifest_path,
                metrics=GenerationMetrics(end_to_end_time=max(0.0, time.perf_counter() - started)),
            )
        store.append(row)

    selected_rows = tuple(
        row for row in store.results(run_id=run_id) if row.sample_id in set(selected_ids)
    )
    write_parquet_aggregate(selected_rows, parquet_output)
    failed = [row for row in selected_rows if row.status is ResultStatus.FAILED]
    reference_artifact_digest: str | None = None
    if failed:
        metric_rows: tuple[MetricObservation, ...] = ()
        operating_point = None
        metric_error = f"{len(failed)} sample(s) failed; native metrics were not aggregated"
    else:
        if {row.method for row in selected_rows} != {model.method}:
            raise ValueError("paper-native run changed effective method or used a fallback")
        operating_point = cache_operating_point(selected_rows)
        if task.name in _PREFIXKV_TASKS:
            if not callable(getattr(model, "teacher_forced_nll", None)):
                raise TypeError("PrefixKV paper-native metrics require teacher_forced_nll")
            reference = None
            if model.method == "full_kv":
                if fullkv_reference_input is not None or fullkv_reference_sha256 is not None:
                    raise ValueError("FullKV reference creation cannot also consume a reference")
                if fullkv_reference_output is None:
                    raise ValueError("FullKV PrefixKV lane requires --fullkv-reference-output")
                reference = build_fullkv_reference_artifact(
                    selected_rows,
                    model_revision=model_revision,
                    tokenizer_revision=tokenizer_revision,
                    generation_config_sha256=generation_config_sha256,
                )
                write_fullkv_reference_artifact(reference, fullkv_reference_output)
            else:
                if fullkv_reference_input is None or fullkv_reference_sha256 is None:
                    raise ValueError(
                        "compressed PrefixKV lane requires a hash-bound FullKV reference input"
                    )
                observed_reference_sha = sha256_bytes(Path(fullkv_reference_input).read_bytes())
                if observed_reference_sha != fullkv_reference_sha256:
                    raise ValueError(
                        "FullKV reference file SHA mismatch: "
                        f"expected {fullkv_reference_sha256}, observed {observed_reference_sha}"
                    )
                reference = load_fullkv_reference_artifact(fullkv_reference_input)
                expected_reference_identity = (
                    model.model_id,
                    model_revision,
                    tokenizer_revision,
                    generation_config_sha256,
                )
                observed_reference_identity = (
                    reference.model_id,
                    reference.model_revision,
                    reference.tokenizer_revision,
                    reference.generation_config_sha256,
                )
                if observed_reference_identity != expected_reference_identity:
                    raise ValueError(
                        "FullKV reference model/tokenizer/generation identity does not match "
                        "the compressed run"
                    )
            targets = {
                sample_id: cast("str", sample_by_id[sample_id].teacher_forcing_target)
                for sample_id in selected_ids
            }
            metric_rows = evaluate_prefixkv_native_metrics(
                cast("TeacherForcedEvaluationModel", model),
                tuple(requests[sample_id] for sample_id in selected_ids),
                targets,
                selected_rows,
                reference,
                manifest_path=manifest_path,
                evaluator_commit=evaluator_commit,
            )
            reference_artifact_digest = artifact_sha256(reference)
        else:
            metric_rows = _milebench_observations(
                selected_rows,
                sample_by_id,
                manifest_path=manifest_path,
                evaluator_commit=evaluator_commit,
                ledger_sha=ledger.sha256,
            )
        _write_metric_outputs(
            metric_rows,
            metric_output=metric_output,
            metric_observations_parquet_output=metric_observations_parquet_output,
            metric_aggregates_parquet_output=metric_aggregates_parquet_output,
        )
        metric_error = None
    return cast(
        "JsonObject",
        {
            "run_id": run_id,
            "tasks": [task.name],
            "selected_samples": len(selected_ids),
            "resumed_samples": len(set(selected_ids) & prior),
            "failed_samples": len(failed),
            "raw_output": str(Path(raw_output).resolve()),
            "parquet_output": str(Path(parquet_output).resolve()),
            "metric_output": str(Path(metric_output).resolve()) if not failed else None,
            "metric_observations_parquet_output": (
                str(Path(metric_observations_parquet_output).resolve()) if not failed else None
            ),
            "metric_aggregates_parquet_output": (
                str(Path(metric_aggregates_parquet_output).resolve()) if not failed else None
            ),
            "metric_error": metric_error,
            "paper_native_ledger_sha256": ledger.sha256,
            "cache_operating_point": (
                None if operating_point is None else operating_point.to_json_object()
            ),
            "fullkv_reference_artifact_sha256": reference_artifact_digest,
        },
    )


def resolve_protocol(
    suite: PublishedProtocolSuite,
    protocol_id: str,
) -> PublishedBaselineProtocol:
    matches = [protocol for protocol in suite.protocols if protocol.protocol_id == protocol_id]
    if len(matches) != 1:
        raise LookupError(f"unknown or duplicate published protocol {protocol_id!r}")
    return matches[0]


def validate_protocol_run(
    protocol: PublishedBaselineProtocol,
    *,
    task: TaskSpec,
    model_id: str,
    model_revision: str,
    method: str,
    retention_ratio: float,
) -> None:
    """Validate one requested run against a versioned reproduction contract."""

    if not protocol.execution_authorized:
        raise ValueError(f"protocol {protocol.protocol_id!r} is not execution-authorized")
    if (model_id, model_revision) != (protocol.model, protocol.model_revision):
        raise ValueError("model ID/revision does not match the published protocol")
    task_contract = next((item for item in protocol.tasks if item.name == task.name), None)
    if task_contract is None:
        raise ValueError(f"task {task.name!r} is not part of protocol {protocol.protocol_id!r}")
    if (task.dataset_id, task.split) != (task_contract.dataset, task_contract.split):
        raise ValueError("task registry dataset/split does not match the published protocol")
    method_contract = next((item for item in protocol.methods if item.name == method), None)
    if method_contract is None:
        raise ValueError(f"method {method!r} is not declared by protocol {protocol.protocol_id!r}")
    if not method_contract.execution_enabled:
        raise ValueError(
            f"method {method!r} is explicitly unsupported/disabled in {protocol.protocol_id!r}"
        )
    if retention_ratio not in protocol.retention_ratios:
        raise ValueError(
            f"retention ratio {retention_ratio} is not declared by {protocol.protocol_id!r}"
        )


__all__ = [
    "CacheOperatingPoint",
    "MaterializedMedia",
    "MaterializedSample",
    "PaperNativeLedger",
    "cache_operating_point",
    "load_paper_native_ledger",
    "resolve_protocol",
    "run_materialized_paper_native_evaluation",
    "validate_byte_matched_operating_points",
    "validate_protocol_run",
]
