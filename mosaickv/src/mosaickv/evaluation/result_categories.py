"""Fail-closed separation of paper reproductions and unified comparisons."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from mosaickv.evaluation.metrics.types import MetricDirection, MetricObservation
from mosaickv.types import JsonObject, ResultCategory, ResultOrigin


def _required(value: str, field: str) -> None:
    if not value.strip():
        raise ValueError(f"{field} must be non-empty")


def _digest(value: str, field: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class PaperReportedResult:
    """A number transcribed from a paper, never represented as a measured run."""

    paper_id: str
    paper_title: str
    paper_source: str
    source_artifact_sha256: str
    source_locator: str
    method_family: str
    method_label: str
    model_id: str
    model_revision: str
    dataset_id: str
    dataset_revision: str
    dataset_split: str
    metric_name: str
    metric_implementation: str
    metric_version: str
    metric_unit: str
    metric_direction: MetricDirection
    budget_convention: str
    budget_value: float
    budget_unit: str
    supported_environment: str
    metric_value: float
    preregistered_absolute_tolerance: float
    result_category: ResultCategory = ResultCategory.PAPER_REPRODUCTION
    result_origin: ResultOrigin = ResultOrigin.PAPER_REPORTED
    schema_version: int = 1

    def __post_init__(self) -> None:
        for field in (
            "paper_id",
            "paper_title",
            "paper_source",
            "source_locator",
            "method_family",
            "method_label",
            "model_id",
            "model_revision",
            "dataset_id",
            "dataset_revision",
            "dataset_split",
            "metric_name",
            "metric_implementation",
            "metric_version",
            "metric_unit",
            "budget_convention",
            "budget_unit",
            "supported_environment",
        ):
            _required(str(getattr(self, field)), field)
        _digest(self.source_artifact_sha256, "source_artifact_sha256")
        if self.result_category is not ResultCategory.PAPER_REPRODUCTION:
            raise ValueError("paper-reported results must use category=paper_reproduction")
        if self.result_origin is not ResultOrigin.PAPER_REPORTED:
            raise ValueError("PaperReportedResult must use origin=paper_reported")
        for field in ("budget_value", "metric_value", "preregistered_absolute_tolerance"):
            value = getattr(self, field)
            if isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"{field} must be finite")
        if self.budget_value <= 0:
            raise ValueError("budget_value must be positive")
        if self.preregistered_absolute_tolerance < 0:
            raise ValueError("preregistered_absolute_tolerance cannot be negative")
        if self.schema_version != 1:
            raise ValueError("schema_version must be 1")

    def to_json_object(self) -> JsonObject:
        payload = asdict(self)
        payload["metric_direction"] = self.metric_direction.value
        payload["result_category"] = self.result_category.value
        payload["result_origin"] = self.result_origin.value
        return cast("JsonObject", payload)

    @classmethod
    def from_json_object(cls, payload: Mapping[str, Any]) -> PaperReportedResult:
        data = dict(payload)
        data["metric_direction"] = MetricDirection(str(data["metric_direction"]))
        data["result_category"] = ResultCategory(
            str(data.get("result_category", ResultCategory.PAPER_REPRODUCTION.value))
        )
        data["result_origin"] = ResultOrigin(
            str(data.get("result_origin", ResultOrigin.PAPER_REPORTED.value))
        )
        return cls(**data)


@dataclass(frozen=True, slots=True)
class LocallyReproducedResult:
    """A locally measured reproduction reduced from immutable run artifacts."""

    protocol_id: str
    method_family: str
    method_label: str
    model_id: str
    model_revision: str
    dataset_id: str
    dataset_revision: str
    dataset_split: str
    metric_name: str
    metric_implementation: str
    metric_version: str
    metric_unit: str
    metric_direction: MetricDirection
    budget_convention: str
    budget_value: float
    budget_unit: str
    supported_environment: str
    metric_value: float
    sample_count: int
    attempted_sample_count: int
    failed_sample_count: int
    run_ids: tuple[str, ...]
    manifest_paths: tuple[str, ...]
    manifest_sha256: tuple[str, ...]
    fullkv_parity_artifact_path: str
    fullkv_parity_artifact_sha256: str
    result_category: ResultCategory = ResultCategory.PAPER_REPRODUCTION
    result_origin: ResultOrigin = ResultOrigin.LOCALLY_REPRODUCED
    schema_version: int = 1

    def __post_init__(self) -> None:
        for field in (
            "protocol_id",
            "method_family",
            "method_label",
            "model_id",
            "model_revision",
            "dataset_id",
            "dataset_revision",
            "dataset_split",
            "metric_name",
            "metric_implementation",
            "metric_version",
            "metric_unit",
            "budget_convention",
            "budget_unit",
            "supported_environment",
        ):
            _required(str(getattr(self, field)), field)
        if self.result_category is not ResultCategory.PAPER_REPRODUCTION:
            raise ValueError("local reproductions must use category=paper_reproduction")
        if self.result_origin is not ResultOrigin.LOCALLY_REPRODUCED:
            raise ValueError("local reproductions must use origin=locally_reproduced")
        if self.sample_count < 1:
            raise ValueError("sample_count must be positive")
        if self.attempted_sample_count < 1:
            raise ValueError("attempted_sample_count must be positive")
        if self.failed_sample_count < 0:
            raise ValueError("failed_sample_count cannot be negative")
        if self.sample_count + self.failed_sample_count != self.attempted_sample_count:
            raise ValueError(
                "attempted_sample_count must equal sample_count plus failed_sample_count"
            )
        if not self.run_ids or len(set(self.run_ids)) != len(self.run_ids):
            raise ValueError("run_ids must be non-empty and unique")
        if not (len(self.run_ids) == len(self.manifest_paths) == len(self.manifest_sha256)):
            raise ValueError("run IDs, manifest paths, and manifest hashes must align")
        for digest in self.manifest_sha256:
            _digest(digest, "manifest_sha256")
        _required(self.fullkv_parity_artifact_path, "fullkv_parity_artifact_path")
        _digest(self.fullkv_parity_artifact_sha256, "fullkv_parity_artifact_sha256")
        for field in ("budget_value", "metric_value"):
            value = getattr(self, field)
            if isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"{field} must be finite")
        if self.budget_value <= 0:
            raise ValueError("budget_value must be positive")
        if self.schema_version != 1:
            raise ValueError("schema_version must be 1")

    def to_json_object(self) -> JsonObject:
        payload = asdict(self)
        payload["metric_direction"] = self.metric_direction.value
        payload["result_category"] = self.result_category.value
        payload["result_origin"] = self.result_origin.value
        return cast("JsonObject", payload)

    @classmethod
    def from_json_object(cls, payload: Mapping[str, Any]) -> LocallyReproducedResult:
        data = dict(payload)
        data["metric_direction"] = MetricDirection(str(data["metric_direction"]))
        data["run_ids"] = tuple(data["run_ids"])
        data["manifest_paths"] = tuple(data["manifest_paths"])
        data["manifest_sha256"] = tuple(data["manifest_sha256"])
        data["result_category"] = ResultCategory(
            str(data.get("result_category", ResultCategory.PAPER_REPRODUCTION.value))
        )
        data["result_origin"] = ResultOrigin(
            str(data.get("result_origin", ResultOrigin.LOCALLY_REPRODUCED.value))
        )
        return cls(**data)


@dataclass(frozen=True, slots=True)
class ReproductionComparison:
    """Nested paper-versus-local values with an explicit reproduction delta."""

    paper_reported: PaperReportedResult
    locally_reproduced: LocallyReproducedResult
    reproduction_delta: float
    relative_reproduction_delta: float | None
    within_preregistered_tolerance: bool
    schema_version: int = 1

    def __post_init__(self) -> None:
        identity_fields = (
            "method_family",
            "model_id",
            "model_revision",
            "dataset_id",
            "dataset_revision",
            "dataset_split",
            "metric_name",
            "metric_implementation",
            "metric_version",
            "metric_unit",
            "metric_direction",
            "budget_convention",
            "budget_value",
            "budget_unit",
            "supported_environment",
        )
        mismatches = [
            field
            for field in identity_fields
            if getattr(self.paper_reported, field) != getattr(self.locally_reproduced, field)
        ]
        if mismatches:
            raise ValueError(
                "paper-reported and locally reproduced identities differ: " + ", ".join(mismatches)
            )
        expected = self.locally_reproduced.metric_value - self.paper_reported.metric_value
        if not math.isclose(self.reproduction_delta, expected, rel_tol=0, abs_tol=1e-12):
            raise ValueError("reproduction_delta must equal local minus paper-reported value")
        expected_relative = (
            None
            if self.paper_reported.metric_value == 0
            else expected / abs(self.paper_reported.metric_value)
        )
        if expected_relative is None:
            if self.relative_reproduction_delta is not None:
                raise ValueError("relative delta must be null for a zero paper value")
        elif self.relative_reproduction_delta is None or not math.isclose(
            self.relative_reproduction_delta,
            expected_relative,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError("relative reproduction delta is inconsistent")
        if self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        expected_pass = abs(expected) <= self.paper_reported.preregistered_absolute_tolerance
        if self.within_preregistered_tolerance is not expected_pass:
            raise ValueError("reproduction tolerance verdict is inconsistent")

    @classmethod
    def build(
        cls,
        paper_reported: PaperReportedResult,
        locally_reproduced: LocallyReproducedResult,
    ) -> ReproductionComparison:
        delta = locally_reproduced.metric_value - paper_reported.metric_value
        relative = (
            None if paper_reported.metric_value == 0 else delta / abs(paper_reported.metric_value)
        )
        return cls(
            paper_reported,
            locally_reproduced,
            delta,
            relative,
            abs(delta) <= paper_reported.preregistered_absolute_tolerance,
        )

    def to_json_object(self) -> JsonObject:
        return {
            "schema_version": self.schema_version,
            "result_category": ResultCategory.PAPER_REPRODUCTION.value,
            "paper_reported": self.paper_reported.to_json_object(),
            "locally_reproduced": self.locally_reproduced.to_json_object(),
            "reproduction_delta": self.reproduction_delta,
            "relative_reproduction_delta": self.relative_reproduction_delta,
            "preregistered_absolute_tolerance": (
                self.paper_reported.preregistered_absolute_tolerance
            ),
            "within_preregistered_tolerance": self.within_preregistered_tolerance,
        }


def validate_locally_reproduced_manifests(result: LocallyReproducedResult) -> None:
    """Verify that every declared local value is backed by its paper-lane manifest."""

    parity_path = Path(result.fullkv_parity_artifact_path).resolve(strict=True)
    parity_raw = parity_path.read_bytes()
    parity_sha = hashlib.sha256(parity_raw).hexdigest()
    if parity_sha != result.fullkv_parity_artifact_sha256:
        raise ValueError(
            "FullKV parity artifact SHA mismatch: "
            f"expected {result.fullkv_parity_artifact_sha256}, observed {parity_sha}"
        )
    parity = json.loads(parity_raw)
    if not isinstance(parity, Mapping):
        raise ValueError("FullKV parity artifact must be a JSON object")
    if (parity.get("model_id"), parity.get("model_revision")) != (
        result.model_id,
        result.model_revision,
    ):
        raise ValueError("FullKV parity artifact model identity does not match reproduction")
    cache_budget = parity.get("cache_budget")
    if not isinstance(cache_budget, Mapping) or cache_budget.get("retention_ratio") != 1.0:
        raise ValueError("FullKV parity artifact must use retention ratio 1.0")
    tolerance = parity.get("logit_absolute_tolerance")
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(float(tolerance))
        or tolerance < 0
    ):
        raise ValueError("FullKV parity artifact lacks a valid numerical tolerance")
    for field in ("generate_reference", "retention_1_reinjection"):
        comparison = parity.get(field)
        if not isinstance(comparison, Mapping):
            raise ValueError(f"FullKV parity artifact lacks {field}")
        generated_tokens = comparison.get("generated_tokens")
        if (
            isinstance(generated_tokens, bool)
            or not isinstance(generated_tokens, int)
            or generated_tokens < 16
        ):
            raise ValueError(f"FullKV parity artifact {field} must compare at least 16 tokens")
        if comparison.get("token_agreement") != 1.0:
            raise ValueError(f"FullKV parity artifact {field} lacks exact token agreement")
        maximum = comparison.get("maximum_logit_difference")
        if (
            isinstance(maximum, bool)
            or not isinstance(maximum, (int, float))
            or not math.isfinite(float(maximum))
            or maximum > tolerance
        ):
            raise ValueError(f"FullKV parity artifact {field} exceeds its logit tolerance")

    expected_identity = (
        result.model_id,
        result.model_revision,
        result.dataset_id,
        result.dataset_revision,
        result.dataset_split,
    )
    for run_id, path, expected_sha in zip(
        result.run_ids,
        result.manifest_paths,
        result.manifest_sha256,
        strict=True,
    ):
        manifest, observed_sha = load_manifest(path)
        if observed_sha != expected_sha:
            raise ValueError(
                f"local reproduction manifest SHA mismatch for {path}: "
                f"expected {expected_sha}, observed {observed_sha}"
            )
        if manifest.get("run_id") != run_id:
            raise ValueError(f"local reproduction run ID does not match manifest {path}")
        if manifest.get("result_category") != ResultCategory.PAPER_REPRODUCTION.value:
            raise ValueError(f"local reproduction manifest has the wrong category: {path}")
        if manifest.get("result_origin") != ResultOrigin.LOCALLY_REPRODUCED.value:
            raise ValueError(f"local reproduction manifest has the wrong origin: {path}")
        raw_source = manifest.get("source")
        if not isinstance(raw_source, Mapping):
            raise ValueError(f"local reproduction manifest lacks source provenance: {path}")
        source = cast("Mapping[str, Any]", raw_source)
        if source.get("git_dirty") is not False or source.get("canonical_eligible") is not True:
            raise ValueError(f"local reproduction manifest is not from a clean commit: {path}")
        if parity.get("git_sha") != source.get("git_sha"):
            raise ValueError(f"FullKV parity and reproduction use different source commits: {path}")
        raw_model = manifest.get("model")
        raw_dataset = manifest.get("dataset")
        resolved = manifest.get("resolved_config")
        if (
            not isinstance(raw_model, Mapping)
            or not isinstance(raw_dataset, Mapping)
            or not isinstance(resolved, Mapping)
        ):
            raise ValueError(
                f"local reproduction manifest lacks model/dataset/method identity: {path}"
            )
        model = cast("Mapping[str, Any]", raw_model)
        dataset = cast("Mapping[str, Any]", raw_dataset)
        observed_identity = (
            model.get("id"),
            model.get("revision"),
            dataset.get("id"),
            dataset.get("revision"),
            dataset.get("split"),
        )
        if observed_identity != expected_identity:
            raise ValueError(f"local reproduction identity does not match manifest {path}")
        if resolved.get("method") != result.method_label:
            raise ValueError(f"local reproduction method does not match manifest {path}")
        raw_evaluation = manifest.get("evaluation")
        if not isinstance(raw_evaluation, Mapping):
            raise ValueError(f"local reproduction manifest lacks evaluation provenance: {path}")
        evaluation = cast("Mapping[str, Any]", raw_evaluation)
        if evaluation.get("baseline_name") != result.method_label:
            raise ValueError(f"local reproduction baseline label does not match manifest {path}")
        for field in ("evaluator_name", "evaluator_version"):
            value = evaluation.get(field)
            if not isinstance(value, str) or value in {"", "not_used", "not_applicable"}:
                raise ValueError(f"local reproduction manifest lacks evaluation.{field}: {path}")
        evaluator_commit = evaluation.get("evaluator_commit")
        if not isinstance(evaluator_commit, str) or len(evaluator_commit) not in {40, 64}:
            raise ValueError(
                f"local reproduction manifest lacks an immutable evaluator commit: {path}"
            )
        reference_sha = evaluation.get("reference_output_sha256")
        if not isinstance(reference_sha, str) or len(reference_sha) != 64:
            raise ValueError(f"local reproduction manifest lacks a reference-output hash: {path}")
        baseline_commit = evaluation.get("baseline_commit")
        if (
            not isinstance(baseline_commit, str)
            or len(baseline_commit) != 40
            or any(character not in "0123456789abcdef" for character in baseline_commit)
        ):
            raise ValueError(f"local reproduction manifest lacks a baseline commit: {path}")


def write_reproduction_comparisons(
    comparisons: Sequence[ReproductionComparison], output: str | Path
) -> Path:
    """Write a separate artifact; paper values never enter measured metric tables."""

    if not comparisons:
        raise ValueError("at least one reproduction comparison is required")
    for comparison in comparisons:
        validate_locally_reproduced_manifests(comparison.locally_reproduced)
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite reproduction comparisons: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "result_category": ResultCategory.PAPER_REPRODUCTION.value,
        "columns": [
            "paper_reported",
            "locally_reproduced",
            "reproduction_delta",
            "preregistered_absolute_tolerance",
            "within_preregistered_tolerance",
        ],
        "comparisons": [comparison.to_json_object() for comparison in comparisons],
    }
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=destination.parent, delete=False
        ) as handle:
            temporary = handle.name
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
        raise
    return destination


_UNIFIED_CONTROL_PATHS = (
    "source.git_sha",
    "source.git_dirty",
    "model.id",
    "model.revision",
    "model.precision",
    "dataset.id",
    "dataset.revision",
    "dataset.split",
    "execution.backend",
    "execution.attention_implementation",
    "execution.decode_attention_implementation",
    "execution.seed",
    "execution.deterministic_algorithms",
    "inputs.prompt_set_sha",
    "inputs.media_set_sha",
    "inputs.preprocessing_sha",
    "inputs.tokenization_sha",
    "generation.parameters_sha",
    "generation.output_length_policy",
    "cache.budget_unit",
    "cache.accounting_spec_sha",
    "environment.lock_sha256",
    "hardware.gpu_type",
    "hardware.gpu_count",
    "software.cuda",
    "software.driver",
    "software.pytorch",
    "software.transformers",
    "software.vllm",
    "software.sglang",
    "resolved_config.execution",
    "resolved_config.cache.block_size",
    "resolved_config.cache.physical_budget_block_size",
    "resolved_config.cache.physical_budget_mandatory_policy",
    "resolved_config.cache.exact_gather_mode",
    "resolved_config.cache.packing_mode",
    "resolved_config.cache.total_storage_budget_bytes",
)

_BACKEND_EXECUTION_KEYS = (
    "engine_execution_mode",
    "cuda_graph",
    "attention_backend_configuration",
    "vllm_block_size",
    "prefix_caching",
    "model_source",
)


def _lookup(payload: Mapping[str, Any], path: str) -> Any:
    value: Any = payload
    for component in path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            raise ValueError(f"manifest lacks unified-comparison control {path!r}")
        value = value[component]
    return value


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def load_manifest(path: str | Path) -> tuple[JsonObject, str]:
    source = Path(path).resolve(strict=True)
    raw = source.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"manifest is not an object: {source}")
    return cast("JsonObject", payload), hashlib.sha256(raw).hexdigest()


def validate_unified_comparison_manifests(
    manifest_paths: Sequence[str | Path],
) -> JsonObject:
    """Require identical execution controls before unified aggregation."""

    if len(manifest_paths) < 2:
        raise ValueError("unified comparison requires at least two manifests")
    loaded = [load_manifest(path) for path in manifest_paths]
    manifests = [item[0] for item in loaded]
    for manifest in manifests:
        if manifest.get("result_category") != ResultCategory.UNIFIED_COMPARISON.value:
            raise ValueError("unified comparison received a non-unified manifest")
        if manifest.get("result_origin") != ResultOrigin.LOCALLY_MEASURED.value:
            raise ValueError("unified comparison manifests must be locally measured")
    reference = cast("Mapping[str, Any]", manifests[0])
    mismatches: list[str] = []
    for candidate in manifests[1:]:
        candidate_mapping = cast("Mapping[str, Any]", candidate)
        for path in _UNIFIED_CONTROL_PATHS:
            if _canonical(_lookup(reference, path)) != _canonical(_lookup(candidate_mapping, path)):
                mismatches.append(path)
        reference_execution = cast("Mapping[str, Any]", _lookup(reference, "execution"))
        candidate_execution = cast("Mapping[str, Any]", _lookup(candidate_mapping, "execution"))
        for key in _BACKEND_EXECUTION_KEYS:
            if _canonical(reference_execution.get(key, "not_used")) != _canonical(
                candidate_execution.get(key, "not_used")
            ):
                mismatches.append(f"execution.{key}")
    unique_mismatches = tuple(sorted(set(mismatches)))
    if unique_mismatches:
        raise ValueError("unified comparison controls differ: " + ", ".join(unique_mismatches))
    controls = {path: _lookup(reference, path) for path in _UNIFIED_CONTROL_PATHS}
    controls_sha = hashlib.sha256(_canonical(controls).encode("utf-8")).hexdigest()
    return {
        "status": "comparable",
        "result_category": ResultCategory.UNIFIED_COMPARISON.value,
        "control_sha256": controls_sha,
        "manifest_sha256": [digest for _, digest in loaded],
        "control_mismatches": [],
    }


def require_metric_category(rows: Sequence[MetricObservation], category: ResultCategory) -> None:
    if not rows:
        raise ValueError("metric cohort cannot be empty")
    observed = {(row.result_category, row.result_origin) for row in rows}
    expected_origin = (
        ResultOrigin.LOCALLY_REPRODUCED
        if category is ResultCategory.PAPER_REPRODUCTION
        else ResultOrigin.LOCALLY_MEASURED
    )
    if observed != {(category, expected_origin)}:
        labels = sorted(f"{item[0].value}/{item[1].value}" for item in observed)
        raise ValueError(
            f"metric cohort is not exclusively {category.value}/{expected_origin.value}: {labels}"
        )


__all__ = [
    "LocallyReproducedResult",
    "PaperReportedResult",
    "ReproductionComparison",
    "load_manifest",
    "require_metric_category",
    "validate_locally_reproduced_manifests",
    "validate_unified_comparison_manifests",
    "write_reproduction_comparisons",
]
