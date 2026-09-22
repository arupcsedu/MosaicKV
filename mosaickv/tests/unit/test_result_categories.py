from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from mosaickv.config import synthetic_smoke_config
from mosaickv.evaluation.metrics.comparison import compare_paired_methods
from mosaickv.evaluation.metrics.types import (
    MetricDirection,
    MetricObservation,
    MetricScope,
)
from mosaickv.evaluation.model import GenerationMetrics
from mosaickv.evaluation.result_categories import (
    LocallyReproducedResult,
    PaperReportedResult,
    ReproductionComparison,
    validate_unified_comparison_manifests,
    write_reproduction_comparisons,
)
from mosaickv.evaluation.results import EvaluationResult
from mosaickv.manifest import (
    ArtifactProvenance,
    EvaluationProvenance,
    InputProvenance,
    RunManifestWriter,
    sha256_text,
)
from mosaickv.types import MeasurementType, ResultCategory, ResultOrigin


def _metric(
    method: str,
    *,
    category: ResultCategory = ResultCategory.UNIFIED_COMPARISON,
) -> MetricObservation:
    origin = (
        ResultOrigin.LOCALLY_REPRODUCED
        if category is ResultCategory.PAPER_REPRODUCTION
        else ResultOrigin.LOCALLY_MEASURED
    )
    return MetricObservation(
        run_id=f"run-{method}",
        sample_id="sample-1",
        task="task",
        model="model",
        backend="hf",
        method=method,
        retention_ratio=0.5,
        metric_name="accuracy",
        metric_value=1.0,
        metric_direction=MetricDirection.HIGHER,
        metric_scope=MetricScope.SAMPLE,
        metric_implementation="fixture",
        metric_version="1",
        metric_unit="fraction",
        measurement_type=(
            "paper_reproduction_metric_measured"
            if category is ResultCategory.PAPER_REPRODUCTION
            else "unified_comparison_metric_measured"
        ),
        manifest_path=f"{method}-manifest.json",
        result_category=category,
        result_origin=origin,
    )


def _paper() -> PaperReportedResult:
    return PaperReportedResult(
        paper_id="paper-1",
        paper_title="A paper",
        paper_source="https://example.invalid/paper",
        source_artifact_sha256="a" * 64,
        source_locator="Table 2, row method",
        method_family="method",
        method_label="paper method",
        model_id="model",
        model_revision="revision",
        dataset_id="dataset",
        dataset_revision="dataset-revision",
        dataset_split="test",
        metric_name="accuracy",
        metric_implementation="official",
        metric_version="1",
        metric_unit="percentage_points",
        metric_direction=MetricDirection.HIGHER,
        budget_convention="paper-token-ratio",
        budget_value=0.5,
        budget_unit="retention_ratio",
        supported_environment="paper environment",
        metric_value=80.0,
        preregistered_absolute_tolerance=2.5,
    )


def _local() -> LocallyReproducedResult:
    return LocallyReproducedResult(
        protocol_id="protocol-1",
        method_family="method",
        method_label="method_reimpl",
        model_id="model",
        model_revision="revision",
        dataset_id="dataset",
        dataset_revision="dataset-revision",
        dataset_split="test",
        metric_name="accuracy",
        metric_implementation="official",
        metric_version="1",
        metric_unit="percentage_points",
        metric_direction=MetricDirection.HIGHER,
        budget_convention="paper-token-ratio",
        budget_value=0.5,
        budget_unit="retention_ratio",
        supported_environment="paper environment",
        metric_value=78.0,
        sample_count=10,
        attempted_sample_count=11,
        failed_sample_count=1,
        run_ids=("run-1",),
        manifest_paths=("manifest.json",),
        manifest_sha256=("b" * 64,),
        fullkv_parity_artifact_path="parity.json",
        fullkv_parity_artifact_sha256="f" * 64,
    )


def test_measured_stream_rejects_paper_reported_values_and_category_mixing() -> None:
    with pytest.raises(ValueError, match="PaperReportedResult"):
        replace(_metric("method"), result_origin=ResultOrigin.PAPER_REPORTED)
    with pytest.raises(ValueError, match="require origin"):
        replace(
            _metric("method"),
            result_category=ResultCategory.PAPER_REPRODUCTION,
        )
    with pytest.raises(ValueError, match="category, or origin"):
        compare_paired_methods(
            (_metric("reference"),),
            (_metric("candidate", category=ResultCategory.PAPER_REPRODUCTION),),
            resamples=10,
        )


def test_execution_stream_rejects_paper_reported_values() -> None:
    with pytest.raises(ValueError, match="paper-reported"):
        EvaluationResult.from_generation(
            run_id="run",
            sample_id="sample",
            task="task",
            model="model",
            backend="hf",
            method="method",
            retention_ratio=0.5,
            answer="answer",
            reference="answer",
            task_score=1.0,
            metrics=GenerationMetrics(),
            manifest_path="manifest.json",
            result_origin=ResultOrigin.PAPER_REPORTED,
        )


def test_legacy_rows_gain_backward_compatible_category_defaults() -> None:
    unified_payload = _metric("method").to_json_object()
    unified_payload.pop("result_category")
    unified_payload.pop("result_origin")
    unified = MetricObservation.from_json_object(unified_payload)
    assert unified.result_category is ResultCategory.UNIFIED_COMPARISON
    assert unified.result_origin is ResultOrigin.LOCALLY_MEASURED

    paper_payload = _metric("method", category=ResultCategory.PAPER_REPRODUCTION).to_json_object()
    paper_payload.pop("result_category")
    paper_payload.pop("result_origin")
    paper = MetricObservation.from_json_object(paper_payload)
    assert paper.result_category is ResultCategory.PAPER_REPRODUCTION
    assert paper.result_origin is ResultOrigin.LOCALLY_REPRODUCED

    result = EvaluationResult.from_generation(
        run_id="legacy",
        sample_id="sample",
        task="task",
        model="model",
        backend="hf",
        method="method",
        retention_ratio=1.0,
        answer="answer",
        reference="answer",
        task_score=1.0,
        metrics=GenerationMetrics(),
        manifest_path="manifest.json",
    ).to_json_object()
    result.pop("result_category")
    result.pop("result_origin")
    restored = EvaluationResult.from_json_object(result)
    assert restored.result_category is ResultCategory.UNIFIED_COMPARISON
    assert restored.result_origin is ResultOrigin.LOCALLY_MEASURED


def test_reproduction_comparison_keeps_values_nested_and_records_delta(
    tmp_path: Path,
) -> None:
    paper = _paper()
    parity_path = tmp_path / "parity.json"
    parity_path.write_text(
        json.dumps(
            {
                "git_sha": "a" * 40,
                "model_id": "model",
                "model_revision": "revision",
                "cache_budget": {"retention_ratio": 1.0, "unit": "retained_slots"},
                "logit_absolute_tolerance": 1e-4,
                "generate_reference": {
                    "generated_tokens": 16,
                    "token_agreement": 1.0,
                    "maximum_logit_difference": 0.0,
                },
                "retention_1_reinjection": {
                    "generated_tokens": 16,
                    "token_agreement": 1.0,
                    "maximum_logit_difference": 0.0,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "result_category": "paper_reproduction",
                "result_origin": "locally_reproduced",
                "source": {
                    "git_sha": "a" * 40,
                    "git_dirty": False,
                    "canonical_eligible": True,
                },
                "model": {"id": "model", "revision": "revision"},
                "dataset": {
                    "id": "dataset",
                    "revision": "dataset-revision",
                    "split": "test",
                },
                "resolved_config": {"method": "method_reimpl"},
                "evaluation": {
                    "evaluator_name": "official evaluator",
                    "evaluator_version": "1",
                    "evaluator_commit": "c" * 40,
                    "reference_output_sha256": "d" * 64,
                    "baseline_name": "method_reimpl",
                    "baseline_commit": "e" * 40,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    local = replace(
        _local(),
        manifest_paths=(str(manifest_path),),
        manifest_sha256=(hashlib.sha256(manifest_path.read_bytes()).hexdigest(),),
        fullkv_parity_artifact_path=str(parity_path),
        fullkv_parity_artifact_sha256=hashlib.sha256(parity_path.read_bytes()).hexdigest(),
    )
    comparison = ReproductionComparison.build(paper, local)
    assert comparison.reproduction_delta == -2.0
    assert comparison.relative_reproduction_delta == -0.025
    assert comparison.within_preregistered_tolerance is True
    output = write_reproduction_comparisons((comparison,), tmp_path / "comparison.json")
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["columns"] == [
        "paper_reported",
        "locally_reproduced",
        "reproduction_delta",
        "preregistered_absolute_tolerance",
        "within_preregistered_tolerance",
    ]
    row = payload["comparisons"][0]
    assert row["paper_reported"]["result_origin"] == "paper_reported"
    assert row["locally_reproduced"]["result_origin"] == "locally_reproduced"
    assert row["reproduction_delta"] == -2.0
    assert row["within_preregistered_tolerance"] is True
    assert row["locally_reproduced"]["attempted_sample_count"] == 11
    assert row["locally_reproduced"]["failed_sample_count"] == 1
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_reproduction_comparisons((comparison,), output)
    bad_hash = ReproductionComparison.build(
        paper,
        replace(local, manifest_sha256=("0" * 64,)),
    )
    with pytest.raises(ValueError, match="manifest SHA mismatch"):
        write_reproduction_comparisons((bad_hash,), tmp_path / "bad.json")
    with pytest.raises(ValueError, match="identities differ"):
        ReproductionComparison.build(paper, replace(local, dataset_split="validation"))

    bad_parity = replace(local, fullkv_parity_artifact_sha256="0" * 64)
    with pytest.raises(ValueError, match="parity artifact SHA mismatch"):
        write_reproduction_comparisons(
            (ReproductionComparison.build(paper, bad_parity),),
            tmp_path / "bad-parity.json",
        )


def test_reproduction_gate_fails_outside_preregistered_tolerance() -> None:
    comparison = ReproductionComparison.build(
        replace(_paper(), preregistered_absolute_tolerance=1.0),
        _local(),
    )
    assert comparison.reproduction_delta == -2.0
    assert comparison.within_preregistered_tolerance is False


def test_reproduction_result_json_round_trip() -> None:
    assert PaperReportedResult.from_json_object(_paper().to_json_object()) == _paper()
    assert LocallyReproducedResult.from_json_object(_local().to_json_object()) == _local()


def test_reproduction_denominator_must_include_failures() -> None:
    with pytest.raises(ValueError, match="sample_count plus failed_sample_count"):
        replace(_local(), attempted_sample_count=10)


def _write_manifest(
    path: Path,
    *,
    category: ResultCategory = ResultCategory.UNIFIED_COMPARISON,
    tokenization_sha: str | None = None,
) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    inputs = InputProvenance(
        prompt_set_sha=sha256_text("prompts"),
        media_set_sha=sha256_text("media"),
        preprocessing_sha=sha256_text("preprocessing"),
        tokenization_sha=tokenization_sha or sha256_text("tokenization"),
    )
    RunManifestWriter(repo_root).write(
        path,
        synthetic_smoke_config(),
        MeasurementType.METHOD,
        inputs,
        ArtifactProvenance(
            raw_output_sha=sha256_text(path.name),
            metrics_sha="not_applicable",
            log_sha=sha256_text(f"log-{path.name}"),
        ),
        result_category=category,
        evaluation_provenance=(
            EvaluationProvenance(
                evaluator_name="fixture",
                evaluator_version="1",
                evaluator_commit="a" * 40,
                reference_output_sha256="b" * 64,
                baseline_name="method_reimpl",
                baseline_commit="c" * 40,
            )
            if category is ResultCategory.PAPER_REPRODUCTION
            else None
        ),
    )


def test_unified_manifest_validation_enforces_identical_controls(tmp_path: Path) -> None:
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    _write_manifest(left)
    _write_manifest(right)
    result = validate_unified_comparison_manifests((left, right))
    assert result["status"] == "comparable"
    assert len(result["control_sha256"]) == 64

    different = tmp_path / "different.json"
    _write_manifest(different, tokenization_sha=sha256_text("different"))
    with pytest.raises(ValueError, match=r"inputs\.tokenization_sha"):
        validate_unified_comparison_manifests((left, different))

    paper = tmp_path / "paper.json"
    _write_manifest(paper, category=ResultCategory.PAPER_REPRODUCTION)
    with pytest.raises(ValueError, match="non-unified"):
        validate_unified_comparison_manifests((left, paper))
