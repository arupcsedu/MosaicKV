from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from mosaickv.evaluation.messages import build_multimodal_messages
from mosaickv.evaluation.metrics.comparison import compare_paired_methods
from mosaickv.evaluation.metrics.registry import default_metric_registry
from mosaickv.evaluation.metrics.scorers import (
    aggregate_perplexity,
    anls_score,
    exact_match_score,
    rouge_l_f1,
)
from mosaickv.evaluation.metrics.statistics import paired_bootstrap
from mosaickv.evaluation.metrics.storage import (
    JsonlMetricStore,
    load_metric_jsonl,
    write_metric_parquet_tables,
)
from mosaickv.evaluation.metrics.types import (
    METRIC_COLUMNS,
    MetricDirection,
    MetricObservation,
    MetricSample,
    MetricScope,
)
from mosaickv.evaluation.milebench import (
    MileBenchDatasetScore,
    MileBenchMetric,
    aggregate_milebench,
    milebench_multichoice_accuracy,
    milebench_needle_accuracy,
    milebench_rouge_l,
)
from mosaickv.evaluation.model import EvaluationRequest, GenerationMetrics, ModelGeneration
from mosaickv.evaluation.prefixkv_metrics import (
    TeacherForcedNll,
    artifact_sha256,
    build_fullkv_reference_artifact,
    evaluate_prefixkv_native_metrics,
    perplexity_sample,
    prefixkv_metric_observations,
    rouge_samples,
)
from mosaickv.evaluation.published_protocols import load_published_protocols
from mosaickv.evaluation.results import EvaluationResult


def test_rouge_anls_and_exact_match_fixtures() -> None:
    assert rouge_l_f1("a b c", ("a x c",)) == pytest.approx(2 / 3)
    assert rouge_l_f1("", ("answer",)) == 0
    assert anls_score("invoice 42", ("invoice 43",)) == pytest.approx(0.9)
    assert anls_score("unrelated", ("x",)) == 0
    assert exact_match_score("Blue!", ("blue",)) == 1


def test_perplexity_uses_token_weighted_corpus_nll() -> None:
    samples = (
        perplexity_sample("a", "target", TeacherForcedNll(math.log(2), 1)),
        perplexity_sample("b", "target", TeacherForcedNll(3 * math.log(4), 3)),
    )
    mean_nll = (math.log(2) + 3 * math.log(4)) / 4
    assert aggregate_perplexity(samples) == pytest.approx(math.exp(mean_nll))


def test_metric_aware_bootstrap_recomputes_perplexity() -> None:
    definition = default_metric_registry().resolve("perplexity")
    reference = (
        MetricSample("a", "", ("x",), numerator=1.0, denominator=1),
        MetricSample("b", "", ("x",), numerator=2.0, denominator=2),
    )
    candidate = (
        MetricSample("a", "", ("x",), numerator=1.5, denominator=1),
        MetricSample("b", "", ("x",), numerator=2.5, denominator=2),
    )
    result = paired_bootstrap(definition, reference, candidate, resamples=100, seed=7)
    assert result.sample_count == 2
    assert result.candidate_value > result.reference_value
    assert result.bootstrap_resamples == 100


def test_metric_store_is_append_only(tmp_path: Path) -> None:
    row = MetricObservation(
        run_id="run",
        sample_id="sample",
        task="task",
        model="model",
        backend="hf",
        method="full_kv",
        retention_ratio=1.0,
        metric_name="accuracy",
        metric_value=1.0,
        metric_direction=MetricDirection.HIGHER,
        metric_scope=MetricScope.SAMPLE,
        metric_implementation="fixture",
        metric_version="1",
        metric_unit="fraction",
        measurement_type="unit_test",
        manifest_path="manifest.json",
    )
    store = JsonlMetricStore(tmp_path / "metrics.jsonl")
    assert store.append(row)
    assert not store.append(row)
    assert load_metric_jsonl(store.path) == (row,)


def test_metric_schema_contains_required_long_form_columns() -> None:
    required = {
        "run_id",
        "sample_id",
        "task",
        "metric_name",
        "metric_value",
        "metric_direction",
        "metric_scope",
        "metric_implementation",
        "metric_version",
        "evaluator_commit",
        "reference_artifact_sha",
        "evaluated_token_count",
        "measurement_type",
    }
    assert required <= set(METRIC_COLUMNS)


def test_metric_reader_accepts_legacy_sha_and_token_denominator(tmp_path: Path) -> None:
    row = MetricObservation(
        run_id="run",
        sample_id="sample",
        task="task",
        model="model",
        backend="hf",
        method="full_kv",
        retention_ratio=1.0,
        metric_name="perplexity",
        metric_value=2.0,
        metric_direction=MetricDirection.LOWER,
        metric_scope=MetricScope.SAMPLE,
        metric_implementation="fixture",
        metric_version="1",
        metric_unit="ratio",
        measurement_type="unit_test",
        manifest_path="manifest.json",
        reference_artifact_sha="a" * 64,
        evaluated_token_count=7,
        numerator=1.0,
        denominator=7,
    )
    legacy = row.to_json_object()
    legacy["reference_artifact_sha256"] = legacy.pop("reference_artifact_sha")
    legacy.pop("evaluated_token_count")
    path = tmp_path / "legacy_metrics.jsonl"
    path.write_text(json.dumps(legacy) + "\n", encoding="utf-8")

    assert load_metric_jsonl(path) == (row,)


def test_metric_parquet_tables_separate_sample_and_corpus_rows(tmp_path: Path) -> None:
    pyarrow_parquet = pytest.importorskip("pyarrow.parquet")
    sample = MetricObservation(
        run_id="run",
        sample_id="sample",
        task="task",
        model="model",
        backend="hf",
        method="full_kv",
        retention_ratio=1.0,
        metric_name="accuracy",
        metric_value=1.0,
        metric_direction=MetricDirection.HIGHER,
        metric_scope=MetricScope.SAMPLE,
        metric_implementation="fixture",
        metric_version="1",
        metric_unit="fraction",
        measurement_type="unit_test",
        manifest_path="manifest.json",
    )
    aggregate = MetricObservation(
        run_id="run",
        sample_id=None,
        task="task",
        model="model",
        backend="hf",
        method="full_kv",
        retention_ratio=1.0,
        metric_name="cider",
        metric_value=10.0,
        metric_direction=MetricDirection.HIGHER,
        metric_scope=MetricScope.CORPUS,
        metric_implementation="fixture",
        metric_version="1",
        metric_unit="score",
        measurement_type="unit_test",
        manifest_path="manifest.json",
        sample_count=1,
    )
    observations = tmp_path / "observations.parquet"
    aggregates = tmp_path / "aggregates.parquet"

    written = write_metric_parquet_tables(
        (aggregate, sample),
        observations_output=observations,
        aggregates_output=aggregates,
    )

    assert written == (observations, aggregates)
    observation_table = pyarrow_parquet.read_table(observations).to_pylist()
    aggregate_table = pyarrow_parquet.read_table(aggregates).to_pylist()
    assert [row["metric_scope"] for row in observation_table] == ["sample"]
    assert [row["metric_scope"] for row in aggregate_table] == ["corpus"]
    assert required_metric_columns() <= set(observation_table[0])


def required_metric_columns() -> set[str]:
    return {
        "run_id",
        "sample_id",
        "task",
        "metric_name",
        "metric_value",
        "metric_direction",
        "metric_scope",
        "metric_implementation",
        "metric_version",
        "evaluator_commit",
        "reference_artifact_sha",
        "evaluated_token_count",
        "measurement_type",
    }


def _fullkv_row(sample_id: str, answer: str) -> EvaluationResult:
    return EvaluationResult.from_generation(
        run_id="fullkv",
        sample_id=sample_id,
        task="mmvet",
        model="llava",
        backend="hf",
        method="full_kv",
        retention_ratio=1.0,
        answer=answer,
        reference="ground truth",
        task_score=1.0,
        metrics=GenerationMetrics(),
        manifest_path="manifest.json",
    )


def test_prefixkv_reference_artifact_is_hashable_and_scores_rouge() -> None:
    artifact = build_fullkv_reference_artifact(
        (_fullkv_row("a", "red green"), _fullkv_row("b", "blue")),
        model_revision="model-sha",
        tokenizer_revision="tokenizer-sha",
        generation_config_sha256="a" * 64,
    )
    assert len(artifact_sha256(artifact)) == 64
    samples = rouge_samples(
        (
            _fullkv_row("a", "red green"),
            _fullkv_row("b", "wrong"),
        ),
        artifact,
    )
    assert [sample.value for sample in samples] == [1.0, 0.0]


def test_milebench_scoring_and_equal_dataset_aggregation() -> None:
    assert milebench_multichoice_accuracy("B. second", "second", ("first", "second")) == 1
    assert milebench_needle_accuracy("the key is kiwi", "kiwi", token_bounded=True) == 1
    assert milebench_rouge_l("red car", "red vehicle") == pytest.approx(0.5)
    aggregate = aggregate_milebench(
        (
            MileBenchDatasetScore("a", "T-1", MileBenchMetric.ACCURACY, 0.25, 100),
            MileBenchDatasetScore("b", "T-1", MileBenchMetric.ACCURACY, 0.75, 10),
            MileBenchDatasetScore("n1", "N-1", MileBenchMetric.ACCURACY, 0.2, 5),
            MileBenchDatasetScore("n2", "N-2", MileBenchMetric.ACCURACY, 0.4, 5),
        )
    )
    assert aggregate["T-1"] == 0.5
    assert aggregate["NH"] == pytest.approx(0.3)


def test_milebench_preserves_official_empty_answer_fallback() -> None:
    choices = ("first", "second", "third")
    assert milebench_multichoice_accuracy("", "third", choices) == 1
    assert milebench_multichoice_accuracy("", "first", choices) == 0


def test_prefixkv_observations_preserve_ppl_sufficient_statistics() -> None:
    rouge = (
        MetricSample("a", "x", ("x",), value=1.0),
        MetricSample("b", "y", ("z",), value=0.0),
    )
    ppl = (
        perplexity_sample("a", "x", TeacherForcedNll(1.0, 1)),
        perplexity_sample("b", "z", TeacherForcedNll(2.0, 2)),
    )
    rows = prefixkv_metric_observations(
        run_id="run",
        task="mmvet",
        model="llava",
        backend="hf",
        method="prefixkv_reimpl",
        retention_ratio=0.5,
        manifest_path="manifest.json",
        rouge=rouge,
        perplexity=ppl,
        reference_artifact_sha256="a" * 64,
        target_artifact_sha256="b" * 64,
        evaluator_commit="597f1ab",
    )
    ppl_rows = [row for row in rows if row.metric_name == "perplexity"]
    assert len(ppl_rows) == 3
    assert sum(row.metric_scope is MetricScope.SAMPLE for row in ppl_rows) == 2
    assert ppl_rows[-1].metric_value == pytest.approx(math.exp(1.0))


def test_prefixkv_native_harness_pairs_generation_and_teacher_forcing() -> None:
    class FakeModel:
        model_id = "llava"
        backend = "hf"
        method = "prefixkv_reimpl"
        retention_ratio = 0.5
        supports_video = False

        def generate(self, _request: EvaluationRequest) -> ModelGeneration:
            raise AssertionError("generation rows are already supplied")

        def teacher_forced_nll(self, request: EvaluationRequest, target: str) -> TeacherForcedNll:
            assert request.sample_id == target
            return TeacherForcedNll(1.0, 2)

    reference_rows = (_fullkv_row("a", "alpha"), _fullkv_row("b", "beta"))
    artifact = build_fullkv_reference_artifact(
        reference_rows,
        model_revision="model-sha",
        tokenizer_revision="tokenizer-sha",
        generation_config_sha256="a" * 64,
    )
    generated_rows = tuple(
        EvaluationResult.from_generation(
            run_id="candidate",
            sample_id=sample_id,
            task="mmvet",
            model="llava",
            backend="hf",
            method="prefixkv_reimpl",
            retention_ratio=0.5,
            answer=answer,
            reference=sample_id,
            task_score=None,
            metrics=GenerationMetrics(),
            manifest_path="manifest.json",
        )
        for sample_id, answer in (("a", "alpha"), ("b", "different"))
    )
    requests = tuple(
        EvaluationRequest(
            run_id="candidate",
            sample_id=sample_id,
            task="mmvet",
            messages=build_multimodal_messages("question"),
            generation_kwargs={},
        )
        for sample_id in ("a", "b")
    )

    rows = evaluate_prefixkv_native_metrics(
        FakeModel(),
        requests,
        {"a": "a", "b": "b"},
        generated_rows,
        artifact,
        manifest_path="manifest.json",
        evaluator_commit="597f1ab",
    )

    assert len([row for row in rows if row.metric_name == "rouge_l_f1"]) == 2
    ppl_corpus = [
        row
        for row in rows
        if row.metric_name == "perplexity" and row.metric_scope is MetricScope.CORPUS
    ]
    assert ppl_corpus[0].metric_value == pytest.approx(math.exp(0.5))


def _accuracy_row(method: str, sample_id: str, value: float) -> MetricObservation:
    return MetricObservation(
        run_id=f"run-{method}",
        sample_id=sample_id,
        task="docvqa",
        model="llava",
        backend="hf",
        method=method,
        retention_ratio=1.0 if method == "full_kv" else 0.5,
        metric_name="accuracy",
        metric_value=value,
        metric_direction=MetricDirection.HIGHER,
        metric_scope=MetricScope.SAMPLE,
        metric_implementation="lmms_eval.task_owned",
        metric_version="1",
        metric_unit="fraction",
        measurement_type="unit_test",
        manifest_path="manifest.json",
    )


def test_paired_method_comparison_rejects_unpaired_or_mixed_metrics() -> None:
    reference = (_accuracy_row("full_kv", "a", 1.0), _accuracy_row("full_kv", "b", 0.0))
    candidate = (
        _accuracy_row("mosaickv_exact", "a", 1.0),
        _accuracy_row("mosaickv_exact", "b", 1.0),
    )
    result = compare_paired_methods(reference, candidate, resamples=50, seed=3)
    assert result.candidate_minus_reference == 0.5
    with pytest.raises(ValueError, match="identical sample IDs"):
        compare_paired_methods(reference, candidate[:1], resamples=10)


def test_versioned_published_protocol_suite_is_valid() -> None:
    root = Path(__file__).resolve().parents[2]
    suite = load_published_protocols(
        root / "configs/experiments/published_paper_native_metrics_v1.yaml"
    )
    assert {protocol.protocol_id for protocol in suite.protocols} == {
        "lookm_native_v1",
        "prefixkv_native_v1",
        "vl_cache_native_v1",
    }
    assert all(not protocol.execution_authorized for protocol in suite.protocols)
    assert len(suite.sha256) == 64
