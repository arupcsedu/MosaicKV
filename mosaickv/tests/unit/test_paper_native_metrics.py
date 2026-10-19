from __future__ import annotations

import copy
import importlib.util
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from mosaickv.evaluation.messages import build_multimodal_messages
from mosaickv.evaluation.metric_export import cider_samples_from_results
from mosaickv.evaluation.metrics.comparison import compare_paired_methods
from mosaickv.evaluation.metrics.registry import default_metric_registry
from mosaickv.evaluation.metrics.scorers import (
    aggregate_cider,
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
    MILEBENCH_DATASET_TO_GROUP,
    MileBenchDatasetScore,
    MileBenchMetric,
    MileBenchQuestionType,
    MileBenchSampleScore,
    MileBenchScoringRoute,
    aggregate_milebench,
    aggregate_milebench_samples,
    milebench_multichoice_accuracy,
    milebench_needle_accuracy,
    milebench_rouge_l,
    milebench_samples_from_observations,
    paired_milebench_bootstrap,
    route_milebench_dataset,
    score_milebench_sample,
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
from mosaickv.evaluation.published_protocols import (
    load_published_protocols,
    metric_readiness_issues,
)
from mosaickv.evaluation.results import EvaluationResult, canonical_backend_name
from mosaickv.types import ResultCategory


def test_rouge_anls_and_exact_match_fixtures() -> None:
    assert rouge_l_f1("a b c", ("a x c",)) == pytest.approx(2 / 3)
    assert rouge_l_f1("", ("answer",)) == 0
    assert anls_score("invoice 42", ("invoice 43",)) == pytest.approx(0.9)
    assert anls_score("unrelated", ("x",)) == 0
    assert exact_match_score("Blue!", ("blue",)) == 1


def test_anls_matches_pinned_lmms_eval_source_on_whitespace_edge() -> None:
    from lmms_eval.api.metrics import anls

    references = ["a b"]
    prediction = "a   c"
    official = float(anls(references, [prediction], thresh_hold=0.5)["anls"])

    assert anls_score(prediction, references) == official


def test_rouge_matches_pinned_prefixkv_and_lookm_dependency() -> None:
    from rouge import Rouge

    prediction = "red car. blue door"
    reference = "red truck. blue door"
    official = float(Rouge().get_scores(prediction, reference)[0]["rouge-l"]["f"])

    assert rouge_l_f1(prediction, (reference,)) == official


def test_rouge_long_sequence_preserves_pinned_score_and_restores_recursion_limit() -> None:
    previous_limit = sys.getrecursionlimit()
    prediction = " ".join(f"token-{index}" for index in range(previous_limit + 32))

    assert rouge_l_f1(prediction, (prediction,)) == pytest.approx(1.0)
    assert sys.getrecursionlimit() == previous_limit


def test_cider_matches_lmms_eval_coco_tokenization_and_scorer() -> None:
    from lmms_eval.tasks.coco_cap import utils as coco_utils

    samples = (
        MetricSample("1", "A red car!", ("A red car.", "The car is red.")),
        MetricSample("2", "Two dogs run", ("Two dogs are running.", "Dogs run together.")),
        MetricSample("3", "A blue door", ("The door is blue.", "A blue door.")),
        MetricSample("4", "A person rides", ("A person is riding.", "Someone rides.")),
    )
    ground_truth = {
        sample.sample_id: [{"caption": reference} for reference in sample.references]
        for sample in samples
    }
    predictions = {sample.sample_id: [{"caption": sample.prediction}] for sample in samples}
    tokenizer = coco_utils.PTBTokenizer()
    official, _ = coco_utils.Cider().compute_score(
        tokenizer.tokenize(ground_truth),
        tokenizer.tokenize(predictions),
    )

    assert aggregate_cider(samples) == pytest.approx(float(official) * 100.0)


def test_cider_reducer_counts_repeated_bootstrap_draws() -> None:
    samples = (
        MetricSample("a", "red car", ("red car", "a red vehicle")),
        MetricSample("b", "blue door", ("blue door", "a blue doorway")),
        MetricSample("c", "two dogs", ("two dogs", "a pair of dogs")),
        MetricSample("d", "green tree", ("green tree", "a green tree")),
    )

    repeated_draw = (samples[0], samples[0], samples[2], samples[3])

    assert math.isfinite(aggregate_cider(repeated_draw))


def _cider_result(
    method: str,
    sample_id: str,
    answer: str,
    references: list[str],
) -> EvaluationResult:
    return EvaluationResult.from_generation(
        run_id=f"run-{method}",
        sample_id=sample_id,
        task="coco_caption",
        model="llava",
        backend="hf",
        method=method,
        retention_ratio=1.0 if method == "full_kv" else 0.5,
        answer=answer,
        reference=json.dumps(references),
        task_score=None,
        metrics=GenerationMetrics(),
        manifest_path="manifest.json",
    )


def _cider_corpus_row(
    method: str,
    samples: tuple[MetricSample, ...],
) -> MetricObservation:
    definition = default_metric_registry().resolve("cider")
    return MetricObservation(
        run_id=f"run-{method}",
        sample_id=None,
        task="coco_caption",
        model="llava",
        backend="hf",
        method=method,
        retention_ratio=1.0 if method == "full_kv" else 0.5,
        metric_name="cider",
        metric_value=definition.aggregate(samples),
        metric_direction=MetricDirection.HIGHER,
        metric_scope=MetricScope.CORPUS,
        metric_implementation=definition.implementation,
        metric_version=definition.version,
        metric_unit=definition.unit,
        measurement_type="unit_test",
        manifest_path="manifest.json",
        sample_count=len(samples),
    )


def test_paired_cider_bootstrap_reruns_corpus_reducer_from_raw_predictions() -> None:
    references = (
        ["red car", "a red vehicle"],
        ["blue door", "a blue doorway"],
        ["two dogs", "a pair of dogs"],
        ["green tree", "a green tree"],
    )
    reference_results = tuple(
        _cider_result("full_kv", str(index), values[0], values)
        for index, values in enumerate(references)
    )
    candidate_answers = ("red car", "wrong caption", "two dogs", "green tree")
    candidate_results = tuple(
        _cider_result("mosaickv_exact", str(index), answer, list(references[index]))
        for index, answer in enumerate(candidate_answers)
    )
    reference_samples = cider_samples_from_results(reference_results)
    candidate_samples = cider_samples_from_results(candidate_results)
    reference_row = _cider_corpus_row("full_kv", reference_samples)
    candidate_row = _cider_corpus_row("mosaickv_exact", candidate_samples)

    with pytest.raises(ValueError, match="sample-level sufficient statistics"):
        compare_paired_methods((reference_row,), (candidate_row,), resamples=2)
    with pytest.raises(ValueError, match="candidate sample count disagrees"):
        compare_paired_methods(
            (reference_row,),
            (replace(candidate_row, sample_count=5),),
            reference_samples=reference_samples,
            candidate_samples=candidate_samples,
            resamples=2,
        )

    comparison = compare_paired_methods(
        (reference_row,),
        (candidate_row,),
        reference_samples=reference_samples,
        candidate_samples=candidate_samples,
        resamples=4,
        seed=3,
    )

    assert comparison.metric_name == "cider"
    assert comparison.sample_count == 4
    assert comparison.candidate_minus_reference < 0
    assert comparison.confidence == 0.95


def test_perplexity_uses_token_weighted_corpus_nll() -> None:
    samples = (
        perplexity_sample("a", "target", TeacherForcedNll(math.log(2), 1)),
        perplexity_sample("b", "target", TeacherForcedNll(3 * math.log(4), 3)),
    )
    mean_nll = (math.log(2) + 3 * math.log(4)) / 4
    assert aggregate_perplexity(samples) == pytest.approx(math.exp(mean_nll))


def test_perplexity_matches_pinned_prefixkv_official_reducer_fixture() -> None:
    torch = pytest.importorskip("torch")
    token_nlls = (math.log(2), math.log(4), math.log(8), math.log(16))
    official = float(torch.exp(torch.stack([torch.tensor(value) for value in token_nlls]).mean()))
    samples = tuple(
        MetricSample(
            sample_id=f"token-{index}",
            prediction="",
            references=("target",),
            numerator=value,
            denominator=1,
        )
        for index, value in enumerate(token_nlls)
    )
    assert aggregate_perplexity(samples) == pytest.approx(official)


def test_metric_aware_bootstrap_recomputes_perplexity() -> None:
    registry = default_metric_registry()
    definition = registry.resolve("perplexity")
    assert definition.name == "teacher_forced_ppl"
    assert registry.resolve("teacher_forced_ppl") is definition
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
    expected_relative = (result.candidate_value - result.reference_value) / result.reference_value
    assert result.relative_difference == pytest.approx(expected_relative)
    assert result.confidence == 0.95


@pytest.mark.parametrize("metric_name", ["accuracy", "anls", "rouge_l_f1"])
def test_metric_aware_bootstrap_averages_sample_metrics(metric_name: str) -> None:
    definition = default_metric_registry().resolve(metric_name)
    reference = (
        MetricSample("a", "", ("x",), value=1.0),
        MetricSample("b", "", ("x",), value=0.0),
        MetricSample("c", "", ("x",), value=0.5),
    )
    candidate = (
        MetricSample("a", "", ("x",), value=1.0),
        MetricSample("b", "", ("x",), value=0.5),
        MetricSample("c", "", ("x",), value=0.5),
    )

    result = paired_bootstrap(definition, reference, candidate, resamples=100, seed=11)

    assert result.reference_value == pytest.approx(0.5)
    assert result.candidate_value == pytest.approx(2 / 3)
    assert result.candidate_minus_reference == pytest.approx(1 / 6)
    assert result.relative_difference == pytest.approx(1 / 3)
    assert result.ci_low <= result.candidate_minus_reference <= result.ci_high
    assert result.relative_ci_low == pytest.approx(result.ci_low / 0.5)
    assert result.relative_ci_high == pytest.approx(result.ci_high / 0.5)


def test_relative_difference_is_undefined_for_zero_reference() -> None:
    definition = default_metric_registry().resolve("accuracy")
    reference = (
        MetricSample("a", "", ("x",), value=0.0),
        MetricSample("b", "", ("x",), value=0.0),
    )
    candidate = (
        MetricSample("a", "", ("x",), value=1.0),
        MetricSample("b", "", ("x",), value=0.0),
    )

    result = paired_bootstrap(definition, reference, candidate, resamples=10)

    assert result.relative_difference is None
    assert result.relative_ci_low is None
    assert result.relative_ci_high is None


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
    assert [sample.value for sample in samples] == pytest.approx([0.999999995, 0.0])


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
    assert aggregate["T-1/accuracy"] == 0.5
    assert aggregate["NH/accuracy"] == pytest.approx(0.3)


def test_milebench_matches_pinned_lookm_official_fixture() -> None:
    source = Path(__file__).resolve().parents[3] / "third_party/LOOK-M/evaluate.py"
    spec = importlib.util.spec_from_file_location("lookm_official_evaluate_fixture", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    official = module.Eval()

    sample = {
        "sample_id": 1,
        "gt_response": "Second!",
        "pred_response": "B. second",
        "choice_list": ["First", "Second!"],
        "image": ["a.jpg"],
    }
    official_sample = copy.deepcopy(sample)
    official.process_sample(official_sample)
    official_accuracy, _ = official.judge_multi_choice(official_sample)
    local_accuracy = milebench_multichoice_accuracy(
        sample["pred_response"],
        sample["gt_response"],
        tuple(sample["choice_list"]),
    )
    assert local_accuracy == official_accuracy

    prediction = "Red car, blue door!"
    reference = "red truck blue door"
    official_rouge = official.evaluate_rouge(
        [
            {
                "sample_id": 1,
                "gt_response": reference,
                "pred_response": prediction,
                "image": ["a.jpg"],
            }
        ],
        {
            "data": [
                {
                    "sample_id": 1,
                    "image_quantity_level": "Few",
                    "task_instance": {"images_path": ["a.jpg"]},
                }
            ]
        },
    )[0]["Rouge-L f"]
    assert milebench_rouge_l(prediction, reference) == pytest.approx(official_rouge)


def test_milebench_routes_every_official_dataset_and_keeps_metrics_separate() -> None:
    assert len(MILEBENCH_DATASET_TO_GROUP) == 29
    for dataset in MILEBENCH_DATASET_TO_GROUP:
        route = route_milebench_dataset(dataset, MileBenchQuestionType.MULTI_CHOICE)
        assert route.group == MILEBENCH_DATASET_TO_GROUP[dataset]

    open_route = route_milebench_dataset("DocVQA", MileBenchQuestionType.OPEN_ENDED)
    assert open_route.route is MileBenchScoringRoute.OPEN_ENDED
    assert open_route.metric is MileBenchMetric.ROUGE_L
    assert score_milebench_sample(open_route, "red car", "red vehicle") == pytest.approx(0.5)

    needle_route = route_milebench_dataset(
        "TextNeedleInAHaystack", MileBenchQuestionType.OPEN_ENDED
    )
    assert needle_route.route is MileBenchScoringRoute.NEEDLE_TOKEN
    assert score_milebench_sample(needle_route, "the key is kiwi", "kiwi") == 1

    mixed = aggregate_milebench(
        (
            MileBenchDatasetScore("accuracy", "S-4", MileBenchMetric.ACCURACY, 0.8, 10),
            MileBenchDatasetScore("rouge", "S-4", MileBenchMetric.ROUGE_L, 0.4, 10),
        )
    )
    assert mixed == {"S-4/accuracy": 0.8, "S-4/rouge_l_f1": 0.4}


def test_milebench_preserves_official_empty_answer_fallback() -> None:
    choices = ("first", "second", "third")
    assert milebench_multichoice_accuracy("", "third", choices) == 1
    assert milebench_multichoice_accuracy("", "first", choices) == 0


def _milebench_samples(
    *,
    first_values: tuple[float, ...],
    second_values: tuple[float, ...],
) -> tuple[MileBenchSampleScore, ...]:
    return tuple(
        MileBenchSampleScore(
            sample_id=f"first-{index}",
            dataset="ActionLocalization",
            question_type=MileBenchQuestionType.MULTI_CHOICE,
            metric=MileBenchMetric.ACCURACY,
            value=value,
        )
        for index, value in enumerate(first_values)
    ) + tuple(
        MileBenchSampleScore(
            sample_id=f"second-{index}",
            dataset="ActionPrediction",
            question_type=MileBenchQuestionType.MULTI_CHOICE,
            metric=MileBenchMetric.ACCURACY,
            value=value,
        )
        for index, value in enumerate(second_values)
    )


def test_milebench_bootstrap_recomputes_equal_dataset_hierarchy() -> None:
    reference = _milebench_samples(
        first_values=(1.0, 0.0),
        second_values=(1.0, 1.0, 1.0, 1.0),
    )
    candidate = _milebench_samples(
        first_values=(1.0, 1.0),
        second_values=(1.0, 1.0, 1.0, 1.0),
    )

    reference_summary = aggregate_milebench_samples(reference)
    comparisons = paired_milebench_bootstrap(
        reference,
        candidate,
        resamples=100,
        seed=17,
    )
    comparison = comparisons["Realistic Temporal/accuracy"]

    # Equal-dataset aggregation gives (0.5 + 1.0) / 2, not the sample mean 5/6.
    assert reference_summary["Realistic Temporal/accuracy"] == pytest.approx(0.75)
    assert comparison.reference_value == pytest.approx(0.75)
    assert comparison.candidate_value == 1.0
    assert comparison.candidate_minus_reference == pytest.approx(0.25)
    assert comparison.relative_difference == pytest.approx(1 / 3)
    assert comparison.ci_low <= comparison.candidate_minus_reference <= comparison.ci_high
    assert comparison.confidence == 0.95

    with pytest.raises(ValueError, match="identical MileBench samples"):
        paired_milebench_bootstrap(reference, candidate[:-1], resamples=2)


def test_milebench_observations_require_routing_metadata() -> None:
    row = _accuracy_row("full_kv", "a", 1.0)
    routed = replace(
        row,
        task="milebench",
        metadata_json=json.dumps(
            {
                "dataset": "ActionLocalization",
                "question_type": "multi-choice",
            }
        ),
    )

    samples = milebench_samples_from_observations((routed,))

    assert samples[0].dataset == "ActionLocalization"
    with pytest.raises(ValueError, match="metadata_json dataset and question_type"):
        milebench_samples_from_observations((replace(routed, metadata_json="{}"),))


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
    ppl_rows = [row for row in rows if row.metric_name == "teacher_forced_ppl"]
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
        if row.metric_name == "teacher_forced_ppl" and row.metric_scope is MetricScope.CORPUS
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
    assert result.relative_difference == 1.0
    assert result.to_json_object()["confidence"] == 0.95
    with pytest.raises(ValueError, match="identical sample IDs"):
        compare_paired_methods(reference, candidate[:1], resamples=10)
    mixed = (candidate[0], replace(candidate[1], metric_name="anls"))
    with pytest.raises(ValueError, match="one task/model/backend/method/ratio/metric"):
        compare_paired_methods(reference, mixed, resamples=10)


def test_huggingface_backend_alias_is_canonicalized_for_analysis() -> None:
    assert canonical_backend_name("hf") == "huggingface"
    assert canonical_backend_name("huggingface") == "huggingface"
    assert canonical_backend_name("vllm") == "vllm"


def test_versioned_published_protocol_suite_is_valid() -> None:
    root = Path(__file__).resolve().parents[2]
    suite = load_published_protocols(
        root / "configs/experiments/published_paper_native_metrics_v1.yaml"
    )
    assert {protocol.protocol_id for protocol in suite.protocols} == {
        "lookm_hf_reimpl_native_metrics_v1",
        "prefixkv_hf_reimpl_native_metrics_v1",
        "vl_cache_native_v1",
    }
    assert all(protocol.execution_authorized for protocol in suite.protocols)
    assert all(
        protocol.result_category.value == "paper_reproduction" for protocol in suite.protocols
    )
    assert all(len(protocol.model_revision) == 40 for protocol in suite.protocols)
    assert all(
        protocol.budget_convention.require_measured_active_bytes
        and len(protocol.budget_convention.active_byte_accounting_sha256) == 64
        and len(protocol.supported_environment.environment_lock_sha256) == 64
        for protocol in suite.protocols
    )
    excluded = {
        method.name
        for protocol in suite.protocols
        for method in protocol.methods
        if not method.execution_enabled
    }
    assert excluded == {
        "lookm_official",
        "prefixkv_official",
    }
    enabled = {
        method.name
        for protocol in suite.protocols
        for method in protocol.methods
        if method.execution_enabled
    }
    assert {
        "h2o_reimpl",
        "elastic_cache_reimpl",
        "pyramidkv_reimpl",
        "roco_reimpl",
        "snapkv_reimpl",
        "streamingllm_reimpl",
    } <= enabled
    with pytest.raises(ValueError, match="exact 40-character model SHA"):
        replace(suite.protocols[0], model_revision="resolve_before_execution")
    with pytest.raises(ValueError, match="paper_reproduction"):
        replace(
            suite.protocols[0],
            result_category=ResultCategory.UNIFIED_COMPARISON,
        )
    assert all(
        method.status != "unsupported_unpinned"
        for protocol in suite.protocols
        for method in protocol.methods
    )
    assert len(suite.sha256) == 64
    # Reconciled with the reviewed official_218 materialized parent ledger;
    # this is not the unrelated lmms-eval MM-Vet route.
    issues = metric_readiness_issues(suite)
    prefix = next(p for p in suite.protocols if p.protocol_id.startswith("prefixkv_"))
    mmvet = next(t for t in prefix.tasks if t.name == "mmvet_prefixkv")
    assert mmvet.dataset_revision == "8ce15108c0b55c67200eb96601ba2bb98183ff65"
    # Module discovery is deliberately side-effect free: optional lmms-eval
    # judge imports are validated only by their specific fixture/task lane.
    assert issues == ()
