"""LOOK-M's paper-native MileBench scoring and subtask aggregation."""

from __future__ import annotations

import json
import math
import random
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from mosaickv.evaluation.metrics.scorers import rouge_l_f1
from mosaickv.evaluation.metrics.statistics import PairedMetricComparison, percentile
from mosaickv.evaluation.metrics.types import MetricObservation, MetricScope


class MileBenchMetric(StrEnum):
    ACCURACY = "accuracy"
    ROUGE_L = "rouge_l_f1"


class MileBenchQuestionType(StrEnum):
    OPEN_ENDED = "open-ended"
    MULTI_CHOICE = "multi-choice"


class MileBenchScoringRoute(StrEnum):
    MULTI_CHOICE = "multiple_choice_accuracy"
    OPEN_ENDED = "rouge_l_f1"
    NEEDLE_TOKEN = "needle_token_accuracy"
    NEEDLE_SUBSTRING = "needle_substring_accuracy"


# Exact dataset taxonomy released in LOOK-M's pinned ``score.py``.
MILEBENCH_DATASET_GROUPS: dict[str, tuple[str, ...]] = {
    "Realistic Temporal": (
        "ActionLocalization",
        "ActionPrediction",
        "ActionSequence",
        "CharacterOrder",
        "CounterfactualInference",
        "EgocentricNavigation",
        "MovingAttribute",
        "MovingDirection",
        "ObjectExistence",
        "ObjectInteraction",
        "ObjectShuffle",
        "SceneTransition",
        "StateChange",
    ),
    "Realistic Semantic": (
        "ALFRED",
        "CLEVR-Change",
        "DocVQA",
        "IEdit",
        "MMCoQA",
        "MultiModalQA",
        "nuscenes",
        "OCR-VQA",
        "SlideVQA",
        "Spot-the-Diff",
        "TQA",
        "WebQA",
        "WikiVQA",
    ),
    "Diagnostic": (
        "TextNeedleInAHaystack",
        "ImageNeedleInAHaystack",
        "GPR1200",
    ),
}
MILEBENCH_DATASET_TO_GROUP = {
    dataset: group for group, datasets in MILEBENCH_DATASET_GROUPS.items() for dataset in datasets
}


@dataclass(frozen=True, slots=True)
class MileBenchRoute:
    dataset: str
    group: str
    question_type: MileBenchQuestionType
    route: MileBenchScoringRoute
    metric: MileBenchMetric


def route_milebench_dataset(
    dataset: str,
    question_type: str | MileBenchQuestionType,
) -> MileBenchRoute:
    """Mirror LOOK-M's dataset/special-case/question-type dispatch exactly."""

    try:
        group = MILEBENCH_DATASET_TO_GROUP[dataset]
    except KeyError as error:
        raise LookupError(f"unknown official MileBench dataset {dataset!r}") from error
    declared = MileBenchQuestionType(question_type)
    if dataset in {"TextNeedleInAHaystack", "ImageNeedleInAHaystack"}:
        route = MileBenchScoringRoute.NEEDLE_TOKEN
        metric = MileBenchMetric.ACCURACY
    elif dataset == "MMCoQA":
        route = MileBenchScoringRoute.NEEDLE_SUBSTRING
        metric = MileBenchMetric.ACCURACY
    elif declared is MileBenchQuestionType.OPEN_ENDED:
        route = MileBenchScoringRoute.OPEN_ENDED
        metric = MileBenchMetric.ROUGE_L
    else:
        route = MileBenchScoringRoute.MULTI_CHOICE
        metric = MileBenchMetric.ACCURACY
    return MileBenchRoute(dataset, group, declared, route, metric)


MILEBENCH_SUBTASK_METRICS: dict[str, MileBenchMetric] = {
    "T-1": MileBenchMetric.ACCURACY,
    "T-2": MileBenchMetric.ACCURACY,
    "T-3": MileBenchMetric.ACCURACY,
    "T-4": MileBenchMetric.ACCURACY,
    "S-1": MileBenchMetric.ACCURACY,
    "S-2": MileBenchMetric.ACCURACY,
    "S-3": MileBenchMetric.ROUGE_L,
    # S-4 contains one accuracy and one ROUGE-L dataset.  Its dataset metric
    # must therefore be supplied explicitly and is validated below.
    "S-4": MileBenchMetric.ACCURACY,
    "S-5": MileBenchMetric.ACCURACY,
    "N-1": MileBenchMetric.ACCURACY,
    "N-2": MileBenchMetric.ACCURACY,
    "I-1": MileBenchMetric.ACCURACY,
}


@dataclass(frozen=True, slots=True)
class MileBenchDatasetScore:
    """One official dataset score before equal-dataset subtask averaging."""

    dataset: str
    subtask: str
    metric: MileBenchMetric
    value: float
    sample_count: int
    question_type: MileBenchQuestionType | None = None

    def __post_init__(self) -> None:
        if not self.dataset.strip():
            raise ValueError("dataset must be non-empty")
        if self.dataset in MILEBENCH_DATASET_TO_GROUP:
            if self.question_type is None:
                raise ValueError("official MileBench dataset scores require question_type")
            route = route_milebench_dataset(self.dataset, self.question_type)
            if self.subtask != route.group:
                raise ValueError(
                    f"dataset {self.dataset} belongs to {route.group!r}, not {self.subtask!r}"
                )
            if self.metric is not route.metric:
                raise ValueError(f"dataset {self.dataset} requires {route.metric}")
        elif self.subtask not in MILEBENCH_SUBTASK_METRICS:
            raise ValueError(f"unknown MileBench dataset/subtask {self.dataset!r}/{self.subtask!r}")
        elif self.subtask != "S-4" and self.metric is not MILEBENCH_SUBTASK_METRICS[self.subtask]:
            expected = MILEBENCH_SUBTASK_METRICS[self.subtask]
            raise ValueError(f"subtask {self.subtask} requires {expected}")
        if not math.isfinite(self.value):
            raise ValueError("dataset score must be finite")
        if self.sample_count < 1:
            raise ValueError("sample_count must be positive")


@dataclass(frozen=True, slots=True)
class MileBenchSampleScore:
    """One sample contribution with enough metadata to rebuild the hierarchy."""

    sample_id: str
    dataset: str
    question_type: MileBenchQuestionType
    metric: MileBenchMetric
    value: float

    def __post_init__(self) -> None:
        if not self.sample_id.strip():
            raise ValueError("sample_id must be non-empty")
        route = route_milebench_dataset(self.dataset, self.question_type)
        if self.metric is not route.metric:
            raise ValueError(f"dataset {self.dataset} requires {route.metric}")
        if not math.isfinite(self.value):
            raise ValueError("sample score must be finite")


def _normalize(value: str) -> str:
    """Match the punctuation/case normalization in official LOOK-M evaluate.py."""

    text = value.replace("\n", " ").replace("\t", " ").strip()
    original = text
    comma_strip = re.compile(r"(\d)(,)(\d)")
    punctuation = [
        ";",
        "/",
        "[",
        "]",
        '"',
        "{",
        "}",
        "(",
        ")",
        "=",
        "+",
        "\\",
        "_",
        "-",
        ">",
        "<",
        "@",
        "`",
        ",",
        "?",
        "!",
    ]
    for mark in punctuation:
        adjacent = f"{mark} " in original or f" {mark}" in original
        replacement = "" if (adjacent or comma_strip.search(original)) else " "
        text = text.replace(mark, replacement)
    # Keep the official repository's regular expression verbatim.  Although
    # the first assertion appears intended as a lookbehind, changing it would
    # make this scorer non-identical to the released code.
    text = re.sub(r"(?!<=\d)(\.)(?!\d)", "", text)
    return text.strip("'").strip('"').strip().lower()


def milebench_rouge_l(prediction: str, reference: str) -> float:
    candidate = _normalize(prediction)
    target = _normalize(reference)
    if not candidate:
        return 0.0
    return rouge_l_f1(candidate, (target,))


def milebench_needle_accuracy(prediction: str, reference: str, *, token_bounded: bool) -> float:
    candidate = _normalize(prediction)
    target = _normalize(reference)
    if token_bounded:
        return float(target in candidate.split())
    return float(target in candidate)


def _choice_label(index: int) -> str:
    if index < 0:
        raise ValueError("choice index cannot be negative")
    if index < 26:
        return chr(index + 65)
    if index < 52:
        return "A" + chr(index + 39)
    return "B" + chr(index + 13)


def milebench_multichoice_accuracy(
    prediction: str, reference: str, choices: Sequence[str]
) -> float:
    if reference not in choices:
        raise ValueError("reference must occur in choices")
    candidate = _normalize(prediction)
    normalized_choices = tuple(_normalize(choice) for choice in choices)
    labels = tuple(_choice_label(index) for index in range(len(choices)))
    if candidate == "":
        return float(labels[choices.index(reference)] == "C")
    selected = ""
    for label, choice in zip(labels, normalized_choices, strict=True):
        if re.search(rf"{re.escape(label.casefold())} {re.escape(choice)}", candidate):
            selected = label
            break
    if not selected:
        compact = candidate.replace(" ", "")
        for label, choice in zip(labels, normalized_choices, strict=True):
            if choice.replace(" ", "") in compact:
                selected = label
                break
    if not selected and candidate.upper() in labels:
        selected = candidate.upper()
    if not selected:
        selected = "".join(
            character.upper() for character in candidate if character.upper() in labels
        )
    return float(selected == labels[choices.index(reference)])


def score_milebench_sample(
    route: MileBenchRoute,
    prediction: str,
    reference: str,
    *,
    choices: Sequence[str] = (),
) -> float:
    """Score one sample using only its validated official route."""

    if route.route is MileBenchScoringRoute.OPEN_ENDED:
        if choices:
            raise ValueError("open-ended MileBench samples cannot contain choices")
        return milebench_rouge_l(prediction, reference)
    if route.route is MileBenchScoringRoute.MULTI_CHOICE:
        if not choices:
            raise ValueError("multiple-choice MileBench samples require choices")
        return milebench_multichoice_accuracy(prediction, reference, choices)
    if choices:
        raise ValueError("needle MileBench samples cannot contain choices")
    return milebench_needle_accuracy(
        prediction,
        reference,
        token_bounded=route.route is MileBenchScoringRoute.NEEDLE_TOKEN,
    )


def aggregate_milebench(
    scores: Iterable[MileBenchDatasetScore],
) -> dict[str, float]:
    """Reproduce LOOK-M's equal-dataset subtask summaries.

    Dataset scores are averaged equally inside each subtask, matching the
    paper.  ``NH`` is the equal mean of N-1 and N-2 and ``IR`` aliases I-1.
    No overall macro combines heterogeneous Accuracy and ROUGE-L values.
    """

    grouped: dict[tuple[str, MileBenchMetric], list[float]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for score in scores:
        key = score.subtask, score.dataset
        if key in seen:
            raise ValueError(f"duplicate MileBench dataset score {key!r}")
        seen.add(key)
        grouped[(score.subtask, score.metric)].append(score.value)
    summaries = {
        f"{subtask}/{metric.value}": math.fsum(values) / len(values)
        for (subtask, metric), values in sorted(grouped.items())
    }
    for metric in MileBenchMetric:
        first = f"N-1/{metric.value}"
        second = f"N-2/{metric.value}"
        if first in summaries and second in summaries:
            summaries[f"NH/{metric.value}"] = (summaries[first] + summaries[second]) / 2.0
        image_retrieval = f"I-1/{metric.value}"
        if image_retrieval in summaries:
            summaries[f"IR/{metric.value}"] = summaries[image_retrieval]
    return summaries


def aggregate_milebench_samples(
    scores: Iterable[MileBenchSampleScore],
) -> dict[str, float]:
    """Reduce sample scores to dataset means and then rebuild the hierarchy."""

    grouped: dict[str, list[MileBenchSampleScore]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for score in scores:
        key = score.dataset, score.sample_id
        if key in seen:
            raise ValueError(f"duplicate MileBench sample score {key!r}")
        seen.add(key)
        grouped[score.dataset].append(score)
    if not grouped:
        raise ValueError("cannot aggregate an empty MileBench sample set")
    datasets: list[MileBenchDatasetScore] = []
    for dataset, samples in sorted(grouped.items()):
        identities = {(sample.question_type, sample.metric) for sample in samples}
        if len(identities) != 1:
            raise ValueError(f"dataset {dataset!r} mixes question types or metrics")
        question_type, metric = next(iter(identities))
        route = route_milebench_dataset(dataset, question_type)
        datasets.append(
            MileBenchDatasetScore(
                dataset=dataset,
                subtask=route.group,
                metric=metric,
                value=math.fsum(sample.value for sample in samples) / len(samples),
                sample_count=len(samples),
                question_type=question_type,
            )
        )
    return aggregate_milebench(datasets)


def milebench_samples_from_observations(
    rows: Sequence[MetricObservation],
) -> tuple[MileBenchSampleScore, ...]:
    """Recover hierarchy contributions from sample metric observations."""

    samples: list[MileBenchSampleScore] = []
    for row in rows:
        if row.metric_scope is not MetricScope.SAMPLE or row.sample_id is None:
            raise ValueError("MileBench hierarchy requires sample-scoped observations")
        try:
            metadata = json.loads(row.metadata_json)
        except json.JSONDecodeError as error:  # already schema-validated; defensive for callers.
            raise ValueError("invalid MileBench metadata_json") from error
        dataset = metadata.get("dataset")
        question_type = metadata.get("question_type")
        if not isinstance(dataset, str) or not isinstance(question_type, str):
            raise ValueError(
                "MileBench observations require metadata_json dataset and question_type"
            )
        samples.append(
            MileBenchSampleScore(
                sample_id=row.sample_id,
                dataset=dataset,
                question_type=MileBenchQuestionType(question_type),
                metric=MileBenchMetric(row.metric_name),
                value=row.metric_value,
            )
        )
    return tuple(samples)


def paired_milebench_bootstrap(
    reference: Sequence[MileBenchSampleScore],
    candidate: Sequence[MileBenchSampleScore],
    *,
    resamples: int = 2_000,
    seed: int = 2027,
    confidence: float = 0.95,
) -> dict[str, PairedMetricComparison]:
    """Recompute MileBench dataset means and hierarchy for every paired draw.

    Sampling is stratified by dataset so every official dataset remains in
    every replicate and equal-dataset aggregation is preserved.
    """

    if resamples < 1:
        raise ValueError("resamples must be positive")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")
    reference_by_key = {(item.dataset, item.sample_id): item for item in reference}
    candidate_by_key = {(item.dataset, item.sample_id): item for item in candidate}
    if len(reference_by_key) != len(reference) or len(candidate_by_key) != len(candidate):
        raise ValueError("paired MileBench sample identities must be unique")
    if reference_by_key.keys() != candidate_by_key.keys():
        raise ValueError("reference and candidate must contain identical MileBench samples")
    if len(reference_by_key) < 2:
        raise ValueError("paired MileBench bootstrap requires at least two samples")
    for sample_key in reference_by_key:
        left = reference_by_key[sample_key]
        right = candidate_by_key[sample_key]
        if (left.question_type, left.metric) != (right.question_type, right.metric):
            raise ValueError(f"paired MileBench metadata differs for {sample_key!r}")

    reference_value = aggregate_milebench_samples(reference_by_key.values())
    candidate_value = aggregate_milebench_samples(candidate_by_key.values())
    if reference_value.keys() != candidate_value.keys():
        raise ValueError("reference and candidate MileBench hierarchies differ")
    keys_by_dataset: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for sample_key in sorted(reference_by_key):
        keys_by_dataset[sample_key[0]].append(sample_key)
    generator = random.Random(seed)
    differences: dict[str, list[float]] = {key: [] for key in reference_value}
    for _index in range(resamples):
        reference_draw: list[MileBenchSampleScore] = []
        candidate_draw: list[MileBenchSampleScore] = []
        for dataset in sorted(keys_by_dataset):
            dataset_keys = keys_by_dataset[dataset]
            for sample_index, _sample in enumerate(dataset_keys):
                selected = dataset_keys[generator.randrange(len(dataset_keys))]
                reference_source = reference_by_key[selected]
                candidate_source = candidate_by_key[selected]
                bootstrap_id = f"{dataset}:bootstrap:{sample_index}"
                reference_draw.append(
                    MileBenchSampleScore(
                        bootstrap_id,
                        dataset,
                        reference_source.question_type,
                        reference_source.metric,
                        reference_source.value,
                    )
                )
                candidate_draw.append(
                    MileBenchSampleScore(
                        bootstrap_id,
                        dataset,
                        candidate_source.question_type,
                        candidate_source.metric,
                        candidate_source.value,
                    )
                )
        reference_summary = aggregate_milebench_samples(reference_draw)
        candidate_summary = aggregate_milebench_samples(candidate_draw)
        if reference_summary.keys() != reference_value.keys():
            raise RuntimeError("MileBench bootstrap changed hierarchy keys")
        for hierarchy_key in differences:
            differences[hierarchy_key].append(
                candidate_summary[hierarchy_key] - reference_summary[hierarchy_key]
            )

    tail = (1.0 - confidence) / 2.0
    comparisons: dict[str, PairedMetricComparison] = {}
    for hierarchy_key in sorted(differences):
        ordered = sorted(differences[hierarchy_key])
        absolute = candidate_value[hierarchy_key] - reference_value[hierarchy_key]
        denominator = abs(reference_value[hierarchy_key])
        ci_low = percentile(ordered, tail)
        ci_high = percentile(ordered, 1.0 - tail)
        comparisons[hierarchy_key] = PairedMetricComparison(
            metric_name=f"milebench/{hierarchy_key}",
            reference_value=reference_value[hierarchy_key],
            candidate_value=candidate_value[hierarchy_key],
            candidate_minus_reference=absolute,
            relative_difference=None if denominator == 0.0 else absolute / denominator,
            ci_low=ci_low,
            ci_high=ci_high,
            relative_ci_low=None if denominator == 0.0 else ci_low / denominator,
            relative_ci_high=None if denominator == 0.0 else ci_high / denominator,
            confidence=confidence,
            bootstrap_resamples=resamples,
            bootstrap_seed=seed,
            sample_count=len(reference_by_key),
        )
    return comparisons


__all__ = [
    "MILEBENCH_DATASET_GROUPS",
    "MILEBENCH_DATASET_TO_GROUP",
    "MILEBENCH_SUBTASK_METRICS",
    "MileBenchDatasetScore",
    "MileBenchMetric",
    "MileBenchQuestionType",
    "MileBenchRoute",
    "MileBenchSampleScore",
    "MileBenchScoringRoute",
    "aggregate_milebench",
    "aggregate_milebench_samples",
    "milebench_multichoice_accuracy",
    "milebench_needle_accuracy",
    "milebench_rouge_l",
    "milebench_samples_from_observations",
    "paired_milebench_bootstrap",
    "route_milebench_dataset",
    "score_milebench_sample",
]
