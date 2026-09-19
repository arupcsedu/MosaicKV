"""LOOK-M's paper-native MileBench scoring and subtask aggregation."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from mosaickv.evaluation.metrics.scorers import rouge_l_f1


class MileBenchMetric(StrEnum):
    ACCURACY = "accuracy"
    ROUGE_L = "rouge_l_f1"


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

    def __post_init__(self) -> None:
        if not self.dataset.strip():
            raise ValueError("dataset must be non-empty")
        if self.subtask not in MILEBENCH_SUBTASK_METRICS:
            raise ValueError(f"unknown MileBench subtask {self.subtask!r}")
        if self.subtask != "S-4" and self.metric is not MILEBENCH_SUBTASK_METRICS[self.subtask]:
            expected = MILEBENCH_SUBTASK_METRICS[self.subtask]
            raise ValueError(f"subtask {self.subtask} requires {expected}")
        if not math.isfinite(self.value):
            raise ValueError("dataset score must be finite")
        if self.sample_count < 1:
            raise ValueError("sample_count must be positive")


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


def aggregate_milebench(
    scores: Iterable[MileBenchDatasetScore],
) -> dict[str, float]:
    """Reproduce LOOK-M's equal-dataset subtask summaries.

    Dataset scores are averaged equally inside each subtask, matching the
    paper.  ``NH`` is the equal mean of N-1 and N-2 and ``IR`` aliases I-1.
    No overall macro combines heterogeneous Accuracy and ROUGE-L values.
    """

    grouped: dict[str, list[float]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for score in scores:
        key = score.subtask, score.dataset
        if key in seen:
            raise ValueError(f"duplicate MileBench dataset score {key!r}")
        seen.add(key)
        grouped[score.subtask].append(score.value)
    summaries = {
        subtask: math.fsum(values) / len(values) for subtask, values in sorted(grouped.items())
    }
    if "N-1" in summaries and "N-2" in summaries:
        summaries["NH"] = (summaries["N-1"] + summaries["N-2"]) / 2.0
    if "I-1" in summaries:
        summaries["IR"] = summaries["I-1"]
    return summaries


__all__ = [
    "MILEBENCH_SUBTASK_METRICS",
    "MileBenchDatasetScore",
    "MileBenchMetric",
    "aggregate_milebench",
    "milebench_multichoice_accuracy",
    "milebench_needle_accuracy",
    "milebench_rouge_l",
]
