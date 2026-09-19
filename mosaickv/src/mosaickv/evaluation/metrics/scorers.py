"""Pure metric scorers and corpus reducers used by paper-native protocols."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

from mosaickv.evaluation.metrics.types import MetricSample


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(value.split())


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    """Return LCS length with O(min(n,m)) auxiliary memory."""

    if len(left) < len(right):
        left, right = right, left
    row = [0] * (len(right) + 1)
    for left_token in left:
        diagonal = 0
        for index, right_token in enumerate(right, start=1):
            previous = row[index]
            if left_token == right_token:
                row[index] = diagonal + 1
            else:
                row[index] = max(row[index], row[index - 1])
            diagonal = previous
    return row[-1]


def rouge_l_f1(prediction: str, references: Sequence[str]) -> float:
    """Return maximum whitespace-tokenized ROUGE-L F1 over references.

    This matches the LCS/F1 definition used by PrefixKV and LOOK-M.  The
    implementation identity is recorded separately so it is never mislabeled
    as an invocation of their third-party ``rouge`` package.
    """

    predicted = _tokens(prediction)
    if not predicted:
        return 0.0
    best = 0.0
    for reference in references:
        target = _tokens(reference)
        if not target:
            continue
        overlap = _lcs_length(predicted, target)
        precision = overlap / len(predicted)
        recall = overlap / len(target)
        if precision + recall:
            best = max(best, 2.0 * precision * recall / (precision + recall))
    return best


def _normalized_text(value: str) -> str:
    return re.sub(r"[^\w]+", " ", value.casefold()).strip()


def exact_match_score(prediction: str, references: Sequence[str]) -> float:
    normalized = _normalized_text(prediction)
    return float(any(normalized == _normalized_text(reference) for reference in references))


def _levenshtein(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    row = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        following = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            following.append(
                min(
                    following[-1] + 1,
                    row[right_index] + 1,
                    row[right_index - 1] + (left_char != right_char),
                )
            )
        row = following
    return row[-1]


def anls_score(prediction: str, references: Sequence[str], *, threshold: float = 0.5) -> float:
    """Compute DocVQA ANLS using the official 0.5 normalized-distance cutoff."""

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must lie in [0, 1]")
    candidate = prediction.casefold().strip()
    best = 0.0
    for reference in references:
        target = reference.casefold().strip()
        normalizer = max(len(candidate), len(target))
        similarity = 1.0 if normalizer == 0 else 1.0 - _levenshtein(candidate, target) / normalizer
        if similarity < threshold:
            similarity = 0.0
        best = max(best, similarity)
    return best


def aggregate_mean(samples: Sequence[MetricSample]) -> float:
    if not samples:
        raise ValueError("cannot aggregate an empty metric sample set")
    values: list[float] = []
    for sample in samples:
        if sample.value is None:
            raise ValueError("mean-reduced metrics require every sample value")
        values.append(sample.value)
    return math.fsum(values) / len(values)


def aggregate_perplexity(samples: Sequence[MetricSample]) -> float:
    """Compute corpus PPL as exp(total NLL / total evaluated tokens)."""

    if not samples:
        raise ValueError("cannot aggregate an empty metric sample set")
    nll_values: list[float] = []
    total_tokens = 0
    for sample in samples:
        if sample.numerator is None or sample.denominator is None:
            raise ValueError("perplexity requires NLL numerator and token-count denominator")
        nll_values.append(sample.numerator)
        total_tokens += sample.denominator
    total_nll = math.fsum(nll_values)
    return math.exp(total_nll / total_tokens)


def aggregate_cider(samples: Sequence[MetricSample], *, scale: float = 100.0) -> float:
    """Compute corpus CIDEr with the evaluator used by lmms-eval 0.7.2.

    The ICLR VL-Cache table reports CIDEr in score points (approximately 100
    for ordinary COCO captions), hence the explicit factor of 100 over the
    pycocoevalcap return value.
    """

    if not samples:
        raise ValueError("cannot aggregate an empty metric sample set")
    try:
        from pycocoevalcap.cider.cider import Cider  # type: ignore[import-untyped]
    except ImportError as error:  # pragma: no cover - exercised by minimal installs.
        raise RuntimeError("CIDEr requires the pinned pycocoevalcap package") from error
    ground_truth = {sample.sample_id: list(sample.references) for sample in samples}
    predictions = {sample.sample_id: [sample.prediction] for sample in samples}
    score, _scores = Cider().compute_score(ground_truth, predictions)
    return float(score) * scale


__all__ = [
    "aggregate_cider",
    "aggregate_mean",
    "aggregate_perplexity",
    "anls_score",
    "exact_match_score",
    "rouge_l_f1",
]
