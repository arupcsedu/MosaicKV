"""Pure metric scorers and corpus reducers used by paper-native protocols."""

from __future__ import annotations

import math
import re
import sys
from collections.abc import Sequence

from mosaickv.evaluation.metrics.types import MetricSample


def rouge_l_f1(prediction: str, references: Sequence[str]) -> float:
    """Return PrefixKV/LOOK-M's pinned ``rouge==1.0.1`` ROUGE-L F1."""

    if not references:
        raise ValueError("ROUGE-L requires at least one reference")
    if not prediction.strip():
        return 0.0
    try:
        from rouge import Rouge
    except ImportError as error:  # pragma: no cover - pinned evaluation dependency.
        raise RuntimeError("ROUGE-L requires pinned rouge==1.0.1") from error
    scorer = Rouge(metrics=("rouge-l",), stats=("f",))
    best = 0.0
    for reference in references:
        if not reference.strip():
            continue
        # ``rouge==1.0.1`` reconstructs the LCS recursively.  Its score is
        # correct for long PrefixKV descriptions, but Python's default depth
        # (normally 1,000) is smaller than valid 1,024-token generations.
        # Raising the limit changes no tokenization or scoring operation; it
        # only permits the pinned implementation to finish.  Always restore
        # the process setting so one metric call cannot affect later work.
        previous_limit = sys.getrecursionlimit()
        required_limit = max(
            previous_limit,
            4 * (len(prediction.split()) + len(reference.split()) + 1),
        )
        try:
            if required_limit != previous_limit:
                sys.setrecursionlimit(required_limit)
            result = scorer.get_scores(prediction, reference)
        finally:
            if sys.getrecursionlimit() != previous_limit:
                sys.setrecursionlimit(previous_limit)
        best = max(best, float(result[0]["rouge-l"]["f"]))
    return best


def _normalized_text(value: str) -> str:
    return re.sub(r"[^\w]+", " ", value.casefold()).strip()


def exact_match_score(prediction: str, references: Sequence[str]) -> float:
    normalized = _normalized_text(prediction)
    return float(any(normalized == _normalized_text(reference) for reference in references))


def anls_score(prediction: str, references: Sequence[str], *, threshold: float = 0.5) -> float:
    """Compute ANLS through the pinned lmms-eval DocVQA implementation."""

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must lie in [0, 1]")
    if not references:
        raise ValueError("ANLS requires at least one reference")
    try:
        from lmms_eval.api.metrics import anls
    except ImportError as error:  # pragma: no cover - pinned evaluation dependency.
        raise RuntimeError("ANLS requires pinned lmms-eval==0.7.2") from error
    result = anls(list(references), [prediction], thresh_hold=threshold)
    return float(result["anls"])


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
        from lmms_eval.tasks.coco_cap import utils as coco_utils
    except ImportError as error:  # pragma: no cover - pinned evaluation dependency.
        raise RuntimeError("CIDEr requires pinned lmms-eval==0.7.2") from error
    # Bootstrap draws legitimately contain repeated source samples.  Give
    # every occurrence a fresh evaluator-local identity so a repeated draw is
    # counted repeatedly instead of being collapsed by the COCO dictionaries.
    # Source-cohort uniqueness is checked before resampling by
    # ``paired_bootstrap``.
    ground_truth = {
        str(index): [{"caption": reference} for reference in sample.references]
        for index, sample in enumerate(samples)
    }
    predictions = {
        str(index): [{"caption": sample.prediction}] for index, sample in enumerate(samples)
    }
    tokenizer = coco_utils.PTBTokenizer()
    tokenized_ground_truth = tokenizer.tokenize(ground_truth)
    tokenized_predictions = tokenizer.tokenize(predictions)
    score, _scores = coco_utils.Cider().compute_score(
        tokenized_ground_truth,
        tokenized_predictions,
    )
    return float(score) * scale


__all__ = [
    "aggregate_cider",
    "aggregate_mean",
    "aggregate_perplexity",
    "anls_score",
    "exact_match_score",
    "rouge_l_f1",
]
