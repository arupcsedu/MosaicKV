"""Synthetic metric tests; no measured experimental results."""

from dataclasses import replace

import numpy as np
import pytest

from mosaickv.evaluation.cider_resampling import ResampledCider, paired_draw_counts
from mosaickv.evaluation.metrics.types import MetricSample


def groups():
    base = [
        MetricSample(
            str(i),
            text,
            (text or "empty scene", text + " today", text + " here", "a bird flies", "red car"),
        )
        for i, text in enumerate(("a red car drives", "a cat sleeps", "a bird flies", ""))
    ]
    base[1] = replace(base[1], references=(*base[1].references, "a kitten sleeps"))
    base[2] = replace(base[2], references=(*base[2].references, "birds fly", "many birds fly"))
    other = [
        replace(s, prediction=("unknown qwerty qwerty" if i % 2 else s.prediction + " here"))
        for i, s in enumerate(base)
    ]
    return {("baseline", 40): base, ("candidate", 20): list(reversed(other))}


@pytest.mark.parametrize("batch_size", [1, 5, 64])
def test_every_draw_rebuilds_official_corpus_cider(batch_size):
    scorer = ResampledCider(groups())
    counts = np.vstack(
        [np.ones((1, 4), dtype=int), np.array([[4, 0, 0, 0]]), paired_draw_counts(4, 17, 42)]
    )
    scores = scorer.score(counts, batch_size=batch_size)
    assert scorer.validate_official(counts, scores) < 1e-10
    assert scores.shape == (19, 2)
    # All copies of one document have zero IDF: frozen image scores would fail.
    assert np.all(scores[1] == 0)


def test_determinism_and_pairing_fail_closed():
    assert np.array_equal(paired_draw_counts(4, 10, 7), paired_draw_counts(4, 10, 7))
    data = groups()
    data["candidate", 20][0] = replace(data["candidate", 20][0], references=("changed",))
    with pytest.raises(ValueError, match="references"):
        ResampledCider(data)
    data = groups()
    data["candidate", 20] = data["candidate", 20][1:]
    with pytest.raises(ValueError, match="identical"):
        ResampledCider(data)
    data = groups()
    data["candidate", 20].append(data["candidate", 20][0])
    with pytest.raises(ValueError, match="duplicate"):
        ResampledCider(data)


@pytest.mark.parametrize("counts", [np.zeros((1, 4)), [[1, 1, 1, 0]], [[4, -1, 1, 0]]])
def test_invalid_resamples_rejected(counts):
    with pytest.raises(ValueError, match="draw counts"):
        ResampledCider(groups()).score(np.asarray(counts))
