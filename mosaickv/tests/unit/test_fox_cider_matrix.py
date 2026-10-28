import random

import numpy as np
import pytest

from mosaickv.evaluation.coco_shared_bytes import TokenizedCider
from mosaickv.evaluation.fox_cider_matrix import FrozenCiderMatrix
from mosaickv.evaluation.metrics.types import MetricSample


def fixture_points():
    refs = (
        ("a red car near a sign", "a car", "the red car", "red car today", "red car near here"),
        ("birds fly", "two birds", "two birds fly", "a bird", "birds here", "birds above"),
        (".", "cat", "a sleeping cat", "the cat sleeps", "a cat", "cat today", "cat now"),
        ("one two two three", "one two", "one three", "three two", "one two two"),
    )
    predictions = (
        ("a red car near a sign", "two birds fly", "", "one two two two three"),
        ("unseen word word", "bird", "cat", "three two one"),
    )
    return tuple(
        tuple(MetricSample(str(i), text, refs[i]) for i, text in enumerate(point))
        for point in predictions
    )


def test_sparse_draws_match_official_each_corpus_and_duplicate_occurrence():
    points = fixture_points()
    tokenizer = TokenizedCider([s for p in points for s in p])
    matrix = FrozenCiderMatrix(points, tokenizer=tokenizer)
    rng = random.Random(17)
    draws = [(0, 1, 2, 3), (0, 0, 0, 0), (1,), (2, 2, 3), (3, 3, 1, 1, 0, 2)]
    draws += [tuple(rng.randrange(4) for _ in range(rng.randint(2, 12))) for _ in range(37)]
    expected = np.asarray([[tokenizer([p[i] for i in draw]) for p in points] for draw in draws])
    result = matrix.score_draws(draws, batch_size=7)
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-10)
    np.testing.assert_allclose(result, matrix.score_draws(draws, batch_size=1), rtol=0, atol=1e-10)


def test_sparse_cider_fail_closed_on_misaligned_references_and_invalid_draw():
    points = fixture_points()
    changed = list(points[1])
    changed[0] = MetricSample("0", "caption", ("changed",))
    with pytest.raises(ValueError, match="reference"):
        FrozenCiderMatrix((points[0], tuple(changed)))
    matrix = FrozenCiderMatrix(points)
    for draws in (((),), ((-1, 2),), ((4,),)):
        with pytest.raises(ValueError, match="resample"):
            matrix.score_draws(draws)
