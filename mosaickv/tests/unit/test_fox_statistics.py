import pytest

from mosaickv.evaluation.fox_statistics import (
    cluster_draws,
    corpus_cider_statistics,
    media_components,
    paired_scalar_statistics,
)
from mosaickv.evaluation.metrics.types import MetricSample


def test_media_closure_is_transitive_and_label_independent():
    assert media_components(("a", "b", "c"), {"a": ("x",), "b": ("x", "y"), "c": ("y",)}) == (
        (0, 1, 2),
    )
    with pytest.raises(ValueError, match="exactly"):
        media_components(("a",), {"b": ("x",)})


def test_single_independent_cluster_cannot_claim_inferential_precision():
    with pytest.raises(ValueError, match="two independent"):
        cluster_draws(
            ("a", "b"), dict(a=("same",), b=("same",)), dict(a="all", b="all"), resamples=3
        )


def test_cluster_resampling_preserves_linked_media_and_each_subtask():
    ids = ("a", "b", "c", "d")
    groups = dict(a=("x",), b=("x",), c=("y",), d=("z",))
    tasks = dict(a="t1", b="t2", c="t1", d="t2")
    first = cluster_draws(ids, groups, tasks, resamples=9, seed=3)
    assert first == cluster_draws(ids, groups, tasks, resamples=9, seed=3)
    for draw in first:
        assert draw.count(0) == draw.count(1)
        assert {tasks[ids[index]] for index in draw} == {"t1", "t2"}


def test_equal_subtask_hierarchy_is_not_pooled_sample_mean():
    reference = dict(a=1.0, b=0.0, c=0.0)
    candidate = dict(a=1.0, b=1.0, c=1.0)
    groups = {key: (key,) for key in reference}
    task = dict(a="one", b="two", c="two")
    result = paired_scalar_statistics(
        reference, candidate, groups, task, resamples=11, hierarchy=True
    )
    assert result["reference_value"] == 0.5
    assert result["candidate_value"] == 1
    assert result["interpretation"].startswith("development_only")


def test_latency_speedup_is_ratio_of_means_not_mean_of_ratios():
    ref = dict(a=10.0, b=20.0)
    cand = dict(a=1.0, b=10.0)
    result = paired_scalar_statistics(
        ref, cand, dict(a=("a",), b=("b",)), dict(a="all", b="all"), resamples=7
    )
    assert result["speedup_ratio_of_means"] == pytest.approx(15 / 5.5)
    assert result["speedup_ratio_of_means"] != 6


def test_cider_recomputes_real_corpus_with_repeated_occurrences():
    samples = tuple(
        MetricSample(
            str(i), text, (text, text + " nearby", text + " today", text + " here", text + ".")
        )
        for i, text in enumerate(("a red car", "two birds fly", "a cat sleeps", "a man walks"))
    )
    result = corpus_cider_statistics(
        {"A": samples, "B": samples},
        [("A", "B")],
        {s.sample_id: (s.sample_id,) for s in samples},
        resamples=7,
        workers=1,
    )
    assert result["contrasts"][0]["difference"] == 0
    assert result["contrasts"][0]["ordinary_95_ci"] == [0, 0]
    assert result["scores"]["A"] > 0


def test_missing_or_changed_corpus_references_fail_closed():
    a = (MetricSample("x", "cat", ("a cat",)), MetricSample("y", "dog", ("a dog",)))
    b = (a[0], MetricSample("y", "dog", ("changed reference",)))
    with pytest.raises(ValueError, match="identical inputs AND references"):
        corpus_cider_statistics(
            {"A": a, "B": b}, [("A", "B")], dict(x=("x",), y=("y",)), resamples=1, workers=1
        )
