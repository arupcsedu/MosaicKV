"""Descriptive features must not score forced post-EOS tokens or drop references."""

import copy

import evicastkv_caption_audit as audit
import pytest


@pytest.fixture
def caption():
    tokens = [1, 3, 2] + [8] * 61
    row = dict(
        status="completed",
        answer="A red bird.",
        token_ids=tokens,
        forward_steps=63,
        references=["one bird", "a bird"],
    )
    trace = dict(
        answer=row["answer"],
        generated_token_ids=list(tokens),
        quality_decoding=dict(
            configured_terminal_token_ids=[2],
            timed_token_count=64,
            terminal_token_index=2,
            scored_token_count=2,
        ),
    )
    return row, trace


def test_saved_quality_boundary_excludes_forced_post_eos_continuation(caption):
    before = copy.deepcopy(caption)
    result = audit.caption_features(*caption)
    assert result["scored_tokens"] == 2 and result["timed_tokens"] == 64
    assert result["word_count"] == 3
    assert result["terminal_observed"] and not result["output_limit_hit"]
    assert caption == before


@pytest.mark.parametrize("count", [5, 6, 7])
def test_every_annotation_is_counted_without_truncation(caption, count):
    row, trace = caption
    row["references"] = [f"reference {i}" for i in range(count)]
    assert audit.caption_features(row, trace)["reference_count"] == count


def test_empty_answers_are_counted_not_excluded(caption):
    row, trace = caption
    row["answer"] = trace["answer"] = "   "
    result = audit.caption_features(row, trace)
    assert result["empty_answer"] and result["word_count"] == 0


def test_limit_hit_is_distinct_from_eos(caption):
    row, trace = caption
    row["token_ids"] = trace["generated_token_ids"] = [1] * 64
    trace["quality_decoding"].update(terminal_token_index=None, scored_token_count=64)
    assert audit.caption_features(row, trace)["output_limit_hit"]


@pytest.mark.parametrize(
    "field,value",
    [("scored_token_count", 3), ("terminal_token_index", 3), ("timed_token_count", 63)],
)
def test_changed_boundary_is_rejected(caption, field, value):
    row, trace = caption
    trace["quality_decoding"][field] = value
    with pytest.raises(ValueError, match="boundaries"):
        audit.caption_features(row, trace)


def test_repetition_feature_and_constant_association_are_explicit(caption):
    row, trace = caption
    row["answer"] = trace["answer"] = "a b c d a b c d"
    assert audit.caption_features(row, trace)["repeated_fourgram_count"] == 1
    assert audit.correlation([1, 1, 1], [1, 2, 3]) is None
    assert audit.correlation([1, 2, 3], [1, 2, 3]) == pytest.approx(1)
