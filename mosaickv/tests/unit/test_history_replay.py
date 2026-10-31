import pytest
from evicastkv_history_replay import identical_predictions


def row(name, tokens=(1, 2), answer="same", digest="a"):
    return dict(
        sample_id=name,
        greedy_ids=list(tokens),
        scored_answer=dict(answer=answer),
        logits_sha256=digest,
    )


def test_identity_is_paired_and_does_not_double_count_quality_N():
    result = identical_predictions([row("a"), row("b")], [row("b"), row("a")])
    assert result == dict(
        sample_count=2, changed_greedy_sequences=0, changed_scored_answers=0, changed_logit_hashes=0
    )


def test_changed_tokens_or_answers_cannot_reuse_the_quality_artifact():
    result = identical_predictions(
        [row("a"), row("b")], [row("b"), row("a", tokens=(1, 3), answer="different")]
    )
    assert result["changed_greedy_sequences"] == result["changed_scored_answers"] == 1


@pytest.mark.parametrize("other", [[row("a")], [row("a"), row("a")]])
def test_missing_or_duplicate_predictions_fail_closed(other):
    with pytest.raises(ValueError, match="complete unique"):
        identical_predictions([row("a"), row("b")], other)
