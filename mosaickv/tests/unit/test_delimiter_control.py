"""Synthetic transaction, deterministic donor, tie handling and immutability tests."""

from types import SimpleNamespace

import pytest
import torch

from mosaickv.backends.hf_allocation_guard import decide_draft_stability
from mosaickv.config import DraftAllocationGuardConfig
from mosaickv.evaluation.delimiter_control import (
    compare_prefix_logits,
    delimiter_positions,
    restore_positions,
)


def test_restore_is_equal_capacity_deterministic_and_preserves_mandatory_and_present_delimiter():
    scores = [0.1, 0.1, 0.9, 0.2, 0.3, 0.4, 0.8]
    restored, event = restore_positions((0, 1, 2, 6), (4, 5, 6), (2, 6), scores)
    assert restored == (2, 4, 5, 6)
    assert event["removed_positions"] == [0, 1]
    assert event["added_positions"] == [4, 5]
    assert len(restored) == 4
    assert restore_positions((0, 1, 2, 6), (4, 5, 6), (2, 6), scores) == (restored, event)


def test_complete_delimiter_is_noop_and_infeasible_restore_is_rejected_not_overfilled():
    assert restore_positions((1, 2, 3), (2, 3), (3,), [1] * 4)[0] == (1, 2, 3)
    with pytest.raises(ValueError, match="cannot fit"):
        restore_positions((0, 3), (1, 2, 3), (0, 3), [1] * 4)


@pytest.mark.parametrize(
    "selected,delimiter,scores",
    [
        ((2, 1), (2,), [1] * 3),
        ((1, 1), (2,), [1] * 3),
        ((0, 2), (1,), [float("nan")] * 3),
        ((0, 2), (4,), [1] * 3),
    ],
)
def test_invalid_source_or_scores_fail_closed(selected, delimiter, scores):
    with pytest.raises(ValueError):
        restore_positions(selected, delimiter, (2,), scores)


def test_delimiter_is_identified_from_actual_suffix_tokenization_not_hardcoded_positions():
    tokenizer = SimpleNamespace(decode=lambda x: "ASSISTANT:" if x == [7, 8, 9] else "other")
    assert delimiter_positions([1, 2, 7, 8, 9], tokenizer) == (2, 3, 4)
    with pytest.raises(ValueError, match="cannot be identified"):
        delimiter_positions([1, 2], tokenizer)


def test_fullkv_tie_does_not_waive_registered_token_guard():
    full = torch.zeros(32, 4)
    candidate = full.clone()
    candidate[22, 1] = 0.001
    rows = compare_prefix_logits(full, full, candidate)
    assert rows[22]["reference_argmax_tie_count"] == 4
    assert rows[22]["candidate"]["greedy_token_agreement"] is False
    verdict = decide_draft_stability(rows, DraftAllocationGuardConfig())
    assert not verdict["accepted"]
    assert any("new_greedy_token_disagreement" in x["causes"] for x in verdict["rejected_steps"])
    with pytest.raises(ValueError, match="complete"):
        compare_prefix_logits(full[:4], full[:4], candidate[:4])
