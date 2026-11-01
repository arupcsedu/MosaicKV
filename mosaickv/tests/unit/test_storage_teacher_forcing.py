from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from mosaickv.adapters.huggingface.types import DecodeState
from mosaickv.evaluation.storage_teacher_forcing import (
    _score,
    score_formats,
    score_ordinary,
    validate_targets,
)


def prefill():
    return SimpleNamespace(
        logits=torch.tensor([0.0, 1.0, 2.0]),
        state=DecodeState(None, torch.ones(1, 5), 5, 5, 5, (), {}),
    )


def test_one_token_never_constructs_cache_or_calls_decode():
    adapter = SimpleNamespace(decode_one_token=lambda *a, **kw: pytest.fail("unexpected decode"))
    formats = (SimpleNamespace(tensors=lambda: (torch.ones(7),)),)
    with patch(
        "mosaickv.evaluation.storage_teacher_forcing.cache_from_formats", side_effect=AssertionError
    ):
        row = score_formats(adapter, prefill(), formats, (2,))
    assert row["evaluated_token_count"] == 1 and row["updates"] == 0
    assert row["storage"]["initial_active_kv_storage_bytes"] == 28
    assert row["storage"]["final_active_kv_storage_bytes"] == 28
    assert row["nll_sum"] == pytest.approx(
        float(-torch.log_softmax(torch.tensor([0.0, 1.0, 2.0]), 0)[2])
    )


def test_complete_teacher_prefix_no_greedy_or_eos_substitution():
    original = prefill()
    state = DecodeState(None, torch.ones(1, 5), 5, 5, 5, (), {})
    fed = []

    def decode(token, state, **kwargs):
        fed.append(token.item())
        state.next_decode_position += 1
        return SimpleNamespace(logits=torch.tensor([2.0, 1.0, 0.0]))

    row, logits = _score(
        SimpleNamespace(device="cpu", decode_one_token=decode),
        original,
        state,
        (0, 2, 1, 0),
        return_logits=True,
    )
    assert fed == [0, 2, 1] and row["evaluated_token_count"] == 4
    assert len(row["per_token_nll"]) == 4 and logits.shape == (4, 3)
    assert row["updates"] == 3 and original.state.next_decode_position == 5
    assert row["nll_sum"] == pytest.approx(sum(row["per_token_nll"]))


@pytest.mark.parametrize("targets", [(), (True,), (1, -1), [1], tuple(range(1025))])
def test_invalid_targets_fail_closed(targets):
    with pytest.raises(ValueError):
        validate_targets(targets)


def test_no_implicit_128_token_limit_or_truncation():
    assert len(validate_targets((1,) * 593)) == 593


def test_nonfinite_or_outside_vocab_rejected():
    with pytest.raises(ValueError):
        _score(None, prefill(), None, (3,))


def test_ordinary_clones_actual_cache_and_preserves_source_state():
    value = prefill()
    original = object()
    value.state.past_key_values = original
    cloned = object()
    seen = []

    def extract(cache, *, clone):
        assert cache is original and clone
        seen.append(cache)
        return cloned

    def decode(token, state, **kwargs):
        assert state.past_key_values is cloned
        state.next_decode_position += 1
        return SimpleNamespace(logits=torch.tensor([2.0, 1.0, 0.0]))

    adapter = SimpleNamespace(
        device="cpu",
        extract_past_key_values=extract,
        inject_past_key_values=lambda snapshot: snapshot,
        decode_one_token=decode,
    )
    row = score_ordinary(adapter, value, (0, 1))
    assert row["evaluated_token_count"] == 2 and seen == [original]
    assert value.state.past_key_values is original and value.state.next_decode_position == 5
