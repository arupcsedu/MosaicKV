"""Synthetic timing/alignment semantics, not measured model quality."""

from types import SimpleNamespace

import evicastkv_token_eviction_audit as controller
import pytest
import torch

from mosaickv.evaluation.token_eviction import TeacherTokenCapture, compare_nll, first_divergence


class FakeAdapter:
    def __init__(self, ring=True):
        self.ring = ring

    def prefill(self):
        layer = SimpleNamespace(eviction_count=0, _generated_positions=[])
        state = SimpleNamespace(
            logical_sequence_length=10,
            next_decode_position=10,
            past_key_values=SimpleNamespace(layers=[layer]),
        )
        return SimpleNamespace(state=state, logits=torch.arange(8).reshape(1, 8).float())

    def decode_one_token(self, token, state):
        layer = state.past_key_values.layers[0]
        if self.ring and len(layer._generated_positions) == 2:
            layer._generated_positions.pop(0)
            layer.eviction_count += 1
        if self.ring:
            layer._generated_positions.append(state.next_decode_position)
        state.next_decode_position += 1
        return SimpleNamespace(state=state, logits=torch.arange(8).reshape(1, 8).float())


def module():
    def maintain(adapter, state, *, step_index):
        event = dict(
            layer=0, step=step_index, removed_logical_position=5 if step_index == 1 else 10
        )
        return state, (event,)

    return SimpleNamespace(maintain_prefixkv_decode_cache=maintain)


def test_fifo_eviction_affects_capacity_plus_two_prediction_not_previous_one():
    adapter = FakeAdapter()
    helpers = module()
    targets = (1, 2, 3, 4, 5)
    original_prefill = adapter.prefill
    original_decode = adapter.decode_one_token
    original_maintain = helpers.maintain_prefixkv_decode_cache
    with TeacherTokenCapture(adapter, targets, helpers) as capture:
        state = adapter.prefill().state
        for token in targets[:-1]:
            state = adapter.decode_one_token(torch.tensor([[token]]), state).state
    record = capture.record()
    assert [x["prediction_position"] for x in record["records"]] == [1, 2, 3, 4, 5]
    assert [x["generated_evictions_visible_per_layer"] for x in record["records"]] == [
        [0],
        [0],
        [0],
        [1],
        [2],
    ]
    assert record["records"][3]["removed_generated_positions_in_this_forward"] == [[10]]
    assert adapter.prefill == original_prefill and adapter.decode_one_token == original_decode
    assert helpers.maintain_prefixkv_decode_cache is original_maintain


def test_native_post_forward_cleanup_does_not_affect_already_scored_logits():
    adapter, helpers = FakeAdapter(ring=False), module()
    targets = (1, 2, 3, 4)
    with TeacherTokenCapture(adapter, targets, helpers) as capture:
        state = adapter.prefill().state
        for step, token in enumerate(targets[:-1], start=1):
            state = adapter.decode_one_token(torch.tensor([[token]]), state).state
            state, _ = helpers.maintain_prefixkv_decode_cache(adapter, state, step_index=step)
    records = capture.record()["records"]
    assert [x["prompt_evictions_visible_per_layer"] for x in records] == [[0], [0], [1], [1]]
    assert [x["generated_evictions_visible_per_layer"] for x in records] == [[0], [0], [0], [1]]
    assert capture.native_events[0]["first_prediction_it_can_affect"] == 3


def test_exception_restores_methods_and_rejects_different_teacher_prefix():
    adapter, helpers = FakeAdapter(), module()
    old = helpers.maintain_prefixkv_decode_cache
    with pytest.raises(RuntimeError, match="prefix"), TeacherTokenCapture(adapter, (1, 2), helpers):
        state = adapter.prefill().state
        adapter.decode_one_token(torch.tensor([[3]]), state)
    assert "prefill" not in adapter.__dict__ and "decode_one_token" not in adapter.__dict__
    assert helpers.maintain_prefixkv_decode_cache is old


def records(values):
    return [
        dict(
            prediction_position=i + 1,
            target_token_id=i,
            nll=v,
            generated_evictions_visible_per_layer=[int(i >= 3)],
        )
        for i, v in enumerate(values)
    ]


def test_pairwise_nll_windows_and_eviction_boundary_are_aligned():
    result = compare_nll(records([1, 2, 3, 4]), records([1, 2, 2, 3]))
    assert result["first_token_nll_increase"] == 3
    assert result["first_cumulative_nll_increase"] == 3
    assert result["first_generated_eviction_affected_prediction"] == 4
    assert result["nll_delta_before_first_generated_eviction"] == 1
    assert result["nll_delta_from_first_generated_eviction"] == 1
    assert result["nll_delta_total"] == 2


def test_pairwise_nll_rejects_target_mismatch_and_incomplete_cohorts():
    a, b = records([1, 2]), records([1, 2])
    b[0]["target_token_id"] = 99
    with pytest.raises(ValueError, match="alignment"):
        compare_nll(a, b)
    with pytest.raises(ValueError, match="lengths"):
        compare_nll(a, b[:1])


def test_greedy_alignment_distinguishes_post_eos_from_answer_changes(monkeypatch):
    monkeypatch.setattr(controller.previous, "phase_trace", lambda x: x)
    a = dict(
        generated_token_ids=[1, 9, 2], quality_decoding=dict(configured_terminal_token_ids=[9])
    )
    b = dict(
        generated_token_ids=[1, 9, 3], quality_decoding=dict(configured_terminal_token_ids=[9])
    )
    result = controller.greedy_alignment(a, b, generated_capacity=2)
    assert result["first_greedy_divergence"] == 3
    assert result["divergence_after_both_eos"]
    assert result["first_possible_fifo_affected_prediction"] == 4
    assert first_divergence([1, 2], [1, 3]) == 2
    with pytest.raises(ValueError, match="equal"):
        first_divergence([1], [1, 2])
