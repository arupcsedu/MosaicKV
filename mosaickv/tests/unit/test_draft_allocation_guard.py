"""Synthetic CPU regressions; never benchmark quality or paper evidence."""

from __future__ import annotations

import copy
import inspect
from types import SimpleNamespace

import pytest

from mosaickv.backends.hf_allocation_guard import (
    decide_draft_stability,
    guard_joint_allocation,
    packed_positions_sha,
    validate_draft_guard_trace,
)
from mosaickv.config import BoundedDecodeConfig, ConfigurationError, DraftAllocationGuardConfig


def scores(horizon=32):
    return [
        dict(
            step=i,
            anchor=dict(logit_kl=0.1, maximum_logit_difference=0.5, greedy_token_agreement=True),
            candidate=dict(
                logit_kl=0.05, maximum_logit_difference=0.4, greedy_token_agreement=True
            ),
        )
        for i in range(1, horizon + 1)
    ]


def test_joint_guard_rejects_late_regression_even_with_early_token_parity():
    rows = scores()
    assert decide_draft_stability(rows, DraftAllocationGuardConfig())["accepted"]
    rows[15]["candidate"]["logit_kl"] = 0.2
    result = decide_draft_stability(rows, DraftAllocationGuardConfig())
    assert not result["accepted"] and result["rejected_steps"] == [
        dict(step=16, causes=["logit_kl"])
    ]
    assert result["later_checkpoints_completed"]


@pytest.mark.parametrize(
    "metric,value",
    [("logit_kl", 0.2), ("maximum_logit_difference", 0.6), ("greedy_token_agreement", False)],
)
def test_every_prefix_is_guarded_not_only_checkpoint_averages(metric, value):
    rows = scores()
    rows[6]["candidate"][metric] = value
    assert not decide_draft_stability(rows, DraftAllocationGuardConfig())["accepted"]


def test_incomplete_and_nonfinite_probe_evidence_cannot_pass():
    with pytest.raises(ValueError, match="entire registered horizon"):
        decide_draft_stability(scores()[:4], DraftAllocationGuardConfig())
    rows = scores()
    rows[-1]["candidate"]["logit_kl"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        decide_draft_stability(rows, DraftAllocationGuardConfig())


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(horizon=True),
        dict(horizon=3),
        dict(horizon=65),
        dict(checkpoints=(1, 16, 4, 32)),
        dict(checkpoints=(1, 4)),
        dict(absolute_tolerance=-1.0),
        dict(relative_tolerance=float("inf")),
    ],
)
def test_strict_guard_config(kwargs):
    with pytest.raises(ConfigurationError, match="draft_guard"):
        DraftAllocationGuardConfig(**kwargs)


def test_guard_is_explicit_opt_in_and_never_implicitly_applied_to_baselines():
    with pytest.raises(ConfigurationError, match="draft_guard"):
        BoundedDecodeConfig(draft_guard=DraftAllocationGuardConfig())
    with pytest.raises(ConfigurationError, match="draft_guard"):
        BoundedDecodeConfig(layer_allocation="draft_logit_guard_v1")
    cfg = BoundedDecodeConfig(
        layer_allocation="draft_logit_guard_v1", draft_guard=DraftAllocationGuardConfig()
    )
    assert (
        BoundedDecodeConfig.from_mapping(
            dict(
                generated_capacity=32,
                sink_tokens=4,
                layer_allocation="draft_logit_guard_v1",
                draft_guard=dict(horizon=32, checkpoints=[1, 4, 16, 32]),
            )
        )
        == cfg
    )
    with pytest.raises(ConfigurationError, match="unknown"):
        DraftAllocationGuardConfig.from_mapping(dict(target="forbidden"))
    assert not {"target", "answers", "references"}.intersection(
        inspect.signature(guard_joint_allocation).parameters
    )


def test_selection_hash_uses_logical_positions_not_renumbered_physical_slots():
    import hashlib
    import json

    packed = SimpleNamespace(
        snapshot=SimpleNamespace(layers=[SimpleNamespace(key=SimpleNamespace(shape=(1, 1, 2, 4)))]),
        slot_records=[
            dict(layer=0, kv_head=0, slot=0, physical_position=1, logical_position=100),
            dict(layer=0, kv_head=0, slot=1, physical_position=2, logical_position=103),
        ],
    )
    expected = hashlib.sha256(
        json.dumps([[[100, 103]]], separators=(",", ":")).encode()
    ).hexdigest()
    assert packed_positions_sha(packed) == expected
    packed.slot_records[0]["physical_position"] = 50
    assert packed_positions_sha(packed) == expected
    packed.slot_records[0]["logical_position"] = 101
    assert packed_positions_sha(packed) != expected


@pytest.mark.parametrize("late_regression", [False, True])
def test_private_probes_preserve_sources_exact_bytes_positions_and_atomic_rollback(
    monkeypatch, late_regression
):
    import torch

    import mosaickv.adapters.huggingface.bounded_history_cache as transport

    def packed(candidate=False):
        layers = tuple(
            SimpleNamespace(
                key=torch.full((1, 1, 2, 4), float(i)), value=torch.full((1, 1, 2, 4), float(i + 1))
            )
            for i in range(2)
        )
        records = tuple(
            dict(layer=layer_index, kv_head=0, slot=s, physical_position=p, logical_position=p)
            for layer_index in range(2)
            for s, p in enumerate((1, 3) if candidate and layer_index == 0 else (0, 3))
        )
        return SimpleNamespace(snapshot=SimpleNamespace(layers=layers), slot_records=records)

    anchor, candidate = packed(), packed(True)
    full = SimpleNamespace(
        past_key_values=[(torch.ones(1, 1, 4, 4), torch.ones(1, 1, 4, 4)) for _ in range(2)],
        active_cache_length=4,
        logical_sequence_length=4,
        next_decode_position=4,
        role="full",
        step=0,
    )
    sources = [t.clone() for pair in full.past_key_values for t in pair]

    def make(original, choice):
        state = copy.copy(original)
        state.role = "anchor" if choice is anchor else "candidate"
        state.past_key_values = [(layer.key, layer.value) for layer in choice.snapshot.layers]
        return state

    def install(_adapter, state, **kwargs):
        assert kwargs["cap_bytes"] == 192
        state.past_key_values = [
            (torch.nn.functional.pad(k, (0, 0, 0, 1)), torch.nn.functional.pad(v, (0, 0, 0, 1)))
            for k, v in state.past_key_values
        ]

    monkeypatch.setattr(transport, "install_bounded_history", install)
    calls = []

    def decode(token, state, **_kwargs):
        state.step += 1
        state.next_decode_position += 1
        for k, v in state.past_key_values:
            k.add_(0.01)
            v.add_(0.01)
        z = (
            [2.0, 0.0]
            if state.role == "full"
            else [1.8, 0.2]
            if state.role == "anchor"
            else [0.0, 2.0]
            if late_regression and state.step >= 16
            else [1.9, 0.1]
        )
        logits = torch.tensor([z])
        calls.append(state.role)
        return SimpleNamespace(state=state, logits=logits, next_token_id=logits.argmax(-1))

    adapter = SimpleNamespace(
        device=torch.device("cpu"),
        _legacy_layers=lambda cache: (cache, False),
        preallocate_decode_cache=lambda *_: True,
        decode_one_token=decode,
    )
    budget = SimpleNamespace(
        prompt_budget_bytes=128, cap_bytes=192, generated_positions=1, reserved_generated_bytes=64
    )
    selected, trace = guard_joint_allocation(
        adapter,
        SimpleNamespace(state=full, next_token_id=torch.tensor([0])),
        anchor,
        candidate,
        budget,
        DraftAllocationGuardConfig(),
        sink_tokens=0,
        clone_state=copy.deepcopy,
        state_from_packed=make,
    )
    assert selected is (anchor if late_regression else candidate)
    assert len(calls) == 96 and trace["probe_updates"] == 96
    assert full.next_decode_position == 4
    assert all(
        torch.equal(x, y)
        for x, y in zip(sources, [t for pair in full.past_key_values for t in pair], strict=True)
    )
    assert validate_draft_guard_trace(trace)["accepted"] is (not late_regression)
    assert trace["final_positions_sha256"] == packed_positions_sha(selected)
    altered = copy.deepcopy(trace)
    altered["final_positions_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="final positions"):
        validate_draft_guard_trace(altered)
