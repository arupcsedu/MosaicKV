"""Synthetic CPU correctness only; no pretrained answer-quality claims."""

from __future__ import annotations

import copy
import inspect
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mosaickv.backends.hf_allocation_guard import decide_draft_stability, packed_positions_sha
from mosaickv.backends.hf_single_exchange_guard import (
    guard_single_exchanges,
    validate_single_exchange_trace,
    validate_single_transaction,
)
from mosaickv.cache_state import FullKVState
from mosaickv.config import (
    BoundedDecodeConfig,
    ConfigurationError,
    DraftAllocationGuardConfig,
    DraftExchangeSearchConfig,
)
from mosaickv.selection.direct_error_exchange import build_single_exchange_layer_plans


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(candidates_per_round=True),
        dict(candidates_per_round=9),
        dict(max_rounds=0),
        dict(candidates_per_round=8, max_rounds=3),
    ],
)
def test_strict_single_exchange_search(kwargs):
    with pytest.raises(ConfigurationError, match="draft_exchange"):
        DraftExchangeSearchConfig(**kwargs)


def test_strict_opt_in_config_and_no_label_parameters():
    policy = DraftAllocationGuardConfig()
    with pytest.raises(ConfigurationError, match="draft_exchange"):
        BoundedDecodeConfig(layer_allocation="draft_single_exchange_v1", draft_guard=policy)
    with pytest.raises(ConfigurationError, match="draft_exchange"):
        BoundedDecodeConfig(draft_exchange=DraftExchangeSearchConfig())
    cfg = BoundedDecodeConfig.from_mapping(
        dict(
            layer_allocation="draft_single_exchange_v1",
            draft_guard={},
            draft_exchange={},
        )
    )
    assert cfg.draft_guard == policy
    assert cfg.draft_exchange == DraftExchangeSearchConfig(4, 2)
    assert not {"target", "answers", "references", "score"}.intersection(
        inspect.signature(guard_single_exchanges).parameters
    )
    with pytest.raises(ConfigurationError, match="unknown"):
        DraftExchangeSearchConfig.from_mapping(dict(answers=[]))


def test_shortlist_is_independent_single_exact_byte_mandatory_preserving_and_deterministic():
    values = ([0.0, 0.0, 0.0, 4.0, 2.0, 2.0, 2.0, 2.0], [0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0])
    state = FullKVState.from_tensors(
        tuple(
            (torch.zeros(1, 2, 8, 1), torch.tensor(row).reshape(1, 1, 8, 1).repeat(1, 2, 1, 1))
            for row in values
        ),
        block_size=1,
        mandatory_logical_positions=(3,),
    )
    attention = {(i, h): np.ones(8) / 8 for i in range(2) for h in range(2)}
    records = tuple(
        dict(layer=i, kv_head=h, logical_position=p, tier="exact")
        for i in range(2)
        for h in range(2)
        for p in range(4)
    )
    kwargs = dict(prompt_budget_bytes=state.active_bytes // 2, max_candidates=4)
    plans = build_single_exchange_layer_plans(state, attention, records, **kwargs)
    again = build_single_exchange_layer_plans(state, attention, records, **kwargs)
    assert plans and tuple(p.positions for p in plans) == tuple(p.positions for p in again)
    for plan in plans:
        assert sorted(plan.capacities) == [3, 5]
        assert plan.prompt_bytes == kwargs["prompt_budget_bytes"]
        assert all(3 in head for layer in plan.positions for head in layer)
        assert len({len(head) for head in plan.positions[0]}) == 1
        assert plan.reconstruction["single_exchange"][
            "promotion_requires_actual_full_horizon_draft_logits"
        ]
    # A passed composition may be ragged across LAYERS, never across heads.
    first = plans[0]
    records = tuple(
        dict(layer=i, kv_head=h, logical_position=p, tier="exact")
        for i, layer in enumerate(first.positions)
        for h, row in enumerate(layer)
        for p in row
    )
    build_single_exchange_layer_plans(state, attention, records, **kwargs)


def setup_probe(monkeypatch):
    import mosaickv.adapters.huggingface.bounded_history_cache as transport

    def packed(name, positions):
        layers = tuple(
            SimpleNamespace(
                key=torch.tensor(row).float().reshape(1, 1, len(row), 1).repeat(1, 1, 1, 4),
                value=torch.ones(1, 1, len(row), 4),
            )
            for row in positions
        )
        records = tuple(
            dict(layer=i, kv_head=0, slot=s, tier="exact", logical_position=p)
            for i, row in enumerate(positions)
            for s, p in enumerate(row)
        )
        return SimpleNamespace(
            name=name, snapshot=SimpleNamespace(layers=layers), slot_records=records
        )

    anchor = packed("anchor", ((0, 3),) * 3)
    full = SimpleNamespace(
        past_key_values=[(torch.ones(1, 1, 4, 4), torch.ones(1, 1, 4, 4)) for _ in range(3)],
        active_cache_length=4,
        logical_sequence_length=4,
        next_decode_position=4,
        role="full",
        step=0,
    )
    budget = SimpleNamespace(
        prompt_budget_bytes=192, cap_bytes=288, generated_positions=1, reserved_generated_bytes=96
    )
    calls, plans = [], {}
    logits = {}

    def exchange(parent, donor, receiver, name):
        rows = [
            [r["logical_position"] for r in parent.slot_records if r["layer"] == i]
            for i in range(3)
        ]
        before = [len(row) for row in rows]
        rows[donor].remove(next(p for p in rows[donor] if p != 3))
        rows[receiver].append(next(p for p in range(4) if p not in rows[receiver]))
        choice = packed(name, tuple(tuple(sorted(row)) for row in rows))
        return SimpleNamespace(
            packed=choice,
            capacities=tuple(len(row) for row in rows),
            reconstruction={
                "single_exchange": dict(
                    donor_layer=donor,
                    receiver_layer=receiver,
                    transferred_bytes=32,
                    initial_layer_capacities=before,
                )
            },
        )

    def make(original, choice):
        state = copy.copy(original)
        state.role = choice.name
        state.past_key_values = [(layer.key, layer.value) for layer in choice.snapshot.layers]
        return state

    def install(_adapter, state, **_kwargs):
        state.past_key_values = [
            (torch.nn.functional.pad(k, (0, 0, 0, 1)), torch.nn.functional.pad(v, (0, 0, 0, 1)))
            for k, v in state.past_key_values
        ]

    monkeypatch.setattr(transport, "install_bounded_history", install)

    def decode(token, state, **_kwargs):
        state.step += 1
        state.next_decode_position += 1
        for k, v in state.past_key_values:
            k.add_(0.01)
            v.add_(0.01)
        value = (
            [2.0, 0.0]
            if state.role == "full"
            else [1.8, 0.2]
            if state.role == "anchor"
            else logits[state.role](state.step)
        )
        z = torch.tensor([value])
        calls.append(state.role)
        return SimpleNamespace(state=state, logits=z, next_token_id=z.argmax(-1))

    adapter = SimpleNamespace(
        device=torch.device("cpu"),
        _legacy_layers=lambda cache: (cache, False),
        preallocate_decode_cache=lambda *_: True,
        decode_one_token=decode,
    )

    def run(rounds=2):
        return guard_single_exchanges(
            adapter,
            SimpleNamespace(state=full, next_token_id=torch.tensor([0])),
            anchor,
            budget,
            DraftAllocationGuardConfig(),
            DraftExchangeSearchConfig(4, rounds),
            sink_tokens=0,
            clone_state=copy.deepcopy,
            state_from_packed=make,
            propose=lambda parent, limit: plans.get(parent.name, ())[:limit],
            pack=lambda plan: plan.packed,
        )

    return SimpleNamespace(
        anchor=anchor,
        full=full,
        budget=budget,
        calls=calls,
        plans=plans,
        logits=logits,
        exchange=exchange,
        run=run,
    )


@pytest.mark.parametrize("failed_step", [1, 2, 4, 16, 32])
def test_rejection_stops_at_first_violating_prefix_and_shares_draft(monkeypatch, failed_step):
    f = setup_probe(monkeypatch)
    candidate = f.exchange(f.anchor, 0, 1, "bad")
    f.plans["anchor"] = (candidate,)
    f.logits["bad"] = lambda step: [0.0, 2.0] if step >= failed_step else [1.9, 0.1]
    saved = copy.deepcopy(f.full)
    selected, trace = f.run()
    assert selected is f.anchor
    assert Counter(f.calls) == dict(full=failed_step, anchor=failed_step, bad=failed_step)
    assert trace["probe_updates"] == 3 * failed_step
    assert len(trace["attempts"][0]["steps"]) == failed_step
    assert validate_single_exchange_trace(trace)["early_rejections"] == int(failed_step < 32)
    assert f.full.next_decode_position == 4
    assert all(
        torch.equal(a, b)
        for p, q in zip(saved.past_key_values, f.full.past_key_values, strict=True)
        for a, b in zip(p, q, strict=True)
    )


def test_full_horizon_logit_ranking_then_composition_requires_parent_nonregression(monkeypatch):
    f = setup_probe(monkeypatch)
    good, best = f.exchange(f.anchor, 0, 1, "good"), f.exchange(f.anchor, 0, 2, "best")
    f.plans["anchor"] = (good, best)
    # Worsens BEST, although it still improves over the ORIGINAL anchor.
    worse = f.exchange(best.packed, 1, 2, "worse")
    f.plans["best"] = (worse,)
    f.logits.update(
        good=lambda _: [1.9, 0.1], best=lambda _: [1.95, 0.05], worse=lambda _: [1.85, 0.15]
    )
    selected, trace = f.run()
    assert selected is best.packed
    assert Counter(f.calls) == dict(full=32, anchor=32, good=32, best=32, worse=1)
    assert [event["accepted"] for event in trace["attempts"]] == [True, True, False]
    assert trace["probe_updates"] == 129
    assert validate_single_exchange_trace(trace)["accepted"]
    tampered = copy.deepcopy(trace)
    tampered["attempts"][0]["steps"].pop()
    with pytest.raises(ValueError):
        validate_single_exchange_trace(tampered)
    tampered = copy.deepcopy(trace)
    tampered["rounds"][0]["winner"] = packed_positions_sha(good.packed)
    with pytest.raises(ValueError, match="ranking"):
        validate_single_exchange_trace(tampered)


def test_partial_evidence_never_passes_and_late_prefix_not_hidden():
    rows = [
        dict(
            step=i,
            anchor=dict(logit_kl=0.1, maximum_logit_difference=0.5, greedy_token_agreement=True),
            candidate=dict(
                logit_kl=0.05, maximum_logit_difference=0.4, greedy_token_agreement=True
            ),
        )
        for i in range(1, 5)
    ]
    with pytest.raises(ValueError, match="entire registered horizon"):
        decide_draft_stability(rows, DraftAllocationGuardConfig(), allow_early_rejection=True)
    rows[-1]["candidate"]["logit_kl"] = 0.2
    assert not decide_draft_stability(
        rows, DraftAllocationGuardConfig(), allow_early_rejection=True
    )["accepted"]


def test_no_moves_and_zero_updates_never_claim_acceptance(monkeypatch):
    f = setup_probe(monkeypatch)
    selected, trace = f.run()
    assert selected is f.anchor and not f.calls
    assert not validate_single_exchange_trace(trace)["accepted"]
    f.budget.generated_positions = 0
    f.budget.reserved_generated_bytes = 0
    f.budget.cap_bytes = f.budget.prompt_budget_bytes
    selected, trace = f.run()
    assert selected is f.anchor and not f.calls and trace["reason"] == "no_decode_updates"
    assert not validate_single_exchange_trace(trace)["accepted"]


def test_composition_is_published_only_after_two_complete_horizon_passes(monkeypatch):
    f = setup_probe(monkeypatch)
    first = f.exchange(f.anchor, 0, 1, "first")
    second = f.exchange(first.packed, 2, 1, "second")
    f.plans.update(anchor=(first,), first=(second,))
    f.logits.update(first=lambda _: [1.9, 0.1], second=lambda _: [1.95, 0.05])
    selected, trace = f.run()
    assert selected is second.packed
    assert trace["final_layer_capacities"] == [1, 4, 1]
    assert Counter(f.calls) == dict(full=32, anchor=32, first=32, second=32)
    assert [len(event["steps"]) for event in trace["attempts"]] == [32, 32]
    assert validate_single_exchange_trace(trace)["accepted"]
    altered = copy.deepcopy(trace)
    altered["attempts"][1]["layer_capacities"][0] += 1
    with pytest.raises(ValueError, match="capacity mismatch"):
        validate_single_exchange_trace(altered)
    altered = copy.deepcopy(trace)
    altered["attempts"][1]["round"] = 2
    with pytest.raises(ValueError, match="search budget"):
        validate_single_exchange_trace(altered)


def test_multi_exchange_and_false_byte_metadata_rejected_before_model_probe(monkeypatch):
    f = setup_probe(monkeypatch)
    one = f.exchange(f.anchor, 0, 1, "one")
    bad = f.exchange(one.packed, 2, 1, "bad")
    with pytest.raises(ValueError, match="single whole-layer"):
        validate_single_transaction(f.anchor, bad.packed, one.reconstruction["single_exchange"])
    one.reconstruction["single_exchange"]["transferred_bytes"] = 64
    f.plans["anchor"] = (one,)
    with pytest.raises(ValueError, match="physical bytes"):
        f.run()
    assert not f.calls
