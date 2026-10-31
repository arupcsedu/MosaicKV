from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.draft_prompt_evidence import reduce_prompt_mass
from mosaickv.evaluation.request_precision import online_frozen


def steps(n=40):
    return tuple((torch.ones(1, 2, 1, n + s + 1) / (n + s + 1),) for s in range(4))


def test_draft_excludes_history_and_preserves_individual_head_demand():
    captured = steps()
    for (p,) in captured:
        p[0, 0, 0, 7] = 2
        p[0, 1, 0, 11] = 3
        p[..., 40:] = 100  # Draft history cannot dominate original-prompt ranking.
    original = [p.clone() for (p,) in captured]
    demand = reduce_prompt_mass(captured, prompt_length=40, heads=(2,))
    assert demand[0].argmax(-1).tolist() == [7, 11]
    assert torch.allclose(demand[0].sum(-1), torch.ones(2))
    assert all(torch.equal(a, b[0]) for a, b in zip(original, captured, strict=True))


def test_registered_horizon_and_layout_fail_closed():
    with pytest.raises(ValueError, match="four-step"):
        reduce_prompt_mass(steps()[:2], prompt_length=40, heads=(2,))
    with pytest.raises(ValueError, match="layout"):
        reduce_prompt_mass(steps(), prompt_length=40, heads=(1,))
    invalid = list(steps())
    invalid[0][0][..., :40] = 0
    with pytest.raises(ValueError, match="positive"):
        reduce_prompt_mass(invalid, prompt_length=40, heads=(2,))


def test_draft_selection_changes_evidence_not_capacity_or_mandatory_positions():
    p = torch.ones(1, 2, 40, 40) / 40
    prefill = SimpleNamespace(attention_weights=(p,))
    old, before = online_frozen(prefill, (0.5,), 0.5, method="mosaickv_exact")
    demand = torch.zeros(2, 40)
    demand[:, 32] = 1
    new, after = online_frozen(
        prefill, (0.5,), 0.5, method="mosaickv_exact", draft_attention=(demand,)
    )
    assert before["actual_source_slot_cap"] == after["actual_source_slot_cap"]
    key = "uniform_0.5_control"
    assert new[key]["anchor"]["positions"] != old[key]["anchor"]["positions"]
    for h in new[key]["anchor"]["positions"][0]:
        assert 32 in h and 39 in h and len(h) == after["per_layer_capacity"][0]
    with pytest.raises(ValueError, match="explicit"):
        online_frozen(
            prefill, (0.5,), 0.5, method="prompt_attention_topk", draft_attention=(demand,)
        )
    with pytest.raises(ValueError, match="normalized"):
        online_frozen(prefill, (0.5,), 0.5, method="mosaickv_exact", draft_attention=(demand * 2,))


def test_collection_reuses_prefill_and_rejects_position_mutation(monkeypatch):
    from mosaickv.evaluation.draft_prompt_evidence import collect
    from mosaickv.forecasting import huggingface

    cache = object()
    state = SimpleNamespace(
        past_key_values=cache,
        active_cache_length=40,
        logical_sequence_length=40,
        next_decode_position=40,
    )
    prefill = SimpleNamespace(state=state)
    tensor = torch.ones(1, 2, 40, 8)
    adapter = SimpleNamespace(_legacy_layers=lambda c: (((tensor, tensor),), None))
    calls = []

    def rollout(a, p, **kwargs):
        calls.append((a, p, kwargs))
        return SimpleNamespace(
            attention_steps=steps(),
            original_cache_unchanged=True,
            cache_clone_seconds=0.1,
            draft_decode_seconds=0.2,
            synchronization_calls=6,
            token_ids=(1, 2, 3, 4),
        )

    monkeypatch.setattr(huggingface, "run_isolated_greedy_query_rollout", rollout)
    demand, proof = collect(adapter, prefill)
    assert len(demand) == 1 and proof["second_prefill"] is False
    assert calls == [
        (adapter, prefill, dict(steps=4, capture_attentions=True, capture_distributions=False))
    ]
    assert prefill.state.past_key_values is cache

    def corrupt(a, p, **kwargs):
        p.state.next_decode_position += 1
        return rollout(a, p, **kwargs)

    monkeypatch.setattr(huggingface, "run_isolated_greedy_query_rollout", corrupt)
    with pytest.raises(RuntimeError, match="positions"):
        collect(adapter, prefill)
