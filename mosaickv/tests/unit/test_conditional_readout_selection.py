import pytest
import torch

from mosaickv.evaluation.conditional_readout_selection import choose, revised_ids, shortlist
from mosaickv.evaluation.storage_six_overlay import OverlayPolicy


def test_exact_substitution_retains_capacity_and_protected_memberships():
    assert revised_ids((2, 5), 2, 7, (0, 1), (3, 4), 8) == (5, 7)
    for donor, receiver in ((0, 7), (2, 1), (2, 3), (2, 8), (2, 5)):
        with pytest.raises(ValueError):
            revised_ids((2, 5), donor, receiver, (0, 1), (3, 4), 8)


def test_real_readout_choice_is_deterministic_and_greedy_token_safe():
    anchor = dict(name="identity", kl=0.2, ordinal=0, greedy_token=3)
    changed_token = dict(name="invalid", kl=0.001, ordinal=1, greedy_token=4)
    stable = dict(name="safe", kl=0.1, ordinal=2, greedy_token=3)
    assert choose([anchor, changed_token, stable]) == stable
    assert choose([anchor, dict(stable, kl=0.2)]) == anchor
    with pytest.raises(ValueError):
        choose([dict(anchor, kl=float("nan"))])


def test_shortlist_uses_no_labels_and_keeps_one_head_equal_capacity():
    torch.manual_seed(84)
    original = tuple(torch.randn(9, 4) for _ in range(2))
    active = tuple(torch.randn(1, 2, 10, 4) for _ in range(2))
    data = dict(
        query=torch.randn(1, 2, 1, 4),
        scaling=0.5,
        mask=None,
        active_key=active[0],
        active_value=active[1],
        parent_key=active[0][:, 1:2].clone(),
        parent_value=active[1][:, 1:2].clone(),
    )
    kw = dict(head=1, exact=(2, 5), receivers=(6, 7, 8), protected=(0, 1), members=(3, 4))
    before = tuple(t.clone() for t in active)
    result = shortlist(data, original, **kw)
    assert result == shortlist(data, original, **kw)
    assert len(result) == 4 and len(set(r[1] for r in result)) == 4
    assert all(len(r[1]) == 2 for r in result)
    assert all(torch.equal(a, b) for a, b in zip(active, before, strict=True))
    with pytest.raises(ValueError):
        shortlist(data, original, **kw, maximum=5)


def test_readout_policy_cannot_silently_use_earlier_layers_or_hide_draft_forecast():
    for changes in (dict(layers=(30,)), dict(provisional_tokens=128), dict(readout_control=True)):
        with pytest.raises(ValueError):
            OverlayPolicy(readout_selection="joint", **changes)
    from storage_six_ablation import component_policies

    policies = component_policies(readout_selection_study=True)
    assert policies["readout_joint"]["readout_selection"] == "joint"
    assert policies["no_forecast"]["readout_selection"] == "none"
    assert policies["no_repair"]["readout_selection"] == "exact"
