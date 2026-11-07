from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.live_anchor_guard import LiveAnchorGuard, _current, decision
from mosaickv.evaluation.storage_six_overlay import OverlayPolicy


def test_live_decision_keeps_registered_thresholds_and_greedy_tokens():
    original = torch.tensor([4.0, 1.0, 0.0])
    assert bool(decision(original, original)[0])
    assert bool(decision(original, original + 0.02)[0])
    assert bool(decision(original, original + 1.0)[0])
    assert not bool(decision(original, original + 1.01)[0])
    assert not bool(decision(original, torch.tensor([3.0, 1.9, 0.9]))[0])
    assert not bool(decision(original, torch.tensor([0.0, 4.0, 0.0]))[0])
    with pytest.raises(ValueError, match="nonfinite"):
        decision(original, torch.full((3,), float("nan")))
    with pytest.raises(ValueError):
        decision(original, original[None])


def test_live_guard_scope_restores_patches_and_parent_owners_on_failure():
    from mosaickv.evaluation.uniform_core_transport import Binding

    score = Binding.scores
    report = {}
    g = LiveAnchorGuard(SimpleNamespace(), SimpleNamespace(head=0), report)
    with pytest.raises(RuntimeError, match="controlled"), g:
        assert _current.get() is g
        raise RuntimeError("controlled")
    assert _current.get() is None and Binding.scores is score
    assert report["live_anchor_guard"]["steps"] == 0
    assert g.parent_key is g.parent_value is None
    with pytest.raises(ValueError):
        OverlayPolicy(live_anchor_guard=True)
    assert OverlayPolicy(inline_repair=True, live_anchor_guard=True)


def test_parent_snapshots_own_only_one_head_and_do_not_alias_the_live_lease():
    observed = []
    lease = torch.randn(1, 32, 9, 128).bfloat16()
    layer = SimpleNamespace(
        attention=SimpleNamespace(layer_idx=31),
        leased=(lease,),
        get_seq_length=lambda: 7,
        observe=lambda t: observed.append(t),
    )
    g = LiveAnchorGuard(SimpleNamespace(), SimpleNamespace(head=4), {})
    g.record_component(layer, 0)
    assert g.parent_key.shape == (1, 1, 7, 128)
    original = g.parent_key.clone()
    lease.zero_()
    assert torch.equal(g.parent_key, original)
    assert g.parent_key.untyped_storage().nbytes() == 7 * 128 * 2
    assert observed and g.peak_temporary_kv_bytes == 7 * 128 * 2
    with pytest.raises(RuntimeError, match="not consumed"):
        g.record_component(layer, 0)


@pytest.mark.parametrize("per_head", [False, True])
def test_single_head_counterfactual_preserves_native_head_values_and_masks(per_head):
    from mosaickv.evaluation.eager_core_operator import reference_scores
    from mosaickv.evaluation.live_anchor_guard import head_mask, replace_head_output

    generator = torch.Generator().manual_seed(71)
    query = torch.randn(1, 32, 1, 128, generator=generator).bfloat16()
    key = torch.randn(1, 32, 33, 128, generator=generator).bfloat16()
    value = torch.randn(1, 32, 33, 128, generator=generator).bfloat16()
    mask = torch.zeros(1, 32 if per_head else 1, 1, 33).bfloat16()
    mask[..., -3:] = float("-inf")
    scores = reference_scores(query, key, mask, 128**-0.5)
    original = torch.matmul(scores, value).transpose(1, 2).contiguous().reshape(1, 1, -1)
    parent_key, parent_value = key[:, 4:5] + 0.1, value[:, 4:5] - 0.1
    key[:, 4:5], value[:, 4:5] = parent_key, parent_value
    reference = torch.matmul(reference_scores(query, key, mask, 128**-0.5), value)
    reference = reference.transpose(1, 2).contiguous().reshape(1, 1, -1)
    single = reference_scores(query[:, 4:5], parent_key, head_mask(mask, 4), 128**-0.5)
    actual = replace_head_output(original, single, parent_value, 4)
    assert torch.equal(actual, reference)
    assert not torch.equal(actual, original)
    assert head_mask(None, 4) is None
    with pytest.raises(ValueError, match="mask"):
        head_mask(torch.zeros(1, 2, 1, 33), 4)
    with pytest.raises(ValueError, match="head/output"):
        replace_head_output(original, single, parent_value, 32)


def test_single_head_scope_is_explicit_and_defaults_to_reference():
    assert OverlayPolicy().anchor_counterfactual == "full_heads"
    with pytest.raises(ValueError, match="counterfactual"):
        OverlayPolicy(anchor_counterfactual="single_head")
    policy = OverlayPolicy(
        inline_repair=True, live_anchor_guard=True, anchor_counterfactual="single_head"
    )
    assert LiveAnchorGuard(
        SimpleNamespace(), SimpleNamespace(head=0), {}, policy=policy
    ).single_head


def test_unused_parent_export_can_be_omitted_without_skipping_any_guard_decision():
    policy = OverlayPolicy(export_parent_logits=False)
    g = LiveAnchorGuard(None, None, {}, policy=policy)
    parent = torch.tensor([3.0, 1.0, -1.0])
    _, record = decision(parent, parent)
    g.record_readout(parent, record)
    assert g.observed_parent_logit_steps == 1 and len(g.records) == 1
    assert not g.parent_logits
    assert torch.equal(g.records[0], record)
    original = LiveAnchorGuard(None, None, {})
    original.record_readout(parent, record)
    assert len(original.parent_logits) == 1


def test_explicit_diagnostic_capture_requires_full_parent_logit_vectors():
    from mosaickv.evaluation.storage_six_overlay import integrated_storage

    with (
        pytest.raises(ValueError, match="complete parent-logit"),
        integrated_storage(
            None, OverlayPolicy(export_parent_logits=False), diagnostic_capture=lambda _: None
        ),
    ):
        pass
