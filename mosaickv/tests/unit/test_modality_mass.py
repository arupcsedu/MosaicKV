import inspect
import math
from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.completion_masks import build_mask
from mosaickv.evaluation.modality_mass import CalibratedFormat, estimate, wrap


def evidence():
    prompt = torch.tensor([0.125, 0.125, 0.125, 0.125, 0.25, 0.25])
    return tuple((torch.cat((prompt, torch.zeros(s + 1)))[None, None, None],) for s in range(4))


def test_exact_visual_text_odds_restored_without_changing_text():
    biases, records = estimate(evidence(), (((0, 1, 4, 5),),), (4, 5), strength=1.0)
    expected = torch.tensor([[math.log(2), math.log(2), 0.0, 0.0]])
    assert torch.allclose(biases[0], expected)
    kept = torch.tensor([0.125, 0.125, 0.25, 0.25]) * biases[0][0].exp()
    assert abs(float(kept[:2].sum() / kept.sum()) - 0.5) < 1e-7
    assert not biases[0][:, -2:].any()
    assert records[0]["generated_history_bias_zero"]


def test_zero_strength_and_all_source_retention_are_identities():
    zero, _ = estimate(evidence(), (((0, 1, 4, 5),),), (4, 5), strength=0.0)
    full, _ = estimate(evidence(), ((tuple(range(6)),),), (4, 5), strength=1.0)
    assert not zero[0].any() and not full[0].any()


def test_zero_mass_is_marked_unidentified_and_never_makes_an_infinite_bias():
    steps = tuple((p.clone(),) for (p,) in evidence())
    for (p,) in steps:
        p[..., :4] = 0
    biases, records = estimate(steps, (((0, 1, 4, 5),),), (4, 5), strength=1.0)
    assert not biases[0].any()
    assert records[0]["unidentifiable_heads_bypassed"] == 1
    assert records[0]["per_draft_query_log_odds_error"] == [[None]] * 4


def test_half_strength_preserves_individual_forecast_queries_before_reduction():
    attn = list(evidence())
    changed = attn[3][0].clone()
    changed[..., 0] *= 0.1
    changed[..., 2] *= 1.9
    attn[3] = (changed,)
    biases, records = estimate(tuple(attn), (((0, 1, 4, 5),),), (4, 5), strength=0.5)
    individual = records[0]["per_draft_query_log_odds_error"]
    assert individual[0] != individual[3]
    assert float(biases[0].abs().max()) <= math.log(2) / 2 + 1e-7


@pytest.mark.parametrize("failure", ["strength", "missing_step", "no_text", "no_visual", "padding"])
def test_malformed_evidence_fails_closed(failure):
    steps, memberships, text, strength = evidence(), (((0, 1, 4, 5),),), (4, 5), 1.0
    if failure == "strength":
        strength = 0.75
    elif failure == "missing_step":
        steps = steps[:3]
    elif failure == "no_text":
        text = ()
    elif failure == "no_visual":
        text = tuple(range(6))
    else:
        memberships = (((0, None, 4, 5),),)
    with pytest.raises(ValueError):
        estimate(steps, memberships, text, strength=strength)


def test_source_positions_padding_and_history_never_receive_visual_bias():
    biases, _ = estimate(evidence(), (((0, 4, 5, None),),), (4, 5), strength=1.0)
    counts = torch.tensor([3], dtype=torch.int16)
    result = build_mask(counts, 7, 4, torch.bfloat16, None, log_mass_bias=biases[0])
    assert result[0, 0, 0, 0] > 0
    assert not result[0, 0, 0, 1:3].any()
    assert result[0, 0, 0, 3] == torch.finfo(torch.bfloat16).min
    assert not result[..., 4:].any()


def test_wrapper_counts_all_persistent_metadata_and_never_changes_original_values():
    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes

    original = torch.arange(8).bfloat16().reshape(1, 1, 4, 2)

    def fill(attention, targets, *, direct):
        for t in targets:
            t.copy_(original)

    base = SimpleNamespace(
        shape=original.shape,
        private=lambda: (original,),
        shared=lambda: (),
        fill=fill,
        materialize=lambda attention: (original, original),
    )
    bias = torch.tensor([[math.log(2), math.log(2), 0.0, 0.0]])
    (wrapped,) = wrap((base,), (bias,), (((0, 1, 4, 5),),))
    assert wrapped.component_bytes == backing_bytes((original, bias, wrapped.valid_prompt_counts))
    assert wrapped.component_bytes > backing_bytes((original,))
    targets = (torch.empty_like(original), torch.empty_like(original))
    wrapped.fill(None, targets, direct=True)
    assert all(torch.equal(t.view(torch.int16), original.view(torch.int16)) for t in targets)
    with pytest.raises(ValueError):
        CalibratedFormat(base, bias * 2, wrapped.valid_prompt_counts)


def test_prospective_estimator_cannot_accept_answers_or_reference_logits():
    assert set(inspect.signature(estimate).parameters) == {
        "attention_steps",
        "memberships",
        "text_positions",
        "strength",
    }
