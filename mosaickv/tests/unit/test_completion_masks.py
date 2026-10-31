from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.completion_masks import CompletionMaskTransport, build_mask


def test_padding_is_excluded_and_generated_history_is_valid():
    counts = torch.tensor([3, 5], dtype=torch.int16)
    old = torch.zeros(1, 1, 1, 9, dtype=torch.bfloat16)
    old[..., 0] = -torch.inf
    copied = old.clone()
    actual = build_mask(counts, 8, 5, torch.bfloat16, old)
    assert actual.shape == (1, 2, 1, 8)
    assert torch.isneginf(actual[:, :, :, 0]).all()
    assert (actual[0, 0, 0, 3:5] == torch.finfo(torch.bfloat16).min).all()
    assert not actual[:, :, :, 5:].any()
    assert not actual[0, 1, 0, 1:].any()
    assert torch.equal(old, copied)


def test_all_valid_zero_mask_and_invalid_shapes():
    counts = torch.tensor([5, 5], dtype=torch.int16)
    assert not build_mask(counts, 7, 5, torch.bfloat16, None).any()
    with pytest.raises(ValueError):
        build_mask(counts, 4, 5, torch.bfloat16, None)
    with pytest.raises(ValueError):
        build_mask(counts.float(), 7, 5, torch.bfloat16, None)


@pytest.mark.parametrize("fail", [False, True])
def test_request_cleanup_and_mask_backing_accounted(fail):
    class Attention(torch.nn.Module):
        config = SimpleNamespace(_attn_implementation="eager")

        def forward(self, hidden_states, attention_mask, past_key_value):
            if fail:
                raise RuntimeError("sentinel")
            return attention_mask

    attention = Attention()
    fmt = SimpleNamespace(
        shape=(1, 2, 4, 128), valid_prompt_counts=torch.tensor([3, 4], dtype=torch.int16)
    )
    cache = SimpleNamespace(
        layers=[SimpleNamespace(representation=fmt, get_seq_length=lambda: 5)],
        shared_tensors=(),
        shared_bytes=0,
        observed=[],
    )
    cache.observe = lambda: cache.observed.append(cache.shared_bytes)
    adapter = SimpleNamespace(_language_layers=lambda: [SimpleNamespace(self_attn=attention)])
    try:
        with CompletionMaskTransport(adapter, cache):
            mask = attention(
                hidden_states=torch.ones(1, 1, 4).bfloat16(),
                attention_mask=None,
                past_key_value=cache,
            )
            assert mask.shape == (1, 2, 1, 6)
            assert mask[0, 0, 0, 3] < 0 and not mask[:, :, :, 4:].any()
    except RuntimeError:
        assert fail
    assert cache.observed == [24]
    assert cache.shared_bytes == 0 and cache.shared_tensors == ()
    assert not attention._forward_hooks and not attention._forward_pre_hooks


@pytest.mark.parametrize("fail", [False, True])
def test_cleanup_preserves_persistent_owners_added_by_history_transport(fail):
    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes

    original = torch.ones(1, dtype=torch.bfloat16)
    phase = torch.ones(1, 1, 128, dtype=torch.bfloat16)

    class Attention(torch.nn.Module):
        config = SimpleNamespace(_attn_implementation="eager")

        def forward(self, hidden_states, attention_mask, past_key_value):
            past_key_value.shared_tensors = (*past_key_value.shared_tensors, phase)
            if fail:
                raise RuntimeError("later history failure")
            return hidden_states

    attention = Attention()
    cache = SimpleNamespace(
        layers=[
            SimpleNamespace(
                representation=SimpleNamespace(
                    shape=(1, 2, 4, 128),
                    valid_prompt_counts=torch.tensor([3, 4], dtype=torch.int16),
                ),
                get_seq_length=lambda: 5,
            )
        ],
        shared_tensors=(original,),
        shared_bytes=2,
        observe=lambda: None,
    )
    adapter = SimpleNamespace(_language_layers=lambda: [SimpleNamespace(self_attn=attention)])
    try:
        with CompletionMaskTransport(adapter, cache):
            attention(
                hidden_states=torch.ones(1, 1, 4).bfloat16(),
                attention_mask=None,
                past_key_value=cache,
            )
    except RuntimeError:
        assert fail
    assert len(cache.shared_tensors) == 2
    assert cache.shared_tensors[0] is original and cache.shared_tensors[1] is phase
    assert cache.shared_bytes == backing_bytes((original, phase))
    assert not attention._forward_hooks and not attention._forward_pre_hooks
