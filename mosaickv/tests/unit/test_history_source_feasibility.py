from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.history_source_feasibility import (
    HistoryProjectionCapture,
    packet_feasibility,
)


def fake_adapter(*, fail=False):
    class LlamaAttention(torch.nn.Module):
        def forward(self, *, hidden_states, past_key_value, **kwargs):
            if fail:
                raise RuntimeError("synthetic forward failure")
            layer = past_key_value.layers[0]
            layer.tail_used += 1
            for tensor in layer.history:
                tensor[..., layer.tail_used - 1 : layer.tail_used, :].copy_(hidden_states[:, None])
            return hidden_states, None

    LlamaAttention.__module__ = "transformers.models.llama.modeling_llama"
    attention = LlamaAttention()
    adapter = SimpleNamespace(_language_layers=lambda: [SimpleNamespace(self_attn=attention)])
    layer = SimpleNamespace(
        tail_used=0, history=tuple(torch.zeros(1, 1, 4, 4, dtype=torch.bfloat16) for _ in range(2))
    )
    return adapter, attention, SimpleNamespace(layers=[layer])


def test_capture_preserves_values_and_original_logical_not_physical_positions():
    adapter, attention, cache = fake_adapter()
    hidden = torch.arange(4).reshape(1, 1, 4).bfloat16()
    phase = (torch.ones(1, 1, 4).bfloat16(), torch.zeros(1, 1, 4).bfloat16())
    before = hidden.clone()
    with HistoryProjectionCapture(adapter) as capture:
        for i in range(2):
            actual, _ = attention(
                hidden_states=hidden,
                past_key_value=cache,
                position_embeddings=phase,
                position_ids=torch.tensor([[600 + i]]),
                cache_position=torch.tensor([300 + i]),
            )
            assert torch.equal(actual, before)
    assert not attention._forward_hooks and not attention._forward_pre_hooks
    record = capture.assemble(2)[0]
    assert record["logical_positions"].tolist() == [600, 601]
    assert torch.equal(record["normalized"], before.expand(1, 2, 4))
    hidden.zero_()
    for tensor in cache.layers[0].history:
        tensor.zero_()
    assert torch.equal(record["key"][0, 0], before[0].expand(2, 4))
    assert torch.equal(record["value"], record["key"])


def test_failed_forward_removes_hooks_and_incomplete_capture_fails_closed():
    adapter, attention, cache = fake_adapter(fail=True)
    with (
        pytest.raises(RuntimeError, match="synthetic forward failure"),
        HistoryProjectionCapture(adapter) as capture,
    ):
        attention(
            hidden_states=torch.zeros(1, 1, 4, dtype=torch.bfloat16),
            past_key_value=cache,
            position_embeddings=(torch.ones(1, 1, 4),) * 2,
            position_ids=torch.tensor([[600]]),
            cache_position=torch.tensor([300]),
        )
    assert not attention._forward_hooks and not attention._forward_pre_hooks
    with pytest.raises(ValueError, match="complete observed"):
        capture.assemble(1)


def test_nonconsecutive_original_positions_fail_closed():
    adapter, attention, cache = fake_adapter()
    with HistoryProjectionCapture(adapter) as capture:
        for position in (600, 602):
            attention(
                hidden_states=torch.zeros(1, 1, 4, dtype=torch.bfloat16),
                past_key_value=cache,
                position_embeddings=(torch.ones(1, 1, 4),) * 2,
                position_ids=torch.tensor([[position]]),
                cache_position=torch.tensor([300]),
            )
    with pytest.raises(ValueError, match="consecutive"):
        capture.assemble(2)


@pytest.mark.parametrize("updates", [0, 1, 16, 128, True])
def test_packet_scope_rejects_unsupported_updates_before_model_access(updates):
    with pytest.raises(ValueError, match="bounded complete"):
        packet_feasibility(None, {}, updates=updates)
