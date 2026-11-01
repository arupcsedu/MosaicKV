from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.source_draft_capture import Capture, collect, tensor_hash
from mosaickv.evaluation.source_query_importance import (
    blend_families,
    future_importance_cpu,
    future_importance_cuda,
)


def test_families_are_query_count_normalized_and_empty_fails():
    x = torch.tensor([2.0, 4.0])
    y = torch.tensor([4.0, 8.0])
    assert torch.equal(blend_families(x, y, 2, 4), x / 2)
    assert torch.equal(blend_families(x * 3, y, 6, 4), x / 2)
    for count in (0, -1, False):
        with pytest.raises(ValueError):
            blend_families(x, y, count, 4)


def test_draft_probabilities_preserve_missing_history_mass():
    h, n, d = 1, 3, 128
    q = torch.ones(1, h, 2, d).bfloat16()
    v = torch.zeros(1, h, n, d).bfloat16()
    w = torch.eye(d)
    phases = (torch.ones(1, n, d), torch.zeros(1, n, d))
    exact = torch.zeros(h, n, dtype=torch.bool)
    p = torch.ones(1, h, 2, n) / 12
    o = torch.ones_like(q)
    a = future_importance_cpu(q, v, w, w, phases, exact, p, o)
    b = future_importance_cpu(q, v, w, w, phases, exact, p * 2, o)
    torch.testing.assert_close(b, a * 4)
    with pytest.raises(ValueError):
        future_importance_cuda(q, v, w, w, phases, exact, p, o)
    with pytest.raises(ValueError):
        future_importance_cpu(q, v, w, w, phases, exact, p * 20, o)
    exact[:] = True
    assert not future_importance_cpu(q, v, w, w, phases, exact, p, o).any()


def test_cpu_collect_cannot_claim_cuda_forecast():
    with pytest.raises(NotImplementedError):
        collect(SimpleNamespace(device=torch.device("cpu")), None)


def test_hooks_clean_up_even_on_exception_and_reject_second_prefill():
    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Identity()
            self.o_proj = torch.nn.Identity()

        def forward(self, hidden_states, **kwargs):
            return hidden_states, None

    attention = Attention()
    adapter = SimpleNamespace(_language_layers=lambda: [SimpleNamespace(self_attn=attention)])
    with pytest.raises(ValueError, match="reuse prefill"), Capture(adapter, length=8):
        attention(hidden_states=torch.ones(1, 2, 128))
    for module in (attention, attention.q_proj, attention.o_proj):
        assert not module._forward_hooks and not module._forward_pre_hooks


def test_capture_preserves_actual_probabilities_and_private_history():
    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Identity()
            self.o_proj = torch.nn.Identity()

        def forward(self, hidden_states, *, position_embeddings, past_key_value):
            q = self.q_proj(hidden_states)
            self.o_proj(q)
            step = past_key_value.layers[0].history_length
            return q, torch.full((1, 1, 1, 3 + step), 0.125)

    attention = Attention()
    layer = SimpleNamespace(
        history_length=0, history=(torch.ones(1, 1, 4, 128), torch.ones(1, 1, 4, 128))
    )
    cache = SimpleNamespace(layers=[layer])
    adapter = SimpleNamespace(_language_layers=lambda: [SimpleNamespace(self_attn=attention)])
    with Capture(adapter, length=3) as capture:
        for step in range(4):
            layer.history_length = step + 1
            attention(
                hidden_states=torch.ones(1, 1, 128),
                position_embeddings=(torch.ones(1, 1, 128), torch.zeros(1, 1, 128)),
                past_key_value=cache,
            )
        evidence = capture.finish()[0]
    assert evidence["query"].shape == (1, 1, 4, 128)
    assert (evidence["probability"].sum(-1) == 0.375).all()
    assert evidence["history"][0].data_ptr() != layer.history[0].data_ptr()
    assert capture.cache is None and capture.layers == []


def test_integrity_hash_detects_value_mutation():
    value = torch.ones(1, 2, 3, 4).bfloat16()
    before = tensor_hash((value,))
    value[0, 0, 0, 0] = 2
    assert tensor_hash((value,)) != before
