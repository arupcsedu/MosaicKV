from types import SimpleNamespace

import pytest
import torch
from transformers.models.llama import modeling_llama as llama

from mosaickv.evaluation import post_attention_segment as segment


def layer(dtype=torch.bfloat16):
    config = llama.LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=4,
        hidden_act="silu",
        attention_bias=False,
        mlp_bias=False,
    )
    return llama.LlamaDecoderLayer(config, 0).to(dtype=dtype).eval()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_explicit_buffers_reproduce_the_pinned_reference_operator_sequence(dtype):
    torch.manual_seed(20261105)
    model, observations = layer(dtype), []
    work = segment.Workspace(model)
    pointers = [t.data_ptr() for t in work.tensors()]
    versions = [p._version for p in model.parameters()]
    with torch.inference_mode():
        for scale in (0.0, 0.01, 1.0, 50.0):
            residual = torch.randn(1, 1, 64).to(dtype) * scale
            attention = torch.randn_like(residual)
            hidden = residual + attention
            reference = hidden + model.mlp(model.post_attention_layernorm(hidden))
            work.fill(residual, attention)
            actual = work.compute(model)
            assert torch.equal(
                actual.view(torch.int16 if dtype == torch.bfloat16 else torch.int32),
                reference.view(torch.int16 if dtype == torch.bfloat16 else torch.int32),
            )
            observations.append(actual.clone())
    assert pointers == [t.data_ptr() for t in work.tensors()]
    assert versions == [p._version for p in model.parameters()]
    assert len(observations) == 4


def test_default_off_does_not_allocate_workspace_or_wrap_layers():
    assert segment.transport(None, None).__enter__() is None


def test_request_scope_is_explicit_non_nested_and_not_reused():
    @segment.managed
    def call(**kw):
        return [], [], {}

    assert call()[2]["post_attention_segment"] == {"enabled": False}
    with pytest.raises(ValueError, match="boolean"):
        call(post_attention_segment=1)
    with pytest.raises(RuntimeError, match="did not engage"):
        call(post_attention_segment=True)
    assert segment._scope.get() is None


def test_non_pinned_layers_are_rejected_before_allocating_workspace():
    state = dict(enabled=True, report=None)
    adapter = SimpleNamespace(_language_layers=lambda: [layer()])
    with pytest.raises(NotImplementedError, match="pinned"):
        segment.Transport(adapter, SimpleNamespace(prompt_graph_pool=object()), state).__enter__()
