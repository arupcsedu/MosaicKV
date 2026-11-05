from types import SimpleNamespace

import pytest
import torch
from transformers.models.llama import modeling_llama as llama

from mosaickv.evaluation import input_norm_segment as segment


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_exact_operator_order_no_weight_or_pointer_mutation(dtype):
    norm = llama.LlamaRMSNorm(64).to(dtype=dtype).eval()
    workspace = segment.Workspace(norm)
    pointers, version = [t.data_ptr() for t in workspace.tensors()], norm.weight._version
    with torch.inference_mode():
        for scale in (0.0, 0.01, 1.0, 50.0):
            hidden = torch.randn(1, 1, 64, dtype=dtype) * scale
            workspace.input.copy_(hidden.reshape_as(workspace.input))
            actual = workspace.compute(norm)
            reference = norm(hidden)
            word = torch.int16 if dtype == torch.bfloat16 else torch.int32
            assert torch.equal(actual.view(word), reference.view(word))
    assert norm.weight._version == version
    assert pointers == [t.data_ptr() for t in workspace.tensors()]


def test_default_off_and_fail_closed_scope():
    assert segment.transport(None, None).__enter__() is None

    @segment.managed
    def call(**kwargs):
        return [], [], {}

    assert call()[2]["common_input_norm"] == dict(enabled=False)
    with pytest.raises(ValueError, match="boolean"):
        call(input_norm_segment=1)
    with pytest.raises(ValueError, match="post-attention"):
        call(input_norm_segment=True)
    with pytest.raises(RuntimeError, match="did not engage"):
        call(input_norm_segment=True, post_attention_segment=True)
    assert segment._scope.get() is None


def test_reject_unpinned_layers_before_workspace_allocation():
    adapter = SimpleNamespace(_language_layers=lambda: [])
    with pytest.raises(NotImplementedError, match="pinned"):
        segment.Transport(adapter, SimpleNamespace(prompt_graph_pool=None), {}).__enter__()
