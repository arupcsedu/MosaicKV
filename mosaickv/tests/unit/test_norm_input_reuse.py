from types import SimpleNamespace

import pytest
import torch
from transformers.models.llama import modeling_llama as llama

from mosaickv.evaluation.input_norm_segment import Transport, Workspace, managed


def setup():
    norm = llama.LlamaRMSNorm(64).bfloat16().eval()
    cache = object()
    owner = Transport(SimpleNamespace(_language_layers=lambda: []), cache, dict(reuse=True))
    owner.workspace = Workspace(norm)
    segment = SimpleNamespace(
        cache=cache,
        closed=False,
        workspace=SimpleNamespace(output=torch.empty(1, 64, dtype=torch.bfloat16)),
    )
    owner.bind_segment(segment)
    return norm, owner, segment


def test_early_embedding_copy_then_no_later_copy_matches_original():
    norm, owner, segment = setup()
    pointers = [t.data_ptr() for t in owner.workspace.tensors()]
    with torch.inference_mode():
        embedding = torch.randn(1, 1, 64).bfloat16()
        owner.prepare_input(embedding, 0)
        assert torch.equal(
            segment.workspace.output.view(torch.int16), embedding.reshape(1, 64).view(torch.int16)
        )
        for index in range(1, 32):
            hidden = segment.workspace.output.view(1, 1, 64)
            owner.prepare_input(hidden, index)
            actual = owner.workspace.compute(norm, owner.shared_input).clone()
            assert torch.equal(actual.view(torch.int16), norm(hidden).view(torch.int16))
            segment.workspace.output.copy_(torch.randn_like(segment.workspace.output))
    assert owner.first_layer_copies == 1 and owner.skipped_copies == 31
    assert pointers == [t.data_ptr() for t in owner.workspace.tensors()]


def test_request_alias_guards_fail_closed():
    _, owner, segment = setup()
    with pytest.raises(RuntimeError, match="exactly once"):
        owner.bind_segment(segment)
    with pytest.raises(RuntimeError, match="later layers"):
        owner.prepare_input(torch.ones(1, 1, 64).bfloat16(), 1)
    with pytest.raises(RuntimeError, match="embedding"):
        owner.prepare_input(segment.workspace.output, 0)


def test_reuse_requires_explicit_normalization_and_boolean():
    @managed
    def call(**kwargs):
        return [], [], {}

    with pytest.raises(ValueError, match="reuse"):
        call(reuse_norm_input=True)
    with pytest.raises(ValueError, match="reuse"):
        call(reuse_norm_input=1)
