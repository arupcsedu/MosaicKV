"""Random tiny CPU model; no benchmark labels or downloaded weights."""

import pytest
import torch
from test_hf_tiny_models import _llava, _runtime_request

from mosaickv.evaluation.delimiter_control import source_sha
from mosaickv.evaluation.projection_archive import (
    ProjectionCapture,
    build_archive,
    reconstruct_archive,
)


@pytest.mark.parametrize("encoded", [False, True])
@pytest.mark.parametrize("coherent", [False, True])
def test_shared_normalized_source_reconstructs_exact_selected_post_rope_KV(encoded, coherent):
    torch.manual_seed(33)
    adapter = _llava(kv_heads=2)
    adapter.model.to(torch.bfloat16)
    with ProjectionCapture(adapter) as capture:
        prefill = adapter.prefill(adapter.prepare_inputs(_runtime_request("archive").messages))
    capture.validate()
    pairs, _ = adapter._legacy_layers(prefill.state.past_key_values)
    before = source_sha(pairs)
    selections = ((0, 3), (0, 3)) if coherent else ((0, 2), (1, 3))
    for index, (key, value) in enumerate(pairs):
        archive = build_archive(capture.records[index], selections, encoded=encoded)
        reconstructed = reconstruct_archive(archive, adapter._language_layers()[index].self_attn)
        selected = torch.tensor(selections)[None, :, :, None].expand(1, 2, 2, 8)
        for original, actual in zip((key, value), reconstructed, strict=True):
            assert torch.equal(original.gather(2, selected), actual)
    assert source_sha(pairs) == before
    assert all(not layer.self_attn._forward_pre_hooks for layer in adapter._language_layers())


def test_projection_capture_hooks_restore_on_failure_and_reject_ragged_source():
    adapter = _llava(kv_heads=2)
    adapter.model.to(torch.bfloat16)
    with pytest.raises(RuntimeError, match="fixture"), ProjectionCapture(adapter):
        raise RuntimeError("fixture")
    assert all(not layer.self_attn._forward_pre_hooks for layer in adapter._language_layers())
    source = dict(normalized=torch.ones(1, 4, 16, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="uniform"):
        build_archive(source, ((0,), (1, 3)), encoded=False)
