"""CPU fixed-weight compact correction and original selector preservation."""

import pytest
import torch
from test_hf_tiny_models import _llava, _runtime_request

from mosaickv.evaluation.compact_storage import build_compact, projection_policy
from mosaickv.evaluation.exact_storage import formats_for_layer
from mosaickv.evaluation.projection_archive import ProjectionCapture


@pytest.mark.parametrize("reduced", [False, True])
def test_compact_hybrid_restores_original_bits_without_changing_positions(reduced):
    torch.manual_seed(251)
    adapter = _llava(kv_heads=2)
    adapter.model.to(torch.bfloat16)
    with ProjectionCapture(adapter) as capture:
        prefill = adapter.prefill(adapter.prepare_inputs(_runtime_request("compact").messages))
    source, _ = adapter._legacy_layers(prefill.state.past_key_values)
    attention = adapter._language_layers()[0].self_attn
    positions = ((0, 1, 3), (0, 2, 3))
    choices, _ = formats_for_layer(capture.records[0], attention, *source[0], positions)
    base = next(f for f in choices if f.kind == "token_hybrid")
    index = torch.tensor(positions)[None, :, :, None].expand(1, 2, 3, 8)
    reference = tuple(t.gather(2, index) for t in source[0])
    original_flag = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    compact, proof = build_compact(base, attention, reference, reduced_precision=reduced)
    assert compact.slots.dtype == torch.int32
    assert compact.slots.shape == (2, 2)
    assert compact.source_map.dtype == torch.int32
    assert compact.source_map.tolist() == [0, -1, -1, 1]
    assert proof["repeated_original_KV_bitwise"]
    backing = tuple(torch.full((1, 2, 7, 8), -71, dtype=torch.bfloat16) for _ in range(2))
    compact.fill(attention, backing, direct=True)
    assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction == original_flag
    for a, b in zip(reference, backing, strict=True):
        assert torch.equal(a.view(torch.int16), b[..., :3, :].view(torch.int16))
        assert bool((b[..., 3:, :] == -71).all())
    assert torch.equal(compact.archive.selected, torch.tensor(positions))


def test_projection_policy_rolls_back_even_after_exception():
    before = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    with pytest.raises(RuntimeError), projection_policy(not before):
        assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction != before
        raise RuntimeError("test rollback")
    assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction == before
