import pytest
import torch
from test_hf_tiny_models import _llava, _runtime_request

from mosaickv.evaluation.direct_constructor import base_formats
from mosaickv.evaluation.exact_storage import formats_for_layer
from mosaickv.evaluation.projection_archive import ProjectionCapture


@pytest.mark.parametrize("shared", [True, False])
def test_direct_payloads_match_reference_format_costs_bits_and_maps(shared):
    torch.manual_seed(42)
    adapter = _llava(kv_heads=2)
    adapter.model.to(torch.bfloat16)
    with ProjectionCapture(adapter) as capture:
        prefill = adapter.prefill(adapter.prepare_inputs(_runtime_request("direct").messages))
    original, _ = adapter._legacy_layers(prefill.state.past_key_values)
    key, value = original[0]
    positions = ((0, 1, 3), (0, 1, 3)) if shared else ((0, 1, 3), (0, 2, 3))
    attention = adapter._language_layers()[0].self_attn
    expected, _ = formats_for_layer(capture.records[0], attention, key, value, positions)
    actual, reference = base_formats(capture.records[0], key, value, positions)
    assert [f.kind for f in actual] == [f.kind for f in expected]
    assert [f.component_bytes for f in actual] == [f.component_bytes for f in expected]
    for a, b in zip(actual, expected, strict=True):
        for tensor, baseline in zip(
            a.materialize(attention), b.materialize(attention), strict=True
        ):
            assert torch.equal(tensor.view(torch.int16), baseline.view(torch.int16))
        if a.archive is not None:
            assert torch.equal(a.archive.union, b.archive.union)
            assert torch.equal(a.archive.selected, b.archive.selected)
            assert a.archive.selected.stride(0) == b.archive.selected.stride(0)
        for tensor, baseline in zip(a.materialize(attention), reference, strict=True):
            assert torch.equal(tensor.view(torch.int16), baseline.view(torch.int16))
