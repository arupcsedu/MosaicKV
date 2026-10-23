"""Tiny random-weight CPU cache transport fixtures, no benchmark downloads."""

import pytest
import torch
from test_hf_tiny_models import _llava, _runtime_request

from mosaickv.adapters.huggingface.layer_local_cache import gather_layer_local_snapshot


def test_gather_preserves_per_head_source_values_and_has_no_padding_or_aliases():
    adapter = _llava(kv_heads=2)
    prepared = adapter.prepare_inputs(_runtime_request("gather").messages)
    prefill = adapter.prefill(prepared)
    source, _ = adapter._legacy_layers(prefill.state.past_key_values)
    positions = (((0, 2), (1, 3)), ((0, 1, 3), (0, 2, 3)))
    snapshot = gather_layer_local_snapshot(adapter, prefill.state.past_key_values, positions)
    assert snapshot.independent_layer_lengths
    assert snapshot.active_sequence_length == 3
    for original, layer, heads in zip(source, snapshot.layers, positions, strict=True):
        for head, indices in enumerate(heads):
            torch.testing.assert_close(
                layer.key[0, head], original[0][0, head, list(indices)], rtol=0, atol=0
            )
            torch.testing.assert_close(
                layer.value[0, head], original[1][0, head, list(indices)], rtol=0, atol=0
            )
        for tensor in (layer.key, layer.value):
            assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
            assert tensor.untyped_storage().data_ptr() not in {
                item.untyped_storage().data_ptr() for item in original
            }


@pytest.mark.parametrize(
    "bad", [(((0, 0),), ((0, 1),)), (((0, 4),), ((0, 1),)), (((2, 0),), ((0, 1),))]
)
def test_invalid_positions_fail_before_reinjection(bad):
    adapter = _llava()
    prepared = adapter.prepare_inputs(_runtime_request("invalid-gather").messages)
    prefill = adapter.prefill(prepared)
    with pytest.raises(ValueError, match="sorted, unique"):
        gather_layer_local_snapshot(adapter, prefill.state.past_key_values, bad)
