from __future__ import annotations

import pytest
import torch

from mosaickv.adapters.huggingface import DecodeState, HuggingFaceMultimodalAdapter
from mosaickv.adapters.huggingface.preallocated_cache import (
    PreallocatedDynamicLayer,
    build_preallocated_dynamic_cache,
)


def test_preallocated_layer_appends_without_replacing_storage() -> None:
    keys = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4)
    values = keys + 100
    layer = PreallocatedDynamicLayer(keys, values, additional_positions=2)
    key_pointer = layer.keys.untyped_storage().data_ptr()
    value_pointer = layer.values.untyped_storage().data_ptr()

    observed_keys, observed_values = layer.update(
        torch.full((1, 1, 1, 4), 7.0),
        torch.full((1, 1, 1, 4), 9.0),
        {"cache_position": torch.tensor([99])},
    )

    assert layer.get_seq_length() == 4
    assert layer.get_mask_sizes(torch.tensor([99])) == (5, 0)
    assert observed_keys.untyped_storage().data_ptr() == key_pointer
    assert observed_values.untyped_storage().data_ptr() == value_pointer
    torch.testing.assert_close(observed_keys[..., :3, :], keys)
    torch.testing.assert_close(observed_values[..., :3, :], values)
    assert observed_keys[..., 3, :].tolist() == [[[7.0, 7.0, 7.0, 7.0]]]


def test_preallocated_layer_fails_when_reserved_capacity_is_exceeded() -> None:
    layer = PreallocatedDynamicLayer(
        torch.zeros((1, 1, 2, 4)),
        torch.zeros((1, 1, 2, 4)),
        additional_positions=1,
    )
    layer.update(torch.ones((1, 1, 1, 4)), torch.ones((1, 1, 1, 4)))

    with pytest.raises(RuntimeError, match="capacity exceeded"):
        layer.update(torch.ones((1, 1, 1, 4)), torch.ones((1, 1, 1, 4)))


def test_preallocated_dynamic_cache_reports_only_occupied_prefix() -> None:
    keys = torch.ones((1, 2, 3, 4))
    cache = build_preallocated_dynamic_cache(((keys, keys.clone()),), 4)

    assert cache.get_seq_length() == 3
    updated_keys, _updated_values = cache.update(
        torch.full((1, 2, 1, 4), 2.0),
        torch.full((1, 2, 1, 4), 3.0),
        0,
        {"cache_position": torch.tensor([100])},
    )

    assert cache.get_seq_length() == 4
    assert tuple(updated_keys.shape) == (1, 2, 4, 4)
    assert updated_keys.untyped_storage().nbytes() == 1 * 2 * 7 * 4 * 4


def test_preallocated_provisional_step_rolls_back_without_copying_storage() -> None:
    keys = torch.ones((1, 1, 3, 4))
    cache = build_preallocated_dynamic_cache(((keys, keys.clone()),), 2)
    pointer = cache.layers[0].keys.untyped_storage().data_ptr()
    cache.update(
        torch.full((1, 1, 1, 4), 2.0),
        torch.full((1, 1, 1, 4), 3.0),
        0,
    )
    state = DecodeState(
        past_key_values=cache,
        attention_mask=torch.ones((1, 4), dtype=torch.long),
        active_cache_length=4,
        logical_sequence_length=4,
        next_decode_position=4,
        modality_map=(),
        model_state={"mosaickv_decode_cache_storage": "preallocated_append_only"},
    )

    rolled_back = HuggingFaceMultimodalAdapter.rollback_preallocated_decode_step(
        None,  # type: ignore[arg-type]
        state,
        active_cache_length=3,
        logical_sequence_length=3,
        next_decode_position=3,
    )

    assert rolled_back
    assert cache.get_seq_length() == 3
    assert state.active_cache_length == 3
    assert state.logical_sequence_length == 3
    assert state.next_decode_position == 3
    assert state.attention_mask.shape[-1] == 3
    assert cache.layers[0].keys.untyped_storage().data_ptr() == pointer
