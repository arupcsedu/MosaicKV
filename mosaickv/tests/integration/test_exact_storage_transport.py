"""Random-weight CPU transport; no downloaded weights or benchmark labels."""

import pytest
import torch
from test_hf_tiny_models import _llava, _runtime_request

from mosaickv.adapters.huggingface.types import DecodeState
from mosaickv.evaluation.exact_storage import construct, formats_for_layer
from mosaickv.evaluation.lossless_lifetime import standard_greedy_reference
from mosaickv.evaluation.lossless_prompt_cache import IsolatedStorageTransport
from mosaickv.evaluation.projection_archive import ProjectionCapture


@pytest.mark.parametrize("mode", ["kv", "token", "patched"])
@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("sealed", [False, True])
def test_all_storage_paths_match_original_32_token_greedy(mode, direct, sealed):
    torch.manual_seed(38)
    adapter = _llava(kv_heads=2)
    adapter.model.to(torch.bfloat16)
    with ProjectionCapture(adapter) as capture:
        prefill = adapter.prefill(adapter.prepare_inputs(_runtime_request(mode).messages))
    positions = (((0, 1, 3), (0, 2, 3)),) * 2  # Shared source and direct-only rows.
    reference = standard_greedy_reference(adapter, prefill, positions)
    cache = construct(
        adapter, prefill, capture, positions, mode=mode, direct=direct, sealed_history=sealed
    )
    state = DecodeState(
        cache,
        torch.ones(1, 3, dtype=torch.long),
        3,
        prefill.state.logical_sequence_length,
        prefill.state.next_decode_position,
        prefill.state.modality_map,
        dict(
            prefill.state.model_state, mosaickv_packed_all_valid=True, mosaickv_cache_all_valid=True
        ),
    )
    values, token = [prefill.logits.detach().reshape(-1).float()], prefill.next_token_id
    with IsolatedStorageTransport(adapter, cache):
        for _ in range(31):
            output = adapter.decode_one_token(token, state, capture_queries=False)
            values.append(output.logits.detach().reshape(-1).float())
            token = output.next_token_id
    assert torch.equal(torch.stack(values), reference)
    record = cache.storage_record()
    assert record["outstanding_layer_leases"] == 0
    assert record["history_updates_per_layer"] == [31, 31]
    assert record["peak_active_kv_storage_bytes"] >= record["final_active_kv_storage_bytes"]
    if sealed:
        assert all(layer.tail_used == 15 and len(layer.sealed) == 1 for layer in cache.layers)


def test_partial_source_format_recovers_all_low_occupancy_direct_values():
    adapter = _llava(kv_heads=2)
    adapter.model.to(torch.bfloat16)
    with ProjectionCapture(adapter) as capture:
        prefill = adapter.prefill(adapter.prepare_inputs(_runtime_request("mixed").messages))
    pairs, _ = adapter._legacy_layers(prefill.state.past_key_values)
    choices, record = formats_for_layer(
        capture.records[0],
        adapter._language_layers()[0].self_attn,
        *pairs[0],
        ((0, 1, 3), (0, 2, 3)),
    )
    assert "token_hybrid" in record["candidate_component_bytes"]
    mixed = next(f for f in choices if f.kind == "token_hybrid")
    assert mixed.slots.tolist() == [1, 4]
    assert mixed.archive.union.tolist() == [0, 3]


def test_sealed_history_128_updates_preserve_all_bits_and_immutable_chunks():
    from mosaickv.evaluation.exact_storage import ExactCache, ExactFormat, encode
    from mosaickv.evaluation.sealed_history import SealedHistoryLayer

    torch.manual_seed(31)
    prompt = tuple(torch.randn(1, 2, 3, 8).bfloat16() for _ in range(2))
    representation = ExactFormat("kv", tuple(prompt[0].shape), tuple(encode(t) for t in prompt))
    attention = torch.nn.Linear(8, 8, dtype=torch.bfloat16)
    cache = ExactCache(audit_workspace=True)
    layer = SealedHistoryLayer(
        representation, attention, observe=cache.observe, direct=True, updates=128, audit=True
    )
    cache.layers.append(layer)
    histories = [[], []]
    snapshots = []
    for _step in range(128):
        pair = tuple(torch.randn(1, 2, 1, 8).bfloat16() for _ in range(2))
        actual = layer.update(*pair)
        for j, component in enumerate(pair):
            histories[j].append(component)
            expected = torch.cat((prompt[j], *histories[j]), dim=-2)
            assert torch.equal(actual[j].view(torch.int16), expected.view(torch.int16))
        for old, saved in zip(layer.sealed, snapshots, strict=False):
            assert all(
                torch.equal(a, b)
                for body, values in zip(old, saved, strict=True)
                for a, b in zip(body.buffers, values, strict=True)
            )
        snapshots = [
            tuple(tuple(t.clone() for t in b.buffers) for b in pair) for pair in layer.sealed
        ]
        layer.release()
    assert layer.history_length == 128
    assert layer.tail_used == 16
    assert len(layer.sealed) == 7
    assert layer.history_capacity == 16
    assert cache.storage_record()["outstanding_layer_leases"] == 0
