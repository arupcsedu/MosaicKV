"""Original-BF16 exact transport controls; tiny random weights, no downloads."""

import dataclasses
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from mosaickv.adapters.huggingface.types import (
    DecodeOutput,
    DecodeState,
    PrefillOutput,
    QueryVectors,
)
from mosaickv.evaluation.exact_selected_control import (
    build_raw_formats,
    mandatory_positions,
    native_same_selected_state,
    raw_format_report,
    run_exact_selected_cpu,
    run_native_same_selected,
    select_uniform_positions,
)
from mosaickv.evaluation.lossless_bf16 import PackedBF16
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.storage_runs import cache_from_formats


@pytest.fixture(scope="module", autouse=True)
def bounded_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


class TinyAdapter:
    """Real HF eager Llama with explicit original positions and mutable state."""

    device = torch.device("cpu")

    def __init__(self):
        torch.manual_seed(782)
        self.model = (
            LlamaForCausalLM(
                LlamaConfig(
                    vocab_size=32,
                    hidden_size=64,
                    intermediate_size=96,
                    num_hidden_layers=2,
                    num_attention_heads=4,
                    num_key_value_heads=4,
                    head_dim=16,
                    max_position_embeddings=2048,
                    attn_implementation="eager",
                )
            )
            .bfloat16()
            .eval()
        )

    def _language_layers(self):
        return self.model.model.layers

    @staticmethod
    def _legacy_layers(cache):
        return tuple((layer.keys, layer.values) for layer in cache.layers), "cache_layers"

    @staticmethod
    def _cache_length(cache):
        return max(layer.get_seq_length() for layer in cache.layers)

    def prefill(self, *, offset=0, length=600):
        ids = ((torch.arange(length) + offset) % 32)[None]
        output = self.model(input_ids=ids, use_cache=True)
        logits = output.logits[:, -1, :]
        return PrefillOutput(
            logits,
            logits.argmax(-1)[:, None],
            DecodeState(output.past_key_values, torch.ones(1, length), length, length, length, ()),
            QueryVectors(()),
        )

    def decode_one_token(self, token, state, *, capture_queries=False):
        assert capture_queries is False
        position = torch.tensor([state.next_decode_position], dtype=torch.long)
        result = self.model(
            input_ids=token,
            past_key_values=state.past_key_values,
            cache_position=position,
            position_ids=position[None],
            attention_mask=None,
            use_cache=True,
        )
        state.past_key_values = result.past_key_values
        state.active_cache_length = self._cache_length(state.past_key_values)
        state.logical_sequence_length += 1
        state.next_decode_position += 1
        state.attention_mask = torch.ones(1, state.active_cache_length, dtype=torch.long)
        logits = result.logits[:, -1, :]
        return DecodeOutput(logits, logits.argmax(-1)[:, None], state, QueryVectors(()))


def formats_for(adapter, prefill, count):
    pairs, _ = adapter._legacy_layers(prefill.state.past_key_values)
    mandatory = mandatory_positions(SimpleNamespace(modality_map=()), pairs[0][0].shape[-2])
    scores = torch.arange(pairs[0][0].shape[-2]).float().repeat(4, 1)
    positions = tuple(select_uniform_positions(scores, count, mandatory) for _ in pairs)
    return build_raw_formats(adapter, pairs, positions, mandatory=mandatory)


def test_mandatory_policy_and_stable_ties_use_only_inputs():
    spans = (
        SimpleNamespace(modality="image", start=0, end=16),
        SimpleNamespace(modality=SimpleNamespace(value="text"), start=16, end=19),
    )
    prepared = SimpleNamespace(modality_map=spans)
    mandatory = mandatory_positions(prepared, 32)
    assert mandatory == (*range(4), 16, 17, 18, *range(24, 32))
    scores = torch.ones(2, 32)
    selected = select_uniform_positions(scores, 20, mandatory)
    assert selected[0] == selected[1] == tuple(sorted((*mandatory, 4, 5, 6, 7, 8)))
    assert mandatory_positions(SimpleNamespace(modality_map=()), 3) == (0, 1, 2)
    with pytest.raises(ValueError, match="within"):
        mandatory_positions(
            SimpleNamespace(modality_map=(SimpleNamespace(modality="text", start=0, end=33),)), 32
        )
    with pytest.raises(ValueError, match="sorted unique"):
        select_uniform_positions(scores, 20, (2, 1))
    with pytest.raises(ValueError, match="feasible"):
        select_uniform_positions(scores, 2, mandatory)


@pytest.mark.parametrize("count", [91, 98])
def test_raw_gather_and_128_prediction_native_transport_parity(count, monkeypatch):
    import mosaickv.evaluation.exact_storage as codecs

    monkeypatch.setattr(codecs, "encode", lambda *_: pytest.fail("encoding forbidden"))
    adapter = TinyAdapter()
    with torch.inference_mode():
        prefill = adapter.prefill()
        pairs, _ = adapter._legacy_layers(prefill.state.past_key_values)
        formats = formats_for(adapter, prefill, count)
        proof = raw_format_report(formats)
        for f, pair in zip(formats, pairs, strict=True):
            assert f.original_positions.device.type == "cpu"
            for head, row in enumerate(f.original_positions.tolist()):
                for packed, source in zip(f.pair, pair, strict=True):
                    assert packed.raw and len(packed.buffers) == 1
                    actual = packed.buffers[0][:, head]
                    expected = source[:, head].index_select(1, torch.tensor(row))
                    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
                    assert (
                        actual.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
                    )
        forced = tuple((i + 3) % 32 for i in range(128))
        native_logits, native = run_native_same_selected(adapter, prefill, formats, forced=forced)
        reader_logits, reader = run_exact_selected_cpu(adapter, prefill, formats, forced=forced)
    assert torch.equal(native_logits, reader_logits)
    assert reader_logits.shape == (128, 32)
    assert proof["raw_KV_backing_bytes"] == 2 * 4 * count * 16 * 2 * 2
    assert proof["CPU_position_map_backing_bytes"] == 2 * 4 * count * 8
    assert proof["FullKV_quality_guard_evaluated"] is False
    assert native["original_next_decode_position"] == 727
    assert native["reference"] == "native_DynamicCache_of_SAME_selected_original_KV_not_FullKV"
    storage = reader["storage"]
    assert storage["initial_active_kv_storage_bytes"] == proof["raw_KV_backing_bytes"]
    assert (
        storage["final_active_kv_storage_bytes"]
        == proof["raw_KV_backing_bytes"] + 2 * 4 * 128 * 16 * 2 * 2
    )
    assert storage["outstanding_layer_leases"] == 0
    assert not reader["whole_request_KV_peak_certified"]


def test_full_retention_and_isolated_native_clones():
    adapter = TinyAdapter()
    with torch.inference_mode():
        prefill = adapter.prefill(length=24)
        formats = formats_for(adapter, prefill, 24)
        first = native_same_selected_state(adapter, prefill, formats)
        second = native_same_selected_state(adapter, prefill, formats)
        native_logits, _ = run_native_same_selected(adapter, prefill, formats, tokens=16)
        reader_logits, _ = run_exact_selected_cpu(adapter, prefill, formats, tokens=16)
    assert torch.equal(native_logits, reader_logits)
    assert first.active_cache_length == 24
    assert first.next_decode_position == second.next_decode_position == 24
    before = second.past_key_values.layers[0].keys.clone()
    with torch.inference_mode():
        first.past_key_values.layers[0].keys.zero_()
    assert torch.equal(second.past_key_values.layers[0].keys, before)
    assert not torch.equal(first.past_key_values.layers[0].keys, formats[0].pair[0].buffers[0])


def test_raw_history_geometry_is_value_and_context_independent():
    adapter = TinyAdapter()
    records = []
    with torch.inference_mode():
        for offset in (0, 7):
            prefill = adapter.prefill(offset=offset)
            formats = formats_for(adapter, prefill, 91)
            cache = cache_from_formats(
                adapter,
                formats,
                updates=128,
                audit=False,
                sealed=False,
                source_history=False,
                shared_workspace=False,
                sequential_lease=False,
            )
            geometry = []
            for step in range(127):
                for layer in cache.layers:
                    tensor = torch.full((1, 4, 1, 16), float(step + offset), dtype=torch.bfloat16)
                    layer.update(tensor, -tensor)
                    assert all(t.shape[-2] == 92 + step for t in layer.leased)
                    layer.release()
                geometry.append(
                    tuple((layer.history_capacity, layer.resident_bytes) for layer in cache.layers)
                )
            records.append((geometry, cache.storage_record()))
    assert records[0] == records[1]
    assert records[0][0][-1][0][0] == 128


def test_full_backing_is_charged_for_views_and_invalid_memberships_fail():
    adapter = TinyAdapter()
    with torch.inference_mode():
        prefill = adapter.prefill(length=24)
        pairs, _ = adapter._legacy_layers(prefill.state.past_key_values)
        formats = formats_for(adapter, prefill, 16)
    f = formats[0]
    larger = torch.empty((1, 4, 32, 16), dtype=torch.bfloat16)
    view = larger[..., :16, :]
    changed = dataclasses.replace(f, pair=(PackedBF16(tuple(view.shape), True, (view,)), f.pair[1]))
    proof = raw_format_report((changed, formats[1]))
    assert proof["raw_KV_backing_bytes"] > proof["raw_KV_view_payload_bytes"]
    assert proof["raw_KV_backing_bytes"] == backing_bytes(
        t for fmt in (changed, formats[1]) for t in fmt.tensors()
    )
    positions = tuple(tuple(tuple(range(16)) for _ in range(4)) for _ in pairs)
    bad = list(positions)
    bad[0] = ((-1, *range(1, 16)), *bad[0][1:])
    with pytest.raises(ValueError, match="bounded"):
        build_raw_formats(adapter, pairs, bad)
    with pytest.raises(ValueError, match="mandatory"):
        build_raw_formats(adapter, pairs, positions, mandatory=(23,))
    mixed = (positions[0], tuple(tuple(range(17)) for _ in range(4)))
    with pytest.raises(ValueError, match="across layers"):
        build_raw_formats(adapter, pairs, mixed)
