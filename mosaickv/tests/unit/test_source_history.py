from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.exact_storage import ExactFormat, encode
from mosaickv.evaluation.source_history import (
    SourceHistoryCache,
    SourceHistoryLayer,
    SourceInputTransport,
    seal_source_full_tails,
)
from mosaickv.evaluation.storage_runs import cache_from_formats


def cache_fixture():
    cache = SourceHistoryCache(audit_workspace=False)
    attention = torch.nn.Module()
    for _ in range(2):
        pair = tuple(encode(torch.zeros(1, 2, 1, 128, dtype=torch.bfloat16)) for _ in range(2))
        format = ExactFormat("kv", (1, 2, 1, 128), pair=pair)
        cache.layers.append(
            SourceHistoryLayer(
                format, attention, observe=cache.observe, direct=True, updates=128, audit=False
            )
        )
    return cache


def step(cache, token):
    hidden = torch.full((1, 1, 256), token, dtype=torch.bfloat16)
    phase = (
        torch.ones(1, 1, 128, dtype=torch.bfloat16),
        torch.zeros(1, 1, 128, dtype=torch.bfloat16),
    )
    positions = torch.tensor([[600 + token]])
    outputs = []
    for index, layer in enumerate(cache.layers):
        cache.source_input(index, hidden, phase, positions)
        pair = hidden.view(1, 1, 2, 128).transpose(1, 2)
        values = layer.update(pair, pair)
        outputs.append(tuple(t.clone() for t in values))
        layer.release()
    return outputs


def exact_cpu_factory(layer, cos, sin):
    return ExactFormat(
        "kv", tuple(layer.history[0].shape), pair=tuple(encode(t) for t in layer.history)
    )


def test_history_transaction_preserves_every_position_and_all_charged_backing(monkeypatch):
    import mosaickv.evaluation.source_history as module

    monkeypatch.setattr(module, "build_chunk", exact_cpu_factory)
    cache = cache_fixture()
    original_prompt = [layer.representation for layer in cache.layers]
    for token in range(17):
        seal_source_full_tails(cache)
        outputs = step(cache, token)
        for key, value in outputs:
            assert key.shape[-2] == token + 2
            assert torch.equal(key, value)
            assert (key[..., 0, :] == 0).all()
            for position in range(token + 1):
                assert (key[..., position + 1, :] == position).all()
    record = cache.storage_record()
    assert record["source_history_seals"] == 1
    assert record["source_history_original_positions"] == [list(range(600, 616))]
    assert record["history_updates_per_layer"] == [17, 17]
    assert record["source_history_normalized_tail_bytes"] == 2 * 16 * 256 * 2
    assert record["outstanding_layer_leases"] == 0
    assert [layer.representation for layer in cache.layers] == original_prompt


def test_failed_layer_construction_rolls_back_the_entire_seal(monkeypatch):
    import mosaickv.evaluation.source_history as module

    cache = cache_fixture()
    for token in range(16):
        step(cache, token)
    before = cache.storage_record()
    calls = []

    def fail_second(layer, cos, sin):
        calls.append(layer)
        if len(calls) == 2:
            raise RuntimeError("synthetic layer reconstruction failure")
        return exact_cpu_factory(layer, cos, sin)

    monkeypatch.setattr(module, "build_chunk", fail_second)
    with pytest.raises(RuntimeError, match="synthetic layer reconstruction failure"):
        seal_source_full_tails(cache)
    assert all(layer.tail_used == 16 and not layer.source_sealed for layer in cache.layers)
    assert cache.source_history_seals == 0 and cache.history_position_chunks == []
    assert (
        cache.storage_record()["final_active_kv_storage_bytes"]
        == before["final_active_kv_storage_bytes"]
    )


def test_unpaired_source_and_cross_layer_phase_mismatch_fail_closed():
    cache = cache_fixture()
    pair = torch.zeros(1, 2, 1, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="captured source"):
        cache.layers[0].update(pair, pair)
    hidden = torch.zeros(1, 1, 256, dtype=torch.bfloat16)
    phase = (torch.ones(1, 1, 128, dtype=torch.bfloat16),) * 2
    positions = torch.tensor([[600]])
    cache.source_input(0, hidden, phase, positions)
    with pytest.raises(ValueError, match="identical original"):
        cache.source_input(1, hidden, tuple(t.clone() for t in phase), positions)


@pytest.mark.parametrize(
    "updates,batched,sealed", [(32, False, True), (128, True, True), (128, False, False)]
)
def test_short_output_and_incompatible_modes_rejected(updates, batched, sealed):
    with pytest.raises(ValueError, match="source history requires"):
        cache_from_formats(
            None,
            (),
            updates=updates,
            audit=False,
            source_history=True,
            batched_history=batched,
            sealed=sealed,
        )


def test_wrong_request_cache_fails_closed_and_all_hooks_are_removed():
    class LlamaAttention(torch.nn.Module):
        def forward(self, **kwargs):
            return None

    LlamaAttention.__module__ = "transformers.models.llama.modeling_llama"
    attention = LlamaAttention()
    adapter = SimpleNamespace(_language_layers=lambda: [SimpleNamespace(self_attn=attention)])
    cache = cache_fixture()
    with (
        pytest.raises(ValueError, match="request-local cache"),
        SourceInputTransport(adapter, cache),
    ):
        attention(
            hidden_states=torch.zeros(1, 1, 256, dtype=torch.bfloat16),
            past_key_value=object(),
            position_ids=torch.tensor([[600]]),
            position_embeddings=(torch.ones(1, 1, 128, dtype=torch.bfloat16),) * 2,
        )
    assert not attention._forward_pre_hooks


@pytest.mark.parametrize("corrupt", [False, True])
def test_batched_checks_commit_only_after_every_layer_passes(monkeypatch, corrupt):
    import mosaickv.evaluation.source_history as module

    cache = cache_fixture()
    cache.source_constructor = "direct_batched_checks"
    for token in range(16):
        step(cache, token)
    calls = []

    def builder(layer, cos, sin, *, bit_flags):
        calls.append(layer)
        bit_flags.extend(torch.tensor(not (corrupt and len(calls) == 2)) for _ in range(6))
        return exact_cpu_factory(layer, cos, sin)

    monkeypatch.setattr(module, "build_chunk_direct", builder)
    if corrupt:
        with pytest.raises(RuntimeError, match="no history sealed"):
            seal_source_full_tails(cache)
        assert all(layer.tail_used == 16 and not layer.source_sealed for layer in cache.layers)
        assert cache.source_history_seals == cache.source_check_syncs == 0
    else:
        assert seal_source_full_tails(cache)
        assert cache.source_check_syncs == 1 and cache.source_check_flags_checked == 12
        assert all(layer.tail_used == 0 and len(layer.source_sealed) == 1 for layer in cache.layers)
        cache.storage_record()
