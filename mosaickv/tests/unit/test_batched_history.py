from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.batched_history import BatchedHistoryLayer, seal_full_tails, split_batch
from mosaickv.evaluation.exact_storage import ExactCache, ExactFormat, encode
from mosaickv.evaluation.lossless_bf16 import PackedBF16, pack_cpu, unpack_cpu
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes


@pytest.mark.parametrize("raw", [True, False])
def test_shared_windows_preserve_every_bf16_bit_and_charge_one_owner(raw):
    words = torch.arange(65536, dtype=torch.int32).to(torch.int16)
    expected_words = words.clone()
    original = words.view(torch.bfloat16).view(2, 1, 2, 128, 128)
    body = (
        PackedBF16(tuple(original.shape), True, (original.clone(),))
        if raw
        else pack_cpu(original, raw_fallback=False)
    )
    parts = split_batch(body, (1, 2, 128, 128), 2)
    for i, part in enumerate(parts):
        actual = unpack_cpu(part)
        assert torch.equal(actual.view(torch.int16), original[i].view(torch.int16))
    assert backing_bytes(tuple(t for p in parts for t in p.buffers)) == backing_bytes(body.buffers)
    original.view(torch.int16).zero_()
    reconstructed = torch.cat([unpack_cpu(p).view(torch.int16).flatten() for p in parts])
    assert torch.equal(reconstructed, expected_words)


def cache_for_test(count=2):
    cache = ExactCache(audit_workspace=False)
    originals = []
    for _ in range(count):
        pair = tuple(torch.randn(1, 2, 5, 32).bfloat16() for _ in range(2))
        originals.append(pair)
        format = ExactFormat("kv", tuple(pair[0].shape), pair=tuple(encode(t) for t in pair))
        layer = BatchedHistoryLayer(
            format,
            SimpleNamespace(parameters=lambda: ()),
            observe=cache.observe,
            direct=True,
            updates=40,
            audit=False,
        )
        cache.layers.append(layer)
    cache.initial_bytes = backing_bytes(
        tuple(t for layer in cache.layers for t in layer.owned_tensors())
    )
    return cache, originals


@pytest.mark.parametrize("count", [2, 9])
def test_full_cache_transaction_keeps_every_history_token_and_exact_backing(count):
    torch.manual_seed(7)
    cache, original = cache_for_test(count)
    histories = [[[], []] for _ in cache.layers]
    for step in range(40):
        sealed = seal_full_tails(cache)
        assert sealed == (step in (16, 32))
        for i, layer in enumerate(cache.layers):
            pair = tuple(torch.randn(1, 2, 1, 32).bfloat16() for _ in range(2))
            output = layer.update(*pair)
            for kind in range(2):
                histories[i][kind].append(pair[kind])
                expected = torch.cat((original[i][kind], *histories[i][kind]), 2)
                assert torch.equal(output[kind].view(torch.int16), expected.view(torch.int16))
            layer.release()
        record = cache.storage_record()
        assert record["history_updates_per_layer"] == [step + 1] * count
        assert record["final_active_kv_storage_bytes"] == backing_bytes(
            tuple(t for layer in cache.layers for t in layer.owned_tensors())
        )
    assert cache.batched_history_seals == 2
    assert cache.batched_history_encode_batches == 2 * ((count + 3) // 4)
    assert all(len(layer.sealed) == 2 and layer.tail_used == 8 for layer in cache.layers)


def test_coordinator_rejects_partial_forward_and_unreleased_lease():
    cache, _ = cache_for_test()
    pair = tuple(torch.zeros(1, 2, 1, 32, dtype=torch.bfloat16) for _ in range(2))
    cache.layers[0].update(*pair)
    with pytest.raises(ValueError, match="synchronized released"):
        seal_full_tails(cache)


def test_split_rejects_partial_codec_blocks():
    source = torch.ones(1, 2, 3, 32, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="block-aligned"):
        split_batch(pack_cpu(source), (1, 2, 3, 32), 1)
