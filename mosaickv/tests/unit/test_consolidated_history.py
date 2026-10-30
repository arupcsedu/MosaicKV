from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.consolidated_history import consolidate, join_patches, merge_sources
from mosaickv.evaluation.exact_storage import ExactFormat, SparseBitPatch, decoded, encode
from mosaickv.evaluation.lossless_bf16 import PackedBF16, pack_cpu, unpack_cpu
from mosaickv.evaluation.source_history import seal_source_full_tails
from mosaickv.evaluation.storage_runs import cache_from_formats


@pytest.mark.parametrize("raw", [(False, False), (True, True), (True, False)])
def test_encoded_concat_preserves_every_original_bit_and_independent_owners(raw):
    generator = torch.Generator().manual_seed(374)
    tensors = [
        torch.randint(-32768, 32767, (1, 1, n, 128), generator=generator, dtype=torch.int16).view(
            torch.bfloat16
        )
        for n in (16, 32)
    ]
    bodies = tuple(
        PackedBF16(tuple(t.shape), True, (t.clone(),)) if flag else pack_cpu(t, raw_fallback=False)
        for t, flag in zip(tensors, raw, strict=True)
    )
    merged = merge_sources(bodies)
    expected = torch.cat(tuple(tensors), 2)
    assert torch.equal(unpack_cpu(merged).view(torch.int16), expected.view(torch.int16))
    old = {t.untyped_storage().data_ptr() for b in bodies for t in b.buffers if t.numel()}
    assert not any(t.untyped_storage().data_ptr() in old for t in merged.buffers if t.numel())


@pytest.mark.parametrize("lengths", [(16, 16), (32, 16), (48, 16)])
def test_patch_reindexing_is_unique_head_local_and_position_preserving(lengths):
    heads, dim = 3, 128
    generator = torch.Generator().manual_seed(4)
    patches = []
    expected = torch.zeros(1, heads, sum(lengths), dim, dtype=torch.bfloat16)
    offset = 0
    for n in lengths:
        indices = torch.randperm(heads * n * dim, generator=generator)[:128].sort().values.int()
        values = torch.randint(-32768, 32767, (128,), generator=generator, dtype=torch.int16).view(
            torch.bfloat16
        )
        patch = SparseBitPatch(indices, values)
        original = torch.zeros(1, heads, n, dim, dtype=torch.bfloat16)
        patch.apply(original)
        expected[..., offset : offset + n, :].copy_(original)
        patches.append(patch)
        offset += n
    merged = join_patches(tuple(patches), lengths, heads=heads, dimension=dim)
    actual = torch.zeros_like(expected)
    merged.apply(actual)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    assert merged.indices.unique().numel() == merged.indices.numel()


def fixture():
    adapter = SimpleNamespace(
        _language_layers=lambda: [SimpleNamespace(self_attn=torch.nn.Module()) for _ in range(2)]
    )
    formats = tuple(
        ExactFormat(
            "kv",
            (1, 2, 3, 128),
            pair=tuple(encode(torch.ones(1, 2, 3, 128, dtype=torch.bfloat16)) for _ in range(2)),
        )
        for _ in range(2)
    )
    return cache_from_formats(
        adapter,
        formats,
        updates=128,
        audit=True,
        source_history=True,
        source_history_constructor="direct",
    )


def step(cache, token):
    phase = (
        torch.ones(1, 1, 128, dtype=torch.bfloat16),
        torch.zeros(1, 1, 128, dtype=torch.bfloat16),
    )
    position = torch.tensor([[600 + token]])
    outputs = []
    for i, layer in enumerate(cache.layers):
        hidden = torch.full((1, 1, 256), token + i, dtype=torch.bfloat16)
        cache.source_input(i, hidden, phase, position)
        pair = hidden.view(1, 1, 2, 128).transpose(1, 2)
        values = layer.update(pair, pair)
        outputs.append(tuple(t.clone() for t in values))
        layer.release()
    return outputs


def fake_seal(layer, cos, sin):
    return ExactFormat(
        "kv", tuple(layer.history[0].shape), pair=tuple(encode(t) for t in layer.history)
    )


def fake_join(layer, cos, sin):
    values = [tuple(decoded(p) for p in f.pair) for f in layer.source_sealed]
    pairs = tuple(torch.cat(tuple(v[i] for v in values), 2) for i in range(2))
    return ExactFormat("kv", tuple(pairs[0].shape), pair=tuple(encode(t) for t in pairs))


def test_streaming_transactions_match_every_old_value_and_original_position(monkeypatch):
    import mosaickv.evaluation.consolidated_history as module
    import mosaickv.evaluation.source_history as history

    monkeypatch.setattr(history, "build_chunk_direct", fake_seal)
    monkeypatch.setattr(module, "build_consolidated", fake_join)
    a, b = fixture(), fixture()
    for token in range(50):
        seal_source_full_tails(a)
        if seal_source_full_tails(b):
            consolidate(b)
        reference, actual = step(a, token), step(b, token)
        assert all(
            torch.equal(x, y)
            for xs, ys in zip(reference, actual, strict=True)
            for x, y in zip(xs, ys, strict=True)
        )
        b.storage_record()  # independent physical-owner enumeration must agree
    assert b.source_consolidation_epochs == 2
    assert b.source_consolidation_layers == 4
    assert len(b.history_rotary_chunks) == 1
    assert all(
        len(layer.source_sealed) == 1 and layer.source_sealed[0].shape[2] == 48
        for layer in b.layers
    )
    assert (
        b.storage_record()["source_history_original_positions"]
        == a.storage_record()["source_history_original_positions"]
    )


def test_late_failure_keeps_every_committed_and_original_layer_semantically_valid(monkeypatch):
    import mosaickv.evaluation.consolidated_history as module
    import mosaickv.evaluation.source_history as history

    monkeypatch.setattr(history, "build_chunk_direct", fake_seal)
    cache = fixture()
    for token in range(32):
        seal_source_full_tails(cache)
        step(cache, token)
    seal_source_full_tails(cache)

    def fail_second(layer, cos, sin):
        if layer is cache.layers[1]:
            raise RuntimeError("injected later layer failure")
        return fake_join(layer, cos, sin)

    monkeypatch.setattr(module, "build_consolidated", fail_second)
    with pytest.raises(RuntimeError, match="later layer"):
        consolidate(cache)
    assert [len(layer.source_sealed) for layer in cache.layers] == [1, 2]
    values = step(cache, 32)
    for i, pair in enumerate(values):
        for tensor in pair:
            expected = torch.arange(33).bfloat16()[None, None, :, None] + i
            assert torch.equal(tensor[..., 3:, :], expected.expand_as(tensor[..., 3:, :]))
    cache.storage_record()


def test_incompatible_policy_fails_closed():
    with pytest.raises(ValueError, match="consolidation requires"):
        cache_from_formats(None, (), updates=128, audit=False, consolidated_history=True)
