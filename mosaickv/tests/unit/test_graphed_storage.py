from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.exact_storage import ExactFormat, encode
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.source_history import seal_source_full_tails
from mosaickv.evaluation.storage_runs import cache_from_formats


def fixture(*, graphs=False, shared=True):
    modules = [SimpleNamespace(self_attn=torch.nn.Module()) for _ in range(2)]
    adapter = SimpleNamespace(_language_layers=lambda: modules)
    formats = tuple(
        ExactFormat(
            "kv",
            (1, 2, n, 128),
            pair=tuple(
                encode(torch.full((1, 2, n, 128), i + offset, dtype=torch.bfloat16))
                for offset in (0, 10)
            ),
        )
        for i, n in enumerate([3, 5])
    )
    return cache_from_formats(
        adapter,
        formats,
        updates=128,
        audit=True,
        source_history=True,
        source_history_constructor="direct",
        shared_workspace=shared,
        workspace_graph=graphs,
    )


def step(cache, token):
    phase = (
        torch.ones(1, 1, 128, dtype=torch.bfloat16),
        torch.zeros(1, 1, 128, dtype=torch.bfloat16),
    )
    positions = torch.tensor([[600 + token]])
    values = []
    for i, layer in enumerate(cache.layers):
        hidden = torch.full((1, 1, 256), token + i, dtype=torch.bfloat16)
        cache.source_input(i, hidden, phase, positions)
        key = hidden.view(1, 1, 2, 128).transpose(1, 2)
        pair = layer.update(key, key + 10)
        values.append(tuple(t.clone() for t in pair))
        layer.release()
    return values


def exact_factory(layer, cos, sin):
    return ExactFormat(
        "kv", tuple(layer.history[0].shape), pair=tuple(encode(t) for t in layer.history)
    )


def test_shared_strided_workspace_matches_legacy_and_every_history_bit(monkeypatch):
    import mosaickv.evaluation.source_history as module

    monkeypatch.setattr(module, "build_chunk_direct", exact_factory)
    reference, candidate = fixture(shared=False), fixture()
    for token in range(33):
        if token and token % 16 == 0:
            candidate.release_workspace()
        seal_source_full_tails(reference)
        seal_source_full_tails(candidate)
        a, b = step(reference, token), step(candidate, token)
        assert all(
            torch.equal(x.view(torch.int16), y.view(torch.int16))
            for pairs in zip(a, b, strict=True)
            for x, y in zip(*pairs, strict=True)
        )
        record = candidate.storage_record()
        assert record["outstanding_layer_leases"] == 0
        assert record["active_workspace_bytes"] == backing_bytes(candidate.workspace)
        assert len(candidate.workspace) == 2
        assert candidate.layers[0].history_length == candidate.layers[1].history_length == token + 1
    candidate.release_workspace()
    assert candidate.storage_record()["active_workspace_bytes"] == 0
    assert (
        candidate.storage_record()["final_active_kv_storage_bytes"]
        == reference.storage_record()["final_active_kv_storage_bytes"]
    )
    assert candidate.workspace_allocations == 3


def test_shared_workspace_cannot_overlap_or_release_live_lease():
    cache = fixture()
    first = cache.layers[0]
    cache.workspace = (torch.empty(1, 2, 133, 128),) * 2
    first.leased = cache.workspace
    with pytest.raises(RuntimeError, match="overlapping"):
        cache.acquire(cache.layers[1])
    with pytest.raises(RuntimeError, match="active"):
        cache.release_workspace()


def test_close_invalidates_every_graph_before_releasing_owners(monkeypatch):
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 0)
    cache = fixture()
    step(cache, 0)
    for layer in cache.layers:
        layer.reconstruction_graph = SimpleNamespace(reset=lambda: None)
    before = cache.storage_record()
    workspace = backing_bytes(cache.workspace)
    cache.release_workspace()
    assert all(layer.reconstruction_graph is None for layer in cache.layers)
    assert (
        before["final_active_kv_storage_bytes"]
        - cache.storage_record()["final_active_kv_storage_bytes"]
        == workspace
    )


def test_private_graph_allocations_are_charged_and_teardown_verified(monkeypatch):
    cache = fixture()
    step(cache, 0)
    live = [1024]
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: live[0])

    def reset():
        live[0] -= 512

    for layer in cache.layers:
        layer.reconstruction_graph = SimpleNamespace(reset=reset)
        layer.graph_private_bytes = 512
    cache.graph_private_live_bytes = 1024
    before = cache.storage_record()
    workspace = backing_bytes(cache.workspace)
    cache.release_workspace()
    after = cache.storage_record()
    assert live == [0] and after["graph_private_live_bytes"] == 0
    assert after["graph_private_accounting_verified"]
    assert (
        before["final_active_kv_storage_bytes"] - after["final_active_kv_storage_bytes"]
        == workspace + 1024
    )


def test_incomplete_private_graph_release_fails_closed(monkeypatch):
    cache = fixture()
    step(cache, 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 512)
    cache.layers[0].reconstruction_graph = SimpleNamespace(reset=lambda: None)
    cache.layers[0].graph_private_bytes = 512
    cache.graph_private_live_bytes = 512
    with pytest.raises(RuntimeError, match="released 0 bytes; expected 512"):
        cache.release_workspace()
    assert not cache.graph_private_accounting_verified


def test_graph_replay_fails_explicitly_on_cpu():
    cache = fixture(graphs=True)
    with pytest.raises(NotImplementedError, match="CUDA"):
        step(cache, 0)
    for layer in cache.layers:
        layer.release()
    cache.release_workspace()


@pytest.mark.parametrize("kwargs", [dict(workspace_graph=True), dict(shared_workspace=True)])
def test_invalid_combinations_fail_before_any_allocation(kwargs):
    with pytest.raises(ValueError, match="shared source-history"):
        cache_from_formats(None, (), updates=128, audit=False, **kwargs)
