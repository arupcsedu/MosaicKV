from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.exact_storage import ExactFormat
from mosaickv.evaluation.lossless_bf16 import PackedBF16
from mosaickv.evaluation.prompt_graph_reader import PromptGraphLayer, PromptGraphPool
from mosaickv.evaluation.storage_runs import cache_from_formats


def arguments():
    return dict(
        updates=8,
        audit=False,
        sealed=True,
        sequential_lease=True,
        source_history=True,
        sequential_source_history=True,
        consolidated_history=True,
        source_history_constructor="direct_batched_checks",
    )


def factory(**change):
    pair = tuple(torch.zeros(1, 2, 5, 128, dtype=torch.bfloat16) for _ in range(2))
    packet = ExactFormat(
        "kv", tuple(pair[0].shape), pair=tuple(PackedBF16(tuple(x.shape), True, (x,)) for x in pair)
    )
    adapter = SimpleNamespace(
        _language_layers=lambda: [SimpleNamespace(self_attn=torch.nn.Linear(256, 256, bias=False))]
    )
    return cache_from_formats(adapter, (packet,), **dict(arguments(), **change))


def test_default_has_no_graph_pool_or_new_cache_behavior():
    assert not isinstance(factory().layers[0], PromptGraphLayer)


def test_explicit_factory_owns_private_pool_without_preallocating():
    a, b = factory(prompt_graph_reader=True), factory(prompt_graph_reader=True)
    assert isinstance(a.layers[0], PromptGraphLayer)
    assert a.prompt_graph_pool is not b.prompt_graph_pool
    assert a.layers[0].prompt_graph_pool is a.prompt_graph_pool
    assert a.prompt_graph_pool.buffer is None
    assert a.storage_record()["initial_active_kv_storage_bytes"] == factory().initial_bytes
    a.prompt_graph_pool.close()
    a.prompt_graph_pool.close()
    assert a.prompt_graph_pool.record()["retired"]


@pytest.mark.parametrize(
    "change",
    [
        {"prompt_graph_reader": 1},
        {"prompt_graph_reader": True, "sequential_source_history": False},
        {"prompt_graph_reader": True, "updates": 1},
    ],
)
def test_unsupported_configs_fail_without_model_loading(change):
    with pytest.raises(ValueError):
        factory(**change)


def test_cpu_replay_fails_explicitly_and_cannot_hide_an_active_lease():
    cache = factory(prompt_graph_reader=True)
    layer = cache.layers[0]
    with pytest.raises(NotImplementedError, match="CUDA BF16"):
        cache.prompt_graph_pool.acquire(layer, (1, 2, 6, 128), layer.history[0])
    layer.leased = (torch.empty(1),)
    with pytest.raises(RuntimeError, match="active lease"):
        cache.prompt_graph_pool.close()
    layer.release()
    cache.prompt_graph_pool.close()


def test_shared_alias_is_not_double_charged_and_full_owner_is_counted():
    cache = factory(prompt_graph_reader=True)
    layer = cache.layers[0]
    owner = torch.empty(1, 2, 12, 128, dtype=torch.bfloat16)
    cache.prompt_graph_pool.buffer = owner
    cache.shared_tensors = (*cache.shared_tensors, owner)
    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes

    cache.shared_bytes = backing_bytes(cache.shared_tensors)
    layer.leased = (owner[..., :6, :],) * 2
    layer.lease_bytes = backing_bytes(layer.leased)
    assert layer.lease_bytes == 0
    assert cache.storage_record()["final_active_kv_storage_bytes"] == (
        cache.initial_bytes + owner.untyped_storage().nbytes()
    )
    layer.release()
    cache.prompt_graph_pool.close()
    assert cache.storage_record()["final_active_kv_storage_bytes"] == cache.initial_bytes


def test_closed_pool_rejects_reuse_and_never_carries_state_to_another_request():
    cache = factory(prompt_graph_reader=True)
    pool = cache.prompt_graph_pool
    pool.close()
    with pytest.raises(RuntimeError, match="private nonoverlapping"):
        pool.acquire(cache.layers[0], (1, 2, 6, 128), cache.layers[0].history[0])
    assert not PromptGraphPool(cache).closed
