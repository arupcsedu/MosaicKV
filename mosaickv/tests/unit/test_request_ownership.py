from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.request_ownership import cache_tensors


def test_inventory_matches_real_cache_shared_plus_layer_owner_contract():
    shared, left, right = (torch.empty(3) for _ in range(3))
    cache = SimpleNamespace(
        shared_tensors=(shared,),
        layers=[SimpleNamespace(owned_tensors=lambda: (left, right, shared))],
    )
    result = cache_tensors(cache)
    assert [id(t) for t in result] == [id(shared), id(left), id(right), id(shared)]
    # The timeline deduplicates aliases by allocation generation; the cache
    # itself deliberately has no owned_tensors() method.
    assert not hasattr(cache, "owned_tensors")


def test_no_implicit_empty_cache_or_unsupported_ownership_fallback():
    with pytest.raises(TypeError):
        cache_tensors(SimpleNamespace(layers=[]))


def test_allocator_observer_replaces_only_weakref_scans_and_restores_default():
    from mosaickv.evaluation.projection_cache import ScratchAudit
    from mosaickv.evaluation.request_ownership import allocator_lifetimes_only

    calls = []
    original = ScratchAudit.__torch_dispatch__
    with allocator_lifetimes_only():
        with ScratchAudit(excluded=set(), observe=calls.append) as mode:
            result = torch.arange(8) + 1
        assert result.tolist() == list(range(1, 9))
        assert mode.operations > 0 and not mode.refs and not calls
    assert ScratchAudit.__torch_dispatch__ is original
