from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation import native_history_reader as native
from mosaickv.evaluation.mixed_verification_barrier import _epoch, flush, transaction, use


def fixture(monkeypatch):
    observations = []
    cache = SimpleNamespace(
        layers=[SimpleNamespace(leased=(), source_pending=False)],
        observe=lambda ts: observations.append(sum(t.numel() * t.element_size() for t in ts)),
    )
    state = SimpleNamespace(closed=False, pool=SimpleNamespace(cache=cache))
    monkeypatch.setattr(native, "current", lambda: state)
    return cache, state, observations


def test_same_predicates_flush_before_return_and_charge_scalar_workspace(monkeypatch):
    cache, _state, observations = fixture(monkeypatch)
    with use() as report, transaction(cache):
        _epoch.get()[1].extend(torch.tensor(True) for _ in range(6))
    assert report["commit_flushes"] == 1 and report["complete_KV_verification_flags"] == 6
    assert max(observations) == 13 and observations[-1] == 0
    assert _epoch.get() is None


def test_wrong_original_bit_fails_without_relaxing_or_skipping(monkeypatch):
    cache, _state, _observations = fixture(monkeypatch)
    with (
        pytest.raises(RuntimeError, match="bitwise verification failed"),
        use(),
        transaction(cache),
    ):
        _epoch.get()[1].extend(torch.tensor(x) for x in (True, True, True, True, True, False))
    assert _epoch.get() is None


@pytest.mark.parametrize("count", [1, 5, 193, 198])
def test_incomplete_or_unbounded_predicate_sets_fail(monkeypatch, count):
    cache, _state, _ = fixture(monkeypatch)
    with pytest.raises(RuntimeError, match="incomplete bounded"):
        flush(cache, [torch.tensor(True) for _ in range(count)])


def test_foreign_or_active_cache_cannot_defer(monkeypatch):
    cache, state, _ = fixture(monkeypatch)
    state.pool.cache = object()
    with pytest.raises(RuntimeError, match="released cache"), use(), transaction(cache):
        pass
    state.pool.cache = cache
    cache.layers[0].leased = (True,)
    with pytest.raises(RuntimeError, match="released cache"), use(), transaction(cache):
        pass


def test_policy_and_failure_scopes_do_not_leak(monkeypatch):
    cache, _state, _ = fixture(monkeypatch)
    with (
        use(),
        transaction(cache),
        pytest.raises(RuntimeError, match="nonnested"),
        transaction(cache),
    ):
        pass
    with pytest.raises(RuntimeError, match="active nonnested"), transaction(cache):
        pass
