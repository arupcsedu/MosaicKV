import threading

import pytest

from mosaickv.evaluation import prefill_native_overlap as mod


def test_bounded_ordered_private_worker_and_retirement():
    calls, closed = [], []

    def task(i, value):
        calls.append((i, threading.get_ident()))
        return value

    q = mod.OrderedQueue(task, lambda: closed.append(threading.get_ident()))
    objects = [object() for _ in range(8)]
    try:
        for i, item in enumerate(objects):
            q.submit(i, item)
        assert [q.take(i) for i in range(8)] == objects
        assert q.maximum_observed == 2
        q.close(complete=True)
        assert q.closed and not q.pending and not q.results
        assert [i for i, _ in calls] == list(range(8))
        assert len({tid for _, tid in calls}) == 1
        assert closed == [calls[0][1]]
        assert closed[0] != threading.get_ident()
        with pytest.raises(RuntimeError):
            q.submit(8, object())
    finally:
        q.close(complete=False)
    assert len(closed) == 1


def test_exception_always_retires_on_owner_thread():
    closed = []

    def task(i, value):
        raise ArithmeticError("synthetic task failure, no silent fallback")

    q = mod.OrderedQueue(task, lambda: closed.append(threading.get_ident()))
    q.submit(0, object())
    with pytest.raises(ArithmeticError):
        q.take(0)
    q.close(complete=False)
    assert q.closed and len(closed) == 1


def test_wrong_order_duplicate_and_unconsumed_are_rejected():
    q = mod.OrderedQueue(lambda i, x: x, lambda: None)
    try:
        with pytest.raises(RuntimeError):
            q.submit(1, object())
        q.submit(0, object())
        with pytest.raises(RuntimeError):
            q.submit(0, object())
        with pytest.raises(RuntimeError):
            q.take(1)
        with pytest.raises(RuntimeError):
            q.close(complete=True)
        assert q.closed
    finally:
        q.close(complete=False)
    with pytest.raises(ValueError):
        mod.OrderedQueue(lambda i, x: x, lambda: None, maximum_pending=3)


def test_default_and_ineligible_baselines_are_truthful_noops():
    @mod.managed
    def call(adapter, messages, profile, method, **kwargs):
        assert mod.current() is None
        mod.sealed(0, object(), object())
        return "values", "packets", {}

    assert call(None, None, None, "mosaickv_exact")[2]["overlapping_native_construction"] == {
        "requested": False,
        "eligible": False,
        "report": None,
    }
    for method in ("prefixkv_reimpl", "lookm_reimpl", "vl_cache_reimpl", "prompt_attention_topk"):
        report = call(None, None, None, method, overlap_native_fit=True)[2]
        assert report["overlapping_native_construction"] == {
            "requested": True,
            "eligible": False,
            "report": None,
        }
    with pytest.raises(ValueError):
        call(None, None, None, "mosaickv_exact", overlap_native_fit=1)
    with pytest.raises(NotImplementedError):
        call(None, None, None, "mosaickv_exact", overlap_native_fit=True)
    assert mod.current() is None


def test_identity_mismatch_is_not_silently_recomputed(monkeypatch):
    import mosaickv.evaluation.lt_source_format as fmt

    monkeypatch.setattr(fmt, "map_sources", lambda packet, f: f(packet))
    original, changed = object(), object()

    class Queue:
        def take(self, index):
            assert index == 0
            return original, object()

    state = mod.Pipeline.__new__(mod.Pipeline)
    state.audit, state.queue = False, Queue()
    with pytest.raises(RuntimeError, match="was replaced"):
        state.convert((changed,), (object(),), policy="direct", audit=False)
