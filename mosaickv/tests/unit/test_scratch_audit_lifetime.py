"""The observer must not keep a source/cache alive until cyclic GC runs."""

import gc
import weakref

import pytest
import torch

from mosaickv.evaluation.projection_cache import ScratchAudit


@pytest.mark.parametrize("nested_outputs", [False, True])
def test_observer_and_payload_retire_by_reference_count_without_gc(nested_outputs):
    class Observer:
        def __init__(self, payload):
            self.payload = payload

        def __call__(self, extra):
            assert extra

    def request():
        payload = torch.empty(128, dtype=torch.uint8)
        observer = Observer(payload)
        audit = ScratchAudit(excluded=set(), observe=observer)
        with audit:
            result = torch.empty(8)
            if nested_outputs:
                result = torch.split(result, 2)
        return weakref.ref(payload), weakref.ref(observer), weakref.ref(audit)

    gc.collect()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        references = request()
        assert all(r() is None for r in references), "audit introduced cyclic owner lifetime"
    finally:
        if was_enabled:
            gc.enable()
        gc.collect()


def test_result_collection_remains_weak_and_counts_all_tuple_outputs():
    observed = []
    audit = ScratchAudit(excluded=set(), observe=lambda ts: observed.append(len(ts)))
    with audit:
        value = torch.empty(8)
        parts = torch.split(value, 2)
    assert len(parts) == 4 and observed[-1] >= 4
    refs = tuple(weakref.ref(t) for t in parts)
    del parts
    assert all(r() is None for r in refs)
