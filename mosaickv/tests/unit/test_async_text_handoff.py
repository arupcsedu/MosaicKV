import weakref
from types import SimpleNamespace

import pytest
import torch
from test_verified_constructor import corrected_fixture

from mosaickv.evaluation.async_text_handoff import Capture, publish, verify_outer
from mosaickv.evaluation.direct_text_construction import build
from mosaickv.evaluation.lt_source_format import fit
from mosaickv.evaluation.prefill_native_overlap import Pipeline
from mosaickv.evaluation.sequential_kv_lease import fill_component
from mosaickv.evaluation.streaming_prefill import RetiringPrefillCache
from mosaickv.evaluation.verified_constructor import State


class Engine:
    def project(self, packet, attention, component, target):
        fill_component(packet, attention, component, target)


@pytest.mark.parametrize(
    "modes",
    [
        ("source", "source"),
        ("resident", "resident"),
        ("encoded", "encoded"),
        ("resident", "encoded"),
    ],
)
def test_existing_native_reference_builds_same_text_and_outer_words(modes):
    attention, base = corrected_fixture()
    capture, engine = Capture(base, (0, 2), modes), Engine()
    refs = []

    def observed(component, reference):
        refs.append(weakref.ref(reference))
        capture.capture(component, reference)

    candidate = fit(base, attention, engine, direct_validation=True, on_reference=observed)
    assert capture.calls == [0, 1]
    assert all(ref() is None for ref in refs)
    prepared, native = capture.finish(candidate)
    if modes != ("source", "source"):
        pair = base.materialize(attention)
        positions = (tuple(range(base.shape[2])),) * base.shape[1]
        original = build(base, pair, positions, (0, 2), modes, attention=attention)
        assert all(
            torch.equal(a.view(torch.int16), b.view(torch.int16))
            for a, b in zip(
                prepared.materialize(attention), original.materialize(attention), strict=True
            )
        )
    else:
        assert prepared is base
    assert verify_outer(prepared, native, attention, engine) == 6


def test_plan_shape_order_and_mode_errors_fail_closed():
    _, base = corrected_fixture()
    for text, modes in (
        ((), ("source", "source")),
        ((2, 0), ("resident", "resident")),
        ((0,), ("source", "encoded")),
    ):
        with pytest.raises(ValueError):
            Capture(base, text, modes)
    capture = Capture(base, (0,), ("resident", "resident"))
    with pytest.raises(RuntimeError, match="K then V"):
        capture.capture(1, torch.ones(base.shape).bfloat16())
    state = State("async_handoff", [("resident", "resident")], audit=False)
    with pytest.raises(RuntimeError, match="private async"):
        state.deferred_plan(0)


def test_publication_is_ordered_keeps_logical_state_and_counts_old_new_overlap():
    cache = RetiringPrefillCache(1, audit=False)
    old, new = torch.ones(7), torch.ones(11)
    original = SimpleNamespace(shape=(1, 2, 9, 128), tensors=lambda: (old,))
    prepared = SimpleNamespace(shape=original.shape, tensors=lambda: (new,))
    cache.layers[0].packet, cache.layers[0].length, cache.next_layer = original, 9, 1
    cache.layer_owners_changed(0)
    publish(cache, 0, original, prepared)
    assert cache.get_seq_length() == 9 and cache.next_layer == 1
    assert cache.record()["final_prefill_cache_bytes"] == new.untyped_storage().nbytes()
    assert cache.peak_bytes == old.untyped_storage().nbytes() + new.untyped_storage().nbytes()
    with pytest.raises(RuntimeError, match="same-cache"):
        publish(cache, 0, original, prepared)


def test_pipeline_rejects_cross_request_publication_and_retires_on_abort():
    pipe = Pipeline(audit=False, frozen=False, deferred_text=True)
    pipe.cache = object()
    with pytest.raises(RuntimeError, match="cross request"):
        pipe.publish_ready(object(), final=False)
    pipe.queue.close(complete=False)
    assert pipe.cache is None and pipe.queue.closed


def test_default_native_fitter_has_no_reference_callback_side_effects():
    attention, base = corrected_fixture()
    engine = Engine()
    original = fit(base, attention, engine, direct_validation=True)
    capture = Capture(base, (0,), ("source", "source"))
    other = fit(base, attention, engine, direct_validation=True, on_reference=capture.capture)
    assert all(
        torch.equal(a.indices, b.indices)
        and torch.equal(a.values.view(torch.int16), b.values.view(torch.int16))
        for a, b in zip(original.repairs, other.repairs, strict=True)
    )
