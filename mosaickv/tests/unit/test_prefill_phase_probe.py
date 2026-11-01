import weakref

import pytest
import torch

from mosaickv.evaluation.prefill_phase_probe import PhaseCache, wrap_phase


def test_phase_probe_preserves_observer_and_never_pins_buffers(monkeypatch):
    instances = []
    monkeypatch.setattr(PhaseCache, "instances", instances)
    cache = PhaseCache(1, audit=True)
    tensor = torch.empty(1024)
    reference = weakref.ref(tensor)
    observe = wrap_phase("outer", wrap_phase("packing", cache.observe, instances), instances)
    observe((tensor,))
    assert cache.peak_events[-1] == dict(
        layer=0, device="cpu", bytes=4096, phase=("outer", "packing")
    )
    assert cache.peak_bytes == 4096 and cache.observations == 1
    del tensor
    assert reference() is None
    assert cache.phase_stack == []
    with pytest.raises(RuntimeError):
        PhaseCache(1)


def test_phase_stack_is_restored_on_failure(monkeypatch):
    instances = []
    monkeypatch.setattr(PhaseCache, "instances", instances)
    cache = PhaseCache(1)

    def fail():
        raise ValueError("synthetic fixture")

    with pytest.raises(ValueError):
        wrap_phase("failure", fail, instances)()
    assert cache.phase_stack == []


@pytest.mark.parametrize(
    "outcome",
    [
        dict(complete=False, failures=[]),
        dict(complete=True, failures=["bad"]),
        dict(complete=True, failures=[], source_net_storage_gate=False),
    ],
)
def test_profile_never_submits_after_failed_parent(outcome, monkeypatch, tmp_path):
    import evicastkv_source_peak_profile as module

    records = []
    monkeypatch.setattr(module.previous.previous.common, "inputs", lambda *a: {})
    monkeypatch.setattr(module.previous.previous, "read", lambda *a: outcome)
    monkeypatch.setattr(module, "sha", lambda *a: "negative-outcome-sha")
    monkeypatch.setattr(module, "write_once", lambda path, value: records.append(value))
    monkeypatch.setattr(
        module.previous.common.control.parent, "submit", lambda *a: pytest.fail("unsafe submit")
    )
    module.submit(tmp_path, "plan")
    assert records[0]["GPU_submitted"] is False
