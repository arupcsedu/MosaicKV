import evicastkv_direct_construction_profile as profile
import pytest

from mosaickv.evaluation import direct_text_construction as construction
from mosaickv.evaluation.construction_phase_timing import PhaseTimer, phase_totals


def test_instrumentation_restores_functions_and_outputs(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(construction, "build", lambda: sentinel)
    original = construction.build
    timer = PhaseTimer()
    with profile.observe(timer):
        assert construction.build() is sentinel
    assert construction.build is original
    assert phase_totals(timer.resolve())["direct.build"]["calls"] == 1


def test_instrumentation_restores_after_failure():
    original = construction.build
    timer = PhaseTimer()
    with pytest.raises(TypeError), profile.observe(timer):
        construction.build()
    assert construction.build is original and not timer.stack


def test_incomplete_profile_cannot_appear_complete():
    with pytest.raises(ValueError, match="complete unique"):
        profile.summarize([])


def test_phase_summary_requires_numerical_controls():
    rows = [
        dict(method=m, trial=t, warmup=t < 0, phases={}, logits_bitwise=True)
        for m in profile.direct.common.previous.METHODS
        for t in (-1, 0, 1)
    ]
    with pytest.raises(ValueError, match="complete constructor"):
        profile.summarize(rows)
