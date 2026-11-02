import pytest

from mosaickv.evaluation.construction_phase_timing import PhaseTimer, phase_totals


def test_nested_timer_preserves_result_counts_and_exclusive_cpu_time():
    clock = iter([0, 1, 3, 5])
    timer = PhaseTimer(clock=lambda: next(clock))

    def leaf(value):
        return value + 1

    wrapped = timer.wrap("leaf", leaf)
    with timer.phase("parent"):
        assert wrapped(4) == 5
        with pytest.raises(RuntimeError):
            timer.resolve()
    totals = phase_totals(timer.resolve())
    assert totals["parent"]["inclusive_wall_seconds"] == 5
    assert totals["parent"]["exclusive_wall_seconds"] == 3
    assert totals["leaf"]["inclusive_wall_seconds"] == 2
    assert totals["leaf"]["calls"] == 1


def test_exception_closes_phase_and_preserves_error():
    timer = PhaseTimer()
    with pytest.raises(ValueError, match="original"), timer.phase("broken"):
        raise ValueError("original")
    assert not timer.stack and timer.resolve()[0]["phase"] == "broken"


def test_events_are_resolved_only_after_scope_not_during_recording():
    log = []

    class Event:
        def record(self):
            log.append("record")

        def elapsed_time(self, end):
            log.append("elapsed")
            return 2.5

    timer = PhaseTimer(event_factory=Event)
    with timer.phase("gpu"):
        pass
    assert log == ["record", "record"]
    assert timer.resolve()[0]["stream_elapsed_ms"] == 2.5


def test_instrumentation_resolves_all_targets_and_restores_aliases_on_error():
    import torch

    from mosaickv.evaluation import shared_source_format
    from mosaickv.evaluation.construction_phase_timing import instrument

    original = shared_source_format.encode
    tensor = torch.ones((1, 1, 16, 128), dtype=torch.bfloat16)
    timer = PhaseTimer()
    with pytest.raises(ValueError, match="control"), instrument(timer):
        assert shared_source_format.encode is not original
        packed = shared_source_format.encode(tensor)
        assert packed.shape == tuple(tensor.shape)
        raise ValueError("control")
    assert shared_source_format.encode is original
    assert any(r["phase"].endswith("shared_source_format.encode") for r in timer.resolve())
