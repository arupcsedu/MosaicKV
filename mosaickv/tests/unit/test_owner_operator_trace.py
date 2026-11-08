import importlib.util
from pathlib import Path

import pytest


def load():
    path = Path(__file__).resolve().parents[2] / "scripts/complete_owner_operator.py"
    spec = importlib.util.spec_from_file_location("owner_operator_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_trace_uses_actual_kernel_durations_not_host_ranges_or_stream_gaps():
    trace = dict(
        traceEvents=[
            dict(ph="X", cat="cpu_op", name="host launch", dur=10000),
            dict(ph="X", cat="kernel", name="unpack", dur=9),
            dict(ph="X", cat="gpu_memcpy", name="Memcpy DtoD", dur=3),
        ]
    )
    rows = load().kernel_summary(trace)
    assert len(rows) == 2
    assert rows[0]["device_ms"] == 0.009
    assert rows[1]["device_ms"] == 0.003


@pytest.mark.parametrize("duration", (-1, float("nan"), True))
def test_invalid_device_durations_fail_closed(duration):
    with pytest.raises(ValueError):
        load().kernel_summary(
            dict(traceEvents=[dict(ph="X", cat="kernel", name="unpack", dur=duration)])
        )


def test_missing_cuda_events_cannot_be_called_zero_kernel_cost():
    with pytest.raises(ValueError, match="no actual"):
        load().kernel_summary(dict(traceEvents=[dict(ph="X", cat="cpu_op", name="replay", dur=1)]))
