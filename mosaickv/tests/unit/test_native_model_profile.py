from types import SimpleNamespace

import evicastkv_native_model_profile as driver
import pytest


def test_windows_require_actual_cuda_activity_and_exact_registered_steps():
    rows = [
        dict(step=i, operators=[dict(device_type="DeviceType.CUDA", device_us=2)])
        for i in driver.STEPS
    ]
    driver.validate_windows(rows)
    with pytest.raises(ValueError, match="unique"):
        driver.validate_windows(rows[:-1])
    rows[0]["operators"][0]["device_type"] = "DeviceType.CPU"
    with pytest.raises(ValueError, match="CUPTI"):
        driver.validate_windows(rows)


def test_event_row_separates_device_and_host_times():
    event = SimpleNamespace(
        key="aten::bmm",
        device_type="DeviceType.CPU",
        count=2,
        cpu_time_total=10,
        self_cpu_time_total=3,
        device_time_total=5,
        self_device_time_total=4,
        input_shapes=[[32, 600, 4096]],
    )
    row = driver.event_row(event)
    assert row["device_us"] == 5 and row["self_cpu_us"] == 3
    assert "total_wall_seconds" not in row


def test_disabled_scope_does_not_call_profiler(monkeypatch):
    import torch

    def forbidden(*_args, **_kwargs):
        raise AssertionError("inactive call must not start profiler scope")

    monkeypatch.setattr(torch.profiler, "record_function", forbidden)
    called = []
    wrapper = driver.scoped(lambda value: called.append(value) or value, "scope", dict(value=False))
    assert wrapper(3) == 3 and called == [3]
