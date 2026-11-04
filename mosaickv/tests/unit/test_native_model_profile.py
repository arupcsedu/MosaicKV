from types import SimpleNamespace

import evicastkv_native_model_profile as driver
import pytest


def test_windows_require_actual_cuda_activity_and_exact_registered_steps():
    rows = [dict(step=i, actual_kernels=dict(count=2, positive_count=2)) for i in driver.STEPS]
    driver.validate_windows(rows)
    with pytest.raises(ValueError, match="unique"):
        driver.validate_windows(rows[:-1])
    rows[0]["actual_kernels"]["positive_count"] = 1
    with pytest.raises(ValueError, match="CUPTI"):
        driver.validate_windows(rows)


def test_annotations_are_not_kernels_and_zero_duration_is_not_valid(tmp_path):
    import json

    path = tmp_path / "trace.json"
    path.write_text(
        json.dumps(
            dict(
                traceEvents=[
                    dict(cat="gpu_user_annotation", ph="X", name="scope", dur=100),
                    dict(cat="kernel", ph="X", name="a", dur=0),
                    dict(cat="kernel", ph="X", name="b", dur=2),
                    dict(cat="kernel", ph="X", name="b", dur=3),
                ]
            )
        )
    )
    result = driver.actual_kernels(path)
    assert result["count"] == 3 and result["positive_count"] == 2
    assert result["sum_kernel_us"] == 5
    assert result["top"][0] == dict(name="b", count=2, device_us=5)
    with pytest.raises(ValueError, match="EVERY"):
        driver.validate_windows([dict(step=i, actual_kernels=result) for i in driver.STEPS])


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
