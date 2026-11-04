import pytest
import torch
from test_prompt_source_subphases import projection_fixture


def test_window_uses_global_rows_and_separate_prefix_value_stride(monkeypatch):
    from mosaickv.evaluation import request_kernel_dispatch, shared_source_kernel

    calls = []
    monkeypatch.setattr(request_kernel_dispatch, "launch", lambda *a, **k: calls.append((a, k)))
    packet, attention, targets = projection_fixture()
    prefix = torch.empty((1, 2, 8, 128), dtype=torch.bfloat16)
    shared_source_kernel.project(packet, attention, (targets[0], prefix), window=(0, 8, True))
    args, options = calls[0]
    assert args[1] == (1, 2)
    assert args[2][9] is targets[0] and args[2][10] is prefix
    assert options["FIRST"] == 0 and options["TAKE"] == 8 and options["V_PREFIX"]
    assert options["num_warps"] == 16 and options["COMPONENT"] == -1
    shared_source_kernel.project(packet, attention, targets, component=1, window=(8, 8, False))
    assert calls[1][1]["FIRST"] == 8 and calls[1][1]["V_PREFIX"] is False
    assert calls[1][0][2][-2:] == (16, targets[0].stride(1))


@pytest.mark.parametrize(
    "window",
    [
        (0, 0, True),
        (-1, 8, True),
        (0, 17, True),
        (True, 8, False),
        [0, 8, True],
        (0, 8, 1),
        (8, 8, True),
    ],
)
def test_invalid_windows_fail_before_gpu(monkeypatch, window):
    from mosaickv.evaluation import request_kernel_dispatch, shared_source_kernel

    monkeypatch.setattr(request_kernel_dispatch, "launch", lambda *a, **k: pytest.fail("launch"))
    packet, attention, targets = projection_fixture()
    with pytest.raises(ValueError):
        shared_source_kernel.project(packet, attention, targets, window=window)


def test_prefix_alias_or_wrong_component_is_rejected():
    from mosaickv.evaluation.shared_source_kernel import project

    packet, attention, targets = projection_fixture()
    with pytest.raises(ValueError, match="separate charged"):
        project(packet, attention, targets, window=(0, 16, True))
    with pytest.raises(ValueError, match="both components"):
        project(packet, attention, targets, component=1, window=(0, 16, True))
