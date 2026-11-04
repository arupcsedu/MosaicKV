from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation import prompt_source_subphases as detail
from mosaickv.evaluation.construction_phase_timing import PhaseTimer
from mosaickv.evaluation.reconstruction_roles import _role


def test_disabled_scope_does_not_retain_tensors_or_measure_unrelated_roles():
    timer = PhaseTimer()
    original = torch.empty
    with detail.phase("validation"):
        pass
    with detail.use(timer):
        with detail.phase("validation"):
            pass
        tensor = torch.empty(2)
        assert tensor.numel() == 2
    assert timer.resolve() == [] and torch.empty is original


def test_nested_failure_restores_role_allocator_and_context():
    timer = PhaseTimer()
    original = torch.empty
    token = _role.set("prompt_source")
    try:
        with pytest.raises(RuntimeError), detail.use(timer):
            with detail.phase("validation"):
                assert torch.empty(2).numel() == 2
            with pytest.raises(ValueError, match="nonnested"), detail.use(timer):
                pass
            with pytest.raises(ValueError, match="registered"), detail.phase("made_up"):
                pass
            raise RuntimeError("control")
    finally:
        _role.reset(token)
    assert torch.empty is original
    assert {x["phase"] for x in timer.resolve()} == {
        "prompt_detail.validation",
        "prompt_detail.temporary_buffers",
    }
    assert not any("tensor" in k for row in timer.records for k in row)


def fixture_phases():
    return {
        f"{window}/prompt_detail.{name}": dict(
            calls=2, exclusive_wall_seconds=0.01, stream_elapsed_ms=1.0
        )
        for window in ("early", "middle", "late")
        for name in detail.NAMES - {"temporary_buffers"}
    }


def test_exclusive_leaf_aggregation_requires_windows_and_rejects_nonfinite():
    phases = fixture_phases()
    phases["early/prompt_source.reconstruction"] = dict(
        calls=2, exclusive_wall_seconds=99.0, stream_elapsed_ms=999.0
    )
    result = detail.totals(phases)
    assert result["operations"]["validation"]["exclusive_wall_seconds"] == 0.03
    assert result["tracked_temporary_allocation_calls"] == 0
    assert result["allocation_inventory_is_not_peak_evidence"]
    assert (
        detail.totals({}, require_source=False)["operations"]["projection_dispatch"]["calls"] == 0
    )
    for fault in ("missing", "nan", "negative", "unknown"):
        bad = deepcopy(phases)
        key = "early/prompt_detail.validation"
        if fault == "missing":
            del bad[key]
        elif fault == "nan":
            bad[key]["stream_elapsed_ms"] = float("nan")
        elif fault == "negative":
            bad[key]["exclusive_wall_seconds"] = -1
        else:
            bad["early/prompt_detail.unknown"] = bad[key]
        with pytest.raises(ValueError):
            detail.totals(bad)


def projection_fixture():
    h, n, d = 2, 16, 128
    body = SimpleNamespace(
        shape=(1, 1, n, h * d),
        bits=3,
        axis=2,
        group=128,
        codes=torch.zeros(n * h * d * 3 // 8, dtype=torch.uint8),
        minimum=torch.zeros(1),
        scale=torch.ones(1),
    )
    packet = SimpleNamespace(
        shape=(1, h, n, d),
        source=body,
        cos=torch.ones(1, n, d),
        sin=torch.zeros(1, n, d),
        source_head_group=1,
        source_row_tile=16,
        source_num_warps=None,
    )
    attention = SimpleNamespace(
        k_proj=torch.nn.Linear(h * d, h * d, bias=False).bfloat16(),
        v_proj=torch.nn.Linear(h * d, h * d, bias=False).bfloat16(),
    )
    target = torch.empty(1, h, n + 4, d, dtype=torch.bfloat16)
    return packet, attention, (target, target)


def test_projection_arguments_checks_and_subphases_preserve_original_launch(monkeypatch):
    from mosaickv.evaluation import request_kernel_dispatch, shared_source_kernel

    captured = []
    monkeypatch.setattr(
        request_kernel_dispatch, "launch", lambda *a, **kw: captured.append((a, kw))
    )
    packet, attention, targets = projection_fixture()
    shared_source_kernel.project(packet, attention, targets, component=1)
    args, options = captured[0]
    assert args[1] == (1, 2) and args[2][-2:] == (16, targets[0].stride(1))
    assert options == dict(
        WIDTH=256,
        DIM=128,
        BITS=3,
        COMPONENT=1,
        HEAD_GROUP=1,
        HAS_COLUMNS=False,
        ROW_TILE=16,
        REUSE_HEADERS=False,
        ROW4=False,
        FIRST=0,
        TAKE=0,
        V_PREFIX=False,
        num_warps=4,
        num_stages=3,
        enable_fp_fusion=False,
    )
    timer = PhaseTimer()
    token = _role.set("prompt_source")
    try:
        with detail.use(timer):
            shared_source_kernel.project(packet, attention, targets, component=1)
    finally:
        _role.reset(token)
    args2, options2 = captured[1]
    assert options2 == options and all(
        x is y for x, y in zip(args2[2][:-2], args[2][:-2], strict=True)
    )
    assert {x["phase"] for x in timer.resolve()} == {
        "prompt_detail.validation",
        "prompt_detail.indexing_views",
        "prompt_detail.projection_dispatch",
    }
    attention.k_proj.bias = torch.nn.Parameter(torch.zeros(256))
    with pytest.raises(ValueError, match="unchanged contiguous"):
        shared_source_kernel.project(packet, attention, targets)
    assert len(captured) == 2


def test_opt_in_nibble_transport_changes_only_code_pointer_and_layout_flag(monkeypatch):
    from mosaickv.evaluation import request_kernel_dispatch, shared_source_kernel
    from mosaickv.evaluation.row_major_source_codes import lookup, use

    calls = []
    monkeypatch.setattr(request_kernel_dispatch, "launch", lambda *a, **kw: calls.append((a, kw)))
    packet, attention, targets = projection_fixture()
    scratch = torch.zeros(16 * 256 // 2, dtype=torch.uint8)
    shared_source_kernel.project(packet, attention, targets, component=1)
    with use(packet.source, scratch):
        shared_source_kernel.project(packet, attention, targets, component=1)
    old_args, old_options = calls[0]
    new_args, new_options = calls[1]
    assert old_args[:2] == new_args[:2]
    assert old_args[2][0] is packet.source.codes and new_args[2][0] is scratch
    assert all(a is b for a, b in zip(old_args[2][1:-2], new_args[2][1:-2], strict=True))
    assert old_args[2][-2:] == new_args[2][-2:]
    assert new_options == dict(old_options, ROW4=True)
    assert old_options["ROW4"] is False and lookup(packet.source) is None
    assert len(calls) == 2


@pytest.mark.parametrize(
    "fault", ["component", "row_tile", "head_group", "warps", "bits", "stride"]
)
def test_invalid_projection_metadata_still_fails_before_launch(monkeypatch, fault):
    from mosaickv.evaluation import request_kernel_dispatch, shared_source_kernel

    def forbidden(*a, **kw):
        pytest.fail("invalid metadata reached GPU launch")

    monkeypatch.setattr(request_kernel_dispatch, "launch", forbidden)
    packet, attention, targets = projection_fixture()
    kwargs = dict(component=0)
    if fault == "component":
        kwargs["component"] = True
    elif fault == "row_tile":
        packet.source_row_tile = True
    elif fault == "head_group":
        packet.source_head_group = 4
    elif fault == "warps":
        packet.source_num_warps = 2
    elif fault == "bits":
        packet.source.bits = 2
    else:
        targets = tuple(t[..., ::2] for t in targets)
    with pytest.raises(ValueError):
        shared_source_kernel.project(packet, attention, targets, **kwargs)


def test_scatter_and_patch_keep_views_grid_and_scalar_arguments(monkeypatch):
    from mosaickv.evaluation import dual_projection_kernel as patcher
    from mosaickv.evaluation import head_local_precision_kernel as scatterer

    calls = []

    class Kernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append((grid, args, kwargs))

            return launch

    monkeypatch.setattr(scatterer, "_scatter_flat", Kernel())
    monkeypatch.setattr(patcher, "_patch", Kernel())
    body = SimpleNamespace(
        raw=True, shape=(1, 1, 2, 128), buffers=(torch.zeros(256, dtype=torch.bfloat16),)
    )
    positions = torch.tensor([0, 17], dtype=torch.int32)
    target = torch.empty(1, 2, 20, 128, dtype=torch.bfloat16)
    scatterer.scatter(body, positions, target, prompt_length=16)
    grid, args, kwargs = calls[-1]
    assert grid == (1,) and args[-3:] == (256, 16, target.stride(1))
    assert args[-4].data_ptr() == target.data_ptr() and kwargs == dict(D=128, RAW=True, BLOCK=1024)
    patch = SimpleNamespace(
        indices=torch.tensor([4, 20], dtype=torch.int32),
        values=torch.tensor([2.0, 3.0], dtype=torch.bfloat16),
    )
    patcher.apply_patch(patch, target[..., :16, :])
    grid, args, kwargs = calls[-1]
    assert grid == (1,) and args[-1].data_ptr() == target.data_ptr()
    assert kwargs == dict(COUNT=2, KEPT=16, HEAD_OUT=target.stride(-3), DIM=128, BLOCK=256)


def test_preparation_rejects_numerical_or_positive_parent_before_submission(monkeypatch, tmp_path):
    import evicastkv_prompt_source_profile as driver

    monkeypatch.setattr(driver, "clean_source", lambda _: "source")
    for fault in ("incomplete", "failed", "already_promoted"):
        decision = dict(complete=True, failures=[], descriptor_end_to_end_improvement_gate=False)
        if fault == "incomplete":
            decision["complete"] = False
        elif fault == "failed":
            decision["failures"] = ["bad control"]
        else:
            decision["descriptor_end_to_end_improvement_gate"] = True
        monkeypatch.setattr(driver.parent.common.previous, "read", lambda p, d=decision: d)
        with pytest.raises(ValueError, match="negative descriptor"):
            driver.prepare(tmp_path / fault, "source")


def test_analysis_preserves_missing_producers_without_submitting_GPU(monkeypatch, tmp_path):
    import evicastkv_prompt_source_profile as driver

    monkeypatch.setattr(
        driver.parent.common.previous.common, "inputs", lambda *a: {"source_sha": "s"}
    )
    driver.analyze(tmp_path, "p")
    decision = driver.parent.common.previous.read(tmp_path / "analysis/decision.json")
    assert not decision["complete"] and len(decision["failures"]) == 2
    assert decision["no_GPU_retry_or_sweep_submitted"] and decision["next_target"] is None
    assert not decision["paper_superiority_established"]
