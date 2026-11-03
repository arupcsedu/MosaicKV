from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation import native_execution_plan as plan


def test_weight_views_alias_original_and_are_request_private():
    weight = torch.randn(256, 256)
    with plan.use(True) as state:
        first = plan.weight_view(weight, 2, 128)
        second = plan.weight_view(weight, 2, 128)
        assert first is second
        assert first.untyped_storage().data_ptr() == weight.untyped_storage().data_ptr()
        assert state["weight_hits"] == 1 and state["weight_misses"] == 1
        with pytest.raises(ValueError, match="nonnested"), plan.use(True):
            pass
        with plan.suspend(True):
            assert plan.weight_view(weight, 2, 128) is not first
        assert state["weight_hits"] == 1
    assert state["weights"] == {} and state["restores"] == {}
    with plan.use(True) as next_state:
        assert plan.weight_view(weight, 2, 128) is not first
        assert next_state["weight_misses"] == 1


def test_weight_mutation_and_exception_clear_all_aliases():
    weight = torch.randn(256, 256)
    with pytest.raises(RuntimeError, match="version changed"), plan.use(True) as state:
        plan.weight_view(weight, 2, 128)
        weight.add_(1)
        plan.weight_view(weight, 2, 128)
    assert state["weights"] == {}
    with plan.use(False):
        assert plan.weight_view(weight, 2, 128).shape == (2, 256, 128)


class Kernel:
    arg_names = [str(i) for i in range(13)] + ["D", "RAW"]

    def __init__(self):
        self.compiled_calls = []
        self.direct_calls = []

    def __getitem__(self, grid):
        owner = self

        class Compiled:
            metadata = SimpleNamespace(global_scratch_size=0)

            def __getitem__(self, dimensions):
                def run(*args):
                    owner.direct_calls.append((dimensions, args))

                return run

        def initial(*args, **kwargs):
            self.compiled_calls.append((grid, args, kwargs))
            return Compiled()

        return initial


def arguments():
    return (
        *[torch.zeros(2, dtype=torch.int32) for _ in range(8)],
        torch.zeros(1, 2, 10, 128, dtype=torch.int16),
        256,
        2,
        5,
        1280,
    )


def test_same_compiled_binary_uses_current_output_and_stride_without_cross_request_cache():
    kernel, args, options = Kernel(), arguments(), dict(D=128, RAW=True, num_warps=4)
    with plan.use(True) as state:
        compiled = plan.launch(kernel, (2,), args, options)
        output = torch.empty(1, 2, 20, 128, dtype=torch.int16)
        changed = (*args[:8], output, *args[9:12], 2560)
        assert plan.launch(kernel, (2,), changed, options) is compiled
        assert state["restore_misses"] == 1 and state["restore_hits"] == 1
        assert kernel.direct_calls[0][1][8] is output
        assert kernel.direct_calls[0][1][12] == 2560
        assert kernel.direct_calls[0][1][-2:] == (128, True)
    assert not state["restores"]
    with plan.use(True) as next_state:
        plan.launch(kernel, (2,), args, options)
        assert next_state["restore_misses"] == 1
    assert len(kernel.compiled_calls) == 2


def test_changed_restore_abi_or_negative_stride_fails_closed():
    kernel, args, options = Kernel(), arguments(), dict(D=128, RAW=True, num_warps=4)
    with plan.use(True):
        plan.launch(kernel, (2,), args, options)
        with pytest.raises(RuntimeError, match="ABI changed"):
            plan.launch(kernel, (2,), args, dict(options, num_warps=8))
        with pytest.raises(ValueError, match="stride"):
            plan.launch(kernel, (2,), (*args[:-1], -1), options)


def test_prepared_operator_gate_keeps_original_threshold_and_requires_activity():
    import evicastkv_native_restore_followup as driver
    from test_native_restore_followup import rows

    reports = rows()
    for report in reports:
        for case in report["cases"]:
            case["correctness"]["prepared"] = dict(case["correctness"]["fused"])
            case["aggregate"]["prepared"] = dict(mean_cuda_ms=0.7)
            case["prepared_activity"] = dict(weight_hits=10, restore_hits=10)
    assert driver.eligible(reports, "prepared")
    reports[0]["cases"][0]["prepared_activity"]["restore_hits"] = 0
    assert not driver.eligible(reports, "prepared")
