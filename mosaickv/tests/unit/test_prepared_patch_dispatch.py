import weakref
from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.prepared_patch_dispatch import launch, report, use


class Kernel:
    def __init__(self, scratch=0):
        self.calls = []
        self.compiled = SimpleNamespace(metadata=SimpleNamespace(global_scratch_size=scratch))

    def __getitem__(self, grid):
        kernel = self

        class Compiled:
            metadata = kernel.compiled.metadata

            def __getitem__(self, full_grid):
                def run(*args):
                    kernel.calls.append(("reuse", full_grid, tuple(x for x in args[3:])))

                return run

        def run(*args, **kwargs):
            assert len(args) == 3
            assert set(kwargs) == {"COUNT", "KEPT", "HEAD_OUT", "DIM", "BLOCK"}
            ordered = tuple(kwargs[n] for n in ("COUNT", "KEPT", "HEAD_OUT", "DIM", "BLOCK"))
            self.calls.append(("original", grid, ordered))
            return Compiled()

        return run


def args():
    return (
        torch.tensor([0, 1], dtype=torch.int32),
        torch.zeros(2, dtype=torch.int16),
        torch.zeros(1, 2, 7, 128, dtype=torch.int16),
        2,
        5,
        7 * 128,
        128,
        256,
    )


def test_exact_arguments_reused_but_changed_stride_or_pointer_gets_new_plan():
    kernel, arguments = Kernel(), args()
    with use(True) as state:
        launch(kernel, (1,), arguments)
        launch(kernel, (1,), arguments)
        launch(kernel, (1,), (*arguments[:5], 8 * 128, *arguments[6:]))
        launch(kernel, (1,), (arguments[0].clone(), *arguments[1:]))
    assert report(state)["hits"] == 1 and report(state)["misses"] == 3
    assert state["handles"] == {}
    assert kernel.calls[:2] == [
        ("original", (1,), arguments[3:]),
        ("reuse", (1, 1, 1), arguments[3:]),
    ]


def test_default_keeps_original_dispatch_and_has_no_owners():
    kernel = Kernel()
    with use(False) as state:
        launch(kernel, (1,), args())
        assert state is None
    assert kernel.calls[0][0] == "original"
    assert report(state)["tensor_owners_retained"] == 0


def test_disabled_scope_is_still_nonnested_and_exception_reset_is_safe():
    with use(False), pytest.raises(ValueError), use(True):
        pass
    with use(True) as state:
        assert not state["handles"]


def test_plans_do_not_retain_tensor_or_stale_request_state():
    kernel, arguments = Kernel(), args()
    tracked = [weakref.ref(tensor) for tensor in arguments[:3]]
    with use(True) as state:
        launch(kernel, (1,), arguments)
        del arguments
        assert all(ref() is None for ref in tracked)
    assert not state["handles"]
    with use(True) as other:
        assert other["hits"] == other["misses"] == 0


@pytest.mark.parametrize("fault", ["nested", "not_bool", "scratch", "geometry"])
def test_invalid_or_opaque_launches_fail_closed(fault):
    if fault == "nested":
        with use(True), pytest.raises(ValueError), use(True):
            pass
    elif fault == "not_bool":
        with pytest.raises(ValueError), use(1):
            pass
    else:
        with use(True), pytest.raises((RuntimeError, ValueError)):
            launch(
                Kernel(scratch=int(fault == "scratch")),
                (1,),
                args() if fault == "scratch" else (*args()[:3], -1, *args()[4:]),
            )
