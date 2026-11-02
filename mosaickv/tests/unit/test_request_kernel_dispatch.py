from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.request_kernel_dispatch import launch, use_dispatch


class FakeKernel:
    arg_names = ("X", "N", "K")
    do_not_specialize = ("N",)

    def __init__(self, *, scratch=0):
        self.jit, self.direct = [], []
        self.compiled = Compiled(self)
        self.compiled.metadata = SimpleNamespace(global_scratch_size=scratch)

    def __getitem__(self, grid):
        def run(*args, **kwargs):
            self.jit.append((grid, args, kwargs))
            return self.compiled

        return run


class Compiled:
    def __init__(self, parent):
        self.parent = parent

    def __getitem__(self, grid):
        def run(*args):
            assert len(grid) == 3
            self.parent.direct.append((grid, args))

        return run


def test_compiled_reuse_changes_dynamic_values_but_preserves_options_and_no_tensor_capture():
    kernel, source = FakeKernel(), torch.ones(16)
    with use_dispatch(True) as state:
        for n in (16, 17, 18):
            assert launch(kernel, (2,), (source, n), K=4, num_warps=4) is kernel.compiled
        assert len(kernel.jit) == 1 and len(kernel.direct) == 2
        assert [x[1][1] for x in kernel.direct] == [17, 18]
        assert state["hits"] == 2 and state["misses"] == 1
        assert all(not isinstance(v, torch.Tensor) for key in state["handles"] for v in key)
        launch(kernel, (2,), (source, 2**33), K=4, num_warps=4)
        launch(kernel, (2,), (source[1:], 16), K=4, num_warps=4)
        launch(kernel, (2,), (source, 16), K=8, num_warps=4)
        launch(kernel, (2,), (source, 16), K=4, num_warps=8)
        assert state["misses"] == 5
    assert not state["handles"]
    launch(kernel, (2,), (source, 16), K=4)
    assert len(kernel.jit) == 6


def test_request_state_isolation_exception_cleanup_and_unknown_scratch_fail_closed():
    kernel, source = FakeKernel(), torch.ones(1)
    with pytest.raises(RuntimeError, match="stop"), use_dispatch(True) as state:
        launch(kernel, (1,), (source, 1), K=4)
        raise RuntimeError("stop")
    assert not state["handles"]
    with use_dispatch(True) as new:
        launch(kernel, (1,), (source, 1), K=4)
        assert new["misses"] == 1 and new["hits"] == 0
    for value in (None, 1):
        with pytest.raises(RuntimeError, match="scratch"), use_dispatch(True):
            launch(FakeKernel(scratch=value), (1,), (source, 1), K=4)
    for invalid in (1, "yes", None):
        with pytest.raises(ValueError), use_dispatch(invalid):
            pytest.fail("invalid policy")


def test_wrong_triton_revision_is_not_silently_accepted(monkeypatch):
    import triton

    monkeypatch.setattr(triton, "__version__", "unverified")
    with pytest.raises(NotImplementedError), use_dispatch(True):
        pytest.fail("wrong revision")
