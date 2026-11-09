import weakref

import pytest
import torch

from mosaickv.evaluation import fused_history_unpack as source
from mosaickv.evaluation import prepared_patch_dispatch as patches
from mosaickv.evaluation.prepared_address_dispatch import addresses, use


def test_fixture_conversion_preserves_actual_addresses_without_tensor_copies():
    args = (
        torch.zeros(5, dtype=torch.int32),
        torch.empty(0, dtype=torch.uint8),
        torch.ones(17, dtype=torch.bfloat16),
    )
    ptrs = addresses(args, device_type="cpu")
    assert ptrs == tuple(t.data_ptr() for t in args)
    assert all(type(p) is int for p in ptrs)
    refs = tuple(weakref.ref(t) for t in args)
    del args
    assert all(t() is None for t in refs)


@pytest.mark.parametrize("args", [(), (None,), (torch.ones(2),)])
def test_actual_launcher_never_accepts_cpu_or_missing_buffers(args):
    with pytest.raises(ValueError):
        addresses(args)


def test_gradients_and_invalid_empty_nonzero_pointer_fail():
    with pytest.raises(ValueError, match="nongrad"):
        addresses((torch.ones(2, requires_grad=True),), device_type="cpu")


def test_policy_restores_both_dispatchers_after_failure_and_disabled_control():
    old_source, old_patch = source.replay, patches.replay
    with pytest.raises(RuntimeError, match="sentinel"), use() as report:
        assert source.replay is not old_source and patches.replay is not old_patch
        with pytest.raises(ValueError, match="nonnested"), use():
            pass
        raise RuntimeError("sentinel")
    assert source.replay is old_source and patches.replay is old_patch
    assert report["retained_tensor_owners"] == report["additional_GPU_storage_bytes"] == 0
    with use(False):
        assert source.replay is old_source and patches.replay is old_patch
