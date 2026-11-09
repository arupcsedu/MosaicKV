"""Default-off dispatch of validated CUDA addresses to the SAME compiled binary.

Pinned Triton 3.4.0 explicitly accepts uint64 addresses. First compilation still
uses real tensors and its CUDA attribute validation. Replays keep private tensor
lifetimes/device checks but avoid redundant CUDA pointer-attribute queries.
There are NO new kernels, GPU pointer tables, workspace or retained owners.
"""

import hashlib
from contextlib import contextmanager
from contextvars import ContextVar
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

import torch

DRIVER_SHA = "448c23dda5888b37a97afa2c1d688358c513e087c4dacfa41ce826827c7602f5"
_scope = ContextVar("evicastkv_prepared_address_dispatch", default=None)


def verify_driver():
    import triton

    path = Path(find_spec("triton.backends.nvidia.driver").origin)
    if triton.__version__ != "3.4.0" or hashlib.sha256(path.read_bytes()).hexdigest() != DRIVER_SHA:
        raise NotImplementedError("exact audited Triton 3.4.0 CUDA launcher required")


def addresses(arguments, *, device_type="cuda"):
    """CPU selection is fixture-only; neither compiles nor executes a kernel."""
    if not arguments or any(not isinstance(t, torch.Tensor) for t in arguments):
        raise ValueError("complete live tensor arguments required")
    device = arguments[-1].device
    if device.type != device_type or any(t.device != device or t.requires_grad for t in arguments):
        raise ValueError("private nongrad buffers on one device required")
    if device_type == "cuda" and torch.cuda.current_device() != device.index:
        raise RuntimeError("current CUDA device differs from the private source")
    pointers = tuple(t.data_ptr() for t in arguments)
    if any(
        not 0 <= p < 2**64 or (not p and t.numel())
        for t, p in zip(arguments, pointers, strict=True)
    ):
        raise ValueError("valid uint64 addresses; NULL only for masked empty buffers")
    return pointers


@contextmanager
def use(enabled=True):
    from mosaickv.evaluation import fused_history_unpack as source
    from mosaickv.evaluation import prepared_patch_dispatch as patches

    if type(enabled) is not bool or _scope.get() is not None:
        raise ValueError("explicit nonnested prepared address policy required")
    verify_driver()
    report = dict(
        enabled=enabled,
        source_replays=0,
        patch_replays=0,
        pointer_arguments=0,
        additional_GPU_storage_bytes=0,
        retained_tensor_owners=0,
        first_tensor_launch_validation_unchanged=True,
        compiled_kernel_and_geometry_unchanged=True,
        CUDA_driver_source_sha=DRIVER_SHA,
    )
    token = _scope.set(report)

    def source_replay(entry, args):
        if len(args) != 24:
            raise ValueError("complete original mixed source ABI required")
        values = addresses(args)
        entry.runner(*values, *entry.geometry.values())
        report["source_replays"] += 1
        report["pointer_arguments"] += 24

    def patch_replay(runner, args):
        if len(args) != 8:
            raise ValueError("complete original sparse patch ABI required")
        values = addresses(args[:3])
        runner(*values, *args[3:])
        report["patch_replays"] += 1
        report["pointer_arguments"] += 3

    try:
        if enabled:
            with (
                patch.object(source, "replay", source_replay),
                patch.object(patches, "replay", patch_replay),
            ):
                yield report
        else:
            yield report
    finally:
        _scope.reset(token)


def gpu_controls():
    from mosaickv.evaluation.fused_history_unpack import gpu_controls as source_controls
    from mosaickv.evaluation.prepared_patch_dispatch import gpu_controls as patch_controls

    with use() as report:
        sources = source_controls()
        patches = patch_controls()
    if report["source_replays"] < 10 or not report["patch_replays"]:
        raise RuntimeError("both actual cached-kernel address paths must execute")
    return dict(
        sources=sources,
        patches=patches,
        dispatch=report,
        actual_CUDA_kernels=True,
        pretrained_weights_loaded=False,
    )
