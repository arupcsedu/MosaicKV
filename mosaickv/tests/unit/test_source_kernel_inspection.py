from types import SimpleNamespace

import torch

from mosaickv.evaluation import shared_source_kernel
from mosaickv.evaluation.precision_tier import pack


def test_kernel_inspection_preserves_unavailable_counters_and_legacy_return(monkeypatch):
    class Kernel:
        def __getitem__(self, grid):
            assert grid == (3, 1)

            def launch(*args, **kwargs):
                assert kwargs["HEAD_GROUP"] == 2 and not kwargs["enable_fp_fusion"]
                return SimpleNamespace(
                    n_regs=42, n_spills=None, metadata=SimpleNamespace(shared=1234)
                )

            return launch

    monkeypatch.setattr(shared_source_kernel, "_project", Kernel())
    packet = SimpleNamespace(
        shape=(1, 2, 33, 128),
        source=pack(torch.ones(1, 1, 33, 256).bfloat16(), bits=3, axis=2),
        source_head_group=2,
        cos=torch.ones(1, 33, 128).bfloat16(),
        sin=torch.zeros(1, 33, 128).bfloat16(),
    )
    attention = SimpleNamespace(
        k_proj=torch.nn.Linear(256, 256, bias=False).bfloat16(),
        v_proj=torch.nn.Linear(256, 256, bias=False).bfloat16(),
    )
    targets = tuple(torch.empty(1, 2, 33, 128, dtype=torch.bfloat16) for _ in range(2))
    assert shared_source_kernel.project(packet, attention, targets) is None
    info = shared_source_kernel.project(packet, attention, targets, inspect_kernel=True)
    assert info["registers"] == 42 and info["spills"] is None
    assert info["shared_memory_bytes"] == 1234
