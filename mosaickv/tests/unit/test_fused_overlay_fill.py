from dataclasses import replace

import pytest
import torch

from mosaickv.evaluation.storage_six_overlay import LayerOverlay, OverlayPolicy


def make_overlay(backend, device="cpu"):
    generator = torch.Generator().manual_seed(13)

    def values(rows):
        return torch.randn(rows, 128, generator=generator).bfloat16().to(device)

    return LayerOverlay(
        head=1,
        positions=(2, 4),
        exact_ids=(8, 10),
        exact_pair=(values(2), values(2)),
        prototype_pair=(values(1), values(1)),
        phases=(torch.ones(3, 128, device=device), torch.zeros(3, 128, device=device)),
        residual_pair=(values(2), values(2)),
        anchor=2,
        dispersion=0.2,
        evidence={},
        fill_backend=backend,
    )


def test_fused_backend_is_explicit_and_keeps_original_default():
    with pytest.raises(ValueError, match="isolated full score-risk"):
        OverlayPolicy(overlay_fill_backend="fused")
    p = OverlayPolicy(repair_execution="score_risk", repair_policy="prototype_risk")
    assert p.overlay_fill_backend == "torch"
    assert replace(p, overlay_fill_backend="fused").overlay_fill_backend == "fused"


def test_fused_fill_retires_unused_mixed_scratch_and_rejects_cpu_execution():
    from mosaickv.evaluation.storage_six_overlay import _bytes

    a, b = make_overlay("torch"), make_overlay("fused")
    assert not b.mixed
    assert _bytes(a.gpu_tensors()) - _bytes(b.gpu_tensors()) == 1024
    with pytest.raises(ValueError, match="BF16 CUDA"):
        b.fill(0, torch.zeros(1, 2, 17, 128).bfloat16())


def test_fused_reader_import_is_cpu_safe():
    from mosaickv.evaluation.fused_overlay_fill import fill

    assert callable(fill)
