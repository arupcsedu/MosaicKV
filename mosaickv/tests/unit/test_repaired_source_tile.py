from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from test_prompt_source_subphases import projection_fixture

from mosaickv.evaluation import repaired_source_tile as tile
from mosaickv.evaluation.exact_storage import SparseBitPatch
from mosaickv.evaluation.repaired_tile_operator import validate_patches


def fixture():
    packet, attention, targets = projection_fixture()
    packet.cos, packet.sin = packet.cos.bfloat16(), packet.sin.bfloat16()
    return packet, attention, targets[0]


@pytest.mark.parametrize("component", [0, 1])
def test_launch_uses_original_storage_weights_and_strided_destination(monkeypatch, component):
    packet, attention, target = fixture()
    calls, rotations = [], []

    class Kernel:
        def __getitem__(self, grid):
            def run(*args, **kwargs):
                calls.append((grid, args, kwargs))
                return SimpleNamespace(
                    n_regs=100,
                    n_spills=0,
                    metadata=SimpleNamespace(global_scratch_size=0, shared=36000),
                )

            return run

    from mosaickv.evaluation import shared_source_kernel

    monkeypatch.setattr(tile, "_project", Kernel())
    monkeypatch.setattr(shared_source_kernel, "rotate_native", lambda *args: rotations.append(args))
    info = tile.project_raw(packet, attention, component, target, inspect_kernel=True)
    grid, args, options = calls[0]
    assert grid == (1, 2) and args[0] is packet.source.codes
    assert args[1] is packet.source.minimum and args[2] is packet.source.scale
    assert args[5] is (attention.k_proj, attention.v_proj)[component].weight
    assert args[6] is target and args[-2:] == (16, target.stride(1))
    assert options == dict(
        WIDTH=256, HAS_COLUMNS=False, num_warps=4, num_stages=3, enable_fp_fusion=False
    )
    assert info["no_full_source_or_output_workspace"]
    assert len(rotations) == int(component == 0)
    if rotations:
        assert rotations[0] == (target, packet.cos, packet.sin, 0, 16)


@pytest.mark.parametrize("fault", ["component", "weight", "shape", "phase", "bits", "stride"])
def test_unsafe_geometry_or_weight_changes_rejected(fault):
    packet, attention, target = fixture()
    component = 0
    if fault == "component":
        component = True
    elif fault == "weight":
        attention.k_proj = torch.nn.Linear(256, 256, bias=True).bfloat16()
    elif fault == "shape":
        packet.shape = (2, 2, 16, 128)
    elif fault == "phase":
        packet.cos = packet.cos.float()
    elif fault == "bits":
        packet.source = replace(packet.source, bits=4)
    else:
        target = target.transpose(-1, -2)
    with pytest.raises(ValueError):
        tile.validate(packet, attention, component, target)


def test_repair_budget_counts_whole_underlying_storage_not_views():
    indices = torch.empty(32768, dtype=torch.int32)
    values = torch.empty(32768, dtype=torch.bfloat16)
    empty = SparseBitPatch(torch.empty(0, dtype=torch.int32), torch.empty(0, dtype=torch.bfloat16))
    patch = SparseBitPatch(indices[:1], values[:1])
    with pytest.raises(RuntimeError, match="repair bytes"):
        validate_patches((patch, empty))
    patch = SparseBitPatch(
        torch.tensor([0, 7], dtype=torch.int32), torch.tensor([0.0, -0.0], dtype=torch.bfloat16)
    )
    assert validate_patches((patch, empty)) == 12
    for bad in ([patch, empty], (patch,)):
        with pytest.raises(ValueError):
            validate_patches(bad)


@pytest.mark.parametrize("fault", [None, "missing", "stack", "local", "scratch"])
def test_cpu_compiler_gate_rejects_spills_and_incomplete_column_paths(fault):
    from evicastkv_repaired_tile_operator import validate_proof

    proof = dict(
        rows=[dict(columns=c, stack_bytes=0, local_bytes=0, scratch_bytes=0) for c in (False, True)]
    )
    if fault == "missing":
        proof["rows"].pop()
    elif fault:
        proof["rows"][1][f"{fault}_bytes"] = 8
    if fault:
        with pytest.raises(ValueError):
            validate_proof(proof)
    else:
        validate_proof(proof)
