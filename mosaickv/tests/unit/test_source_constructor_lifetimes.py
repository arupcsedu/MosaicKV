"""Release dead construction buffers, never remove verification or metadata."""

import weakref
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves


class WeakFloatInventory(TorchDispatchMode):
    def __init__(self, threshold):
        super().__init__()
        self.threshold, self.references = threshold, []

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        result = function(*args, **(kwargs or {}))
        for value in tree_leaves(result):
            if (
                isinstance(value, torch.Tensor)
                and value.dtype == torch.float32
                and value.numel() >= self.threshold
            ):
                self.references.append(weakref.ref(value))
        return result

    def assert_released(self):
        assert self.references
        assert all(reference() is None for reference in self.references)


def test_conditional_preparation_does_not_overlap_packing_workspace(monkeypatch):
    from mosaickv.evaluation import conditional_precision as module

    values = torch.randn(1, 2, 16, 128).bfloat16()
    rows = torch.zeros(2, 16, dtype=torch.bool)
    rows[:, :4] = True
    inventory = WeakFloatInventory(values.numel())

    def pack(prepared, **kwargs):
        inventory.assert_released()
        assert prepared.dtype == torch.bfloat16
        assert kwargs == dict(bits=3, axis=2)
        return "all_original_preparation_checks_executed"

    monkeypatch.setattr(module, "pack", pack)
    with inventory:
        assert (
            module.pack_unprotected(values, rows, bits=3, axis=2, bf16_prepare=False)
            == "all_original_preparation_checks_executed"
        )


def test_quantizer_source_released_before_integer_serialization(monkeypatch):
    from mosaickv.evaluation import precision_tier as module

    values = torch.randn(1, 2, 16, 128).bfloat16()
    inventory = WeakFloatInventory(values.numel())
    original = torch.arange

    def arange(*args, **kwargs):
        inventory.assert_released()
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "arange", arange)
    with inventory:
        packed = module.pack(values, bits=3, axis=2)
    assert packed.codes.dtype == torch.uint8


def test_transport_releases_source_and_old_pair_before_independent_verification(monkeypatch):
    from mosaickv.evaluation import shared_source_format as module

    alive, checkpoints = {}, []

    def source(_packet):
        tensor = torch.zeros(1, 8, 256, dtype=torch.bfloat16)
        alive["source"] = weakref.ref(tensor)
        return tensor

    def project_value(x):
        tensor = torch.zeros_like(x)
        alive["native_value"] = weakref.ref(tensor)
        return tensor

    @dataclass(frozen=True)
    class Packet:
        shape: tuple
        cos: object
        sin: object
        row_positions: object
        exact: tuple
        transport_patches: tuple = ()
        input_overlay: object = None
        protected_patches_disjoint: bool = False

        def materialize(self, _attention):
            assert alive["source"]() is None
            assert alive["native_value"]() is None
            if self.transport_patches:
                assert all(reference() is None for reference in alive["old_pair"])
                checkpoints.append("independent_verification")
            else:
                checkpoints.append("original_materialization")
            pair = tuple(torch.zeros(self.shape, dtype=torch.bfloat16) for _ in range(2))
            alive["old_pair"] = tuple(weakref.ref(t) for t in pair)
            return pair

    monkeypatch.setattr(
        "mosaickv.evaluation.shared_source_kernel.decode_source_for_construction", source
    )
    monkeypatch.setattr(module, "decoded", lambda body: body)
    packet = Packet(
        (1, 2, 8, 128),
        torch.ones(1, 8, 128, dtype=torch.bfloat16),
        torch.zeros(1, 8, 128, dtype=torch.bfloat16),
        torch.empty(0, dtype=torch.int32),
        tuple(torch.empty(1, 1, 0, 128, dtype=torch.bfloat16) for _ in range(2)),
    )
    attention = SimpleNamespace(k_proj=lambda x: torch.zeros_like(x), v_proj=project_value)
    output, proof = module.fit_transport(packet, attention)
    assert checkpoints == ["original_materialization", "independent_verification"]
    assert proof["frozen_construction_tolerance_pass"]
    assert len(output.transport_patches) == 2
    assert output.protected_patches_disjoint
