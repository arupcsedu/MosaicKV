import ctypes as C
from types import SimpleNamespace

import pytest

from mosaickv.evaluation import frozen_joint_plan as frozen
from mosaickv.evaluation import joint_incoming_projection as joint


def test_configuration_and_attribute_widths_are_the_verified_ones():
    attrs = frozen.frozen_attributes()
    assert tuple(value for _, value, _ in attrs) == frozen.FROZEN[1:]
    assert [(attr, C.sizeof(kind)) for attr, _, kind in attrs] == [
        (1, 4),
        (2, 4),
        (3, 4),
        (4, 4),
        (5, 4),
        (6, 4),
        (7, 2),
    ]


class Function:
    def __init__(self):
        self.calls = 0

    def __call__(self, *args):
        self.calls += 1
        return 0


def fake_plans(monkeypatch, *, corrupt=False):
    names = (
        "cublasLtMatmulAlgoInit",
        "cublasLtMatmulAlgoConfigSetAttribute",
        "cublasLtMatmulAlgoCheck",
    )
    lib = SimpleNamespace(**{name: Function() for name in names})
    calls = []

    def call(name, *args):
        assert name in names, "the frozen path must never query IDs or search heuristics"
        calls.append(name)
        return 0

    def init(self, rows, width):
        self.library = SimpleNamespace(
            lib=lib, call=call, abi=dict(header_hashes={"/fixed/cublasLt.h": "hash"})
        )
        self.closed, self.handle, self.operation = False, "handle", "op"
        self.plans = {n: ("a", "w", "d", frozen.Algo(), {}) for n in rows}

    def describe(self, algo):
        keys = (
            "algorithm_id",
            "tile",
            "split_k",
            "reduction",
            "swizzle",
            "custom",
            "stages",
            "inner",
        )
        record = dict(zip(keys, frozen.FROZEN, strict=True), numerical_flags=0x200)
        if corrupt:
            record["tile"] += 1
        return record

    monkeypatch.setattr(frozen.Plans, "__init__", init)
    monkeypatch.setattr(frozen.Plans, "close", lambda self: setattr(self, "closed", True))
    monkeypatch.setattr(frozen.KernelPlans, "describe", describe)
    monkeypatch.setattr(frozen, "header_proof", lambda *a: None)
    return calls


def test_exact_plan_is_initialized_and_checked_without_the_old_search(monkeypatch):
    calls = fake_plans(monkeypatch)
    plans = frozen.FrozenKernelPlans((88, 89, 128))
    assert plans.ids == [31]
    assert len(plans.candidates) == 1
    assert frozen.fingerprint(plans.candidates[0][1]) == frozen.FROZEN
    assert calls == ["cublasLtMatmulAlgoInit"] + ["cublasLtMatmulAlgoConfigSetAttribute"] * 7
    assert plans.library.lib.cublasLtMatmulAlgoCheck.calls == 3


def test_unexpected_configuration_is_rejected_not_replaced_with_a_search(monkeypatch):
    fake_plans(monkeypatch, corrupt=True)
    with pytest.raises(RuntimeError, match="unavailable"):
        frozen.FrozenKernelPlans((88, 89, 128))


def test_frozen_plan_is_explicit_and_request_private():
    @joint.managed
    def call(**kwargs):
        return None, None, {}

    assert call()[2]["joint_incoming_projection"]["frozen_joint_plan"] == dict(
        requested=False, actual_engines=0
    )
    assert call(frozen_joint_plan=True)[2]["joint_incoming_projection"][
        "frozen_joint_plan"
    ] == dict(requested=True, actual_engines=0)
    with pytest.raises(ValueError, match="boolean"):
        call(frozen_joint_plan=1)
    assert joint._scope.get() is None


def test_other_geometries_cannot_silently_reuse_the_audited_configuration():
    with pytest.raises(NotImplementedError, match="audited"):
        frozen.FrozenKernelPlans((87, 88, 128))
