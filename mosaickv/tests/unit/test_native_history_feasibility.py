import pytest
import torch

from mosaickv.evaluation.native_history_feasibility import (
    ROWS,
    Engine,
    RepairBudgetError,
    duplicated_phases,
    make_packet,
    validate_rows,
)


def test_only_complete_bounded_epochs():
    assert tuple(range(16, 113, 16)) == ROWS
    for n in ROWS:
        validate_rows(n)
    for bad in (True, 16.0, 0, 15, 17, 128, -16):
        with pytest.raises(ValueError):
            validate_rows(bad)


def test_budget_failure_keeps_its_measured_diagnostic():
    diagnostic = dict(rows=16, repair_bytes=32769, registered_cap_bytes=32768)
    error = RepairBudgetError(diagnostic)
    assert error.diagnostic is diagnostic
    assert "unchanged" in str(error)


def test_frozen_factory_uses_actual_one_argument_api(monkeypatch):
    from mosaickv.evaluation import joint_kernel_compatibility as mod
    from mosaickv.evaluation.joint_all_layer_compatibility import FROZEN

    algorithm, record = object(), object()

    class Plans:
        def __init__(self, rows):
            assert rows == ROWS
            self.candidates = [(algorithm, record)]
            self.plans = {n: (object(), object(), object(), None, None) for n in rows}

    monkeypatch.setattr(mod, "KernelPlans", Plans)
    monkeypatch.setattr(mod, "fingerprint", lambda r: FROZEN if r is record else None)
    engine = Engine(torch.device("cpu"), frozen=True)
    assert all(p[3] is algorithm and p[4] is record for p in engine.plans.plans.values())
    assert engine.source.shape == (128, 4096)


def test_rotary_fixture_matches_actual_llama_not_independent_halves():
    theta = torch.randn((1, 16, 64))
    cos, sin = duplicated_phases(theta)
    assert cos.shape == sin.shape == (1, 16, 128)
    for t in (cos, sin):
        assert torch.equal(t[..., :64].view(torch.int16), t[..., 64:].view(torch.int16))
    with pytest.raises(ValueError, match="duplicate-half"):
        make_packet(
            torch.zeros((1, 1, 16, 4096), dtype=torch.bfloat16),
            torch.randn((1, 16, 128)).bfloat16(),
            sin,
        )
    with pytest.raises(ValueError):
        duplicated_phases(torch.zeros((1, 16, 128)))
