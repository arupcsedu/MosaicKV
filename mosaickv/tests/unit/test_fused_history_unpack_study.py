import importlib.util
from pathlib import Path

import pytest


def load(name):
    path = Path(__file__).resolve().parents[2] / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_timing_schedule_keeps_storage_and_same_precision_controls():
    module = load("fused_history_unpack_timing")
    from unittest.mock import patch

    with patch.object(module.timing, "ARMS", module.ARMS):
        rows = module.timing.schedule(42)
        assert rows == module.timing.schedule(42)
    assert len(rows) == 75
    for arm in module.ARMS:
        assert sum(row["arm"] == arm and not row["warmup"] for row in rows) == 20
    assert module.POLICIES[module.ARMS[0]] == dict(history=False, mixed=False)
    assert module.POLICIES[module.ARMS[2]] == dict(history=True, mixed=True, fused_unpack=True)


def test_no_timing_submission_when_operator_model_gate_fails(tmp_path, monkeypatch):
    module = load("fused_history_unpack_timing")
    monkeypatch.setattr(
        module.timing.base.common, "read", lambda _: dict(bitwise_and_persistent_storage_gate=False)
    )
    monkeypatch.setattr(
        module.timing.base.common, "submit", lambda *a: pytest.fail("must not submit")
    )
    root = tmp_path / "timing"
    with pytest.raises(ValueError, match="parity/storage"):
        module.prepare(root)
    assert not root.exists()
