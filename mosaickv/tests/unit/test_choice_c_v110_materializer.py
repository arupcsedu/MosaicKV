from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from mosaickv.evaluation.disjoint_samples import SampleProtocolError


def _module():
    package = Path(__file__).parents[2]
    path = package / "scripts/prepare_choice_c_v110_preflight.py"
    spec = importlib.util.spec_from_file_location("mosaickv_test_v110_materializer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v110_materializer_accepts_only_narrow_preflight_authority() -> None:
    module = _module()
    protocol = module._protocol()
    module.validate_protocol(protocol)
    module.require_authorization(protocol)


def test_v110_materializer_rejects_weakened_gate() -> None:
    module = _module()
    protocol = module._protocol()
    protocol["unchanged_memory_and_fidelity_gates"]["maximum_forward_kl"] = 0.03
    with pytest.raises(SampleProtocolError, match="changed"):
        module.validate_protocol(protocol)
