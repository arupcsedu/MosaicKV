from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

from mosaickv.evaluation.disjoint_samples import SampleProtocolError


def _module():
    package = Path(__file__).parents[2]
    path = package / "scripts/prepare_choice_c_v109_preflight.py"
    spec = importlib.util.spec_from_file_location("mosaickv_test_v109_materializer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v109_materializer_accepts_narrow_preflight_authorization() -> None:
    module = _module()
    protocol = module._protocol()
    module.validate_protocol(protocol)
    module.require_authorization(protocol)


def test_v109_materializer_rejects_weakened_fidelity_gate() -> None:
    module = _module()
    protocol = module._protocol()
    protocol["unchanged_memory_and_fidelity_gates"]["maximum_forward_kl"] = 0.03
    with pytest.raises(SampleProtocolError, match="changed"):
        module.validate_protocol(protocol)


def test_v109_protocol_is_valid_yaml() -> None:
    package = Path(__file__).parents[2]
    value = yaml.safe_load(
        (
            package
            / "configs/experiments/choice_c_v109_multi_feasible_ladder_preflight.yaml"
        ).read_text(encoding="utf-8")
    )
    assert value["experiment_id"] == "choice_c_v109_multi_feasible_ladder_preflight"
