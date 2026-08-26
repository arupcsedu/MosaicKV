from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml


def _materializer(package: Path) -> ModuleType:
    scripts = package / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        path = scripts / "prepare_choice_c_v108_preflight.py"
        spec = importlib.util.spec_from_file_location("mosaickv_test_v108_materializer", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


def test_v108_materializer_accepts_narrow_preflight_authorization() -> None:
    package = Path(__file__).parents[2]
    materializer = _materializer(package)
    protocol = yaml.safe_load(
        (
            package / "configs/experiments/choice_c_v108_storage_aligned_preflight.yaml"
        ).read_text(encoding="utf-8")
    )

    materializer.validate_protocol(protocol)
    materializer.require_authorization(protocol)


def test_v108_materializer_rejects_calibration_authority() -> None:
    package = Path(__file__).parents[2]
    materializer = _materializer(package)
    protocol = yaml.safe_load(
        (
            package / "configs/experiments/choice_c_v108_storage_aligned_preflight.yaml"
        ).read_text(encoding="utf-8")
    )
    protocol["downstream_actions"]["calibration_execution_authorized"] = True

    with pytest.raises(materializer.SampleProtocolError, match="calibration or held-out"):
        materializer.require_authorization(protocol)


def test_v108_worker_spec_requires_storage_accounting_trace_fields() -> None:
    package = Path(__file__).parents[2]
    materializer = _materializer(package)
    source = (package / "scripts/prepare_choice_c_v108_preflight.py").read_text(
        encoding="utf-8"
    )

    assert "v108_storage_aligned_feasible_route_logit_feedback" in source
    assert "accounting_trace_must_record" in source
    assert materializer.OUTPUT_ROOT.name.endswith("preflight-v1")
