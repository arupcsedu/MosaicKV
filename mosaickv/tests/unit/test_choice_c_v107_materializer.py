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
        path = scripts / "prepare_choice_c_v107_preflight.py"
        spec = importlib.util.spec_from_file_location("mosaickv_test_v107_materializer", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


def test_v107_materializer_validates_draft_but_fails_closed_on_execution() -> None:
    package = Path(__file__).parents[2]
    materializer = _materializer(package)
    protocol = yaml.safe_load(
        (
            package / "configs/experiments/choice_c_v107_feasible_route_preflight.yaml"
        ).read_text(encoding="utf-8")
    )

    materializer.validate_protocol(protocol)
    with pytest.raises(PermissionError, match="not authorized"):
        materializer.require_authorization(protocol)


def test_v107_worker_spec_is_preflight_only_and_uses_new_candidate() -> None:
    package = Path(__file__).parents[2]
    materializer = _materializer(package)
    source = (package / "scripts/prepare_choice_c_v107_preflight.py").read_text(
        encoding="utf-8"
    )

    assert "v107_position_feasible_route_baseline_logit_feedback_restore_donor" in source
    assert "calibration or held-out work is authorized" in source
    assert materializer.OUTPUT_ROOT.name.endswith("preflight-v1")
