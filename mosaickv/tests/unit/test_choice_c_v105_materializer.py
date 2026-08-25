from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import yaml


def _materializer(package: Path) -> ModuleType:
    scripts = package / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        path = scripts / "prepare_choice_c_v105_preflight.py"
        spec = importlib.util.spec_from_file_location("mosaickv_test_v105_materializer", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


def test_v105_authorization_is_narrow_and_keeps_downstream_blocked() -> None:
    package = Path(__file__).parents[2]
    materializer = _materializer(package)
    protocol = yaml.safe_load(
        (package / "configs/experiments/choice_c_v105_reserved_beam_preflight.yaml").read_text(
            encoding="utf-8"
        )
    )

    materializer.require_authorization(protocol)
    assert all(value is False for value in protocol["downstream_actions"].values())
    assert protocol["corrective_retry"]["predecessor_gpu_execution_started"] is False
    assert protocol["corrective_retry"]["candidate_algorithm_or_threshold_changed"] is False
    assert materializer.OUTPUT_ROOT.name.endswith("preflight-v2")


def test_v105_launcher_is_two_one_a100_workers_and_validates_inputs() -> None:
    package = Path(__file__).parents[2]
    launcher = (package / "slurm/choice_c_v105_preflight_a100.sbatch").read_text(
        encoding="utf-8"
    )

    assert "#SBATCH --array=0-1%2" in launcher
    assert "#SBATCH --gres=gpu:a100:1" in launcher
    assert "choice_c_v105_reserved_beam_preflight" in launcher
    assert "prepare_choice_c_v105_preflight.py --validate-only" in launcher
    assert "choice-c-v105-reserved-beam-preflight-v2" in launcher
