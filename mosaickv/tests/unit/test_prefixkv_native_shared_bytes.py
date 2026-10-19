from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _load(name: str):
    scripts = ROOT / "mosaickv/scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(name, scripts / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_prefix_native_execution_modules_are_import_safe():
    _load("promote_prefixkv_native_validation")
    _load("run_prefixkv_native_shared_bytes")
    _load("analyze_prefixkv_native_shared_bytes")


def test_prefix_native_slurm_enforces_registered_controls():
    text = (ROOT / "mosaickv/slurm/prefixkv_native_shared_bytes.sbatch").read_text()
    assert "--cpus-per-task=12" in text
    assert "--gres=gpu:a100:1" in text
    assert "--time=02:00:00" in text
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in text
    assert text.index("CUBLAS_WORKSPACE_CONFIG=:4096:8") < text.index(
        "run_prefixkv_native_shared_bytes.py"
    )
