from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

PIN_PATTERN = re.compile(r"^([A-Za-z0-9_.-]+)(?:\[[A-Za-z0-9_,.-]+\])?==([^;\s]+)$")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _pins(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", "--")):
            continue
        match = PIN_PATTERN.fullmatch(line)
        assert match is not None, f"non-exact requirement in {path}: {line}"
        name = re.sub(r"[-_.]+", "-", match.group(1)).lower()
        assert name not in pins, f"duplicate requirement in {path}: {name}"
        pins[name] = match.group(2)
    return pins


def test_direct_requirements_are_exactly_present_in_locks() -> None:
    env_root = _project_root() / "env"
    for profile in ("common",):
        direct = _pins(env_root / profile / "requirements.in")
        locked = _pins(env_root / profile / "requirements.lock")
        assert direct
        assert locked
        assert all(locked.get(name) == version for name, version in direct.items())


def test_common_attention_dependencies_are_explicit() -> None:
    env_root = _project_root() / "env"
    common = _pins(env_root / "common" / "requirements.lock")
    assert "flash-attn" not in common
    assert "flash-attn-4" not in common
    assert common["flashinfer-python"] == "0.2.3"


def test_backend_stacks_share_one_declared_intersection() -> None:
    env_root = _project_root() / "env"
    common = _pins(env_root / "common" / "requirements.lock")
    assert (common["torch"], common["transformers"]) == ("2.5.1", "4.49.0")
    assert (common["vllm"], common["sglang"]) == ("0.7.2", "0.4.3.post1")


def test_mock_verifier_passes_without_cuda(tmp_path: Path) -> None:
    project_root = _project_root()
    cache_root = tmp_path / "cache"
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PIP_CONFIG_FILE": "/dev/null",
            "PIP_CACHE_DIR": str(cache_root / "pip"),
            "UV_CACHE_DIR": str(cache_root / "uv"),
            "XDG_CACHE_HOME": str(cache_root / "xdg"),
            "HF_HOME": str(cache_root / "huggingface"),
            "HF_HUB_CACHE": str(cache_root / "huggingface" / "hub"),
            "HF_ASSETS_CACHE": str(cache_root / "huggingface" / "assets"),
            "HF_DATASETS_CACHE": str(cache_root / "datasets"),
            "TRANSFORMERS_CACHE": str(cache_root / "transformers"),
            "TORCH_HOME": str(cache_root / "torch"),
            "TORCHINDUCTOR_CACHE_DIR": str(cache_root / "torchinductor"),
            "TRITON_CACHE_DIR": str(cache_root / "triton"),
            "NUMBA_CACHE_DIR": str(cache_root / "numba"),
            "CUDA_CACHE_PATH": str(cache_root / "cuda"),
            "FLASHINFER_WORKSPACE_BASE": str(cache_root / "flashinfer"),
            "VLLM_CACHE_ROOT": str(cache_root / "vllm"),
            "SGLANG_CACHE_DIR": str(cache_root / "sglang"),
            "PRE_COMMIT_HOME": str(cache_root / "pre-commit"),
            "MPLCONFIGDIR": str(cache_root / "matplotlib"),
            "WANDB_CACHE_DIR": str(cache_root / "wandb" / "cache"),
            "WANDB_DATA_DIR": str(cache_root / "wandb" / "data"),
            "RAY_TMPDIR": str(cache_root / "ray"),
            "TMPDIR": str(cache_root / "tmp"),
        }
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "verify_envs.py"),
            "--environment",
            "mock",
            "--lock",
            str(project_root / "env" / "mock" / "requirements.lock"),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(completed.stdout)
    assert report["status"] == "mock_verified"
    assert report["mock_verified"] is True
    assert report["support_verified"] is False
    assert report["cuda"]["matmul_passed"] is False
