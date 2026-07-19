from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

PIN_PATTERN = re.compile(r"^([A-Za-z0-9_.-]+)==([^;\s]+)$")


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
    for profile in ("hf", "vllm", "sglang"):
        direct = _pins(env_root / profile / "requirements.in")
        locked = _pins(env_root / profile / "requirements.lock")
        assert direct
        assert locked
        assert all(locked.get(name) == version for name, version in direct.items())


def test_attention_packages_are_isolated() -> None:
    env_root = _project_root() / "env"
    hf = _pins(env_root / "hf" / "requirements.lock")
    vllm = _pins(env_root / "vllm" / "requirements.lock")
    sglang = _pins(env_root / "sglang" / "requirements.lock")
    assert hf["flash-attn"] == "2.8.3.post1"
    assert "flash-attn" not in vllm
    assert "flash-attn" not in sglang
    assert sglang["flash-attn-4"] == "4.0.0b22"


def test_backend_stacks_remain_separate() -> None:
    env_root = _project_root() / "env"
    hf = _pins(env_root / "hf" / "requirements.lock")
    vllm = _pins(env_root / "vllm" / "requirements.lock")
    sglang = _pins(env_root / "sglang" / "requirements.lock")
    assert (hf["torch"], hf["transformers"]) == ("2.11.0", "4.57.6")
    assert (vllm["torch"], vllm["transformers"]) == ("2.9.0", "4.57.6")
    assert (sglang["torch"], sglang["transformers"]) == ("2.9.1", "5.3.0")


def test_mock_verifier_passes_without_cuda(tmp_path: Path) -> None:
    project_root = _project_root()
    cache_root = tmp_path / "cache"
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "XDG_CACHE_HOME": str(cache_root / "xdg"),
            "HF_HOME": str(cache_root / "huggingface"),
            "HF_HUB_CACHE": str(cache_root / "huggingface" / "hub"),
            "HF_DATASETS_CACHE": str(cache_root / "datasets"),
            "TRANSFORMERS_CACHE": str(cache_root / "transformers"),
            "TORCH_HOME": str(cache_root / "torch"),
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
