from __future__ import annotations

import hashlib
import importlib.metadata
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
    assert common["flash-attn"] == "2.8.3"
    assert "flash-attn-4" not in common
    assert common["flashinfer-python"] == "0.2.14.post1"


def test_flash_attention_source_build_is_versioned_and_specialized() -> None:
    project_root = _project_root()
    patch_path = (
        project_root / "env" / "patches" / "flash-attn-2.8.3-a100-multimodal-inference.patch"
    )
    assert hashlib.sha256(patch_path.read_bytes()).hexdigest() == (
        "e919aa142eba57abd98af403de282ee7ecf622ad22d745da22355a935b654768"
    )
    patch_text = patch_path.read_text(encoding="utf-8")
    assert "FLASHATTENTION_DISABLE_BACKWARD" in patch_text
    assert (
        '+                "cxx": compiler_c17_flag + '
        '["-DFLASHATTENTION_DISABLE_BACKWARD"]' in patch_text
    )
    assert "MosaicKV inference wheel supports head dimensions <=96 or exactly 128" in patch_text
    assert '"flash_bwd" not in source' not in patch_text
    assert "FLASHATTENTION_DISABLE_UNEVEN_K" not in patch_text

    build_text = (project_root / "scripts" / "build_flash_attn_wheel.sh").read_text(
        encoding="utf-8"
    )
    assert str(patch_path.name) in build_text
    assert "TORCH_CUDA_ARCH_LIST" in build_text
    assert "FLASH_ATTN_CUDA_ARCHS" in build_text
    assert "sha256sum" in build_text
    assert "build-source.XXXXXX" in build_text
    assert "1e71dd64a9e0280e0447b8a0c2541bad4bf6ac65bdeaa2f90e51a9e57de0370d" in build_text


def test_backend_stacks_share_one_declared_intersection() -> None:
    env_root = _project_root() / "env"
    common = _pins(env_root / "common" / "requirements.lock")
    assert (common["torch"], common["transformers"]) == ("2.8.0", "4.55.2")
    assert (common["vllm"], common["sglang"]) == ("0.10.2", "0.5.1.post3")


def test_environment_installer_rejects_retired_profile_names() -> None:
    project_root = _project_root()
    installer = project_root / "scripts" / "create_envs.sh"
    for profile in ("all", "hf", "vllm", "sglang", "mock"):
        completed = subprocess.run(
            [str(installer), profile],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 2
        assert "[--sync] common" in completed.stderr
        assert "alias" not in completed.stderr.lower()


def test_all_container_entry_points_use_the_unpatched_common_environment() -> None:
    project_root = _project_root()
    setup_text = (project_root / "scripts" / "create_envs.sh").read_text(encoding="utf-8")
    assert "apply_env_patches.sh" not in setup_text
    for name in ("Dockerfile", "Dockerfile.hf", "Dockerfile.vllm", "Dockerfile.sglang"):
        dockerfile = (project_root / name).read_text(encoding="utf-8")
        assert "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel" in dockerfile
        assert "scripts/apply_env_patches.sh" not in dockerfile
        assert "MOSAICKV_ENV_DIR=/opt/mosaickv-venv" in dockerfile


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
    mock_lock = tmp_path / "mock.requirements.lock"
    mock_lock.write_text(
        "\n".join(
            f"{name}=={importlib.metadata.version(name)}"
            for name in ("mosaickv", "numpy", "pytest")
        )
        + "\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "verify_envs.py"),
            "--environment",
            "mock",
            "--lock",
            str(mock_lock),
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
