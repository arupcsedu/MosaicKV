#!/usr/bin/env python3
"""Verify exact environment pins, imports, cache policy, and CUDA execution."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

LOCK_PATTERN = re.compile(r"^([A-Za-z0-9_.-]+)==([^;\s]+)$")
CACHE_VARIABLES = (
    "PIP_CACHE_DIR",
    "UV_CACHE_DIR",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_ASSETS_CACHE",
    "HF_DATASETS_CACHE",
    "TRANSFORMERS_CACHE",
    "XDG_CACHE_HOME",
    "TORCH_HOME",
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "NUMBA_CACHE_DIR",
    "CUDA_CACHE_PATH",
    "FLASHINFER_WORKSPACE_BASE",
    "VLLM_CACHE_ROOT",
    "SGLANG_CACHE_DIR",
    "PRE_COMMIT_HOME",
    "MPLCONFIGDIR",
    "WANDB_CACHE_DIR",
    "WANDB_DATA_DIR",
    "RAY_TMPDIR",
    "TMPDIR",
)
PROFILE_IMPORTS = {
    "common": (
        "mosaickv",
        "numpy",
        "torch",
        "torchvision",
        "transformers",
        "accelerate",
        "datasets",
        "lmms_eval",
        "qwen_vl_utils",
        "av",
        "decord",
        "cv2",
        "PIL",
        "sentencepiece",
        "safetensors",
        "tokenizers",
        "pyarrow",
        "yaml",
    ),
    "mock": ("mosaickv", "numpy", "pytest"),
}
COMMON_GPU_IMPORTS = (
    "vllm",
    "vllm.entrypoints.llm",
    "sglang",
    "sglang.srt.entrypoints.engine",
    "flashinfer",
    "sgl_kernel",
    "xformers",
    "ray",
    "fastapi",
)
SGLANG_PATCH_RELATIVE_PATH = Path(
    "env/patches/sglang-0.4.3.post4-transformers-4.49.patch"
)
SGLANG_PATCH_TARGET = Path("sglang/srt/configs/qwen2_5_vl_config.py")
SGLANG_PATCH_MARKERS = (
    "Qwen2_5_VLConfig, None, Qwen2_5_VLImageProcessor, None, exist_ok=True",
    "Qwen2_5_VLConfig, Qwen2_5_VLProcessor, exist_ok=True",
)


def normalize_distribution(name: str) -> str:
    """Return the canonical comparison form used by package metadata."""

    return re.sub(r"[-_.]+", "-", name).lower()


def parse_lock(path: Path) -> dict[str, str]:
    """Parse an exact-pin lock and reject ranges, duplicates, and malformed rows."""

    pins: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        match = LOCK_PATTERN.fullmatch(line)
        if match is None:
            raise ValueError(f"{path}:{line_number}: expected an exact name==version pin")
        name = normalize_distribution(match.group(1))
        if name in pins:
            raise ValueError(f"{path}:{line_number}: duplicate distribution {name}")
        pins[name] = match.group(2)
    if not pins:
        raise ValueError(f"{path}: lock contains no package pins")
    return pins


def verify_pins(pins: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    """Verify every locked distribution is installed at exactly the locked version."""

    installed: dict[str, str] = {}
    errors: list[str] = []
    for name, expected in sorted(pins.items()):
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"missing distribution: {name}=={expected}")
            continue
        installed[name] = actual
        if actual != expected:
            errors.append(f"version mismatch: {name} locked={expected} installed={actual}")
    return installed, errors


def verify_environment_patches(
    profile: str,
    project_root: Path,
) -> tuple[dict[str, Any], list[str]]:
    """Verify exact source state for the versioned environment compatibility patch."""

    if profile != "common":
        return {}, []
    errors: list[str] = []
    patch_path = project_root / SGLANG_PATCH_RELATIVE_PATH
    report: dict[str, Any] = {
        "sglang_transformers_compat": {
            "patch_path": str(patch_path),
            "status": "failed",
        }
    }
    patch_report = report["sglang_transformers_compat"]
    assert isinstance(patch_report, dict)
    if not patch_path.is_file():
        errors.append(f"environment patch is missing: {patch_path}")
        return report, errors
    patch_report["patch_sha256"] = hashlib.sha256(patch_path.read_bytes()).hexdigest()
    try:
        distribution = importlib.metadata.distribution("sglang")
    except importlib.metadata.PackageNotFoundError:
        errors.append("cannot verify environment patch: sglang is not installed")
        return report, errors
    target = Path(distribution.locate_file(SGLANG_PATCH_TARGET)).resolve()
    patch_report["target"] = str(target)
    if not target.is_file():
        errors.append(f"environment patch target is missing: {target}")
        return report, errors
    source = target.read_text(encoding="utf-8")
    missing = [marker for marker in SGLANG_PATCH_MARKERS if source.count(marker) != 1]
    if missing:
        errors.append(
            "environment patch is not applied exactly once: "
            + ", ".join(repr(marker) for marker in missing)
        )
        return report, errors
    patch_report["target_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    patch_report["status"] = "applied"
    return report, errors


def verify_native_library_paths(profile: str) -> tuple[dict[str, Any], list[str]]:
    """Verify that wheel-provided native CUDA libraries are loader-visible."""

    if profile != "common":
        return {}, []
    expected = (
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "nvidia"
        / "cuda_nvrtc"
        / "lib"
    ).resolve()
    loader_paths = [
        Path(item).resolve(strict=False)
        for item in os.environ.get("LD_LIBRARY_PATH", "").split(":")
        if item
    ]
    report: dict[str, Any] = {
        "expected_nvrtc_directory": str(expected),
        "ld_library_path": [str(path) for path in loader_paths],
        "libnvrtc_present": (expected / "libnvrtc.so.12").is_file(),
        "loader_visible": expected in loader_paths,
    }
    errors: list[str] = []
    if not report["libnvrtc_present"]:
        errors.append(f"locked NVRTC library is missing: {expected / 'libnvrtc.so.12'}")
    if not report["loader_visible"]:
        errors.append(f"locked NVRTC directory is absent from LD_LIBRARY_PATH: {expected}")
    return report, errors


def _import_in_bounded_subprocess(
    module_name: str,
    timeout_seconds: float,
) -> tuple[bool, str, float]:
    """Import one module in an isolated process with a hard wall-clock deadline."""

    command = [
        sys.executable,
        "-c",
        "import importlib, sys; importlib.import_module(sys.argv[1])",
        module_name,
    ]
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        elapsed = time.monotonic() - started
        return False, f"timed out after {timeout_seconds:g}s", elapsed

    elapsed = time.monotonic() - started
    if process.returncode == 0:
        return True, "", elapsed
    detail = stderr.strip() or stdout.strip() or f"exit status {process.returncode}"
    return False, detail, elapsed


def verify_imports(
    profile: str,
    require_cuda: bool,
    timeout_seconds: float,
) -> tuple[list[str], list[str], dict[str, float]]:
    """Import the complete smoke surface with an independent deadline per module."""

    imported: list[str] = []
    errors: list[str] = []
    durations: dict[str, float] = {}
    module_names = list(PROFILE_IMPORTS[profile])
    if profile == "common" and require_cuda:
        module_names.extend(COMMON_GPU_IMPORTS)
    for module_name in module_names:
        print(f"verifying_import={module_name}", file=sys.stderr, flush=True)
        succeeded, detail, elapsed = _import_in_bounded_subprocess(
            module_name,
            timeout_seconds,
        )
        durations[module_name] = round(elapsed, 6)
        if succeeded:
            imported.append(module_name)
        else:
            errors.append(f"import failed: {module_name}: {detail}")
    return imported, errors, durations


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def verify_cache_policy(profile: str) -> tuple[dict[str, str], list[str]]:
    """Require all model/runtime caches to be absolute and outside the home tree."""

    home = Path.home().resolve()
    names = list(CACHE_VARIABLES)
    values: dict[str, str] = {}
    errors: list[str] = []
    for name in names:
        value = os.environ.get(name)
        if not value:
            errors.append(f"cache variable is unset: {name}")
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            errors.append(f"cache path is not absolute: {name}={value}")
            continue
        resolved = path.resolve(strict=False)
        values[name] = str(resolved)
        if is_within(resolved, home):
            errors.append(f"cache path is inside home: {name}={resolved}")
    if os.environ.get("PIP_CONFIG_FILE") != "/dev/null":
        errors.append("PIP_CONFIG_FILE must be /dev/null to ignore user-level pip config")
    return values, errors


def nvidia_driver_version() -> str:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unavailable"
    if completed.returncode != 0:
        return "unavailable"
    versions = sorted({line.strip() for line in completed.stdout.splitlines() if line.strip()})
    return ",".join(versions) if versions else "unavailable"


def driver_major(version: str) -> int | None:
    match = re.match(r"^(\d+)", version)
    return int(match.group(1)) if match else None


def cuda_smoke(profile: str, require_cuda: bool) -> tuple[dict[str, Any], list[str]]:
    """Run a deterministic 2x2 CUDA matrix multiplication when torch is available."""

    if profile == "mock":
        return {"required": False, "available": False, "matmul_passed": False}, []

    import torch

    available = bool(torch.cuda.is_available())
    report: dict[str, Any] = {
        "required": require_cuda,
        "available": available,
        "build_cuda": torch.version.cuda or "not_used",
        "driver": nvidia_driver_version(),
        "matmul_passed": False,
    }
    errors: list[str] = []
    if not available:
        if require_cuda:
            errors.append("CUDA is required but torch.cuda.is_available() is false")
        return report, errors

    device = torch.device("cuda:0")
    capability = torch.cuda.get_device_capability(device)
    report.update(
        {
            "device_name": torch.cuda.get_device_name(device),
            "compute_capability": list(capability),
            "device_count": torch.cuda.device_count(),
        }
    )
    left = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
    right = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device=device)
    actual = left @ right
    expected = torch.tensor([[19.0, 22.0], [43.0, 50.0]], device=device)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    torch.cuda.synchronize(device)
    report["matmul_passed"] = True

    minimum_driver = 525
    actual_driver = driver_major(str(report["driver"]))
    if profile == "common" and str(torch.version.cuda) != "12.4":
        errors.append(f"common lock requires a CUDA 12.4 torch build, got {torch.version.cuda}")
    if actual_driver is None or actual_driver < minimum_driver:
        errors.append(
            f"{profile} requires NVIDIA driver major >= {minimum_driver}, got {report['driver']}"
        )
    if profile == "common" and capability < (8, 0):
        errors.append(f"common backend lock requires compute capability >= 8.0, got SM {capability}")
    return report, errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", required=True, choices=sorted(PROFILE_IMPORTS))
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument(
        "--import-timeout-seconds",
        type=float,
        default=120.0,
        help="hard wall-clock deadline for each isolated module import (default: 120)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    profile = str(args.environment)
    if args.import_timeout_seconds <= 0:
        print(json.dumps({"status": "error", "error": "import timeout must be positive"}))
        return 2
    if profile == "mock" and args.require_cuda:
        print(json.dumps({"status": "error", "error": "mock profile cannot require CUDA"}))
        return 2

    project_root = Path(__file__).resolve().parents[1]
    lock_path = args.lock or project_root / "env" / profile / "requirements.lock"
    errors: list[str] = []
    try:
        pins = parse_lock(lock_path)
    except (OSError, ValueError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, sort_keys=True))
        return 2

    installed, pin_errors = verify_pins(pins)
    environment_patches, patch_errors = verify_environment_patches(profile, project_root)
    native_libraries, native_library_errors = verify_native_library_paths(profile)
    imported, import_errors, import_durations = verify_imports(
        profile,
        bool(args.require_cuda),
        float(args.import_timeout_seconds),
    )
    cache_paths, cache_errors = verify_cache_policy(profile)
    try:
        cuda, cuda_errors = cuda_smoke(profile, bool(args.require_cuda))
    except BaseException as error:
        cuda = {"required": bool(args.require_cuda), "matmul_passed": False}
        cuda_errors = [f"CUDA smoke failed: {type(error).__name__}: {error}"]
    errors.extend(pin_errors)
    errors.extend(patch_errors)
    errors.extend(native_library_errors)
    errors.extend(import_errors)
    errors.extend(cache_errors)
    errors.extend(cuda_errors)

    runtime_profile = profile != "mock"
    support_verified = (
        runtime_profile
        and bool(args.require_cuda)
        and not errors
        and bool(cuda.get("matmul_passed", False))
    )
    mock_verified = profile == "mock" and not errors
    if errors:
        status = "failed"
    elif support_verified:
        status = "support_verified"
    elif mock_verified:
        status = "mock_verified"
    else:
        status = "imports_only"

    payload = {
        "schema_version": 1,
        "status": status,
        "environment": profile,
        "support_verified": support_verified,
        "mock_verified": mock_verified,
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "lock_path": str(lock_path.resolve()),
        "lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        "locked_distribution_count": len(pins),
        "verified_distribution_count": len(installed),
        "imported_modules": imported,
        "import_durations_seconds": import_durations,
        "import_timeout_seconds": float(args.import_timeout_seconds),
        "environment_patches": environment_patches,
        "native_libraries": native_libraries,
        "cache_paths": cache_paths,
        "hf_token_present": bool(os.environ.get("HF_TOKEN")),
        "cuda": cuda,
        "errors": errors,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
