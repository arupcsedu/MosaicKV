"""Read-only host, CUDA, and optional-backend diagnostics."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Final, cast

from mosaickv import __version__
from mosaickv.types import JsonObject, JsonValue

_OPTIONAL_PACKAGES: Final[dict[str, tuple[str, tuple[str, ...]]]] = {
    "torch": ("torch", ("torch",)),
    "transformers": ("transformers", ("transformers",)),
    "accelerate": ("accelerate", ("accelerate",)),
    "flash_attn": ("flash_attn", ("flash-attn", "flash_attn")),
    "vllm": ("vllm", ("vllm",)),
    "sglang": ("sglang", ("sglang",)),
    "lmms_eval": ("lmms_eval", ("lmms-eval", "lmms_eval")),
    "datasets": ("datasets", ("datasets",)),
}


def _metadata_version(names: tuple[str, ...]) -> str | None:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _probe_import(module: str, timeout_seconds: int = 20) -> tuple[bool, str | None]:
    script = (
        "import importlib, json; "
        f"importlib.import_module({module!r}); "
        "print(json.dumps({'ok': True}))"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return False, f"import timed out after {timeout_seconds}s"
    if completed.returncode == 0:
        return True, None
    stderr = completed.stderr.strip().splitlines()
    detail = stderr[-1] if stderr else f"import exited with code {completed.returncode}"
    return False, detail


def probe_packages() -> JsonObject:
    """Report package metadata separately from actual importability."""

    packages: JsonObject = {}
    for label, (module, distributions) in _OPTIONAL_PACKAGES.items():
        version = _metadata_version(distributions)
        installed = version is not None
        importable = False
        error: str | None = None
        if installed:
            importable, error = _probe_import(module)
        record: JsonObject = {
            "installed": installed,
            "version": version or "not_installed",
            "importable": importable,
        }
        if error is not None:
            record["error"] = error
        packages[label] = record
    return packages


def _nvidia_smi_report() -> JsonObject:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return {"available": False, "driver_version": "not_used", "gpus": [], "error": str(error)}
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        return {
            "available": False,
            "driver_version": "not_used",
            "gpus": [],
            "error": detail or f"nvidia-smi exited with code {completed.returncode}",
        }

    gpus: list[JsonValue] = []
    drivers: set[str] = set()
    for index, line in enumerate(completed.stdout.splitlines()):
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        name, driver, memory_mib = parts
        drivers.add(driver)
        try:
            parsed_memory: JsonValue = int(memory_mib)
        except ValueError:
            parsed_memory = memory_mib
        gpus.append(
            {"index": index, "name": name, "driver_version": driver, "memory_mib": parsed_memory}
        )
    return {
        "available": bool(gpus),
        "driver_version": ",".join(sorted(drivers)) if drivers else "not_used",
        "gpus": gpus,
    }


def _torch_cuda_report(torch_importable: bool) -> JsonObject:
    if not torch_importable:
        return {
            "torch_importable": False,
            "available": False,
            "build_cuda": "not_used",
            "device_count": 0,
            "devices": [],
        }
    script = """
import json
import torch
devices = []
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        devices.append({
            "index": index,
            "name": props.name,
            "total_memory_bytes": props.total_memory,
            "capability": list(torch.cuda.get_device_capability(index)),
        })
print(json.dumps({
    "torch_importable": True,
    "available": torch.cuda.is_available(),
    "build_cuda": torch.version.cuda or "not_used",
    "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    "devices": devices,
}))
"""
    try:
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return {"torch_importable": True, "available": False, "error": "torch CUDA probe timed out"}
    if completed.returncode != 0:
        return {
            "torch_importable": True,
            "available": False,
            "error": completed.stderr.strip() or "torch CUDA probe failed",
        }
    try:
        return cast("JsonObject", json.loads(completed.stdout))
    except json.JSONDecodeError as error:
        return {"torch_importable": True, "available": False, "error": str(error)}


def doctor_report() -> JsonObject:
    """Collect a diagnostic report without loading checkpoints."""

    packages = probe_packages()
    torch_record = cast("JsonObject", packages["torch"])
    torch_cuda = _torch_cuda_report(bool(torch_record["importable"]))
    nvidia_smi = _nvidia_smi_report()
    nvidia_available = bool(nvidia_smi["available"])
    torch_available = bool(torch_cuda.get("available", False))
    gpu_count = len(cast("list[JsonValue]", nvidia_smi["gpus"]))
    if gpu_count == 0:
        torch_device_count = torch_cuda.get("device_count", 0)
        gpu_count = torch_device_count if isinstance(torch_device_count, int) else 0

    def package_importable(label: str) -> bool:
        record = cast("JsonObject", packages[label])
        return bool(record["importable"])

    return {
        "schema_version": 1,
        "status": "ok" if nvidia_available or torch_available else "cpu_only",
        "mosaickv_version": __version__,
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
            "isolated_environment": sys.prefix != sys.base_prefix,
        },
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "pid": os.getpid(),
            "cwd": str(Path.cwd()),
        },
        "packages": packages,
        "cuda": {
            "available": nvidia_available or torch_available,
            "gpu_count": gpu_count,
            "nvidia_smi": nvidia_smi,
            "torch": torch_cuda,
        },
        "backends": {
            "huggingface": {
                "available": package_importable("transformers"),
                "model_weights_loaded": False,
            },
            "vllm": {"available": package_importable("vllm"), "model_weights_loaded": False},
            "sglang": {
                "available": package_importable("sglang"),
                "model_weights_loaded": False,
            },
        },
    }


__all__ = ["doctor_report", "probe_packages"]
