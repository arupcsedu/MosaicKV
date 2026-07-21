"""Read-only NVIDIA telemetry and process-concurrency capture."""

from __future__ import annotations

import csv
import io
import os
import subprocess
from datetime import UTC, datetime
from typing import Any

from mosaickv.measurements.types import (
    GpuDeviceState,
    GpuEnvironmentSnapshot,
    GpuProcessState,
)


def _run_nvidia_smi(arguments: list[str]) -> tuple[str, str | None]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return "", f"{type(error).__name__}: {error}"
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        return "", f"nvidia-smi exited {completed.returncode}: {detail}"
    return completed.stdout, None


def _visible_identifiers() -> frozenset[str] | None:
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if value is None or not value.strip():
        return None
    return frozenset(item.strip() for item in value.split(",") if item.strip())


def capture_gpu_environment(torch_module: Any) -> GpuEnvironmentSnapshot:
    """Capture GPU/device/process state without changing clocks or power."""

    device_output, device_error = _run_nvidia_smi(
        [
            "--query-gpu=index,uuid,name,pstate,clocks.current.graphics,clocks.current.sm,"
            "clocks.current.memory,power.draw,power.limit,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    visible = _visible_identifiers()
    devices: list[GpuDeviceState] = []
    for row in csv.reader(io.StringIO(device_output)):
        values = [item.strip() for item in row]
        if len(values) != 10:
            continue
        index, uuid, name, pstate, graphics, sm, memory, power, limit, driver = values
        is_visible = visible is None or index in visible or uuid in visible
        devices.append(
            GpuDeviceState(
                index=index,
                uuid=uuid,
                name=name,
                power_state=pstate,
                graphics_clock_mhz=graphics,
                sm_clock_mhz=sm,
                memory_clock_mhz=memory,
                power_draw_w=power,
                power_limit_w=limit,
                driver_version=driver,
                visible_to_process=is_visible,
            )
        )

    process_output, process_error = _run_nvidia_smi(
        [
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    current_pid = os.getpid()
    processes: list[GpuProcessState] = []
    for row in csv.reader(io.StringIO(process_output)):
        values = [item.strip() for item in row]
        if len(values) != 4:
            continue
        uuid, pid_text, name, memory = values
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        processes.append(
            GpuProcessState(
                gpu_uuid=uuid,
                pid=pid,
                process_name=name,
                used_gpu_memory_mib=memory,
                is_current_process=pid == current_pid,
            )
        )
    visible_uuids = {device.uuid for device in devices if device.visible_to_process}
    visible_processes = tuple(item for item in processes if item.gpu_uuid in visible_uuids)
    background = tuple(item for item in visible_processes if not item.is_current_process)
    errors = "; ".join(item for item in (device_error, process_error) if item is not None)
    return GpuEnvironmentSnapshot(
        captured_at_utc=datetime.now(UTC).isoformat(),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
        torch_version=str(getattr(torch_module, "__version__", "unknown")),
        cuda_runtime=str(getattr(torch_module.version, "cuda", None) or "not_used"),
        devices=tuple(devices),
        processes=tuple(processes),
        background_processes=background,
        visible_process_concurrency=len(visible_processes),
        query_error=errors or None,
    )


__all__ = ["capture_gpu_environment"]
