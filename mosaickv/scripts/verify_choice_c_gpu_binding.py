#!/usr/bin/env python3
"""Fail fast unless a two-rank Slurm step has one distinct GPU per rank."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise RuntimeError(f"{name} is required for the GPU-binding preflight")
    return value


def _visible_gpu_uuid() -> str:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"nvidia-smi failed during binding preflight: {detail}")
    uuids = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(uuids) != 1:
        raise RuntimeError(
            "each Slurm rank must see exactly one GPU; "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not_set')!r}, "
            f"visible_uuids={uuids}"
        )
    return uuids[0]


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--round", required=True, type=int)
    parser.add_argument("--wave", required=True, type=int)
    parser.add_argument("--rank", required=True, type=int)
    parser.add_argument("--timeout-seconds", default=45.0, type=float)
    args = parser.parse_args()
    if args.rank not in {0, 1}:
        raise ValueError("binding preflight rank must be 0 or 1")
    if args.timeout_seconds <= 0 or args.timeout_seconds > 60:
        raise ValueError("timeout-seconds must lie in (0, 60]")

    job_id = _required_environment("SLURM_JOB_ID")
    step_id = _required_environment("SLURM_STEP_ID")
    local_id = _required_environment("SLURM_LOCALID")
    proc_id = _required_environment("SLURM_PROCID")
    if int(proc_id) != args.rank or int(local_id) != args.rank:
        raise RuntimeError(
            f"unexpected Slurm rank assignment: proc_id={proc_id}, "
            f"local_id={local_id}, expected={args.rank}"
        )

    binding_root = (
        args.output_root.resolve()
        / "gpu_binding"
        / f"job-{job_id}"
        / f"round-{args.round}-wave-{args.wave}-step-{step_id}"
    )
    binding_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "job_id": job_id,
        "step_id": step_id,
        "round": args.round,
        "wave": args.wave,
        "rank": args.rank,
        "local_id": int(local_id),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
        "step_gpus": os.environ.get("SLURM_STEP_GPUS", "not_set"),
        "visible_gpu_uuid": _visible_gpu_uuid(),
    }
    path = binding_root / f"rank-{args.rank}.json"
    if path.exists():
        raise RuntimeError(f"refusing to overwrite binding record: {path}")
    _atomic_write(path, payload)

    paths = [binding_root / "rank-0.json", binding_root / "rank-1.json"]
    deadline = time.monotonic() + args.timeout_seconds
    while not all(item.exists() for item in paths):
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for peer binding record in {binding_root}")
        time.sleep(0.1)
    records = [json.loads(item.read_text()) for item in paths]
    uuids = {str(item["visible_gpu_uuid"]) for item in records}
    if len(uuids) != 2:
        raise RuntimeError(
            "GPU isolation failed: both Slurm ranks resolved to the same physical GPU; "
            f"records={records}"
        )
    if any(
        item["job_id"] != job_id
        or item["step_id"] != step_id
        or item["round"] != args.round
        or item["wave"] != args.wave
        for item in records
    ):
        raise RuntimeError(f"binding records do not describe the same Slurm step: {records}")
    print(
        f"gpu-binding preflight passed round={args.round} wave={args.wave} "
        f"rank={args.rank} uuid={payload['visible_gpu_uuid']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
