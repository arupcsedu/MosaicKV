#!/usr/bin/env python3
"""Verify paired LLaVA-1.5 HF/vLLM runs for the selected paper tasks."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

from mosaickv.backends.vllm_verification import TASK_DATASETS, verify_campaign


def _write_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary_name = handle.name
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", action="append", choices=sorted(TASK_DATASETS))
    parser.add_argument("--expected-samples", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    tasks = tuple(args.task or TASK_DATASETS)
    payload = verify_campaign(args.root, tasks=tasks, expected_samples=args.expected_samples)
    output = args.output or args.root / (
        "verification.json" if len(tasks) > 1 else f"verification-{tasks[0]}.json"
    )
    _write_atomic(output.resolve(), payload)
    print(json.dumps(payload, sort_keys=True, indent=2))
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
