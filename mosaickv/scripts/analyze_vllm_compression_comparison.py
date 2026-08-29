#!/usr/bin/env python3
"""Analyze equal-budget common-selector projections decoded by vLLM."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from mosaickv.types import JsonObject, JsonValue

TASKS = ("mmstar", "textvqa", "docvqa")
METHODS = ("full_kv", "random_kv", "prompt_attention_topk", "mosaickv_exact")


def _rows(path: Path) -> dict[str, dict[str, Any]]:
    files = sorted(path.glob("*/results.jsonl"))
    if len(files) != 1:
        raise ValueError(f"expected one results.jsonl below {path}, found {len(files)}")
    rows: dict[str, dict[str, Any]] = {}
    for line in files[0].read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        if payload.get("status") != "completed" or payload.get("error") is not None:
            raise ValueError(f"incomplete result row in {files[0]}")
        sample_id = str(payload["sample_id"])
        if sample_id in rows:
            raise ValueError(f"duplicate sample {sample_id}")
        rows[sample_id] = cast("dict[str, Any]", payload)
    return rows


def _trace_isolation(path: Path) -> bool:
    traces = sorted(path.glob("*/*.json"))
    if not traces:
        raise ValueError(f"missing traces below {path}")
    return any(
        json.loads(trace.read_text(encoding="utf-8"))
        .get("request_isolation", {})
        .get("token_ids_match_anchor")
        is True
        for trace in traces
    )


def _mean(rows: Mapping[str, Mapping[str, Any]], field: str) -> float:
    values = tuple(float(row[field]) for row in rows.values())
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError(f"field {field} is missing or non-finite")
    return statistics.mean(values)


def analyze(root: Path) -> JsonObject:
    records: list[JsonValue] = []
    for task in TASKS:
        paths = {
            "full_kv": root / task / "full_kv",
            **{
                method: root / task / method / "vllm"
                for method in METHODS
                if method != "full_kv"
            },
        }
        by_method = {method: _rows(path) for method, path in paths.items()}
        identities = set(by_method["full_kv"])
        if len(identities) < 2 or any(set(rows) != identities for rows in by_method.values()):
            raise ValueError(f"{task} methods do not contain the same samples")
        full = by_method["full_kv"]
        full_bytes = _mean(full, "active_kv_bytes")
        for method in METHODS:
            rows = by_method[method]
            active = _mean(rows, "active_kv_bytes")
            agreement = statistics.mean(
                rows[sample_id]["answer"] == full[sample_id]["answer"]
                for sample_id in sorted(identities)
            )
            score = _mean(rows, "task_score")
            full_score = _mean(full, "task_score")
            records.append(
                {
                    "task": task,
                    "method": method,
                    "samples": len(rows),
                    "task_score": score,
                    "score_delta_from_fullkv": score - full_score,
                    "answer_agreement_with_fullkv": agreement,
                    "active_kv_bytes": active,
                    "active_memory_reduction_vs_fullkv": 1.0 - active / full_bytes,
                    "compression_seconds": _mean(rows, "compression_time"),
                    "ttft_seconds": _mean(rows, "ttft"),
                    "decode_seconds": _mean(rows, "decode_time"),
                    "end_to_end_seconds": _mean(rows, "end_to_end_time"),
                    "request_isolation_verified": _trace_isolation(paths[method] / "traces"),
                }
            )
    return {
        "schema_version": 1,
        "classification": "exploratory_dirty_worktree_common_selector_projection",
        "model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "tasks": list(TASKS),
        "methods": list(METHODS),
        "records": records,
    }


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = analyze(args.root.resolve())
    _write(args.output.resolve(), json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "completed", "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
