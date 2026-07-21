#!/usr/bin/env python3
"""Validate measured vLLM trace structure without inventing missing data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mosaickv.backends.vllm_runtime import native_integration_capability


def _validate_trace(
    payload: dict[str, Any], *, require_gpu_measurements: bool = False
) -> dict[str, object]:
    if payload.get("measurement_type") != "vllm_fullkv":
        raise ValueError("trace is not labeled vllm_fullkv")
    if payload.get("backend") != "vllm" or payload.get("method") != "full_kv":
        raise ValueError("trace backend/method must be vllm/full_kv")
    if payload.get("native_mosaickv") is not False:
        raise ValueError("Stage A trace must not claim native MosaicKV")
    trials = payload.get("trials")
    if not isinstance(trials, list) or not trials:
        raise ValueError("trace must contain at least one raw trial")
    token_ids: tuple[int, ...] | None = None
    for index, trial in enumerate(trials):
        if not isinstance(trial, dict):
            raise ValueError(f"trial {index} must be an object")
        ids = tuple(int(value) for value in trial["token_ids"])
        if token_ids is None:
            token_ids = ids
        elif ids != token_ids:
            raise ValueError(f"trial {index} token IDs differ from trial 0")
        timestamps = trial["token_timestamps_seconds"]
        itls = trial["inter_token_latencies_seconds"]
        if len(timestamps) != len(ids) or len(itls) != max(0, len(ids) - 1):
            raise ValueError(f"trial {index} token timing cardinality is inconsistent")
        for field in (
            "ttft_seconds",
            "request_latency_seconds",
            "throughput_tokens_per_second",
        ):
            value = trial[field]
            if value is not None and float(value) < 0:
                raise ValueError(f"trial {index} has negative {field}")
        for field in ("prefix_cache_hit_rate", "mm_cache_hit_rate"):
            value = trial[field]
            if value is not None and not 0 <= float(value) <= 1:
                raise ValueError(f"trial {index} has invalid {field}")
        if require_gpu_measurements:
            required_fields = (
                "ttft_seconds",
                "request_latency_seconds",
                "throughput_tokens_per_second",
                "engine_prefill_seconds",
                "engine_decode_seconds",
                "engine_ttft_seconds",
                "gpu_memory_baseline_bytes",
                "gpu_memory_peak_bytes",
                "num_cached_tokens",
                "prefix_cache_hit_rate",
                "mm_cache_queries",
                "mm_cache_hits",
                "mm_cache_hit_rate",
            )
            missing = [field for field in required_fields if trial.get(field) is None]
            if missing:
                raise ValueError(f"trial {index} is missing required GPU measurements: {missing}")
    cache_measurement = payload.get("cache_measurement")
    if not isinstance(cache_measurement, dict) or set(cache_measurement) != {
        "prefix_cache",
        "multimodal_preprocessor_cache",
        "encoder_output_cache",
    }:
        raise ValueError("trace must document all three cache-observation boundaries")
    if require_gpu_measurements and len(trials) < 2:
        raise ValueError("GPU cache validation requires at least two identical trials")
    return {
        "status": "validated",
        "trials": len(trials),
        "generated_tokens": len(token_ids or ()),
        "deterministic_token_ids": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--vllm-version", default="0.7.2")
    parser.add_argument("--require-gpu-measurements", action="store_true")
    args = parser.parse_args()
    payload: dict[str, object] = {
        "native_capability": native_integration_capability(args.vllm_version).to_json_object()
    }
    if args.trace is not None:
        trace = json.loads(args.trace.read_text(encoding="utf-8"))
        if not isinstance(trace, dict):
            raise ValueError("trace root must be an object")
        payload["stage_a"] = _validate_trace(
            trace, require_gpu_measurements=bool(args.require_gpu_measurements)
        )
    print(json.dumps(payload, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
