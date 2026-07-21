#!/usr/bin/env python3
"""Validate measured SGLang traces without manufacturing missing observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from mosaickv.backends.sglang_runtime import native_integration_capability


def _object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"trace root must be an object: {path}")
    return payload


def _validate_trace(
    payload: dict[str, Any], *, require_gpu_measurements: bool = False
) -> dict[str, object]:
    if payload.get("measurement_type") != "sglang_fullkv":
        raise ValueError("trace is not labeled sglang_fullkv")
    if payload.get("backend") != "sglang" or payload.get("method") != "full_kv":
        raise ValueError("trace backend/method must be sglang/full_kv")
    if payload.get("native_mosaickv") is not False:
        raise ValueError("Stage A trace must not claim native MosaicKV")
    isolation = payload.get("request_isolation")
    if not isinstance(isolation, dict) or isolation.get("session_params") is not None:
        raise ValueError("trace must prove that no SGLang session was reused")
    engine = payload.get("engine")
    if not isinstance(engine, dict):
        raise ValueError("trace omitted engine metadata")
    command = engine.get("server_command")
    if not isinstance(command, list):
        raise ValueError("trace omitted exact server command")
    for required in (
        "--enable-deterministic-inference",
        "--disable-overlap-schedule",
        "--disable-cuda-graph",
        "--skip-server-warmup",
        "--disable-fast-image-processor",
    ):
        if required not in command:
            raise ValueError(f"correctness server command omitted {required}")
    geometry = engine.get("kv_cache_geometry")
    if not isinstance(geometry, dict):
        raise ValueError("trace omitted KV cache geometry")
    bytes_per_position = int(geometry["bytes_per_position"])
    trials = payload.get("trials")
    if not isinstance(trials, list) or not trials:
        raise ValueError("trace must contain at least one raw trial")
    token_ids: tuple[int, ...] | None = None
    request_ids: list[str] = []
    for index, trial in enumerate(trials):
        if not isinstance(trial, dict):
            raise ValueError(f"trial {index} must be an object")
        ids = tuple(int(value) for value in trial["token_ids"])
        if token_ids is None:
            token_ids = ids
        elif ids != token_ids:
            raise ValueError(f"trial {index} token IDs differ from trial 0")
        request_ids.append(str(trial["request_id"]))
        timestamps = trial["token_timestamps_seconds"]
        itls = trial["inter_token_latencies_seconds"]
        if len(timestamps) != len(ids) or len(itls) != max(0, len(ids) - 1):
            raise ValueError(f"trial {index} token timing cardinality is inconsistent")
        expected_positions = int(trial["prompt_tokens"]) + max(0, len(ids) - 1)
        if int(trial["active_cache_positions"]) != expected_positions:
            raise ValueError(f"trial {index} active cache position accounting is wrong")
        if int(trial["active_kv_bytes"]) != expected_positions * bytes_per_position:
            raise ValueError(f"trial {index} active KV byte accounting is wrong")
        if not 0 <= float(trial["prefix_cache_hit_rate"]) <= 1:
            raise ValueError(f"trial {index} prefix cache hit rate is invalid")
        for field in (
            "ttft_seconds",
            "request_latency_seconds",
            "decode_seconds",
            "throughput_tokens_per_second",
        ):
            value = trial.get(field)
            if value is not None and float(value) < 0:
                raise ValueError(f"trial {index} has negative {field}")
        if require_gpu_measurements:
            required_fields = (
                "ttft_seconds",
                "request_latency_seconds",
                "decode_seconds",
                "throughput_tokens_per_second",
                "cached_tokens",
                "prometheus_cache_hit_rate",
                "prometheus_generation_throughput",
                "prometheus_token_usage",
                "gpu_memory_baseline_bytes",
                "gpu_memory_peak_bytes",
            )
            missing = [field for field in required_fields if trial.get(field) is None]
            if missing:
                raise ValueError(f"trial {index} is missing GPU observations: {missing}")
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("request IDs are not unique within the trace")
    if require_gpu_measurements and len(trials) < 2:
        raise ValueError("GPU cache validation requires two identical trials")
    probe = isolation.get("post_intervening_request_probe")
    probe_request_id: str | None = None
    probe_verified = False
    if probe is not None:
        if not isinstance(probe, dict):
            raise ValueError("request-isolation probe must be an object")
        probe_request_id = str(probe.get("request_id"))
        probe_verified = (
            probe.get("performed") is True
            and probe.get("token_ids_match_anchor") is True
            and int(probe.get("intervening_distinct_input_fingerprints", 0)) >= 1
        )
        if not probe_verified:
            raise ValueError("request-isolation A-B-A probe did not pass")
    return {
        "sample_id": str(payload["sample_id"]),
        "trials": len(trials),
        "generated_tokens": len(token_ids or ()),
        "deterministic_token_ids": True,
        "active_kv_accounting": True,
        "request_ids": request_ids + ([probe_request_id] if probe_request_id else []),
        "isolation_probe_verified": probe_verified,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, action="append", default=[])
    parser.add_argument("--results-jsonl", type=Path)
    parser.add_argument("--sglang-version", default="0.5.10.post1")
    parser.add_argument("--require-gpu-measurements", action="store_true")
    args = parser.parse_args()
    validations = [
        _validate_trace(path_payload, require_gpu_measurements=args.require_gpu_measurements)
        for path_payload in (_object(path) for path in args.trace)
    ]
    sample_ids = [str(item["sample_id"]) for item in validations]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("multiple traces reuse a sample ID")
    all_request_ids = [
        str(request_id)
        for item in validations
        for request_id in cast("list[str]", item["request_ids"])
    ]
    if len(all_request_ids) != len(set(all_request_ids)):
        raise ValueError("request IDs are not globally unique across traces")
    isolation_rows = 0
    if args.results_jsonl is not None:
        result_rows = [
            json.loads(line)
            for line in args.results_jsonl.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        by_sample = {
            str(row["sample_id"]): row for row in result_rows if isinstance(row, dict)
        }
        if set(sample_ids) != set(by_sample):
            raise ValueError("trace and result sample IDs differ")
        for sample_id in sample_ids:
            row = by_sample[sample_id]
            if row.get("status") != "completed":
                raise ValueError(f"request-isolation sample failed: {sample_id}")
            isolation_rows += 1
        if len(sample_ids) >= 2 and not any(
            bool(item["isolation_probe_verified"]) for item in validations
        ):
            raise ValueError("multi-input run omitted the request-isolation A-B-A probe")
    payload = {
        "native_capability": native_integration_capability(
            args.sglang_version
        ).to_json_object(),
        "stage_a": validations,
        "request_isolation": {
            "unique_session_free_requests": True,
            "completed_distinct_samples_checked": isolation_rows,
            "post_intervening_anchor_recheck": any(
                bool(item["isolation_probe_verified"]) for item in validations
            ),
        },
    }
    print(json.dumps(payload, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
