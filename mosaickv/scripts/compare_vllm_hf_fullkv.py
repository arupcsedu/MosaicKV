#!/usr/bin/env python3
"""Compare controlled HF and vLLM FullKV artifacts without editing raw runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(
    *,
    vllm_manifest_path: Path,
    vllm_trace_path: Path,
    hf_manifest_path: Path,
    hf_trace_path: Path,
) -> dict[str, Any]:
    vllm_manifest = _load_object(vllm_manifest_path)
    vllm_trace = _load_object(vllm_trace_path)
    hf_manifest = _load_object(hf_manifest_path)
    hf_trace = _load_object(hf_trace_path)
    trials = vllm_trace.get("trials")
    if not isinstance(trials, list) or not trials or not isinstance(trials[0], dict):
        raise ValueError("vLLM trace contains no trial")
    vllm_ids = [int(value) for value in trials[0].get("token_ids", [])]
    hf_ids = [int(value) for value in hf_trace.get("generated_token_ids", [])]
    if not vllm_ids or not hf_ids:
        raise ValueError("both traces must contain generated token IDs")
    common = min(len(vllm_ids), len(hf_ids))
    matches = sum(vllm_ids[index] == hf_ids[index] for index in range(common))
    agreement = matches / max(len(vllm_ids), len(hf_ids))
    checks = {
        "model": vllm_manifest.get("model") == hf_manifest.get("model"),
        "input_hashes": vllm_manifest.get("inputs") == hf_manifest.get("inputs"),
        "generation_parameters": (
            vllm_manifest.get("generation", {}).get("parameters_sha")
            == hf_manifest.get("generation", {}).get("parameters_sha")
        ),
        "sample_id": vllm_trace.get("sample_id") == hf_trace.get("sample_id"),
        "token_ids": vllm_ids == hf_ids,
        "decoded_text": trials[0].get("answer") == hf_trace.get("answer"),
    }
    return {
        "schema_version": 1,
        "status": "passed" if all(checks.values()) else "failed",
        "comparison": "hf_fullkv_vs_vllm_fullkv",
        "checks": checks,
        "generated_tokens": {"hf": len(hf_ids), "vllm": len(vllm_ids)},
        "token_agreement": agreement,
        "artifacts": {
            "hf_manifest": str(hf_manifest_path.resolve()),
            "hf_manifest_sha256": _sha256(hf_manifest_path),
            "hf_trace": str(hf_trace_path.resolve()),
            "hf_trace_sha256": _sha256(hf_trace_path),
            "vllm_manifest": str(vllm_manifest_path.resolve()),
            "vllm_manifest_sha256": _sha256(vllm_manifest_path),
            "vllm_trace": str(vllm_trace_path.resolve()),
            "vllm_trace_sha256": _sha256(vllm_trace_path),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-manifest", type=Path, required=True)
    parser.add_argument("--vllm-trace", type=Path, required=True)
    parser.add_argument("--hf-manifest", type=Path, required=True)
    parser.add_argument("--hf-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(
        vllm_manifest_path=args.vllm_manifest,
        vllm_trace_path=args.vllm_trace,
        hf_manifest_path=args.hf_manifest,
        hf_trace_path=args.hf_trace,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
