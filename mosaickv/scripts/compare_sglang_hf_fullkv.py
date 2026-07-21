#!/usr/bin/env python3
"""Compare controlled HF and SGLang FullKV artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(
    *,
    sglang_manifest_path: Path,
    sglang_trace_path: Path,
    hf_manifest_path: Path,
    hf_trace_path: Path,
) -> dict[str, Any]:
    sglang_manifest = _object(sglang_manifest_path)
    sglang_trace = _object(sglang_trace_path)
    hf_manifest = _object(hf_manifest_path)
    hf_trace = _object(hf_trace_path)
    trials = sglang_trace.get("trials")
    if not isinstance(trials, list) or not trials or not isinstance(trials[0], dict):
        raise ValueError("SGLang trace contains no trial")
    sglang_ids = [int(value) for value in trials[0].get("token_ids", [])]
    hf_ids = [int(value) for value in hf_trace.get("generated_token_ids", [])]
    if not sglang_ids or not hf_ids:
        raise ValueError("both traces must contain generated token IDs")
    common = min(len(sglang_ids), len(hf_ids))
    matches = sum(sglang_ids[index] == hf_ids[index] for index in range(common))
    checks = {
        "model": sglang_manifest.get("model") == hf_manifest.get("model"),
        "input_hashes": sglang_manifest.get("inputs") == hf_manifest.get("inputs"),
        "generation_parameters": (
            sglang_manifest.get("generation", {}).get("parameters_sha")
            == hf_manifest.get("generation", {}).get("parameters_sha")
        ),
        "sample_id": sglang_trace.get("sample_id") == hf_trace.get("sample_id"),
        "token_ids": sglang_ids == hf_ids,
        "decoded_text": trials[0].get("answer") == hf_trace.get("answer"),
    }
    return {
        "schema_version": 1,
        "status": "passed" if all(checks.values()) else "failed",
        "comparison": "hf_fullkv_vs_sglang_fullkv",
        "checks": checks,
        "generated_tokens": {"hf": len(hf_ids), "sglang": len(sglang_ids)},
        "token_agreement": matches / max(len(sglang_ids), len(hf_ids)),
        "artifacts": {
            "hf_manifest": str(hf_manifest_path.resolve()),
            "hf_manifest_sha256": _sha256(hf_manifest_path),
            "hf_trace": str(hf_trace_path.resolve()),
            "hf_trace_sha256": _sha256(hf_trace_path),
            "sglang_manifest": str(sglang_manifest_path.resolve()),
            "sglang_manifest_sha256": _sha256(sglang_manifest_path),
            "sglang_trace": str(sglang_trace_path.resolve()),
            "sglang_trace_sha256": _sha256(sglang_trace_path),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sglang-manifest", type=Path, required=True)
    parser.add_argument("--sglang-trace", type=Path, required=True)
    parser.add_argument("--hf-manifest", type=Path, required=True)
    parser.add_argument("--hf-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(
        sglang_manifest_path=args.sglang_manifest,
        sglang_trace_path=args.sglang_trace,
        hf_manifest_path=args.hf_manifest,
        hf_trace_path=args.hf_trace,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
