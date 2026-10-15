#!/usr/bin/env python3
"""Recheck completed OCR adaptation artifacts without modifying or rerunning them."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(root):
    summary = json.loads((root / "analysis/prefixkv-validation.json").read_text())
    results = dict(
        audit="read_only_ocr_adaptation_not_native_fidelity",
        inputs={},
        source_sha=None,
        rows=0,
        errors=[],
        compressed_initial_bytes=[],
        compressed_final_bytes=[],
        generated_tokens=[],
        decode_evictions=[],
        strict_peak_backing_storage_verified=False,
        reason="historical payload rows do not instrument the new live-storage peak contract",
        tasks=summary["tasks"],
        native_fidelity_pass=False,
        large_execution_allowed=False,
    )
    seen = set()
    controls = {}
    sources = set()
    for name, expected in summary["inputs"].items():
        path = Path(name)
        actual = sha(path)
        results["inputs"][str(path)] = actual
        if actual != expected:
            raise ValueError(f"completion hash changed: {path}")
        completion = json.loads(path.read_text())
        sources.add(completion["source_sha"])
        if (
            completion["status"] != "completed"
            or completion["source_dirty"]
            or completion["failures"]
        ):
            results["errors"].append(f"incomplete or dirty worker {path}")
        for record in completion["records"]:
            if set(row["sample_id"] for row in record["rows"]) != set(completion["planned_ids"]):
                results["errors"].append(f"wrong cohort {path}/{record['variant']}")
            for row in record["rows"]:
                key = (completion["task"], row["sample_id"], record["variant"])
                if key in seen:
                    raise ValueError("duplicate paired row")
                seen.add(key)
                if row["status"] != "completed" or row["error"] or not row["trace_validated"]:
                    results["errors"].append(f"failed row {key}")
                trace_path = Path(row["trace_path"])
                if sha(trace_path) != row["trace_sha256"]:
                    raise ValueError(f"trace hash changed: {trace_path}")
                trace = json.loads(trace_path.read_text())
                if trace["status"] != "completed":
                    results["errors"].append(f"failed trace {key}")
                control_key = (completion["task"], row["sample_id"])
                control = row["input_controls_sha256"]
                if controls.setdefault(control_key, control) != control:
                    results["errors"].append(f"input controls differ {control_key}")
                if record["variant"] != "full_kv":
                    if row["effective_method"] != record["variant"]:
                        results["errors"].append(f"wrong effective label {key}")
                    for target, field in (
                        ("compressed_initial_bytes", "active_kv_bytes"),
                        ("compressed_final_bytes", "final_active_kv_bytes"),
                        ("generated_tokens", "generated_tokens"),
                        ("decode_evictions", "decode_evictions"),
                    ):
                        results[target].append(row[field])
                    if row["final_active_kv_bytes"] != row["active_kv_bytes"] + 524288 * 63:
                        results["errors"].append(f"unexpected append growth {key}")
                results["rows"] += 1
    if len(sources) != 1:
        raise ValueError("mixed source versions")
    results["source_sha"] = next(iter(sources))
    for field in (
        "compressed_initial_bytes",
        "compressed_final_bytes",
        "generated_tokens",
        "decode_evictions",
    ):
        results[field] = sorted(set(results[field]))
    results["artifact_integrity_pass"] = not results["errors"]
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = audit(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps({key: value for key, value in report.items() if key not in {"tasks", "inputs"}})
    )
