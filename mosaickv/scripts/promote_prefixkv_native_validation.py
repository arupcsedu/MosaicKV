#!/usr/bin/env python3
"""Create a byte-validation plan only after complete PrefixKV native parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_byte_capped_published_baseline_pilot import _require_clean_source
from run_prefixkv_decode_preflight import digest, write_once


def promote(args: argparse.Namespace) -> dict:
    source = Path(__file__).resolve().parents[2]
    _require_clean_source(source, args.source_sha)
    parent_plan_path = args.parent_root / "plan.json"
    if digest(parent_plan_path) != args.parent_plan_sha:
        raise ValueError("parent PrefixKV plan changed")
    parent = json.loads(parent_plan_path.read_text(encoding="utf-8"))
    gate = json.loads(args.parity_gate.read_text(encoding="utf-8"))
    comparison = json.loads(args.parity_comparison.read_text(encoding="utf-8"))
    if gate != {
        "checked_samples": 4,
        "failures": [],
        "passed": True,
        "thresholds": gate.get("thresholds"),
    }:
        raise ValueError("the registered four-sample PrefixKV parity gate did not pass")
    samples = comparison.get("samples", [])
    if (
        comparison.get("status") != "comparable"
        or comparison.get("official_repository_sha")
        != "597f1ab032704951550f93bcc8a23f1454b80aa4"
        or comparison.get("control_mismatches")
        or comparison.get("corpus", {}).get("sample_count") != 4
        or comparison.get("corpus", {}).get("teacher_forced_ppl_delta") != 0.0
        or len(samples) != 4
        or any(
            sample.get("per_layer_cache_sizes_match") is not True
            or sample.get("answers_exact_match") is not True
            or sample.get("actual_active_kv_byte_delta") != 0
            or sample.get("retained_byte_delta") != 0
            or sample.get("perplexity_delta") != 0.0
            or sample.get("rouge_l_f1_delta") != 0.0
            or sample.get("token_agreement") != 1.0
            for sample in samples
        )
    ):
        raise ValueError("PrefixKV official/reimplementation parity comparison failed")
    if args.output_root.exists():
        raise FileExistsError("validation promotion requires a new immutable output root")
    plan = dict(parent)
    plan.update(
        source_sha=args.source_sha,
        classification="native_metric_byte_validation",
        fidelity_execution_authorized=False,
        byte_validation_authorized=True,
        full_campaign_authorized=False,
        parent_plan=str(parent_plan_path),
        parent_plan_sha256=args.parent_plan_sha,
        parity_gate=str(args.parity_gate),
        parity_gate_sha256=digest(args.parity_gate),
        parity_comparison=str(args.parity_comparison),
        parity_comparison_sha256=digest(args.parity_comparison),
    )
    contract = source / "mosaickv/configs/experiments/shared_byte_comparison_v2.yaml"
    bound_files = dict(plan["bound_files"])
    bound_files.pop(str(parent["comparison_contract"]), None)
    bound_files[str(contract)] = digest(contract)
    bound_files[str(args.parity_gate)] = digest(args.parity_gate)
    bound_files[str(args.parity_comparison)] = digest(args.parity_comparison)
    plan["bound_files"] = bound_files
    plan["comparison_contract"] = str(contract)
    plan["comparison_contract_sha256"] = digest(contract)
    for job in plan["jobs"]:
        task = str(job["task"])
        shard = int(job["shard"])
        method = str(job["method"])
        cap = 40 if job["cap_mib"] is None else int(job["cap_mib"])
        job["reference"] = str(
            args.output_root / "references" / task / f"{shard:02d}.json"
        )
        job["output"] = str(
            args.output_root / "outputs" / task / f"{shard:02d}-{method}-{cap}"
        )
    write_once(args.output_root / "plan.json", plan)
    payload = {
        "plan": str(args.output_root / "plan.json"),
        "sha256": digest(args.output_root / "plan.json"),
        "group_count": len({(job["task"], job["shard"]) for job in plan["jobs"]}),
        "submitted": False,
    }
    print(json.dumps(payload, sort_keys=True))
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--parent-plan-sha", required=True)
    parser.add_argument("--parity-gate", type=Path, required=True)
    parser.add_argument("--parity-comparison", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    promote(parser.parse_args())
