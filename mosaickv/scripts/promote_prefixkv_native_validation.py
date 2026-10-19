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
    fidelity = json.loads(args.fidelity_report.read_text(encoding="utf-8"))
    required = {
        "status": "passed",
        "plan_sha256": args.parent_plan_sha,
        "sample_count": 64,
        "report_count": 32,
        "retention_one_passed": True,
        "selected_positions_and_per_layer_lengths_match": True,
        "decode_evictions_match": True,
        "greedy_tokens_and_logits_match": True,
        "teacher_forced_nll_and_token_counts_match": True,
        "large_execution_authorized": False,
    }
    if any(fidelity.get(key) != value for key, value in required.items()):
        raise ValueError("PrefixKV native fidelity gate did not pass exactly")
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
        fidelity_report=str(args.fidelity_report),
        fidelity_report_sha256=digest(args.fidelity_report),
    )
    contract = source / "mosaickv/configs/experiments/shared_byte_comparison_v2.yaml"
    bound_files = dict(plan["bound_files"])
    bound_files.pop(str(parent["comparison_contract"]), None)
    bound_files[str(contract)] = digest(contract)
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
    parser.add_argument("--fidelity-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    promote(parser.parse_args())
