#!/usr/bin/env python3
"""Reduce the complete PrefixKV native differential gate, fail closed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_prefixkv_decode_preflight import digest, write_once

UPSTREAM_SHA = "597f1ab032704951550f93bcc8a23f1454b80aa4"


def _check_trajectory(value: dict, *, atol: float, teacher_forced: bool) -> None:
    if (
        value.get("passed") is not True
        or value.get("upstream_sha") != UPSTREAM_SHA
        or value.get("token_agreement") != 1.0
        or value.get("source_isolated") is not True
        or float(value.get("maximum_logit_difference", float("inf"))) > atol
    ):
        raise ValueError("PrefixKV upstream/local trajectory parity failed")
    steps = value.get("steps")
    if not isinstance(steps, list):
        raise ValueError("PrefixKV trajectory steps are malformed")
    if any(
        step.get("selected_positions_and_cache_match") is not True
        or float(step.get("maximum_logit_difference", float("inf"))) > atol
        for step in steps
    ):
        raise ValueError("PrefixKV step-level selection/cache/logit parity failed")
    forced = value.get("teacher_forced")
    if teacher_forced:
        if not isinstance(forced, dict) or (
            float(forced.get("nll_absolute_difference", float("inf")))
            > float(forced.get("nll_absolute_tolerance", -1.0))
        ):
            raise ValueError("PrefixKV teacher-forced NLL parity failed")
    elif forced is not None or value.get("generated_token_ids") != value.get(
        "official_generated_token_ids"
    ):
        raise ValueError("PrefixKV greedy generation parity failed")


def analyze(root: Path, plan_sha: str, output: Path) -> dict:
    plan_path = root / "plan.json"
    if digest(plan_path) != plan_sha:
        raise ValueError("immutable plan hash changed")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    jobs = [job for job in plan["jobs"] if job["method"] == "full_kv"]
    if len(jobs) != 32:
        raise ValueError("native fidelity gate requires 32 two-sample reference shards")
    atol = float(plan["fidelity_logit_atol"])
    reports: list[dict] = []
    observed_ids: set[str] = set()
    for job in jobs:
        path = root / "fidelity" / job["task"] / f"{int(job['shard']):02d}.json"
        if not path.is_file():
            raise ValueError(f"missing native fidelity report: {path}")
        report = json.loads(path.read_text(encoding="utf-8"))
        if (
            report.get("status") != "passed"
            or report.get("failures")
            or report.get("plan_sha256") != plan_sha
            or report.get("planned_ids") != job["sample_ids"]
            or len(report.get("results", [])) != len(job["sample_ids"])
        ):
            raise ValueError(f"failed or mismatched native fidelity report: {path}")
        for result in report["results"]:
            sample_id = result.get("sample_id")
            if sample_id not in job["sample_ids"] or sample_id in observed_ids:
                raise ValueError("duplicate or unregistered PrefixKV fidelity identity")
            observed_ids.add(sample_id)
            retention = result.get("retention_one")
            if (
                not isinstance(retention, dict)
                or float(retention.get("token_agreement", -1.0)) != 1.0
                or float(retention.get("maximum_logit_difference", float("inf"))) > atol
            ):
                raise ValueError("retention-one parity failed")
            _check_trajectory(result["greedy"], atol=atol, teacher_forced=False)
            _check_trajectory(result["teacher_forced"], atol=atol, teacher_forced=True)
        reports.append(
            {
                "path": str(path),
                "sha256": digest(path),
                "task": job["task"],
                "shard": job["shard"],
                "sample_ids": job["sample_ids"],
            }
        )
    expected = {sample_id for job in jobs for sample_id in job["sample_ids"]}
    if observed_ids != expected or len(expected) != 64:
        raise ValueError("PrefixKV native fidelity denominator changed")
    payload = {
        "schema_version": 1,
        "status": "passed",
        "plan_sha256": plan_sha,
        "source_sha": plan["source_sha"],
        "upstream_prefixkv_sha": UPSTREAM_SHA,
        "tasks": sorted({job["task"] for job in jobs}),
        "sample_count": len(observed_ids),
        "report_count": len(reports),
        "retention_one_passed": True,
        "selected_positions_and_per_layer_lengths_match": True,
        "decode_evictions_match": True,
        "greedy_tokens_and_logits_match": True,
        "teacher_forced_nll_and_token_counts_match": True,
        "large_execution_authorized": False,
        "reports": reports,
    }
    write_once(output, payload)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(analyze(arguments.root, arguments.plan_sha, arguments.output), indent=2))
