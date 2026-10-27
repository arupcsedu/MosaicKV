#!/usr/bin/env python3
"""Prepare and analyze only the two existing consumed-input V9 diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from bounded_history_pilot import SOURCE, digest, write_once
from prepare_prefix_output_replay import prepare_output_replay

POLICY = dict(
    prefixes=[1, 4],
    derivative_seconds_limit=120,
    loss_scale=1024.0,
    diagnostic_jvp_atol=1e-5,
    diagnostic_jvp_rtol=0.05,
    shared_anchor_VJP=True,
    one_JVP_crosscheck=True,
    past_cache_held_fixed=True,
    parameters_updated=False,
)
BLOCKED = dict(
    fresh_cohort_authorized=False, calibration_authorized=False, paper_campaign_authorized=False
)


def prepare_downstream(root, source_sha, protocol_path, scope):
    if scope["downstream_jacobian"] != POLICY or any(scope[key] for key in BLOCKED):
        raise ValueError("V9 cannot change its registered scope or derivative controls")
    base = SOURCE / scope["base_protocol"]
    report_path = Path(scope["negative_readout_report"])
    if (
        digest(base) != scope["base_protocol_sha256"]
        or digest(report_path) != scope["negative_readout_report_sha256"]
    ):
        raise ValueError("negative V8/V8.2 evidence or base protocol changed")
    report = json.loads(report_path.read_text())
    if report["useful_prospective_signals"] or any(
        t["registered_accepted_proposals"] for t in report["tasks"]
    ):
        raise ValueError("V9 requires the preserved negative consumed-input diagnostic")
    protocol = yaml.safe_load(base.read_text())
    protocol.update(scope)
    return prepare_output_replay(
        root,
        source_sha,
        protocol_path,
        protocol,
        extra_bound={
            str(base): scope["base_protocol_sha256"],
            str(report_path): scope["negative_readout_report_sha256"],
        },
    )


def analyze(root):
    plan = json.loads((root / "plan.json").read_text())
    if plan["protocol"]["downstream_jacobian"] != POLICY or any(plan[key] for key in BLOCKED):
        raise ValueError("V9 scope changed")
    rows, errors = [], []
    for job in plan["jobs"]:
        i = job["index"]
        worker = root / f"workers/{i}"
        try:
            from head_substitution_v7 import validate

            validate(root, digest(root / "plan.json"), i)
            completion = json.loads((worker / "completed.json").read_text())
            if completion["status"] != "completed":
                raise ValueError("worker completion reports failure")
            manifest_path = Path(completion["manifest"])
            if digest(manifest_path) != completion["manifest_sha256"]:
                raise ValueError("manifest changed")
            manifest = json.loads(manifest_path.read_text())
            if (
                manifest["source"]["git_sha"] != plan["source_sha"]
                or manifest["source"]["git_dirty"]
            ):
                raise ValueError("manifest source mismatch")
            if (
                manifest["execution"]["plan_sha256"] != digest(root / "plan.json")
                or manifest["artifacts"]["raw_output_sha"] != digest(worker / "diagnostic.json")
                or manifest["artifacts"]["log_sha"] != digest(worker / "artifact-hashes.json")
            ):
                raise ValueError("manifest artifact or plan binding changed")
            for path, sha in json.loads((worker / "artifact-hashes.json").read_text()).items():
                if digest(path) != sha:
                    raise ValueError("diagnostic artifact changed")
            diagnostic = json.loads((worker / "diagnostic.json").read_text())
            if diagnostic["status"] != "completed":
                raise ValueError(diagnostic.get("error", "failed worker"))
            result = diagnostic["result"]
            from mosaickv.config import DraftAllocationGuardConfig
            from mosaickv.evaluation.head_substitution_followup import audit_null_control
            from mosaickv.selection.draft_shortlist import validate_shortlist_trace

            if result["selection"]["events"]:
                validate_shortlist_trace(result["selection"])
            guard = DraftAllocationGuardConfig.from_mapping(result["policy"]["guard"])
            if guard != DraftAllocationGuardConfig():
                raise ValueError("registered guard changed")
            controls = (
                all(
                    audit_null_control(c["steps"], guard)["passed"]
                    for c in result["controls"].values()
                )
                and len(result["controls"]) == 2
            )
            derivative = result["downstream_sensitivity"]
            derivative_ok = bool(
                derivative
                and derivative["seconds"] <= 120
                and derivative["candidate_forward_calls_for_ranking"] == 0
                and [r["prefix"] for r in derivative["prefixes"]] == [1, 4]
                and all(r["primal_bitwise_replayed"] for r in derivative["prefixes"])
                and derivative["prefixes"][0]["jvp_crosscheck_passed"]
                and not derivative["parameters_updated"]
            )
            passed = bool(
                result["diagnostic_passed"]
                and controls
                and derivative_ok
                and result["controls_passed"]
                and result["original_guard_unchanged"]
                and result["active_cap_bytes"] == 40 * 2**20
                and result["initial_prompt_storage_bytes"] == 32 * 2**20
                and result["padding_bytes"] == 0
                and result["measured_source_cache_unchanged"]
                and result["logical_positions_unchanged"]
                and result["live_private_states_after_diagnostic"] == 0
                and not result["used_fullkv_fallback"]
                and result["selection"]["accepted"]
                and result["selection"]["meaningful"]
                and result["ranking_screen"]["passed"]
                and not any(result[k] for k in BLOCKED)
            )
            rows.append(
                dict(
                    worker=i,
                    task=job["task"],
                    passed=passed,
                    accepted=result["selection"]["accepted"],
                    meaningful=result["selection"]["meaningful"],
                    proposals=len(result["proposals"]),
                    ranking_screen=result["ranking_screen"],
                    derivative=result["downstream_sensitivity"],
                    controls_rechecked=controls,
                    derivative_feasibility_passed=derivative_ok,
                    diagnostic_wall_seconds=result["diagnostic_wall_seconds"],
                    active_cap_bytes=result["active_cap_bytes"],
                    padding_bytes=result["padding_bytes"],
                )
            )
        except (OSError, ValueError, KeyError) as error:
            failure = dict(worker=i, error=f"{type(error).__name__}: {error}")
            raw = worker / "diagnostic.json"
            if raw.is_file():
                try:
                    saved = json.loads(raw.read_text())
                    failure.update(
                        raw_path=str(raw),
                        raw_sha256=digest(raw),
                        worker_error=saved.get("error"),
                        downstream_sensitivity=saved.get("downstream_sensitivity"),
                        probe_storage=saved.get("probe_storage"),
                    )
                except (OSError, ValueError):
                    failure["raw_record_unreadable"] = True
            errors.append(failure)
    passed = len(rows) == 2 and not errors and all(row["passed"] for row in rows)
    result = dict(
        status="completed" if len(rows) == 2 and not errors else "incomplete",
        diagnostic_passed=passed,
        tasks=rows,
        errors=errors,
        source_sha=plan["source_sha"],
        plan_sha256=digest(root / "plan.json"),
        classification="consumed_input_diagnostic_not_benchmark_quality",
        action="bounded_design_review_only_no_automatic_promotion"
        if passed
        else "stop_substitution_strategy_no_automatic_successor",
        no_new_inference=True,
        **BLOCKED,
    )
    write_once(root / "analysis/report.json", result)
    write_once(
        root / "analysis/decision.json",
        dict(
            action=result["action"],
            report_sha256=digest(root / "analysis/report.json"),
            gpu_submission_authorized=False,
            **BLOCKED,
        ),
    )
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.root.resolve())
