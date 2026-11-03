#!/usr/bin/env python3
"""Measured repair/restore follow-up: operators -> memory/parity -> paired timing.

Exactly one new execution policy, no new inputs or benchmark quality campaign.
Every failed/missing producer is analyzed; unsuccessful policies stop unchanged.
"""

import argparse
import os
from pathlib import Path

import evicastkv_full_prompt_native as driver
import evicastkv_native_operator_study as operators

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = driver.SOURCE
PARENT = Path("/scratch/djy8hg/runs/evicastkv/native-operator-consumed-20261103-v1")
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_native_restore_followup.sbatch"
DOC = SOURCE / "mosaickv/docs/native_restore_followup_20261103.md"


def prepare(root, revision, *, candidate="fused"):
    global PARENT, DOC

    if candidate == "prepared":
        PARENT = Path("/scratch/djy8hg/runs/evicastkv/native-restore-consumed-20261103-v1")
        DOC = SOURCE / "mosaickv/docs/native_prepared_restore_20261103.md"
    elif candidate != "fused":
        raise ValueError("registered fused or prepared candidate required")
    if root.exists() or clean_source(SOURCE) != revision:
        raise ValueError("new immutable root and full clean source SHA required")
    negative = driver.read(PARENT / "analysis/decision.json")
    if not negative["complete"] or negative["failures"]:
        raise ValueError("complete measured operator evidence required")
    if candidate == "prepared" and negative["fused_operator_gate"]:
        raise ValueError("preserve the negative fused-only gate before this bounded follow-up")
    old = driver.read(PARENT / "plan.json")
    bound = dict(old["bound_files"])
    for p in (PARENT / "plan.json", PARENT / "analysis/decision.json", DOC):
        bound[str(p)] = sha(p)
    for i in (0, 1):
        result = driver.read(PARENT / f"workers/{i}/result.json")
        if result["status"] != "completed":
            raise ValueError("negative operator results must be preserved complete")
        for name, value in result["outputs"].items():
            p = PARENT / f"workers/{i}" / name
            if sha(p) != value:
                raise ValueError("changed parent operator output")
            bound[str(p)] = value
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=revision,
            bound_files=bound,
            chain_root=str(root),
            stage="consumed_native_restore_operator_gate",
            products=["bmm", "fused", "prepared"] if candidate == "prepared" else ["bmm", "fused"],
            operator_candidate=candidate,
            native_policies=[
                "kv_fused_prepared" if candidate == "prepared" else "kv_fused_restore"
            ],
            maximum_policy_attempts=1,
            automatic_full_request_submission=True,
            automatic_quality_submission=False,
            full_request_requires_operator_improvement=True,
            frozen_operator_ratio_upper=0.9,
            maximum_worker_minutes=60,
        ),
    )
    driver.pair.common.previous.cpu(root, sha(root / "plan.json"))
    queue = driver.subprocess.check_output(
        ["squeue", "-h", "-r", "-u", os.environ["USER"], "-o", "%b"], text=True
    )
    if "gpu" in queue.lower():
        raise ValueError("finish preceding owned GPU jobs first")
    driver.submit_array(root, "operator", "operator_analysis")


def eligible(rows, candidate="fused"):
    if len(rows) != 2:
        return False
    grid = {(layer, padding) for layer in (0, 16, 31) for padding in (0, 64, 127)}
    for row in rows:
        cases = row["cases"]
        if len(cases) != 9 or {(c["layer"], c["padding"]) for c in cases} != grid:
            return False
        for case in cases:
            if (
                not case["correctness"][candidate]["bitwise"]
                or not case["correctness"][candidate]["history_sentinel"]
                or case["correctness"][candidate]["extra_output_workspace_bytes"] != 0
                or any(
                    case["aggregate"][candidate]["mean_cuda_ms"]
                    > 0.9 * case["aggregate"][mode]["mean_cuda_ms"]
                    for mode in ("bmm", "original")
                )
            ):
                return False
            if candidate == "prepared" and (
                case["aggregate"][candidate]["mean_cuda_ms"]
                > 0.99 * case["aggregate"]["fused"]["mean_cuda_ms"]
                or not case["prepared_activity"]["weight_hits"]
                or not case["prepared_activity"]["restore_hits"]
            ):
                return False
    return True


def operator_analysis(root, expected):
    plan = driver.checked(root, expected)
    rows, failures = driver.load_producers(root, plan, "profile.json")
    candidate = plan.get("operator_candidate", "fused")
    passed = not failures and eligible(rows, candidate)
    decision = dict(
        complete=not failures,
        failures=failures,
        fused_operator_gate=passed,
        operator_candidate=candidate,
        source_sha=plan["source_sha"],
        independent_quality_N=0,
        paper_superiority_established=False,
        full_request_submitted=False,
        operator_ratios=[
            [
                dict(
                    layer=c["layer"],
                    padding=c["padding"],
                    versus_bmm=c["aggregate"][candidate]["mean_cuda_ms"]
                    / c["aggregate"]["bmm"]["mean_cuda_ms"],
                    versus_original=c["aggregate"][candidate]["mean_cuda_ms"]
                    / c["aggregate"]["original"]["mean_cuda_ms"],
                )
                for c in r["cases"]
            ]
            for r in rows
        ],
    )
    if passed:
        decision["full_request_submitted"] = True
    write_once(root / "analysis/decision.json", decision)
    if passed:
        target = driver.child(root, "feasibility", stage="native_restore_model_feasibility")
        driver.submit_array(target, "feasibility", "feasibility_analysis")
    else:
        driver.closeout(root, decision)


def main():
    driver.SCRIPT = SCRIPT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "prepare",
            "operator",
            "operator_analysis",
            "feasibility",
            "feasibility_analysis",
            "paired",
            "paired_analysis",
            "all_five",
            "final_analysis",
        ),
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--source-sha")
    parser.add_argument("--plan-sha")
    parser.add_argument("--candidate", choices=("fused", "prepared"), default="fused")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root, args.source_sha, candidate=args.candidate)
    elif args.action == "operator":
        operators.worker(args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    elif args.action == "operator_analysis":
        operator_analysis(args.root, args.plan_sha)
    elif args.action in ("feasibility", "paired", "all_five"):
        getattr(driver, args.action)(
            args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"])
        )
    else:
        getattr(driver, args.action)(args.root, args.plan_sha)


if __name__ == "__main__":
    main()
