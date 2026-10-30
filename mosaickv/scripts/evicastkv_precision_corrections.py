#!/usr/bin/env python3
"""Four frozen same-consumed-input precision-risk ablations, no cohort expansion."""

import argparse
import copy
import sys
from pathlib import Path

import evicastkv_precision_tier as previous

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

base = previous.base
ROOT = Path("/scratch/djy8hg/runs/evicastkv/precision-corrections-consumed-20261030-v1")
PARENT = Path("/scratch/djy8hg/runs/evicastkv/precision-threebit-consumed-20261030-v2/canary")
REFERENCE = Path("/scratch/djy8hg/runs/evicastkv/precision-tier-consumed-20261030-v2/comparison")


def prepare(root, phase):
    if phase != "baselines" or root.exists() or root.name != "diagnostic":
        raise ValueError("only new four-worker consumed diagnostic permitted")
    parent = base.common.read(PARENT / "plan.json")
    decision = PARENT / "analysis/precision_decision.json"
    if not base.common.read(decision)["registered_precision_passed"]:
        raise ValueError("both encoded exact-text canaries must pass")
    bound = dict(parent["bound_files"])
    for p in (
        PARENT / "plan.json",
        decision,
        PARENT / "analysis/manifest.json",
        REFERENCE / "plan.json",
    ):
        bound[str(p)] = sha(p)
    for ref_job in base.common.read(REFERENCE / "plan.json")["jobs"]:
        if ref_job["method"] == "full_kv":
            for name in ("result.json", "manifest.json", "logits.pt", "ordinary_parity.json"):
                p = REFERENCE / f"workers/{ref_job['index']}" / name
                bound[str(p)] = sha(p)
    jobs = []
    for old in parent["jobs"]:
        path = PARENT / f"workers/{old['index']}/precision_controls.json"
        controls = base.common.read(path)["controls"]["3"]
        bound[str(path)] = sha(path)
        for mode in ("mse", "query"):
            job = copy.deepcopy(old)
            job.update(
                index=len(jobs),
                correction_mode=mode,
                uncorrected_raw_text_storage=controls["raw_exact_control"]["storage"],
            )
            jobs.append(job)
    plan = dict(
        parent,
        source_sha=clean_source(base.SOURCE),
        bound_files=bound,
        jobs=jobs,
        protocol="storage_funded_exact_precision_risk_exceptions_consumed_v1",
        stage="comparison",
        phase="baselines",
        requested_workers=4,
        correction_fraction=0.005,
        correction_prompt_window=16,
        maximum_correction_fraction=0.005,
        common_history_consolidated=True,
        comparison_precision_bits=3,
        worker_script="evicastkv_precision_corrections.sbatch",
        external_fullkv_reference_root=str(REFERENCE),
        no_exception_bytes_above_saved_exact_text_bytes=True,
        reference_oracle_available_to_proposer=False,
        benchmark_answers_available_to_proposer=False,
        finite_next_phase_authorized=False,
        fresh_evaluation_authorized=False,
        calibration_authorized=False,
        paper_campaign_authorized=False,
    )
    write_once(root / "plan.json", plan)
    return sha(root / "plan.json")


def analyze(root, expected):
    from mosaickv.evaluation.precision_quality import derive

    previous.analyze(root, expected)
    plan = base.common.read(root / "plan.json")
    if not base.common.read(root / "analysis/GATE.json")["failures"]:
        derive(
            root,
            expected,
            root / "analysis/quality_prefixes",
            source=base.SOURCE,
            reference_root=Path(plan["external_fullkv_reference_root"]),
        )


if __name__ == "__main__":
    base.BASE = ROOT
    base.__dict__["PHASES"] = ("baselines",)
    base.SCRIPT_NAME = "evicastkv_precision_corrections.sbatch"
    base.prepare, base.baseline_study, base.analyze = prepare, previous.study, analyze
    if len(sys.argv) > 1 and sys.argv[1] in ("cpu", "admit"):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("mode", choices=("cpu", "admit"))
        parser.add_argument("--root", type=Path, required=True)
        parser.add_argument("--plan-sha", required=True)
        args = parser.parse_args()
        if args.mode == "cpu":
            from evicastkv_consolidated_history import cpu

            cpu(args.root, args.plan_sha)
        else:
            previous.admit(args.root, args.plan_sha)
    else:
        base.main()
