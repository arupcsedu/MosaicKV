#!/usr/bin/env python3
"""Same consumed identities, same 3-bit policy for every baseline control."""

import argparse
import sys
from pathlib import Path

import evicastkv_precision_tier as previous

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

base = previous.base
ROOT = Path("/scratch/djy8hg/runs/evicastkv/precision-threebit-consumed-20261030-v1")
PARENT = Path("/scratch/djy8hg/runs/evicastkv/precision-tier-consumed-20261030-v2")


def prepare(root, phase):
    if phase != "baselines" or root.exists() or root.name != "comparison":
        raise ValueError("only a new finite consumed comparison permitted")
    canary = base.common.read(PARENT / "canary/plan.json")
    original = base.common.read(PARENT / "comparison/plan.json")
    bound = dict(original["bound_files"])
    for name in (
        "canary/plan.json",
        "comparison/plan.json",
        "canary/analysis/GATE.json",
        "canary/analysis/manifest.json",
    ):
        path = PARENT / name
        bound[str(path)] = sha(path)
    for job in canary["jobs"]:
        folder = PARENT / f"canary/workers/{job['index']}"
        result = base.common.read(folder / "result.json")
        if (
            result["status"] != "completed"
            or result["failures"]
            or result["source_sha"] != canary["source_sha"]
        ):
            raise ValueError("both original 3-bit canaries must have complete provenance")
        for name, digest in result["outputs"].items():
            path = folder / name
            if sha(path) != digest:
                raise ValueError("canary evidence changed")
            bound[str(path)] = digest
        path = folder / "result.json"
        bound[str(path)] = sha(path)
        controls = base.common.read(folder / "precision_controls.json")["controls"]["3"]
        if not controls["numerical_gate"] or not controls["smaller_initial_final_peak"]:
            raise ValueError("both registered 3-bit model and storage canaries must pass")
    plan = dict(
        original,
        source_sha=clean_source(base.SOURCE),
        bound_files=bound,
        protocol="three_bit_common_baseline_controls_consumed_only_v1",
        comparison_precision_bits=3,
        precision_bits=[3],
        worker_script="evicastkv_precision_threebit.sbatch",
        same_precision_policy_for_all_methods=True,
        retrospective_engineering_extension=True,
        original_32_token_guard_unchanged=True,
        all_128_token_differences_must_be_reported=True,
        fresh_evaluation_authorized=False,
        calibration_authorized=False,
        paper_campaign_authorized=False,
        finite_next_phase_authorized=False,
    )
    write_once(root / "plan.json", plan)
    return sha(root / "plan.json")


def analyze(root, expected):
    from mosaickv.evaluation.precision_quality import derive

    previous.analyze(root, expected)
    if not base.common.read(root / "analysis/GATE.json")["failures"]:
        derive(root, expected, root / "analysis/quality_prefixes", source=base.SOURCE)


if __name__ == "__main__":
    base.BASE = ROOT
    base.__dict__["PHASES"] = ("baselines",)
    base.SCRIPT_NAME = "evicastkv_precision_threebit.sbatch"
    base.prepare, base.baseline_study, base.analyze = prepare, previous.study, analyze
    if len(sys.argv) > 1 and sys.argv[1] in ("cpu", "admit", "derive"):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("mode", choices=("cpu", "admit", "derive"))
        parser.add_argument("--root", type=Path, required=True)
        parser.add_argument("--plan-sha", required=True)
        parser.add_argument("--output", type=Path)
        args = parser.parse_args()
        if args.mode == "cpu":
            from evicastkv_consolidated_history import cpu

            cpu(args.root, args.plan_sha)
        elif args.mode == "admit":
            previous.admit(args.root, args.plan_sha)
        else:
            from mosaickv.evaluation.precision_quality import derive

            if args.output is None:
                raise ValueError("new derived output root required")
            derive(args.root, args.plan_sha, args.output, source=base.SOURCE)
    else:
        base.main()
