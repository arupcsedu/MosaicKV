#!/usr/bin/env python3
"""Finite zero-workspace Lt ABI/CPU -> operators -> terminal afterany analysis."""

import argparse
import os
import traceback
from pathlib import Path

import evicastkv_row4_operator as common

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE, driver = common.SOURCE, common.driver
common.SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_lt_operator.sbatch"
PARENT = Path("/scratch/djy8hg/runs/evicastkv/repaired-tile-operator-20261104-v1")


def prepare(root, revision):
    if root.exists() or clean_source(SOURCE) != revision:
        raise ValueError("new immutable root and clean source required")
    negative = driver.read(PARENT / "analysis/decision.json")
    if not negative["complete"] or negative["operator_gate"]:
        raise ValueError("preserved complete failed native-tile operator required")
    old = driver.read(PARENT / "plan.json")
    bound = dict(old["bound_files"])
    for p in (
        PARENT / "plan.json",
        PARENT / "analysis/decision.json",
        SOURCE / "mosaickv/docs/zero_workspace_lt_20261104.md",
    ):
        bound[str(p)] = sha(p)
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=revision,
            bound_files=bound,
            stage="zero_workspace_FP32_cublasLt_128_source",
            two_explicit_BF16_workspace_bytes=2 * 2**20,
            vendor_internal_registers_and_spills="not_exposed_not_claimed_zero",
            vendor_internal_GPU_peak_certified=False,
            automatic_model_submission=False,
            automatic_quality_submission=False,
        ),
    )
    common.stage(root, "cpu")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("prepare", "cpu", "cpu_analysis", "operator", "operator_analysis")
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-sha")
    parser.add_argument("--plan-sha")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root, args.source_sha)
    elif args.action == "cpu":
        driver.pair.common.previous.cpu(args.root, args.plan_sha)
        driver.cpu_proof(args.root, driver.checked(args.root, args.plan_sha))
        try:
            from mosaickv.evaluation.zero_workspace_lt import library

            write_once(
                args.root / "abi.json", dict(complete=True, proof=library().abi, no_GPU_called=True)
            )
        except Exception as error:
            write_once(
                args.root / "abi.json",
                dict(
                    complete=False,
                    error=f"{type(error).__name__}: {error}",
                    traceback=traceback.format_exc(),
                ),
            )
            raise
    elif args.action == "cpu_analysis":
        try:
            proof = driver.read(args.root / "abi.json")
            if not proof["complete"]:
                raise ValueError(proof)
        except Exception as error:
            write_once(
                args.root / "cpu-analysis/decision.json",
                dict(
                    complete=False,
                    failures=[str(error)],
                    no_GPU_submitted=True,
                ),
            )
        else:
            common.analyze(args.root, args.plan_sha, args.action)
    elif args.action == "operator":
        from mosaickv.evaluation import repaired_tile_operator, row_major_source_operator

        row_major_source_operator.run = lambda i: repaired_tile_operator.run(i, engine_name="lt128")
        common.worker(args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    else:
        common.analyze(args.root, args.plan_sha, args.action)


if __name__ == "__main__":
    main()
