#!/usr/bin/env python3
"""Finite full CPU suite -> two bounded synthetic prefix operators -> analysis."""

import argparse
import os
from pathlib import Path

import evicastkv_row4_operator as common

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE, driver = common.SOURCE, common.driver
common.SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_paired_prefix_operator.sbatch"


def prepare(root, revision):
    if root.exists() or clean_source(SOURCE) != revision:
        raise ValueError("new root and clean immutable source required")
    old = driver.read(common.PARENT / "plan.json")
    bound = dict(old["bound_files"])
    for p in (
        common.PARENT / "plan.json",
        common.PARENT / "analysis/decision.json",
        common.PROFILE / "decision.json",
        Path("/scratch/djy8hg/runs/evicastkv/row4-operator-20261104-retry1/analysis/decision.json"),
        SOURCE / "mosaickv/docs/paired_prefix_operator_20261104.md",
    ):
        bound[str(p)] = sha(p)
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=revision,
            bound_files=bound,
            stage="synthetic_paired_prefix_operator_only",
            operator_upper_ratio=0.8,
            no_checkpoint_loaded=True,
            automatic_model_submission=False,
            automatic_quality_submission=False,
            paper_campaign_authorized=False,
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
    elif args.action == "operator":
        from mosaickv.evaluation import paired_prefix_operator, row_major_source_operator

        row_major_source_operator.run = paired_prefix_operator.run
        common.worker(args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    else:
        common.analyze(args.root, args.plan_sha, args.action)


if __name__ == "__main__":
    main()
