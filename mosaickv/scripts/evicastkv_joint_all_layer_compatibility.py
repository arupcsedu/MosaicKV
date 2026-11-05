#!/usr/bin/env python3
"""Frozen all-layer operator gate on the two consumed inputs only."""

from pathlib import Path

import evicastkv_joint_row_feasibility as parent

from mosaickv.evaluation import joint_all_layer_compatibility as probe
from mosaickv.evaluation.fox_campaign import write_once

SOURCE = parent.SOURCE
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_joint_all_layer_compatibility.sbatch"
ORIGINAL_PREPARE = parent.prepare


def prepare(root, revision):
    gate = parent.driver.read(parent.WAIT / "analysis/decision.json")
    if (
        not gate["complete"]
        or gate["failures"]
        or list(probe.FROZEN) not in gate["common_exact_configurations_both_inputs"]
    ):
        raise ValueError("positive frozen-configuration preceding operator gate required")
    ORIGINAL_PREPARE(root, revision)


def analyze(root, expected):
    plan = parent.driver.checked(root, expected)
    rows, failures = parent.driver.load_producers(root, plan, "profile.json")
    passed = (
        not failures and len(rows) == 2 and all(r["records"][0]["all_layer_exact"] for r in rows)
    )
    write_once(
        root / "analysis/decision.json",
        dict(
            complete=not failures,
            failures=failures,
            source_sha=plan["source_sha"],
            plan_sha256=expected,
            rows=rows,
            all_layer_exact_both_inputs=passed,
            runtime_transaction_may_be_implemented_not_promoted=passed,
            no_runtime_integrated=True,
            no_latency_or_quality_acceptance=True,
            independent_quality_N=0,
            automatic_followup=False,
        ),
    )


def main():
    parent.WAIT = Path(
        "/scratch/djy8hg/runs/evicastkv/joint-kernel-compatibility-consumed-20261105-v1"
    )
    parent.SCRIPT, parent.probe = SCRIPT, probe
    parent.ONLY_LAYER = None
    parent.STAGE = parent.VARIANT = "consumed_joint_all_layer_compatibility"
    parent.DOC = SOURCE / "mosaickv/docs/joint_all_layer_compatibility.md"
    parent.EXTRA_PLAN = dict(
        frozen_configuration=probe.FROZEN,
        registered_layers=list(range(32)),
        observed_components_per_input=192,
        no_configuration_search_or_new_samples=True,
    )
    parent.EXTRA_BOUND_FILES = (parent.WAIT / "analysis/decision.json",)
    parent.prepare = prepare
    parent.analyze = analyze
    parent.main()


if __name__ == "__main__":
    main()
