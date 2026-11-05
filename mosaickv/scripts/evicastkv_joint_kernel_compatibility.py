#!/usr/bin/env python3
"""Finite consumed-input numerical study, never a runtime promotion."""

from pathlib import Path

import evicastkv_joint_row_feasibility as parent

from mosaickv.evaluation import joint_kernel_compatibility as probe
from mosaickv.evaluation.fox_campaign import sha, write_once

SOURCE = parent.SOURCE
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_joint_kernel_compatibility.sbatch"


def analyze(root, expected):
    plan = parent.driver.checked(root, expected)
    rows, failures = parent.driver.load_producers(root, plan, "profile.json")
    exact = [set(map(tuple, r["records"][0]["common_exact_configurations"])) for r in rows]
    common = sorted(set.intersection(*exact)) if len(exact) == 2 and not failures else []
    write_once(
        root / "analysis/decision.json",
        dict(
            complete=not failures,
            failures=failures,
            source_sha=plan["source_sha"],
            plan_sha256=expected,
            rows=rows,
            common_exact_configurations_both_inputs=common,
            same_parent_integration_eligible_for_further_operator_checks=bool(common),
            no_runtime_integrated=True,
            no_latency_or_quality_acceptance=True,
            independent_quality_N=0,
            automatic_followup=False,
        ),
    )


def main():
    parent.WAIT = Path("/scratch/djy8hg/runs/evicastkv/joint-row-feasibility-consumed-20261105-v1")
    parent.SCRIPT = SCRIPT
    parent.probe = probe
    parent.STAGE = "consumed_joint_kernel_compatibility"
    parent.VARIANT = parent.STAGE
    parent.DOC = SOURCE / "mosaickv/docs/joint_kernel_compatibility.md"
    trace = Path(
        "/scratch/djy8hg/runs/evicastkv/lt-model-profile-consumed-20261104-v1/"
        "workers/0/mosaickv_exact-graph/step-1.json"
    )
    parent.EXTRA_PLAN = dict(
        maximum_algorithm_ids=probe.MAX_IDS,
        maximum_executed_configurations=probe.MAX_PLANS,
        maximum_custom_option=probe.MAX_CUSTOM,
        algorithm_selection_not_based_on_timing_or_benchmark_scores=True,
        inspected_original_kernel_trace=str(trace),
        inspected_original_kernel_trace_sha256=sha(trace),
        observed_kernel_geometry_NOT_public_algorithm_identity=True,
    )
    # Protocol preparation and the generic controller both bind the new files.
    parent.analyze = analyze
    probe.WINDOWS = probe.original.WINDOWS
    parent.main()


if __name__ == "__main__":
    main()
