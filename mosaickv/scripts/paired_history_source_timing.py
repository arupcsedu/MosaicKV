#!/usr/bin/env python3
"""Apply common immediate-pair reuse to storage-only AND the history variant."""

from pathlib import Path
from unittest.mock import patch

import header_history_timing as timing
import history_three_arm as three

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

PARENT = Path("/scratch/djy8hg/runs/evicastkv/mixed-history-projection-20261108-v1")
ORIGINAL = Path("/scratch/djy8hg/runs/evicastkv/selective-history64-20261108-v1")
ARMS = ["storage_common_headers", "storage_common_pair_reuse", "selective_fused_pair_reuse"]
POLICIES = {
    ARMS[0]: dict(history=False, mixed=False),
    ARMS[1]: dict(history=False, mixed=False, paired_source=True),
    ARMS[2]: dict(history=True, mixed=True, fused_unpack=True, paired_source=True),
}


def prepare(root):
    if root.exists():
        raise FileExistsError("new immutable timing root required")
    gate = timing.base.common.read(root.parent / "analysis/pair-decision.json")
    if not gate["bitwise_and_persistent_storage_gate"]:
        raise ValueError("actual pair reuse parity/storage required")
    p = timing.base.common.read(PARENT / "paired-timing/plan.json")
    original = timing.base.common.read(ORIGINAL / "plan.json")
    p["controls"]["references"] = original["controls"]["references"]
    p["same_precision_references"] = {
        str(i): str(PARENT / f"model/{i}/selective_unfused.logits.pt") for i in (0, 1)
    }
    paths = [
        PARENT / "paired-timing/plan.json",
        ORIGINAL / "plan.json",
        root.parent / "analysis/pair-decision.json",
    ]
    paths += [Path(v["storage"]) for v in p["controls"]["references"].values()]
    paths += [Path(v) for v in p["same_precision_references"].values()]
    p["bound_files"].update({str(path): sha(path) for path in paths})
    p.update(
        source_sha=clean_source(timing.SOURCE),
        study="direct_common_pair_reuse_storage_history_v1",
        measurement_type="consumed_direct_three_arm_paired_timing",
        arms=ARMS,
        arm_policies=POLICIES,
        timing_family_comparisons=12,
        precision_equivalence_required=False,
        profile_first_warmup=True,
        profile_is_excluded_from_timed_trials=True,
        no_multiplication_of_ratios_from_separate_jobs=True,
        fresh_or_paper_campaign_authorized=False,
        independent_quality_validation_authorized=False,
    )
    with patch.object(timing, "ARMS", ARMS):
        p["timing_schedules"] = {str(i): timing.schedule(p["seed"] + i) for i in (0, 1)}
    root.mkdir()
    (root / "logs").mkdir()
    write_once(root / "plan.json", p)
    timing.base.common.WORKER_WRAPPER = (
        timing.SOURCE / "mosaickv/slurm/paired_history_source_timing.sbatch"
    )
    cpu = timing.base.common.submit(root, "cpu")
    write_once(
        root / "submission.json",
        dict(cpu=cpu, follow=timing.base.common.submit(root, "after_cpu", cpu)),
    )


def analyze(root, p):
    three.analyze(root, p)
    result = timing.base.common.read(root / "analysis/decision.json")
    actual = [
        r for r in result["results"] if r["baseline"] == ARMS[1] and r["candidate"] == ARMS[2]
    ]
    write_once(
        root / "analysis/common_control-decision.json",
        dict(
            complete=result["complete"],
            both_methods_receive_common_optimization=True,
            correct_comparison_baseline=ARMS[1],
            correct_comparison_candidate=ARMS[2],
            strict_latency_pass=result["complete"]
            and len(actual) == 2
            and all(r["end_to_end_seconds"]["equal_or_faster"] for r in actual),
            results=actual,
            independent_quality_established=False,
            initial_memory_advantage_NOT_established=True,
            fresh_cohort_authorized=False,
        ),
    )


if __name__ == "__main__":
    timing.ARMS = three.ARMS = ARMS
    three.COMPARISONS = [(ARMS[0], ARMS[1]), (ARMS[0], ARMS[2]), (ARMS[1], ARMS[2])]
    timing.base.common.WORKER_WRAPPER = (
        timing.SOURCE / "mosaickv/slurm/paired_history_source_timing.sbatch"
    )
    timing.base.EXTRA_TESTS = (
        *timing.base.EXTRA_TESTS,
        "test_mixed_history_projection.py",
        "test_fused_history_unpack.py",
        "test_paired_history_source.py",
        "test_history_three_arm.py",
    )
    timing.prepare, timing.analyze = prepare, analyze
    timing.main()
