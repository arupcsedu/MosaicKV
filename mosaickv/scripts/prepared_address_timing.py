#!/usr/bin/env python3
"""Direct timing with common optimizations and finite external-baseline dependency."""

from pathlib import Path
from unittest.mock import patch

import header_history_timing as timing
import history_three_arm as three

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

PARENT = Path("/scratch/djy8hg/runs/evicastkv/mixed-history-projection-20261108-v1")
ORIGINAL = Path("/scratch/djy8hg/runs/evicastkv/selective-history64-20261108-v1")
BASELINES = Path(
    "/scratch/djy8hg/runs/evicastkv/private-prefill-scores-20261108-v1/compatible-baselines"
)
ARMS = ["storage_common_address_dispatch", "mixed_tensor_arguments", "mixed_address_arguments"]
POLICIES = {
    ARMS[0]: dict(
        history=False,
        mixed=False,
        paired_source=True,
        private_prefill_scores=True,
        prepared_addresses=True,
    ),
    ARMS[1]: dict(
        history=True, mixed=True, fused_unpack=True, paired_source=True, private_prefill_scores=True
    ),
    ARMS[2]: dict(
        history=True,
        mixed=True,
        fused_unpack=True,
        paired_source=True,
        private_prefill_scores=True,
        prepared_addresses=True,
    ),
}
original_cpu = timing.base.cpu


def prepare(root):
    if root.exists():
        raise FileExistsError("new immutable timing root required")
    if not timing.base.common.read(root.parent / "analysis/address-decision.json")[
        "bitwise_and_storage_gate"
    ]:
        raise ValueError("complete address dispatch parity required")
    p = timing.base.common.read(PARENT / "paired-timing/plan.json")
    original = timing.base.common.read(ORIGINAL / "plan.json")
    p["controls"]["references"] = original["controls"]["references"]
    p["same_precision_references"] = {
        str(i): str(PARENT / f"model/{i}/selective_unfused.logits.pt") for i in (0, 1)
    }
    dependency = timing.base.common.read(BASELINES / "model-submission.json")["analyzer"]
    paths = [
        PARENT / "paired-timing/plan.json",
        ORIGINAL / "plan.json",
        root.parent / "analysis/address-decision.json",
        BASELINES / "plan.json",
        BASELINES / "model-submission.json",
    ]
    paths += [Path(v["storage"]) for v in p["controls"]["references"].values()]
    paths += [Path(v) for v in p["same_precision_references"].values()]
    p["bound_files"].update({str(path): sha(path) for path in paths})
    p.update(
        source_sha=clean_source(timing.SOURCE),
        study="common_prepared_address_storage_history_timing_v1",
        measurement_type="consumed_direct_three_arm_paired_timing",
        arms=ARMS,
        arm_policies=POLICIES,
        timing_family_comparisons=12,
        precision_equivalence_required=False,
        profile_first_warmup=True,
        profile_is_excluded_from_timed_trials=True,
        no_multiplication_of_ratios_from_separate_jobs=True,
        external_baseline_analysis_dependency=dependency,
        fresh_or_paper_campaign_authorized=False,
        independent_quality_validation_authorized=False,
    )
    with patch.object(timing, "ARMS", ARMS):
        p["timing_schedules"] = {str(i): timing.schedule(p["seed"] + i) for i in (0, 1)}
    root.mkdir()
    (root / "logs").mkdir()
    write_once(root / "plan.json", p)
    timing.base.common.WORKER_WRAPPER = (
        timing.SOURCE / "mosaickv/slurm/prepared_address_timing.sbatch"
    )
    cpu = timing.base.common.submit(root, "cpu", after=dependency)
    write_once(
        root / "submission.json",
        dict(
            cpu=cpu,
            follow=timing.base.common.submit(root, "after_cpu", cpu),
            external_afterany=dependency,
            no_polling=True,
        ),
    )


def cpu(root, p):
    path = BASELINES / "analysis/decision.json"
    if not path.exists() or not timing.base.common.read(path)["complete"]:
        write_once(
            root / "cpu.json",
            dict(
                status="failed",
                reason="common-baseline analysis missing/negative; no GPU timing permitted",
            ),
        )
        return
    write_once(
        root / "external-baseline-seal.json",
        dict(
            path=str(path),
            sha256=sha(path),
            source_plan_sha=sha(BASELINES / "plan.json"),
            producer_job_dependency=p["external_baseline_analysis_dependency"],
        ),
    )
    original_cpu(root, p)


if __name__ == "__main__":
    timing.ARMS = three.ARMS = ARMS
    three.COMPARISONS = [(ARMS[0], ARMS[1]), (ARMS[0], ARMS[2]), (ARMS[1], ARMS[2])]
    timing.base.common.WORKER_WRAPPER = (
        timing.SOURCE / "mosaickv/slurm/prepared_address_timing.sbatch"
    )
    timing.base.EXTRA_TESTS = (
        *timing.base.EXTRA_TESTS,
        "test_mixed_history_projection.py",
        "test_fused_history_unpack.py",
        "test_paired_history_source.py",
        "test_prepared_address_dispatch.py",
        "test_private_prefill_scores.py",
        "test_history_three_arm.py",
    )
    timing.base.cpu = cpu
    timing.prepare, timing.analyze = prepare, three.analyze
    timing.main()
