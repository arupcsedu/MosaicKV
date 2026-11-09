#!/usr/bin/env python3
"""One conditional paired timing follow-up after bitwise same-precision parity."""

import json

import header_history_timing as timing

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

timing.ARMS = ["selective_unfused", "selective_mixed"]
timing.base.common.WORKER_WRAPPER = timing.SOURCE / "mosaickv/slurm/mixed_history_timing.sbatch"


def prepare(root):
    if root.exists():
        raise FileExistsError("one write-once conditional timing root required")
    parent = root.parent
    gate = timing.base.common.read(parent / "analysis/mixed-decision.json")
    if not gate["complete"] or not gate["same_precision_storage_gate"]:
        raise ValueError("unchanged mixed-reader parity/storage gate must pass first")
    p = timing.base.common.read(parent / "plan.json")
    for path in (parent / "plan.json", parent / "analysis/mixed-decision.json"):
        p["bound_files"][str(path)] = sha(path)
    for i in (0, 1):
        path = parent / f"model/{i}/selective_unfused.logits.pt"
        p["bound_files"][str(path)] = sha(path)
        p["controls"]["references"][str(i)]["storage"] = str(path)
    p.update(
        source_sha=clean_source(timing.SOURCE),
        study="same_precision_mixed_reader_paired_timing_v1",
        measurement_type="consumed_same_precision_randomized_paired_timing",
        timing_warmups=5,
        timing_repetitions=20,
        timing_bootstrap_replicates=10000,
        timing_worker_seconds=3300,
        timing_history_on_both=True,
        timing_mixed_candidate=True,
        precision_equivalence_required=True,
        prior_negative_memory_gate_NOT_overridden=True,
        independent_quality_validation_authorized=False,
        fresh_or_paper_campaign_authorized=False,
    )
    p["timing_schedules"] = {str(i): timing.schedule(p["seed"] + i) for i in (0, 1)}
    root.mkdir()
    (root / "logs").mkdir()
    write_once(root / "plan.json", p)
    cpu = timing.base.common.submit(root, "cpu")
    write_once(
        root / "submission.json",
        dict(cpu=cpu, follow=timing.base.common.submit(root, "after_cpu", cpu)),
    )
    print(json.dumps(timing.base.common.read(root / "submission.json")), flush=True)


if __name__ == "__main__":
    timing.prepare = prepare
    timing.main()
