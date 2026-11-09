#!/usr/bin/env python3
"""Move exact headers off the prefill peak, on the same consumed inputs only."""

import json
from pathlib import Path

import exact_scale_replay_study as base

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = base.SOURCE
PARENT = Path("/scratch/djy8hg/runs/evicastkv/exact-scale-replay-20261108-v1")
base.common.WORKER_WRAPPER = SOURCE / "mosaickv/slurm/early_scale_headers.sbatch"
base.EXTRA_TESTS = ("test_early_scale_headers.py", "test_async_text_handoff.py")


def prepare(root, *, overrides=None, dependencies=()):
    if root.exists():
        raise FileExistsError("new immutable root required")
    gate = base.common.read(PARENT / "analysis/decision.json")
    if not gate["complete"]:
        raise ValueError("complete predecessor numerical and ownership evidence required")
    p = base.common.read(PARENT / "plan.json")
    for name in ("plan.json", "analysis/decision.json"):
        path = PARENT / name
        p["bound_files"][str(path)] = sha(path)
    p.update(
        source_sha=clean_source(SOURCE),
        study="early_exact_header_lifetime_v1",
        measurement_type="consumed_early_header_lifetime_ablation",
        earlier_peak_failure_preserved=True,
        early_headers="completed_async_publication_only; restored_before_downstream_reads",
        future_prototypes_and_history_blocked=True,
        independent_quality_validation_authorized=False,
        fresh_or_paper_campaign_authorized=False,
    )
    if overrides:
        if not overrides.keys() <= {
            "study",
            "measurement_type",
            "arms",
            "selective_history",
            "mixed_history_projection",
        }:
            raise ValueError("narrow consumed-study configuration overrides only")
        p.update(overrides)
    for path in dependencies:
        p["bound_files"][str(path)] = sha(path)
    root.mkdir()
    (root / "logs").mkdir()
    write_once(root / "plan.json", p)
    cpu = base.common.submit(root, "cpu")
    write_once(
        root / "submission.json", dict(cpu=cpu, follow=base.common.submit(root, "after_cpu", cpu))
    )
    print(json.dumps(base.common.read(root / "submission.json")), flush=True)


def model(root, p, folder, i):
    from mosaickv.evaluation.early_scale_headers import use as early
    from mosaickv.evaluation.exact_scale_replay import use as late

    def context(stack, arm):
        if arm == base.ARMS[1]:
            return dict(early=stack.enter_context(early()), late=stack.enter_context(late()))
        return None

    base.common.extra_model_context = context
    base.common.model(root, p, folder, i)


if __name__ == "__main__":
    base.prepare, base.model = prepare, model
    base.main()
