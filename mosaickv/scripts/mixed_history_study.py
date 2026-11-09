#!/usr/bin/env python3
"""Bounded same-precision mixed-history reader gate before clean timing."""

from pathlib import Path

import early_scale_header_study as early
import exact_scale_replay_study as base
import selective_history_study as precision

from mosaickv.evaluation.fox_campaign import write_once

SOURCE = base.SOURCE
PARENT = Path("/scratch/djy8hg/runs/evicastkv/selective-history64-20261108-v1")
TIMING = Path("/scratch/djy8hg/runs/evicastkv/header-history-timing-20261108-v1")
ARMS = ["selective_unfused", "selective_mixed"]
base.ARMS = ARMS
base.common.WORKER_WRAPPER = SOURCE / "mosaickv/slurm/mixed_history.sbatch"
base.EXTRA_TESTS = (*base.EXTRA_TESTS, "test_mixed_history_projection.py")
original_operator, original_analyze = base.operator, base.analyze


def prepare(root):
    if not base.common.read(TIMING / "analysis/decision.json")["complete"]:
        raise ValueError("complete predecessor paired timing required")
    early.PARENT = PARENT
    early.prepare(
        root,
        dependencies=(TIMING / "plan.json", TIMING / "analysis/decision.json"),
        overrides=dict(
            study="single_projection_same_precision_history_v1",
            measurement_type="consumed_same_precision_history_execution_parity",
            arms=ARMS,
            mixed_history_projection="shared source/output workspace; original 32KiB repairs cap",
        ),
    )


def operator(root, p, folder, i):
    from mosaickv.evaluation.mixed_history_projection import gpu_controls

    write_once(
        folder / "mixed-operator.json",
        dict(
            passed=True,
            controls=gpu_controls(),
            pretrained_weights_loaded=False,
        ),
    )
    original_operator(root, p, folder, i)


def model(root, p, folder, i):
    import torch

    from mosaickv.evaluation.early_scale_headers import use as early_headers
    from mosaickv.evaluation.exact_scale_replay import use as late_headers
    from mosaickv.evaluation.history_source_precision import use as history
    from mosaickv.evaluation.mixed_history_projection import use as mixed

    reference = []

    def context(stack, arm):
        return dict(
            early=stack.enter_context(early_headers()),
            late=stack.enter_context(late_headers()),
            selective_history=stack.enter_context(
                history(feature_group=32, exact_recent_tokens=64)
            ),
            mixed_projection=stack.enter_context(mixed()) if arm == ARMS[1] else None,
        )

    def compare(actual, parent, plan, arm):
        proof = precision.fidelity(actual, parent, plan, precision.ARMS[1])
        if arm == ARMS[0]:
            reference.append(actual.clone())
        if not reference:
            raise ValueError("unfused same-precision reference required")
        proof["all_128_words_equal_unfused"] = bool(torch.equal(actual, reference[0]))
        proof["passed"] &= proof["all_128_words_equal_unfused"]
        return proof

    base.common.extra_model_context, base.common.assess_fidelity = context, compare
    base.common.model(root, p, folder, i)


def analyze(root, p):
    original_analyze(root, p)
    proof = base.common.read(root / "analysis/decision.json")
    eligible = proof["complete"]
    pairs = []
    for i in (0, 1):
        rows = [r for r in proof["rows"] if r["input"] == i]
        if len(rows) != 2 or any(r["status"] != "completed" for r in rows):
            eligible = False
            continue
        a, b = rows
        info = b.get("common_header", {}).get("mixed_projection") or {}
        good = (
            b["numerically_admissible"]
            and b["fidelity"].get("all_128_words_equal_unfused")
            and b["initial_GPU_KV_bytes"] == a["initial_GPU_KV_bytes"]
            and b["final_GPU_KV_bytes"] <= a["final_GPU_KV_bytes"]
            and info.get("mixed_constructions") == 192
            and info.get("no_extra_persistent_source_copy") is True
        )
        eligible &= good
        pairs.append(
            dict(
                input=i,
                same_precision_and_storage_gate=bool(good),
                projection=b.get("common_header", {}).get("mixed_projection"),
            )
        )
    write_once(
        root / "analysis/mixed-decision.json",
        dict(
            complete=proof["complete"],
            same_precision_storage_gate=bool(eligible),
            pairs=pairs,
            initial_memory_superiority_established=False,
            clean_latency_established=False,
            independent_quality_established=False,
            fresh_cohort_authorized=False,
        ),
    )
    if eligible:
        from mixed_history_timing import prepare as prepare_timing

        prepare_timing(root / "paired-timing")


if __name__ == "__main__":
    base.prepare, base.operator, base.model, base.analyze = prepare, operator, model, analyze
    base.main()
