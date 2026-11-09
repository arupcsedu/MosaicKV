#!/usr/bin/env python3
"""Unchanged three-pass mixed verification with one pre-decode epoch commit."""

from pathlib import Path

import early_scale_header_study as early
import exact_scale_replay_study as base
import selective_history_study as precision

from mosaickv.evaluation.fox_campaign import write_once

SOURCE = base.SOURCE
PARENT = Path("/scratch/djy8hg/runs/evicastkv/paired-history-source-20261108-v1")
ARMS = ["immediate_pair_reference", "immediate_pair_epoch_verification"]
base.ARMS = ARMS
base.common.WORKER_WRAPPER = SOURCE / "mosaickv/slurm/mixed_verification_barrier.sbatch"
base.EXTRA_TESTS = (
    *base.EXTRA_TESTS,
    "test_fused_history_unpack.py",
    "test_mixed_history_projection.py",
    "test_paired_history_source.py",
    "test_mixed_verification_barrier.py",
)
original_analyze = base.analyze


def prepare(root):
    if not base.common.read(PARENT / "analysis/pair-decision.json")[
        "bitwise_and_persistent_storage_gate"
    ]:
        raise ValueError("complete immediate-pair parity/storage required")
    early.PARENT = PARENT
    early.prepare(
        root,
        dependencies=(PARENT / "analysis/pair-decision.json",),
        overrides=dict(
            study="unchanged_mixed_predicates_epoch_commit_v1",
            measurement_type="consumed_bitwise_verification_batching_ablation",
            arms=ARMS,
            mixed_history_projection=(
                "same three original K/V verification passes; one epoch commit synchronization"
            ),
        ),
    )


def operator(root, p, folder, i):
    from compact_hybrid_study import manifest

    from mosaickv.evaluation.mixed_verification_barrier import gpu_controls

    manifest(p, base.sha(root / "plan.json"), folder, i, gpu=True)
    write_once(
        folder / "result.json",
        dict(
            status="completed",
            passed=True,
            controls=gpu_controls(),
            pretrained_weights_loaded=False,
        ),
    )


def model(root, p, folder, i):
    import torch

    from mosaickv.evaluation.early_scale_headers import use as early_headers
    from mosaickv.evaluation.exact_scale_replay import use as late_headers
    from mosaickv.evaluation.fused_history_unpack import use as fused
    from mosaickv.evaluation.history_source_precision import use as history
    from mosaickv.evaluation.mixed_history_projection import use as mixed
    from mosaickv.evaluation.mixed_verification_barrier import use as barrier
    from mosaickv.evaluation.paired_history_source import use as pairs

    reference = []

    def context(stack, arm):
        return dict(
            early=stack.enter_context(early_headers()),
            late=stack.enter_context(late_headers()),
            selective_history=stack.enter_context(
                history(feature_group=32, exact_recent_tokens=64)
            ),
            mixed_projection=stack.enter_context(mixed()),
            fused_unpack=stack.enter_context(fused()),
            paired_source=stack.enter_context(pairs()),
            verification_barrier=stack.enter_context(barrier()) if arm == ARMS[1] else None,
        )

    def compare(actual, parent, plan, arm):
        proof = precision.fidelity(actual, parent, plan, precision.ARMS[1])
        if arm == ARMS[0]:
            reference.append(actual.clone())
        if not reference:
            raise ValueError("synchronous original verification reference required")
        proof["all_128_logits_bitwise_equal_original"] = bool(torch.equal(actual, reference[0]))
        proof["passed"] &= proof["all_128_logits_bitwise_equal_original"]
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
        info = b.get("common_header", {}).get("verification_barrier") or {}
        good = bool(
            b["fidelity"].get("all_128_logits_bitwise_equal_original")
            and a["initial_GPU_KV_bytes"] == b["initial_GPU_KV_bytes"]
            and a["final_GPU_KV_bytes"] == b["final_GPU_KV_bytes"]
            and info.get("commit_flushes") == 6
            and info.get("complete_KV_verification_flags") == 1152
            and info.get("maximum_epoch_flags") == 192
        )
        eligible &= good
        pairs.append(dict(input=i, gate=good, verification=info))
    write_once(
        root / "analysis/verification-decision.json",
        dict(
            complete=proof["complete"],
            bitwise_and_persistent_storage_gate=bool(eligible),
            pairs=pairs,
            fresh_cohort_authorized=False,
            independent_quality_established=False,
            complete_superiority_established=False,
        ),
    )
    if eligible:
        from mixed_verification_barrier_timing import prepare as timing_prepare

        timing_prepare(root / "paired-timing")


if __name__ == "__main__":
    base.prepare, base.operator, base.model, base.analyze = prepare, operator, model, analyze
    base.main()
