#!/usr/bin/env python3
"""Consumed-input mixed-source launch fusion: operator -> parity -> clean timing."""

from pathlib import Path

import early_scale_header_study as early
import exact_scale_replay_study as base
import selective_history_study as precision

from mosaickv.evaluation.fox_campaign import write_once

SOURCE = base.SOURCE
PARENT = Path("/scratch/djy8hg/runs/evicastkv/mixed-history-projection-20261108-v1")
ARMS = ["mixed_source_reference", "mixed_source_single_launch"]
base.ARMS = ARMS
base.common.WORKER_WRAPPER = SOURCE / "mosaickv/slurm/fused_history_unpack.sbatch"
base.EXTRA_TESTS = (
    *base.EXTRA_TESTS,
    "test_mixed_history_projection.py",
    "test_fused_history_unpack.py",
    "test_fused_history_unpack_study.py",
)
original_analyze = base.analyze


def prepare(root):
    decision = base.common.read(PARENT / "analysis/mixed-decision.json")
    if not decision["same_precision_storage_gate"]:
        raise ValueError("completed original mixed precision equivalence required")
    early.PARENT = PARENT
    early.prepare(
        root,
        dependencies=(PARENT / "analysis/mixed-decision.json",),
        overrides=dict(
            study="same_precision_mixed_source_unpack_fusion_v1",
            measurement_type="consumed_exact_decoder_launch_fusion_parity",
            arms=ARMS,
            mixed_history_projection="same GEMM/RoPE/repairs; fuse source decode launches only",
        ),
    )


def operator(root, p, folder, i):
    from compact_hybrid_study import manifest

    from mosaickv.evaluation.fused_history_unpack import gpu_controls

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

    reference = []

    def context(stack, arm):
        return dict(
            early=stack.enter_context(early_headers()),
            late=stack.enter_context(late_headers()),
            selective_history=stack.enter_context(
                history(feature_group=32, exact_recent_tokens=64)
            ),
            mixed_projection=stack.enter_context(mixed()),
            fused_unpack=stack.enter_context(fused()) if arm == ARMS[1] else None,
        )

    def compare(actual, parent, plan, arm):
        proof = precision.fidelity(actual, parent, plan, precision.ARMS[1])
        if arm == ARMS[0]:
            reference.append(actual.clone())
        if not reference:
            raise ValueError("original mixed history reference required")
        proof["all_128_logits_bitwise_equal_original_mixed"] = bool(
            torch.equal(actual, reference[0])
        )
        proof["passed"] &= proof["all_128_logits_bitwise_equal_original_mixed"]
        return proof

    base.common.extra_model_context, base.common.assess_fidelity = context, compare
    base.common.model(root, p, folder, i)


def analyze(root, p):
    original_analyze(root, p)
    report = base.common.read(root / "analysis/decision.json")
    pairs = []
    eligible = report["complete"]
    for i in (0, 1):
        rows = [r for r in report["rows"] if r["input"] == i]
        if len(rows) != 2 or any(r["status"] != "completed" for r in rows):
            eligible = False
            continue
        a, b = rows
        runtime = b.get("common_header", {}).get("fused_unpack") or {}
        good = bool(
            b["fidelity"].get("all_128_logits_bitwise_equal_original_mixed")
            and a["initial_GPU_KV_bytes"] == b["initial_GPU_KV_bytes"]
            and a["final_GPU_KV_bytes"] == b["final_GPU_KV_bytes"]
            and runtime.get("prepared", 0) >= 192
            and runtime.get("reused", 0) > 0
            and runtime.get("additional_GPU_storage_bytes") == 0
        )
        eligible &= good
        pairs.append(dict(input=i, gate=good, runtime=runtime))
    write_once(
        root / "analysis/unpack-decision.json",
        dict(
            complete=report["complete"],
            bitwise_and_persistent_storage_gate=bool(eligible),
            pairs=pairs,
            initial_memory_advantage_over_common_storage=False,
            independent_quality_established=False,
            complete_superiority_established=False,
            fresh_cohort_authorized=False,
        ),
    )
    if eligible:
        from fused_history_unpack_timing import prepare as timing_prepare

        timing_prepare(root / "paired-timing")


if __name__ == "__main__":
    base.prepare, base.operator, base.model, base.analyze = prepare, operator, model, analyze
    base.main()
