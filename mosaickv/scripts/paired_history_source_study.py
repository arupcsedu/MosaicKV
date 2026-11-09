#!/usr/bin/env python3
"""Preserve all checks but reuse exact source in immediate construction K/V pairs."""

from pathlib import Path

import early_scale_header_study as early
import exact_scale_replay_study as base
import selective_history_study as precision

from mosaickv.evaluation.fox_campaign import write_once

SOURCE = base.SOURCE
PARENT = Path("/scratch/djy8hg/runs/evicastkv/fused-history-unpack-20261108-v1")
ARMS = ["fused_mixed_reference", "fused_mixed_immediate_pair_reuse"]
base.ARMS = ARMS
base.common.WORKER_WRAPPER = SOURCE / "mosaickv/slurm/paired_history_source.sbatch"
base.EXTRA_TESTS = (
    *base.EXTRA_TESTS,
    "test_fused_history_unpack.py",
    "test_mixed_history_projection.py",
    "test_native_history_feasibility.py",
    "test_native_history_reader.py",
    "test_paired_history_source.py",
)
original_analyze = base.analyze


def prepare(root):
    if not base.common.read(PARENT / "analysis/unpack-decision.json")[
        "bitwise_and_persistent_storage_gate"
    ]:
        raise ValueError("complete fused unpack parity/storage required")
    early.PARENT = PARENT
    early.prepare(
        root,
        dependencies=(PARENT / "analysis/unpack-decision.json",),
        overrides=dict(
            study="immediate_history_KV_pair_source_reuse_v1",
            measurement_type="consumed_common_immediate_pair_reuse_parity",
            arms=ARMS,
            mixed_history_projection=(
                "same precision, all original projections/checks; reuse source within pair only"
            ),
        ),
    )


def operator(root, p, folder, i):
    from compact_hybrid_study import manifest

    from mosaickv.evaluation.fused_history_unpack import use as fused
    from mosaickv.evaluation.mixed_history_projection import gpu_controls
    from mosaickv.evaluation.paired_history_source import use as pairs

    manifest(p, base.sha(root / "plan.json"), folder, i, gpu=True)
    with fused(), pairs() as report:
        controls = gpu_controls()
    if not report["reused_sources"] or not report["paired_transactions"]:
        raise RuntimeError("no actual immediate K/V source reuse")
    write_once(
        folder / "result.json",
        dict(
            status="completed",
            passed=True,
            controls=controls,
            source_reuse=report,
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
            paired_source=stack.enter_context(pairs()) if arm == ARMS[1] else None,
        )

    def compare(actual, parent, plan, arm):
        proof = precision.fidelity(actual, parent, plan, precision.ARMS[1])
        if arm == ARMS[0]:
            reference.append(actual.clone())
        if not reference:
            raise ValueError("original exact source/projection reference required")
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
        info = b.get("common_header", {}).get("paired_source") or {}
        good = bool(
            b["fidelity"].get("all_128_logits_bitwise_equal_original")
            and a["initial_GPU_KV_bytes"] == b["initial_GPU_KV_bytes"]
            and a["final_GPU_KV_bytes"] == b["final_GPU_KV_bytes"]
            and info.get("reused_sources", 0) > 0
            and info.get("added_GPU_storage_bytes") == 0
        )
        eligible &= good
        pairs.append(dict(input=i, gate=good, reuse=info))
    write_once(
        root / "analysis/pair-decision.json",
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
        from paired_history_source_timing import prepare as timing_prepare

        timing_prepare(root / "paired-timing")


if __name__ == "__main__":
    base.prepare, base.operator, base.model, base.analyze = prepare, operator, model, analyze
    base.main()
