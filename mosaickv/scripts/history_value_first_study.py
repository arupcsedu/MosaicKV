#!/usr/bin/env python3
"""Finite schedule-only history V-before-prompt V verification on consumed inputs."""

from contextlib import ExitStack
from pathlib import Path
from time import perf_counter

import early_scale_header_study as early
import exact_scale_replay_study as base
import header_history_timing as statistics
import selective_history_study as precision

from mosaickv.evaluation.fox_campaign import write_once

SOURCE = base.SOURCE
PARENT = Path("/scratch/djy8hg/runs/evicastkv/private-prefill-scores-20261108-v1")
ARMS = ["original_value_order", "history_value_first"]
base.ARMS = ARMS
base.common.WORKER_WRAPPER = SOURCE / "mosaickv/slurm/history_value_first.sbatch"
base.EXTRA_TESTS = (
    *base.EXTRA_TESTS,
    "test_history_value_first.py",
    "test_history_value_first_study.py",
    "test_paired_history_source.py",
)
original_analyze = base.analyze


def prepare(root):
    if not base.common.read(PARENT / "analysis/private-score-decision.json")[
        "bitwise_and_peak_gate"
    ]:
        raise ValueError("complete preceding pretrained numerical and peak evidence required")
    early.PARENT = PARENT
    early.prepare(
        root,
        dependencies=(PARENT / "analysis/private-score-decision.json",),
        overrides=dict(
            study="same_storage_history_value_before_prompt_value_v1",
            measurement_type="consumed_reconstruction_order_only_ablation",
            arms=ARMS,
            mixed_history_projection="same precision/projector; early history V source reuse",
        ),
    )


def operator(root, p, folder, i):
    import numpy as np
    import torch
    from compact_hybrid_study import manifest

    from mosaickv.evaluation.fused_history_unpack import use as fused
    from mosaickv.evaluation.history_value_first import use
    from mosaickv.evaluation.history_value_first_controls import build
    from mosaickv.evaluation.paired_history_source import use as pairs

    manifest(p, base.sha(root / "plan.json"), folder, i, gpu=True)
    rows, correct = [], True
    rng = np.random.default_rng(p["seed"] + i)
    with torch.inference_mode():
        for kind, n in (("native", 64), ("mixed", 112)):
            layer, state, scope = build(kind, n)
            try:
                with ExitStack() as stack:
                    scope(stack)
                    stack.enter_context(fused())
                    stack.enter_context(pairs())
                    expected = []
                    with use(False):
                        for c in (0, 1):
                            layer.materialize_component(c)
                            expected.append(layer.leased[0].clone())
                    for rep in range(-5, 20):
                        for flag in rng.permutation([False, True]):
                            with use(bool(flag)) as report:
                                torch.cuda.synchronize()
                                a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                                a.record()
                                start = perf_counter()
                                for _ in range(5):
                                    for c in (0, 1):
                                        layer.materialize_component(c)
                                host = (perf_counter() - start) / 5
                                b.record()
                                torch.cuda.synchronize()
                                stream = a.elapsed_time(b) / 5000
                            equal = torch.equal(
                                layer.leased[0].view(torch.int16), expected[1].view(torch.int16)
                            )
                            correct &= bool(equal)
                            if flag:
                                correct &= report["skipped_source_unpacks"] == 5
                            rows.append(
                                dict(
                                    kind=kind,
                                    rows=n,
                                    repetition=rep,
                                    warmup=rep < 0,
                                    candidate=bool(flag),
                                    host_seconds=host,
                                    stream_seconds=stream,
                                    bitwise=bool(equal),
                                    schedule=report,
                                )
                            )
                    with use(True):
                        for c in (0, 1):
                            layer.materialize_component(c)
                            correct &= torch.equal(
                                layer.leased[0].view(torch.int16), expected[c].view(torch.int16)
                            )
            finally:
                state.engine.close()
    stats = []
    for kind in ("native", "mixed"):
        values = [
            [
                r["host_seconds"]
                for r in rows
                if r["kind"] == kind and not r["warmup"] and r["candidate"] == flag
            ]
            for flag in (False, True)
        ]
        stats.append(
            dict(
                kind=kind,
                host=statistics.paired(
                    values, seed=p["seed"] + i, replicates=10000, family_comparisons=4
                ),
            )
        )
    passed = bool(correct and all(r["host"]["equal_or_faster"] for r in stats))
    write_once(
        folder / "result.json",
        dict(
            status="completed",
            passed=passed,
            correctness_passed=bool(correct),
            trials=rows,
            paired=stats,
            synthetic_operator_NOT_pretrained_quality=True,
            original_GEMMs_and_positions_unchanged=True,
        ),
    )


def contexts(stack, p, enabled):
    from mosaickv.evaluation.early_scale_headers import use as early_headers
    from mosaickv.evaluation.exact_scale_replay import use as late_headers
    from mosaickv.evaluation.fused_history_unpack import use as fused
    from mosaickv.evaluation.history_source_precision import use as history
    from mosaickv.evaluation.history_value_first import use as schedule
    from mosaickv.evaluation.mixed_history_projection import use as mixed
    from mosaickv.evaluation.paired_history_source import use as pairs
    from mosaickv.evaluation.tiled_softmax import use as tile

    return dict(
        early=stack.enter_context(early_headers()),
        late=stack.enter_context(late_headers()),
        selective_history=stack.enter_context(history(feature_group=32, exact_recent_tokens=64)),
        mixed_projection=stack.enter_context(mixed()),
        fused_unpack=stack.enter_context(fused()),
        paired_source=stack.enter_context(pairs()),
        private_scores=stack.enter_context(tile(rows=p["tiled_softmax_rows"], inplace=True)),
        history_value_first=stack.enter_context(schedule(enabled)),
    )


def model(root, p, folder, i):
    import torch

    references = []

    def compare(actual, parent, plan, arm):
        proof = precision.fidelity(actual, parent, plan, precision.ARMS[1])
        if arm == ARMS[0]:
            references.append(actual.clone())
        proof["all_128_logits_bitwise_equal_original"] = bool(
            references and torch.equal(actual, references[0])
        )
        proof["passed"] &= proof["all_128_logits_bitwise_equal_original"]
        return proof

    base.common.extra_model_context = lambda stack, arm: contexts(stack, p, arm == ARMS[1])
    base.common.assess_fidelity = compare
    base.common.model(root, p, folder, i)


def analyze(root, p):
    original_analyze(root, p)
    proof = base.common.read(root / "analysis/decision.json")
    pairs = []
    for i in (0, 1):
        rows = [r for r in proof["rows"] if r["input"] == i]
        passed = False
        if len(rows) == 2 and all(r["status"] == "completed" for r in rows):
            a, b = rows
            report = b["common_header"]["history_value_first"]
            passed = bool(
                b["fidelity"]["all_128_logits_bitwise_equal_original"]
                and a["initial_GPU_KV_bytes"] == b["initial_GPU_KV_bytes"]
                and a["final_GPU_KV_bytes"] == b["final_GPU_KV_bytes"]
                and report["early_values"] >= 32
                and report["early_values"] == report["skipped_source_unpacks"]
                and report["early_values"] == report["skipped_duplicate_history_fills"]
                and report["additional_GPU_storage_bytes"] == 0
            )
        pairs.append(dict(input=i, gate=passed))
    eligible = proof["complete"] and all(r["gate"] for r in pairs)
    write_once(
        root / "analysis/value-order-decision.json",
        dict(
            complete=proof["complete"],
            bitwise_and_storage_gate=eligible,
            pairs=pairs,
            independent_quality_established=False,
            complete_superiority_established=False,
            fresh_cohort_authorized=False,
        ),
    )
    if eligible:
        from history_value_first_baselines import prepare as baseline_prepare

        baseline_prepare(root / "compatible-baselines")


if __name__ == "__main__":
    base.prepare, base.operator, base.model, base.analyze = prepare, operator, model, analyze
    base.main()
