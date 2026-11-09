#!/usr/bin/env python3
"""Pinned cached-kernel argument transport, operator improvement before model work."""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import early_scale_header_study as early
import exact_scale_replay_study as base
import header_history_timing as statistics
import selective_history_study as precision

from mosaickv.evaluation.fox_campaign import write_once

SOURCE = base.SOURCE
PARENT = Path("/scratch/djy8hg/runs/evicastkv/private-prefill-scores-20261108-v1")
ARMS = ["tensor_argument_reference", "prepared_address_arguments"]
base.ARMS = ARMS
base.common.WORKER_WRAPPER = SOURCE / "mosaickv/slurm/prepared_address.sbatch"
base.EXTRA_TESTS = (
    *base.EXTRA_TESTS,
    "test_prepared_address_dispatch.py",
    "test_prepared_address_study.py",
    "test_prepared_patch_dispatch.py",
    "test_fused_history_unpack.py",
    "test_mixed_history_projection.py",
    "test_private_prefill_scores.py",
)
original_analyze = base.analyze


def prepare(root):
    from mosaickv.evaluation.prepared_address_dispatch import verify_driver

    verify_driver()
    if not base.common.read(PARENT / "analysis/private-score-decision.json")[
        "bitwise_and_peak_gate"
    ]:
        raise ValueError("complete private prefill/FullKV gate required")
    early.PARENT = PARENT
    early.prepare(
        root,
        dependencies=(PARENT / "analysis/private-score-decision.json",),
        overrides=dict(
            study="same_binary_prepared_address_transport_v1",
            measurement_type="consumed_same_binary_argument_transport",
            arms=ARMS,
            mixed_history_projection=(
                "same binary/geometry; live private CUDA buffers; avoid redundant pointer queries"
            ),
        ),
    )


def operator(root, p, folder, i):
    import numpy as np
    import torch
    from compact_hybrid_study import manifest

    from mosaickv.evaluation.exact_storage import encode
    from mosaickv.evaluation.fused_history_unpack import launch
    from mosaickv.evaluation.fused_history_unpack import use as fused
    from mosaickv.evaluation.history_source_precision import pack_source
    from mosaickv.evaluation.prepared_address_dispatch import gpu_controls
    from mosaickv.evaluation.prepared_address_dispatch import use as addresses

    manifest(p, base.sha(root / "plan.json"), folder, i, gpu=True)
    controls = gpu_controls()
    recent = tuple(
        SimpleNamespace(
            shape=(1, 32, 16, 128),
            archive=SimpleNamespace(
                source=encode(torch.randn(1, 1, 16, 4096, device="cuda").bfloat16())
            ),
        )
        for _ in range(4)
    )
    older = SimpleNamespace(
        shape=(1, 32, 48, 128),
        body=pack_source(torch.randn(1, 1, 48, 4096, device="cuda").bfloat16(), group=32),
    )

    @dataclass
    class Packet:
        shape: tuple
        recent: tuple
        older: object

    packet = Packet((1, 32, 112, 128), recent, older)
    target = torch.empty(128, 4096, device="cuda", dtype=torch.bfloat16)
    rows = []
    rng = np.random.default_rng(p["seed"] + i)
    with fused():
        launch(packet, target)
        expected = target[:112].clone()
        for rep in range(-5, 20):
            for flag in rng.permutation([False, True]):
                with addresses(bool(flag)) as report:
                    torch.cuda.synchronize()
                    begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                    begin.record()
                    started = perf_counter()
                    for _ in range(50):
                        launch(packet, target)
                    host = perf_counter() - started
                    end.record()
                    torch.cuda.synchronize()
                    elapsed = begin.elapsed_time(end)
                if not torch.equal(expected.view(torch.int16), target[:112].view(torch.int16)):
                    raise RuntimeError("same binary address dispatch changed a source word")
                rows.append(
                    dict(
                        repetition=rep,
                        warmup=rep < 0,
                        raw_addresses=bool(flag),
                        host_enqueue_seconds_per_launch=host / 50,
                        CUDA_stream_seconds_per_launch=elapsed / 50000,
                        dispatch=report,
                    )
                )
    stats = {}
    for metric in ("host_enqueue_seconds_per_launch", "CUDA_stream_seconds_per_launch"):
        pairs = [
            [r[metric] for r in rows if not r["warmup"] and r["raw_addresses"] == flag]
            for flag in (False, True)
        ]
        stats[metric] = statistics.paired(
            pairs, seed=p["seed"] + i, replicates=10000, family_comparisons=4
        )
    passed = stats["host_enqueue_seconds_per_launch"]["equal_or_faster"]
    write_once(
        folder / "result.json",
        dict(
            status="completed",
            passed=passed,
            correctness_passed=True,
            controls=controls,
            trials=rows,
            paired=stats,
            bounded_operator_gate_passed=passed,
            operator_intervals_NOT_model_latency=True,
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
    from mosaickv.evaluation.prepared_address_dispatch import use as addresses
    from mosaickv.evaluation.tiled_softmax import use as tile

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
            private_scores=stack.enter_context(tile(rows=p["tiled_softmax_rows"], inplace=True)),
            prepared_addresses=stack.enter_context(addresses(arm == ARMS[1])),
        )

    def compare(actual, parent, plan, arm):
        proof = precision.fidelity(actual, parent, plan, precision.ARMS[1])
        if arm == ARMS[0]:
            reference.append(actual.clone())
        if not reference:
            raise ValueError("original Tensor-argument reference required")
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
        info = b.get("common_header", {}).get("prepared_addresses") or {}
        good = bool(
            b["fidelity"].get("all_128_logits_bitwise_equal_original")
            and a["initial_GPU_KV_bytes"] == b["initial_GPU_KV_bytes"]
            and a["final_GPU_KV_bytes"] == b["final_GPU_KV_bytes"]
            and info.get("source_replays", 0) > 0
            and info.get("patch_replays", 0) > 0
            and info.get("additional_GPU_storage_bytes") == 0
        )
        eligible &= good
        pairs.append(dict(input=i, gate=good, dispatch=info))
    write_once(
        root / "analysis/address-decision.json",
        dict(
            complete=proof["complete"],
            bitwise_and_storage_gate=bool(eligible),
            pairs=pairs,
            fresh_cohort_authorized=False,
            independent_quality_established=False,
            complete_superiority_established=False,
        ),
    )
    if eligible:
        from prepared_address_timing import prepare as timing_prepare

        timing_prepare(root / "paired-timing")


if __name__ == "__main__":
    base.prepare, base.operator, base.model, base.analyze = prepare, operator, model, analyze
    base.main()
