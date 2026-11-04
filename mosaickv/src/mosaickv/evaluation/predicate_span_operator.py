"""Bounded same-ATen predicate ablation, never benchmark-quality evidence."""

import math
import random
from time import perf_counter

import numpy as np
import torch

from mosaickv.evaluation.deferred_transport_checks import use
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.shared_source_format import assert_transport_close, transport_corrections


def summarize(rows, seed=20261104):
    keys = [(r["trial"], r["full_span"]) for r in rows]
    if len(set(keys)) != len(keys) or set(keys) != {
        (t, full) for t in range(-5, 20) for full in (False, True)
    }:
        raise ValueError("complete unique five-warmup/twenty-paired operator trials required")
    if any(
        type(r["full_span"]) is not bool
        or type(r["trial"]) is not int
        or r["patch_words_identical"] is not True
        or type(r["wall_seconds"]) not in (int, float)
        or not math.isfinite(r["wall_seconds"])
        or r["wall_seconds"] <= 0
        for r in rows
    ):
        raise ValueError("identical patch words and finite positive timings required")
    arrays = {
        full: np.asarray(
            [
                r["wall_seconds"]
                for r in sorted(rows, key=lambda r: r["trial"])
                if r["full_span"] is full and r["trial"] >= 0
            ]
        )
        for full in (False, True)
    }
    draws = np.random.default_rng(seed).integers(0, 20, (10000, 20))
    ratios = arrays[True][draws].mean(1) / arrays[False][draws].mean(1)
    bounds = np.quantile(ratios, [0.05 / 8, 1 - 0.05 / 8]).tolist()
    return dict(
        mean_ratio=float(arrays[True].mean() / arrays[False].mean()),
        adjusted_ratio_CI=bounds,
        operator_improvement_gate=bounds[1] <= 0.85,
        registered_operator_upper_ratio=0.85,
        bootstrap_replicates=10000,
        request_latency_evidence=False,
        independent_quality_N=0,
    )


def run(index):
    if index not in (0, 1):
        raise ValueError("only the two consumed diagnostic geometries are registered")
    rows, rng = [], random.Random(20261104)
    generator = torch.Generator(device="cuda").manual_seed(20261104)
    n = (600, 604)[index]
    reference = torch.randn((1, 32, n, 128), generator=generator, device="cuda").bfloat16()
    actual = reference.clone()
    actual.flatten()[::127] += 0.25
    protected = torch.empty(0, device="cuda", dtype=torch.int32)
    frozen = None
    for trial in range(-5, 20):
        order = [False, True]
        rng.shuffle(order)
        for full in order:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start = perf_counter()
            begin.record()
            with use(True, coalesce_nonzero=True, full_predicate_span=full) as state:
                patch = transport_corrections(
                    (reference,), (actual,), protected, chunk_scalars=524288
                )[0]
                repaired = actual.clone()
                patch.apply(repaired)
                assert_transport_close(repaired, reference)
            end.record()
            torch.cuda.synchronize()
            wall = perf_counter() - start
            peak = torch.cuda.max_memory_allocated() - before
            if frozen is None:
                frozen = tuple(t.clone() for t in patch.tensors)
            equal = all(
                torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                for a, b in zip(frozen, patch.tensors, strict=True)
            )
            if not equal:
                raise RuntimeError("larger predicate span changed numerical patch words")
            rows.append(
                dict(
                    trial=trial,
                    full_span=full,
                    wall_seconds=wall,
                    stream_milliseconds=begin.elapsed_time(end),
                    patch_words_identical=equal,
                    peak_added_allocated_bytes_including_outputs=peak,
                    output_patch_backing_bytes=backing_bytes(patch.tensors),
                    validation=state.report(),
                )
            )
            del patch, repaired
    return dict(
        rows=rows,
        summary=summarize(rows),
        synthetic_operator_fixture=True,
        shape=list(reference.shape),
        not_model_or_quality_evidence=True,
        unchanged_ATen_operations_tolerances_and_dtypes=True,
    )
