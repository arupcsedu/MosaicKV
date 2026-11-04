"""Synthetic operator feasibility, not model or task-quality evidence."""

import math
import random
import time
from types import SimpleNamespace

import numpy as np
import torch

from mosaickv.evaluation.row_major_source_codes import geometry, pack_into, reference, use


def summarize(rows, seed=20261104):
    keys = [(x["trial"], x["row4"]) for x in rows]
    if len(keys) != len(set(keys)) or set(keys) != {
        (t, r) for t in range(-5, 20) for r in (False, True)
    }:
        raise ValueError("complete unique five-warmup/twenty-paired operator trials required")
    if any(
        type(x["row4"]) is not bool
        or type(x["trial"]) is not int
        or x["bitwise"] is not True
        or type(x["wall_seconds"]) not in (float, int)
        or not math.isfinite(x["wall_seconds"])
        or x["wall_seconds"] <= 0
        for x in rows
    ):
        raise ValueError("bitwise results and positive finite nonboolean timings required")
    arrays = {
        r: np.asarray(
            [
                x["wall_seconds"]
                for x in sorted(rows, key=lambda x: x["trial"])
                if x["row4"] is r and x["trial"] >= 0
            ]
        )
        for r in (False, True)
    }
    draws = np.random.default_rng(seed).integers(0, 20, (10000, 20))
    ratio = arrays[True][draws].mean(1) / arrays[False][draws].mean(1)
    ci = np.quantile(ratio, [0.05 / 8, 1 - 0.05 / 8]).tolist()
    return dict(
        mean_ratio=float(arrays[True].mean() / arrays[False].mean()),
        adjusted_ratio_CI=ci,
        operator_gate=ci[1] <= 0.8,
        registered_operator_upper_ratio=0.8,
        bootstrap_replicates=10000,
        not_request_latency_evidence=True,
        independent_quality_N=0,
        fixture_seed=20261104,
    )


@torch.inference_mode()
def run(index):
    from mosaickv.evaluation.lossless_affine_minimum import encode_minimum
    from mosaickv.evaluation.precision_tier import pack
    from mosaickv.evaluation.shared_source_kernel import project

    if index not in (0, 1):
        raise ValueError("two frozen consumed geometries only")
    n, h, d = (600, 604)[index], 32, 128
    generator = torch.Generator(device="cuda").manual_seed(20261104)
    x = torch.randn((1, 1, n, h * d), device="cuda", generator=generator).bfloat16()
    body = encode_minimum(pack(x, bits=3, axis=2))
    scratch = torch.empty(geometry(n, h * d), dtype=torch.uint8, device="cuda")
    pack_into(body, scratch)
    if not torch.equal(scratch.cpu(), reference(body.codes.cpu(), n, h * d)):
        raise RuntimeError("integer row-major transport changed an original quantizer code")
    attention = SimpleNamespace(
        k_proj=torch.nn.Linear(h * d, h * d, bias=False, device="cuda", dtype=torch.bfloat16),
        v_proj=torch.nn.Linear(h * d, h * d, bias=False, device="cuda", dtype=torch.bfloat16),
    )
    phase = torch.linspace(-3, 3, n * 64, device="cuda").reshape(1, n, 64)
    packet = SimpleNamespace(
        source=body,
        shape=(1, h, n, d),
        cos=torch.cat((phase.cos(), phase.cos()), -1).bfloat16(),
        sin=torch.cat((phase.sin(), phase.sin()), -1).bfloat16(),
        source_head_group=4,
        source_row_tile=16,
        source_num_warps=None,
    )
    expected = tuple(
        torch.full((1, h, n + 127, d), -71.0, device="cuda", dtype=torch.bfloat16) for _ in range(2)
    )
    for c in (0, 1):
        project(packet, attention, (expected[c], expected[c]), component=c)
    actual = tuple(torch.full_like(t, -71.0) for t in expected)
    resources = []
    for mode in (False, True):
        from contextlib import nullcontext

        with use(body, scratch) if mode else nullcontext():
            resources.append(
                project(packet, attention, (actual[0], actual[0]), component=0, inspect_kernel=True)
            )
    if (
        any(type(r["spills"]) is not int for r in resources)
        or resources[1]["spills"] > resources[0]["spills"]
    ):
        raise RuntimeError("row-major transport introduced unverified or increased spills")
    records, rng = [], random.Random(20261104)
    for trial in range(-5, 20):
        placement = [False, True]
        rng.shuffle(placement)
        for row4 in placement:
            for t in actual:
                t.fill_(-71.0)
            torch.cuda.synchronize()
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            started = time.perf_counter()
            begin.record()
            if row4:
                pack_into(body, scratch)
            with use(body, scratch) if row4 else nullcontext():
                for c in (0, 1):
                    project(packet, attention, (actual[c], actual[c]), component=c)
            end.record()
            torch.cuda.synchronize()
            wall = time.perf_counter() - started
            equal = all(
                torch.equal(a.view(torch.int16), b.view(torch.int16))
                for a, b in zip(actual, expected, strict=True)
            )
            if not equal:
                raise RuntimeError("row-major projection changed a K/V or surrounding history bit")
            records.append(
                dict(
                    trial=trial,
                    row4=row4,
                    wall_seconds=wall,
                    cuda_interval_ms=begin.elapsed_time(end),
                    bitwise=equal,
                    extra_workspace_backing_bytes=scratch.untyped_storage().nbytes() if row4 else 0,
                )
            )
    return dict(
        rows=records,
        summary=summarize(records),
        shape=[1, h, n, d],
        resources=resources,
        bitwise_integer_transport=True,
        includes_conversion_and_both_components=True,
        synthetic_operator_fixture=True,
        no_checkpoint_loaded=True,
        independent_quality_N=0,
    )
