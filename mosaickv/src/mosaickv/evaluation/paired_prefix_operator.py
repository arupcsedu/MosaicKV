"""Bounded synthetic K/V-prefix sharing test; not a model latency result."""

import random
import time
from types import SimpleNamespace

import torch

from mosaickv.evaluation.row_major_source_operator import summarize


@torch.inference_mode()
def run(index):
    from mosaickv.evaluation.lossless_affine_minimum import encode_minimum
    from mosaickv.evaluation.precision_tier import pack
    from mosaickv.evaluation.shared_source_kernel import project

    if type(index) is not int or index not in (0, 1):
        raise ValueError("two frozen consumed geometries only")
    n, h, d, prefix = (600, 604)[index], 32, 128, 256
    generator = torch.Generator(device="cuda").manual_seed(20261104)
    x = torch.randn((1, 1, n, h * d), device="cuda", generator=generator).bfloat16()
    packet = SimpleNamespace(
        source=encode_minimum(pack(x, bits=3, axis=2)),
        shape=(1, h, n, d),
        source_head_group=4,
        source_row_tile=16,
        source_num_warps=None,
    )
    phase = torch.linspace(-3, 3, n * 64, device="cuda").reshape(1, n, 64)
    packet.cos = torch.cat((phase.cos(), phase.cos()), -1).bfloat16()
    packet.sin = torch.cat((phase.sin(), phase.sin()), -1).bfloat16()
    attention = SimpleNamespace(
        k_proj=torch.nn.Linear(h * d, h * d, bias=False, device="cuda", dtype=torch.bfloat16),
        v_proj=torch.nn.Linear(h * d, h * d, bias=False, device="cuda", dtype=torch.bfloat16),
    )
    expected = tuple(
        torch.full((1, h, n + 127, d), -71.0, device="cuda", dtype=torch.bfloat16) for _ in range(2)
    )
    for c in (0, 1):
        project(packet, attention, (expected[c], expected[c]), component=c)
    actual = tuple(torch.full_like(t, -71.0) for t in expected)
    workspace = torch.empty((1, h, prefix, d), device="cuda", dtype=torch.bfloat16)

    def execute(paired, inspect=False):
        if not paired:
            return [
                project(
                    packet, attention, (actual[c], actual[c]), component=c, inspect_kernel=inspect
                )
                for c in (0, 1)
            ]
        joint = project(
            packet,
            attention,
            (actual[0], workspace),
            component=-1,
            window=(0, prefix, True),
            inspect_kernel=inspect,
        )
        key_tail = project(
            packet,
            attention,
            (actual[0], actual[0]),
            component=0,
            window=(prefix, n - prefix, False),
            inspect_kernel=inspect,
        )
        actual[1][..., :prefix, :].copy_(workspace)
        value_tail = project(
            packet,
            attention,
            (actual[1], actual[1]),
            component=1,
            window=(prefix, n - prefix, False),
            inspect_kernel=inspect,
        )
        return [joint, key_tail, value_tail]

    resources = [execute(False, True), execute(True, True)]
    if any(type(r["spills"]) is not int or r["spills"] != 0 for group in resources for r in group):
        raise RuntimeError("paired projection has unverified or nonzero spills")
    rows, rng = [], random.Random(20261104)
    for trial in range(-5, 20):
        modes = [False, True]
        rng.shuffle(modes)
        for paired in modes:
            for t in actual:
                t.fill_(-71.0)
            workspace.fill_(float("nan"))
            torch.cuda.synchronize()
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start = time.perf_counter()
            begin.record()
            execute(paired)
            end.record()
            torch.cuda.synchronize()
            wall = time.perf_counter() - start
            equal = all(
                torch.equal(a.view(torch.int16), b.view(torch.int16))
                for a, b in zip(actual, expected, strict=True)
            )
            if not equal:
                raise RuntimeError("paired K/V changed a prompt or history-sentinel bit")
            rows.append(
                dict(
                    trial=trial,
                    row4=paired,
                    paired_prefix=paired,
                    wall_seconds=wall,
                    cuda_interval_ms=begin.elapsed_time(end),
                    bitwise=equal,
                    extra_workspace_backing_bytes=workspace.untyped_storage().nbytes()
                    if paired
                    else 0,
                )
            )
    # The common reducer's historical `row4` field is ONLY its binary arm key;
    # the actual input remains the original three-bit source in BOTH arms.
    summary = summarize(rows)
    summary["binary_arm_key_legacy_name"] = "row4_not_code_layout"
    return dict(
        rows=rows,
        summary=summary,
        shape=[1, h, n, d],
        resources=resources,
        prefix_rows=prefix,
        all_original_rows_retained=True,
        original_three_bit_code_layout=True,
        includes_prefix_copy=True,
        no_checkpoint_loaded=True,
        independent_quality_N=0,
    )
