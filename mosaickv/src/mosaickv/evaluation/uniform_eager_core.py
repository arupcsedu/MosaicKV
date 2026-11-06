"""Uniform-only exact replay capability. Adaptive layouts retain original math.

The predecessor universal replay policy FAILED; this is not a pass of that
gate. This narrower feature never instantiates a Core for nonuniform layouts.
"""

import random
import statistics
import time

import torch

from mosaickv.evaluation.eager_core_operator import Core, controls, opportunities, reference_scores


def strategy(prompt_lengths, updates):
    if (
        not prompt_lengths
        or any(type(n) is not int or n < 1 for n in prompt_lengths)
        or type(updates) is not int
        or not 1 <= updates <= 127
    ):
        raise ValueError("positive immutable layer lengths and registered decode horizon required")
    if len(prompt_lengths) != 32 or max(prompt_lengths) + updates > 1024:
        return "original_eager_ineligible_geometry"
    if len(set(prompt_lengths)) != 1:
        return "original_eager_nonuniform"
    return "uniform_split_graph"


def supported(prompt_lengths, updates):
    return strategy(prompt_lengths, updates) == "uniform_split_graph"


@torch.inference_mode()
def study():
    checked = controls()  # Same strict words/owner/384-KiB gate, including all masks.
    baseline_checked = controls(lengths=(301, 305, 365, 369, 427, 431, 301), capacity=431)
    cached_checked = controls(reuse_views=True)
    cached_baseline_checked = controls(
        lengths=(301, 305, 365, 369, 427, 431, 301), capacity=431, reuse_views=True
    )
    generator = torch.Generator(device="cuda").manual_seed(20261106)
    rng = random.Random(20261106)
    records = []
    lease = torch.empty(1, 32, 731, 128, device="cuda", dtype=torch.bfloat16)
    for pattern in ("uniform", "adaptive"):
        lengths = (665,) * 32 if pattern == "uniform" else tuple(417 + 8 * i for i in range(32))
        prompt_lengths = tuple(n - 64 for n in lengths)
        frequencies = opportunities(prompt_lengths, 127)
        q = torch.randn(32, 1, 32, 1, 128, device="cuda", generator=generator).bfloat16()
        k = torch.randn(32, 1, 32, 731, 128, device="cuda", generator=generator).bfloat16()
        v = torch.randn(k.shape, device="cuda", generator=generator).bfloat16()
        masks = [torch.zeros(1, 1, 1, n, device="cuda", dtype=torch.bfloat16) for n in lengths]
        trials = []
        for trial in range(-5, 20):
            order = ["original", "uniform_uncached", "uniform_feature"]
            rng.shuffle(order)
            for place, policy in enumerate(order):
                torch.cuda.synchronize()
                allocated = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                before = time.perf_counter()
                start.record()
                enabled = policy != "original" and supported(prompt_lengths, 127)
                core = (
                    Core(
                        lease,
                        mask_dtype=torch.bfloat16,
                        graph=True,
                        frequencies=frequencies,
                        reuse_views=policy == "uniform_feature",
                    )
                    if enabled
                    else None
                )
                for index, length in enumerate(lengths):
                    lease[..., :length, :].copy_(k[index, ..., :length, :])
                    weights = (
                        reference_scores(q[index], lease[..., :length, :], masks[index], 128**-0.5)
                        if core is None
                        else core.scores_for(q[index], masks[index], length)
                    )
                    lease[..., :length, :].copy_(v[index, ..., :length, :])
                    output = (
                        torch.matmul(weights, lease[..., :length, :]).transpose(1, 2).contiguous()
                        if core is None
                        else core.values_for(length)
                    )
                captures = 0 if core is None else core.captures
                view_builds = 0 if core is None else core.view_builds
                view_reuses = 0 if core is None else core.view_reuses
                del weights, output
                if core is not None:
                    core.close()
                end.record()
                torch.cuda.synchronize()
                trials.append(
                    dict(
                        trial=trial,
                        warmup=trial < 0,
                        method=policy,
                        order=place,
                        selected_strategy=strategy(prompt_lengths, 127),
                        acceleration_eligible=supported(prompt_lengths, 127),
                        wall_seconds=time.perf_counter() - before,
                        gpu_seconds=start.elapsed_time(end) / 1000,
                        captures=captures,
                        view_descriptor_builds=view_builds,
                        view_descriptor_reuses=view_reuses,
                        torch_peak_increment_bytes=torch.cuda.max_memory_allocated() - allocated,
                        original_eager_passthrough=core is None,
                        cold_capture_and_retirement_included=True,
                    )
                )
                del core
        means = {
            policy: statistics.mean(
                t["wall_seconds"] for t in trials if t["method"] == policy and not t["warmup"]
            )
            for policy in ("original", "uniform_uncached", "uniform_feature")
        }
        records.append(
            dict(
                pattern=pattern,
                means=means,
                ratio=means["uniform_feature"] / means["original"],
                trials=trials,
                adaptive_acceleration_claim=False,
            )
        )
        del q, k, v, masks
    adaptive = records[1]
    passthrough = all(
        t["captures"] == 0 and t["original_eager_passthrough"] for t in adaptive["trials"]
    )
    return dict(
        complete=True,
        failures=[],
        controls=checked,
        shorter_uniform_baseline_controls=baseline_checked,
        cached_view_controls=cached_checked,
        cached_shorter_baseline_controls=cached_baseline_checked,
        records=records,
        uniform_feature_operator_feasible=records[0]["ratio"] <= 0.96 and passthrough,
        adaptive_acceleration_supported=False,
        adaptive_original_path_and_zero_graphs=passthrough,
        previous_universal_operator_gate_remains_negative=True,
        no_runtime_integration=True,
        no_model_latency_or_quality_acceptance=True,
        independent_quality_N=0,
    )
