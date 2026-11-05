"""Sparse CUDA-event attribution without CUPTI; not latency acceptance."""

import time
from contextlib import ExitStack
from unittest.mock import patch

from mosaickv.evaluation.lt_model_profile import STEPS, phase_summary


def summarize(steps, windows):
    if [s["step"] for s in steps] != list(range(127)):
        raise RuntimeError("all 127 decode updates required")
    if [w["step"] for w in windows] != list(STEPS) or any(
        len(w["reader_stream_intervals_ms"]) != 64 for w in windows
    ):
        raise RuntimeError("all 64 readers in every registered early/middle/late window required")
    import math

    values = [s["step_stream_ms"] for s in steps]
    values += [v for w in windows for v in w["reader_stream_intervals_ms"]]
    if any(type(v) not in (float, int) or not math.isfinite(v) or v <= 0 for v in values):
        raise RuntimeError("finite positive actual CUDA-event intervals required")
    return dict(
        steps=steps,
        windows=windows,
        CUPTI_enabled=False,
        CUDA_events_include_host_gaps_NOT_pure_kernel_time=True,
        instrumentation_NOT_request_latency_acceptance=True,
        independent_quality_N=0,
    )


def collect(adapter, call, folder):
    import torch

    from mosaickv.evaluation.prompt_graph_reader import PromptGraphPool

    original_step, original_fill = adapter.decode_one_token, PromptGraphPool.reconstruct
    step_events, reader_events, phases = [], [], {"capture_host": [], "replay_host": []}
    index = -1

    def event():
        return torch.cuda.Event(enable_timing=True)

    def step(*args, **kwargs):
        nonlocal index
        index += 1
        a, b = event(), event()
        a.record()
        result = original_step(*args, **kwargs)
        b.record()
        step_events.append((index, a, b))
        return result

    def fill(pool, layer, component, destination):
        capture = (id(layer), component) not in pool.entries
        active = index in STEPS
        if active and capture:
            raise RuntimeError("window must follow initial graph capture")
        if active:
            a, b = event(), event()
            a.record()
        start = time.perf_counter()
        result = original_fill(pool, layer, component, destination)
        phases["capture_host" if capture else "replay_host"].append(time.perf_counter() - start)
        if active:
            b.record()
            reader_events.append((index, a, b))
        return result

    with ExitStack() as stack:
        stack.enter_context(patch.object(adapter, "decode_one_token", step))
        stack.enter_context(patch.object(PromptGraphPool, "reconstruct", fill))
        measured = call()
    torch.cuda.synchronize()
    if measured["logits_bitwise"] is not True or measured["no_FullKV_fallback"] is not True:
        raise RuntimeError("all 128 vocabulary vectors must match the own-parent words")
    steps = [dict(step=i, step_stream_ms=a.elapsed_time(b)) for i, a, b in step_events]
    windows = [
        dict(
            step=i,
            reader_stream_intervals_ms=[a.elapsed_time(b) for s, a, b in reader_events if s == i],
        )
        for i in STEPS
    ]
    return dict(
        summarize(steps, windows),
        host_phases=phase_summary(phases),
        parent_logits_bitwise=True,
        no_FullKV_fallback=True,
        initial_encoded_bytes=measured["initial_encoded_bytes"],
        final_encoded_bytes=measured["final_encoded_bytes"],
        prompt_graph_reader=measured["prompt_graph_reader"],
        CUDA_event_resources_not_certified_whole_request_memory=True,
    )
