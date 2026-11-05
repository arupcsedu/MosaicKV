"""Attribution only: consumed inputs, no latency acceptance or new evidence."""

import time
from contextlib import ExitStack
from unittest.mock import patch

STEPS = (1, 64, 126)  # Capture has finished before CUPTI starts.


def phase_summary(values):
    return {
        name: dict(
            calls=len(v), total_host_seconds=sum(v), maximum_host_seconds=max(v, default=0.0)
        )
        for name, v in values.items()
    }


def collect(adapter, call, folder):
    import torch
    from evicastkv_native_model_profile import actual_kernels, event_row

    from mosaickv.evaluation import lt_source_format, prompt_graph_reader
    from mosaickv.evaluation.fox_campaign import sha

    phases = {
        k: []
        for k in (
            "conversion",
            "capture_inclusive",
            "replay_host",
            "allocator_snapshot",
            "retirement",
        )
    }
    windows, events = [], []
    active, count = False, 0
    original_step = adapter.decode_one_token
    original_fill = prompt_graph_reader.PromptGraphPool.reconstruct

    def step(*args, **kwargs):
        nonlocal count, active
        index = count
        count += 1
        if index not in STEPS:
            return original_step(*args, **kwargs)
        torch.cuda.synchronize()
        first = len(events)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
        ) as profiler:
            active = True
            try:
                result = original_step(*args, **kwargs)
            finally:
                torch.cuda.synchronize()
                active = False
        path = folder / f"step-{index}.json"
        profiler.export_chrome_trace(str(path))
        kernels = actual_kernels(path)
        if kernels["count"] < 1 or kernels["positive_count"] != kernels["count"]:
            raise RuntimeError("every actual model window requires positive CUPTI kernel durations")
        window_events = events[first:]
        if len(window_events) != 64:
            raise RuntimeError("every late model window must replay all 64 prompt graphs")
        windows.append(
            dict(
                step=index,
                trace=str(path),
                trace_sha256=sha(path),
                actual_kernels=kernels,
                operators=[event_row(e) for e in profiler.key_averages(group_by_input_shape=True)],
                reader_stream_intervals_ms=[a.elapsed_time(b) for a, b in window_events],
                stream_intervals_include_host_gaps_NOT_pure_kernel_time=True,
            )
        )
        return result

    def fill(pool, layer, component, destination):
        capture = (id(layer), component) not in pool.entries
        if active and capture:
            raise RuntimeError("CUPTI attribution must not perturb initial graph ownership checks")
        if active:
            a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            a.record()
        start = time.perf_counter()
        result = original_fill(pool, layer, component, destination)
        phases["capture_inclusive" if capture else "replay_host"].append(
            time.perf_counter() - start
        )
        if active:
            b.record()
            events.append((a, b))
        return result

    def timed(function, name):
        def wrapped(*args, **kwargs):
            start = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                phases[name].append(time.perf_counter() - start)

        return wrapped

    with ExitStack() as stack:
        stack.enter_context(patch.object(adapter, "decode_one_token", step))
        stack.enter_context(patch.object(prompt_graph_reader.PromptGraphPool, "reconstruct", fill))
        stack.enter_context(
            patch.object(
                prompt_graph_reader,
                "allocator_blocks",
                timed(prompt_graph_reader.allocator_blocks, "allocator_snapshot"),
            )
        )
        stack.enter_context(
            patch.object(
                prompt_graph_reader.PromptGraphPool,
                "close",
                timed(prompt_graph_reader.PromptGraphPool.close, "retirement"),
            )
        )
        stack.enter_context(
            patch.object(lt_source_format, "convert", timed(lt_source_format.convert, "conversion"))
        )
        measured = call()
    if count != 127 or [r["step"] for r in windows] != list(STEPS):
        raise RuntimeError("complete consumed trajectory and all three windows required")
    if measured["logits_bitwise"] is not True or measured["no_FullKV_fallback"] is not True:
        raise RuntimeError("all 128 parent logit vectors must remain bitwise exact")
    # Deliberately remove instrumented full-request timings from the report.
    return dict(
        windows=windows,
        phases=phase_summary(phases),
        nested_phases_must_NOT_be_summed=True,
        parent_logits_bitwise=True,
        no_FullKV_fallback=True,
        initial_encoded_bytes=measured["initial_encoded_bytes"],
        final_encoded_bytes=measured["final_encoded_bytes"],
        prompt_graph_reader=measured["prompt_graph_reader"],
        independent_quality_N=0,
        instrumentation_NOT_request_latency_evidence=True,
    )
