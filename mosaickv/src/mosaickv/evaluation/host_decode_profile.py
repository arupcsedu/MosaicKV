"""Consumed-input CPU-call attribution; instrumented times are not acceptance."""

import cProfile
import math
import pstats
from unittest.mock import patch


def summarize(stats):
    rows = []
    for (path, line, name), (primitive, calls, own, cumulative, _callers) in stats.items():
        if (
            type(primitive) is not int
            or type(calls) is not int
            or not 0 <= primitive <= calls
            or any(
                type(v) not in (int, float) or not math.isfinite(v) or v < 0
                for v in (own, cumulative)
            )
        ):
            raise RuntimeError("finite actual CPU-call counters required")
        rows.append(
            dict(
                file=path,
                line=line,
                function=name,
                primitive_calls=primitive,
                calls=calls,
                self_seconds=own,
                cumulative_seconds=cumulative,
            )
        )
    if not rows:
        raise RuntimeError("nonempty actual decode CPU-call profile required")
    def tie(row):
        return row["file"], row["line"], row["function"]

    return dict(
        top_self=sorted(rows, key=lambda r: (-r["self_seconds"], tie(r)))[:100],
        top_cumulative=sorted(rows, key=lambda r: (-r["cumulative_seconds"], tie(r)))[:100],
        function_count=len(rows),
        cumulative_times_are_nested_do_NOT_sum=True,
        C_function_time_can_include_GPU_wait=True,
        instrumentation_NOT_request_latency_acceptance=True,
        independent_quality_N=0,
    )


def collect(adapter, call, folder):
    from mosaickv.evaluation.fox_campaign import sha

    profiler = cProfile.Profile()
    original = adapter.decode_one_token
    count = 0

    def step(*args, **kwargs):
        nonlocal count
        index = count
        count += 1
        if index == 0:  # Exclude all initial graph compilation/capture.
            return original(*args, **kwargs)
        profiler.enable()
        try:
            return original(*args, **kwargs)
        finally:
            profiler.disable()

    with patch.object(adapter, "decode_one_token", step):
        measured = call()
    if (
        count != 127
        or measured["logits_bitwise"] is not True
        or measured["no_FullKV_fallback"] is not True
    ):
        raise RuntimeError("complete parent-bitwise consumed trajectory required")
    path = folder / "decode.prof"
    profiler.dump_stats(str(path))
    return dict(
        summarize(pstats.Stats(profiler).stats),
        profile=str(path),
        profile_sha256=sha(path),
        CPU_profiled_decode_updates=126,
        first_capture_update_excluded=True,
        parent_logits_bitwise=True,
        no_FullKV_fallback=True,
        no_new_GPU_workspace=True,
        whole_request_GPU_peak_certified=False,
    )
