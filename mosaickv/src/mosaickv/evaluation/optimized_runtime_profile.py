"""Nonoverlapping CPU attribution of the already optimized consumed request.

Construction includes preprocessing/prefill/setup; it is NOT pure compression.
The initial decode/capture is excluded. Instrumented durations cannot establish
latency acceptance. No weights, cache operators, or numerical guards change.
"""

import cProfile
import pstats
from contextlib import ExitStack
from unittest.mock import patch

from mosaickv.evaluation.host_decode_profile import summarize


def collect(adapter, call, folder):
    from mosaickv.evaluation.fox_campaign import sha

    construction, decode = cProfile.Profile(), cProfile.Profile()
    prepare, step = adapter.prepare_inputs, adapter.decode_one_token
    prepares = updates = 0

    def prepared(*args, **kwargs):
        nonlocal prepares
        prepares += 1
        if prepares != 1 or updates:
            raise RuntimeError("single unchanged prefill required")
        construction.enable()
        return prepare(*args, **kwargs)

    def decoded(*args, **kwargs):
        nonlocal updates
        index = updates
        updates += 1
        if index == 0:
            construction.disable()
            return step(*args, **kwargs)
        decode.enable()
        try:
            return step(*args, **kwargs)
        finally:
            decode.disable()

    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(adapter, "prepare_inputs", prepared))
            stack.enter_context(patch.object(adapter, "decode_one_token", decoded))
            measured = call()
    finally:
        construction.disable()
        decode.disable()
    proof = measured.get("graph_cache_accounting", {})
    if (
        prepares != 1
        or updates != 127
        or measured.get("logits_bitwise") is not True
        or measured.get("no_FullKV_fallback") is not True
        or proof.get("policy") != "event_sum"
        or proof.get("final_independent_tensor_enumeration_preserved") is not True
    ):
        raise RuntimeError("complete own-parent-bitwise event-sum trajectory required")
    result = {}
    for name, profiler in (("construction", construction), ("decode", decode)):
        path = folder / f"{name}.prof"
        profiler.dump_stats(str(path))
        result[name] = dict(
            summarize(pstats.Stats(profiler).stats),
            profile=str(path),
            profile_sha256=sha(path),
        )
    return dict(
        phases=result,
        construction_includes_preprocessing_prefill_setup=True,
        decode_updates=126,
        initial_decode_capture_excluded=True,
        graph_cache_accounting=proof,
        parent_logits_bitwise=True,
        no_FullKV_fallback=True,
        no_new_GPU_workspace=True,
        whole_request_GPU_peak_certified=False,
        instrumentation_NOT_request_latency_acceptance=True,
        independent_quality_N=0,
    )
