"""Host attribution from input preparation to first decode, no acceptance."""

import cProfile
import pstats
from contextlib import ExitStack
from unittest.mock import patch

from mosaickv.evaluation.host_decode_profile import summarize


def collect(adapter, call, folder):
    from mosaickv.evaluation.fox_campaign import sha

    profiler = cProfile.Profile()
    prepare, decode = adapter.prepare_inputs, adapter.decode_one_token
    prepares = steps = 0

    def prepared(*args, **kwargs):
        nonlocal prepares
        prepares += 1
        if prepares != 1 or steps:
            raise RuntimeError("one unchanged original prefill preparation required")
        profiler.enable()
        return prepare(*args, **kwargs)

    def step(*args, **kwargs):
        nonlocal steps
        if steps == 0:
            profiler.disable()
        steps += 1
        return decode(*args, **kwargs)

    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(adapter, "prepare_inputs", prepared))
            stack.enter_context(patch.object(adapter, "decode_one_token", step))
            measured = call()
    finally:
        profiler.disable()
    if (
        prepares != 1
        or steps != 127
        or measured["logits_bitwise"] is not True
        or measured["no_FullKV_fallback"] is not True
    ):
        raise RuntimeError("complete single-prefill own-parent-bitwise trajectory required")
    path = folder / "constructor.prof"
    profiler.dump_stats(str(path))
    return dict(
        summarize(pstats.Stats(profiler).stats),
        profile=str(path),
        profile_sha256=sha(path),
        boundary="prepare_inputs_through_prefill_construction_and_cache_setup_until_first_decode",
        decode_excluded=True,
        not_pure_compression_time=True,
        parent_logits_bitwise=True,
        no_FullKV_fallback=True,
        no_new_GPU_workspace=True,
        whole_request_GPU_peak_certified=False,
    )
