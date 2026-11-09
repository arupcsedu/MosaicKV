"""CPU-only profiling of the frozen schedule's metadata/control flow.

Mock projector/format methods, not the scheduling module's metadata guards.
These numbers are not GPU/full-request latency and omit CUDA stream-query
overhead. No candidate is constructed or promoted.
"""

import cProfile
import os
import pstats
import weakref
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch

from mosaickv.evaluation import mixed_history_projection as mixed
from mosaickv.evaluation import native_history_reader as native
from mosaickv.evaluation import sequential_source_history as sequential
from mosaickv.evaluation.dispatch_affinity_audit import pinned_thread
from mosaickv.evaluation.history_value_first import use


def fixture(stack):
    class State:
        closed, graphs = False, None

    class Layer(sequential.SequentialSourceHistoryLayer):
        def get_seq_length(self):
            return 68

    state = State()
    state.engine = SimpleNamespace(source=torch.empty(128, 4096, dtype=torch.bfloat16))
    cos = torch.ones(1, 64, 128, dtype=torch.bfloat16)
    packet = mixed.MixedHistory(
        None,
        (),
        cos,
        cos,
        (),
        weakref.ref(state),
        (1, 32, 64, 128),
        (native.phase_token(cos), native.phase_token(cos)),
    )
    layer = Layer.__new__(Layer)
    layer.prompt_length, layer.tail_used = 3, 1
    layer.source_sealed = [packet]
    layer.representation, layer.attention = object(), object()
    lease = torch.empty(1, 32, 68, 128, dtype=torch.bfloat16)
    layer.leased, layer.current_component = (lease, lease), None
    counters = dict(mock_source_unpacks=0, mock_projections=0)

    def no_gpu_unpack(unused, source):
        if source is not state.engine.source:
            raise RuntimeError("unexpected CPU mock source")
        counters["mock_source_unpacks"] += 1

    def fill(packet, attention, component, target, **kwargs):
        if packet is not layer.representation:
            mixed.unpack_sources(packet, state.engine.source)
            counters["mock_projections"] += 1

    def materialize(layer, component):
        sequential.fill_component(layer.representation, layer.attention, component, lease)
        sequential.fill_component(packet, layer.attention, component, lease[..., 3:67, :])
        layer.current_component = component

    stack.enter_context(patch.object(native, "current", lambda: state))
    stack.enter_context(patch.object(mixed, "unpack_sources", no_gpu_unpack))
    stack.enter_context(patch.object(sequential, "fill_component", fill))
    stack.enter_context(
        patch.object(sequential.SequentialSourceHistoryLayer, "materialize_component", materialize)
    )
    return layer, counters


@torch.inference_mode()
def profile(directory, *, transactions=1000, warmups=50):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise RuntimeError("explicit CPU-only uninitialized-CUDA process required")
    if type(transactions) is not int or transactions < 1 or type(warmups) is not int or warmups < 0:
        raise ValueError("fixed positive profile count and nonnegative warmups required")
    if directory.exists():
        raise FileExistsError("new immutable CPU profile directory required")
    directory.mkdir()
    results = []
    with pinned_thread() as affinity, ExitStack() as stack:
        layer, counters = fixture(stack)
        for enabled in (False, True):
            with use(enabled) as report:
                for _ in range(warmups):
                    for c in (0, 1):
                        layer.materialize_component(c)
                before = dict(counters)
                begin = {k: report[k] for k in ("early_values", "skipped_source_unpacks")}
                profiler = cProfile.Profile()
                profiler.enable()
                for _ in range(transactions):
                    for c in (0, 1):
                        layer.materialize_component(c)
                profiler.disable()
            profiler.dump_stats(str(directory / ("enabled.prof" if enabled else "original.prof")))
            stats = pstats.Stats(profiler)
            functions = []
            for (file, line, name), (
                primitive,
                calls,
                own,
                cumulative,
                _callers,
            ) in stats.stats.items():
                functions.append(
                    dict(
                        file=file,
                        line=line,
                        function=name,
                        primitive_calls=primitive,
                        calls=calls,
                        self_seconds=own,
                        cumulative_seconds=cumulative,
                    )
                )
            delta = {k: counters[k] - before[k] for k in counters}
            if delta != dict(
                mock_projections=2 * transactions,
                mock_source_unpacks=transactions * (1 if enabled else 2),
            ):
                raise RuntimeError("CPU profile did not exercise the intended metadata path")
            results.append(
                dict(
                    enabled=enabled,
                    transactions=transactions,
                    profile_seconds=stats.total_tt,
                    counters=delta,
                    schedule=report,
                    measured_early_values=report["early_values"] - begin["early_values"],
                    measured_skips=report["skipped_source_unpacks"]
                    - begin["skipped_source_unpacks"],
                    functions=sorted(functions, key=lambda r: (-r["self_seconds"], r["file"])),
                )
            )
    return dict(
        measurement_type="CPU_mocked_GPU_metadata_profile_NOT_model_latency",
        affinity=affinity,
        results=results,
        GPU_or_model_work_executed=False,
        frozen_schedule_metadata_guards_and_bookkeeping_profiled=True,
        original_projector_and_format_methods_mocked=True,
        CUDA_stream_query_and_GPU_work_NOT_profiled=True,
        cProfile_overhead_NOT_removed=True,
        independent_quality_or_GPU_performance_inference_authorized=False,
        candidate_combined_or_promoted=False,
    )
