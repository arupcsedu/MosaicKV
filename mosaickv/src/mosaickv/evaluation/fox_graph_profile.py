"""Evaluation-only graph cProfile instrumentation; never an online policy.

The clock is the calling thread's CPU clock, not GPU time or elapsed latency.
Inclusive subphase times nest and MUST NOT be added. Instrumented request times
are contaminated diagnostics and are never eligible as headline latency.
"""

from __future__ import annotations

import contextlib
import cProfile
import functools
import pstats
import time
from pathlib import Path
from unittest.mock import patch

SUBPHASES = {
    "pool_block_descriptors": "descriptor_pooling",
    "pool_prompt_attention_coactivation": "attention_pooling",
    "_add_cosine_edges": "cosine_edges",
    "_add_structural_edges": "structural_edges",
    "finalize": "edge_finalization",
    "_diagnostics": "graph_diagnostics",
    "build_evidence_graph": "complete_graph",
}


def function_rows(profile):
    """Export every function, including NumPy internals, without saved tensors."""
    rows = []
    for (filename, line, name), (primitive, calls, exclusive, cumulative, _) in pstats.Stats(
        profile
    ).stats.items():
        in_graph = "/mosaickv/graph/" in filename.replace("\\", "/")
        rows.append(
            dict(
                filename=filename,
                line=line,
                function=name,
                graph_subphase=SUBPHASES.get(name) if in_graph else None,
                primitive_calls=primitive,
                total_calls=calls,
                exclusive_thread_cpu_seconds=exclusive,
                cumulative_thread_cpu_seconds=cumulative,
            )
        )
    return sorted(
        rows, key=lambda r: (-r["exclusive_thread_cpu_seconds"], r["filename"], r["line"])
    )


class GraphCPUProfiler:
    """Bounded, exception-safe wrapper returning the ORIGINAL graph unchanged."""

    def __init__(self, *, maximum_graph_calls=2):
        if type(maximum_graph_calls) is not int or maximum_graph_calls < 1:
            raise ValueError("positive integer graph-call limit required")
        self.maximum_graph_calls = maximum_graph_calls
        self.records = []
        self._active = False

    def wrap(self, function):
        @functools.wraps(function)
        def call(*args, **kwargs):
            if self._active:
                raise RuntimeError("nested graph profiling is unsupported")
            if len(self.records) >= self.maximum_graph_calls:
                raise RuntimeError("registered graph-call budget exceeded")
            profile = cProfile.Profile(timer=time.thread_time)
            record = dict(call_index=len(self.records), status="failed", error=None)
            self.records.append(record)
            wall, cpu = time.perf_counter(), time.thread_time()
            self._active = True
            try:
                profile.enable()
                result = function(*args, **kwargs)
                profile.disable()
                record.update(
                    status="passed",
                    nodes=len(result.nodes),
                    edges=len(result.row_indices),
                )
                return result
            except BaseException as error:
                record["error"] = f"{type(error).__name__}: {error}"
                raise
            finally:
                profile.disable()
                self._active = False
                record.update(
                    wall_seconds=time.perf_counter() - wall,
                    calling_thread_cpu_seconds=time.thread_time() - cpu,
                    functions=function_rows(profile),
                )

        return call

    @contextlib.contextmanager
    def install(self):
        # hf_runtime imports this function by name. Patch that actual call site,
        # not just the package export; all function internals stay untouched.
        import mosaickv.backends.hf_runtime as runtime

        with patch.object(runtime, "build_evidence_graph", self.wrap(runtime.build_evidence_graph)):
            yield self

    def report(self):
        return dict(
            schema_version=1,
            measurement_type="consumed_cpu_graph_profile_diagnostic",
            clock="time.thread_time; calling thread only, excludes other BLAS threads",
            profiled_request_latency_eligible=False,
            nested_cumulative_times_additive=False,
            maximum_graph_calls=self.maximum_graph_calls,
            actual_graph_calls=len(self.records),
            calls=self.records,
        )


def profile_bindings(source):
    """Diagnostic source provenance, separate from production algorithm files."""
    from mosaickv.evaluation.fox_campaign import sha

    relative = "mosaickv/src/mosaickv/evaluation/fox_graph_profile.py"
    return {relative: sha(Path(source) / relative)}
