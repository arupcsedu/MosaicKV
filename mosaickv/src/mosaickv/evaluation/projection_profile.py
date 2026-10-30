"""Bounded consumed-input operator profile; profile durations are not latency evidence."""

from __future__ import annotations

import gc
from contextlib import ExitStack, contextmanager
from functools import wraps
from pathlib import Path
from typing import Any
from unittest.mock import patch


@contextmanager
def ranges() -> Any:
    import torch

    import mosaickv.evaluation.projection_cache as source
    from mosaickv.evaluation.lossless_prompt_cache import PromptStorageLayer

    def annotate(function: Any, name: str) -> Any:
        @wraps(function)
        def call(*args: Any, **kwargs: Any) -> Any:
            with torch.profiler.record_function(name):
                return function(*args, **kwargs)

        return call

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                source,
                "reconstruct_archive",
                annotate(source.reconstruct_archive, "evicastkv::source_reconstruction"),
            )
        )
        stack.enter_context(
            patch.object(
                source.ProjectionLayer,
                "materialize",
                annotate(source.ProjectionLayer.materialize, "evicastkv::source_materialize"),
            )
        )
        stack.enter_context(
            patch.object(
                PromptStorageLayer,
                "materialize",
                annotate(PromptStorageLayer.materialize, "evicastkv::encoded_KV_materialize"),
            )
        )
        yield


def run(adapter: Any, prefill: Any, positions: Any, cache: Any) -> Any:
    import torch

    from mosaickv.adapters.huggingface.types import DecodeState
    from mosaickv.evaluation.lossless_prompt_cache import IsolatedStorageTransport

    length = max(len(h[0]) for h in positions)
    state = DecodeState(
        cache,
        torch.ones((1, length), device=adapter.device, dtype=torch.long),
        length,
        prefill.state.logical_sequence_length,
        prefill.state.next_decode_position,
        prefill.state.modality_map,
        dict(
            prefill.state.model_state, mosaickv_packed_all_valid=True, mosaickv_cache_all_valid=True
        ),
    )
    values = [prefill.logits.detach().reshape(-1).float()]
    token = prefill.next_token_id
    with IsolatedStorageTransport(adapter, cache):
        for _ in range(4):
            out = adapter.decode_one_token(token, state, capture_queries=False)
            token = out.next_token_id
            values.append(out.logits.detach().reshape(-1).float())
    if state.next_decode_position != prefill.state.next_decode_position + 4:
        raise RuntimeError("operator profile changed logical positions")
    return torch.stack(values)


def profile(
    adapter: Any, prefill: Any, capture: Any, positions: Any, output_dir: Path, role: str
) -> list[Any]:
    import torch

    from mosaickv.evaluation.fox_campaign import sha, write_once
    from mosaickv.evaluation.lossless_lifetime import standard_greedy_reference
    from mosaickv.evaluation.lossless_prompt_cache import construct_cache
    from mosaickv.evaluation.projection_cache import construct_projection_cache

    reference = standard_greedy_reference(adapter, prefill, positions, tokens=5)

    def build(variant: str) -> Any:
        if variant == "hybrid_source":
            return construct_projection_cache(
                adapter, prefill, capture, positions, policy="hybrid", audit_workspace=False
            )
        return construct_cache(
            adapter,
            prefill.state.past_key_values,
            positions,
            encode=True,
            gpu_encoder=True,
            fused_unpack=True,
        )

    results = []
    for variant in ("encoded_KV", "hybrid_source"):
        for _ in range(5):
            cache = build(variant)
            logits = run(adapter, prefill, positions, cache)
            if not torch.equal(logits.cpu(), reference):
                raise RuntimeError("profile warmup changed reference logits")
            del cache, logits
            gc.collect()
        cache = build(variant)
        torch.cuda.synchronize()
        with (
            torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
            ) as prof,
            ranges(),
        ):
            logits = run(adapter, prefill, positions, cache)
            torch.cuda.synchronize()
        if not torch.equal(logits.cpu(), reference):
            raise RuntimeError("profile instrumentation changed reference logits")
        if cache.storage_record()["outstanding_layer_leases"] != 0:
            raise RuntimeError("profile leaked layer lease")
        trace = output_dir / f"{role}-{variant}-operator_trace.json.gz"
        prof.export_chrome_trace(str(trace))
        operators = [
            dict(
                name=e.key,
                input_shapes=e.input_shapes,
                calls=e.count,
                self_cpu_us=e.self_cpu_time_total,
                cpu_inclusive_us=e.cpu_time_total,
                self_device_us=e.self_device_time_total,
                device_inclusive_us=e.device_time_total,
            )
            for e in prof.key_averages(group_by_input_shape=True)
        ]
        record = dict(
            role=role,
            variant=variant,
            warmups=5,
            profiled_decode_updates=4,
            bitwise_reference_parity=True,
            trace_sha256=sha(trace),
            trace=trace.name,
            operators=operators,
            top_self_device_operators=sorted(
                operators, key=lambda r: r["self_device_us"], reverse=True
            )[:20],
            durations_are_instrumented_not_comparable_latency=True,
            inclusive_ranges_must_not_be_summed=True,
            selector_changed=False,
            no_benchmark_quality_claim=True,
            whole_request_peak_certified=False,
        )
        write_once(output_dir / f"{role}-{variant}-operator_profile.json", record)
        results.append({k: v for k, v in record.items() if k != "operators"})
        del cache, logits, prof
        gc.collect()
    return results
