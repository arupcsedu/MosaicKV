"""Randomized same-selection source-storage timing; not request latency or quality."""

from __future__ import annotations

import gc
import random
import time
from typing import Any

VARIANTS = ("raw_separate", "encoded_KV", "hybrid_source")


def orders(seed: int) -> list[tuple[int, list[str]]]:
    rng = random.Random(seed)
    result = []
    for block in range(-5, 20):
        order = list(VARIANTS)
        rng.shuffle(order)
        result.append((block, order))
    return result


def benchmark(
    adapter: Any,
    prefill: Any,
    capture: Any,
    positions: Any,
    *,
    seed: int,
    compact_projection: bool = False,
    streamed_projection: bool = False,
) -> list[Any]:
    import torch

    from mosaickv.adapters.huggingface.types import DecodeState
    from mosaickv.evaluation.lossless_lifetime import standard_greedy_reference
    from mosaickv.evaluation.lossless_prompt_cache import (
        IsolatedStorageTransport,
        PromptStorageCache,
        construct_cache,
    )
    from mosaickv.evaluation.projection_cache import construct_projection_cache

    reference = standard_greedy_reference(adapter, prefill, positions)
    rows = []
    for block, order in orders(seed):
        for order_index, variant in enumerate(order):
            torch.cuda.synchronize()
            before = time.monotonic()
            cache: PromptStorageCache
            if variant == "hybrid_source":
                cache = construct_projection_cache(
                    adapter,
                    prefill,
                    capture,
                    positions,
                    policy="hybrid",
                    audit_workspace=False,
                    compact_projection=compact_projection,
                    streamed_projection=streamed_projection,
                )
            else:
                encoded = variant == "encoded_KV"
                cache = construct_cache(
                    adapter,
                    prefill.state.past_key_values,
                    positions,
                    encode=encoded,
                    gpu_encoder=encoded,
                    fused_unpack=encoded,
                )
            torch.cuda.synchronize()
            constructor = time.monotonic() - before
            length = max(len(heads[0]) for heads in positions)
            state = DecodeState(
                cache,
                torch.ones((1, length), device=adapter.device, dtype=torch.long),
                length,
                prefill.state.logical_sequence_length,
                prefill.state.next_decode_position,
                prefill.state.modality_map,
                dict(
                    prefill.state.model_state,
                    mosaickv_packed_all_valid=True,
                    mosaickv_cache_all_valid=True,
                ),
            )
            token = prefill.next_token_id
            values = [prefill.logits.detach().reshape(-1).float()]
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            before = time.monotonic()
            with IsolatedStorageTransport(adapter, cache):
                start.record()
                for _ in range(31):
                    output = adapter.decode_one_token(token, state, capture_queries=False)
                    token = output.next_token_id
                    values.append(output.logits.detach().reshape(-1).float())
                end.record()
            torch.cuda.synchronize()
            wall = time.monotonic() - before
            elapsed = start.elapsed_time(end)
            logits = torch.stack(values).cpu()
            if not torch.equal(logits, reference):
                raise RuntimeError("timed storage changed original-selection greedy logits")
            if state.next_decode_position != prefill.state.next_decode_position + 31:
                raise RuntimeError("timing changed original logical positions")
            record = cache.storage_record()
            if record["outstanding_layer_leases"] != 0:
                raise RuntimeError("timing leaked a layer lease")
            rows.append(
                dict(
                    block=block,
                    warmup=block < 0,
                    order=order_index,
                    variant=variant,
                    seed=seed,
                    compact_projection=compact_projection,
                    streamed_projection=streamed_projection,
                    construction_seconds=constructor,
                    decode_wall_seconds=wall,
                    decode_cuda_ms=elapsed,
                    bitwise_reference_parity=True,
                    persistent_storage={
                        key: record[key]
                        for key in (
                            "initial_active_kv_storage_bytes",
                            "final_active_kv_storage_bytes",
                        )
                    },
                    workspace_audit_enabled=False,
                    peak_from_timed_trial_admissible=False,
                    memory_reference="separately_instrumented_same_selection_runtime_control",
                    boundary="post_selection_shared_prefill_constructor_and_31_decode_updates",
                    not_end_to_end_or_quality_measurement=True,
                )
            )
            del output, values, logits, state, cache
            gc.collect()  # Cache/layer observer cycles are collected outside timing.
    return rows


def summarize(trials: list[Any]) -> list[Any]:
    import numpy as np

    groups = {}
    for variant in VARIANTS:
        group = sorted((t for t in trials if t["variant"] == variant), key=lambda t: t["block"])
        if (
            len(group) != 25
            or [t["block"] for t in group] != list(range(-5, 20))
            or any(
                not t["bitwise_reference_parity"]
                or t["warmup"] != (t["block"] < 0)
                or t["workspace_audit_enabled"]
                or t["peak_from_timed_trial_admissible"]
                for t in group
            )
            or any(
                not np.isfinite(t[k]) or t[k] <= 0
                for t in group
                for k in ("construction_seconds", "decode_wall_seconds", "decode_cuda_ms")
            )
        ):
            raise ValueError("complete finite unaudited bitwise-paired 5/20 timing required")
        if any(t["persistent_storage"] != group[0]["persistent_storage"] for t in group):
            raise ValueError("persistent storage changed between trials")
        groups[variant] = group
    rng = np.random.default_rng(20261030)
    indices = rng.integers(0, 20, size=(10000, 20))
    rows = []
    for variant, group in groups.items():
        row: dict[str, Any] = dict(
            variant=variant, warmups=5, timed_repetitions=20, **group[0]["persistent_storage"]
        )
        for metric in ("decode_cuda_ms", "decode_wall_seconds", "construction_seconds"):
            values = np.array([t[metric] for t in group[5:]])
            row[metric] = dict(
                mean=float(values.mean()),
                p50=float(np.median(values)),
                p95=float(np.quantile(values, 0.95)),
            )
            for comparator in ("raw_separate", "encoded_KV"):
                baseline = np.array([t[metric] for t in groups[comparator][5:]])
                ratio = values[indices].mean(1) / baseline[indices].mean(1)
                row[f"{metric}_ratio_vs_{comparator}"] = dict(
                    mean_ratio=float(values.mean() / baseline.mean()),
                    paired_ci95=[
                        float(np.quantile(ratio, 0.025)),
                        float(np.quantile(ratio, 0.975)),
                    ],
                )
        rows.append(row)
    return rows
