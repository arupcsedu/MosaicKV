"""One exact FullKV draft update from temporary CPU prefill snapshots.

No second prefill, label/oracle/reference artifact access, or retained GPU
full-cache clone. Temporary CPU originals and CUDA layer leases are reported.
This is a correctness-first forecasting diagnostic, not a free-memory claim.
"""

from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

import torch
from transformers.cache_utils import CacheLayerMixin

from mosaickv.evaluation.lossless_prompt_cache import (
    IsolatedStorageTransport,
    PromptStorageCache,
    PromptStorageLayer,
    backing_bytes,
)


def signatures(pairs):
    import hashlib

    # Inference snapshots intentionally have no version counter. Hash their
    # ORIGINAL bytes rather than making an unchecked immutability assertion.
    return tuple(
        hashlib.sha256(t.view(torch.uint8).numpy()).hexdigest() for pair in pairs for t in pair
    )


def snapshot(pair):
    if len(pair) != 2 or pair[0].shape != pair[1].shape or pair[0].ndim != 4:
        raise ValueError("complete matching original prompt K/V required")
    result = tuple(t.detach().to("cpu", copy=True).contiguous() for t in pair)
    if pair[0].device.type == "cuda":
        result = tuple(t.pin_memory() for t in result)
    return result


class DraftLayer(PromptStorageLayer):
    """Exact CPU prompt, one GPU token, two temporary native eager leases."""

    def __init__(self, pair, *, device, observe):
        if (
            len(pair) != 2
            or pair[0].shape != pair[1].shape
            or pair[0].ndim != 4
            or pair[0].shape[0] != 1
            or any(t.device.type != "cpu" or t.dtype != torch.bfloat16 for t in pair)
        ):
            raise ValueError("aligned batch-one CPU BF16 original prompt required")
        CacheLayerMixin.__init__(self)
        self.cpu_prompt = pair
        self._initialize_storage(
            tuple(pair[0].shape),
            device,
            0,
            observe=observe,
            fused_unpack=False,
            maximum_decode_updates=1,
        )

    def owned_tensors(self):
        return (*self.history, *self.leased)

    def materialize(self):
        for target, source, history in zip(self.leased, self.cpu_prompt, self.history, strict=True):
            target[..., : self.prompt_length, :].copy_(
                source, non_blocking=target.device.type == "cuda"
            )
            target[..., self.prompt_length :, :].copy_(history[..., : self.history_length, :])


@torch.inference_mode()
def run(adapter, prefill, pairs):
    """Isolated exact draft forecast; return TWO predictions and no cache owners."""
    from mosaickv.adapters.huggingface.types import DecodeState

    total_started = perf_counter()
    if len(pairs) != len(adapter._language_layers()) or not pairs:
        raise ValueError("one original CPU K/V pair per model layer required")
    n = pairs[0][0].shape[-2]
    if any(pair[0].shape != pairs[0][0].shape for pair in pairs):
        raise ValueError("uniform exact FullKV prompt geometry required")
    if prefill.state.logical_sequence_length != n or prefill.state.next_decode_position != n:
        raise ValueError("original unadvanced prefill and original logical positions required")
    if adapter.model.training:
        raise ValueError("frozen evaluation model required")
    cpu_bytes = backing_bytes(tuple(t for pair in pairs for t in pair))
    versions = signatures(pairs)
    weights = tuple(p._version for p in adapter.model.parameters())

    @torch.inference_mode()
    def task():
        cache = PromptStorageCache()
        cache.layers.extend(
            DraftLayer(pair, device=adapter.device, observe=cache.observe) for pair in pairs
        )
        state = DecodeState(
            cache,
            torch.ones(1, n, dtype=torch.long, device=adapter.device),
            n,
            n,
            n,
            prefill.state.modality_map,
            dict(
                prefill.state.model_state,
                mosaickv_packed_all_valid=True,
                mosaickv_cache_all_valid=True,
            ),
        )
        cuda = adapter.device.type == "cuda"
        if cuda:
            torch.cuda.synchronize()
        started = perf_counter()
        begin = torch.cuda.Event(enable_timing=True) if cuda else None
        end = torch.cuda.Event(enable_timing=True) if cuda else None
        if cuda:
            begin.record()
        with IsolatedStorageTransport(adapter, cache):
            output = adapter.decode_one_token(prefill.next_token_id, state, capture_queries=False)
        if cuda:
            end.record()
            end.synchronize()
        elapsed = perf_counter() - started
        logits = torch.stack(
            tuple(t.detach().float().reshape(-1).cpu() for t in (prefill.logits, output.logits))
        )
        evidence = dict(
            forecast_mode="draft_rollout",
            draft_predictions=2,
            draft_cache_updates=1,
            second_prefill_performed=False,
            source="original_prefill_cpu_KV_not_reference_artifact",
            temporary_cpu_original_kv_bytes=cpu_bytes,
            cpu_snapshots_pinned=all(t.is_pinned() for pair in pairs for t in pair)
            if cuda
            else False,
            draft_peak_gpu_cache_backing_bytes=cache.peak_bytes,
            prompt_gpu_cache_clone_bytes=0,
            layer_updates=[layer.history_length for layer in cache.layers],
            outstanding_layer_leases=sum(bool(layer.leased) for layer in cache.layers),
            logical_sequence_length=output.state.logical_sequence_length,
            next_decode_position=output.state.next_decode_position,
            forecast_wall_seconds=elapsed,
            forecast_cuda_ms=begin.elapsed_time(end) if cuda else None,
            benchmark_labels_accessed=False,
            reference_decode_artifacts_accessed=False,
            whole_request_peak_certified=False,
        )
        return logits, evidence

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="exact-fullkv-draft") as pool:
        logits, evidence = pool.submit(task).result()
    if versions != signatures(pairs):
        raise RuntimeError("FullKV draft mutated original prompt snapshots")
    if weights != tuple(p._version for p in adapter.model.parameters()):
        raise RuntimeError("FullKV draft modified model weights")
    if evidence["outstanding_layer_leases"] or set(evidence["layer_updates"]) != {1}:
        raise RuntimeError("exactly one isolated update and complete lease retirement required")
    evidence["forecast_with_snapshot_hash_validation_wall_seconds"] = perf_counter() - total_started
    return logits, evidence
