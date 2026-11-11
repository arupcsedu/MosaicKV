"""Layer-local lossless normalized-source prefill, not an adaptive selector.

Use this encoder only inside the existing ``SealAfterAttention`` scope with a
fresh ``RetiringPrefillCache``. Each native attention consumes its original K/V
before the hook constructs an exact source packet and retires that layer's raw
cache. No global activation capture or future-token evidence is required.

Observed ownership includes encoder tensors, not opaque vendor workspace or
model activations outside the encoder. This does not certify a whole-request
GPU peak or a latency improvement.
"""

from __future__ import annotations

import torch

from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.lossless_source_control import KIND, construct
from mosaickv.evaluation.streaming_prefill import RetiringPrefillCache


class StreamingLosslessSource:
    """Request-private encoder for the unchanged exact BF16 source schema.

    ``prepared.logical_sequence_length`` is the fully expanded prompt length.
    Optional per-layer byte caps are fixed before prefill. A declined packet
    propagates its failure: neither a raw-K/V fallback nor a partial cache is
    published as a successful encoded request.
    """

    def __init__(self, adapter, prepared, cache, *, layer_byte_caps=None):
        self.layers = tuple(adapter._language_layers())
        length = prepared.logical_sequence_length
        if (
            type(length) is not int
            or length < 1
            or type(cache) is not RetiringPrefillCache
            or len(cache.layers) != len(self.layers)
            or not self.layers
            or cache.next_layer != 0
            or cache.extra_tensors
            or any(
                layer.keys is not None or layer.values is not None or layer.packet is not None
                for layer in cache.layers
            )
        ):
            raise ValueError("One fresh complete retiring cache and expanded prompt required")
        if layer_byte_caps is not None and (
            not isinstance(layer_byte_caps, (tuple, list))
            or len(layer_byte_caps) != len(self.layers)
            or any(type(cap) is not int or cap < 1 for cap in layer_byte_caps)
        ):
            raise ValueError("One positive predetermined byte cap per layer required")
        self.length, self.cache = length, cache
        self.caps = (
            (None,) * len(self.layers) if layer_byte_caps is None else tuple(layer_byte_caps)
        )
        # Metadata only; no original activation, K/V, or phase owner is kept.
        self.records = []
        self.finished = self.failed = False
        self.failure = None

    @torch.no_grad()
    def __call__(self, index, attention, original, args, kwargs, output):
        if (
            self.finished
            or self.failed
            or type(index) is not int
            or index != len(self.records)
            or index != self.cache.next_layer
            or not 0 <= index < len(self.layers)
            or attention is not self.layers[index].self_attn
        ):
            raise ValueError("One original ordered layer-local source transaction required")
        layer = self.cache.layers[index]
        if len(original) != 2 or original[0] is not layer.keys or original[1] is not layer.values:
            raise ValueError("Only the current original retiring-cache K/V may be encoded")
        normalized = kwargs.get("hidden_states", args[0] if args else None)
        phases, positions = kwargs.get("position_embeddings"), kwargs.get("cache_position")
        try:
            if (
                not isinstance(normalized, torch.Tensor)
                or normalized.ndim != 3
                or normalized.shape[:2] != (1, self.length)
                or not isinstance(phases, (tuple, list))
                or len(phases) != 2
                or not isinstance(positions, torch.Tensor)
                or positions.shape != (self.length,)
                or not torch.equal(positions, torch.arange(self.length, device=positions.device))
                or any(pair.shape[-2] != self.length for pair in original)
            ):
                raise ValueError(
                    "Original normalized input, phases and contiguous positions required"
                )
            packet = construct(
                normalized,
                original,
                attention,
                phases,
                positions,
                maximum_persistent_bytes=self.caps[index],
                workspace_observer=self.cache.observe,
            )
            self.records.append(dict(layer=index, **packet.diagnostics))
            return packet
        except BaseException as error:
            self.failed = True
            self.failure = dict(layer=index, error_type=type(error).__name__)
            raise

    def finish(self):
        if self.finished or self.failed or len(self.records) != len(self.layers):
            raise RuntimeError("All exact source layers must complete once without failure")
        packets = self.cache.formats()
        if any(
            packet.kind != KIND
            or tuple(packet.shape) != tuple(self.cache.layers[index].packet.shape)
            or packet.component_bytes != self.records[index]["persistent_bytes"]
            or self.cache.get_seq_length(index) != self.length
            for index, packet in enumerate(packets)
        ):
            raise RuntimeError("Published lossless source packets changed after construction")
        storage = self.cache.record()
        persistent = backing_bytes(
            tuple(tensor for packet in packets for tensor in packet.tensors())
        )
        if persistent != storage["final_prefill_cache_bytes"] or persistent != sum(
            record["persistent_bytes"] for record in self.records
        ):
            raise RuntimeError("Complete independent source ownership accounting differs")
        self.finished = True
        self.layers = ()
        return {
            "kind": KIND,
            "construction": "layer_local_after_native_attention_before_next_layer",
            "global_source_or_query_capture": False,
            "prompt_selection_or_precision_policy_changed": False,
            "all_prompt_positions_retained": True,
            "original_KV_all_word_bitwise_validation": True,
            "persistent_bytes": persistent,
            "original_raw_KV_bytes": sum(record["raw_kv_bytes"] for record in self.records),
            "layer_records": list(self.records),
            "prefill_storage": storage,
            "whole_request_gpu_peak_certified": False,
            "benchmark_quality_or_latency_claim": False,
        }
