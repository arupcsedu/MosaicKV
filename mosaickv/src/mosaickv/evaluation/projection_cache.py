"""Isolated exact source/KV hybrid diagnostic, never a production backend.

All required archived inputs, indices, RoPE tensors, history and reconstructed
tensor workspaces are charged. Original capture/source and model weights are
outside the declared post-construction decode-owned tensor-backing boundary.
"""

from __future__ import annotations

import weakref
from contextlib import nullcontext
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from transformers.cache_utils import CacheLayerMixin

from mosaickv.evaluation.lossless_bf16 import pack_cpu
from mosaickv.evaluation.lossless_prompt_cache import (
    PromptStorageCache,
    PromptStorageLayer,
    backing_bytes,
)
from mosaickv.evaluation.projection_archive import (
    ProjectionArchive,
    build_archive,
    reconstruct_archive,
)


def identity(tensor: Any) -> tuple[str, int, int]:
    storage = tensor.untyped_storage()
    return str(tensor.device), storage.data_ptr(), storage.nbytes()


class ScratchAudit(TorchDispatchMode):
    """Weak, per-ATen-result backing audit; no lifetime extension or weight charge."""

    def __init__(self, *, excluded: set[Any], observe: Any) -> None:
        super().__init__()
        self.excluded, self.observe = excluded, observe
        self.refs: dict[int, Any] = {}
        self.operations = 0

    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        result = func(*args, **(kwargs or {}))

        def remember(value: Any) -> None:
            if isinstance(value, torch.Tensor):
                self.refs[id(value)] = weakref.ref(value)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    remember(item)

        remember(result)
        live = {i: ref for i, ref in self.refs.items() if ref() is not None}
        self.refs = live
        extras = tuple(
            t
            for ref in live.values()
            if (t := ref()) is not None and identity(t) not in self.excluded
        )
        self.operations += 1
        self.observe(extras)
        return result


class ProjectionLayer(PromptStorageLayer):
    def __init__(
        self,
        archive: ProjectionArchive,
        attention: Any,
        *,
        observe: Any,
        maximum_decode_updates: int,
        audit_workspace: bool,
    ) -> None:
        CacheLayerMixin.__init__(self)
        self.archive, self.attention = archive, attention
        self.audit_workspace = audit_workspace
        self.workspace_operations = 0
        self.excluded_weights = {identity(t) for t in attention.parameters()}
        heads, length = archive.selected.shape
        prefix = (*archive.source.buffers, archive.union, archive.selected)
        self._initialize_storage(
            (1, heads, length, attention.head_dim),
            archive.union.device,
            backing_bytes(prefix),
            observe=observe,
            fused_unpack=False,
            maximum_decode_updates=maximum_decode_updates,
        )

    def owned_tensors(self) -> tuple[Any, ...]:
        return (*self.archive.storage_tensors(), *self.history, *self.leased)

    def materialize(self) -> None:
        excluded = self.excluded_weights | {identity(t) for t in self.owned_tensors()}
        mode = (
            ScratchAudit(excluded=excluded, observe=self.observe) if self.audit_workspace else None
        )
        with mode if mode is not None else nullcontext():
            prompts = reconstruct_archive(self.archive, self.attention)
        if mode is not None:
            self.workspace_operations += mode.operations
        self.observe(prompts)
        for source, dest, history in zip(prompts, self.leased, self.history, strict=True):
            dest[..., : self.prompt_length, :].copy_(source)
            dest[..., self.prompt_length :, :].copy_(history[..., : self.history_length, :])
        del source, dest, history, prompts


class ProjectionCache(PromptStorageCache):
    def __init__(self, *, audit_workspace: bool) -> None:
        super().__init__()
        self.audit_workspace = audit_workspace

    def get_seq_length(self, layer_idx: int | None = None) -> int:
        # Native HF mask construction needs the maximum physical extent, not
        # an assumption that the first layer is always the longest. Explicit
        # layer queries still report their actual unpadded length.
        if layer_idx is None:
            return max((layer.get_seq_length() for layer in self.layers), default=0)
        return self.layers[layer_idx].get_seq_length() if layer_idx < len(self.layers) else 0

    def get_mask_sizes(self, cache_position: Any, layer_idx: int) -> tuple[int, int]:
        # Llama creates one causal mask before visiting its layers. Native eager
        # attention slices that maximum-extent mask to each actual key length.
        del layer_idx
        return self.get_seq_length() + cache_position.shape[0], 0

    def storage_record(self) -> dict[str, Any]:
        row = super().storage_record()
        row.update(
            boundary="post_construction_archive_and_history_and_reconstruction_tensor_results",
            includes="encoded_KV_or_normalized_source,union_and_head_indices,shared_rotary,raw_history,lease,observed_reconstruction_results",
            excludes="inactive_original_KV_and_normalized_capture,weights,attention_activations,internal_GEMM_workspace,allocator_reserved",
            reconstruction_tensor_results_observed=self.audit_workspace,
            reconstruction_operator_observations=sum(
                getattr(layer, "workspace_operations", 0) for layer in self.layers
            ),
            peak_admissible_only_for_declared_tensor_boundary=self.audit_workspace,
        )
        return row


def source_layer_choices(
    source_costs: list[int], kv_costs: list[int], shared_cost: int, maximum_layers: int
) -> tuple[int, ...]:
    """Exact cardinality saving rule with one shared-RoPE activation charge."""
    if (
        not source_costs
        or len(source_costs) != len(kv_costs)
        or any(type(v) is not int or v < 0 for v in (*source_costs, *kv_costs, shared_cost))
        or type(maximum_layers) is not int
        or not 0 <= maximum_layers <= len(source_costs)
    ):
        raise ValueError("nonnegative integer costs and valid source-layer bound required")
    candidates = sorted(range(len(source_costs)), key=lambda i: (source_costs[i] - kv_costs[i], i))
    selected = tuple(i for i in candidates[:maximum_layers] if source_costs[i] < kv_costs[i])
    if sum(kv_costs[i] - source_costs[i] for i in selected) <= shared_cost:
        return ()
    return tuple(sorted(selected))


def construct_projection_cache(
    adapter: Any,
    prefill: Any,
    capture: Any,
    positions: Any,
    *,
    policy: str = "hybrid",
    audit_workspace: bool = True,
    maximum_decode_updates: int = 32,
    maximum_source_layers: int = 8,
) -> ProjectionCache:
    if policy not in ("hybrid", "source_all", "kv_only", "bounded_sources"):
        raise ValueError("explicit hybrid, source_all, kv_only or bounded_sources policy required")
    source, _ = adapter._legacy_layers(prefill.state.past_key_values)
    if len(source) != len(positions) or len(capture.records) != len(source):
        raise ValueError("complete captured layers and selections required")
    cache = ProjectionCache(audit_workspace=audit_workspace)
    prepared = []
    shared: list[Any] = []
    for layer_index, ((key, value), selected) in enumerate(zip(source, positions, strict=True)):
        archive = build_archive(capture.records[layer_index], selected, encoded=True)
        index = archive.selected[None, :, :, None].expand(
            1, key.shape[1], len(selected[0]), key.shape[-1]
        )
        gathered = (key.gather(2, index), value.gather(2, index))
        if key.device.type == "cuda":
            from mosaickv.evaluation.lossless_bf16_kernel import encode
        else:
            encode = pack_cpu
        bodies = tuple(encode(t) for t in gathered)
        source_cost = backing_bytes(archive.storage_tensors())
        kv_cost = sum(body.storage_bytes for body in bodies)
        private_cost = backing_bytes((*archive.source.buffers, archive.union, archive.selected))
        prepared.append((archive, bodies, source_cost, kv_cost, private_cost))
    bounded: tuple[int, ...] = ()
    if policy == "bounded_sources":
        rotary_ids = {tuple(identity(t) for t in (a.cos, a.sin)) for a, *_ in prepared}
        if len(rotary_ids) != 1:
            raise NotImplementedError(
                "bounded format allocation requires shared original RoPE storage"
            )
        fee = backing_bytes((prepared[0][0].cos, prepared[0][0].sin))
        bounded = source_layer_choices(
            [r[4] for r in prepared],
            [r[3] for r in prepared],
            fee,
            min(maximum_source_layers, len(prepared)),
        )
    for layer_index, (archive, bodies, source_cost, kv_cost, private_cost) in enumerate(prepared):
        chosen = (
            policy == "source_all"
            or (policy == "hybrid" and source_cost < kv_cost)
            or (policy == "bounded_sources" and layer_index in bounded)
        )
        if chosen:
            layer = ProjectionLayer(
                archive,
                adapter._language_layers()[layer_index].self_attn,
                observe=cache.observe,
                maximum_decode_updates=maximum_decode_updates,
                audit_workspace=audit_workspace,
            )
            shared.extend((archive.cos, archive.sin))
        else:
            layer = PromptStorageLayer(
                *bodies,
                observe=cache.observe,
                fused_unpack=True,
                maximum_decode_updates=maximum_decode_updates,
            )
        cache.layers.append(layer)
        cache.build_records.append(
            dict(
                layer=layer_index,
                format="projection_source" if chosen else "encoded_KV",
                source_component_cost_with_rotary=source_cost,
                encoded_KV_cost=kv_cost,
                original_selected_positions_preserved=True,
                source_private_bytes=private_cost,
                source_layer_bound=maximum_source_layers if policy == "bounded_sources" else None,
            )
        )
    cache.shared_tensors = tuple(shared)
    cache.shared_bytes = backing_bytes(cache.shared_tensors)
    cache.initial_bytes = backing_bytes(
        tuple(t for layer in cache.layers for t in layer.owned_tensors())
    )
    cache.observe()
    return cache


def runtime_controls(
    adapter: Any,
    prefill: Any,
    capture: Any,
    positions: Any,
    *,
    policies: tuple[str, ...] = ("kv_only", "hybrid", "bounded_sources"),
) -> dict[str, Any]:
    """32-token memory-audited own-greedy control, NOT a latency benchmark."""
    import hashlib
    import time

    from mosaickv.adapters.huggingface.types import DecodeState
    from mosaickv.evaluation.lossless_lifetime import standard_greedy_reference
    from mosaickv.evaluation.lossless_prompt_cache import IsolatedStorageTransport, construct_cache
    from mosaickv.evaluation.worker_support import settle

    reference = standard_greedy_reference(adapter, prefill, positions)
    settle(adapter)
    rows: dict[str, Any] = {}
    for policy in policies:
        if policy == "kv_only":
            cache = construct_cache(
                adapter,
                prefill.state.past_key_values,
                positions,
                encode=True,
                gpu_encoder=adapter.device.type == "cuda",
                fused_unpack=True,
            )
        else:
            cache = construct_projection_cache(
                adapter,
                prefill,
                capture,
                positions,
                policy=policy,
                maximum_source_layers=8,
                audit_workspace=True,
            )
        length = max(len(h[0]) for h in positions)
        state = DecodeState(
            cache,
            torch.ones(1, length, device=adapter.device, dtype=torch.long),
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
        token, values = prefill.next_token_id, [prefill.logits.detach().reshape(-1).float()]
        if adapter.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        with IsolatedStorageTransport(adapter, cache):
            for _ in range(31):
                output = adapter.decode_one_token(token, state, capture_queries=False)
                values.append(output.logits.detach().reshape(-1).float())
                token = output.next_token_id
        if adapter.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        logits = torch.stack(values).cpu()
        if not bool(torch.isfinite(logits).all()) or not torch.equal(logits, reference):
            raise RuntimeError(f"{policy} changed exact reference logits/tokens")
        if state.next_decode_position != prefill.state.next_decode_position + 31:
            raise RuntimeError("source archive changed logical positions")
        storage = cache.storage_record()
        if storage["outstanding_layer_leases"] != 0 or storage["history_updates_per_layer"] != [
            31
        ] * len(positions):
            raise RuntimeError("incomplete or leaked archive decode state")
        rows[policy] = dict(
            bitwise_logits_equal_reference=True,
            maximum_logit_difference=0.0,
            used_fullkv_fallback=False,
            original_selected_positions_unchanged=True,
            logits_sha256=hashlib.sha256(logits.numpy().tobytes()).hexdigest(),
            generated_token_ids=logits.argmax(-1).tolist(),
            tokens=32,
            storage=storage,
            instrumented_wall_seconds=elapsed,
            wall_time_not_comparable_latency=True,
            source_layer_bound=8 if policy == "bounded_sources" else None,
            source_archived_layers=sum(
                r.get("format") == "projection_source" for r in cache.build_records
            ),
            process_cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated()
            if adapter.device.type == "cuda"
            else None,
            process_cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved()
            if adapter.device.type == "cuda"
            else None,
        )
        del output, values, logits, cache, state
        settle(adapter)
    return rows
