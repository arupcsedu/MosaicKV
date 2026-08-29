"""vLLM 0.10.2 native whole-block compaction plugin.

The patch keeps scheduler-visible logical token counts unchanged.  Dropped
request-owned blocks are replaced by vLLM's null block in the allocator table,
while the worker builds a compact attention-only block table and active
sequence length.  Query and M-RoPE positions therefore remain logical.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import math
import os
import time
from typing import Any

from mosaickv.backends.native_blocks import (
    require_scratch_path,
    select_native_blocks,
    validated_native_block_plan,
    write_native_audit,
)

_VERSION = "0.10.2"
_ENABLE_ENV = "MOSAICKV_NATIVE_VLLM"
_RATIO_ENV = "MOSAICKV_NATIVE_RETENTION_RATIO"
_BLOCK_ENV = "MOSAICKV_NATIVE_BLOCK_SIZE"
_AUDIT_ENV = "MOSAICKV_NATIVE_AUDIT_DIR"
_VIT_SDPA_ENV = "MOSAICKV_VLLM_VIT_SDPA"


def _enabled() -> bool:
    return os.environ.get(_ENABLE_ENV) == "1"


def _settings() -> tuple[float, int, str]:
    try:
        ratio = float(os.environ[_RATIO_ENV])
        block_size = int(os.environ[_BLOCK_ENV])
        audit_directory = os.environ[_AUDIT_ENV]
    except KeyError as error:
        raise RuntimeError(
            f"native vLLM is missing required environment {error.args[0]}"
        ) from error
    # Reuse the selector's strict validation.
    select_native_blocks(1, block_size, ratio)
    try:
        require_scratch_path(audit_directory)
    except ValueError as error:
        raise RuntimeError(str(error)) from error
    return ratio, block_size, audit_directory


def _patch_vit_attention_to_sdpa(
    platform: Any | None = None,
    torch_sdpa_backend: Any | None = None,
) -> None:
    """Route cache-free ViT attention through vLLM's Torch SDPA branch.

    vLLM 0.10.2 otherwise sends LLaVA's CLIP encoder through xFormers.
    xFormers 0.0.32.post1 rejects the common environment's external
    FlashAttention 2.8.3 before inference. Decoder paged attention is not
    affected by this narrowly scoped platform hook.
    """

    if platform is None or torch_sdpa_backend is None:
        # ``_Backend`` is deliberately private in the audited vLLM 0.10.2
        # boundary. Resolve it dynamically so this version-pinned plugin does
        # not claim that vLLM exposes a public typed API here.
        vllm_platforms = importlib.import_module("vllm.platforms")
        platform = vllm_platforms.current_platform
        torch_sdpa_backend = vllm_platforms._Backend.TORCH_SDPA
    platform_class: Any = type(platform)
    if getattr(platform_class, "_mosaickv_vit_sdpa_patch_v1", False):
        return

    def get_vit_attn_backend(cls: Any, support_fa: bool = False) -> Any:
        del cls, support_fa
        return torch_sdpa_backend

    platform_class.get_vit_attn_backend = classmethod(get_vit_attn_backend)
    platform_class._mosaickv_vit_sdpa_patch_v1 = True


def _request_selection(
    request: Any,
    *,
    logical_positions: int,
    block_size: int,
    retention_ratio: float,
) -> tuple[Any, dict[str, Any]]:
    """Resolve an optional hash-bound common-selector plan from SamplingParams."""

    sampling = getattr(request, "sampling_params", None)
    extra_args = getattr(sampling, "extra_args", None)
    native = extra_args.get("mosaickv_native") if isinstance(extra_args, dict) else None
    if not isinstance(native, dict) or "selected_blocks" not in native:
        return (
            select_native_blocks(logical_positions, block_size, retention_ratio),
            {"selector_source": "transport_validation_fallback"},
        )
    if int(native.get("logical_positions", -1)) != logical_positions:
        raise RuntimeError(
            "common-selector plan logical length differs from vLLM prefill: "
            f"plan={native.get('logical_positions')}, vllm={logical_positions}"
        )
    if int(native.get("block_size", -1)) != block_size:
        raise RuntimeError("common-selector plan block size differs from vLLM")
    plan_retention_ratio = float(native.get("retention_ratio", -1.0))
    if not math.isfinite(plan_retention_ratio) or not 0 < plan_retention_ratio <= 1:
        raise RuntimeError("common-selector plan retention ratio must be finite and in (0, 1]")
    raw_selected = native.get("selected_blocks")
    if not isinstance(raw_selected, list) or not all(
        isinstance(value, int) and not isinstance(value, bool) for value in raw_selected
    ):
        raise RuntimeError("common-selector selected_blocks must be an integer list")
    raw_mandatory = native.get("mandatory_blocks")
    if not isinstance(raw_mandatory, list) or not all(
        isinstance(value, int) and not isinstance(value, bool) for value in raw_mandatory
    ):
        raise RuntimeError("common-selector mandatory_blocks must be an integer list")
    raw_selected_count = native.get("selected_block_count")
    if (
        not isinstance(raw_selected_count, int)
        or isinstance(raw_selected_count, bool)
        or raw_selected_count < 1
    ):
        raise RuntimeError("common-selector selected_block_count must be a positive integer")
    policy = str(native.get("policy", ""))
    selection = validated_native_block_plan(
        logical_positions,
        block_size,
        plan_retention_ratio,
        tuple(raw_selected),
        policy=policy,
        mandatory_blocks=tuple(raw_mandatory),
        expected_selected_blocks=raw_selected_count,
    )
    catalog_sha = str(native.get("catalog_sha256", ""))
    if len(catalog_sha) != 64:
        raise RuntimeError("common-selector catalog SHA-256 is invalid")
    planning_seconds = float(native.get("selector_planning_seconds", -1.0))
    source_trace_sha = str(native.get("source_trace_sha256", ""))
    method = str(native.get("method", ""))
    if not method or not math.isfinite(planning_seconds) or planning_seconds < 0:
        raise RuntimeError("common-selector method/planning timing is invalid")
    if len(source_trace_sha) != 64:
        raise RuntimeError("common-selector source-trace SHA-256 is invalid")
    return selection, {
        "selector_source": "hash_bound_common_hf_selector_projection",
        "selector_method": method,
        "selector_catalog_sha256": catalog_sha,
        "selector_planning_seconds": planning_seconds,
        "source_trace_sha256": source_trace_sha,
    }


def _compress_ready_requests(scheduler: Any, scheduler_output: Any) -> None:
    ratio, configured_block_size, audit_directory = _settings()
    manager = scheduler.kv_cache_manager
    if manager.enable_caching:
        raise RuntimeError("native vLLM requires prefix caching to be disabled")
    coordinators = manager.coordinator.single_type_managers
    if len(coordinators) != 1 or coordinators[0].__class__.__name__ != "FullAttentionManager":
        raise RuntimeError("native vLLM supports exactly one full-attention KV cache group")
    group = coordinators[0]
    if group.block_size != configured_block_size:
        raise RuntimeError(
            "native vLLM block-size mismatch: "
            f"engine={group.block_size}, configured={configured_block_size}"
        )
    null_block = manager.coordinator.block_pool.null_block
    for request_id in scheduler_output.num_scheduled_tokens:
        request = scheduler.requests.get(request_id)
        if request is None or getattr(request, "_mosaickv_native_compacted", False):
            continue
        if request.num_computed_tokens < request.num_prompt_tokens:
            continue
        if len(request.output_token_ids) != 0:
            raise RuntimeError("native vLLM reached first decode without a prefill transaction")
        started = time.perf_counter()
        blocks = group.req_to_blocks[request_id]
        selection, selector_provenance = _request_selection(
            request,
            logical_positions=request.num_prompt_tokens,
            block_size=configured_block_size,
            retention_ratio=ratio,
        )
        if len(blocks) != selection.source_blocks:
            raise RuntimeError(
                "native vLLM requires prompt allocation to contain exactly its logical blocks; "
                f"allocated={len(blocks)}, logical={selection.source_blocks}"
            )
        selected = set(selection.selected_blocks)
        dropped_physical_ids: list[int] = []
        for logical_index, block in enumerate(tuple(blocks)):
            if logical_index in selected:
                if block.is_null:
                    raise RuntimeError("native vLLM selected an unexpected null source block")
                continue
            if block.is_null:
                raise RuntimeError("native vLLM source table was already sparse")
            dropped_physical_ids.append(block.block_id)
            manager.coordinator.block_pool.free_blocks([block])
            blocks[logical_index] = null_block
        request._mosaickv_native_compacted = True
        request._mosaickv_force_block_reset = True
        compression_seconds = time.perf_counter() - started
        write_native_audit(
            audit_directory,
            "vllm",
            request_id,
            {
                "schema_version": 1,
                "status": "scheduler_transaction_committed",
                "selection": selection.to_json_object(),
                "dropped_physical_block_ids": dropped_physical_ids,
                "retained_physical_block_ids": [
                    block.block_id for block in blocks if not block.is_null
                ],
                "null_block_id": null_block.block_id,
                "compression_seconds": compression_seconds,
                "logical_next_position": request.num_computed_tokens,
                "prefix_cache_disabled": True,
                **selector_provenance,
            },
        )


def _patch_scheduler() -> None:
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager

    scheduler_class: Any = Scheduler
    if getattr(scheduler_class, "_mosaickv_native_patch_v1", False):
        return
    original_update = Scheduler.update_from_output
    original_cached = Scheduler._make_cached_request_data

    def update_from_output(self: Any, scheduler_output: Any, model_runner_output: Any) -> Any:
        _compress_ready_requests(self, scheduler_output)
        return original_update(self, scheduler_output, model_runner_output)

    def make_cached_request_data(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_cached(self, *args, **kwargs)
        for index, request_id in enumerate(result.req_ids):
            request = self.requests[request_id]
            if not getattr(request, "_mosaickv_force_block_reset", False):
                continue
            blocks = self.kv_cache_manager.coordinator.get_blocks(request_id)
            result.new_block_ids[index] = tuple(
                [block.block_id for block in group] for group in blocks
            )
            result.resumed_from_preemption[index] = True
            request._mosaickv_force_block_reset = False
        return result

    scheduler_class.update_from_output = update_from_output
    scheduler_class._make_cached_request_data = make_cached_request_data
    scheduler_class._mosaickv_native_patch_v1 = True

    manager_class: Any = FullAttentionManager
    if not getattr(manager_class, "_mosaickv_native_free_patch_v1", False):

        def free_non_null(self: Any, request_id: str) -> None:
            # Upstream FullAttentionManager never owns null placeholders, so
            # its inherited free() decrements every table entry.  Native
            # compaction introduces null entries whose reference count is
            # intentionally not maintained by vLLM.  Filter them explicitly.
            blocks = self.req_to_blocks.pop(request_id, [])
            self.block_pool.free_blocks(block for block in reversed(blocks) if not block.is_null)
            self.num_cached_block.pop(request_id, None)

        manager_class.free = free_non_null
        manager_class._mosaickv_native_free_patch_v1 = True


def _active_computed_positions(block_ids: list[int], logical: int, block_size: int) -> int:
    if logical == 0:
        return 0
    logical_blocks = math.ceil(logical / block_size)
    if len(block_ids) < logical_blocks:
        raise RuntimeError("native vLLM worker block table is shorter than the logical sequence")
    last = logical_blocks - 1
    tail = logical - last * block_size
    active = sum(
        tail if index == last else block_size
        for index, block_id in enumerate(block_ids[:logical_blocks])
        if block_id != 0
    )
    if active < 1:
        raise RuntimeError("native vLLM compacted attention table is empty")
    return active


def _patch_worker() -> None:
    import torch
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    runner_class: Any = GPUModelRunner
    if getattr(runner_class, "_mosaickv_native_patch_v1", False):
        return
    original_update = GPUModelRunner._update_states
    original_prepare = GPUModelRunner._prepare_inputs

    def update_states(self: Any, scheduler_output: Any) -> None:
        reset_indices: list[int] = []
        request_data = scheduler_output.scheduled_cached_reqs
        for index, request_id in enumerate(request_data.req_ids):
            if not request_data.resumed_from_preemption[index]:
                continue
            existing = self.input_batch.req_id_to_index.get(request_id)
            if existing is None:
                continue
            new_ids = request_data.new_block_ids[index]
            if new_ids is None:
                continue
            reset_indices.append(existing)
        # Upstream append_row assumes a genuinely resumed request is absent
        # from the persistent batch.  For an in-place sparse-table reset, make
        # the existing row empty so the same code writes a replacement.
        for request_index in reset_indices:
            for block_table in self.input_batch.block_table.block_tables:
                block_table.num_blocks_per_row[request_index] = 0
        original_update(self, scheduler_output)

    def prepare_inputs(self: Any, scheduler_output: Any) -> Any:
        result = original_prepare(self, scheduler_output)
        attn_metadata, _, _, scheduled, _, _ = result
        request_ids = self.input_batch.req_ids
        compressed = False
        compact_rows: list[list[int]] = []
        active_lengths: list[int] = []
        block_size = self.input_batch.block_table[0].block_size
        for request_index, request_id in enumerate(request_ids):
            request = self.requests[request_id]
            if len(request.block_ids) != 1:
                raise RuntimeError("native vLLM worker supports one KV cache group")
            logical = int(self.input_batch.num_computed_tokens_cpu[request_index])
            block_ids = request.block_ids[0]
            logical_blocks = math.ceil(logical / block_size) if logical else 0
            visible = [block_id for block_id in block_ids[:logical_blocks] if block_id != 0]
            if any(block_id == 0 for block_id in block_ids[:logical_blocks]):
                compressed = True
                if int(scheduled[request_index]) != 1:
                    raise RuntimeError(
                        "native vLLM compressed requests require one-token deterministic decode"
                    )
            # Include a newly allocated decode block when logical lies exactly
            # on a block boundary.
            required_after = math.ceil((logical + int(scheduled[request_index])) / block_size)
            for block_id in block_ids[logical_blocks:required_after]:
                if block_id == 0:
                    raise RuntimeError("native vLLM allocated a null decode block")
                visible.append(block_id)
            compact_rows.append(visible)
            active_lengths.append(
                _active_computed_positions(block_ids, logical, block_size)
                + int(scheduled[request_index])
                if logical
                else int(scheduled[request_index])
            )
        if not compressed:
            return result
        width = max(len(row) for row in compact_rows)
        compact_table = torch.zeros(
            (len(compact_rows), width), dtype=torch.int32, device=self.device
        )
        for row_index, row in enumerate(compact_rows):
            compact_table[row_index, : len(row)] = torch.tensor(
                row, dtype=torch.int32, device=self.device
            )
        active_seq_lens = torch.tensor(active_lengths, dtype=torch.int32, device=self.device)
        seen: set[int] = set()
        for metadata in attn_metadata.values():
            if id(metadata) in seen:
                continue
            seen.add(id(metadata))
            if metadata.__class__.__name__ != "FlashAttentionMetadata":
                raise RuntimeError(
                    "native vLLM is audited only for FlashAttentionMetadata in eager execution"
                )
            if metadata.use_cascade or metadata.scheduler_metadata is not None:
                raise RuntimeError("native vLLM requires cascade/AOT scheduling to be disabled")
            metadata.block_table = compact_table
            metadata.seq_lens = active_seq_lens
            metadata.max_seq_len = max(active_lengths)
        return result

    runner_class._update_states = update_states
    runner_class._prepare_inputs = prepare_inputs
    runner_class._mosaickv_native_patch_v1 = True


def register() -> None:
    """Install the audited patch in every vLLM scheduler/worker process."""

    native_enabled = _enabled()
    vit_sdpa_enabled = os.environ.get(_VIT_SDPA_ENV) == "1"
    if not native_enabled and not vit_sdpa_enabled:
        return
    installed = importlib.metadata.version("vllm")
    if installed != _VERSION:
        raise RuntimeError(f"MosaicKV audited vLLM {_VERSION}; installed {installed}")
    if vit_sdpa_enabled:
        _patch_vit_attention_to_sdpa()
    if native_enabled:
        _settings()
        _patch_scheduler()
        _patch_worker()


__all__ = ["_patch_vit_attention_to_sdpa", "register"]
