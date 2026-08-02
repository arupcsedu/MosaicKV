"""SGLang 0.5.1.post3 native token-block compaction patch."""

from __future__ import annotations

import importlib.metadata
import os
import time
from typing import Any

from mosaickv.backends.native_blocks import (
    require_scratch_path,
    select_native_blocks,
    write_native_audit,
)

_VERSION = "0.5.1.post3"
_ENABLE_ENV = "MOSAICKV_NATIVE_SGLANG"
_RATIO_ENV = "MOSAICKV_NATIVE_RETENTION_RATIO"
_BLOCK_ENV = "MOSAICKV_NATIVE_BLOCK_SIZE"
_AUDIT_ENV = "MOSAICKV_NATIVE_AUDIT_DIR"


def run_scheduler_process_with_native_patch(*args: Any, **kwargs: Any) -> Any:
    """Spawn-safe SGLang scheduler entry point.

    SGLang uses multiprocessing ``spawn`` for its GPU scheduler, so class
    monkeypatches installed in the HTTP parent are not inherited.  The engine
    process target must itself install the version-pinned transaction before
    constructing ``Scheduler``.
    """

    install()
    from sglang.srt.managers.scheduler import run_scheduler_process

    return run_scheduler_process(*args, **kwargs)


def _enabled() -> bool:
    return os.environ.get(_ENABLE_ENV) == "1"


def _settings() -> tuple[float, int, str]:
    try:
        ratio = float(os.environ[_RATIO_ENV])
        block_size = int(os.environ[_BLOCK_ENV])
        audit_directory = os.environ[_AUDIT_ENV]
    except KeyError as error:
        raise RuntimeError(
            f"native SGLang is missing required environment {error.args[0]}"
        ) from error
    select_native_blocks(1, block_size, ratio)
    try:
        require_scratch_path(audit_directory)
    except ValueError as error:
        raise RuntimeError(str(error)) from error
    return ratio, block_size, audit_directory


def _selected_token_positions(logical: int, block_size: int, blocks: tuple[int, ...]) -> list[int]:
    positions: list[int] = []
    for block in blocks:
        start = block * block_size
        positions.extend(range(start, min(start + block_size, logical)))
    return positions


def _validate_scheduler(scheduler: Any) -> None:
    if scheduler.enable_overlap:
        raise RuntimeError("native SGLang requires overlap scheduling disabled")
    if not scheduler.server_args.disable_cuda_graph:
        raise RuntimeError("native SGLang requires CUDA graphs disabled")
    if not scheduler.server_args.disable_radix_cache:
        raise RuntimeError("native SGLang requires Radix/prefix caching disabled")
    if scheduler.server_args.attention_backend != "torch_native":
        raise RuntimeError("native SGLang requires the torch_native attention backend")
    if not scheduler.spec_algorithm.is_none():
        raise RuntimeError("native SGLang does not support speculative decoding")
    if scheduler.token_to_kv_pool_allocator.page_size != 1:
        raise RuntimeError("native SGLang currently requires allocator page_size=1")


def _release_active_request(cache: Any, request: Any) -> None:
    active_prompt = getattr(request, "_mosaickv_active_prompt_positions", None)
    if active_prompt is None:
        raise RuntimeError("native SGLang release omitted active prompt ownership")
    active_cached = int(active_prompt) + max(0, len(request.output_ids) - 1)
    locations = cache.req_to_token_pool.req_to_token[request.req_pool_idx, :active_cached]
    cache.token_to_kv_pool_allocator.free(locations)
    cache.req_to_token_pool.free(request.req_pool_idx)


def _patch_output_processor() -> None:
    from sglang.srt.managers.scheduler_output_processor_mixin import (
        SchedulerOutputProcessorMixin,
    )

    if getattr(SchedulerOutputProcessorMixin, "_mosaickv_native_patch_v1", False):
        return
    original = SchedulerOutputProcessorMixin.process_batch_result_prefill

    def process_prefill(self: Any, batch: Any, result: Any, launch_done: Any = None) -> Any:
        _validate_scheduler(self)
        ratio, block_size, audit_directory = _settings()
        changed = False
        for request_index, request in enumerate(batch.reqs):
            if request.is_retracted or getattr(request, "_mosaickv_native_compacted", False):
                continue
            if request.is_chunked > 0:
                raise RuntimeError("native SGLang does not support chunked prefill")
            logical = int(batch.seq_lens[request_index].item())
            if logical != len(request.fill_ids):
                raise RuntimeError(
                    "native SGLang prefill transaction requires a complete logical prompt"
                )
            started = time.perf_counter()
            selection = select_native_blocks(logical, block_size, ratio)
            selected_positions = _selected_token_positions(
                logical, block_size, selection.selected_blocks
            )
            row = self.req_to_token_pool.req_to_token[request.req_pool_idx]
            source_locations = row[:logical].clone()
            selected_locations = source_locations[selected_positions]
            selected_mask = source_locations.new_zeros(logical, dtype=bool)
            selected_mask[selected_positions] = True
            dropped_locations = source_locations[~selected_mask]
            if selected_locations.numel() != selection.selected_positions:
                raise RuntimeError("native SGLang selected-position accounting mismatch")
            row[: selection.selected_positions] = selected_locations
            if selection.selected_positions < logical:
                row[selection.selected_positions : logical] = 0
                self.token_to_kv_pool_allocator.free(dropped_locations)
                changed = True
            batch.seq_lens[request_index] = selection.selected_positions
            request._mosaickv_native_compacted = True
            request._mosaickv_active_prompt_positions = selection.selected_positions
            request._mosaickv_logical_prompt_positions = logical
            write_native_audit(
                audit_directory,
                "sglang",
                request.rid,
                {
                    "schema_version": 1,
                    "status": "scheduler_transaction_committed",
                    "selection": selection.to_json_object(),
                    "dropped_physical_token_ids": [
                        int(value) for value in dropped_locations.cpu().tolist()
                    ],
                    "retained_physical_token_ids": [
                        int(value) for value in selected_locations.cpu().tolist()
                    ],
                    "compression_seconds": time.perf_counter() - started,
                    "logical_next_position": logical,
                    "radix_cache_disabled": True,
                },
            )
        if changed:
            batch.seq_lens_sum = int(batch.seq_lens.sum().item())
        return original(self, batch, result, launch_done)

    SchedulerOutputProcessorMixin.process_batch_result_prefill = process_prefill
    SchedulerOutputProcessorMixin._mosaickv_native_patch_v1 = True


def _patch_forward_positions() -> None:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, clamp_position

    if getattr(ForwardBatch, "_mosaickv_native_patch_v1", False):
        return
    original = ForwardBatch.init_new.__func__

    def init_new(cls: type[Any], batch: Any, model_runner: Any) -> Any:
        result = original(cls, batch, model_runner)
        if not result.forward_mode.is_decode():
            return result
        if result.orig_seq_lens is None:
            raise RuntimeError("native SGLang decode omitted original logical sequence lengths")
        result.positions = clamp_position(result.orig_seq_lens)
        if model_runner.model_is_mrope:
            active_lengths = result.seq_lens
            try:
                result.seq_lens = result.orig_seq_lens.to(
                    dtype=active_lengths.dtype, device=active_lengths.device
                )
                result._compute_mrope_positions(model_runner, batch)
            finally:
                result.seq_lens = active_lengths
        return result

    ForwardBatch.init_new = classmethod(init_new)
    ForwardBatch._mosaickv_native_patch_v1 = True


def _patch_request_release() -> None:
    from sglang.srt.mem_cache.chunk_cache import ChunkCache
    from sglang.srt.mem_cache.radix_cache import RadixCache

    if not getattr(RadixCache, "_mosaickv_native_patch_v1", False):
        original_radix = RadixCache.cache_finished_req

        def cache_finished_radix(self: Any, request: Any) -> Any:
            if not getattr(request, "_mosaickv_native_compacted", False):
                return original_radix(self, request)
            if not self.disable:
                raise RuntimeError(
                    "native SGLang compressed request reached an enabled Radix cache"
                )
            _release_active_request(self, request)
            return None

        RadixCache.cache_finished_req = cache_finished_radix
        RadixCache._mosaickv_native_patch_v1 = True

    if not getattr(ChunkCache, "_mosaickv_native_patch_v1", False):
        original_chunk = ChunkCache.cache_finished_req

        def cache_finished_chunk(self: Any, request: Any) -> Any:
            if not getattr(request, "_mosaickv_native_compacted", False):
                return original_chunk(self, request)
            _release_active_request(self, request)
            return None

        ChunkCache.cache_finished_req = cache_finished_chunk
        ChunkCache._mosaickv_native_patch_v1 = True


def _patch_retraction_gate() -> None:
    from sglang.srt.managers.schedule_batch import ScheduleBatch

    if getattr(ScheduleBatch, "_mosaickv_native_patch_v1", False):
        return
    original = ScheduleBatch.retract_decode

    def retract_decode(self: Any, *args: Any, **kwargs: Any) -> Any:
        if any(getattr(request, "_mosaickv_native_compacted", False) for request in self.reqs):
            raise RuntimeError("native SGLang does not support request retraction/recompute")
        return original(self, *args, **kwargs)

    ScheduleBatch.retract_decode = retract_decode
    ScheduleBatch._mosaickv_native_patch_v1 = True


def _patch_scheduler_entrypoint() -> None:
    from sglang.srt.entrypoints import engine

    engine.run_scheduler_process = run_scheduler_process_with_native_patch
    # The audited native runtime requires dp_size=1, but patch the imported
    # alias as well so an accidental unsupported launch cannot silently bypass
    # installation before the scheduler's own fail-closed validation runs.
    from sglang.srt.managers import data_parallel_controller

    data_parallel_controller.run_scheduler_process = run_scheduler_process_with_native_patch


def install() -> None:
    """Install the audited patch before SGLang constructs its scheduler."""

    if not _enabled():
        return
    installed = importlib.metadata.version("sglang")
    if installed != _VERSION:
        raise RuntimeError(f"native MosaicKV audited SGLang {_VERSION}; installed {installed}")
    _settings()
    _patch_scheduler_entrypoint()
    _patch_output_processor()
    _patch_forward_positions()
    _patch_request_release()
    _patch_retraction_gate()


__all__ = ["install", "run_scheduler_process_with_native_patch"]
