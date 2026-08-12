"""CPU residual-tier encoding, pinning, indexing, and restoration."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from mosaickv.cache_state import ExactTier, FullKVState, ResidualTier, tensor_storage_bytes
from mosaickv.config import ResidualConfig
from mosaickv.residual.types import (
    ResidualIndexEntry,
    ResidualPayloadMetadata,
    ResidualPositionIndex,
    ResidualStorageReport,
    ResidualTransferBatch,
)
from mosaickv.types import ResidualStorageDType


class ResidualStorageError(RuntimeError):
    """Raised when a requested residual representation cannot be constructed."""


class PinnedMemoryUnavailableError(ResidualStorageError):
    """Raised when the configured host cannot allocate required pinned memory."""


def _is_torch_tensor(tensor: Any) -> bool:
    return tensor.__class__.__module__.startswith("torch") and hasattr(tensor, "detach")


def _source_dtype_name(tensor: Any) -> str:
    return str(getattr(tensor, "dtype", "unknown"))


def _encode_torch(tensor: Any, storage_dtype: ResidualStorageDType) -> tuple[Any, float | None]:
    import torch

    source = tensor.detach()
    scale: float | None = None
    if storage_dtype is ResidualStorageDType.LOSSLESS:
        encoded = source
    elif storage_dtype is ResidualStorageDType.FP16:
        encoded = source.to(dtype=torch.float16)
    elif storage_dtype is ResidualStorageDType.BF16:
        encoded = source.to(dtype=torch.bfloat16)
    elif storage_dtype is ResidualStorageDType.FP8:
        dtype = getattr(torch, "float8_e4m3fn", None)
        if dtype is None:
            raise ResidualStorageError("this PyTorch build does not expose float8_e4m3fn")
        encoded = source.to(dtype=dtype)
    elif storage_dtype is ResidualStorageDType.INT8:
        maximum = float(source.detach().float().abs().max().item())
        scale = maximum / 127.0 if maximum > 0 else 1.0
        encoded = torch.round(source.detach().float() / scale).clamp(-127, 127).to(torch.int8)
    else:  # pragma: no cover - guarded by the enum
        raise ResidualStorageError(f"unsupported residual storage dtype: {storage_dtype}")
    return encoded.to(device="cpu").contiguous(), scale


def _encode_numpy(
    tensor: Any, storage_dtype: ResidualStorageDType
) -> tuple[np.ndarray[Any, Any], float | None]:
    source = np.asarray(tensor)
    scale: float | None = None
    if storage_dtype is ResidualStorageDType.LOSSLESS:
        encoded = source.copy()
    elif storage_dtype is ResidualStorageDType.FP16:
        encoded = source.astype(np.float16, copy=True)
    elif storage_dtype in {ResidualStorageDType.BF16, ResidualStorageDType.FP8}:
        raise ResidualStorageError(
            f"{storage_dtype.value} residual storage requires a compatible PyTorch build"
        )
    elif storage_dtype is ResidualStorageDType.INT8:
        maximum = float(np.max(np.abs(source.astype(np.float32, copy=False))))
        scale = maximum / 127.0 if maximum > 0 else 1.0
        encoded = np.clip(np.rint(source.astype(np.float32) / scale), -127, 127).astype(np.int8)
    else:  # pragma: no cover - guarded by the enum
        raise ResidualStorageError(f"unsupported residual storage dtype: {storage_dtype}")
    return np.ascontiguousarray(encoded), scale


def _pin_torch(tensor: Any, *, required: bool) -> tuple[Any, bool]:
    import torch

    try:
        pinned = torch.empty_like(tensor, device="cpu", pin_memory=True)
        pinned.copy_(tensor)
    except RuntimeError as error:
        if required:
            raise PinnedMemoryUnavailableError(
                "pinned residual memory was required but the PyTorch/CUDA runtime "
                "could not allocate it"
            ) from error
        return tensor.clone(), bool(tensor.is_pinned())
    if not bool(pinned.is_pinned()):
        if required:
            raise PinnedMemoryUnavailableError(
                "PyTorch returned non-pinned memory for a required residual allocation"
            )
        return pinned, False
    return pinned, True


def _encode_and_place(tensor: Any, config: ResidualConfig) -> tuple[Any, float | None, bool]:
    if _is_torch_tensor(tensor):
        encoded, scale = _encode_torch(tensor, config.storage_dtype)
        if not config.require_pinned_memory:
            # Experimentally requested pageable placement must be real rather
            # than a best-effort pin.  Otherwise a pinned/pageable placement
            # sweep can silently execute the same storage path in both rows.
            return encoded, scale, False
        placed, pinned = _pin_torch(encoded, required=config.require_pinned_memory)
        return placed, scale, pinned
    if config.require_pinned_memory:
        raise PinnedMemoryUnavailableError(
            "pinned residual memory requires torch tensors; disable the requirement only "
            "for CPU/mock validation"
        )
    encoded, scale = _encode_numpy(tensor, config.storage_dtype)
    return encoded, scale, False


def _batch_lossless_torch_payloads(
    tensors: tuple[Any, ...],
    *,
    require_pinned: bool,
) -> tuple[tuple[Any, ...], bool]:
    """Copy lossless torch payloads through one contiguous host allocation.

    Thousands of per-block ``pin_memory`` allocations dominate construction
    for long documents and videos.  One flat slab preserves every payload
    bit-for-bit; the returned shaped views remain distinct tensor objects and
    therefore retain the existing per-block residual interface and accounting.
    """

    if not tensors:
        return (), False
    import torch

    source_dtype = tensors[0].dtype
    if any(tensor.dtype != source_dtype for tensor in tensors):
        raise ResidualStorageError("one batched residual slab cannot mix torch dtypes")
    shapes = tuple(tuple(int(size) for size in tensor.shape) for tensor in tensors)
    counts = tuple(int(tensor.numel()) for tensor in tensors)
    flattened = torch.cat(tuple(tensor.detach().reshape(-1) for tensor in tensors))
    if not require_pinned:
        slab = flattened.to(device="cpu").contiguous()
        pageable_payloads: list[Any] = []
        offset = 0
        for shape, count in zip(shapes, counts, strict=True):
            pageable_payloads.append(slab.narrow(0, offset, count).view(shape))
            offset += count
        if offset != int(slab.numel()):  # pragma: no cover - arithmetic invariant
            raise RuntimeError("batched residual slab offsets do not cover its storage")
        return tuple(pageable_payloads), False
    try:
        slab = torch.empty(
            int(flattened.numel()),
            dtype=source_dtype,
            device="cpu",
            pin_memory=True,
        )
        slab.copy_(flattened)
        pinned = bool(slab.is_pinned())
        if not pinned:
            raise PinnedMemoryUnavailableError(
                "PyTorch returned non-pinned memory for a required residual slab"
            )
    except RuntimeError as error:
        raise PinnedMemoryUnavailableError(
            "pinned residual memory was required but the PyTorch/CUDA runtime "
            "could not allocate a batched slab"
        ) from error
    payloads: list[Any] = []
    offset = 0
    for shape, count in zip(shapes, counts, strict=True):
        payloads.append(slab.narrow(0, offset, count).view(shape))
        offset += count
    if offset != int(slab.numel()):  # pragma: no cover - arithmetic invariant
        raise RuntimeError("batched residual slab offsets do not cover its storage")
    return tuple(payloads), pinned


def empty_residual_storage() -> ResidualStorageReport:
    """Return an empty residual report for exact-only and residual-disabled runs."""

    return ResidualStorageReport(ResidualTier(), (), (), 0)


def _encoded_payload_bytes(
    key: Any,
    value: Any,
    storage_dtype: ResidualStorageDType,
) -> int:
    """Return the exact tensor-payload bytes before allocating host storage."""

    if storage_dtype is ResidualStorageDType.LOSSLESS:
        return tensor_storage_bytes(key) + tensor_storage_bytes(value)
    bytes_per_element = (
        2
        if storage_dtype in {ResidualStorageDType.FP16, ResidualStorageDType.BF16}
        else 1
    )

    def numel(tensor: Any) -> int:
        method = getattr(tensor, "numel", None)
        if method is not None:
            return int(method())
        return int(np.asarray(tensor).size)

    return (numel(key) + numel(value)) * bytes_per_element


def _bounded_residual_node_ids(
    node_ids: Sequence[int],
    payloads: Mapping[int, tuple[Any, Any]],
    config: ResidualConfig,
    priority_by_node: Mapping[int, float] | None,
) -> tuple[int, ...]:
    """Choose a deterministic repairable subset without allocating it first."""

    if config.max_bytes is None and config.max_source_blocks is None:
        return tuple(node_ids)
    priorities = {} if priority_by_node is None else {
        int(node_id): float(priority) for node_id, priority in priority_by_node.items()
    }
    if any(not np.isfinite(priority) or priority < 0 for priority in priorities.values()):
        raise ValueError("residual priorities must be finite and nonnegative")
    unknown = set(priorities) - set(node_ids)
    if unknown:
        raise ValueError(f"residual priorities reference unknown source nodes: {sorted(unknown)}")
    ranked = (
        tuple(node_ids)
        if config.selection_policy == "source_order"
        else tuple(sorted(node_ids, key=lambda node_id: (-priorities.get(node_id, 0.0), node_id)))
    )
    maximum_blocks = len(ranked) if config.max_source_blocks is None else config.max_source_blocks
    maximum_bytes = float("inf") if config.max_bytes is None else config.max_bytes
    selected: set[int] = set()
    consumed = 0
    for node_id in ranked:
        if len(selected) >= maximum_blocks:
            break
        key, value = payloads[node_id]
        cost = _encoded_payload_bytes(key, value, config.storage_dtype)
        if consumed + cost > maximum_bytes:
            continue
        selected.add(node_id)
        consumed += cost
    # Physical/index order remains the declared prototype-member order even
    # when risk scores determined which members survived the storage cap.
    return tuple(node_id for node_id in node_ids if node_id in selected)


def build_residual_storage(
    full_state: FullKVState,
    assignments: Mapping[int, int],
    config: ResidualConfig,
    *,
    node_order: Sequence[int] | None = None,
    gathered: ExactTier | None = None,
    priority_by_node: Mapping[int, float] | None = None,
) -> ResidualStorageReport:
    """Preserve assigned source blocks on CPU, indexed by original position.

    ``assignments`` maps source graph node IDs to prototype IDs.  Graph node IDs
    are required to align with ``full_state.blocks``; this is checked here rather
    than inferred from tensor order.
    """

    if not config.enabled or not assignments:
        return empty_residual_storage()
    node_ids = (
        tuple(sorted(assignments))
        if node_order is None
        else tuple(int(node_id) for node_id in node_order)
    )
    if len(node_ids) != len(set(node_ids)) or set(node_ids) != set(assignments):
        raise ValueError("residual node_order must contain every assigned source node exactly once")
    if any(node_id < 0 or node_id >= len(full_state.blocks) for node_id in node_ids):
        raise ValueError("residual assignment references a non-source graph node")
    blocks = tuple(full_state.blocks[node_id] for node_id in node_ids)
    if gathered is None:
        gathered = full_state.gather_exact_blocks(blocks)
    elif gathered.blocks != blocks:
        raise ValueError(
            "pre-gathered residual payloads must align exactly with residual node_order"
        )
    payload_by_node = {
        node_id: (key, value)
        for node_id, key, value in zip(
            node_ids,
            gathered.key_blocks,
            gathered.value_blocks,
            strict=True,
        )
    }
    node_ids = _bounded_residual_node_ids(
        node_ids,
        payload_by_node,
        config,
        priority_by_node,
    )
    if not node_ids:
        return empty_residual_storage()
    blocks = tuple(full_state.blocks[node_id] for node_id in node_ids)
    gathered = ExactTier(
        blocks,
        tuple(payload_by_node[node_id][0] for node_id in node_ids),
        tuple(payload_by_node[node_id][1] for node_id in node_ids),
    )
    batched_lossless = bool(
        config.storage_dtype is ResidualStorageDType.LOSSLESS
        and gathered.key_blocks
        and all(_is_torch_tensor(tensor) for tensor in gathered.key_blocks)
        and all(_is_torch_tensor(tensor) for tensor in gathered.value_blocks)
    )
    batched_keys: tuple[Any, ...] = ()
    batched_values: tuple[Any, ...] = ()
    batch_key_pinned = False
    batch_value_pinned = False
    if batched_lossless:
        batched_keys, batch_key_pinned = _batch_lossless_torch_payloads(
            gathered.key_blocks,
            require_pinned=config.require_pinned_memory,
        )
        batched_values, batch_value_pinned = _batch_lossless_torch_payloads(
            gathered.value_blocks,
            require_pinned=config.require_pinned_memory,
        )
    key_payloads: list[Any] = []
    value_payloads: list[Any] = []
    metadata: list[ResidualPayloadMetadata] = []
    for payload_index, (node_id, _block, key, value) in enumerate(
        zip(node_ids, blocks, gathered.key_blocks, gathered.value_blocks, strict=True)
    ):
        if batched_lossless:
            key_payload = batched_keys[payload_index]
            value_payload = batched_values[payload_index]
            key_scale = value_scale = None
            key_pinned = batch_key_pinned
            value_pinned = batch_value_pinned
        else:
            key_payload, key_scale, key_pinned = _encode_and_place(key, config)
            value_payload, value_scale, value_pinned = _encode_and_place(value, config)
        key_payloads.append(key_payload)
        value_payloads.append(value_payload)
        prototype_id = assignments[node_id]
        metadata.append(
            ResidualPayloadMetadata(
                payload_index=payload_index,
                source_node_id=node_id,
                prototype_id=prototype_id,
                storage_dtype=config.storage_dtype,
                source_key_dtype=_source_dtype_name(key),
                source_value_dtype=_source_dtype_name(value),
                key_scale=key_scale,
                value_scale=value_scale,
                key_pinned=key_pinned,
                value_pinned=value_pinned,
            )
        )
    tier = ResidualTier(blocks, tuple(key_payloads), tuple(value_payloads))
    payload_metadata = tuple(metadata)
    if config.index_mode == "compact":
        index: Sequence[ResidualIndexEntry] = ResidualPositionIndex.from_tier_metadata(
            tier,
            payload_metadata,
        )
    else:
        materialized = [
            ResidualIndexEntry(
                layer=block.layer,
                kv_head=block.kv_head,
                prototype_id=payload.prototype_id,
                original_position=logical,
                physical_position=physical,
                payload_index=payload.payload_index,
                block_offset=offset,
            )
            for block, payload in zip(blocks, payload_metadata, strict=True)
            for offset, (physical, logical) in enumerate(
                zip(
                    block.physical_cache_indices,
                    block.original_logical_positions,
                    strict=True,
                )
            )
        ]
        materialized.sort(key=lambda entry: entry.identity)
        index = tuple(materialized)
    report = ResidualStorageReport(tier, payload_metadata, index, tier.active_bytes)
    if config.max_bytes is not None and report.cpu_bytes > config.max_bytes:
        raise RuntimeError("encoded residual payloads exceeded residual.max_bytes")
    if (
        config.max_source_blocks is not None
        and len(report.payloads) > config.max_source_blocks
    ):
        raise RuntimeError("encoded residual payloads exceeded residual.max_source_blocks")
    return report


def _decode_payload(
    payload: Any,
    *,
    storage_dtype: ResidualStorageDType,
    scale: float | None,
    reference: Any,
) -> Any:
    if _is_torch_tensor(payload):
        target_dtype = reference.dtype
        target_device = reference.device
        value = payload
        if storage_dtype is ResidualStorageDType.INT8:
            if scale is None:  # pragma: no cover - schema guards this
                raise ResidualStorageError("INT8 residual payload is missing its scale")
            value = value.float() * scale
        return value.to(device=target_device, dtype=target_dtype)
    value = np.asarray(payload)
    if storage_dtype is ResidualStorageDType.INT8:
        if scale is None:  # pragma: no cover - schema guards this
            raise ResidualStorageError("INT8 residual payload is missing its scale")
        value = value.astype(np.float32) * scale
    return value.astype(np.asarray(reference).dtype, copy=False)


def restore_residual_payload(
    report: ResidualStorageReport,
    payload_index: int,
    full_state: FullKVState,
) -> tuple[Any, Any]:
    """Decode one residual block back to the source cache dtype and device."""

    if payload_index < 0 or payload_index >= len(report.payloads):
        raise IndexError(f"residual payload does not exist: {payload_index}")
    metadata = report.payloads[payload_index]
    block = report.tier.source_blocks[payload_index]
    source = full_state.gather_exact_blocks((block,))
    return (
        _decode_payload(
            report.tier.key_residuals[payload_index],
            storage_dtype=metadata.storage_dtype,
            scale=metadata.key_scale,
            reference=source.key_blocks[0],
        ),
        _decode_payload(
            report.tier.value_residuals[payload_index],
            storage_dtype=metadata.storage_dtype,
            scale=metadata.value_scale,
            reference=source.value_blocks[0],
        ),
    )


def _decode_torch_to_reference(
    payload: Any,
    metadata: ResidualPayloadMetadata,
    *,
    scale: float | None,
    reference: Any,
    non_blocking: bool,
) -> Any:
    value = payload.to(device=reference.device, non_blocking=non_blocking)
    if metadata.storage_dtype is ResidualStorageDType.INT8:
        if scale is None:  # pragma: no cover - schema guards this
            raise ResidualStorageError("INT8 residual payload is missing its scale")
        value = value.float() * scale
    return value.to(dtype=reference.dtype)


def restore_residual_payloads_async(
    report: ResidualStorageReport,
    payload_indices: tuple[int, ...],
    references: tuple[tuple[Any, Any], ...],
) -> ResidualTransferBatch:
    """Restore a batch, using a CUDA stream and nonblocking pinned copies when possible."""

    indices = tuple(int(index) for index in payload_indices)
    if indices != tuple(sorted(set(indices))):
        raise ValueError("residual transfer payload indices must be sorted and unique")
    if len(indices) != len(references):
        raise ValueError("residual transfer references must align with payload indices")
    if any(index < 0 or index >= len(report.payloads) for index in indices):
        raise IndexError("residual transfer references an unknown payload")
    if not indices:
        return ResidualTransferBatch((), (), (), 0.0, False)

    torch_batch = all(
        _is_torch_tensor(report.tier.key_residuals[index])
        and _is_torch_tensor(report.tier.value_residuals[index])
        and _is_torch_tensor(key_reference)
        and _is_torch_tensor(value_reference)
        for index, (key_reference, value_reference) in zip(indices, references, strict=True)
    )
    cuda_devices = {
        str(reference.device)
        for pair in references
        for reference in pair
        if _is_torch_tensor(reference) and reference.device.type == "cuda"
    }
    if torch_batch and cuda_devices:
        if len(cuda_devices) != 1:
            raise ResidualStorageError("one residual transfer batch cannot span CUDA devices")
        import torch

        device = torch.device(next(iter(cuda_devices)))
        asynchronous = all(
            report.payloads[index].key_pinned
            and report.payloads[index].value_pinned
            and bool(report.tier.key_residuals[index].is_pinned())
            and bool(report.tier.value_residuals[index].is_pinned())
            for index in indices
        )
        torch.cuda.synchronize(device)
        stream = torch.cuda.Stream(device=device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        key_blocks: list[Any] = []
        value_blocks: list[Any] = []
        with torch.cuda.stream(stream):
            start.record(stream)
            for index, (key_reference, value_reference) in zip(indices, references, strict=True):
                metadata = report.payloads[index]
                key_blocks.append(
                    _decode_torch_to_reference(
                        report.tier.key_residuals[index],
                        metadata,
                        scale=metadata.key_scale,
                        reference=key_reference,
                        non_blocking=asynchronous,
                    )
                )
                value_blocks.append(
                    _decode_torch_to_reference(
                        report.tier.value_residuals[index],
                        metadata,
                        scale=metadata.value_scale,
                        reference=value_reference,
                        non_blocking=asynchronous,
                    )
                )
            end.record(stream)
        torch.cuda.current_stream(device).wait_event(end)
        end.synchronize()
        return ResidualTransferBatch(
            indices,
            tuple(key_blocks),
            tuple(value_blocks),
            float(start.elapsed_time(end)),
            asynchronous,
        )

    started = time.perf_counter()
    keys: list[Any] = []
    values: list[Any] = []
    for index, (key_reference, value_reference) in zip(indices, references, strict=True):
        metadata = report.payloads[index]
        keys.append(
            _decode_payload(
                report.tier.key_residuals[index],
                storage_dtype=metadata.storage_dtype,
                scale=metadata.key_scale,
                reference=key_reference,
            )
        )
        values.append(
            _decode_payload(
                report.tier.value_residuals[index],
                storage_dtype=metadata.storage_dtype,
                scale=metadata.value_scale,
                reference=value_reference,
            )
        )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return ResidualTransferBatch(indices, tuple(keys), tuple(values), elapsed_ms, False)


def discard_residual_payloads(
    report: ResidualStorageReport,
    payload_indices: tuple[int, ...],
) -> ResidualStorageReport:
    """Remove promoted residual payloads and rebuild contiguous payload/index IDs."""

    removed = frozenset(int(index) for index in payload_indices)
    if any(index < 0 or index >= len(report.payloads) for index in removed):
        raise IndexError("cannot discard an unknown residual payload")
    kept = tuple(index for index in range(len(report.payloads)) if index not in removed)
    blocks = tuple(report.tier.source_blocks[index] for index in kept)
    keys = tuple(report.tier.key_residuals[index] for index in kept)
    values = tuple(report.tier.value_residuals[index] for index in kept)
    payloads = tuple(
        replace(report.payloads[old], payload_index=new) for new, old in enumerate(kept)
    )
    tier = ResidualTier(blocks, keys, values)
    index = ResidualPositionIndex.from_tier_metadata(tier, payloads)
    return ResidualStorageReport(tier, payloads, index, tier.active_bytes)


__all__ = [
    "PinnedMemoryUnavailableError",
    "ResidualStorageError",
    "build_residual_storage",
    "discard_residual_payloads",
    "empty_residual_storage",
    "restore_residual_payload",
    "restore_residual_payloads_async",
]
