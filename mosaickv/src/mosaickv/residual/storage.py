"""CPU residual-tier encoding, pinning, indexing, and restoration."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np

from mosaickv.cache_state import FullKVState, ResidualTier
from mosaickv.config import ResidualConfig
from mosaickv.residual.types import (
    ResidualIndexEntry,
    ResidualPayloadMetadata,
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
        placed, pinned = _pin_torch(encoded, required=config.require_pinned_memory)
        return placed, scale, pinned
    if config.require_pinned_memory:
        raise PinnedMemoryUnavailableError(
            "pinned residual memory requires torch tensors; disable the requirement only "
            "for CPU/mock validation"
        )
    encoded, scale = _encode_numpy(tensor, config.storage_dtype)
    return encoded, scale, False


def empty_residual_storage() -> ResidualStorageReport:
    """Return an empty residual report for exact-only and residual-disabled runs."""

    return ResidualStorageReport(ResidualTier(), (), (), 0)


def build_residual_storage(
    full_state: FullKVState,
    assignments: Mapping[int, int],
    config: ResidualConfig,
) -> ResidualStorageReport:
    """Preserve assigned source blocks on CPU, indexed by original position.

    ``assignments`` maps source graph node IDs to prototype IDs.  Graph node IDs
    are required to align with ``full_state.blocks``; this is checked here rather
    than inferred from tensor order.
    """

    if not config.enabled or not assignments:
        return empty_residual_storage()
    node_ids = tuple(sorted(assignments))
    if any(node_id < 0 or node_id >= len(full_state.blocks) for node_id in node_ids):
        raise ValueError("residual assignment references a non-source graph node")
    blocks = tuple(full_state.blocks[node_id] for node_id in node_ids)
    gathered = full_state.gather_exact_blocks(blocks)
    key_payloads: list[Any] = []
    value_payloads: list[Any] = []
    metadata: list[ResidualPayloadMetadata] = []
    index: list[ResidualIndexEntry] = []
    for payload_index, (node_id, block, key, value) in enumerate(
        zip(node_ids, blocks, gathered.key_blocks, gathered.value_blocks, strict=True)
    ):
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
        for offset, (physical, logical) in enumerate(
            zip(
                block.physical_cache_indices,
                block.original_logical_positions,
                strict=True,
            )
        ):
            index.append(
                ResidualIndexEntry(
                    layer=block.layer,
                    kv_head=block.kv_head,
                    prototype_id=prototype_id,
                    original_position=logical,
                    physical_position=physical,
                    payload_index=payload_index,
                    block_offset=offset,
                )
            )
    tier = ResidualTier(blocks, tuple(key_payloads), tuple(value_payloads))
    index.sort(key=lambda item: item.identity)
    return ResidualStorageReport(tier, tuple(metadata), tuple(index), tier.active_bytes)


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
    old_to_new = {old: new for new, old in enumerate(kept)}
    blocks = tuple(report.tier.source_blocks[index] for index in kept)
    keys = tuple(report.tier.key_residuals[index] for index in kept)
    values = tuple(report.tier.value_residuals[index] for index in kept)
    payloads = tuple(
        replace(report.payloads[old], payload_index=new) for new, old in enumerate(kept)
    )
    index = tuple(
        sorted(
            (
                replace(entry, payload_index=old_to_new[entry.payload_index])
                for entry in report.index
                if entry.payload_index in old_to_new
            ),
            key=lambda entry: entry.identity,
        )
    )
    tier = ResidualTier(blocks, keys, values)
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
