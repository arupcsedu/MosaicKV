"""One lossless encode/synchronization for all full layer tails, no eviction."""

from contextlib import nullcontext
from typing import Any

import torch

from mosaickv.evaluation.exact_storage import encode
from mosaickv.evaluation.lossless_bf16 import BLOCK, PackedBF16
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.projection_cache import ScratchAudit, identity
from mosaickv.evaluation.sealed_history import SealedHistoryLayer


def split_batch(body: PackedBF16, shape: tuple[int, ...], count: int) -> tuple[PackedBF16, ...]:
    """Block-aligned codec windows share backing and global literal offsets."""
    length = 1
    for dimension in shape:
        length *= dimension
    if count < 1 or length % BLOCK or body.buffers[0].numel() != count * length:
        raise ValueError("complete block-aligned equal-sized batch required")
    result = []
    blocks = length // BLOCK
    for i in range(count):
        buffers: tuple[Any, ...]
        if body.raw:
            buffers = (body.buffers[0].reshape(-1)[i * length : (i + 1) * length].view(shape),)
        else:
            sm, codes, bases, offsets, literals = body.buffers
            buffers = (
                sm[i * length : (i + 1) * length],
                codes[i * blocks * 384 : (i + 1) * blocks * 384],
                bases[i * blocks : (i + 1) * blocks],
                offsets[i * blocks : (i + 1) * blocks + 1],
                literals,
            )
        result.append(PackedBF16(shape, body.raw, buffers, shared_literal_window=not body.raw))
    return tuple(result)


class BatchedHistoryLayer(SealedHistoryLayer):
    def update(self, key_states: Any, value_states: Any, cache_kwargs: Any = None) -> Any:
        if self.history_capacity and self.tail_used == self.history_capacity:
            raise RuntimeError("batch coordinator must seal every full tail before the next token")
        return super().update(key_states, value_states, cache_kwargs)


def seal_full_tails(cache: Any, *, maximum_layers_per_batch: int = 4) -> bool:
    """Atomic encode before next decode; unequal progress/geometry fails closed."""
    layers = cache.layers
    if type(maximum_layers_per_batch) is not int or not 1 <= maximum_layers_per_batch <= 4:
        raise ValueError("one through four layers per bounded staging batch required")
    if not layers or not all(isinstance(layer, BatchedHistoryLayer) for layer in layers):
        raise ValueError("complete batched-history cache required")
    if len({layer.history_length for layer in layers}) != 1 or any(
        layer.leased for layer in layers
    ):
        raise ValueError("synchronized released layer states required")
    full = [
        bool(layer.history_capacity and layer.tail_used == layer.history_capacity)
        for layer in layers
    ]
    if not any(full):
        return False
    if not all(full) or len({tuple(layer.history[0].shape) for layer in layers}) != 1:
        raise ValueError("all tails must have matching full geometry")
    tails = tuple(t for layer in layers for t in layer.history)
    if any(
        t.shape != tails[0].shape or t.dtype != torch.bfloat16 or t.device != tails[0].device
        for t in tails
    ):
        raise ValueError("matching BF16 history tails required")
    excluded = {identity(t) for layer in layers for t in layer.owned_tensors()}
    excluded.update(t for layer in layers for t in layer.weights)
    pending: list[Any] = []
    parts: list[PackedBF16] = []

    def observe(extras: Any = ()) -> None:
        cache.observe((*pending, *extras))

    # Stage <= four layers (two MiB for this checkpoint), not all layer tails.
    # Prepare every immutable body before committing any layer's tail counter.
    batch_count = 0
    for start in range(0, len(tails), 2 * maximum_layers_per_batch):
        group = tails[start : start + 2 * maximum_layers_per_batch]
        mode = ScratchAudit(excluded=excluded, observe=observe) if cache.audit_workspace else None
        with mode if mode is not None else nullcontext():
            batch = torch.stack(group)
            body = encode(batch)
            windows = split_batch(body, tuple(tails[0].shape), len(group))
        if mode is not None:
            layers[0].workspace_operations += mode.operations
        pending.extend(body.buffers)
        parts.extend(windows)
        batch_count += 1
        observe((batch,))
        del batch, body, windows
        observe(())
    tail_owners = {identity(t) for t in tails}
    if any(identity(t) in tail_owners for part in parts for t in part.buffers):
        raise RuntimeError("sealed batch aliases mutable history tails")
    # One shared owner, never count its complete backing once per layer view.
    cache.shared_tensors = (*cache.shared_tensors, *pending)
    cache.shared_bytes = backing_bytes(cache.shared_tensors)
    for i, layer in enumerate(layers):
        layer.sealed.append((parts[2 * i], parts[2 * i + 1]))
        layer.tail_used = 0
    cache.batched_history_seals = getattr(cache, "batched_history_seals", 0) + 1
    cache.batched_history_encode_batches = (
        getattr(cache, "batched_history_encode_batches", 0) + batch_count
    )
    pending.clear()
    parts.clear()
    cache.observe(())
    return True


def gpu_controls() -> list[dict[str, Any]]:
    from mosaickv.evaluation.lossless_bf16 import unpack_into
    from mosaickv.evaluation.lossless_bf16_kernel import encode as encode_gpu

    words = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16)
    tensor = words.view(torch.bfloat16).view(2, 1, 2, 128, 128)
    records = []
    for raw in (True, False):
        body = (
            PackedBF16(tuple(tensor.shape), True, (tensor.clone(),))
            if raw
            else encode_gpu(tensor, raw_fallback=False)
        )
        parts = split_batch(body, (1, 2, 128, 128), 2)
        for i, part in enumerate(parts):
            backing = torch.full((1, 2, 140, 128), -71, device="cuda", dtype=torch.bfloat16)
            target = backing[..., 6:134, :]
            unpack_into(part, target)
            if not torch.equal(target.view(torch.int16), tensor[i].view(torch.int16)):
                raise RuntimeError("shared history literals changed a BF16 pattern")
            if not bool(
                (backing[..., :6, :] == -71).all() and (backing[..., 134:, :] == -71).all()
            ):
                raise RuntimeError("shared history decode overwrote a strided sentinel")
        if backing_bytes(tuple(t for p in parts for t in p.buffers)) != backing_bytes(body.buffers):
            raise RuntimeError("shared history backing was duplicated or undercounted")
        records.append(
            dict(
                raw=raw,
                all_65536_bits=True,
                strided_sentinels=True,
                shared_backing_counted_once=True,
            )
        )
    return records
