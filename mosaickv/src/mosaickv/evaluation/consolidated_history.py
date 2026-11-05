"""Coalesce exact source-history packets; no history eviction or approximation.

Each layer publishes its replacement only after three original-bit checks.
Other layers remain semantically valid if a later layer fails. Old/new body,
reference and constructor overlap is explicitly audited one layer at a time.
"""

from contextlib import nullcontext
from typing import Any

import torch

from mosaickv.evaluation.exact_storage import SparseBitPatch, decoded, encode
from mosaickv.evaluation.lossless_bf16 import PackedBF16
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.projection_cache import ScratchAudit, identity


def merge_sources(bodies: tuple[PackedBF16, ...]) -> PackedBF16:
    if len(bodies) < 2 or any(
        b.shape[:2] != (1, 1)
        or len(b.shape) != 4
        or b.shape[-1] != bodies[0].shape[-1]
        or b.shared_literal_window
        for b in bodies
    ):
        raise ValueError("two or more complete compatible independent source packets required")
    width = bodies[0].shape[-1]
    if any(b.shape[-2] * width % 1024 for b in bodies):
        raise ValueError("block-aligned source packets required")
    shape = (1, 1, sum(b.shape[-2] for b in bodies), width)
    if any(b.raw for b in bodies):
        return encode(torch.cat(tuple(decoded(b) for b in bodies), dim=2).contiguous())
    offsets = []
    literal_start = 0
    for b in bodies:
        offsets.append(b.buffers[3][:-1] + literal_start)
        literal_start += b.buffers[4].numel()
    last = bodies[-1]
    offsets.append(last.buffers[3][-1:] + literal_start - last.buffers[4].numel())
    buffers = tuple(torch.cat(tuple(b.buffers[i] for b in bodies)) for i in range(3))
    buffers += (torch.cat(tuple(offsets)), torch.cat(tuple(b.buffers[4] for b in bodies)))
    return PackedBF16(shape, False, buffers)


def join_patches(
    patches: tuple[SparseBitPatch, ...], lengths: tuple[int, ...], *, heads: int, dimension: int
) -> SparseBitPatch:
    if (
        len(patches) != len(lengths)
        or not patches
        or any(type(n) is not int or n < 1 for n in (*lengths, heads, dimension))
        or heads * sum(lengths) * dimension >= 2**31
    ):
        raise ValueError("matching patches and positive integer source geometry required")
    total, offset = sum(lengths), 0
    indices, values = [], []
    for patch, n in zip(patches, lengths, strict=True):
        if (
            patch.indices.dtype != torch.int32
            or patch.values.dtype != torch.bfloat16
            or patch.indices.ndim != 1
            or patch.values.ndim != 1
            or patch.indices.numel() != patch.values.numel()
        ):
            raise ValueError("original flat int32/BF16 sparse patches required")
        valid = ((patch.indices >= 0) & (patch.indices < heads * n * dimension)).all()
        if patch.indices.device.type == "cuda":
            torch._assert_async(valid, "invalid original patch bounds")
        elif not bool(valid):
            raise ValueError("invalid original patch bounds")
        old = patch.indices
        head, within = old // (n * dimension), old % (n * dimension)
        indices.append((head * total * dimension + offset * dimension + within).to(torch.int32))
        values.append(patch.values)
        offset += n
    return SparseBitPatch(torch.cat(tuple(indices)), torch.cat(tuple(values)))


def build_consolidated(
    layer: Any, cos: Any, sin: Any, *, bit_flags: list[Any] | None = None
) -> Any:
    from mosaickv.evaluation.native_history_reader import consolidate as native_consolidate
    from mosaickv.evaluation.native_history_reader import current as native_state

    if native_state() is not None:
        return native_consolidate(layer, cos, sin, bit_flags=bit_flags)
    from mosaickv.evaluation.encoded_projection import EncodedFormat
    from mosaickv.evaluation.encoded_projection_kernel import checkpoints
    from mosaickv.evaluation.projection_archive import ProjectionArchive

    formats = tuple(layer.source_sealed)
    if any(
        not isinstance(f, EncodedFormat) or f.archive is None or len(f.patches) != 2
        for f in formats
    ):
        raise ValueError("complete validated encoded-source history formats required")
    lengths = tuple(f.shape[-2] for f in formats)
    count, heads, dimension = sum(lengths), layer.prompt_shape[1], layer.prompt_shape[-1]
    body = merge_sources(tuple(f.archive.source for f in formats))
    union = torch.arange(count, device=cos.device, dtype=torch.int32)
    selected = union[None].expand(heads, -1)
    archive = ProjectionArchive(body, union, selected, count, cos, sin)
    shape = (1, heads, count, dimension)
    candidate = EncodedFormat(
        "encoded_projection",
        shape,
        archive=archive,
        output_slots=torch.arange(count, device=cos.device, dtype=torch.int16)[None].expand(
            heads, -1
        ),
        literal_checkpoints=checkpoints(body),
        patches=tuple(
            join_patches(
                tuple(f.patches[i] for f in formats), lengths, heads=heads, dimension=dimension
            )
            for i in range(2)
        ),
    )
    from mosaickv.evaluation.history_verification_workspace import verify

    pool = getattr(layer, "prompt_graph_pool", None)
    with pool.borrow_history(shape) if pool is not None else nullcontext(None) as scratch:
        verify(layer, candidate, formats, shape, bit_flags, scratch)
    return candidate


def consolidate(cache: Any) -> bool:
    layers = cache.layers
    if (
        not layers
        or any(layer.leased or layer.source_pending for layer in layers)
        or len({layer.history_length for layer in layers}) != 1
        or len({layer.tail_used for layer in layers}) != 1
    ):
        raise ValueError("synchronized released original source-history layers required")
    counts = [sum(f.shape[-2] for f in layer.source_sealed) for layer in layers]
    if len(set(counts)) != 1:
        raise ValueError("history source memberships differ between layers")
    if not any(len(layer.source_sealed) >= 2 for layer in layers):
        return False
    if (
        not cache.history_rotary_chunks
        or sum(t[0].shape[1] for t in cache.history_rotary_chunks) != counts[0]
    ):
        raise ValueError("complete original rotary history required")
    excluded = {identity(t) for layer in layers for t in layer.owned_tensors()}
    excluded.update(identity(t) for t in cache.shared_tensors)
    excluded.update(owner for layer in layers for owner in layer.weights)
    mode = ScratchAudit(excluded=excluded, observe=cache.observe) if cache.audit_workspace else None
    old_phase_owners = {identity(t) for p in cache.history_rotary_chunks for t in p}
    with mode if mode is not None else nullcontext():
        cos, sin = tuple(
            torch.cat(tuple(p[i] for p in cache.history_rotary_chunks), 1) for i in range(2)
        )
        cache.consolidated_phase_versions.append((cos, sin))
        cache.shared_tensors = (*cache.shared_tensors, cos, sin)
        cache.shared_bytes = backing_bytes(cache.shared_tensors)
        excluded.update(identity(t) for t in (cos, sin))
        cache.observe()
        for layer in layers:
            if len(layer.source_sealed) < 2:
                continue
            old = tuple(layer.source_sealed)
            old_private = tuple(t for f in old for t in f.private())
            old_bytes = backing_bytes(old_private)
            flags = [] if getattr(cache, "sequential_source_history", False) else None
            candidate = (
                build_consolidated(layer, cos, sin, bit_flags=flags)
                if flags is not None
                else build_consolidated(layer, cos, sin)
            )
            if flags is not None:
                if len(flags) != 6:
                    raise RuntimeError("three independent K and V checks per layer required")
                passed = torch.stack(flags).all()
                cache.observe((*flags, passed))
                if not bool(passed):
                    raise RuntimeError("consolidated original history KV bit check failed")
                cache.source_consolidation_check_syncs = (
                    getattr(cache, "source_consolidation_check_syncs", 0) + 1
                )
                cache.source_consolidation_checks = getattr(
                    cache, "source_consolidation_checks", 0
                ) + len(flags)
            if candidate.shape[-2] != counts[0]:
                raise RuntimeError("history consolidation changed source membership")
            protected = {identity(t) for t in (*layer.history, layer.normalized_tail)}
            if any(
                t.untyped_storage().nbytes() and identity(t) in protected
                for t in candidate.private()
            ):
                raise RuntimeError("consolidated packet aliases a mutable source tail")
            layer.source_sealed = [candidate]
            layer.resident_bytes += backing_bytes(candidate.private()) - old_bytes
            excluded.update(identity(t) for t in candidate.private())
            cache.observe(old_private)
            cache.source_consolidation_layers += 1
            del old, old_private, candidate
            if mode is not None:
                excluded.clear()
                excluded.update(identity(t) for item in layers for t in item.owned_tensors())
                excluded.update(identity(t) for t in cache.shared_tensors)
                excluded.update(owner for item in layers for owner in item.weights)
            cache.observe()
        used = {identity(t) for layer in layers for f in layer.source_sealed for t in f.shared()}
        used.update(identity(t) for t in (cos, sin))
        obsolete = {
            identity(t)
            for p in cache.consolidated_phase_versions
            for t in p
            if identity(t) not in used
        }
        obsolete.update(t for t in old_phase_owners if t not in used)
        cache.shared_tensors = tuple(t for t in cache.shared_tensors if identity(t) not in obsolete)
        cache.history_rotary_chunks = [(cos, sin)]
        cache.consolidated_phase_versions = [
            p for p in cache.consolidated_phase_versions if any(identity(t) in used for t in p)
        ]
        cache.shared_bytes = backing_bytes(cache.shared_tensors)
        cache.source_consolidation_epochs += 1
        cache.observe()
    if mode is not None:
        layers[0].workspace_operations += mode.operations
    return True


def gpu_controls() -> list[dict[str, Any]]:
    from mosaickv.evaluation.lossless_bf16_kernel import encode as gpu_encode

    data = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16)
    data = data.view(torch.bfloat16).reshape(1, 1, 256, 256)
    halves = (data[..., :128, :].contiguous(), data[..., 128:, :].contiguous())
    records = []
    for flags in [(False, False), (True, False), (True, True)]:
        bodies = tuple(
            PackedBF16(tuple(t.shape), True, (t.clone(),))
            if raw
            else gpu_encode(t, raw_fallback=False)
            for t, raw in zip(halves, flags, strict=True)
        )
        merged = merge_sources(bodies)
        if not torch.equal(decoded(merged).view(torch.int16), data.view(torch.int16)):
            raise RuntimeError("source consolidation codec changed a BF16 bit pattern")
        old = {identity(t) for b in bodies for t in b.buffers if t.numel()}
        if any(identity(t) in old for t in merged.buffers if t.numel()):
            raise RuntimeError("merged source body aliases a previous packet")
        records.append(
            dict(
                raw_flags=flags,
                all_65536_bit_patterns=True,
                independent_new_owner=True,
                storage_bytes=merged.storage_bytes,
            )
        )
    return records
