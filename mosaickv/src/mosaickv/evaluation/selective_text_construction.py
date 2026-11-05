"""Exact text-row construction; default-off, no selection or K/V changes."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace

import torch

_state = ContextVar("evicastkv_selective_text_constructor", default=None)


@contextmanager
def use(active):
    if type(active) is not bool or _state.get() is not None:
        raise ValueError("explicit isolated nonnested selective-text constructor required")
    state = (
        dict(
            source_layers=0,
            projected_rows=0,
            original_full_rows=0,
            cpu_metadata_hits=0,
            cpu_metadata_misses=0,
            windows=[],
        )
        if active
        else None
    )
    token = _state.set(state)
    try:
        yield state
    finally:
        _state.reset(token)


def enabled():
    return _state.get() is not None


def aligned_windows(positions, length, tile):
    if (
        type(length) is not int
        or length < 1
        or type(tile) is not int
        or tile not in (16, 32, 64)
        or not positions
        or any(type(p) is not int or not 0 <= p < length for p in positions)
    ):
        raise ValueError("nonempty bounded original text positions and unchanged row tile required")
    blocks = sorted({p // tile for p in positions})
    intervals = []
    for block in blocks:
        start, end = block * tile, min(length, (block + 1) * tile)
        if intervals and intervals[-1][1] == start:
            intervals[-1] = (intervals[-1][0], end)
        else:
            intervals.append((start, end))
    return tuple((start, end - start) for start, end in intervals)


def source_rows(packet):
    cached = packet.row_positions_cpu
    if cached is None:
        if enabled():
            _state.get()["cpu_metadata_misses"] += 1
        cached = packet.row_positions.detach().cpu()
    else:
        if (
            cached.device.type != "cpu"
            or cached.dtype != torch.int32
            or cached.ndim != 1
            or not cached.is_contiguous()
            or cached.numel() != packet.row_positions.numel()
        ):
            raise ValueError("constructor-owned exact CPU index metadata required")
        if enabled():
            _state.get()["cpu_metadata_hits"] += 1
    ids = cached.tolist()
    if (
        not ids
        or ids != sorted(set(ids))
        or ids[0] < 0
        or ids[-1] >= packet.shape[1] * packet.shape[2]
    ):
        raise ValueError("complete sorted unique protected source rows required")
    if packet.row_positions.device.type == "cpu" and not torch.equal(cached, packet.row_positions):
        raise ValueError("CPU row metadata differs from protected source rows")
    return ids


@torch.inference_mode()
def strip_exact(packet, text):
    from mosaickv.evaluation.exact_storage import decoded, encode
    from mosaickv.evaluation.shared_source_format import SharedSourceFormat

    if not isinstance(packet, SharedSourceFormat) or packet.protected_patches_disjoint is not True:
        raise ValueError("constructor-certified disjoint shared source required")
    ids = source_rows(packet)
    selected = [(i, p) for i, p in enumerate(ids) if p % packet.shape[2] not in text]
    if not selected:
        selected = [(0, ids[0])]  # Original nonempty scatter contract.
    cpu = torch.tensor([p for _, p in selected], dtype=torch.int32)
    if len(selected) == len(ids):
        exact = packet.exact
    else:
        ordinal = torch.tensor(
            [i for i, _ in selected], device=packet.row_positions.device, dtype=torch.long
        )
        exact = tuple(
            encode(decoded(body).index_select(2, ordinal).contiguous()) for body in packet.exact
        )
    return replace(
        packet,
        row_positions=cpu.to(packet.row_positions.device),
        exact=exact,
        row_positions_cpu=cpu,
    )


@torch.inference_mode()
def text_pair(packet, attention, positions):
    from mosaickv.evaluation.shared_source_format import SharedSourceFormat

    if (
        not enabled()
        or not isinstance(packet, SharedSourceFormat)
        or packet.projection_backend != "fused"
        or packet.protected_patches_disjoint is not True
    ):
        raise ValueError("explicit selective verified fused source required")
    n = packet.shape[2]
    windows = aligned_windows(tuple(positions), n, packet.source_row_tile)
    state = _state.get()
    state["source_layers"] += 1
    state["projected_rows"] += sum(take for _, take in windows)
    state["original_full_rows"] += n
    state["windows"].append(list(windows))
    if packet.cos.device.type != "cuda":
        # CPU arithmetic oracle only; no CPU performance claim.
        return packet.materialize(attention)
    from mosaickv.evaluation.fused_source_restore import restore
    from mosaickv.evaluation.shared_source_kernel import project

    targets = tuple(
        torch.empty(packet.shape, device=packet.cos.device, dtype=torch.bfloat16) for _ in (0, 1)
    )
    for first, take in windows:
        project(packet, attention, targets, component=-1, window=(first, take, False))
    for component, target in enumerate(targets):
        restore(packet, component, target, record=False)
    # ONLY text rows may be read from these partially initialized buffers.
    # Retained visual protection comes directly from packet.exact, not targets.
    return targets


def cpu_metadata_bytes(packets):
    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
    from mosaickv.evaluation.shared_source_format import SharedSourceFormat

    tensors = []
    for packet in packets:
        while not isinstance(packet, SharedSourceFormat):
            if hasattr(packet, "template"):
                packet = packet.template
            elif hasattr(packet, "base") and packet.base is not None:
                packet = packet.base
            else:
                break
        if isinstance(packet, SharedSourceFormat) and packet.row_positions_cpu is not None:
            tensors.append(packet.row_positions_cpu)
    return backing_bytes(tuple(tensors))
