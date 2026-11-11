"""Unpadded original-K/V packets on the per-head BF16 token-byte lattice.

This is an exact selected-cache transport control, not a FullKV-quality claim.
Every persistent row is used by its head. Padding exists only in the common
bounded-reader workspace and in the explicitly separate native verifier.
"""

from __future__ import annotations

import hashlib
from itertools import pairwise

import torch

from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.request_precision import ranked_positions


class RaggedRawPacket:
    """Exact-cover K/V head owners; contiguous views and CPU position maps.

    A joint owner is precisely [2, N, 128]: K and V cover disjoint halves with
    offsets zero and N*128. Legacy separate owners remain admissible only when
    each covers its complete allocation. No allocation may span two heads.
    """

    def __init__(self, pairs, positions, *, original_length, prompt_capacity):
        pairs, positions = tuple(pairs), tuple(positions)
        if (
            not pairs
            or len(pairs) != len(positions)
            or type(original_length) is not int
            or original_length < 1
            or type(prompt_capacity) is not int
            or not 1 <= prompt_capacity <= original_length
        ):
            raise ValueError("nonempty per-head pairs, maps and bounded prompt capacity required")
        device = pairs[0][0].device
        for pair, logical in zip(pairs, positions, strict=True):
            if (
                len(pair) != 2
                or pair[0].ndim != 2
                or pair[0].shape != pair[1].shape
                or pair[0].shape[-1] != 128
                or not 1 <= pair[0].shape[0] <= prompt_capacity
                or any(
                    t.dtype != torch.bfloat16 or t.device != device or not t.is_contiguous()
                    for t in pair
                )
                or not isinstance(logical, torch.Tensor)
                or logical.device.type != "cpu"
                or logical.dtype != torch.int64
                or logical.shape != (pair[0].shape[0],)
                or not logical.is_contiguous()
            ):
                raise ValueError(
                    "exact contiguous BF16 N128 heads and sorted CPU positions required"
                )
            # The maps are already required to be contiguous CPU int64. Exact
            # Python-integer checks preserve bounds and strict monotonicity
            # without manufacturing three Boolean tensors per head. This also
            # avoids thousands of unrelated CPU operations entering the full
            # GPU-owner audit; no executed ownership observation is skipped.
            values = logical.tolist()
            if any(not 0 <= position < original_length for position in values) or any(
                before >= after for before, after in pairwise(values)
            ):
                raise ValueError(
                    "exact contiguous BF16 N128 heads and sorted CPU positions required"
                )
        owner_intervals, layouts = [], []
        for key, value in pairs:
            owners = tuple(t.untyped_storage() for t in (key, value))
            same_owner = owners[0]._cdata == owners[1]._cdata
            component_elements = key.numel()
            component_bytes = component_elements * key.element_size()
            if same_owner:
                if (
                    key.storage_offset() != 0
                    or value.storage_offset() != component_elements
                    or owners[0].nbytes() != 2 * component_bytes
                ):
                    raise ValueError(
                        "joint K/V views must be disjoint exact-cover [2,N,128] halves"
                    )
                layouts.append("joint_KV_per_head_2ND_exact_cover")
                used = owners[:1]
            else:
                if any(
                    t.storage_offset() != 0 or owner.nbytes() != component_bytes
                    for t, owner in zip((key, value), owners, strict=True)
                ):
                    raise ValueError("per-head owners require exact-size backing, not padded views")
                layouts.append("separate_K_and_V_legacy_exact_cover")
                used = owners
            # Address-range checks additionally reject overlapping independently
            # wrapped storages; pointer/_cdata inequality alone is insufficient.
            for owner in used:
                start, stop = owner.data_ptr(), owner.data_ptr() + owner.nbytes()
                if any(
                    start < other_stop and other_start < stop
                    for other_start, other_stop in owner_intervals
                ):
                    raise ValueError("K/V owners must not alias another component or head")
                owner_intervals.append((start, stop))
        tensors = tuple(t for pair in pairs for t in pair)
        if backing_bytes(tensors) != sum(t.numel() * t.element_size() for t in tensors):
            raise ValueError("all backing bytes must be used by exactly one K/V component")
        self.pair_by_head, self.original_positions = pairs, positions
        self.prompt_lengths = tuple(pair[0].shape[0] for pair in pairs)
        self.original_length = original_length
        self.shape = (1, len(pairs), prompt_capacity, 128)
        self.allocation_layout_per_head = tuple(layouts)
        self.allocation_layout = layouts[0] if len(set(layouts)) == 1 else "mixed_exact_cover_heads"

    def tensors(self):
        return tuple(t for pair in self.pair_by_head for t in pair)

    @property
    def component_bytes(self):
        return backing_bytes(self.tensors())

    def fill_component(
        self, attention, component, target, *, rope_scratch=None, workspace_observer=None
    ):
        del attention, rope_scratch
        self.fill_group(
            None, component, 0, self.shape[1], target, workspace_observer=workspace_observer
        )

    def fill_group(
        self, attention, component, start, stop, target, *, workspace_observer=None, **kwargs
    ):
        del attention, kwargs
        if (
            type(component) is not int
            or component not in (0, 1)
            or type(start) is not int
            or type(stop) is not int
            or not 0 <= start < stop <= self.shape[1]
            or tuple(target.shape) != (1, stop - start, self.shape[-2], 128)
            or target.dtype != torch.bfloat16
            or target.device != self.tensors()[0].device
            or target.stride(-1) != 1
        ):
            raise ValueError("bounded BF16 head-group prompt workspace required")
        owned = {
            (t.untyped_storage().data_ptr(), t.untyped_storage()._cdata) for t in self.tensors()
        }
        if (target.untyped_storage().data_ptr(), target.untyped_storage()._cdata) in owned:
            raise ValueError("workspace must not alias immutable packet owners")
        target.zero_()
        for offset, head in enumerate(range(start, stop)):
            target[0, offset, : self.prompt_lengths[head]].copy_(self.pair_by_head[head][component])
        if workspace_observer is not None:
            workspace_observer((target,))


class PackedRaggedRawPacket(RaggedRawPacket):
    """Opt-in exact layer owner, disjoint head/component views, no padding.

    This is deliberately a separate layout: the legacy packet continues to
    reject every cross-head shared allocation. Here one flat owner is covered
    exactly, in head order, by each head's K rows followed by its V rows.
    Exposing the single owner to ownership accounting avoids repeatedly
    resolving many head allocations; it does not omit a byte or observation.
    """

    def __init__(self, owner, pairs, positions, *, original_length, prompt_capacity, mandatory=()):
        pairs, positions = tuple(pairs), tuple(positions)
        if (
            not isinstance(owner, torch.Tensor)
            or owner.dtype != torch.bfloat16
            or owner.device.type not in ("cpu", "cuda")
            or owner.ndim != 1
            or not owner.is_contiguous()
            or owner.storage_offset() != 0
            or owner.numel() < 1
            or owner.untyped_storage().nbytes() != owner.numel() * owner.element_size()
            or not pairs
            or len(pairs) != len(positions)
            or type(original_length) is not int
            or original_length < 1
            or type(prompt_capacity) is not int
            or not 1 <= prompt_capacity <= original_length
        ):
            raise ValueError("one exact contiguous BF16 layer owner and per-head maps required")
        if (
            type(mandatory) is not tuple
            or any(type(p) is not int or not 0 <= p < original_length for p in mandatory)
            or tuple(sorted(set(mandatory))) != mandatory
        ):
            raise ValueError("sorted unique original mandatory token positions required")
        storage, cursor = owner.untyped_storage(), 0
        for pair, logical in zip(pairs, positions, strict=True):
            if (
                not isinstance(pair, (tuple, list))
                or len(pair) != 2
                or any(not isinstance(t, torch.Tensor) for t in pair)
                or pair[0].ndim != 2
                or pair[0].shape != pair[1].shape
                or pair[0].shape[-1] != 128
                or not 1 <= pair[0].shape[0] <= prompt_capacity
                or any(
                    t.device != owner.device
                    or t.dtype != owner.dtype
                    or not t.is_contiguous()
                    or t.untyped_storage()._cdata != storage._cdata
                    for t in pair
                )
                or not isinstance(logical, torch.Tensor)
                or logical.device.type != "cpu"
                or logical.dtype != torch.int64
                or logical.shape != (pair[0].shape[0],)
                or not logical.is_contiguous()
            ):
                raise ValueError("exact contiguous BF16 head views and sorted CPU maps required")
            count = pair[0].numel()
            if (
                pair[0].storage_offset() != cursor
                or pair[1].storage_offset() != cursor + count
                or cursor + 2 * count > owner.numel()
            ):
                raise ValueError("layer head K/V intervals must be disjoint ordered exact cover")
            cursor += 2 * count
            values = logical.tolist()
            if (
                any(not 0 <= position < original_length for position in values)
                or any(before >= after for before, after in pairwise(values))
                or not set(mandatory).issubset(values)
            ):
                raise ValueError("sorted bounded CPU positions must preserve all mandatory tokens")
        if cursor != owner.numel():
            raise ValueError("every layer-owner byte must be covered, without tail or padding")
        self.owner = owner
        self.pair_by_head, self.original_positions = pairs, positions
        self.prompt_lengths = tuple(pair[0].shape[0] for pair in pairs)
        self.original_length = original_length
        self.shape = (1, len(pairs), prompt_capacity, 128)
        self.allocation_layout = "joint_KV_layer_ragged_head_exact_cover"
        self.allocation_layout_per_head = (self.allocation_layout,) * len(pairs)

    def tensors(self):
        return (self.owner,)

    def fill_group(
        self, attention, component, start, stop, target, *, workspace_observer=None, **kwargs
    ):
        if not isinstance(target, torch.Tensor) or target.device != self.owner.device:
            raise ValueError("bounded BF16 head-group prompt workspace required")
        storage, target_storage = self.owner.untyped_storage(), target.untyped_storage()
        if (
            storage.data_ptr() < target_storage.data_ptr() + target_storage.nbytes()
            and target_storage.data_ptr() < storage.data_ptr() + storage.nbytes()
        ):
            raise ValueError("workspace must not alias immutable packet owners")
        return super().fill_group(
            attention,
            component,
            start,
            stop,
            target,
            workspace_observer=workspace_observer,
            **kwargs,
        )


def _scores(window, heads, length):
    if not isinstance(window, torch.Tensor):
        raise ValueError("prompt-derived tensor attention evidence required")
    if window.ndim == 4 and window.shape[0] == 1:
        window = window[0]
    if window.ndim == 3:
        window = window.float().mean(-2)
    if window.ndim != 2 or tuple(window.shape) != (heads, length):
        raise ValueError("one finite prompt attention distribution per KV head required")
    if not torch.isfinite(window).all():
        raise ValueError("finite prompt-derived attention scores required")
    return window


def _grouped_cpu_positions(scores, lower, remainder, *, layer_index, mandatory):
    """Exactly the old stable ranking, batched by its at most two capacities.

    The global remainder gives a contiguous prefix of ``lower + 1`` heads and
    a suffix of ``lower`` heads in any layer. Each group receives the SAME
    independent stable score sort, mandatory priority, and source-position
    reorder as the former one-row calls. Never truncate the already
    source-sorted larger set to obtain a smaller-capacity head.
    """
    if (
        not isinstance(scores, torch.Tensor)
        or scores.device.type != "cpu"
        or scores.ndim != 2
        or scores.shape[0] < 1
        or type(lower) is not int
        or lower < 1
        or type(remainder) is not int
        or remainder < 0
        or type(layer_index) is not int
        or layer_index < 0
    ):
        raise ValueError("CPU head scores and positive global balanced-capacity metadata required")
    heads = scores.shape[0]
    upper_heads = min(heads, max(0, remainder - layer_index * heads))
    if not upper_heads:
        return ranked_positions(scores, lower, mandatory)
    if upper_heads == heads:
        return ranked_positions(scores, lower + 1, mandatory)
    return (
        *ranked_positions(scores[:upper_heads], lower + 1, mandatory),
        *ranked_positions(scores[upper_heads:], lower, mandatory),
    )


@torch.inference_mode()
def build_raw_packet_layer(
    original_pair,
    attention_window,
    mandatory,
    *,
    layer_index,
    total_layers,
    target_bytes,
    workspace,
    workspace_observer=None,
    packed_layer=False,
):
    """Stream one native layer into its globally balanced exact raw capacity.

    ``attention_window`` is the CPU H-by-N reduction of the original GPU prompt
    window.  Its reduction must precede transfer, matching ``_scores``.  GPU
    indices reuse the already charged head scratch; no gather tensor exists.
    The old whole-cache builder and its semantics remain unchanged.
    """
    from mosaickv.evaluation.bounded_component_attention import ComponentWorkspace

    if type(packed_layer) is not bool:
        raise ValueError("packed_layer must be an explicit Boolean layout opt-in")
    if (
        type(original_pair) is not tuple
        or len(original_pair) != 2
        or any(not isinstance(t, torch.Tensor) for t in original_pair)
        or original_pair[0].ndim != 4
        or original_pair[0].shape[0] != 1
        or original_pair[0].shape[-1] != 128
        or min(original_pair[0].shape) < 1
        or any(
            t.shape != original_pair[0].shape
            or t.dtype != torch.bfloat16
            or t.device != original_pair[0].device
            for t in original_pair
        )
    ):
        raise ValueError("one complete original same-device BF16 BHN128 layer pair required")
    _, heads, length, dimension = original_pair[0].shape
    device = original_pair[0].device
    if (
        type(layer_index) is not int
        or type(total_layers) is not int
        or not 0 <= layer_index < total_layers
        or type(target_bytes) is not int
        or target_bytes < 1
        or target_bytes % 512
    ):
        raise ValueError("valid global layer index/count and exact 512-byte target required")
    if (
        type(mandatory) is not tuple
        or any(type(p) is not int or not 0 <= p < length for p in mandatory)
        or tuple(sorted(set(mandatory))) != mandatory
    ):
        raise ValueError("sorted unique original mandatory token positions required")
    lower, remainder = divmod(target_bytes // 512, total_layers * heads)
    capacity = lower + bool(remainder)
    if lower < max(1, len(mandatory)) or capacity > length:
        raise ValueError("global target cannot preserve mandatory tokens within original capacity")
    if (
        not isinstance(attention_window, torch.Tensor)
        or attention_window.device.type != "cpu"
        or attention_window.shape != (heads, length)
    ):
        raise ValueError("pre-reduced original CPU H-by-N attention scores required")
    scores = _scores(attention_window, heads, length)
    if (
        not isinstance(workspace, ComponentWorkspace)
        or workspace.heads != heads
        or workspace.head_dim != dimension
        or workspace.component.shape != (workspace.slots, heads * dimension)
        or workspace.head.shape != (workspace.slots, dimension)
        or workspace.component.dtype != torch.bfloat16
        or workspace.component.device != device
        or not workspace.component.is_contiguous()
        or workspace.component.untyped_storage().data_ptr()
        == workspace.head.untyped_storage().data_ptr()
        or workspace.head.dtype != torch.bfloat16
        or workspace.head.device != device
        or not workspace.head.is_contiguous()
        or workspace.head.numel() % 4
        or workspace.head.numel() // 4 < capacity
        or length > workspace.slots
    ):
        raise ValueError("compatible charged head scratch with sufficient index capacity required")
    scratch_storage = workspace.head.untyped_storage()
    scratch_start = scratch_storage.data_ptr()
    scratch_stop = scratch_start + scratch_storage.nbytes()
    for original in original_pair:
        storage = original.untyped_storage()
        if (
            scratch_start < storage.data_ptr() + storage.nbytes()
            and storage.data_ptr() < scratch_stop
        ):
            raise ValueError("index scratch cannot alias immutable original K/V")
    index_pool = workspace.head.view(torch.int64).view(-1)
    per_head, position_maps = [], []
    counts = tuple(lower + (layer_index * heads + head < remainder) for head in range(heads))
    packed_owner = (
        torch.empty(2 * sum(counts) * dimension, device=device, dtype=torch.bfloat16)
        if packed_layer
        else None
    )

    def observe(*active):
        if workspace_observer is not None:
            workspace_observer(
                (
                    *workspace.owned_tensors(),
                    *original_pair,
                    *((packed_owner,) if packed_layer else (t for pair in per_head for t in pair)),
                    *active,
                )
            )

    observe()
    logical_by_head = _grouped_cpu_positions(
        scores, lower, remainder, layer_index=layer_index, mandatory=mandatory
    )
    cursor = 0
    for head, count in enumerate(counts):
        logical = logical_by_head[head]
        host_positions = torch.tensor(logical, dtype=torch.int64, device="cpu")
        indices = index_pool[:count]
        indices.copy_(host_positions)
        if packed_layer:
            elements = count * dimension
            retained = (
                packed_owner[cursor : cursor + elements].view(count, dimension),
                packed_owner[cursor + elements : cursor + 2 * elements].view(count, dimension),
            )
            cursor += 2 * elements
        else:
            owner = torch.empty((2, count, dimension), device=device, dtype=torch.bfloat16)
            retained = owner[0], owner[1]
        observe(*retained)
        for source, target in zip(original_pair, retained, strict=True):
            torch.index_select(source[0, head], 0, indices, out=target)
        per_head.append(retained)
        position_maps.append(host_positions)
        observe()
    packet = (
        PackedRaggedRawPacket(
            packed_owner,
            per_head,
            position_maps,
            original_length=length,
            prompt_capacity=capacity,
            mandatory=mandatory,
        )
        if packed_layer
        else RaggedRawPacket(
            per_head, position_maps, original_length=length, prompt_capacity=capacity
        )
    )
    actual = packet.component_bytes
    expected = 512 * sum(lower + (layer_index * heads + h < remainder) for h in range(heads))
    if actual != expected:
        raise RuntimeError("constructed streaming layer differs from its global exact byte share")
    return packet, dict(
        kind=(
            "original_BF16_streaming_layer_packed_exact_selection_transport_control"
            if packed_layer
            else "original_BF16_streaming_ragged_exact_selection_transport_control"
        ),
        layer_index=layer_index,
        total_layers=total_layers,
        global_target_bytes=target_bytes,
        actual_raw_KV_bytes=actual,
        expected_layer_bytes=expected,
        prompt_lengths=list(packet.prompt_lengths),
        workspace_prompt_capacity=capacity,
        persistent_padding_bytes=0,
        constructor_KV_gather_temporary_allocations=0,
        constructor_index_workspace="charged_shared_head_scratch",
        CPU_ranking_policy="stable_per_head_sort_batched_by_global_lower_upper_capacity",
        CPU_ranking_groups=(2 if 0 < remainder - layer_index * heads < heads else 1),
        allocation_layout=packet.allocation_layout,
        CPU_position_metadata_bytes=backing_bytes(tuple(position_maps)),
        original_length=length,
        functional_head_token_quantum=512,
        original_KV_values_preserved=True,
        original_positions=[p.tolist() for p in packet.original_positions],
        position_map_sha256=hashlib.sha256(
            b"".join(p.numpy().tobytes() for p in packet.original_positions)
        ).hexdigest(),
        benchmark_labels_or_reference_answers_used=False,
        whole_request_KV_peak_certified=False,
    )


@torch.inference_mode()
def packed_selftest(*, device="cuda"):
    """Bounded operator-only parity fixture; no checkpoint or quality claim.

    Run before model admission. The legacy path remains the independent
    word/selection reference; no allocator equality is inferred from this
    fixture, because the model lifecycle ledger must measure that separately.
    """
    from mosaickv.evaluation.bounded_component_attention import ComponentWorkspace

    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(471)
    heads, layers = 32, 3
    workspace = ComponentWorkspace(device=device, heads=heads)
    cases = []
    for length, lower, remainder, layer, mode in (
        (13, 6, 17, 0, "ties"),
        (17, 7, 70, 2, "ascending"),
        (9, 4, 0, 1, "zeros"),
    ):
        original = tuple(
            torch.randn((1, heads, length, 128), device=device, generator=generator).bfloat16()
            for _ in range(2)
        )
        scores = torch.arange(length, dtype=torch.float32).repeat(heads, 1)
        if mode == "ties":
            scores.remainder_(3)
        elif mode == "zeros":
            scores.zero_()
        kwargs = dict(
            layer_index=layer,
            total_layers=layers,
            target_bytes=(layers * heads * lower + remainder) * 512,
            workspace=workspace,
        )
        legacy, reference = build_raw_packet_layer(original, scores, (0, length - 1), **kwargs)
        observed_roots = []

        def observe(tensors, observed_roots=observed_roots):
            observed_roots.append(tuple(tensors))

        packet, report = build_raw_packet_layer(
            original,
            scores,
            (0, length - 1),
            packed_layer=True,
            workspace_observer=observe,
            **kwargs,
        )
        if (
            packet.prompt_lengths != legacy.prompt_lengths
            or packet.component_bytes != legacy.component_bytes
            or report["position_map_sha256"] != reference["position_map_sha256"]
            or any(not any(t is packet.owner for t in row) for row in observed_roots)
        ):
            raise RuntimeError("packed raw operator changed selection, bytes or owner observation")
        for actual, expected in zip(packet.pair_by_head, legacy.pair_by_head, strict=True):
            if any(
                not torch.equal(a.view(torch.int16), b.view(torch.int16))
                for a, b in zip(actual, expected, strict=True)
            ):
                raise RuntimeError("packed raw operator changed original K/V words")
        cases.append(
            dict(
                length=length,
                layer=layer,
                lower=lower,
                remainder=remainder,
                mode=mode,
                word_parity=True,
                position_map_sha256=report["position_map_sha256"],
                actual_raw_KV_bytes=packet.component_bytes,
                legacy_allocation_owners=heads,
                packed_allocation_owners=1,
                persistent_padding_bytes=0,
                constructor_observations=len(observed_roots),
            )
        )
    return dict(
        device=str(device),
        cases=cases,
        failures=[],
        measured_allocator_or_quality_claim=False,
    )


@torch.inference_mode()
def build_raw_packets(adapter, pairs, attention_windows, target_bytes, mandatory):
    """Spend EXACT target bytes on balanced, functional original-K/V head rows.

    In the audited D=128, BF16, K+V geometry a head-token is exactly 512 bytes.
    Every head retains floor(total_rows / routes) or one additional row. Tie
    breaks follow layer/head order, then original source position. No labels,
    synthetic fill, source approximation, or uncharged GPU position maps.
    """
    pairs = tuple(pairs)
    if not pairs or len(pairs) != len(adapter._language_layers()):
        raise ValueError("one original K/V pair and attention window per model layer required")
    shape = tuple(pairs[0][0].shape)
    if len(shape) != 4 or shape[0] != 1 or shape[-1] != 128 or min(shape) < 1:
        raise ValueError("nonempty batch-one original BF16 BHN128 geometry required")
    _, heads, length, dimension = shape
    device = pairs[0][0].device
    if any(
        len(pair) != 2
        or any(
            tuple(t.shape) != shape or t.dtype != torch.bfloat16 or t.device != device for t in pair
        )
        for pair in pairs
    ):
        raise ValueError("same original BF16 layer geometry and device required")
    if (
        type(mandatory) is not tuple
        or any(type(p) is not int or not 0 <= p < length for p in mandatory)
        or tuple(sorted(set(mandatory))) != mandatory
    ):
        raise ValueError("sorted unique original mandatory token positions required")
    quantum = 2 * 2 * dimension
    if type(target_bytes) is not int or target_bytes < 1 or target_bytes % quantum:
        raise ValueError("exact raw target must lie on the 512-byte functional head-token lattice")
    layers, total_rows = len(pairs), target_bytes // quantum
    lower, remainder = divmod(total_rows, layers * heads)
    if lower < max(1, len(mandatory)) or lower + bool(remainder) > length:
        raise ValueError("exact target cannot preserve mandatory tokens within original capacity")
    capacity = lower + bool(remainder)
    packets = []
    for layer, original in enumerate(pairs):
        scores = _scores(attention_windows[layer], heads, length)
        per_head, position_maps = [], []
        for head in range(heads):
            count = lower + (layer * heads + head < remainder)
            logical = ranked_positions(scores[head : head + 1], count, mandatory)[0]
            host_positions = torch.tensor(logical, dtype=torch.int64, device="cpu")
            indices = host_positions.to(device)
            # A head's complete K+V payload is 512*N bytes, including odd N.
            # One functional owner eliminates per-component allocator rounding
            # without padding or changing any selection/value. The pinned Torch
            # index_select(out=...) writes directly into both disjoint halves.
            owner = torch.empty((2, count, dimension), device=device, dtype=torch.bfloat16)
            retained = owner[0], owner[1]
            for source, target in zip(original, retained, strict=True):
                torch.index_select(source[0, head], 0, indices, out=target)
            per_head.append(retained)
            position_maps.append(host_positions)
        packets.append(
            RaggedRawPacket(
                per_head, position_maps, original_length=length, prompt_capacity=capacity
            )
        )
    packets = tuple(packets)
    actual = backing_bytes(tuple(t for packet in packets for t in packet.tensors()))
    if actual != target_bytes:
        raise RuntimeError("constructed unpadded original-K/V owners do not exactly match target")
    maps = tuple(p for packet in packets for p in packet.original_positions)
    return packets, dict(
        kind="original_BF16_ragged_exact_selection_transport_control",
        actual_raw_KV_bytes=actual,
        target_bytes=target_bytes,
        functional_head_token_quantum=quantum,
        functional_head_tokens=total_rows,
        persistent_padding_bytes=0,
        allocation_layout="joint_KV_per_head_2ND_exact_cover",
        raw_KV_allocation_owner_count=layers * heads,
        every_raw_owner_bytes_multiple_512=True,
        constructor_KV_gather_temporary_allocations=0,
        no_unused_fill=True,
        original_KV_values_preserved=True,
        mandatory_positions=list(mandatory),
        original_positions=[[p.tolist() for p in packet.original_positions] for packet in packets],
        prompt_lengths=[list(packet.prompt_lengths) for packet in packets],
        original_length=length,
        workspace_prompt_capacity=capacity,
        CPU_position_metadata_bytes=backing_bytes(maps),
        position_map_sha256=hashlib.sha256(b"".join(p.numpy().tobytes() for p in maps)).hexdigest(),
        benchmark_labels_or_reference_answers_used=False,
        FullKV_quality_preservation_claim=False,
        whole_request_KV_peak_certified=False,
    )


@torch.inference_mode()
def native_same_selected_state(adapter, prefill, packets):
    """Independent padded HF verifier; its padding is NOT stored by the packet."""
    from transformers import DynamicCache

    from mosaickv.adapters.huggingface.types import CacheLayerSnapshot, CacheSnapshot
    from mosaickv.backends.hf_runtime import PackedRuntimeCache, _state_from_packed

    packets = tuple(packets)
    if not packets or len(packets) != len(adapter._language_layers()):
        raise ValueError("one verified ragged packet per model layer required")
    if len({p.shape for p in packets}) != 1:
        raise ValueError("common virtual prompt capacity required for native verification")
    capacity = packets[0].shape[-2]
    snapshots, validity = [], []
    for packet in packets:
        key, value = (
            torch.empty(packet.shape, device=packet.tensors()[0].device, dtype=torch.bfloat16)
            for _ in range(2)
        )
        packet.fill_component(None, 0, key)
        packet.fill_component(None, 1, value)
        mask = torch.zeros(packet.shape[1:3], device=key.device, dtype=torch.bool)
        for head, count in enumerate(packet.prompt_lengths):
            mask[head, :count] = True
        snapshots.append(CacheLayerSnapshot(key, value, 2))
        validity.append(mask)
    snapshot = CacheSnapshot(
        tuple(snapshots),
        DynamicCache,
        "cache_object",
        capacity,
        adapter.capabilities.cached_key_state,
    )
    packed = PackedRuntimeCache(
        snapshot, tuple(validity), capacity, all(bool(v.all()) for v in validity), (), (), ()
    )
    return _state_from_packed(adapter, prefill.state, packed)


@torch.inference_mode()
def run_native_same_selected(adapter, prefill, packets, *, forced=None, tokens=128):
    from mosaickv.evaluation.exact_selected_control import _decode

    packets = tuple(packets)
    state = native_same_selected_state(adapter, prefill, packets)
    logits = _decode(adapter, prefill, state, tokens, forced)
    pairs, _ = adapter._legacy_layers(state.past_key_values)
    return logits, dict(
        reference="native_HF_padded_validity_mask_of_SAME_ragged_original_KV_not_FullKV",
        output_predictions=tokens,
        generated_KV_updates=tokens - 1,
        original_next_decode_position=state.next_decode_position,
        native_reference_final_backing_bytes=backing_bytes(t for pair in pairs for t in pair),
        native_verifier_padding_excluded_from_candidate_measurements=True,
        timing_measured=False,
        no_latency_or_benchmark_quality_claim=True,
    )
