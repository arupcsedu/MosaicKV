"""CPU metadata-only feasibility of skipping wholly overwritten projection tiles.

No tensor/kernel modification. Partial tiles remain entirely projected. The
reported fractions count programs, not FLOPs, latency savings or quality gains.
"""

import json
from hashlib import sha256


def tile_coverage(rows, *, heads, length, row_tile, head_group):
    if (
        any(type(x) is not int or x <= 0 for x in (heads, length, row_tile, head_group))
        or row_tile not in (16, 32, 64)
        or head_group not in (1, 2, 4)
        or heads % head_group
        or type(rows) is not tuple
        or any(type(x) is not int or not 0 <= x < heads * length for x in rows)
        or rows != tuple(sorted(set(rows)))
    ):
        raise ValueError(
            "sorted unique bounded rows and audited divisible source geometry required"
        )
    protected = set(rows)
    complete, partial, untouched = 0, 0, 0
    redundant, projected = 0, heads * length
    per_head = []
    for first_head in range(0, heads, head_group):
        counts = dict(complete=0, partial=0, untouched=0)
        for first in range(0, length, row_tile):
            positions = [
                h * length + p
                for h in range(first_head, first_head + head_group)
                for p in range(first, min(first + row_tile, length))
            ]
            hits = sum(x in protected for x in positions)
            redundant += hits
            if hits == len(positions):
                complete += 1
                counts["complete"] += 1
            elif hits:
                partial += 1
                counts["partial"] += 1
            else:
                untouched += 1
                counts["untouched"] += 1
        per_head.append(dict(first_head=first_head, **counts))
    total = complete + partial + untouched
    return dict(
        total_programs=total,
        wholly_overwritten_programs=complete,
        partially_overwritten_programs=partial,
        untouched_programs=untouched,
        wholly_overwritten_fraction=complete / total,
        overwritten_rows=redundant,
        source_rows=projected,
        overwritten_row_fraction=redundant / projected,
        per_head_group=per_head,
        rows_sha256=sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest(),
        partial_tiles_must_not_be_skipped=True,
        measured_latency_savings=False,
        additional_gpu_metadata_bytes=0,
    )


def packet_coverage(packet):
    """Copy only existing position metadata to CPU, outside all timed execution."""
    from mosaickv.evaluation.shared_source_format import SharedSourceFormat
    from mosaickv.evaluation.text_source_storage import TextSourceFormat

    overlay = packet.projection if isinstance(packet, TextSourceFormat) else None
    base = packet.base if overlay is not None else packet
    if not isinstance(base, SharedSourceFormat) or base.projection_backend != "fused":
        raise ValueError("original fused source packet required, not an arbitrary baseline")
    _, heads, length, dim = base.shape
    protected = tuple(base.row_positions.cpu().tolist())
    kw = dict(
        heads=heads,
        length=length,
        row_tile=base.source_row_tile,
        head_group=base.source_head_group,
    )
    exact = tile_coverage(protected, **kw)
    combined = set(protected)
    if overlay is not None:
        slots = overlay.slots.cpu().tolist()
        if len(slots) != heads or any(len(r) != len(overlay.original_text) for r in slots):
            raise ValueError("complete original text-slot metadata required")
        for h, row in enumerate(slots):
            for p in row:
                if type(p) is not int or not -1 <= p < length:
                    raise ValueError("text overlay position outside original prompt")
                if p >= 0:
                    combined.add(h * length + p)
    both = tile_coverage(tuple(sorted(combined)), **kw)
    patches = [int(p.indices.numel()) for p in base.transport_patches]
    return dict(
        shape=list(base.shape),
        row_tile=base.source_row_tile,
        head_group=base.source_head_group,
        dimension=dim,
        protected_only=exact,
        protected_plus_text=both,
        correction_scalars_per_component=patches,
        existing_protected_index_bytes=base.row_positions.numel()
        * base.row_positions.element_size(),
        text_union_requires_separate_kernel_design=True,
        no_selected_positions_or_values_changed=True,
    )
