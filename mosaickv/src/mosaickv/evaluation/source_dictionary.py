"""CPU-only, lossless all-consumer source dictionary feasibility variant.

Dictionary rows are ORIGINAL packed quantizer symbols, not averaged K/V.
They predict every projection consumer without changing any source bit or
RoPE position. No model-quality/latency claim follows from this CPU format.
"""

from dataclasses import dataclass

import numpy as np

from mosaickv.evaluation import all_head_source_codec as base
from mosaickv.evaluation import lossless_code_delta as byte_codec

DICTIONARY = 128


def common_encode(previous, current):
    """Actual runtime control: best byte/symbol codec, stable legacy tie."""
    symbol, legacy = base.encode(previous, current), byte_codec.encode(previous, current)
    return symbol if symbol.serialized_bytes < legacy.serialized_bytes else legacy


@dataclass(frozen=True)
class DictionaryCodes:
    length: int
    row_symbols: int
    offsets: np.ndarray
    payload: np.ndarray
    anchors: np.ndarray
    row_anchor: np.ndarray

    @property
    def serialized_bytes(self):
        return sum(t.nbytes for t in (self.offsets, self.payload, self.anchors, self.row_anchor))


def _geometry(current, row_symbols, anchor_count):
    current = base.validate(current)
    if (
        type(row_symbols) is not int
        or row_symbols < base.SYMBOLS
        or row_symbols % base.SYMBOLS
        or len(current) % (row_symbols * 3 // 8)
        or type(anchor_count) is not int
        or not 1 <= anchor_count <= 32
    ):
        raise ValueError("complete source rows and 1--32 dictionary anchors required")
    return current.reshape(-1, row_symbols * 3 // 8)


def _palette(row, plan, group, flag=0):
    a, b = int(plan[0][group]), int(plan[1][group])
    other = row != a
    rare = row[other] != b
    body = np.concatenate(
        (
            np.array([a | (b << 3) | flag], dtype=np.uint8),
            np.packbits(other, bitorder="little"),
            np.packbits(rare, bitorder="little"),
            base.pack_symbols(row[other][rare]),
        )
    )
    if len(body) != plan[2][group]:
        raise RuntimeError("dictionary payload differs from planned physical bytes")
    return body


def facility_anchors(previous, current, *, row_symbols=4096, maximum=4, candidates=16):
    """Source-to-anchor gain graph, O(rows * bounded candidates).

    Integer residual costs, not semantic similarity or value norm, define
    coverage. An anchor must pay for its packed row and the first also pays
    for the row-ID map. Exact final packet acceptance remains mandatory.
    """
    rows = _geometry(current, row_symbols, maximum)
    if type(candidates) is not int or not maximum <= candidates <= 32:
        raise ValueError("bounded candidate count in [maximum,32] required")
    ids = tuple(
        sorted(set(np.linspace(0, len(rows) - 1, min(candidates, len(rows)), dtype=int).tolist()))
    )
    symbols = base.symbols(rows.reshape(-1)).reshape(-1, base.SYMBOLS)
    g = row_symbols // base.SYMBOLS
    best = np.minimum(base._plan(symbols)[2], base.BYTES)
    if previous is not None:
        previous = base.validate(previous)
        if previous.shape != current.shape:
            raise ValueError("same original predecessor shape required")
        delta = symbols ^ base.symbols(previous).reshape(symbols.shape)
        best = np.minimum(best, base._plan(delta)[2])
    gain = []
    for candidate in ids:
        prototype = symbols.reshape(len(rows), g, base.SYMBOLS)[candidate]
        sizes = base._plan(symbols ^ np.tile(prototype, (len(rows), 1)))[2]
        gain.append((best - np.minimum(best, sizes)).reshape(len(rows), g).sum(1))
    gain = np.stack(gain)
    covered = np.zeros(len(rows), dtype=np.int64)
    selected, records = [], []
    for _ in range(maximum):
        marginal = np.maximum(gain - covered, 0).sum(1)
        for i in selected:
            marginal[i] = -1
        winner = int(marginal.argmax())
        cost = rows.shape[1] + (len(rows) if not selected else 0)
        if marginal[winner] <= cost:
            break
        records.append(
            dict(
                anchor_row=ids[winner],
                gross_gain_bytes=int(marginal[winner]),
                new_dictionary_and_map_bytes=cost,
            )
        )
        covered = np.maximum(covered, gain[winner])
        selected.append(winner)
    return tuple(sorted(ids[i] for i in selected)), dict(
        candidate_rows=list(ids),
        accepted_anchor_steps=records,
        gain_graph_edges=gain.size,
        bounded_candidates=len(ids),
        predicted_total_group_payload_gain_bytes=int(covered.sum()),
        actual_packet_acceptance_still_required=True,
    )


def encode(previous, current, *, row_symbols=4096, anchor_count=4, anchor_rows=None):
    """Keep the common codec unless actual complete dictionary storage wins.

    Fixed uniform anchors and exact entropy costs are label-blind. One anchor
    per row serves every 512-symbol subblock; direct/previous-layer prediction
    remain available for every group. Stable ties preserve the common codec.
    """
    rows = _geometry(current, row_symbols, anchor_count)
    current = rows.reshape(-1)
    old_packet = common_encode(previous, current)
    if anchor_rows is None:
        anchor_rows = tuple(range(0, len(rows), max(1, len(rows) // anchor_count)))[:anchor_count]
    if (
        not isinstance(anchor_rows, tuple)
        or not anchor_rows
        or len(anchor_rows) > anchor_count
        or any(type(i) is not int or not 0 <= i < len(rows) for i in anchor_rows)
        or tuple(sorted(set(anchor_rows))) != anchor_rows
    ):
        raise ValueError("unique sorted original row anchors required")
    symbols = base.symbols(current).reshape(-1, base.SYMBOLS)
    groups_per_row = row_symbols // base.SYMBOLS
    direct = base._plan(symbols)
    best_sizes = np.minimum(direct[2], base.BYTES)
    modes = np.zeros(len(symbols), dtype=np.uint8)
    old_values, old_plan = None, None
    if previous is not None:
        previous = base.validate(previous)
        if previous.shape != current.shape:
            raise ValueError("original predecessor source geometry changed")
        old_values = symbols ^ base.symbols(previous).reshape(symbols.shape)
        old_plan = base._plan(old_values)
        better = old_plan[2] < best_sizes
        modes[better] = 64
        best_sizes = np.minimum(best_sizes, old_plan[2])
    anchors = rows[list(anchor_rows)].copy()
    anchor_symbols = base.symbols(anchors.reshape(-1)).reshape(-1, groups_per_row, base.SYMBOLS)
    row_anchor = np.zeros(len(rows), dtype=np.uint8)
    row_cost = best_sizes.reshape(len(rows), groups_per_row).sum(1)
    # Two tensor-sized passes, not a source-row dictionary or dense token graph.
    # This is still a CPU feasibility encoder, not an optimized online planner.
    for i, prototype in enumerate(anchor_symbols):
        values = symbols ^ np.tile(prototype, (len(rows), 1))
        size = base._plan(values)[2]
        cost = np.minimum(best_sizes, size).reshape(len(rows), groups_per_row).sum(1)
        better = cost < row_cost
        row_anchor[better] = i
        row_cost[better] = cost[better]
    predicted = anchor_symbols[row_anchor].reshape(symbols.shape)
    dictionary_values = symbols ^ predicted
    dictionary_plan = base._plan(dictionary_values)
    better = dictionary_plan[2] < best_sizes
    modes[better] = DICTIONARY
    parts, offsets, cursor = [], [], 0
    for group, mode in enumerate(modes):
        plan = direct if mode == 0 else old_plan if mode == 64 else dictionary_plan
        values = symbols if mode == 0 else old_values if mode == 64 else dictionary_values
        if plan[2][group] >= base.BYTES:
            offsets.append(cursor | int(base.RAW))
            body = current[group * base.BYTES : (group + 1) * base.BYTES].copy()
        else:
            offsets.append(cursor)
            body = _palette(values[group], plan, group, int(mode))
        parts.append(body)
        cursor += len(body)
    offsets.append(cursor)
    packet = DictionaryCodes(
        len(current),
        row_symbols,
        np.array(offsets, dtype=np.uint32),
        np.concatenate(parts),
        anchors,
        row_anchor,
    )
    accepted = packet.serialized_bytes < old_packet.serialized_bytes
    proof = dict(
        dictionary_accepted=accepted,
        common_source_code_bytes=old_packet.serialized_bytes + 1,
        dictionary_source_code_bytes=packet.serialized_bytes + 1,
        common_codec_kind=type(old_packet).__name__,
        dictionary_predicted_groups=int(np.count_nonzero(modes == DICTIONARY)),
        original_source_rows=len(rows),
        anchor_rows=list(anchor_rows),
        charged_anchor_bytes=anchors.nbytes,
        charged_row_anchor_bytes=row_anchor.nbytes,
        charged_offset_bytes=packet.offsets.nbytes,
        charged_payload_bytes=packet.payload.nbytes,
        charged_codec_selector_bytes=1,
        all_heads_and_both_projections_share_identical_decoded_source=True,
        source_rows_and_logical_positions_not_removed=True,
        active_KV_or_whole_request_peak_measured=False,
        benchmark_labels_accessed=False,
        model_or_native_GPU_validated=False,
    )
    return packet if accepted else old_packet, proof


def decode(previous, packet):
    if isinstance(packet, base.SymbolCodes):
        return base.decode(previous, packet)
    if isinstance(packet, byte_codec.DeltaCodes):
        return byte_codec.decode(previous, packet)
    if not isinstance(packet, DictionaryCodes):
        raise ValueError("explicit source dictionary packet required")
    if (
        packet.row_symbols % base.SYMBOLS
        or packet.row_symbols <= 0
        or packet.length <= 0
        or packet.length % (packet.row_symbols * 3 // 8)
        or packet.anchors.dtype != np.uint8
        or packet.anchors.ndim != 2
        or not 1 <= len(packet.anchors) <= 32
        or packet.anchors.shape[1] != packet.row_symbols * 3 // 8
        or packet.row_anchor.dtype != np.uint8
        or packet.row_anchor.shape != (packet.length // (packet.row_symbols * 3 // 8),)
        or np.any(packet.row_anchor >= len(packet.anchors))
    ):
        raise ValueError("invalid complete source dictionary geometry")
    if packet.payload.dtype != np.uint8 or packet.offsets.dtype != np.uint32:
        raise ValueError("invalid dictionary packet dtypes")
    groups = packet.length // base.BYTES
    if packet.offsets.shape != (groups + 1,):
        raise ValueError("invalid dictionary offset count")
    offsets = packet.offsets & base.OFFSET
    if (
        offsets[0] != 0
        or offsets[-1] != len(packet.payload)
        or np.any(offsets[1:] < offsets[:-1])
        or packet.offsets[-1] & base.RAW
    ):
        raise ValueError("invalid dictionary offset bounds")
    # Reuse the strict, already tested palette/escape decoder after turning
    # dictionary prediction into one explicit per-group predecessor stream.
    patched = packet.payload.copy()
    needed = np.zeros(groups, dtype=np.uint8)
    for group in range(groups):
        if packet.offsets[group] & base.RAW:
            continue
        body = patched[offsets[group] : offsets[group + 1]]
        if not len(body) or body[0] & 192 == 192:
            raise ValueError("invalid combined dictionary/previous-layer mode")
        if body[0] & DICTIONARY:
            needed[group] = 2
            body[0] = (body[0] & 127) | 64
        elif body[0] & 64:
            needed[group] = 1
    if np.any(needed == 1):
        if previous is None or len(base.validate(previous)) != packet.length:
            raise ValueError("original previous-layer codes required")
        predictor = previous.copy().reshape(groups, base.BYTES)
    else:
        predictor = np.zeros((groups, base.BYTES), dtype=np.uint8)
    dictionary = packet.anchors[packet.row_anchor].reshape(groups, base.BYTES)
    predictor[needed == 2] = dictionary[needed == 2]
    return base.decode(
        predictor.reshape(-1),
        base.SymbolCodes(packet.length, packet.offsets.copy(), patched, False),
    )
