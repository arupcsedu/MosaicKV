"""Lossless all-consumer source replacement, CPU reference only initially.

The symbols are ORIGINAL three-bit quantizer codes, not post-RoPE keys or
averaged source values. A palette prototype and exact escape residual preserve
all input features for both projections of every head. This is common storage
coding, not evidence that the six adaptive mechanisms improve quality.
"""

from dataclasses import dataclass

import numpy as np

RAW = np.uint32(1 << 31)
OFFSET = np.uint32((1 << 31) - 1)
SYMBOLS = 512
BYTES = SYMBOLS * 3 // 8


@dataclass(frozen=True)
class SourceCoverage:
    layer: int
    heads: tuple[int, ...]
    components: tuple[str, ...]
    logical_positions: tuple[int, ...]
    feature_intervals: tuple[tuple[int, int], ...]

    def verify(self, *, kv_heads, head_dim, length):
        if any(type(x) is not int or x < 1 for x in (kv_heads, head_dim, length)):
            raise ValueError("nonempty original source geometry required")
        if (
            type(self.layer) is not int
            or self.layer < 0
            or self.heads != tuple(range(kv_heads))
            or self.components != ("key", "value")
            or self.logical_positions != tuple(range(length))
        ):
            raise ValueError("source retirement requires every original head, K/V and position")
        cursor = 0
        for start, end in self.feature_intervals:
            if type(start) is not int or type(end) is not int or start != cursor or end <= start:
                raise ValueError("replacement features must cover the source exactly once")
            cursor = end
        if cursor != kv_heads * head_dim:
            raise ValueError("replacement omits source features used by other heads")
        return dict(
            all_shared_consumers_covered=True,
            kv_head_count=kv_heads,
            projection_consumer_count=2 * kv_heads,
            original_logical_positions_unchanged=True,
            source_replacement_not_single_head_overlay=True,
        )


@dataclass(frozen=True)
class SymbolCodes:
    length: int
    offsets: np.ndarray
    payload: np.ndarray
    raw: bool

    @property
    def serialized_bytes(self):
        return self.offsets.nbytes + self.payload.nbytes


def validate(codes):
    codes = np.asarray(codes)
    if (
        codes.dtype != np.uint8
        or codes.ndim != 1
        or not codes.flags.c_contiguous
        or not codes.size
        or codes.size % BYTES
        or codes.size >= int(OFFSET)
    ):
        raise ValueError("complete 512-symbol three-bit source groups required")
    return codes


def symbols(codes):
    byte = validate(codes).reshape(-1, 3).astype(np.uint32)
    word = byte[:, 0] | (byte[:, 1] << 8) | (byte[:, 2] << 16)
    return ((word[:, None] >> (np.arange(8, dtype=np.uint32) * 3)) & 7).astype(np.uint8).ravel()


def pack_symbols(value):
    value = np.asarray(value, dtype=np.uint8)
    count = value.size
    if np.any(value > 7):
        raise ValueError("symbol exceeds original three-bit width")
    padding = (-count) % 8
    if padding:
        value = np.pad(value, (0, padding))
    word = (value.reshape(-1, 8).astype(np.uint32) << (np.arange(8, dtype=np.uint32) * 3)).sum(
        -1, dtype=np.uint32
    )
    encoded = np.stack([((word >> (8 * i)) & 255).astype(np.uint8) for i in range(3)], -1)
    return encoded.ravel()[: (count * 3 + 7) // 8].copy()


def _plan(value):
    counts = (value[..., None] == np.arange(8)).sum(1)
    first = counts.argmax(1)
    counts[np.arange(len(value)), first] = -1
    second = counts.argmax(1)
    other = value != first[:, None]
    rare = other & (value != second[:, None])
    nonfirst, nrare = other.sum(1), rare.sum(1)
    sizes = 1 + SYMBOLS // 8 + (nonfirst + 7) // 8 + (nrare * 3 + 7) // 8
    return first, second, sizes


def encode(previous, current):
    current = validate(current)
    if previous is not None:
        previous = validate(previous)
        if previous.shape != current.shape:
            raise ValueError("same original logical source layout required")
    value = symbols(current).reshape(-1, SYMBOLS)
    direct = _plan(value)
    delta, delta_plan = None, None
    if previous is not None:
        delta = value ^ symbols(previous).reshape(value.shape)
        delta_plan = _plan(delta)
    offsets, parts, cursor = [], [], 0
    for group in range(len(value)):
        use_delta = delta_plan is not None and delta_plan[2][group] < direct[2][group]
        first, second, sizes = delta_plan if use_delta else direct
        a, b = int(first[group]), int(second[group])
        if sizes[group] >= BYTES:
            body = current[group * BYTES : (group + 1) * BYTES].copy()
            offsets.append(cursor | int(RAW))
        else:
            row = delta[group] if use_delta else value[group]
            other = row != a
            rare = row[other] != b
            body = np.concatenate(
                (
                    np.array([a | (b << 3) | (64 if use_delta else 0)], dtype=np.uint8),
                    np.packbits(other, bitorder="little"),
                    np.packbits(rare, bitorder="little"),
                    pack_symbols(row[other][rare]),
                )
            )
            if len(body) != sizes[group]:
                raise RuntimeError("palette/residual accounting differs from actual payload")
            offsets.append(cursor)
        parts.append(body)
        cursor += len(body)
    offsets.append(cursor)
    packet = SymbolCodes(
        len(current), np.asarray(offsets, dtype=np.uint32), np.concatenate(parts), False
    )
    if packet.serialized_bytes >= len(current):
        return SymbolCodes(len(current), np.empty(0, dtype=np.uint32), current.copy(), True)
    if np.shares_memory(packet.payload, current):
        raise RuntimeError("replacement retains the original source payload")
    return packet


def decode(previous, packet):
    if (
        not isinstance(packet, SymbolCodes)
        or type(packet.raw) is not bool
        or packet.payload.dtype != np.uint8
        or packet.payload.ndim != 1
        or packet.offsets.dtype != np.uint32
        or packet.length <= 0
        or packet.length % BYTES
    ):
        raise ValueError("explicit complete source prototype/residual packet required")
    if packet.raw:
        if packet.offsets.size or len(packet.payload) != packet.length:
            raise ValueError("raw source metadata differs")
        return packet.payload.copy()
    groups = packet.length // BYTES
    offsets = packet.offsets & OFFSET
    if (
        packet.offsets.shape != (groups + 1,)
        or offsets[0] != 0
        or offsets[-1] != len(packet.payload)
        or packet.offsets[-1] & RAW
        or np.any(offsets[1:] < offsets[:-1])
    ):
        raise ValueError("invalid charged source offsets")
    old = None if previous is None else symbols(previous).reshape(groups, SYMBOLS)
    output = np.empty(packet.length, dtype=np.uint8)
    for group in range(groups):
        start, end = int(offsets[group]), int(offsets[group + 1])
        body = packet.payload[start:end]
        if packet.offsets[group] & RAW:
            if len(body) != BYTES:
                raise ValueError("raw source block differs from original layout")
            output[group * BYTES : (group + 1) * BYTES] = body
            continue
        if len(body) < 65 or body[0] & 128:
            raise ValueError("truncated or unknown source palette header")
        a, b = int(body[0] & 7), int((body[0] >> 3) & 7)
        other = np.unpackbits(body[1:65], bitorder="little").astype(bool)
        count = int(other.sum())
        extra = (count + 7) // 8
        if len(body) < 65 + extra:
            raise ValueError("truncated source exception bitmap")
        flags = np.unpackbits(body[65 : 65 + extra], bitorder="little")
        if np.any(flags[count:]):
            raise ValueError("noncanonical source bitmap padding")
        rare = flags[:count].astype(bool)
        nrare = int(rare.sum())
        literal = body[65 + extra :]
        if len(literal) != (nrare * 3 + 7) // 8:
            raise ValueError("source literal count differs from bitmap")
        restored = np.full(SYMBOLS, a, dtype=np.uint8)
        indices = np.flatnonzero(other)
        restored[indices] = b
        if nrare:
            # Pad the independent bitstream for integer decoding; the actual
            # persisted payload above contains no uncharged padding bytes.
            padded = np.pad(literal, (0, (-len(literal)) % 3))
            byte = padded.reshape(-1, 3).astype(np.uint32)
            word = byte[:, 0] | byte[:, 1] << 8 | byte[:, 2] << 16
            values = ((word[:, None] >> (np.arange(8, dtype=np.uint32) * 3)) & 7).ravel()
            if np.any(values[nrare:]):
                raise ValueError("noncanonical source literal padding")
            restored[indices[rare]] = values[:nrare]
        if body[0] & 64:
            if old is None:
                raise ValueError("source delta requires original predecessor codes")
            restored ^= old[group]
        output[group * BYTES : (group + 1) * BYTES] = pack_symbols(restored)
    return output


def screen_lossless(values, globally_exact, *, layer, previous=None):
    """Actual original source code roundtrip plus old/new CPU payload costs.

    Encoders always reconstruct the SAME bytes; predecessor coding opportunities
    therefore do not change when a layer chooses the alternative encoder.
    """
    import hashlib

    import torch

    from mosaickv.evaluation.conditional_precision import pack_unprotected
    from mosaickv.evaluation.lossless_code_delta import decode as old_decode
    from mosaickv.evaluation.lossless_code_delta import encode as old_encode
    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes

    if (
        values.device.type != "cpu"
        or values.dtype != torch.bfloat16
        or values.ndim != 3
        or values.shape[0] != 1
        or values.shape[2] != 4096
        or globally_exact.shape != (values.shape[1],)
        or globally_exact.dtype != torch.bool
        or not bool(values.isfinite().all())
    ):
        raise ValueError(
            "pinned original full-MHA CPU BF16 source and complete exact mask required"
        )
    coverage = SourceCoverage(
        layer, tuple(range(32)), ("key", "value"), tuple(range(values.shape[1])), ((0, 4096),)
    )
    proof = coverage.verify(kv_heads=32, head_dim=128, length=values.shape[1])
    body = pack_unprotected(values[:, None], globally_exact[None], bits=3, axis=2)
    codes = body.codes.numpy()
    alternative, legacy = encode(previous, codes), old_encode(previous, codes)
    restored = decode(previous, alternative)
    if not np.array_equal(restored, codes) or not np.array_equal(
        old_decode(previous, legacy), codes
    ):
        raise RuntimeError("all-head replacement changed original quantizer bits")
    accepted = alternative.serialized_bytes < legacy.serialized_bytes
    row = dict(
        family="all_head_lossless_source",
        **proof,
        source_codes_bitwise=True,
        source_codes_sha256=hashlib.sha256(codes.tobytes()).hexdigest(),
        original_code_bytes=len(codes),
        legacy_serialized_code_bytes=legacy.serialized_bytes,
        alternative_serialized_code_bytes=alternative.serialized_bytes,
        alternative_accepted=accepted,
        numerical_screen_passed=accepted,
        no_parent_token_lock_needed_for_this_bitwise_format=True,
        benchmark_quality_improvement_established=False,
        gpu_cache_backend_implemented=False,
    )
    return dict(
        rows=[row],
        baseline=dict(codes=body.codes, other_bytes=backing_bytes(body.tensors()) - len(codes)),
        candidates={},
        selected_packet=alternative if accepted else legacy,
        selected_codec="source_symbols" if accepted else "legacy_bytes",
    )


def selected_cpu_storage(bodies, packets):
    from mosaickv.evaluation.source_rank_feasibility import cpu_serialized_storage

    raw = sum(b["codes"].numel() + b["other_bytes"] for b in bodies)
    # Actually materialize and charge one codec selector byte per layer.
    selector = np.array([isinstance(p, SymbolCodes) for p in packets], dtype=np.uint8)
    encoded = (
        sum(p.serialized_bytes + b["other_bytes"] for p, b in zip(packets, bodies, strict=True))
        + selector.nbytes
    )
    scratch = max((p.length for p in packets if not p.raw), default=0)
    baseline = cpu_serialized_storage(bodies)
    accepted = encoded + scratch < baseline["net_cpu_payload_bytes"]
    return dict(
        original_cpu_payload_bytes=raw,
        encoded_cpu_payload_bytes=encoded,
        charged_replay_bytes=scratch,
        codec_selector_bytes=selector.nbytes,
        net_cpu_payload_bytes=encoded + scratch if accepted else baseline["net_cpu_payload_bytes"],
        replacement_accepted=accepted,
        source_codes_bitwise=True,
        gpu_kv_storage_measured=False,
        whole_request_peak_certified=False,
    )
