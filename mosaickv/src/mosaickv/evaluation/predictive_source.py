"""Exact, block-local prediction of shared three-bit projection sources.

This is a common storage variant, not a six-component quality improvement.
Every dependency stays inside the existing 512-symbol decoder block. Unlike
cross-block spatial prediction, it permits safe in-place replay with no halo or
second decoded source. GPU support is gated separately from this CPU oracle.
"""

from dataclasses import dataclass

import numpy as np

from mosaickv.evaluation import all_head_source_codec as symbols
from mosaickv.evaluation import lossless_code_delta as bytes_codec
from mosaickv.evaluation.source_dictionary import common_encode

# Fixed before the consumed-input feasibility screen. 128 is the parent's
# quantization group, not an inferred image grid. No benchmark metadata is read.
LOCAL_MODES = ("xor1", "subtract1", "xor8", "subtract8")
MODES = (*LOCAL_MODES, "delta_add")


def transform(codes, mode, *, inverse=False):
    if mode not in LOCAL_MODES or type(inverse) is not bool:
        raise ValueError("registered block-local predictor and explicit inverse required")
    value = symbols.symbols(codes).reshape(-1, 128)
    lag = 8 if mode.endswith("8") else 1
    xor = mode.startswith("xor")
    if inverse:
        value = value.reshape(-1, 128 // lag, lag)
        restored = (
            np.bitwise_xor.accumulate(value, axis=1)
            if xor
            else (np.cumsum(value, axis=1, dtype=np.uint32) & 7).astype(np.uint8)
        )
    else:
        restored = value.copy()
        restored[:, lag:] = (
            value[:, lag:] ^ value[:, :-lag]
            if xor
            else (value[:, lag:].astype(np.int16) - value[:, :-lag]) & 7
        )
    return symbols.pack_symbols(restored.ravel())


@dataclass(frozen=True)
class PredictiveCodes:
    mode: str
    body: symbols.SymbolCodes
    has_previous: bool = False

    @property
    def serialized_bytes(self):
        # The caller charges one actual selector byte for ALL formats. Values
        # 0/1 denote common byte/symbol packets and 2..6 the fixed predictors.
        return self.body.serialized_bytes


def encode(previous, current, mode):
    if mode == "delta_add":
        value = symbols.symbols(current)
        if previous is not None:
            old = symbols.symbols(previous)
            if old.shape != value.shape:
                raise ValueError("same complete predecessor source geometry required")
            value = ((value.astype(np.int16) - old) & 7).astype(np.uint8)
        return PredictiveCodes(
            mode, symbols.encode(None, symbols.pack_symbols(value)), previous is not None
        )
    required = previous is not None
    previous = None if previous is None else transform(previous, mode)
    return PredictiveCodes(mode, symbols.encode(previous, transform(current, mode)), required)


def decode(previous, packet):
    if isinstance(packet, PredictiveCodes):
        if packet.mode not in MODES or type(packet.has_previous) is not bool:
            raise ValueError("registered predictor and explicit predecessor dependency required")
        if packet.has_previous and previous is None:
            raise ValueError("predictive source packet requires its original predecessor")
        if packet.mode == "delta_add":
            value = symbols.decode(None, packet.body)
            if not packet.has_previous:
                return value
            old, residual = symbols.symbols(previous), symbols.symbols(value)
            if old.shape != residual.shape:
                raise ValueError("same complete predecessor source geometry required")
            return symbols.pack_symbols((old + residual) & 7)
        old = None if previous is None else transform(previous, packet.mode)
        return transform(symbols.decode(old, packet.body), packet.mode, inverse=True)
    if isinstance(packet, symbols.SymbolCodes):
        return symbols.decode(previous, packet)
    if isinstance(packet, bytes_codec.DeltaCodes):
        return bytes_codec.decode(previous, packet)
    raise ValueError("registered common or predictive source packet required")


def screen(previous, current):
    """Charge every actual offset/payload and a real one-byte format selector.

    One existing decoded-source replay owner is needed for the whole stream,
    not another per-packet buffer. The stream analyzer charges its full length
    once for both parent and candidate before admitting operator development.
    """
    parent = common_encode(previous, current)
    best = parent
    rows = []
    for mode in MODES:
        candidate = encode(previous, current, mode)
        if not np.array_equal(decode(previous, candidate), current):
            raise RuntimeError("predictor/residual changed an original source-code bit")
        rows.append(dict(mode=mode, bytes=candidate.serialized_bytes + 1))
        if candidate.serialized_bytes < best.serialized_bytes:
            best = candidate
    if not np.array_equal(decode(previous, parent), current):
        raise RuntimeError("parent code-stream control failed")
    packet = best.body if isinstance(best, PredictiveCodes) else best
    owners = (packet.offsets, packet.payload)
    if any(np.shares_memory(owner, current) for owner in owners):
        raise RuntimeError("replacement retains original source backing")
    if sum(owner.nbytes for owner in owners) != best.serialized_bytes:
        raise RuntimeError("replacement omitted an offset or residual payload owner")
    return best, dict(
        common_bytes=parent.serialized_bytes + 1,
        selected_bytes=best.serialized_bytes + 1,
        selected_mode=best.mode if isinstance(best, PredictiveCodes) else "common",
        proposals=rows,
        original_code_bits_exact=True,
        selector_bytes=1,
        replay_dependency_symbols=512,
        extra_decoded_source_bytes=0,
        source_owner_retained=False,
        numerical_precision_unchanged=True,
        benchmark_labels_accessed=False,
        GPU_or_quality_claim=False,
    )
