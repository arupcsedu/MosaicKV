from dataclasses import replace

import numpy as np
import pytest

from mosaickv.evaluation.lossless_code_delta import RAW, decode, encode


@pytest.mark.parametrize("case", ["identical", "sparse", "random", "mixed", "initial", "ragged"])
def test_exact_serialization_and_raw_fallback_charge_every_offset(case):
    rng = np.random.default_rng(824)
    old = rng.integers(0, 256, 128 * 16, dtype=np.uint8)
    current = old.copy()
    if case == "sparse":
        current[::63] ^= 1
    elif case == "random":
        current = rng.integers(0, 256, old.size, dtype=np.uint8)
    elif case == "mixed":
        current[128:256] = rng.integers(0, 256, 128, dtype=np.uint8)
    elif case == "initial":
        old = None
    elif case == "ragged":
        current = current[:128]
    before = None if old is None else old.copy()
    packet = encode(old, current)
    assert np.array_equal(decode(old, packet), current)
    assert packet.serialized_bytes == packet.payload.nbytes + packet.offsets.nbytes
    assert packet.serialized_bytes <= current.nbytes
    if case in ("identical", "sparse", "mixed"):
        assert not packet.raw
    else:
        assert packet.raw
    if before is not None:
        assert np.array_equal(old, before)


def test_complete_chain_resets_without_cross_request_state():
    rng = np.random.default_rng(92)
    original = rng.integers(0, 256, 128 * 8, dtype=np.uint8)
    chain, values, previous = [], [], None
    for i in range(16):
        value = original.copy()
        value[i::37] ^= i
        chain.append(encode(previous, value))
        values.append(value)
        previous = value
    for _ in range(3):
        restored = None
        for packet, expected in zip(chain, values, strict=True):
            restored = decode(restored, packet)
            assert np.array_equal(restored, expected)


def test_bad_offsets_bitmap_and_missing_predecessor_fail_closed():
    old = np.zeros(256, dtype=np.uint8)
    packet = encode(old, old)
    for broken in (
        replace(packet, offsets=np.asarray([0, 1000, 32], dtype=np.uint32)),
        replace(packet, offsets=packet.offsets | RAW),
        replace(packet, payload=np.ones(32, dtype=np.uint8)),
    ):
        with pytest.raises(ValueError):
            decode(old, broken)
    with pytest.raises(ValueError):
        decode(None, packet)
    with pytest.raises(ValueError):
        encode(old, np.zeros(130, dtype=np.uint8))
