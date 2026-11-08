from dataclasses import replace

import numpy as np
import pytest

from mosaickv.evaluation.all_head_source_codec import pack_symbols, symbols
from mosaickv.evaluation.gray_source_codec import GrayCodes, decode, encode, screen, transform


def test_symbolwise_gray_matches_definition_and_inverts_all_symbols():
    value = np.tile(np.arange(8, dtype=np.uint8), 256)
    packed = pack_symbols(value)
    assert np.array_equal(symbols(transform(packed)), value ^ (value >> 1))
    assert np.array_equal(transform(transform(packed), inverse=True), packed)


@pytest.mark.parametrize("seed", range(8))
def test_raw_delta_mixed_group_roundtrips_exactly(seed):
    rng = np.random.default_rng(seed)
    previous = rng.integers(0, 256, 384 * 6, dtype=np.uint8)
    current = previous.copy()
    current[:384] = rng.integers(0, 256, 384, dtype=np.uint8)
    current[385::31] ^= 3
    for mode in (True, False):
        packet = encode(previous, current, gray=mode)
        assert np.array_equal(decode(previous, packet, gray=mode), current)
        assert packet.serialized_bytes == packet.offsets.nbytes + packet.payload.nbytes
    packet, proof = screen(previous, current)
    assert np.array_equal(decode(previous, packet), current)
    assert proof["selected_bytes"] <= proof["common_bytes"]


def test_transform_is_group_local_and_xor_linear_for_native_in_place_replay():
    rng = np.random.default_rng(71)
    a, b = (rng.integers(0, 256, 384 * 5, dtype=np.uint8) for _ in range(2))
    assert np.array_equal(transform(a ^ b), transform(a) ^ transform(b))
    assert np.array_equal(transform(a), np.concatenate([transform(v) for v in a.reshape(-1, 384)]))
    gray = transform(a)
    assert np.array_equal(
        a, np.concatenate([transform(v, inverse=True) for v in gray.reshape(-1, 384)])
    )


def test_bad_lengths_offsets_and_literal_streams_fail_closed():
    previous = np.zeros(384, dtype=np.uint8)
    current = previous.copy()
    current[31] = 7
    packet = encode(previous, current)
    assert isinstance(packet, GrayCodes)
    offsets = packet.offsets.copy()
    offsets[-1] += 1
    with pytest.raises(ValueError, match="offsets"):
        decode(previous, replace(packet, offsets=offsets))
    payload = packet.payload.copy()
    payload[-1] = 0
    with pytest.raises(ValueError, match="literal"):
        decode(previous, replace(packet, payload=payload))
    with pytest.raises(ValueError, match="384"):
        encode(previous[:3], previous[:3])
    assert np.array_equal(decode(None, encode(None, current)), current)
