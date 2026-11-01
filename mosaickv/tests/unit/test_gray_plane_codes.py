from dataclasses import replace

import numpy as np
import pytest

from mosaickv.evaluation.gray_plane_codes import decode, encode, pack_indices, unpack_indices


@pytest.mark.parametrize("kind", ["random", "zero", "seven", "central", "sparse", "alternating"])
def test_every_original_three_bit_index_is_preserved_with_all_headers_charged(kind):
    rng = np.random.default_rng(961)
    q = rng.integers(0, 8, 128 * 32, dtype=np.uint8)
    if kind in ("zero", "seven"):
        q.fill(0 if kind == "zero" else 7)
    elif kind == "central":
        q = rng.choice(np.asarray([3, 4], dtype=np.uint8), q.size)
    elif kind == "sparse":
        q.fill(3)
        q[::67] = 1
    elif kind == "alternating":
        q = np.resize(np.arange(8, dtype=np.uint8), q.size)
    original = pack_indices(q)
    assert np.array_equal(unpack_indices(original), q)
    packet = encode(original)
    assert np.array_equal(decode(packet), original)
    assert packet.serialized_bytes <= original.nbytes
    assert (
        packet.serialized_bytes
        == packet.headers.nbytes + packet.offsets.nbytes + packet.payload.nbytes
    )
    if kind == "random":
        assert packet.raw
    elif kind in ("zero", "seven", "central", "sparse"):
        assert not packet.raw


def test_malformed_offset_header_and_truncated_payload_are_rejected():
    original = pack_indices(np.full(256, 3, dtype=np.uint8))
    packet = encode(original)
    for changed in (
        replace(packet, headers=packet.headers | 128),
        replace(packet, offsets=np.array([0, 10, 1], dtype=np.uint32)),
        replace(packet, payload=np.asarray([17], dtype=np.uint8)),
    ):
        with pytest.raises(ValueError):
            decode(changed)


def test_non_group_aligned_payload_cannot_claim_padding_savings():
    with pytest.raises(ValueError):
        encode(pack_indices(np.zeros(8, dtype=np.uint8)))
    with pytest.raises(ValueError):
        pack_indices(np.asarray([8] * 8, dtype=np.uint8))
