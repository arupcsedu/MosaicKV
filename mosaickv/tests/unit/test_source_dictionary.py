from dataclasses import replace

import numpy as np
import pytest

from mosaickv.evaluation import all_head_source_codec as base
from mosaickv.evaluation import lossless_code_delta as byte_codec
from mosaickv.evaluation.source_dictionary import DictionaryCodes, decode, encode


def fixture():
    rng = np.random.default_rng(17)
    anchor = rng.integers(0, 8, (4, 4096), dtype=np.uint8)
    rows = np.tile(anchor, (40, 1))
    rows[:, 73:75] ^= 1
    return base.pack_symbols(rows.reshape(-1))


def test_all_source_bits_and_accounting_survive_dictionary_replacement():
    current = fixture()
    packet, proof = encode(None, current, anchor_count=4, anchor_rows=(0, 1, 2, 3))
    assert isinstance(packet, DictionaryCodes)
    assert proof["dictionary_accepted"]
    assert np.array_equal(decode(None, packet), current)
    assert packet.serialized_bytes == sum(
        proof[k]
        for k in (
            "charged_anchor_bytes",
            "charged_row_anchor_bytes",
            "charged_offset_bytes",
            "charged_payload_bytes",
        )
    )
    assert not np.shares_memory(packet.anchors, current)
    assert not np.shares_memory(packet.payload, current)
    assert packet.serialized_bytes < proof["common_source_code_bytes"]


def test_predecessor_is_optional_only_when_no_group_uses_it():
    current = fixture()
    previous = current ^ np.uint8(7)
    packet, _ = encode(previous, current, anchor_count=4, anchor_rows=(0, 1, 2, 3))
    assert np.array_equal(decode(previous, packet), current)
    if isinstance(packet, DictionaryCodes):
        # Force one valid palette group to require a previous-layer predictor.
        payload = packet.payload.copy()
        payload[0] = (payload[0] & 63) | 64
        with pytest.raises(ValueError, match="previous-layer"):
            decode(None, replace(packet, payload=payload))


def test_no_gain_preserves_common_codec_and_deterministic_output():
    rng = np.random.default_rng(13)
    current = rng.integers(0, 256, 1536 * 32, dtype=np.uint8)
    first, proof = encode(current, current)
    second, other = encode(current, current)
    assert proof == other
    assert not proof["dictionary_accepted"]
    assert isinstance(first, byte_codec.DeltaCodes)
    assert np.array_equal(first.payload, second.payload)
    assert np.array_equal(decode(current, first), current)
    assert (
        proof["common_source_code_bytes"]
        == byte_codec.encode(current, current).serialized_bytes + 1
    )


def test_corruption_and_invalid_anchor_geometry_fail_closed():
    current = fixture()
    packet, _ = encode(None, current, anchor_count=4, anchor_rows=(0, 1, 2, 3))
    assert isinstance(packet, DictionaryCodes)
    with pytest.raises(ValueError, match="anchors"):
        encode(None, current, anchor_rows=(4, 2))
    indices = packet.row_anchor.copy()
    indices[0] = len(packet.anchors)
    with pytest.raises(ValueError, match="dictionary geometry"):
        decode(None, replace(packet, row_anchor=indices))
    payload = packet.payload.copy()
    payload[0] |= 192
    with pytest.raises(ValueError, match="combined"):
        decode(None, replace(packet, payload=payload))


@pytest.mark.parametrize("seed", range(8))
def test_mixed_prediction_modes_roundtrip_and_never_inflate_selected_storage(seed):
    rng = np.random.default_rng(seed)
    original = rng.integers(0, 8, (16, 4096), dtype=np.uint8)
    original[4:8] = original[:4]
    previous = original.copy()
    previous[::2, ::31] ^= 3
    current = base.pack_symbols(original.reshape(-1))
    for predecessor in (None, base.pack_symbols(previous.reshape(-1))):
        packet, proof = encode(predecessor, current)
        assert np.array_equal(decode(predecessor, packet), current)
        assert packet.serialized_bytes + 1 <= proof["common_source_code_bytes"]
