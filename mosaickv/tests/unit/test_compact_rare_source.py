"""Integer codec fixtures; no synthetic number is a measured model result."""

from dataclasses import replace

import numpy as np
import pytest

from mosaickv.evaluation import all_head_source_codec as base
from mosaickv.evaluation import compact_rare_source as compact
from mosaickv.evaluation import predictive_source as old


@pytest.mark.parametrize("bits", (2, 3))
@pytest.mark.parametrize("count", (0, 1, 2, 3, 4, 7, 8, 9, 15, 31, 32, 33, 65, 511))
def test_literal_bit_boundaries_and_zero_padding(bits, count):
    x = np.random.default_rng(bits * 1000 + count).integers(0, 1 << bits, count, dtype=np.uint8)
    payload = compact.pack_small(x, bits)
    assert np.array_equal(x, compact.unpack_small(payload, count, bits))
    if count * bits % 8:
        assert int(payload[-1]) >> (count * bits % 8) == 0


@pytest.mark.parametrize("seed", range(10))
def test_native_geometry_roundtrip_owner_replacement_and_nonincreasing_cost(seed):
    rng = np.random.default_rng(seed)
    previous = base.pack_symbols(rng.integers(0, 8, 512 * 8, dtype=np.uint8))
    # Concentrated residuals with every support size, plus full-entropy groups.
    values = np.concatenate([rng.choice(np.arange(2 + (g % 7)), 512, p=None) for g in range(8)])
    current = base.pack_symbols((base.symbols(previous) + values) & 7)
    packet = compact.encode(previous, current)
    original = old.encode(previous, current, "delta_add")
    assert np.array_equal(compact.decode(previous, packet), current)
    assert packet.serialized_bytes <= min(original.serialized_bytes, current.nbytes)
    assert not np.shares_memory(packet.payload, current)
    assert not np.shares_memory(packet.payload, previous)


def test_real_two_bit_groups_and_raw_groups_coexist_without_changing_any_source_bit():
    rng = np.random.default_rng(99)
    previous = base.pack_symbols(rng.integers(0, 8, 4096, dtype=np.uint8))
    residual = np.concatenate(
        [
            rng.choice([0, 1, 2, 3, 4, 5], 512, p=[0.5, 0.25, 0.1, 0.06, 0.05, 0.04])
            if i % 2 == 0
            else rng.integers(0, 8, 512)
            for i in range(8)
        ]
    )
    current = base.pack_symbols((base.symbols(previous) + residual) & 7)
    packet = compact.encode(previous, current)
    assert not packet.raw
    assert (packet.offsets[:-1] & base.RAW).any()
    starts = packet.offsets[:-1] & base.OFFSET
    assert any(
        not (o & base.RAW) and packet.payload[s] & 128
        for o, s in zip(packet.offsets[:-1], starts, strict=True)
    )
    assert np.array_equal(compact.decode(previous, packet), current)
    assert packet.serialized_bytes < old.encode(previous, current, "delta_add").serialized_bytes


def test_first_layer_and_incompressible_fallback_keep_original_not_residual_bytes():
    rng = np.random.default_rng(2)
    previous, current = (
        base.pack_symbols(rng.integers(0, 8, 4096, dtype=np.uint8)) for _ in range(2)
    )
    for p in (None, previous):
        packet = compact.encode(p, current)
        assert packet.raw and not packet.offsets.size
        assert np.array_equal(packet.payload, current)
        assert np.array_equal(compact.decode(p, packet), current)


def test_truncation_predecessor_and_unknown_scope_fail_closed():
    previous = np.zeros(384, dtype=np.uint8)
    current = base.pack_symbols(
        np.resize(np.array([0] * 30 + [1] * 10 + [2, 3, 4, 5] * 4, dtype=np.uint8), 1024)
    )
    packet = compact.encode(previous, current)
    assert not packet.raw
    for damaged in (
        replace(packet, payload=packet.payload[:-1]),
        replace(packet, offsets=packet.offsets[:-1]),
    ):
        with pytest.raises(ValueError):
            compact.decode(previous, damaged)
    with pytest.raises(ValueError):
        compact.decode(None, packet)


def test_compact_native_policy_cannot_be_activated_with_an_unaudited_decoder():
    from mosaickv.evaluation.predictive_source_gpu import use

    with (
        pytest.raises(ValueError),
        use(False, encoding_policy="direct_additive_compact_rare", decoder="reference"),
    ):
        pass
