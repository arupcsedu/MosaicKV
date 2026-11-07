import dataclasses

import numpy as np
import pytest

from mosaickv.evaluation.all_head_source_codec import (
    SourceCoverage,
    SymbolCodes,
    decode,
    encode,
    pack_symbols,
    symbols,
)


def test_one_head_or_one_projection_cannot_authorize_source_retirement():
    complete = SourceCoverage(31, tuple(range(32)), ("key", "value"), tuple(range(8)), ((0, 4096),))
    assert complete.verify(kv_heads=32, head_dim=128, length=8)["projection_consumer_count"] == 64
    for bad in (
        dataclasses.replace(complete, heads=(24,)),
        dataclasses.replace(complete, components=("key",)),
        dataclasses.replace(complete, logical_positions=tuple(range(7))),
        dataclasses.replace(complete, feature_intervals=((0, 2048),)),
        dataclasses.replace(complete, feature_intervals=((0, 2048), (2047, 4096))),
    ):
        with pytest.raises(ValueError):
            bad.verify(kv_heads=32, head_dim=128, length=8)


@pytest.mark.parametrize("seed", range(8))
def test_lossless_palette_and_escape_residuals_with_and_without_predecessor(seed):
    rng = np.random.default_rng(seed)
    values = rng.choice(8, 512 * 8, p=[0.57, 0.27, 0.04, 0.03, 0.03, 0.02, 0.02, 0.02]).astype(
        np.uint8
    )
    current = pack_symbols(values)
    previous = pack_symbols(rng.integers(0, 8, len(values), dtype=np.uint8))
    original = current.copy()
    for parent in (None, previous, current.copy()):
        packet = encode(parent, current)
        assert np.array_equal(decode(parent, packet), current)
        assert np.array_equal(symbols(current), values)
        assert np.array_equal(current, original)
        if not packet.raw:
            assert packet.serialized_bytes < len(current)
            assert not np.shares_memory(packet.payload, current)
            assert not np.shares_memory(packet.payload, previous)


def test_random_raw_fallback_and_canonical_errors_are_not_silently_accepted():
    rng = np.random.default_rng(82)
    current = pack_symbols(rng.integers(0, 8, 1024, dtype=np.uint8))
    raw = encode(None, current)
    assert raw.raw
    assert np.array_equal(decode(None, raw), current)
    with pytest.raises(ValueError):
        encode(None, np.zeros(128, dtype=np.uint8))
    constant = pack_symbols(np.ones(1024, dtype=np.uint8) * 4)
    packet = encode(None, constant)
    assert not packet.raw
    damaged = packet.payload.copy()
    damaged[0] |= 128
    with pytest.raises(ValueError, match="header"):
        decode(None, dataclasses.replace(packet, payload=damaged))
    with pytest.raises(ValueError):
        decode(None, SymbolCodes(packet.length, packet.offsets[:-1], packet.payload, False))


def test_delta_reference_is_required_and_no_original_owner_is_stored():
    rng = np.random.default_rng(88)
    previous = pack_symbols(rng.integers(0, 8, 4096, dtype=np.uint8))
    packet = encode(previous, previous.copy())
    assert not packet.raw
    with pytest.raises(ValueError, match="predecessor"):
        decode(None, packet)
    assert set(dataclasses.asdict(packet)) == {"length", "offsets", "payload", "raw"}


def test_quality_improvement_policy_cannot_lock_answers_to_storage_parent():
    from mosaickv.evaluation.storage_six_overlay import OverlayPolicy

    with pytest.raises(ValueError, match="greedy-token lock"):
        OverlayPolicy(quality_objective="fullkv_draft_improvement", live_anchor_guard=True)
    with pytest.raises(ValueError, match="lossy head overlays"):
        OverlayPolicy(quality_objective="bitwise_storage_only")
    assert OverlayPolicy(quality_objective="bitwise_storage_only", enabled=False)
    assert OverlayPolicy(
        quality_objective="fullkv_draft_improvement",
        readout_target="fullkv_draft",
        readout_selection="joint",
    )
