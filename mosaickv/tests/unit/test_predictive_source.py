import gc
import weakref

import numpy as np
import pytest

from mosaickv.evaluation.all_head_source_codec import SourceCoverage, pack_symbols
from mosaickv.evaluation.predictive_source import (
    LOCAL_MODES,
    MODES,
    decode,
    encode,
    screen,
    transform,
)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("seed", range(4))
def test_predictors_preserve_all_bits_with_and_without_predecessor(mode, seed):
    rng = np.random.default_rng(seed)
    current = pack_symbols(rng.integers(0, 8, 512 * 3, dtype=np.uint8))
    before = current.copy()
    if mode in LOCAL_MODES:
        assert np.array_equal(transform(transform(current, mode), mode, inverse=True), current)
    for previous in (None, current.copy(), current ^ np.uint8(13)):
        packet = encode(previous, current, mode)
        assert np.array_equal(decode(previous, packet), current)
        assert packet.serialized_bytes == packet.body.offsets.nbytes + packet.body.payload.nbytes
    assert np.array_equal(current, before)


@pytest.mark.parametrize("mode", LOCAL_MODES)
def test_block_locality_permits_in_place_native_replay_without_halo(mode):
    rng = np.random.default_rng(31)
    a = pack_symbols(rng.integers(0, 8, 512 * 4, dtype=np.uint8))
    # Each native block can reconstruct independently; no neighboring old
    # source block can be read after another program overwrites it.
    transformed = transform(a, mode)
    for first in range(0, len(a), 192):
        assert np.array_equal(
            transform(transformed[first : first + 192].copy(), mode, inverse=True),
            a[first : first + 192],
        )


def test_complete_consumer_coverage_and_source_owner_retirement():
    values = np.repeat(np.arange(8, dtype=np.uint8), 256)
    original = pack_symbols(values)
    ref = weakref.ref(original)
    packet = encode(None, original, "xor1")
    expected = original.copy()
    proof = SourceCoverage(0, tuple(range(4)), ("key", "value"), tuple(range(32)), ((0, 64),))
    assert proof.verify(kv_heads=4, head_dim=16, length=32)["projection_consumer_count"] == 8
    del original
    gc.collect()
    assert ref() is None
    assert np.array_equal(decode(None, packet), expected)


def test_deterministic_selection_uses_actual_complete_storage_not_estimates():
    current = pack_symbols(np.repeat(np.arange(8, dtype=np.uint8), 256))
    first, evidence = screen(None, current)
    second, again = screen(None, current)
    assert evidence == again
    assert evidence["selected_bytes"] <= evidence["common_bytes"]
    assert evidence["selected_bytes"] == first.serialized_bytes + 1
    assert np.array_equal(decode(None, second), current)
    assert not evidence["source_owner_retained"]
    assert evidence["extra_decoded_source_bytes"] == 0


def test_unknown_predictor_and_partial_geometry_fail_closed():
    codes = np.zeros(192, dtype=np.uint8)
    with pytest.raises(ValueError, match="registered"):
        transform(codes, "dictionary")
    with pytest.raises(ValueError, match="512-symbol"):
        transform(codes[:-1], "xor1")


def test_additive_dependency_is_explicit_and_cannot_be_silently_omitted():
    current = pack_symbols(np.tile(np.arange(8, dtype=np.uint8), 128))
    packet = encode(current, current, "delta_add")
    with pytest.raises(ValueError, match="predecessor"):
        decode(None, packet)


def test_additive_gpu_scope_restores_common_runtime_without_loading_weights():
    from mosaickv.evaluation import layer_code_delta as runtime
    from mosaickv.evaluation.predictive_source_gpu import use

    before = runtime.encode, runtime.decode, runtime.convert
    with use(False) as state:
        assert state.report()["additional_persistent_replay_bytes"] == 0
    assert (runtime.encode, runtime.decode, runtime.convert) == before
    with pytest.raises(ValueError, match="explicit"), use("yes"):
        pass
