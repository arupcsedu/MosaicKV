import numpy as np
import pytest

from mosaickv.evaluation.packed_code_arithmetic import FULL, add, subtract


def unpack(words):
    return (words[..., None] >> (np.arange(8, dtype=np.uint32) * 3)) & 7


def pack(fields):
    return (fields.astype(np.uint32) << (np.arange(8, dtype=np.uint32) * 3)).sum(
        -1, dtype=np.uint32
    )


def test_all_64_symbol_pairs_in_every_lane_without_carry_or_borrow_leakage():
    a, b = np.meshgrid(np.arange(8, dtype=np.uint32), np.arange(8, dtype=np.uint32))
    left = np.repeat(a.ravel()[:, None], 8, axis=1)
    right = np.repeat(b.ravel()[:, None], 8, axis=1)
    assert np.array_equal(unpack(add(pack(left), pack(right))), (left + right) & 7)
    assert np.array_equal(unpack(subtract(pack(left), pack(right))), (left - right) & 7)


@pytest.mark.parametrize("seed", range(8))
def test_mixed_neighboring_fields_and_extreme_carries_preserve_every_original_bit(seed):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, int(FULL) + 1, 4096, dtype=np.uint32)
    b = rng.integers(0, int(FULL) + 1, 4096, dtype=np.uint32)
    assert np.array_equal(unpack(add(a, b)), (unpack(a) + unpack(b)) & 7)
    assert np.array_equal(unpack(subtract(a, b)), (unpack(a) - unpack(b)) & 7)
    assert np.array_equal(add(subtract(a, b), b), a)


def test_invalid_fields_or_geometry_fail_closed():
    good = np.array([0, int(FULL)], dtype=np.uint32)
    with pytest.raises(ValueError):
        add(good, np.array([0], dtype=np.uint32))
    with pytest.raises(ValueError):
        subtract(good, np.array([0, 1 << 24], dtype=np.uint32))
    with pytest.raises(ValueError):
        add(good, good.astype(np.int64))
