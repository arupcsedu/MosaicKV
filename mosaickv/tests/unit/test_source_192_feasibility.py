import pytest

from mosaickv.evaluation.source_192_feasibility import CAPACITY, WIDTH, Plans192, ranges


@pytest.mark.parametrize("n,tail", [(600, 24), (604, 28)])
def test_registered_tiling_covers_every_source_position_once(n, tail):
    chunks = ranges(n)
    assert chunks == ((0, 192), (192, 192), (384, 192), (576, tail))
    assert [i for first, take in chunks for i in range(first, first + take)] == list(range(n))
    assert len(chunks) == 4 and len(ranges(n, 128)) == 5
    assert 2 * (CAPACITY - 128) * WIDTH * 2 == 2**20


@pytest.mark.parametrize("n", [True, 16, 599, 608, 600.0, "600"])
def test_unregistered_shapes_fail_before_creating_gpu_descriptors(n):
    with pytest.raises(ValueError, match="consumed"):
        Plans192(n)
