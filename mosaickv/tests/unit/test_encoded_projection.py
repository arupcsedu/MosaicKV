import pytest

from mosaickv.evaluation.encoded_projection import checkpoint_bytes_bound


@pytest.mark.parametrize("rows", [1, 7, 16, 17, 128, 600])
def test_exact_checkpoint_bound_and_saved_decoded_source_storage(rows):
    assert checkpoint_bytes_bound((1, 1, rows, 4096), False) == rows * 64
    assert checkpoint_bytes_bound((1, 1, rows, 4096), True) == 0
    assert rows * 64 < rows * 4096 * 2


@pytest.mark.parametrize("shape", [(1, 2, 16, 128), (1, 1, 0, 128), (1, 1, 16, 64)])
def test_rejects_other_source_layouts(shape):
    with pytest.raises(ValueError):
        checkpoint_bytes_bound(shape, False)


def test_growing_decode_stride_and_row_counts_are_not_compile_constants():
    from mosaickv.evaluation.encoded_projection_kernel import _project

    params = {p.name: p for p in _project.params}
    for name in ("ROWS", "HEAD_OUT", "MAP_STRIDE"):
        assert params[name].do_not_specialize
        assert not params[name].is_constexpr
