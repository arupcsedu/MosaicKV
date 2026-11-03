import pytest

from mosaickv.evaluation.projection_opportunity import tile_coverage


def test_only_complete_tiles_may_be_omitted_including_partial_last_tile():
    # Group 0: full first tile and short final tile. Group 1: empty first,
    # fully protected short final tile. No padded rows count as evidence.
    rows = (*range(17), 33)
    result = tile_coverage(rows, heads=2, length=17, row_tile=16, head_group=1)
    assert result["total_programs"] == 4
    assert result["wholly_overwritten_programs"] == 3
    assert result["untouched_programs"] == 1
    assert result["overwritten_rows"] == len(rows)
    assert result["source_rows"] == 34
    assert result["additional_gpu_metadata_bytes"] == 0
    assert not result["measured_latency_savings"]


def test_one_unprotected_head_prevents_skipping_a_group():
    result = tile_coverage(tuple(range(16)), heads=2, length=17, row_tile=16, head_group=2)
    assert result["wholly_overwritten_programs"] == 0
    assert result["partially_overwritten_programs"] == 1
    assert result["untouched_programs"] == 1


def test_empty_full_and_single_missing_row():
    kw = dict(heads=4, length=33, row_tile=16, head_group=4)
    assert tile_coverage((), **kw)["wholly_overwritten_fraction"] == 0
    all_rows = tuple(range(4 * 33))
    full = tile_coverage(all_rows, **kw)
    assert full["wholly_overwritten_fraction"] == 1
    missing = tile_coverage(all_rows[:-1], **kw)
    assert missing["wholly_overwritten_programs"] == 2
    assert missing["partially_overwritten_programs"] == 1
    assert missing["rows_sha256"] != full["rows_sha256"]


@pytest.mark.parametrize("rows", [(1, 0), (0, 0), (-1,), (34,), (True,), [0]])
def test_invalid_rows_fail_closed(rows):
    with pytest.raises(ValueError):
        tile_coverage(rows, heads=2, length=17, row_tile=16, head_group=1)


@pytest.mark.parametrize(
    "name,value", [("heads", True), ("length", 0), ("row_tile", 8), ("head_group", 3)]
)
def test_unsupported_geometry_fails_closed(name, value):
    kw = dict(heads=2, length=17, row_tile=16, head_group=1)
    kw[name] = value
    with pytest.raises(ValueError):
        tile_coverage((), **kw)


def test_packet_audit_distinguishes_existing_indices_from_hypothetical_text_union():
    import torch

    from mosaickv.evaluation.projection_opportunity import packet_coverage
    from mosaickv.evaluation.shared_source_format import SharedSourceFormat
    from mosaickv.evaluation.text_source_storage import TextProjection, TextSourceFormat

    base = SharedSourceFormat(
        None,
        (1, 2, 17, 128),
        torch.tensor([16, 33], dtype=torch.int32),
        (),
        None,
        None,
        None,
        None,
    )
    projection = TextProjection(
        None,
        torch.arange(16),
        None,
        torch.arange(16).expand(2, 16),
        None,
        None,
        None,
        base.shape,
    )
    result = packet_coverage(TextSourceFormat(base, projection))
    assert result["protected_only"]["wholly_overwritten_fraction"] == 0.5
    assert result["protected_plus_text"]["wholly_overwritten_fraction"] == 1.0
    assert result["correction_scalars_per_component"] == []
    assert result["existing_protected_index_bytes"] == 8
