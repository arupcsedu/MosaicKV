from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.shared_source_format import SharedSourceFormat
from mosaickv.evaluation.source_row_geometry import GEOMETRIES, apply, geometry


def packet():
    return SharedSourceFormat(
        SimpleNamespace(minimum=torch.ones(1, dtype=torch.bfloat16), scale=torch.ones(1)),
        (1, 32, 600, 128),
        None,
        (),
        None,
        None,
        None,
        None,
        source_head_group=4,
    )


@pytest.mark.parametrize("rows,heads", GEOMETRIES)
def test_geometry_preserves_all_storage_position_and_correction_objects(rows, heads):
    old = packet()
    new = geometry(old, rows, heads)
    for key in (
        "source",
        "row_positions",
        "exact",
        "cos",
        "sin",
        "union",
        "slots",
        "transport_patches",
        "input_overlay",
        "channel_exceptions",
    ):
        assert getattr(old, key) is getattr(new, key)
    assert new.shape == old.shape
    assert new.source_row_tile == rows and new.source_head_group == heads
    assert old.source_row_tile == 16 and old.source_head_group == 4


def test_unsupported_geometry_and_backend_fail_explicitly():
    p = packet()
    for args in ((True, 4), (17, 4), (32, 8), (32, True)):
        with pytest.raises(ValueError):
            geometry(p, *args)
    with pytest.raises(NotImplementedError):
        geometry(replace(p, projection_backend="native_bmm_256"), 32, 4)
    with pytest.raises(ValueError):
        geometry(replace(p, shape=(1, 3, 600, 128)), 32, 4)


def test_no_reprojection_for_unrelated_baseline_formats():
    baseline = object()
    assert apply((baseline,), (32, 4))[0] is baseline
    with pytest.raises(ValueError):
        apply((baseline,), (32, 8))
