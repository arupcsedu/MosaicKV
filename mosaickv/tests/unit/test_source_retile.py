from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.shared_source_format import SharedSourceFormat
from mosaickv.evaluation.source_retile import retile


def packet():
    return SharedSourceFormat(
        SimpleNamespace(minimum=torch.ones(1), scale=torch.ones(1)),
        (1, 32, 600, 128),
        None,
        (),
        None,
        None,
        None,
        None,
    )


def test_hardware_tiling_does_not_recode_any_packet_field():
    old = packet()
    new = retile(old, 4)
    assert new.source is old.source and new.exact is old.exact
    assert new.transport_patches is old.transport_patches and new.shape == old.shape
    assert new.source_head_group == 4 and old.source_head_group == 1


def test_invalid_precision_backend_and_head_group_fail_closed():
    old = packet()
    with pytest.raises(ValueError):
        retile(old, True)
    with pytest.raises(ValueError):
        retile(replace(old, shape=(1, 3, 600, 128)), 2)
    with pytest.raises(ValueError):
        retile(
            replace(
                old,
                source=SimpleNamespace(minimum=torch.ones(1).half(), scale=torch.ones(1).half()),
            ),
            2,
        )
    with pytest.raises(NotImplementedError):
        retile(replace(old, projection_backend="native_bmm_256"), 2)
