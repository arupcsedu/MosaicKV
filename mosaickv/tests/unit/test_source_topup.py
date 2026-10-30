"""Label-blind source sharing adds original evidence without donor eviction."""

import pytest
import torch

from mosaickv.evaluation.storage_runs import source_topup


def test_source_sharing_is_deterministic_uniform_and_strict_superset():
    positions = (((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)),)
    attention = (torch.arange(16).float().reshape(1, 4, 1, 4) + 1,)
    result, proof = source_topup(positions, attention, slots=32, delimiter=(0,))
    assert result == (((0, 1, 2, 3),) * 4,)
    assert all(e["added_count"] == 1 and e["removed_count"] == 0 for e in proof["events"])
    assert source_topup(positions, attention, slots=32, delimiter=(0,))[0] == result
    assert proof["no_benchmark_label_access"]


def test_source_sharing_does_not_invent_rows_when_no_free_source_exists():
    positions = (((0, 1), (0, 1)),)
    attention = (torch.ones(1, 2, 1, 2),)
    result, proof = source_topup(positions, attention, slots=2)
    assert result == positions
    assert all(e["added_count"] == 0 for e in proof["events"])
    with pytest.raises(ValueError, match="positive"):
        source_topup(positions, attention, slots=0)
