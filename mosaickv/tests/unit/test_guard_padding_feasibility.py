import pytest
import torch

from mosaickv.evaluation.eager_core_operator import reference_scores
from mosaickv.evaluation.guard_padding_feasibility import padded


def test_padding_is_only_masked_future_storage_not_changed_existing_positions():
    q = torch.ones(1, 1, 1, 8)
    k = torch.zeros(1, 1, 3, 8)
    v = torch.arange(24, dtype=torch.float32).reshape(1, 1, 3, 8)
    p, out = padded(q, k, v, None, 5, 8**-0.5)
    reference = reference_scores(q, k, None, 8**-0.5)
    assert p.shape == reference.shape
    assert torch.allclose(p, reference)
    assert torch.allclose(out, torch.matmul(reference, v))
    assert k.shape[-2] == v.shape[-2] == 3


@pytest.mark.parametrize("cap", (2, 0, False))
def test_padding_cannot_drop_original_positions_or_use_an_implicit_cap(cap):
    q = torch.ones(1, 1, 1, 8)
    kv = torch.ones(1, 1, 3, 8)
    with pytest.raises(ValueError, match="feasible fixed capacity"):
        padded(q, kv, kv, None, cap, 8**-0.5)
