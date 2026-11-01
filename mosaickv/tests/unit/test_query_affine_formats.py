import pytest
import torch

from mosaickv.evaluation.affine_refit import refit_cuda
from mosaickv.evaluation.precision_tier import pack
from mosaickv.evaluation.source_query_importance import _probabilities


def test_actual_cuda_refit_never_simulates_cpu_support():
    x = torch.ones(1, 1, 16, 128).bfloat16()
    with pytest.raises(ValueError):
        refit_cuda(pack(x, bits=3, axis=2), x)


def test_no_available_causal_key_is_ignored_not_made_visible():
    q = torch.zeros(1, 1, 3, 128).bfloat16()
    key = torch.ones(1, 1, 1, 128).bfloat16()
    p, dist, ignored = _probabilities(q, key, key, torch.tensor([[7]]), torch.tensor([5, 6, 7]))
    assert ignored == 2 and torch.isfinite(p).all() and torch.isfinite(dist).all()
    assert (p[..., :2, :] == 0).all() and (p[..., 2, :] == 1).all()
