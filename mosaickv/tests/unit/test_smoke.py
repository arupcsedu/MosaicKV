from __future__ import annotations

import pytest

from mosaickv.config import ConfigurationError
from mosaickv.smoke import run_cpu_smoke


def test_cpu_smoke_uses_synthetic_tensors_and_is_exact() -> None:
    result = run_cpu_smoke(seed=11, layers=2, sequence_length=8, kv_heads=2, head_dim=4)
    assert result.validation_only is True
    assert result.synthetic is True
    assert result.exact_equivalence is True
    assert result.max_abs_error == 0.0
    assert result.shape == (2, 2, 8, 2, 4)
    assert len(result.tensor_sha256) == 64


def test_smoke_is_deterministic() -> None:
    first = run_cpu_smoke(seed=5)
    second = run_cpu_smoke(seed=5)
    assert first.tensor_sha256 == second.tensor_sha256


def test_smoke_refuses_to_claim_unimplemented_compression() -> None:
    with pytest.raises(ConfigurationError, match="no compression implementation"):
        run_cpu_smoke(retention_ratio=0.5)
