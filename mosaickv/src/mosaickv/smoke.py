"""Synthetic CPU-only installation and tensor-path smoke test."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import cast

import numpy as np
import numpy.typing as npt

from mosaickv.config import ConfigurationError
from mosaickv.types import JsonObject


@dataclass(frozen=True, slots=True)
class SmokeResult:
    """Validation-only output; never an experimental measurement row."""

    status: str
    validation_only: bool
    synthetic: bool
    seed: int
    shape: tuple[int, ...]
    dtype: str
    retention_ratio: float
    exact_equivalence: bool
    max_abs_error: float
    tensor_sha256: str

    def to_json_object(self) -> JsonObject:
        return cast("JsonObject", asdict(self))


def _validate_dimension(name: str, value: int) -> None:
    if value < 1:
        raise ConfigurationError(f"smoke.{name} must be >= 1")


def run_cpu_smoke(
    *,
    seed: int = 0,
    layers: int = 2,
    sequence_length: int = 32,
    kv_heads: int = 2,
    head_dim: int = 8,
    retention_ratio: float = 1.0,
) -> SmokeResult:
    """Create synthetic K/V tensors and verify the lossless retention-1.0 path."""

    if seed < 0:
        raise ConfigurationError("smoke.seed must be >= 0")
    for name, value in (
        ("layers", layers),
        ("sequence_length", sequence_length),
        ("kv_heads", kv_heads),
        ("head_dim", head_dim),
    ):
        _validate_dimension(name, value)
    if retention_ratio != 1.0:
        raise ConfigurationError(
            "smoke.retention_ratio must equal 1.0; the scaffold has no compression implementation"
        )

    rng = np.random.default_rng(seed)
    shape = (layers, 2, sequence_length, kv_heads, head_dim)
    full_cache: npt.NDArray[np.float32] = rng.standard_normal(shape, dtype=np.float32)
    candidate_cache = full_cache.copy()
    difference = np.abs(candidate_cache - full_cache)
    max_abs_error = float(difference.max(initial=0.0))
    exact = bool(np.array_equal(candidate_cache, full_cache))
    digest = hashlib.sha256(candidate_cache.tobytes(order="C")).hexdigest()
    return SmokeResult(
        status="passed" if exact else "failed",
        validation_only=True,
        synthetic=True,
        seed=seed,
        shape=shape,
        dtype=str(candidate_cache.dtype),
        retention_ratio=retention_ratio,
        exact_equivalence=exact,
        max_abs_error=max_abs_error,
        tensor_sha256=digest,
    )


__all__ = ["SmokeResult", "run_cpu_smoke"]
