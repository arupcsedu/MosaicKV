"""Shared invariants for experimental native whole-block cache compaction."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class NativeBlockSelection:
    """Deterministic request-wide token-block selection.

    Native serving engines expose one physical token-block table shared by all
    decoder layers and KV heads.  This object therefore represents the coarse
    Stage-B transport boundary, not MosaicKV's per-layer/head selector.
    """

    logical_positions: int
    block_size: int
    retention_ratio: float
    selected_blocks: tuple[int, ...]
    mandatory_blocks: tuple[int, ...]
    policy: str = "mandatory_even_coverage_v1"

    @property
    def source_blocks(self) -> int:
        return math.ceil(self.logical_positions / self.block_size)

    @property
    def selected_positions(self) -> int:
        return selected_position_count(
            self.logical_positions,
            self.block_size,
            self.selected_blocks,
        )

    def to_json_object(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["selected_positions"] = self.selected_positions
        payload["source_blocks"] = self.source_blocks
        return payload


def _validate_inputs(logical_positions: int, block_size: int, retention_ratio: float) -> None:
    if logical_positions < 1:
        raise ValueError("logical_positions must be >= 1")
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    if not math.isfinite(retention_ratio) or not 0 < retention_ratio <= 1:
        raise ValueError("retention_ratio must be finite and in (0, 1]")


def select_native_blocks(
    logical_positions: int,
    block_size: int,
    retention_ratio: float,
) -> NativeBlockSelection:
    """Select mandatory boundary blocks plus deterministic even coverage.

    This fallback is used only to validate an engine's native mutation
    transaction.  It must not be reported as the full MosaicKV algorithm.
    Production MosaicKV rows must supply the common selector's request-wide
    block plan through a later audited selector bridge.
    """

    _validate_inputs(logical_positions, block_size, retention_ratio)
    source_blocks = math.ceil(logical_positions / block_size)
    mandatory = tuple(sorted({0, source_blocks - 1}))
    target = source_blocks if retention_ratio == 1 else math.ceil(source_blocks * retention_ratio)
    target = min(source_blocks, max(target, len(mandatory)))
    selected = set(mandatory)
    candidates = [index for index in range(source_blocks) if index not in selected]
    remaining = target - len(selected)
    if remaining >= len(candidates):
        selected.update(candidates)
    elif remaining > 0:
        # Choose quantiles of the remaining ordered blocks.  Integer midpoint
        # arithmetic gives deterministic tie-breaking without floating error.
        for rank in range(remaining):
            candidate_offset = ((2 * rank + 1) * len(candidates)) // (2 * remaining)
            candidate_offset = min(candidate_offset, len(candidates) - 1)
            selected.add(candidates[candidate_offset])
        # The quantile formula is injective for remaining <= candidates, but
        # retain a defensive deterministic fill for future policy changes.
        for candidate in candidates:
            if len(selected) == target:
                break
            selected.add(candidate)
    result = NativeBlockSelection(
        logical_positions=logical_positions,
        block_size=block_size,
        retention_ratio=retention_ratio,
        selected_blocks=tuple(sorted(selected)),
        mandatory_blocks=mandatory,
    )
    if len(result.selected_blocks) != target:
        raise AssertionError("native selector did not meet its block cardinality")
    return result


def validated_native_block_plan(
    logical_positions: int,
    block_size: int,
    retention_ratio: float,
    selected_blocks: tuple[int, ...],
    *,
    policy: str,
    mandatory_blocks: tuple[int, ...] | None = None,
    expected_selected_blocks: int | None = None,
) -> NativeBlockSelection:
    """Validate an externally planned selection against the native boundary."""

    _validate_inputs(logical_positions, block_size, retention_ratio)
    source_blocks = math.ceil(logical_positions / block_size)
    boundaries = {0, source_blocks - 1}
    mandatory = (
        tuple(sorted(boundaries))
        if mandatory_blocks is None
        else tuple(sorted(set(mandatory_blocks)))
    )
    if mandatory_blocks is not None and mandatory != mandatory_blocks:
        raise ValueError("external mandatory blocks must be strictly increasing and unique")
    if not mandatory or mandatory[0] < 0 or mandatory[-1] >= source_blocks:
        raise ValueError("external mandatory block is outside the logical prompt")
    if not boundaries.issubset(mandatory):
        raise ValueError("external mandatory blocks omit a request boundary block")
    expected = (
        source_blocks
        if retention_ratio == 1
        else math.ceil(source_blocks * retention_ratio)
    )
    if expected_selected_blocks is not None:
        if expected_selected_blocks < 1 or expected_selected_blocks > source_blocks:
            raise ValueError("external selected-block target is outside the source block table")
        expected = expected_selected_blocks
    expected = min(source_blocks, max(expected, len(mandatory)))
    normalized = tuple(sorted(set(selected_blocks)))
    if normalized != selected_blocks:
        raise ValueError("external selected blocks must be strictly increasing and unique")
    if len(normalized) != expected:
        raise ValueError(
            f"external selected-block count {len(normalized)} does not match target {expected}"
        )
    if not normalized or normalized[0] < 0 or normalized[-1] >= source_blocks:
        raise ValueError("external selected block is outside the logical prompt")
    if not set(mandatory).issubset(normalized):
        raise ValueError("external selected blocks omit a mandatory block")
    if not policy.strip():
        raise ValueError("external selection policy must be non-empty")
    return NativeBlockSelection(
        logical_positions=logical_positions,
        block_size=block_size,
        retention_ratio=retention_ratio,
        selected_blocks=normalized,
        mandatory_blocks=mandatory,
        policy=policy,
    )


def selected_position_count(
    logical_positions: int,
    block_size: int,
    selected_blocks: tuple[int, ...],
) -> int:
    """Return the exact number of source positions owned by selected blocks."""

    if logical_positions < 0 or block_size < 1:
        raise ValueError("logical_positions must be >= 0 and block_size must be >= 1")
    if logical_positions == 0:
        if selected_blocks:
            raise ValueError("an empty sequence cannot select blocks")
        return 0
    source_blocks = math.ceil(logical_positions / block_size)
    normalized = tuple(sorted(set(selected_blocks)))
    if normalized != selected_blocks:
        raise ValueError("selected_blocks must be strictly increasing and unique")
    if not normalized or normalized[0] < 0 or normalized[-1] >= source_blocks:
        raise ValueError("selected block index is outside the logical sequence")
    last_block = source_blocks - 1
    tail = logical_positions - last_block * block_size
    return sum(tail if index == last_block else block_size for index in normalized)


def native_audit_path(directory: str | Path, backend: str, request_id: str) -> Path:
    """Map an arbitrary request ID to a safe scratch-rooted audit filename."""

    digest = hashlib.sha256(request_id.encode("utf-8")).hexdigest()
    return Path(directory) / backend / f"{digest}.json"


def require_scratch_path(directory: str | Path) -> Path:
    """Validate a cache/artifact path against the allocation's real path.

    The cluster exposes ``/scratch`` through a canonical ``/sfs/weka/scratch``
    mount, so lexical prefix checks reject valid paths after ``Path.resolve``.
    Comparing resolved paths keeps both spellings safe while still rejecting
    home-directory storage.
    """

    destination = Path(directory).expanduser().resolve()
    scratch_root = Path("/scratch/djy8hg").resolve()
    if destination != scratch_root and scratch_root not in destination.parents:
        raise ValueError(f"native backend path must be under /scratch/djy8hg: {destination}")
    return destination


def write_native_audit(
    directory: str | Path,
    backend: str,
    request_id: str,
    payload: dict[str, Any],
) -> Path:
    """Atomically write one scheduler-owned native compaction audit."""

    destination = native_audit_path(require_scratch_path(directory), backend, request_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    record = {"backend": backend, "request_id": request_id, **payload}
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(record, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


__all__ = [
    "NativeBlockSelection",
    "native_audit_path",
    "require_scratch_path",
    "select_native_blocks",
    "selected_position_count",
    "validated_native_block_plan",
    "write_native_audit",
]
