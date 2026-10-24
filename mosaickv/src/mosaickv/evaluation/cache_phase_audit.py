"""Physical-storage validation applies separately to generation and PPL."""

from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def audit_phase_trace(trace: dict[str, Any], cap_bytes: int | None) -> dict[str, Any]:
    if trace.get("status") != "completed":
        raise ValueError("incomplete phase trace cannot satisfy storage/metric gates")
    teacher = trace.get("teacher_forced_evaluation", {})
    phase = "teacher_forced" if teacher.get("enabled") else "generation"
    life = trace["cache_lifecycle"]
    sizes = {b: life[f"{b}_active_kv_storage_bytes"] for b in ("initial", "final", "peak")}
    if any(type(value) is not int or value < 1 for value in sizes.values()):
        raise ValueError("phase storage requires positive measured backing bytes")
    if cap_bytes is not None and any(value != cap_bytes for value in sizes.values()):
        raise ValueError(f"{phase} storage exceeds or underfills the exact physical byte cap")
    if sizes["peak"] < max(sizes["initial"], sizes["final"]):
        raise ValueError("phase peak storage is smaller than an observed boundary")
    output: dict[str, Any] = dict(
        sample_id=trace["sample_id"],
        phase=phase,
        storage_bytes=sizes,
        effective_method=trace["effective_method"],
    )
    if phase == "teacher_forced":
        nll, tokens = teacher["negative_log_likelihood_sum"], teacher["evaluated_tokens"]
        if (
            type(tokens) is not int
            or tokens < 1
            or isinstance(nll, bool)
            or not isinstance(nll, (int, float))
            or not math.isfinite(nll)
            or nll < 0
        ):
            raise ValueError("teacher-forced phase lacks complete NLL sufficient statistics")
        if not teacher.get("target_was_not_truncated"):
            raise ValueError("truncated PPL target is not admissible")
        if "requested_output_tokens" in trace and trace["requested_output_tokens"] != tokens:
            raise ValueError("teacher-forced target token count changed")
        output.update(nll_sum=nll, tokens=tokens)
    return output


def audit_phase_directory(
    root: str | Path,
    sample_ids: Iterable[str],
    cap_bytes: int | None,
) -> tuple[dict[str, Any], ...]:
    import json

    records = {}
    for path in Path(root).rglob("*.json"):
        result = audit_phase_trace(json.loads(path.read_text()), cap_bytes)
        key = result["sample_id"], result["phase"]
        if key in records:
            raise ValueError("duplicate sample/phase traces")
        records[key] = result
    expected = {
        (sample, phase) for sample in sample_ids for phase in ("generation", "teacher_forced")
    }
    if set(records) != expected:
        raise ValueError("missing phase storage trace")
    return tuple(records[key] for key in sorted(records))
