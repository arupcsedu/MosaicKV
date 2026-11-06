"""Fail-closed rows for a measured-operating-point study, NOT byte matching."""

import math
from collections.abc import Mapping, Sequence
from typing import Any


def endpoints(row: Mapping[str, Any]) -> tuple[int, int, int | None]:
    storage = row.get("storage", {})
    initial = row.get("initial_encoded_bytes", storage.get("initial_active_kv_storage_bytes"))
    final = row.get("final_encoded_bytes", storage.get("final_active_kv_storage_bytes"))
    peak = row.get("decode_owned_peak_bytes")
    if any(type(v) is not int or v <= 0 for v in (initial, final)):
        raise ValueError("positive measured backing endpoints required")
    if peak is not None and (type(peak) is not int or peak < max(initial, final)):
        raise ValueError("audited peak cannot be below a measured endpoint")
    return initial, final, peak


def validate_row(row: Mapping[str, Any], *, audited: bool) -> None:
    if row.get("status") != "completed" or row.get("used_fullkv_fallback") is not False:
        raise ValueError("complete, non-fallback prediction required")
    if len(row.get("generated_token_ids", ())) != 128:
        raise ValueError("all 128 fixed predictions must be preserved")
    if not isinstance(row.get("answer"), str):
        raise ValueError("caption text required, including explicitly empty captions")
    initial, final, peak = endpoints(row)
    if final < initial or (audited and peak is None):
        raise ValueError("append-only endpoint and actual audited peak required")
    latency = row.get("end_to_end_seconds")
    if type(latency) not in (int, float) or not math.isfinite(latency) or latency <= 0:
        raise ValueError("finite synchronized request timing required")


def complete_grid(
    rows: Sequence[Mapping[str, Any]],
    methods: Sequence[str],
    indices: Sequence[int],
    *,
    trials: int = 1,
) -> None:
    expected = {(m, i, t) for m in methods for i in indices for t in range(trials)}
    keys = [(r["method"], r["sample_index"], r.get("trial", 0)) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("missing, duplicate, or unexpected method/sample/trial identities")
