"""Immutable JSONL/JSON storage for raw trials and derived aggregates."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

from mosaickv.measurements.types import FullKVAggregate, FullKVTrialMeasurement
from mosaickv.types import JsonObject


def _write_text_atomically(destination: Path, payload: str, description: str) -> Path:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {description}: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise
    return destination


def write_trial_jsonl(trials: Sequence[FullKVTrialMeasurement], output: str | Path) -> Path:
    """Write every trial atomically and refuse to overwrite raw evidence."""

    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite FullKV trial output: {destination}")
    keys = [(trial.run_id, trial.sample_id, trial.trial_index) for trial in trials]
    if len(set(keys)) != len(keys):
        raise ValueError("FullKV trials contain duplicate run/sample/trial keys")
    payload = "".join(
        json.dumps(
            trial.to_json_object(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
        for trial in trials
    )
    return _write_text_atomically(destination, payload, "FullKV trial output")


def write_aggregate_json(aggregate: FullKVAggregate, output: str | Path) -> Path:
    """Write a derived aggregate atomically without replacing prior output."""

    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite FullKV aggregate: {destination}")
    payload = json.dumps(
        aggregate.to_json_object(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return _write_text_atomically(destination, payload + "\n", "FullKV aggregate")


def write_json_object(payload: JsonObject, output: str | Path) -> Path:
    """Write one structured JSON record atomically without overwriting evidence."""

    destination = Path(output)
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return _write_text_atomically(destination, serialized + "\n", "structured JSON output")


__all__ = ["write_aggregate_json", "write_json_object", "write_trial_jsonl"]
