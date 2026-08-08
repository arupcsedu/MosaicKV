#!/usr/bin/env python3
"""Serially materialize one Choice-C design subset before parallel GPU workers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import yaml

from mosaickv.evaluation.lmms_adapter import prepare_seeded_lmms_tasks
from mosaickv.evaluation.tasks import default_task_registry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument(
        "--ledger-key",
        choices=("calibration_ledger", "calibration_validation_ledger", "evaluation_ledger"),
        default="calibration_ledger",
    )
    args = parser.parse_args()
    spec = cast("dict[str, Any]", yaml.safe_load(args.spec.resolve().read_text()))
    ledger_path = (args.spec.resolve().parents[2] / str(spec[args.ledger_key])).resolve()
    ledger = cast("dict[str, Any]", json.loads(ledger_path.read_text()))
    sample_ids = tuple(cast("list[str]", ledger["sample_ids"]))
    task = default_task_registry().resolve(str(spec["task"]))
    _tasks, prepared, selected, _manager = prepare_seeded_lmms_tasks(
        (task,),
        seed=int(spec["seed"]),
        subset_size=None,
        model_name=str(ledger["materialization_model_name"]),
        explicit_sample_ids=sample_ids,
        dataset_revisions={str(ledger["dataset_id"]): str(ledger["dataset_revision"])},
    )
    if selected != sample_ids or len(prepared) != len(sample_ids):
        raise RuntimeError("prewarmed subset differs from the immutable design ledger")
    print(f"prewarmed {len(selected)} samples from {ledger_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
