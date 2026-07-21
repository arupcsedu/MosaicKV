#!/usr/bin/env python3
"""Generate a leakage-checked PrefixKV profile from captured prompt scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mosaickv.baselines import (
    PrefixKVCalibrationObservation,
    generate_prefixkv_profile,
)


def _ids(path: Path) -> tuple[str, ...]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return tuple(sorted(line.strip() for line in text.splitlines() if line.strip()))
    if not isinstance(payload, list) or not all(isinstance(value, str) for value in payload):
        raise ValueError("evaluation ID file must be a JSON string array or one ID per line")
    return tuple(sorted(payload))


def _observations(path: Path) -> tuple[PrefixKVCalibrationObservation, ...]:
    result: list[PrefixKVCalibrationObservation] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload: Any = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number}: row must be a JSON object")
        result.append(
            PrefixKVCalibrationObservation(
                sample_id=str(payload["sample_id"]),
                layer_scores=tuple(
                    tuple(float(score) for score in layer)
                    for layer in payload["layer_scores"]
                ),
            )
        )
    return tuple(result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention-jsonl", type=Path, required=True)
    parser.add_argument("--evaluation-sample-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--calibration-split", required=True)
    parser.add_argument("--retention-ratio", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-size", type=int, default=1)
    parser.add_argument("--protect-size", type=int, default=1)
    args = parser.parse_args()
    profile = generate_prefixkv_profile(
        _observations(args.attention_jsonl),
        model_id=args.model,
        model_revision=args.model_revision,
        dataset_id=args.dataset,
        dataset_revision=args.dataset_revision,
        calibration_split=args.calibration_split,
        evaluation_sample_ids=_ids(args.evaluation_sample_ids),
        retention_ratio=args.retention_ratio,
        seed=args.seed,
        start_size=args.start_size,
        protect_size=args.protect_size,
    )
    path = profile.write(args.output)
    print(
        json.dumps(
            {
                "status": "completed",
                "profile": str(path.resolve()),
                "profile_sha256": profile.profile_sha256,
                "calibration_samples": len(profile.calibration_sample_ids),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
