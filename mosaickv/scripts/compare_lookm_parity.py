#!/usr/bin/env python3
"""Validate and compare official LOOK-M and ``lookm_reimpl`` artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from mosaickv.baselines import (
    LookMParityArtifact,
    LookMParityError,
    build_lookm_parity_report,
)
from mosaickv.types import JsonObject


def _load(path: Path) -> LookMParityArtifact:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LookMParityError(f"cannot read {path}: {error}") from error
    if not isinstance(payload, dict):
        raise LookMParityError(f"{path} must contain one JSON object")
    return LookMParityArtifact.from_json_object(cast("JsonObject", payload))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare selected positions, active KV bytes, generated tokens, task score, "
            "and latency only after validating every controlled input."
        )
    )
    parser.add_argument("--official", required=True, type=Path)
    parser.add_argument("--reimpl", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> int:
    """Write a machine-readable parity report; return 2 when not comparable."""

    args = _parser().parse_args()
    try:
        official = _load(args.official)
        reimplementation = _load(args.reimpl)
        report = build_lookm_parity_report(official, reimplementation)
    except LookMParityError as error:
        raise SystemExit(f"LOOK-M parity input error: {error}") from error
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "comparable" else 2


if __name__ == "__main__":
    raise SystemExit(main())
