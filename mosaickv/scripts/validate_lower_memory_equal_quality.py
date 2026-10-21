#!/usr/bin/env python3
"""Read-only validation of the lower-memory/equal-quality preregistration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from isoquality_gate import validate_isoquality_protocol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("mosaickv/configs/experiments/lower_memory_equal_quality_v1.yaml"),
    )
    args = parser.parse_args()
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    result = validate_isoquality_protocol(protocol)
    result.update(
        protocol=str(args.protocol.resolve()),
        execution_authorized=bool(protocol["execution_authorized"]),
        next_required_action=(
            "seal_clean_source_and_immutable_ledgers"
            if not protocol["execution_authorized"]
            else "materialize_write_once_plan"
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
