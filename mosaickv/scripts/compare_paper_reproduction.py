#!/usr/bin/env python3
"""Create an explicit paper-reported versus locally-reproduced artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from mosaickv.evaluation.result_categories import (
    LocallyReproducedResult,
    PaperReportedResult,
    ReproductionComparison,
    write_reproduction_comparisons,
)


def _object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected one JSON object in {path}")
    return cast("dict[str, Any]", payload)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a transcribed paper value with a locally reproduced value without "
            "putting the published number in a measured-results table."
        )
    )
    parser.add_argument("--paper-reported", type=Path, required=True)
    parser.add_argument("--locally-reproduced", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paper = PaperReportedResult.from_json_object(_object(args.paper_reported))
    local = LocallyReproducedResult.from_json_object(_object(args.locally_reproduced))
    comparison = ReproductionComparison.build(paper, local)
    output = write_reproduction_comparisons((comparison,), args.output)
    print(json.dumps({"output": str(output.resolve()), **comparison.to_json_object()}))
    return 0 if comparison.within_preregistered_tolerance else 2


if __name__ == "__main__":
    raise SystemExit(main())
