#!/usr/bin/env python3
"""Compare controlled official PrefixKV and prefixkv_reimpl artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mosaickv.baselines import compare_prefixkv_artifacts, load_prefixkv_parity_artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", required=True)
    parser.add_argument("--reimplementation", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare_prefixkv_artifacts(
        load_prefixkv_parity_artifact(args.official),
        load_prefixkv_parity_artifact(args.reimplementation),
    )
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if report["status"] == "comparable" else 2


if __name__ == "__main__":
    raise SystemExit(main())
