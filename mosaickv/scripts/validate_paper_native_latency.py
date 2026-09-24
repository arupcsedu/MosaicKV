#!/usr/bin/env python3
"""Validate the specialized paper-native persistent-latency protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mosaickv.evaluation.paper_native_latency import load_paper_native_latency_protocol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    path = args.protocol.resolve(strict=True)
    protocol = load_paper_native_latency_protocol(path)
    payload = {
        "status": "valid",
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "experiment_id": protocol.experiment_id,
        "lane": protocol.timing.lane,
        "methods": list(protocol.timing.methods),
        "sample_count": protocol.timing.sample_count,
        "warmups_per_method": protocol.timing.warmups_per_method,
        "timed_repetitions_per_method": protocol.timing.timed_repetitions_per_method,
        "maximum_latency_ratio": protocol.timing.maximum_latency_ratio,
        "expected_mosaickv_mean_active_kv_bytes": (protocol.expected_mosaickv_mean_active_kv_bytes),
    }
    print(json.dumps(payload, sort_keys=True) if args.json else json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
