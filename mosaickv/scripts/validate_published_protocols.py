#!/usr/bin/env python3
"""Validate and summarize versioned paper-native evaluation protocols."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mosaickv.evaluation.published_protocols import (
    load_published_protocols,
    metric_readiness_issues,
    protocol_summary,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument(
        "--require-executable",
        action="store_true",
        help="fail unless every method and protocol is authorized and executable",
    )
    parser.add_argument(
        "--require-metrics-ready",
        action="store_true",
        help="fail unless every paper-native metric has its pinned dependency and route",
    )
    args = parser.parse_args()
    suite = load_published_protocols(args.protocol)
    if args.require_metrics_ready:
        metric_blockers = metric_readiness_issues(suite)
        if metric_blockers:
            raise RuntimeError(
                "protocol metric paths are not ready:\n- " + "\n- ".join(metric_blockers)
            )
    if args.require_executable:
        blocked: list[str] = []
        for protocol in suite.protocols:
            if not protocol.execution_authorized:
                blocked.append(f"{protocol.protocol_id}: execution_authorized=false")
            if len(protocol.model_revision) != 40 or any(
                character not in "0123456789abcdef" for character in protocol.model_revision
            ):
                blocked.append(f"{protocol.protocol_id}: model revision is not an exact Git SHA")
            blocked.extend(
                f"{protocol.protocol_id}/{method.name}: {method.status}"
                for method in protocol.methods
                if method.execution_enabled and method.status == "unsupported_unpinned"
            )
        if blocked:
            raise RuntimeError("protocol suite is not executable:\n- " + "\n- ".join(blocked))
    print(json.dumps(protocol_summary(suite), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
