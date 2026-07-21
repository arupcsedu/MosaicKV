#!/usr/bin/env python3
"""Expand one versioned experiment matrix into immutable Slurm task configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mosaickv.experiment_matrix import (
    load_experiment_matrix,
    materialize_experiment_matrix,
    verify_expanded_index,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="verify and reuse an existing byte-identical immutable expansion",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    matrix = load_experiment_matrix(args.matrix)
    index_path = materialize_experiment_matrix(
        args.matrix,
        output_directory=args.output_directory,
        resume=bool(args.resume),
    )
    payload = {
        "array_max_index": max(-1, verify_expanded_index(index_path) - 1),
        "enabled": matrix.enabled,
        "experiment_id": matrix.experiment_id,
        "index_path": str(index_path),
        "job_count": verify_expanded_index(index_path),
        "matrix_sha": matrix.sha256,
        "status": "expanded" if matrix.enabled else "blocked_no_runnable_jobs",
    }
    print(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":") if args.json else None,
            indent=None if args.json else 2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
