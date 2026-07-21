#!/usr/bin/env python3
"""Validate experiment matrices or a materialized Slurm array index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mosaickv.experiment_matrix import (
    expand_experiment_matrix,
    load_experiment_matrix,
    verify_expanded_index,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path, nargs="*")
    parser.add_argument("--expanded-index", type=Path)
    parser.add_argument("--array-index", type=int)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if not args.matrix and args.expanded_index is None:
        parser.error("provide at least one matrix or --expanded-index")
    if args.array_index is not None and args.expanded_index is None:
        parser.error("--array-index requires --expanded-index")
    matrices = []
    for path in args.matrix:
        matrix = load_experiment_matrix(path)
        runs = expand_experiment_matrix(matrix)
        matrices.append(
            {
                "blocked_scopes": len(matrix.blocked),
                "enabled": matrix.enabled,
                "experiment_id": matrix.experiment_id,
                "matrix_revision": matrix.matrix_revision,
                "matrix_sha": matrix.sha256,
                "path": str(path.resolve()),
                "runnable_jobs": len(runs),
                "status": "valid",
            }
        )
    payload: dict[str, object] = {"matrices": matrices, "status": "valid"}
    if args.expanded_index is not None:
        count = verify_expanded_index(args.expanded_index, array_index=args.array_index)
        payload["expanded_index"] = {
            "array_max_index": max(-1, count - 1),
            "job_count": count,
            "path": str(args.expanded_index.resolve()),
            "status": "valid",
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
