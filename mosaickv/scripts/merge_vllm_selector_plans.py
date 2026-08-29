#!/usr/bin/env python3
"""Merge hash-bound selector plans for an A-B-A persistent-engine probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from mosaickv.backends.native_selector_plan import (
    load_native_selector_plan_catalog,
    merge_native_selector_plan_catalogs,
)
from mosaickv.types import JsonObject


def _write_once(path: Path, payload: JsonObject) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite merged selector plan: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"path": str(path), "sha256": hashlib.sha256(encoded).hexdigest()}))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, action="append", required=True)
    parser.add_argument("--catalog-sha256", action="append", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.catalog) != len(args.catalog_sha256):
        raise ValueError("catalog paths and SHA-256 values must have equal lengths")
    catalogs = tuple(
        load_native_selector_plan_catalog(
            path.resolve(),
            digest,
            model_id=args.model_id,
            model_revision=args.model_revision,
            task=args.task,
            dataset_revision=args.dataset_revision,
            method=args.method,
        )
        for path, digest in zip(args.catalog, args.catalog_sha256, strict=True)
    )
    _write_once(args.output.resolve(), merge_native_selector_plan_catalogs(catalogs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
