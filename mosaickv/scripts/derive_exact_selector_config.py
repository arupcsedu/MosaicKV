#!/usr/bin/env python3
"""Derive an exact-only selector config from a frozen MosaicKV candidate."""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any, cast

import yaml  # type: ignore[import-untyped]


def derive_exact_selector_config(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("source configuration must be a YAML object")
    config = cast("dict[str, Any]", value)
    if config.get("method") not in {"mosaickv_full", "mosaickv_proto", "mosaickv_exact"}:
        raise ValueError("source configuration is not a MosaicKV candidate")
    config["method"] = "mosaickv_exact"
    prototypes = cast("dict[str, Any]", config.get("prototypes", {}))
    prototypes["enabled"] = False
    config["prototypes"] = prototypes
    residual = cast("dict[str, Any]", config.get("residual", {}))
    residual["enabled"] = False
    config["residual"] = residual
    repair = cast("dict[str, Any]", config.get("repair", {}))
    repair.update(
        {
            "enabled": False,
            "evaluation_only": False,
            "counterfactual_draft_steps": 0,
            "counterfactual_route_heads": False,
            "decision_high_risk_threshold": None,
            "draft_kl_threshold": None,
            "draft_require_token_mismatch": False,
            "max_blocks_per_step": 0,
            "policy": "none",
        }
    )
    config["repair"] = repair
    cache = cast("dict[str, Any]", config.get("cache", {}))
    cache["total_storage_budget_bytes"] = None
    config["cache"] = cache
    return config


def _write_once(path: Path, payload: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite derived configuration: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.input.resolve()
    output = args.output.resolve()
    value = yaml.safe_load(source.read_text(encoding="utf-8"))
    derived = derive_exact_selector_config(value)
    encoded = yaml.safe_dump(derived, sort_keys=True)
    _write_once(output, encoded)
    print(
        {
            "input": str(source),
            "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "output": str(output),
            "output_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
