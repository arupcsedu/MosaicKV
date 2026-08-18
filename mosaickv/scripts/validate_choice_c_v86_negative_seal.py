#!/usr/bin/env python3
"""Validate the immutable v86 negative calibration seal and artifact inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pandas as pd  # type: ignore[import-untyped]
import yaml  # type: ignore[import-untyped]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inventory(root: Path) -> tuple[int, int, str]:
    records: list[str] = []
    total_bytes = 0
    paths = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    for path in paths:
        payload = path.read_bytes()
        total_bytes += len(payload)
        records.append(
            f"{hashlib.sha256(payload).hexdigest()}  "
            f"{path.relative_to(root).as_posix()}\n"
        )
    return len(paths), total_bytes, hashlib.sha256("".join(records).encode()).hexdigest()


def _sample_ids(path: Path) -> set[str]:
    payload = cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))
    values = cast("list[object]", payload["sample_ids"])
    result = {str(value) for value in values}
    if len(result) != len(values):
        raise RuntimeError(f"sample ledger contains duplicate identifiers: {path}")
    return result


def validate(seal_path: Path) -> dict[str, Any]:
    seal = cast("dict[str, Any]", yaml.safe_load(seal_path.read_text(encoding="utf-8")))
    package_root = Path(__file__).resolve().parents[1]
    if seal.get("experiment_id") != "choice_c_v86_disjoint_robust_fidelity_calibration":
        raise RuntimeError("unexpected v86 experiment id")
    if (
        seal.get("classification") != "sealed_negative_label_blind_calibration_result"
        or seal.get("immutable_result") is not True
        or seal.get("overall_freeze_gate_pass") is not False
        or seal.get("frozen_candidate_created") is not False
        or seal.get("heldout_evaluation_authorized") is not False
    ):
        raise RuntimeError("v86 seal does not preserve a negative, held-out-blocking result")

    protocol = cast("dict[str, Any]", seal["protocol"])
    protocol_path = package_root / str(protocol["path"])
    if _sha(protocol_path) != protocol["sha256"]:
        raise RuntimeError("v86 protocol hash differs from the negative seal")
    parent = cast("dict[str, Any]", seal["parent_negative_result_seal"])
    parent_path = package_root / str(parent["path"])
    if _sha(parent_path) != parent["sha256"]:
        raise RuntimeError("v85 parent seal hash differs from the v86 seal")

    run_root = Path(str(seal["run_root"]))
    if not run_root.is_dir():
        raise RuntimeError(f"sealed v86 run root is missing: {run_root}")
    inventory = cast("dict[str, Any]", seal["artifact_inventory"])
    observed_inventory = _inventory(run_root)
    expected_inventory = (
        int(inventory["file_count"]),
        int(inventory["total_bytes"]),
        str(inventory["inventory_sha256"]),
    )
    if observed_inventory != expected_inventory:
        raise RuntimeError(
            f"v86 artifact inventory changed: expected {expected_inventory}, "
            f"observed {observed_inventory}"
        )
    for name, record_raw in cast("dict[str, Any]", seal["artifacts"]).items():
        record = cast("dict[str, Any]", record_raw)
        path = run_root / str(record["path"])
        if not path.is_file() or _sha(path) != record["sha256"]:
            raise RuntimeError(f"sealed v86 artifact changed or disappeared: {name}")

    base_path = run_root / "analysis/label_blind_calibration.json"
    robust_path = run_root / "analysis/robust_fidelity_calibration.json"
    base = cast("dict[str, Any]", json.loads(base_path.read_text(encoding="utf-8")))
    robust = cast("dict[str, Any]", json.loads(robust_path.read_text(encoding="utf-8")))
    if (
        base.get("selected_candidate") is not None
        or robust.get("selected_candidate") is not None
        or base.get("labels_accessed") is not False
        or robust.get("labels_accessed") is not False
        or base.get("official_task_scores_accessed") is not False
        or robust.get("official_task_scores_accessed") is not False
    ):
        raise RuntimeError("v86 analysis no longer represents the sealed label-blind failure")
    if (run_root / "analysis/frozen_candidate.yaml").exists():
        raise RuntimeError("a frozen v86 candidate exists despite the negative seal")

    trials = pd.read_parquet(run_root / "analysis/label_blind_calibration_trials.parquet")
    if len(trials) != 1280 or set(trials["status"].astype(str)) != {"completed"}:
        raise RuntimeError("v86 calibration trial coverage or completion status changed")
    if "reference" in trials or "task_score" in trials:
        raise RuntimeError("label-bearing fields entered the v86 calibration trials")
    if "error" in trials and bool(trials["error"].notna().any()):
        raise RuntimeError("v86 calibration contains an error row")
    required = (
        "answer",
        "active_kv_bytes",
        "residual_kv_bytes",
        "compression_time",
        "decode_time",
        "end_to_end_time",
        "generated_tokens",
    )
    if any(column not in trials or bool(trials[column].isna().any()) for column in required):
        raise RuntimeError("v86 calibration contains a missing required value")

    v85_inputs = Path(
        "/scratch/djy8hg/runs/mosaickv/"
        "choice-c-v85-v84-frozen-untouched-heldout-quality-memory-v1/inputs"
    )
    overlaps: dict[str, int] = {}
    for task in ("docvqa", "textvqa"):
        current = _sample_ids(run_root / f"inputs/calibration_{task}.json")
        prior = _sample_ids(v85_inputs / f"heldout_{task}.json")
        overlaps[task] = len(current & prior)
    if any(overlaps.values()):
        raise RuntimeError(f"v86 calibration overlaps v85 held-out inputs: {overlaps}")

    return {
        "experiment_id": seal["experiment_id"],
        "classification": seal["classification"],
        "artifact_file_count": observed_inventory[0],
        "artifact_total_bytes": observed_inventory[1],
        "artifact_inventory_sha256": observed_inventory[2],
        "measured_rows": len(trials),
        "v85_overlap": overlaps,
        "frozen_candidate_exists": False,
        "status": "valid_negative_seal",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.seal.resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
