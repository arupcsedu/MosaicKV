#!/usr/bin/env python3
"""Validate the immutable aggregate-only V92 negative calibration seal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import pandas as pd  # type: ignore[import-untyped]
import yaml
from validate_choice_c_v87_negative_seal import _inventory, _sha


def validate(seal_path: Path) -> dict[str, Any]:
    seal = cast("dict[str, Any]", yaml.safe_load(seal_path.read_text(encoding="utf-8")))
    package_root = Path(__file__).resolve().parents[1]
    if (
        seal.get("experiment_id") != "choice_c_v92_output_impact_stability_calibration"
        or seal.get("classification") != "sealed_negative_label_blind_calibration_result"
        or seal.get("immutable_result") is not True
        or seal.get("overall_freeze_gate_pass") is not False
        or seal.get("memory_gate_pass") is not True
        or seal.get("aggregate_fidelity_gate_pass") is not False
        or seal.get("robust_fold_fidelity_gate_pass") is not False
        or seal.get("frozen_candidate_created") is not False
        or seal.get("heldout_evaluation_authorized") is not False
    ):
        raise RuntimeError("V92 seal does not preserve the negative calibration decision")
    for relation in ("protocol", "parent_negative_result_seal"):
        record = cast("dict[str, Any]", seal[relation])
        path = package_root / str(record["path"])
        if not path.is_file() or _sha(path) != record["sha256"]:
            raise RuntimeError(f"V92 {relation} hash differs from the seal")
    run_root = Path(str(seal["run_root"]))
    inventory = cast("dict[str, Any]", seal["artifact_inventory"])
    observed = _inventory(run_root)
    expected = (
        int(inventory["file_count"]),
        int(inventory["total_bytes"]),
        str(inventory["inventory_sha256"]),
    )
    if observed != expected:
        raise RuntimeError(f"V92 artifact inventory changed: expected {expected}, got {observed}")
    for name, raw in cast("dict[str, Any]", seal["artifacts"]).items():
        record = cast("dict[str, Any]", raw)
        path = run_root / str(record["path"])
        if not path.is_file() or _sha(path) != record["sha256"]:
            raise RuntimeError(f"sealed V92 artifact changed or disappeared: {name}")
    base = json.loads((run_root / "analysis/label_blind_calibration.json").read_text())
    robust = json.loads((run_root / "analysis/robust_fidelity_calibration.json").read_text())
    ranking = cast("list[dict[str, Any]]", base["candidate_ranking"])
    if (
        base.get("selected_candidate") is not None
        or robust.get("selected_candidate") is not None
        or base.get("all_configured_gates_eligible_candidate_found") is not False
        or robust.get("all_configured_gates_eligible_candidate_found") is not False
        or base.get("labels_accessed") is not False
        or robust.get("labels_accessed") is not False
        or len(ranking) != 1
        or ranking[0].get("memory_eligible") is not True
        or ranking[0].get("fidelity_eligible") is not False
    ):
        raise RuntimeError("V92 analyses no longer encode memory-pass/fidelity-fail behavior")
    if (run_root / "analysis/frozen_candidate.yaml").exists():
        raise RuntimeError("a frozen V92 candidate exists despite the negative seal")
    trials = pd.read_parquet(run_root / "analysis/label_blind_calibration_trials.parquet")
    if len(trials) != 192 or set(trials["status"].astype(str)) != {"completed"}:
        raise RuntimeError("V92 trial coverage or completion status changed")
    if "reference" in trials or "task_score" in trials:
        raise RuntimeError("label-bearing fields entered the V92 calibration trials")
    if "error" in trials and bool(trials["error"].notna().any()):
        raise RuntimeError("V92 calibration contains an error row")
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
        raise RuntimeError("V92 calibration contains a missing required value")
    if bool((trials["residual_kv_bytes"] != 0).any()):
        raise RuntimeError("V92 exact-only calibration unexpectedly stored residual bytes")
    folds = pd.read_parquet(run_root / "analysis/robust_fidelity_folds.parquet")
    if (
        len(folds) != 8
        or int(folds["answer_gate_pass"].sum()) != 3
        or int(folds["token_gate_pass"].sum()) != 4
    ):
        raise RuntimeError("V92 robust fold outcomes changed")
    return {
        "experiment_id": seal["experiment_id"],
        "classification": seal["classification"],
        "artifact_file_count": observed[0],
        "artifact_total_bytes": observed[1],
        "artifact_inventory_sha256": observed[2],
        "measured_rows": len(trials),
        "answer_folds_passed": int(folds["answer_gate_pass"].sum()),
        "token_folds_passed": int(folds["token_gate_pass"].sum()),
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
