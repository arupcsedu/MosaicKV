#!/usr/bin/env python3
"""Validate the immutable aggregate-only V96 negative calibration seal."""

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
    required_decision = {
        "experiment_id": "choice_c_v96_pareto_atomic_group_calibration",
        "classification": "sealed_negative_label_blind_calibration_result",
        "immutable_result": True,
        "overall_freeze_gate_pass": False,
        "memory_gate_pass": True,
        "aggregate_fidelity_gate_pass": False,
        "robust_fold_fidelity_gate_pass": False,
        "pareto_knapsack_activity_gate_pass": True,
        "non_group_output_impact_activity_gate_pass": True,
        "atomic_group_utilization_gate_pass": False,
        "frozen_candidate_created": False,
        "heldout_evaluation_authorized": False,
    }
    if any(seal.get(key) != value for key, value in required_decision.items()):
        raise RuntimeError("V96 seal does not preserve the negative calibration decision")
    for relation in ("protocol", "parent_negative_result_seal"):
        record = cast("dict[str, Any]", seal[relation])
        path = package_root / str(record["path"])
        if not path.is_file() or _sha(path) != record["sha256"]:
            raise RuntimeError(f"V96 {relation} hash differs from the seal")

    run_root = Path(str(seal["run_root"]))
    inventory = cast("dict[str, Any]", seal["artifact_inventory"])
    expected = (
        int(inventory["file_count"]),
        int(inventory["total_bytes"]),
        str(inventory["inventory_sha256"]),
    )
    observed = _inventory(run_root)
    if observed != expected:
        raise RuntimeError(f"V96 artifact inventory changed: expected {expected}, got {observed}")
    for name, raw in cast("dict[str, Any]", seal["artifacts"]).items():
        record = cast("dict[str, Any]", raw)
        path = run_root / str(record["path"])
        if not path.is_file() or _sha(path) != record["sha256"]:
            raise RuntimeError(f"sealed V96 artifact changed or disappeared: {name}")

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
        or ranking[0].get("pareto_knapsack_activity_eligible") is not True
        or ranking[0].get("layout_group_activity_eligible") is not True
        or ranking[0].get("layout_group_utilization_eligible") is not False
    ):
        raise RuntimeError("V96 analyses no longer encode the sealed negative behavior")
    if (run_root / "analysis/frozen_candidate.yaml").exists():
        raise RuntimeError("a frozen V96 candidate exists despite the negative seal")

    trials_path = run_root / "analysis/label_blind_calibration_trials.parquet"
    trials = pd.read_parquet(
        trials_path,
        columns=["status", "error", "residual_kv_bytes"],
    )
    if (
        not trials_path.is_file()
        or len(trials) != 192
        or set(trials["status"].astype(str)) != {"completed"}
        or bool(trials["error"].notna().any())
        or bool((trials["residual_kv_bytes"] != 0).any())
    ):
        raise RuntimeError("V96 trial coverage, completion, or residual storage changed")
    folds = pd.read_parquet(
        run_root / "analysis/robust_fidelity_folds.parquet",
        columns=["answer_gate_pass", "token_gate_pass"],
    )
    if (
        len(folds) != 8
        or int(folds["answer_gate_pass"].sum()) != 5
        or int(folds["token_gate_pass"].sum()) != 4
    ):
        raise RuntimeError("V96 robust fold outcomes changed")
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
