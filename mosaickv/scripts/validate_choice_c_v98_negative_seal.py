#!/usr/bin/env python3
"""Validate the immutable aggregate-only V98 negative calibration seal."""

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
    required = {
        "experiment_id": "choice_c_v98_short_horizon_logit_calibration",
        "classification": "sealed_negative_label_blind_calibration_result",
        "immutable_result": True,
        "overall_freeze_gate_pass": False,
        "memory_gate_pass": True,
        "aggregate_fidelity_gate_pass": True,
        "robust_fold_fidelity_gate_pass": False,
        "layout_group_activity_gate_pass": True,
        "atomic_group_utilization_gate_pass": False,
        "pareto_knapsack_activity_gate_pass": True,
        "joint_group_donor_activity_gate_pass": True,
        "short_horizon_logit_stability_gate_pass": False,
        "frozen_candidate_created": False,
        "heldout_evaluation_authorized": False,
    }
    if any(seal.get(key) != value for key, value in required.items()):
        raise RuntimeError("V98 seal does not preserve the negative calibration decision")
    for relation in ("protocol", "parent_negative_result_seal"):
        record = cast("dict[str, Any]", seal[relation])
        path = package_root / str(record["path"])
        if not path.is_file() or _sha(path) != record["sha256"]:
            raise RuntimeError(f"V98 {relation} hash differs from the seal")

    run_root = Path(str(seal["run_root"]))
    ledger_root = Path(str(seal["immutable_ledger_root"]))
    inventory = cast("dict[str, Any]", seal["artifact_inventory"])
    expected = (
        int(inventory["file_count"]),
        int(inventory["total_bytes"]),
        str(inventory["inventory_sha256"]),
    )
    observed = _inventory(run_root)
    if observed != expected:
        raise RuntimeError(f"V98 artifact inventory changed: expected {expected}, got {observed}")
    for name, raw in cast("dict[str, Any]", seal["artifacts"]).items():
        record = cast("dict[str, Any]", raw)
        root = ledger_root if record["root"] == "immutable_ledger_root" else run_root
        path = root / str(record["path"])
        if not path.is_file() or _sha(path) != record["sha256"]:
            raise RuntimeError(f"sealed V98 artifact changed or disappeared: {name}")

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
        or ranking[0].get("fidelity_eligible") is not True
        or ranking[0].get("layout_group_utilization_eligible") is not False
        or ranking[0].get("short_horizon_logit_stability_eligible") is not False
    ):
        raise RuntimeError("V98 analyses no longer encode the sealed negative behavior")
    if (run_root / "analysis/frozen_candidate.yaml").exists():
        raise RuntimeError("a frozen V98 candidate exists despite the negative seal")

    trials_path = run_root / "analysis/label_blind_calibration_trials.parquet"
    trials = pd.read_parquet(
        trials_path,
        columns=[
            "status",
            "error",
            "residual_kv_bytes",
            "arm",
            "logit_stability_selected_candidate_passed",
            "logit_stability_selected_greedy_token_match",
            "logit_stability_used_fail_closed_scalar_anchor",
        ],
    )
    candidate = trials[trials["arm"] == "short_logit_stable_joint_group_exact_b4_28_5"]
    if (
        len(trials) != 192
        or set(trials["status"].astype(str)) != {"completed"}
        or bool(trials["error"].notna().any())
        or bool((trials["residual_kv_bytes"] != 0).any())
        or len(candidate) != 64
        or int(candidate["logit_stability_selected_candidate_passed"].sum()) != 3
        or int(candidate["logit_stability_selected_greedy_token_match"].sum()) != 49
        or int(candidate["logit_stability_used_fail_closed_scalar_anchor"].sum()) != 61
    ):
        raise RuntimeError("V98 trial coverage, completion, or logit outcomes changed")
    folds = pd.read_parquet(
        run_root / "analysis/robust_fidelity_folds.parquet",
        columns=["answer_gate_pass", "token_gate_pass"],
    )
    if (
        len(folds) != 8
        or int(folds["answer_gate_pass"].sum()) != 6
        or int(folds["token_gate_pass"].sum()) != 4
    ):
        raise RuntimeError("V98 robust fold outcomes changed")
    return {
        "experiment_id": seal["experiment_id"],
        "classification": seal["classification"],
        "artifact_file_count": observed[0],
        "artifact_total_bytes": observed[1],
        "artifact_inventory_sha256": observed[2],
        "measured_rows": len(trials),
        "logit_gate_passed_samples": int(
            candidate["logit_stability_selected_candidate_passed"].sum()
        ),
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
