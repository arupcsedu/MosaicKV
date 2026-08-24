#!/usr/bin/env python3
"""Validate the immutable aggregate-only V99 negative calibration seal."""

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
        (candidate for candidate in root.rglob("*") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(root).as_posix(),
    )
    for path in paths:
        payload = path.read_bytes()
        total_bytes += len(payload)
        records.append(
            f"{hashlib.sha256(payload).hexdigest()}  {path.relative_to(root).as_posix()}\n"
        )
    return len(records), total_bytes, hashlib.sha256("".join(records).encode()).hexdigest()


def validate(seal_path: Path) -> dict[str, Any]:
    seal = cast("dict[str, Any]", yaml.safe_load(seal_path.read_text(encoding="utf-8")))
    package_root = Path(__file__).resolve().parents[1]
    required = {
        "experiment_id": "choice_c_v99_progressive_logit_safe_eviction_calibration",
        "classification": "sealed_negative_label_blind_calibration_result",
        "immutable_result": True,
        "overall_freeze_gate_pass": False,
        "memory_gate_pass": True,
        "aggregate_fidelity_gate_pass": True,
        "robust_fold_fidelity_gate_pass": False,
        "atomic_group_utilization_gate_pass": False,
        "progressive_logit_safe_eviction_gate_pass": False,
        "frozen_candidate_created": False,
        "heldout_evaluation_authorized": False,
    }
    if any(seal.get(key) != value for key, value in required.items()):
        raise RuntimeError("V99 seal does not preserve the negative calibration decision")
    for relation in ("protocol", "parent_negative_result_seal"):
        record = cast("dict[str, Any]", seal[relation])
        path = package_root / str(record["path"])
        if not path.is_file() or _sha(path) != record["sha256"]:
            raise RuntimeError(f"V99 {relation} hash differs from the seal")

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
        raise RuntimeError(f"V99 artifact inventory changed: expected {expected}, got {observed}")
    for name, raw in cast("dict[str, Any]", seal["artifacts"]).items():
        record = cast("dict[str, Any]", raw)
        root = ledger_root if record["root"] == "immutable_ledger_root" else run_root
        path = root / str(record["path"])
        if not path.is_file() or _sha(path) != record["sha256"]:
            raise RuntimeError(f"sealed V99 artifact changed or disappeared: {name}")

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
        or ranking[0].get("progressive_logit_safe_eviction_eligible") is not False
    ):
        raise RuntimeError("V99 analyses no longer encode the sealed negative behavior")
    if (run_root / "analysis/frozen_candidate.yaml").exists():
        raise RuntimeError("a frozen V99 candidate exists despite the negative seal")

    trials = pd.read_parquet(
        run_root / "analysis/label_blind_calibration_trials.parquet",
        columns=[
            "status",
            "error",
            "residual_kv_bytes",
            "arm",
            "progressive_eviction_target_reached",
            "progressive_eviction_selected_candidate_passed",
            "progressive_eviction_used_target_fallback",
            "progressive_eviction_probes_used",
            "progressive_eviction_accepted_checkpoints",
            "progressive_eviction_rejected_checkpoints",
            "progressive_eviction_started_from_fullkv",
            "progressive_eviction_fullkv_prefill_reused",
            "progressive_eviction_second_prefill_performed",
            "progressive_eviction_probe_states_discarded_before_measured_decode",
            "progressive_eviction_accepted_greedy_token_match",
            "progressive_eviction_accepted_bytes_strictly_decrease",
            "external_stable_query_coverage_initial_layout_group_protected_bytes",
            "external_stable_query_coverage_initial_layout_group_target_bytes",
        ],
    )
    candidate = trials[trials["arm"] == "progressive_fullkv_logit_safe_exact_b4_28_5"]
    utilization = (
        candidate["external_stable_query_coverage_initial_layout_group_protected_bytes"]
        / candidate["external_stable_query_coverage_initial_layout_group_target_bytes"]
    )
    if (
        len(trials) != 192
        or set(trials["status"].astype(str)) != {"completed"}
        or bool(trials["error"].notna().any())
        or bool((trials["residual_kv_bytes"] != 0).any())
        or len(candidate) != 64
        or int(candidate["progressive_eviction_target_reached"].sum()) != 1
        or int(candidate["progressive_eviction_selected_candidate_passed"].sum()) != 1
        or int(candidate["progressive_eviction_used_target_fallback"].sum()) != 63
        or int(candidate["progressive_eviction_probes_used"].sum()) != 8073
        or int(candidate["progressive_eviction_accepted_checkpoints"].sum()) != 1146
        or int(candidate["progressive_eviction_rejected_checkpoints"].sum()) != 6927
        or int(candidate["progressive_eviction_started_from_fullkv"].sum()) != 64
        or int(candidate["progressive_eviction_fullkv_prefill_reused"].sum()) != 64
        or bool(candidate["progressive_eviction_second_prefill_performed"].any())
        or int(
            candidate["progressive_eviction_probe_states_discarded_before_measured_decode"].sum()
        )
        != 64
        or int(candidate["progressive_eviction_accepted_greedy_token_match"].sum()) != 64
        or int(candidate["progressive_eviction_accepted_bytes_strictly_decrease"].sum()) != 64
        or int((utilization >= 0.75).sum()) != 1
    ):
        raise RuntimeError("V99 trial coverage, completion, or progressive outcomes changed")
    folds = pd.read_parquet(
        run_root / "analysis/robust_fidelity_folds.parquet",
        columns=["answer_gate_pass", "token_gate_pass"],
    )
    if (
        len(folds) != 8
        or int(folds["answer_gate_pass"].sum()) != 6
        or int(folds["token_gate_pass"].sum()) != 3
    ):
        raise RuntimeError("V99 robust fold outcomes changed")
    return {
        "experiment_id": seal["experiment_id"],
        "classification": seal["classification"],
        "artifact_file_count": observed[0],
        "artifact_total_bytes": observed[1],
        "artifact_inventory_sha256": observed[2],
        "measured_rows": len(trials),
        "progressive_target_reached_samples": int(
            candidate["progressive_eviction_target_reached"].sum()
        ),
        "answer_folds_passed": int(folds["answer_gate_pass"].sum()),
        "token_folds_passed": int(folds["token_gate_pass"].sum()),
        "frozen_candidate_exists": False,
        "heldout_evaluation_authorized": False,
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
