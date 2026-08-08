#!/usr/bin/env python3
"""Finalize a failed Choice C repair gate without loading model weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import yaml


def finalize(spec_path: Path) -> int:
    spec = cast("dict[str, Any]", yaml.safe_load(spec_path.read_text()))
    output_root = Path(str(spec["output_root"])).resolve()
    prototype = json.loads((output_root / "prototype_calibration_gate.json").read_text())
    profile = json.loads((output_root / "frozen_fidelity_profile.json").read_text())
    repair = json.loads((output_root / "frozen_repair_profile.json").read_text())
    if prototype.get("prototype_beats_exact_control") is not True:
        raise RuntimeError("this finalizer requires a passing prototype calibration gate")
    repair_spec = cast("dict[str, Any]", spec.get("repair_calibration", {}))
    minimum_precision = float(repair_spec.get("minimum_precision", 0.5))
    target_trigger_rate = float(repair_spec.get("target_trigger_rate", 0.1))
    enabled = repair.get("enabled") is True
    precision = float(repair.get("predicted_precision", 0.0))
    recall = float(repair.get("predicted_recall", 0.0))
    passed = bool(enabled and precision >= minimum_precision)
    if passed:
        raise RuntimeError("failure-only finalizer received a passing repair profile")
    repair_gate = {
        "enabled": enabled,
        "predicted_precision": precision,
        "predicted_recall": recall,
        "predicted_triggered_repairs": int(
            repair.get("predicted_triggered_repairs", 0)
        ),
        "predicted_beneficial_repairs": int(
            repair.get("predicted_beneficial_repairs", 0)
        ),
        "minimum_precision": minimum_precision,
        "target_trigger_rate": target_trigger_rate,
        "pass": False,
    }
    failure = {
        "status": "failed",
        "reason": "decision_repair_calibration_gate_failed",
        "prototype_calibration_pass": True,
        "repair_calibration_enabled": enabled,
        "repair_calibration_precision": precision,
        "repair_calibration_recall": recall,
        "repair_calibration_minimum_precision": minimum_precision,
        "internal_validation_started": True,
        "internal_validation_completed": False,
        "internal_validation_rows_used_in_gate": 0,
        "internal_validation_partial_archive": (
            "/scratch/djy8hg/runs/mosaickv/archives/"
            "choice-c-v8-internal-started-after-repair-gate-bug-17985857"
        ),
        "heldout_evaluation_started": False,
        "full_sweep_allowed": False,
    }
    (output_root / "repair_calibration_gate.json").write_text(
        json.dumps(repair_gate, indent=2, sort_keys=True) + "\n"
    )
    (output_root / "gate.json").write_text(
        json.dumps(failure, indent=2, sort_keys=True) + "\n"
    )
    report = [
        "# Choice C calibration gate",
        "",
        f"- Frozen variant: `{profile['selected_variant']}` "
        f"(`{profile['profile_sha256']}`)",
        "- Prototype beats exact calibration control: True",
        f"- Decision-aware repair enabled: {enabled}",
        f"- Calibrated repair precision/recall: {precision:.4f}/{recall:.4f}",
        "- Internal validation started: True (one canceled partial trace; excluded)",
        "- Internal validation rows used in gate: 0",
        "- Held-out evaluation started: False",
        "- Full sweep allowed: **False**",
        "",
        "The preregistered repair gate failed. Calibration artifacts are retained; "
        "one prematurely started internal trace is archived and excluded, and the "
        "held-out ledger remains unconsumed.",
    ]
    (output_root / "gate_report.md").write_text("\n".join(report) + "\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    args = parser.parse_args()
    return finalize(args.spec.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
