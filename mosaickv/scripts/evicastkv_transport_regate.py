#!/usr/bin/env python3
"""Preregistered consumed-input common-transport correctness, never quality."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from evicastkv_campaign import SOURCE, submit

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once


def prepare(root, parent, diagnostic):
    revision = clean_source(SOURCE)
    result = json.loads((diagnostic / "analysis.json").read_text())
    if result.get("transport_controls_passed") is not True:
        raise ValueError("two-case transport diagnosis has not passed unchanged controls")
    cases = result.get("cases", [])
    if (
        len(cases) != 2
        or {c["sample_id"] for c in cases} != {"ActionPrediction-0035", "ActionPrediction-0014"}
        or not all(c.get("padding_control_passed") and c.get("parity_passed") for c in cases)
    ):
        raise ValueError("the registered two consumed-case gates must both pass")
    if root.exists():
        raise FileExistsError("transport regate root must be new")
    lock = copy.deepcopy(json.loads((parent / "protocol_lock.json").read_text()))
    if "milebench" not in lock["cohorts"]:
        raise ValueError("wrong consumed cohort")
    info = lock["cohorts"]["milebench"]["validation"]
    path = Path(info["path"])
    if sha(path) != info["sha256"]:
        raise ValueError("parent input ledger changed")
    ledger = json.loads(path.read_text())
    by_id = {s["sample_id"]: s for s in ledger["samples"]}
    ids = ["ActionPrediction-0035", "ActionPrediction-0014"]
    ledger["samples"] = [by_id[sid] for sid in ids]
    small = root / "manifests/milebench-consumed-two.json"
    write_once(small, ledger)
    lock.update(source_root=str(SOURCE), source_sha=revision)
    lock["cohorts"] = {"milebench": lock["cohorts"]["milebench"]}
    lock["cohorts"]["milebench"]["validation"] = dict(
        path=str(small), sha256=sha(small), sample_ids=ids
    )
    lock["budgets_mib"] = {"milebench": [16, 40]}
    lock["methods"] = {"milebench": ["full_kv", "prompt_attention_topk", "mosaickv_exact"]}
    lock["arithmetic"] = dict(
        allow_fp16_reduced_precision_reduction=True, attention_layout_policy="valid_slot_decode"
    )
    lock["authorization"].update(automatic_stage_progression=False, confirmation_authorized=False)
    lock["resources"].update(
        maximum_a100s=6, array_concurrency=1, six_a100s_explicitly_authorized=True
    )
    lock["transport_regate"] = dict(
        diagnostic_analysis_sha256=sha(diagnostic / "analysis.json"),
        parent_protocol_sha256=sha(parent / "protocol_lock.json"),
        all_compared_methods_share_transport=True,
        exact_methods_only=True,
        lookm_transformed_cache_oracle="not_validated_by_exact_source_identity",
        original_native_gate_not_reclassified=True,
        task_scores_not_used=True,
        no_fresh_inputs=True,
        no_quality_or_timing_successor=True,
    )
    write_once(root / "protocol_lock.json", lock)
    submit(root, "correctness")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--diagnostic", required=True, type=Path)
    a = parser.parse_args()
    for p in (a.root, a.parent, a.diagnostic):
        if not p.resolve().is_relative_to(Path("/scratch/djy8hg").resolve()):
            raise ValueError("scratch-only regate")
    prepare(a.root, a.parent, a.diagnostic)
