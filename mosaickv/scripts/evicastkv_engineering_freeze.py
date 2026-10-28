#!/usr/bin/env python3
"""Freeze the existing policy for bounded consumed-input correctness only."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from evicastkv_campaign import SOURCE, load_lock, resolved_config, submit

from mosaickv.config import canonical_config, config_sha256
from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once


def freeze(root, coco_parent, fullgraph_retry):
    for path in (root, coco_parent, fullgraph_retry):
        if not path.resolve().is_relative_to(Path("/scratch/djy8hg").resolve()):
            raise ValueError("scratch-only engineering freeze")
    revision = clean_source(SOURCE)
    gate = fullgraph_retry / "analysis/full_graph_pilot/GATE_full_graph_pilot.json"
    passed = json.loads(gate.read_text())
    if passed["status"] != "passed" or passed["tests_actually_executed"]["completed"] != 16:
        raise ValueError("full-routing repair has not passed its bounded gate")
    pilot_lock = json.loads((fullgraph_retry / "protocol_lock.json").read_text())
    runtime = Path("mosaickv/src/mosaickv/backends/hf_runtime.py")
    if sha(SOURCE / runtime) != sha(Path(pilot_lock["source_root"]) / runtime):
        raise ValueError("runtime differs from validated full-routing source")
    parent = json.loads((coco_parent / "protocol_lock.json").read_text())
    lock = copy.deepcopy(parent)
    lock.update(source_sha=revision, source_root=str(SOURCE))
    lock["authorization"].update(automatic_stage_progression=False, confirmation_authorized=False)
    lock["resources"].update(
        maximum_a100s=6, array_concurrency=1, six_a100s_explicitly_authorized=True
    )
    lock.setdefault("inherited_gates", {})[str(gate)] = sha(gate)
    info = parent["cohorts"]["coco"]["validation"]
    path = Path(info["path"])
    if sha(path) != info["sha256"]:
        raise ValueError("consumed parent ledger changed")
    ledger = json.loads(path.read_text())
    ledger["samples"] = ledger["samples"][:2]
    small = root / "manifests/coco-consumed-correctness.json"
    write_once(small, ledger)
    lock["cohorts"]["coco"]["validation"] = dict(
        path=str(small), sha256=sha(small), sample_ids=[s["sample_id"] for s in ledger["samples"]]
    )
    write_once(root / "protocol_lock.json", lock)
    configs = []
    for method in ("full_kv", "prompt_attention_topk", "vl_cache_reimpl", "mosaickv_exact"):
        for cap in [None] if method == "full_kv" else [20, 40]:
            cfg = resolved_config(lock, "coco", method, cap)
            configs.append(
                dict(
                    method=method,
                    cap_mib=cap,
                    config=canonical_config(cfg),
                    config_sha256=config_sha256(cfg),
                )
            )
    write_once(
        root / "engineering_config_lock.json",
        dict(
            source_sha=revision,
            scope="consumed_COCO_boundary_A_correctness_only",
            configs=configs,
            status="engineering_frozen_not_paper_approved",
            no_outcome_based_parameter_selection=True,
            runtime_sha256=sha(SOURCE / runtime),
            fullgraph_gate_sha256=sha(gate),
            confirmation_authorized=False,
            human_reviewed=False,
            MileBench_execution_blocked=(
                "padding control and media/source-group verification unresolved"
            ),
            boundary_B_inference_blocked=True,
        ),
    )
    load_lock(root)
    submit(root, "correctness")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--coco-parent", required=True, type=Path)
    parser.add_argument("--fullgraph-retry", required=True, type=Path)
    args = parser.parse_args()
    if args.root.exists():
        raise FileExistsError("engineering freeze is immutable")
    freeze(args.root, args.coco_parent, args.fullgraph_retry)
