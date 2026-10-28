#!/usr/bin/env python3
"""Expand passed common-transport controls using only remaining consumed inputs."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from evicastkv_campaign import SOURCE, load_lock, submit

from mosaickv.evaluation.fox_campaign import TASKS, clean_source, sha, write_once

TRANSPORT_FILES = (
    "mosaickv/src/mosaickv/evaluation/fox_valid_slots.py",
    "mosaickv/src/mosaickv/backends/hf_runtime.py",
    "mosaickv/scripts/evicastkv_worker.py",
    "mosaickv/src/mosaickv/adapters/huggingface/base.py",
    "mosaickv/src/mosaickv/adapters/huggingface/parity.py",
    "mosaickv/src/mosaickv/adapters/huggingface/llava.py",
    "mosaickv/src/mosaickv/adapters/huggingface/types.py",
)
CONSUMED_TWO = {"ActionPrediction-0035", "ActionPrediction-0014"}


def extension_plan(parent, prior, *, source=SOURCE):
    """Validate immutable inputs and return a label-independent 30-input extension."""
    parent_lock = json.loads((parent / "protocol_lock.json").read_text())
    prior_lock = json.loads((prior / "protocol_lock.json").read_text())
    gate_path = prior / "analysis/correctness/GATE_correctness.json"
    gate = json.loads(gate_path.read_text())
    prior_sha = sha(prior / "protocol_lock.json")
    if (
        gate.get("status") != "passed"
        or gate.get("protocol_sha256") != prior_sha
        or gate.get("source_sha") != prior_lock["source_sha"]
        or gate.get("blockers")
        or any(
            gate["tests_actually_executed"].get(k) != v
            for k, v in (("expected", 8), ("actual", 8), ("completed", 8), ("pairing_count", 4))
        )
    ):
        raise ValueError("the two-input eight-request correctness gate must pass")
    old_source = Path(prior_lock["source_root"])
    clean_source(old_source, prior_lock["source_sha"])
    runtime_bindings = {}
    for name in TRANSPORT_FILES:
        digest = sha(source / name)
        if digest != sha(old_source / name):
            raise ValueError("cannot reuse controls across changed transport/runtime source")
        runtime_bindings[name] = digest
    if prior_lock["transport_regate"]["parent_protocol_sha256"] != sha(
        parent / "protocol_lock.json"
    ):
        raise ValueError("prior controls do not bind this consumed parent")
    arithmetic = prior_lock["arithmetic"]
    if arithmetic != dict(
        allow_fp16_reduced_precision_reduction=True,
        attention_layout_policy="valid_slot_decode",
    ):
        raise ValueError("prior arithmetic policy is not the validated common transport")
    info = parent_lock["cohorts"]["milebench"]["validation"]
    path = Path(info["path"])
    if sha(path) != info["sha256"]:
        raise ValueError("consumed validation ledger changed")
    ledger = json.loads(path.read_text())
    ids = [s["sample_id"] for s in ledger["samples"]]
    if (
        len(ids) != 32
        or len(set(ids)) != 32
        or ids != info["sample_ids"]
        or any(sum(s.startswith(task + "-") for s in ids) != 8 for task in TASKS)
    ):
        raise ValueError("expected the unchanged four-task, eight-per-task consumed ledger")
    old_info = prior_lock["cohorts"]["milebench"]["validation"]
    old_path = Path(old_info["path"])
    if sha(old_path) != old_info["sha256"] or set(old_info["sample_ids"]) != CONSUMED_TWO:
        raise ValueError("prior consumed identities changed")
    old_samples = json.loads(old_path.read_text())["samples"]
    by_id = {s["sample_id"]: s for s in ledger["samples"]}
    if {s["sample_id"] for s in old_samples} != CONSUMED_TWO or any(
        s != by_id.get(s["sample_id"]) for s in old_samples
    ):
        raise ValueError("prior input controls differ from the consumed parent")
    ledger["samples"] = [s for s in ledger["samples"] if s["sample_id"] not in CONSUMED_TWO]
    lock = copy.deepcopy(parent_lock)
    lock["arithmetic"] = copy.deepcopy(arithmetic)
    lock["cohorts"] = {"milebench": lock["cohorts"]["milebench"]}
    lock["budgets_mib"] = {"milebench": [16, 40]}
    lock["methods"] = {"milebench": ["full_kv", "prompt_attention_topk", "mosaickv_exact"]}
    lock["authorization"].update(automatic_stage_progression=False, confirmation_authorized=False)
    lock["resources"].update(
        maximum_a100s=6,
        array_concurrency=4,
        six_a100s_explicitly_authorized=True,
        wall_minutes=45,
    )
    lock.setdefault("inherited_gates", {})[str(gate_path)] = sha(gate_path)
    lock["transport_extension"] = dict(
        scope="consumed_representation_correctness_only",
        parent_protocol_sha256=sha(parent / "protocol_lock.json"),
        prior_protocol_sha256=prior_sha,
        reused_control_ids=sorted(CONSUMED_TWO),
        pending_control_ids=[s["sample_id"] for s in ledger["samples"]],
        runtime_bindings=runtime_bindings,
        fresh_inputs=False,
        task_scores_used=False,
        quality_or_timing_successor_authorized=False,
        lookm_transformed_cache_fidelity="not_validated_by_exact_source_oracle",
        original_native_failures_preserved=True,
    )
    return lock, ledger


def prepare(root, parent, prior):
    revision = clean_source(SOURCE)
    if root.exists():
        raise FileExistsError("extension root must be new")
    lock, ledger = extension_plan(parent, prior)
    path = root / "manifests/milebench-consumed-remaining-thirty.json"
    write_once(path, ledger)
    lock["cohorts"]["milebench"]["validation"] = dict(
        path=str(path), sha256=sha(path), sample_ids=[s["sample_id"] for s in ledger["samples"]]
    )
    lock.update(source_root=str(SOURCE), source_sha=revision)
    write_once(root / "protocol_lock.json", lock)
    load_lock(root)
    submit(root, "correctness")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "parent", "prior"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    for path in (args.root, args.parent, args.prior):
        if not path.resolve().is_relative_to(Path("/scratch/djy8hg").resolve()):
            raise ValueError("scratch-only extension")
    prepare(args.root, args.parent, args.prior)
