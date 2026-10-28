#!/usr/bin/env python3
"""Promote only measured common-runtime gates into controlled development.

Never authorizes confirmation, never tunes from quality outcomes, never changes
the runtime algorithm or inherited numerical/byte controls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from evicastkv_campaign import SOURCE, submit

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once


def runtime_files(source):
    package = source / "mosaickv/src/mosaickv"
    return {
        str(p.relative_to(package)): sha(p)
        for p in sorted(package.rglob("*.py"))
        if not p.name.startswith("fox_")
    }


def prepare(parent, root):
    clean_source(SOURCE)
    lock = json.loads((parent / "protocol_lock.json").read_text())
    old_source = Path(lock["source_root"])
    clean_source(old_source, lock["source_sha"])
    if runtime_files(SOURCE) != runtime_files(old_source):
        raise ValueError("production algorithm changed; common runtime gates cannot be inherited")
    gates = {}
    for stage in (
        "smoke",
        "representation_canary",
        "correctness",
        "scoring",
        "profiling",
        "measurement",
    ):
        path = parent / "analysis" / stage / f"GATE_{stage}.json"
        if not path.exists() or json.loads(path.read_text())["status"] != "passed":
            raise ValueError(f"common-runtime gate not passed: {stage}")
        gates[str(path)] = sha(path)
    hotspots = parent / "writing/after-measurement/profiler_hotspots.json"
    if not hotspots.exists():
        raise ValueError("actual diagnostic hotspot report is required before development")
    hotspot_report = json.loads(hotspots.read_text())
    if not hotspot_report["ranges"]:
        raise ValueError("empty profiler evidence")
    lock.update(
        name="evicastkv_fox_controlled_mechanisms_v1",
        source_root=str(SOURCE),
        source_sha=clean_source(SOURCE),
        source_patch_sha256=hashlib.sha256(b"").hexdigest(),
        parent_protocol_sha256=sha(parent / "protocol_lock.json"),
        inherited_gates=gates,
        optimization_decision=dict(
            changes=0,
            status="no_algorithm_or_runtime_optimization_applied",
            explanation=(
                "Retain measured reference runtime for mechanism study. "
                "No unverified speedup claimed."
            ),
            measured_hotspots_sha256=sha(hotspots),
        ),
        confirmation_authorized=False,
        mechanism_contract=dict(
            fraction=1 / 16,
            fixed_candidates=True,
            fixed_prompt_quotas=True,
            raw_score_terms=True,
            cross_modal_removal="remove inter-modality edges after identical graph; no refill",
            routing_policy="existing pooled-prompt entropy, NOT new per-query entropy",
            seeds=[13, 29, 47],
        ),
    )
    lock["historical_bindings"][str(parent / "protocol_lock.json")] = sha(
        parent / "protocol_lock.json"
    )
    write_once(root / "protocol_lock.json", lock)
    write_once(
        root / "inherited_controls.json",
        dict(
            parent=str(parent),
            gate_hashes=gates,
            production_files=runtime_files(SOURCE),
            confirmation_authorized=False,
        ),
    )
    write_once(
        root / "readiness.json",
        dict(
            status="controlled_development_only",
            fresh_confirmation="blocked",
            optimization="no change, no speedup claim",
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "submit"))
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.parent, args.root)
    else:
        prepare(args.parent, args.root)
        submit(args.root, "mechanisms")


if __name__ == "__main__":
    main()
