#!/usr/bin/env python3
"""Extend passed shared-byte pilots to a preregistered final COCO curve.

No parameter tuning or submission. Compatible pilot completions are hash-bound
and reused, including FullKV exactly once per input across all four caps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from manage_coco_shared_bytes import passed_report
from prepare_native_metric_shared_bytes import CAPS, SEED, configurations
from run_byte_capped_published_baseline_pilot import _require_clean_source
from run_coco_shared_bytes import verified_success
from run_prefixkv_decode_preflight import digest, write_once

from mosaickv.config import DatasetConfig, canonical_config, config_sha256
from mosaickv.evaluation.paper_native_lanes import load_paper_native_ledger
from mosaickv.evaluation.tasks import default_task_registry


def prepare(args):
    source = Path(__file__).resolve().parents[2]
    parent_path = args.parent / "plan.json"
    if digest(parent_path) != args.parent_sha:
        raise ValueError("parent plan changed")
    parent = json.loads(parent_path.read_text())
    _require_clean_source(source, parent["source_sha"])
    gate, _ = passed_report(args.parent, "larger", args.parent_sha)
    registration = args.parent / "final-preregistration.json"
    registered = json.loads(registration.read_text())
    if registered["plan_sha256"] != args.parent_sha or registered["pilot_report_sha256"] != digest(
        gate
    ):
        raise ValueError("sample-size registration mismatch")
    if digest(args.pool) not in parent["bound_files"].values():
        raise ValueError("final pool is not the original frozen COCO pool")
    task = default_task_registry().resolve("coco_caption")
    payload = json.loads(args.pool.read_text())
    ledger = load_paper_native_ledger(
        args.pool,
        expected_sha256=digest(args.pool),
        task=task,
        dataset_revision=payload["dataset_revision"],
    )
    n = registered["sample_count"]
    if not 256 <= n <= min(4096, len(ledger.samples)):
        raise ValueError("invalid registered final sample count")
    ids = sorted(
        (s.sample_id for s in ledger.samples),
        key=lambda sid: hashlib.sha256(f"{SEED}:coco_caption:{sid}".encode()).digest(),
    )[:n]
    samples = {s["sample_id"]: s for s in payload["samples"]}
    root = args.root.resolve()
    if (
        root.exists()
        or Path("/scratch/djy8hg").resolve() not in root.parents
        or source in root.parents
    ):
        raise ValueError("new external scratch root required")
    reuse = {}
    for job in parent["jobs"]:
        for sid in job["sample_ids"]:
            if sid not in ids:
                continue
            sample_root = Path(job["output"]) / "samples" / hashlib.sha256(sid.encode()).hexdigest()
            record = verified_success(sample_root, args.parent_sha, job["config_sha"])
            if record is None:
                continue
            path = next(sample_root.glob("attempt-*/success.json"))
            key = json.dumps([job["method"], job["cap_mib"], sid])
            if key in reuse:
                raise ValueError("duplicate compatible pilot identity")
            reuse[key] = dict(path=str(path), sha256=digest(path), plan_sha256=args.parent_sha)
    plan = dict(
        parent,
        jobs=[],
        reuse_records=reuse,
        final_registration=str(registration),
        final_registration_sha256=digest(registration),
        parent_intermediate_report=str(gate),
        parent_intermediate_report_sha256=digest(gate),
        final_sample_count=n,
        final_execution_authorized=args.authorize_final,
        validation_execution_authorized=False,
        larger_execution_authorized=False,
        final_cohort_includes_previously_evaluated_pilots=True,
    )
    dataset = DatasetConfig(ledger.dataset_id, ledger.dataset_revision, ledger.split)
    for shard, offset in enumerate(range(0, len(ids), 16)):
        chosen = ids[offset : offset + 16]
        shard_path = root / "ledgers" / f"{shard:03d}.json"
        write_once(
            shard_path,
            dict(
                schema_version=1,
                task=task.name,
                dataset_id=ledger.dataset_id,
                dataset_revision=ledger.dataset_revision,
                split=ledger.split,
                samples=[samples[s] for s in chosen],
            ),
        )
        for cap in CAPS:
            for method, cfg in configurations(source, task.name, dataset, cap).items():
                if method == "full_kv" and cap != CAPS[0]:
                    continue
                config_path = root / "configs" / f"{method}-{cap}.json"
                if not config_path.exists():
                    write_once(config_path, canonical_config(cfg))
                plan["jobs"].append(
                    dict(
                        stage="final",
                        shard=shard,
                        method=method,
                        cap_mib=None if method == "full_kv" else cap,
                        sample_ids=chosen,
                        ledger=str(shard_path),
                        ledger_sha=digest(shard_path),
                        config=str(config_path),
                        config_sha=config_sha256(cfg),
                        output=str(root / "outputs" / f"{shard:03d}-{method}-{cap}"),
                    )
                )
    write_once(root / "plan.json", plan)
    print(
        json.dumps(
            dict(
                plan=str(root / "plan.json"),
                sha256=digest(root / "plan.json"),
                reused_predictions=len(reuse),
                samples=n,
                submitted=False,
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--parent-sha", required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--authorize-final", action="store_true")
    prepare(parser.parse_args())
