#!/usr/bin/env python3
"""Merge immutable VL-Cache completions and recovery shards, rejecting omissions.

Direct latency comparisons are always within a shared realized byte cap. Cross-cap
quality/memory comparisons are separately labeled curve tests, not same-budget rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

METHODS = ("full_kv", "prompt_attention_topk", "vl_cache_reimpl", "mosaickv_exact")
CAPS = (10, 20, 40, 80)
TASKS = ("textvqa", "docvqa")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest_controls(data, *, compressed):
    cfg = data["resolved_config"]
    out = {k: data[k] for k in ("model", "dataset", "execution", "generation", "inputs")}
    out["environment_lock"] = data["environment"]["lock_sha256"]
    out["hardware"] = data["hardware"]
    out["generation_resolved"] = cfg["generation"]
    if compressed:
        out["cache"] = cfg["cache"]
    return out


def collect(parent, recovery):
    index = [json.loads(s) for s in (parent / "matrix/jobs.jsonl").read_text().splitlines() if s]
    jobs = [j for j in index if j["family"] == "vl_cache"]
    if len(jobs) != 128:
        raise ValueError("expected 128 preregistered VL-Cache shards")
    rows, inputs, seen = [], {}, set()
    for job in jobs:
        relative = (
            f"outputs/vl_cache/{job['task']}/cap{job['cap_mib']}MiB/"
            f"shard-{job['shard_index']:02d}/job_complete.json"
        )
        candidates = [p for p in (parent / relative, recovery / relative) if p.exists()]
        if len(candidates) != 1:
            raise ValueError(f"missing/duplicate completion for original job {job['array_index']}")
        path = candidates[0]
        d = json.loads(path.read_text())
        if (
            d["status"] != "completed"
            or d["source_git_dirty"]
            or d["source_git_sha"] != job["source_git_sha"]
            or d["configured_byte_cap"] != job["byte_cap"]
            or d["sample_ledger_sha256"] != job["sample_ledger_sha256"]
        ):
            raise ValueError(f"completion identity mismatch: {path}")
        ledger = Path(job["sample_ledger_path"])
        if sha(ledger) != job["sample_ledger_sha256"]:
            raise ValueError("sample ledger changed")
        ids = json.loads(ledger.read_text())["sample_ids"]
        if len(ids) != 16 or len(set(ids)) != 16:
            raise ValueError("invalid shard denominator")
        if tuple(r["method"] for r in d["records"]) != METHODS:
            raise ValueError("incomplete method set")
        controls, shared = None, None
        for rec in d["records"]:
            if len(rec["rows"]) != 16 or {r["sample_id"] for r in rec["rows"]} != set(ids):
                raise ValueError("incomplete or duplicated samples")
            manifests = {r["manifest_path"] for r in rec["rows"]}
            if len(manifests) != 1:
                raise ValueError("unexpected manifest topology")
            mf = Path(next(iter(manifests)))
            m = json.loads(mf.read_text())
            if (
                m["source"]["git_dirty"]
                or m["source"]["git_sha"] != job["source_git_sha"]
                or not m["source"]["canonical_eligible"]
            ):
                raise ValueError("noncanonical manifest")
            common = manifest_controls(m, compressed=False)
            if shared is not None and common != shared:
                raise ValueError("shared model/input/generation/environment controls differ")
            shared = common
            if rec["method"] != "full_kv":
                c = manifest_controls(m, compressed=True)
                if controls is not None and c != controls:
                    raise ValueError("compressed cache/mandatory policies differ")
                controls = c
            for row in rec["rows"]:
                key = (job["task"], job["cap_mib"], rec["method"], row["sample_id"])
                if key in seen:
                    raise ValueError("duplicate sample-method-cap")
                seen.add(key)
                if row["status"] != "completed" or row.get("error") is not None:
                    raise ValueError("failed sample; campaign cannot be silently reduced")
                for field in ("task_score", "end_to_end_time", "compression_time", "decode_time"):
                    if (
                        not isinstance(row[field], (int, float))
                        or not math.isfinite(row[field])
                        or row[field] < 0
                    ):
                        raise ValueError("non-finite measurement")
                if row["generated_tokens"] != job["output_length"]:
                    raise ValueError("output-length mismatch")
                if rec["method"] != "full_kv" and row["active_kv_bytes"] != job["byte_cap"]:
                    raise ValueError("exact physical-byte target failed")
                rows.append(
                    {
                        **row,
                        "task": job["task"],
                        "cap_mib": job["cap_mib"],
                        "method": rec["method"],
                        "original_index": job["array_index"],
                        "completion_path": str(path),
                    }
                )
            inputs[str(mf)] = sha(mf)
        inputs[str(path)] = sha(path)
        inputs[str(ledger)] = sha(ledger)
    frame = pd.DataFrame(rows)
    for task in TASKS:
        expected = None
        for cap in CAPS:
            for method in METHODS:
                f = frame[(frame.task == task) & (frame.cap_mib == cap) & (frame.method == method)]
                ids = set(f.sample_id)
                if len(f) != 256 or (expected is not None and ids != expected):
                    raise ValueError("all caps/methods must cover identical 256 identities")
                expected = ids
        task_rows = frame[frame.task == task]
        for _, f in task_rows.groupby("sample_id"):
            if len(set(f.reference)) != 1:
                raise ValueError("reference annotation differs between paired runs")
            full = f[f.method == "full_kv"]
            if len(set(full.answer)) != 1 or len(set(full.task_score)) != 1:
                raise ValueError("FullKV deterministic quality differs across caps")
    return frame, inputs


def pair(left, right, *, latency=True):
    a, b = left.set_index("sample_id").sort_index(), right.set_index("sample_id").sort_index()
    if not a.index.equals(b.index) or not a.index.is_unique:
        raise ValueError("paired identity mismatch")
    delta = (a.task_score - b.task_score).to_numpy()
    idx = np.random.default_rng(20271014).integers(0, len(a), size=(10000, len(a)))
    result = {
        "sample_count": len(a),
        "quality_delta": float(delta.mean()),
        "quality_ci95_low": float(np.quantile(delta[idx].mean(1), 0.025)),
        "quality_ci95_high": float(np.quantile(delta[idx].mean(1), 0.975)),
        "candidate_active_bytes": float(a.active_kv_bytes.mean()),
        "baseline_active_bytes": float(b.active_kv_bytes.mean()),
        "memory_reduction": float(1 - a.active_kv_bytes.mean() / b.active_kv_bytes.mean()),
    }
    if latency:
        if not (a.active_kv_bytes == b.active_kv_bytes).all():
            raise ValueError("direct latency requires shared realized bytes")
        ratio = a.end_to_end_time.to_numpy()[idx].mean(1) / b.end_to_end_time.to_numpy()[idx].mean(
            1
        )
        result.update(
            latency_ratio=float(a.end_to_end_time.mean() / b.end_to_end_time.mean()),
            latency_ci95_low=float(np.quantile(ratio, 0.025)),
            latency_ci95_high=float(np.quantile(ratio, 0.975)),
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("write-once derived output exists")
    frame, inputs = collect(args.parent, args.recovery)
    direct, curves, summary = [], [], []
    for (task, cap, method), f in frame.groupby(["task", "cap_mib", "method"]):
        row = dict(
            task=task,
            cap_mib=int(cap),
            method=method,
            samples=len(f),
            task_score=float(f.task_score.mean()),
            active_kv_bytes=float(f.active_kv_bytes.mean()),
        )
        if method != "full_kv":
            row.update(
                end_to_end_time=float(f.end_to_end_time.mean()),
                compression_time=float(f.compression_time.mean()),
                decode_time=float(f.decode_time.mean()),
            )
        summary.append(row)
    for task in TASKS:
        for cap in CAPS:
            a = frame[
                (frame.task == task) & (frame.cap_mib == cap) & (frame.method == "mosaickv_exact")
            ]
            for method in ("prompt_attention_topk", "vl_cache_reimpl"):
                for bcap in CAPS:
                    b = frame[
                        (frame.task == task) & (frame.cap_mib == bcap) & (frame.method == method)
                    ]
                    if cap == bcap:
                        direct.append(dict(task=task, cap_mib=cap, competitor=method, **pair(a, b)))
                    elif cap < bcap:
                        p = pair(a, b, latency=False)
                        curves.append(
                            dict(
                                task=task,
                                candidate_cap_mib=cap,
                                baseline_cap_mib=bcap,
                                competitor=method,
                                comparison="cross_cap_quality_memory_curve_not_same_budget",
                                gate_30pct_equal_quality=p["memory_reduction"] >= 0.30
                                and p["quality_ci95_low"] >= 0,
                                **p,
                            )
                        )
    args.output.mkdir(parents=True)
    frame.to_parquet(args.output / "rows.parquet", index=False)
    pd.DataFrame(summary).to_parquet(args.output / "summary.parquet", index=False)
    pd.DataFrame(direct).to_parquet(args.output / "paired_same_byte.parquet", index=False)
    pd.DataFrame(curves).to_parquet(args.output / "quality_memory_curve_tests.parquet", index=False)
    report = dict(
        status="complete",
        expected_shards=128,
        examples_per_task=256,
        rows=len(frame),
        failed_rows=0,
        inputs=inputs,
        analysis_script_sha256=sha(__file__),
        source_experiments_unchanged=True,
        fullkv_latency_excluded=True,
        curve_quality_margin=0,
        curve_intervals="pointwise exploratory; no multiplicity adjustment",
        paper_superiority_claim_authorized=False,
        paired=direct,
        curve_tests=curves,
    )
    with (args.output / "report.json").open("x") as f:
        json.dump(report, f, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps({"status": "complete", "output": str(args.output), "rows": len(frame)}))


if __name__ == "__main__":
    main()
