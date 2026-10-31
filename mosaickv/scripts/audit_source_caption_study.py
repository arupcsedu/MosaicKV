#!/usr/bin/env python3
"""Derived source-only whole-trajectory identity audit; never paper promotion."""

import argparse
import json
import math
import subprocess
from pathlib import Path

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
METHODS = (
    "mosaickv_exact",
    "prompt_attention_topk",
    "lookm_reimpl",
    "vl_cache_reimpl",
    "prefixkv_reimpl",
)
FIELDS = (
    "initial_active_kv_storage_bytes",
    "final_active_kv_storage_bytes",
    "peak_active_kv_storage_bytes",
)


def audit_identity(rows):
    lookup = {(r["sample_id"], r["method"], r["precision"]): r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError("duplicate sample/method/precision")
    checks = []
    for key, row in lookup.items():
        if row["precision"] != "source_boundary8":
            continue
        anchor = "balanced_boundary8" if row["method"] in METHODS[:2] else "head3_boundary8"
        old = lookup.get((*key[:2], anchor))
        if old is None:
            raise ValueError("missing actual same-selection anchor")
        identity = (
            row["logits_sha256"] == old["logits_sha256"]
            and row["scored_answer"] == old["scored_answer"]
        )
        if not identity:
            raise ValueError("source-only representation changed 128-token trajectory")
        changes = {}
        for field in FIELDS:
            a, b = old["storage"][field], row["storage"][field]
            if any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in (a, b)):
                raise ValueError("finite positive actual storage required")
            changes[field] = dict(anchor=a, source=b, saved_bytes=a - b)
        if changes[FIELDS[0]]["saved_bytes"] < 0 or changes[FIELDS[1]]["saved_bytes"] < 0:
            raise ValueError("lossless source option exceeded its actual anchor storage")
        checks.append(
            dict(
                sample_id=key[0],
                method=key[1],
                logits_128_bitwise=True,
                scored_answer_identical=True,
                storage=changes,
            )
        )
    return checks


def main(root, output):
    plan_path, gate_path = root / "plan.json", root / "analysis/GATE.json"
    plan, gate = (json.loads(p.read_text()) for p in (plan_path, gate_path))
    if (
        not gate["complete"]
        or gate["failures"]
        or gate["new_rows"] != 160
        or gate["total_rows"] != 1008
        or gate["plan_sha256"] != sha(plan_path)
    ):
        raise ValueError("complete hash-bound original analysis required")
    if subprocess.check_output(
        ["git", "diff", plan["source_sha"], "HEAD", "--", "mosaickv/src"], cwd=SOURCE
    ):
        raise ValueError("analysis must not change the producer runtime package")
    rows_path = root / "analysis/rows.json"
    payload = json.loads(rows_path.read_text())
    for path, digest in payload["original_producer_hashes"].items():
        if sha(Path(path)) != digest:
            raise ValueError("original producer artifact changed")
    rows = payload["rows"]
    checks = audit_identity(rows)
    if len(checks) != 80 or {c["method"] for c in checks} != set(METHODS):
        raise ValueError("all five methods and sixteen source-only identities required")
    for method in METHODS:
        for policy in ("source_boundary8", "delimiter_boundary8"):
            selected = [r for r in rows if r["method"] == method and r["precision"] == policy]
            if len(selected) != 16 or any(
                r["status"] != "completed" or r["used_fullkv_fallback"] for r in selected
            ):
                raise ValueError("complete common-option baseline rows without fallback required")
    write_once(
        output,
        dict(
            audit_source_sha=clean_source(SOURCE),
            producer_source_sha=plan["source_sha"],
            inputs={str(p): sha(p) for p in (plan_path, gate_path, rows_path)},
            complete=True,
            whole_128_logit_identity_checks=checks,
            lossless_representation_checks_passed=True,
            completion_is_changed_membership_not_subject_to_source_identity=True,
            identical_raw_source_anchor_not_encoded_byte_matched=True,
            independent_quality_N=0,
            superiority_established=False,
            no_fresh_cohort_or_paper_promotion=True,
            peak_boundary="observed_decode_owned_cache_not_whole_request",
        ),
    )
    print(
        json.dumps(
            dict(complete=True, bitwise_128_trajectory_checks=len(checks), output=str(output))
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.output)
