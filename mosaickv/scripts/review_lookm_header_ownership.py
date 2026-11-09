#!/usr/bin/env python3
"""CPU-only derived review of four immutable LOOK-M ownership captures."""

import argparse
import gc
import gzip
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once
from mosaickv.measurements.kv_timeline import valid_whole_request_certificate
from mosaickv.measurements.lookm_header_review import (
    PRODUCER_SHA,
    SOURCE_FUNCTIONS,
    SOURCE_LINES,
    reviewed,
)

SOURCE = Path(__file__).resolve().parents[2]
PARENT = Path("/scratch/djy8hg/runs/evicastkv/common-baseline-headers-lookm-retry-20261108-v1")


def run(root):
    source = clean_source(SOURCE)
    if root.exists():
        raise FileExistsError("new immutable derived review directory required")
    parent = json.loads((PARENT / "plan.json").read_text())
    if parent["source_sha"] != PRODUCER_SHA:
        raise ValueError("exact reviewed producer required")
    bound = {}
    for suffix in {r[0] for r in (*SOURCE_FUNCTIONS, *SOURCE_LINES)}:
        path = "mosaickv/src" + suffix
        content = subprocess.check_output(["git", "show", PRODUCER_SHA + ":" + path], cwd=SOURCE)
        expected = hashlib.sha256(content).hexdigest()
        if sha(SOURCE / path) != expected:
            raise ValueError("audited function/line source changed: " + path)
        bound[path] = expected
    root.mkdir(parents=True)
    results = []
    write_once(
        root / "manifest.json",
        dict(
            source_sha=source,
            producer_sha=PRODUCER_SHA,
            parent_plan_sha=sha(PARENT / "plan.json"),
            reviewed_source_hashes=bound,
            classification_rules=(*SOURCE_FUNCTIONS, *SOURCE_LINES),
            no_GPU_inference=True,
            scope="requested PyTorch KV plus shared/algorithm workspace; not process HBM",
            fresh_cohort_authorized=False,
        ),
    )
    for i in (0, 1):
        rows = [json.loads(x) for x in (PARENT / f"model/{i}/raw.jsonl").read_text().splitlines()]
        if len(rows) != 2 or any(
            r["status"] != "completed" or not r["numerically_admissible"] for r in rows
        ):
            raise ValueError("complete saved numerical results required")
        for row in rows:
            path = Path(row["ledger_path"])
            if sha(path) != row["ledger_sha"]:
                raise ValueError("immutable raw ledger changed")
            with gzip.open(path, "rt") as stream:
                original = json.load(stream)
            result = reviewed(original)
            valid = valid_whole_request_certificate(result)
            unknown = Counter(r["stack_id"] for r in result["records"] if r["role"] == "unknown")
            unresolved = [
                dict(count=n, stack=result["stack_table"][sid]) for sid, n in unknown.most_common()
            ]
            output = root / f"{i}-{row['arm']}.reviewed-ledger.json.gz"
            with gzip.open(output, "xt", compresslevel=1) as stream:
                json.dump(result, stream, separators=(",", ":"), allow_nan=False)
            results.append(
                dict(
                    input=i,
                    arm=row["arm"],
                    original_ledger=str(path),
                    original_sha=sha(path),
                    derived_ledger=str(output),
                    derived_sha=sha(output),
                    valid=valid,
                    blocks=result["blocks_exact_certification"],
                    unknown_stack_classes=len(unknown),
                    unknown_allocations=result["unknown_allocation_generations"],
                    unresolved_stacks=unresolved,
                    initial_GPU_KV_bytes=row["initial_GPU_KV_bytes"],
                    final_GPU_KV_bytes=row["final_GPU_KV_bytes"],
                    whole_request_requested_KV_plus_workspace_peak_bytes=result[
                        "exact_KV_peak_bytes"
                    ],
                    logits_unchanged=True,
                    clean_latency_established=False,
                )
            )
            del result, original
            gc.collect()
    write_once(
        root / "decision.json",
        dict(
            complete=len(results) == 4,
            all_certificates_valid=all(r["valid"] for r in results),
            results=results,
            scope=(
                "conservative requested PyTorch KV/workspace; "
                "not allocator rounding/vendor/process HBM"
            ),
            independent_quality_established=False,
            superiority_established=False,
        ),
    )
    print(json.dumps(dict(root=str(root), valid=[r["valid"] for r in results])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args().root)
