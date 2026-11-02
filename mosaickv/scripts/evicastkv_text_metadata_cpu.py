#!/usr/bin/env python3
"""Consumed CPU-only metadata microbenchmark; no model weights/answer scoring."""

import argparse
import ast
import hashlib
import json
import random
import statistics
import subprocess
import time
from pathlib import Path

import evicastkv_common_streaming as common

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

ROOT = Path("/scratch/djy8hg/runs/evicastkv")
FROZEN = "007fb3bbc43a19fac651d444e2361a5d808a3002"


def run(root):
    from huggingface_hub import snapshot_download

    from mosaickv.evaluation.protected_projection_source import (
        metadata_adapter,
        original_text_positions,
    )
    from mosaickv.evaluation.text_source_storage import text_slots

    if root.exists():
        raise FileExistsError("new immutable CPU study root required")
    source_sha = clean_source(common.SOURCE)
    source = subprocess.check_output(
        ["git", "show", f"{FROZEN}:mosaickv/src/mosaickv/evaluation/text_source_storage.py"],
        cwd=common.SOURCE,
        text=True,
    )
    tree = ast.parse(source)
    fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "text_slots"
    )
    namespace = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<pinned-text-slots>", "exec"), namespace)
    legacy = namespace["text_slots"]
    parent = ROOT / "optimized-source-request-timing-consumed-20261101-retry2"
    plan = common.previous.read(parent / "plan.json")
    jobs = common.previous.read(common.AUDIT / "plan.json")["jobs"]
    checkpoint = snapshot_download(
        "llava-hf/llava-1.5-7b-hf",
        revision="b234b804b114d9e37bb655e11cbbb5f5e971b7a9",
        local_files_only=True,
    )
    adapter = metadata_adapter(checkpoint)
    results, bindings = [], {str(parent / "plan.json"): sha(parent / "plan.json")}
    for item in plan["jobs"]:
        source_payload = common.previous.read(
            common.AUDIT / f"workers/{item['reference_worker']}/offline_payload.json"
        )
        n = max(source_payload["memberships"][0][0]) + 1
        text, token_hash = original_text_positions(
            adapter,
            common.previous.load_messages(common.previous.read(item["inputs"][0])),
            expected_length=n,
        )
        for job in [j for j in jobs if j["input_index"] == item["input_index"]]:
            path = common.AUDIT / f"workers/{job['index']}/offline_payload.json"
            bindings[str(path)] = sha(path)
            heads = tuple(
                tuple(tuple(h) for h in layer)
                for layer in common.previous.read(path)["memberships"]
            )
            expected = tuple(legacy(layer, text) for layer in heads)
            expected_mask = tuple(
                tuple(tuple(p in text for p in head) for head in layer) for layer in heads
            )
            trials = []
            rng = random.Random(plan["seed"] + job["index"])
            for trial in range(-2, 5):
                variants = ["legacy", "once"]
                rng.shuffle(variants)
                for variant in variants:
                    function = legacy if variant == "legacy" else text_slots
                    started = time.monotonic()
                    slots = tuple(function(layer, text) for layer in heads)
                    slot_seconds = time.monotonic() - started
                    started = time.monotonic()
                    mask_text = text if variant == "legacy" else frozenset(text)
                    mask = tuple(
                        tuple(tuple(p in mask_text for p in head) for head in layer)
                        for layer in heads
                    )
                    mask_seconds = time.monotonic() - started
                    if slots != expected or mask != expected_mask:
                        raise RuntimeError("metadata changed a selected slot or mask bit")
                    trials.append(
                        dict(
                            trial=trial,
                            warmup=trial < 0,
                            variant=variant,
                            slot_seconds=slot_seconds,
                            mask_seconds=mask_seconds,
                        )
                    )
            means = {
                variant: {
                    key: statistics.mean(
                        t[key] for t in trials if not t["warmup"] and t["variant"] == variant
                    )
                    for key in ("slot_seconds", "mask_seconds")
                }
                for variant in ("legacy", "once")
            }
            results.append(
                dict(
                    input_index=job["input_index"],
                    method=common.method_of(job),
                    tokenization_sha256=token_hash,
                    sample_id=common.previous.read(item["inputs"][0])["sample_id"],
                    text_count=len(text),
                    all_slots_and_masks_identical=True,
                    means=means,
                    trials=trials,
                    parent_manifest_path=str(
                        common.AUDIT / f"workers/{job['index']}/manifest.json"
                    ),
                )
            )
    write_once(
        root / "decision.json",
        dict(
            source_sha=source_sha,
            legacy_source_sha=FROZEN,
            legacy_source_file_sha256=hashlib.sha256(source.encode()).hexdigest(),
            bindings=bindings,
            cases=results,
            complete=len(results) == 10,
            measurement_type="consumed_CPU_metadata_microbenchmark",
            no_GPU_or_task_score_claim=True,
            independent_quality_N=0,
            environment=common.control.environment(),
        ),
    )
    print(json.dumps({"cases": len(results), "complete": True}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args().root)
