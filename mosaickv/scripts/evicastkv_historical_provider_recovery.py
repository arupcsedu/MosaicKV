#!/usr/bin/env python3
"""Selective hash-only recovery of two failed InfographicVQA LFS groups.

All successful parent file transactions and assets are imported unchanged.
Only two preregistered encoded bodies may exceed the old 40M-pixel resource
limit; their exact dimensions and a 64M hard bound are verified. Canonical
EXIF-RGB semantics are unchanged: no resizing, cropping or row exclusion.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import evicastkv_historical_provider_superset as parent_audit
from evicastkv_freshness_producer_canary import read_bytes, read_json
from evicastkv_review_followups import cpu_gate
from evicastkv_review_freshness_stream import signature

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
MAX_PIXELS, SECONDS, SHARDS = 64_000_000, 240, 2
LARGE_BODIES = {
    "d1cb05d10262efbc62f52ad0c4932a605e49ee12b8b75dd25ef81aee4a9223c8": [9024, 5648],
    "918c702e34d0f78d6fca365f1a3de1daaadfa10d66eafa4d9f1267e524333801": [6250, 9375],
}
FAILED_LFS = {
    "12cbc18ad5a499deebca691fcdf9a852c3d3917e448b30dc604bfdfb4559e756",
    "254fdbd3f69499aca70292167f61d89430fb485b3f6a2fce939ab84f02ae5776",
}
ORIGINAL_IDENTITY = parent_audit.image_identity


def read(path, expected=None):
    return read_json(read_bytes(path, bound=128 * 2**20, expected=expected)[0])


def image_identity(raw):
    if not isinstance(raw, bytes) or not 0 < len(raw) <= 32 * 2**20:
        raise ValueError("unchanged encoded-image resource bound exceeded")
    encoded = hashlib.sha256(raw).hexdigest()
    if encoded not in LARGE_BODIES:
        return ORIGINAL_IDENTITY(raw)
    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(raw)) as image:
        if [image.width, image.height] != LARGE_BODIES[
            encoded
        ] or image.width * image.height > MAX_PIXELS:
            raise ValueError("known large body does not match its exact registered geometry")
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        hashed = hashlib.sha256(f"RGB:{rgb.width}:{rgb.height}:".encode())
        hashed.update(rgb.tobytes())
    return dict(encoded_sha256=encoded, pixel_sha256=hashed.hexdigest())


def prepare(root, *, cpu_gate_root, parent):
    root, parent = Path(root).resolve(), Path(parent).resolve(strict=True)
    if root.exists() or not root.is_relative_to(Path("/scratch/djy8hg").resolve()):
        raise ValueError("new write-once scratch recovery root required")
    revision = clean_source(SOURCE)
    cpu = cpu_gate(Path(cpu_gate_root), revision)
    old = read(parent / "plan.json")
    gate_path = parent / "GATE_historical_provider_superset.json"
    gate = read(gate_path)
    if gate["source_sha"] != old["source_sha"] or gate["missing_shards"]:
        raise ValueError("all source-matched provider worker evidence required")
    bound = {str(parent / "plan.json"): sha(parent / "plan.json"), str(gate_path): sha(gate_path)}
    inventory_path = parent / "typed_asset_inventory.json"
    inventory = read(inventory_path, gate["typed_inventory_sha256"])
    if inventory["worker_bindings"] != gate["worker_bindings"]:
        raise ValueError("parent typed inventory differs from its gated worker bindings")
    bound[str(inventory_path)] = gate["typed_inventory_sha256"]
    observations, originals = {}, []
    if len(gate["worker_bindings"]) != 8:
        raise ValueError("all eight parent provider worker bindings required")
    indices = []
    for name, expected in gate["worker_bindings"].items():
        path = Path(name).resolve(strict=True)
        match = re.fullmatch(r"shard-(\d+)\.json", path.name)
        if path.parent != parent / "workers" or match is None:
            raise ValueError("only exact parent provider worker paths are admissible")
        result = read(name, expected)
        indices.append(int(match.group(1)))
        if result["source_sha"] != old["source_sha"] or result["index"] != indices[-1]:
            raise ValueError("parent provider worker source differs")
        bound[name] = expected
        for row in result["files"]:
            key = (row["path"], row["pool"])
            if key in observations:
                raise ValueError("duplicated parent source observation")
            observations[key] = row
            if not row["complete"]:
                originals.append(row)
    if sorted(indices) != list(range(8)):
        raise ValueError("every parent provider worker index required exactly once")
    expected = {(r["path"], r["pool"]): r for r in old["files"]}
    if set(observations) != set(expected):
        raise ValueError("parent source-file denominator differs")
    retry = []
    for row in originals:
        item = expected[row["path"], row["pool"]]
        if (
            item["expected_sha256"] not in FAILED_LFS
            or item["pool"] != "docvqa/InfographicVQA/validation"
            or row["rows_hashed"] != 0
        ):
            raise ValueError("only the two fully rejected registered file groups may be retried")
        if signature(Path(item["path"])) != item["signature"]:
            raise ValueError("registered failed source changed")
        retry.append(dict(item, index=len(retry)))
    if len(retry) != SHARDS or {r["expected_sha256"] for r in retry} != FAILED_LFS:
        raise ValueError("exact two failed LFS groups required; no successful work is rerun")
    plan = dict(
        source_sha=revision,
        cpu_gate_root=str(cpu_gate_root),
        cpu_gate=cpu,
        parent=str(parent),
        bindings=bound,
        files=retry,
        parent_files=old["files"],
        parent_observations=list(observations.values()),
        original_errors=originals,
        parent_pools=old["pools"],
        parent_unresolved=old["unresolved"],
        route_contradictions=old["route_contradictions"],
        libraries=old["libraries"],
        maximum_file_bytes=old["maximum_file_bytes"],
        seconds_per_worker=SECONDS,
        large_body_geometry=LARGE_BODIES,
        maximum_large_body_pixels=MAX_PIXELS,
        shard_count=SHARDS,
        canonical_RGB_semantics_unchanged=True,
        automatic_resubmission=False,
        fresh_confirmation_authorized=False,
    )
    write_once(root / "plan.json", plan)
    return dict(
        status="prepared_two_failed_provider_groups_only",
        workers=SHARDS,
        retry_rows=sum(row["rows"] for row in retry),
        plan_sha256=sha(root / "plan.json"),
    )


def load(root):
    plan = read(Path(root) / "plan.json")
    clean_source(SOURCE, plan["source_sha"])
    cpu_gate(Path(plan["cpu_gate_root"]), plan["source_sha"])
    if (
        plan["large_body_geometry"] != LARGE_BODIES
        or plan["maximum_large_body_pixels"] != MAX_PIXELS
    ):
        raise ValueError("registered hash-only resource exception changed")
    if any(version(name) != expected for name, expected in plan["libraries"].items()):
        raise ValueError("parent canonical image/Arrow library versions changed")
    for name, expected in plan["bindings"].items():
        read_bytes(name, bound=128 * 2**20, expected=expected)
    return plan


def worker(root, index):
    plan = load(root)
    if type(index) is not int or not 0 <= index < SHARDS:
        raise ValueError("one of two selective CPU workers required")
    # Reuse the parent transaction, image-only projection and exact per-row
    # accounting. The only behavior difference is the explicit body exception.
    with (
        patch.object(parent_audit, "load", lambda _root: plan),
        patch.object(parent_audit, "image_identity", image_identity),
    ):
        return parent_audit.worker(Path(root), index)


def analyze(root):
    root = Path(root).resolve()
    plan = load(root)
    old_inventory = read(Path(plan["parent"]) / "typed_asset_inventory.json")
    assets = {
        row["encoded_sha256"]: dict(row, references=list(row["references"]))
        for row in old_inventory["assets"]
    }
    observations = {(r["path"], r["pool"]): r for r in plan["parent_observations"]}
    expected = {(r["path"], r["pool"]): r for r in plan["files"]}
    missing, bindings = [], {}
    for index in range(SHARDS):
        path = root / "workers" / f"shard-{index}.json"
        if not path.exists():
            missing.append(index)
            continue
        result = read(path)
        if result["source_sha"] != plan["source_sha"] or result["index"] != index:
            raise ValueError("selective provider worker source/index differs")
        bindings[str(path)] = sha(path)
        own = {key: item for key, item in expected.items() if item["index"] == index}
        if {(r["path"], r["pool"]) for r in result["files"]} != set(own) or len(
            result["files"]
        ) != len(own):
            raise ValueError("selective provider worker file denominator differs")
        refs = []
        allowed = set()
        for row in result["files"]:
            key = row["path"], row["pool"]
            item = own[key]
            if row["rows_hashed"] > 0 or row["complete"]:
                if (
                    row["signature"] != item["signature"]
                    or row["whole_file_sha256_before"] != item["expected_sha256"]
                    or row["whole_file_sha256_after"] != item["expected_sha256"]
                    or not 0 <= row["rows_hashed"] <= item["rows"]
                    or (row["complete"] and row["rows_hashed"] != item["rows"])
                ):
                    raise ValueError("selective provider source transaction differs")
                allowed.update(
                    (item["pool"], item["expected_sha256"], i) for i in range(row["rows_hashed"])
                )
            observations[key] = row
        for record in result["assets"]:
            encoded = record["encoded_sha256"]
            if encoded in assets and assets[encoded]["pixel_sha256"] != record["pixel_sha256"]:
                raise ValueError("canonical RGB conflicts with a preserved parent asset")
            target = assets.setdefault(encoded, dict(record, references=[]))
            for ref in record["references"]:
                refs.append((ref["pool"], ref["source_sha256"], ref["source_row"]))
                if ref in target["references"]:
                    raise ValueError("retry duplicates an already preserved source row")
                target["references"].append(ref)
        if len(refs) != len(set(refs)) or set(refs) != allowed:
            raise ValueError("selective image references missing or duplicated")
    pool_rows = {}
    errors = []
    for row in observations.values():
        pool_rows[row["pool"]] = pool_rows.get(row["pool"], 0) + row["rows_hashed"]
        if not row["complete"]:
            errors.append(row)
    coverage = [
        dict(
            pool,
            hashed_rows=pool_rows.get(pool["pool"], 0),
            conservative_media_superset_complete=pool["metadata_pool_complete"]
            and pool_rows.get(pool["pool"], 0) == pool["expected_rows"],
        )
        for pool in plan["parent_pools"]
    ]
    write_once(
        root / "typed_asset_inventory.json",
        dict(
            source_sha=plan["source_sha"],
            assets=list(assets.values()),
            pool_coverage=coverage,
            parent_inventory_sha256=sha(Path(plan["parent"]) / "typed_asset_inventory.json"),
            worker_bindings=bindings,
            historical_parent_errors_preserved=plan["original_errors"],
            unresolved=[*plan["parent_unresolved"], *errors],
            global_historical_media_inventory_complete=False,
            fresh_confirmation_authorized=False,
        ),
    )
    report = dict(
        status="selective_provider_recovery_complete"
        if not (missing or errors or plan["parent_unresolved"])
        else "selective_provider_recovery_incomplete",
        source_sha=plan["source_sha"],
        pool_coverage=coverage,
        missing_shards=missing,
        worker_errors=len(errors),
        unresolved_providers=len(plan["parent_unresolved"]),
        typed_assets=len(assets),
        worker_bindings=bindings,
        original_errors_preserved=True,
        retried_files=len(expected),
        successful_parent_files_rerun=0,
        typed_inventory_sha256=sha(root / "typed_asset_inventory.json"),
        video_exposure_resolved=False,
        fresh_confirmation_authorized=False,
    )
    write_once(root / "GATE_historical_provider_recovery.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "worker", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--cpu-gate-root", type=Path)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.mode == "prepare":
        result = prepare(args.root, cpu_gate_root=args.cpu_gate_root, parent=args.parent)
    elif args.mode == "worker":
        result = worker(args.root, args.index)
    else:
        result = analyze(args.root)
    print(json.dumps(result))
