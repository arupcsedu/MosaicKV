#!/usr/bin/env python3
"""Finite COCO ID/media audit, never cohort selection or a freshness license.

The registered optional hash stage reads only cached image/identity Arrow
columns. It emits encoded and canonical RGB hashes, not images or labels.
Unobserved transformations and source-group aliases remain explicitly unknown.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import time
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

from evicastkv_freshness_frontier_fork import canonical_bindings
from evicastkv_freshness_producer_canary import read_bytes, read_json
from evicastkv_review_followups import cpu_gate
from evicastkv_review_freshness_stream import identity_projection, signature

from mosaickv.evaluation.coco_shared_bytes import COCO_REVISION
from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
DATASET = "lmms-lab/COCO-Caption"
CACHE = Path("/scratch/djy8hg/cache/mosaickv/datasets/lmms-lab___coco-caption/default/0.0.0")
AVAILABILITY = Path(
    "/scratch/djy8hg/runs/evicastkv/fox-review-20261028-writing-closure-v2/"
    "coco_availability/report.json"
)
AVAILABILITY_SHA = "154c5f7cb9913f933c49a78c0c43d171b4b4189260f0286f23e63e35f47b5a3f"
MEDIA = Path(
    "/scratch/djy8hg/runs/evicastkv/fox-review-20261028-milebench-media-content-v1/"
    "media_identities.json"
)
MEDIA_SHA = "9a5c43e1ea09fd5522e9c190e0d7ac9527146eaa14546758c982069c15462d0e"
COCO = re.compile(r"COCO_val2014_\d{12}\.jpg")
HASH = re.compile(r"[a-f0-9]{64}")
SHARDS, WORK_SECONDS, MAX_IMAGE, MAX_PIXELS = 8, 240, 32 * 2**20, 40_000_000
ID_COLUMNS = ("question_id", "file_name", "id")
HASH_COLUMNS = (*ID_COLUMNS, "image")


def metadata(path, expected=None, *, deadline=None):
    raw, actual = read_bytes(path, bound=128 * 2**20, expected=expected)
    projected = identity_projection(
        raw, budget=SimpleNamespace(deadline=deadline or time.monotonic() + 60)
    )
    return projected, actual


def image_identity(raw):
    from PIL import Image, ImageOps

    if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_IMAGE:
        raise ValueError("embedded image bytes unavailable or outside registered bound")
    with Image.open(io.BytesIO(raw)) as image:
        if image.width * image.height > MAX_PIXELS:
            raise ValueError("image pixels exceed registered hash-only decode bound")
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        prefix = f"RGB:{rgb.width}:{rgb.height}:".encode()
        pixel = hashlib.sha256(prefix + rgb.tobytes()).hexdigest()
    return dict(encoded_sha256=hashlib.sha256(raw).hexdigest(), pixel_sha256=pixel)


def canonical_id(row):
    value = row["file_name"]
    if not isinstance(value, str) or not COCO.fullmatch(value) or row["question_id"] != value:
        raise ValueError("canonical COCO question_id and file_name disagree")
    return value  # Auxiliary id is a row index, never a source-image identity.


def historical_inventory(path, expected=None):
    """Only the scanner's hash-bound metadata export, never a model result."""
    raw, actual = read_bytes(path, bound=128 * 2**20, expected=expected)
    result = read_json(raw)
    required = {
        "sample_ids",
        "coco_ids",
        "media_hashes",
        "source_group_ids",
        "scopes",
        "scoped_identities",
    }
    if set(result["identities"]) != required:
        raise ValueError("historical metadata inventory identity schema differs")
    if any(not isinstance(v, str) for v in result["identities"]["coco_ids"]):
        raise ValueError("historical COCO identity is not a canonical string")
    return result, actual


def arrow_rows(path, *, images=False):
    """Convert only declared columns; never touch question/caption/answer arrays."""
    import pyarrow as pa

    columns = HASH_COLUMNS if images else ID_COLUMNS
    with pa.memory_map(str(path), "r") as stream:
        reader = pa.ipc.open_stream(stream)
        if not set(columns) <= set(reader.schema.names):
            raise ValueError("pinned Arrow identity/image schema differs")
        for batch in reader:
            yield from batch.select(list(columns)).to_pylist()


def file_sha(path, deadline):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("finite source-hash deadline reached")
            chunk = stream.read(2**20)
            if not chunk:
                return value.hexdigest()
            value.update(chunk)


def prepare(
    root,
    *,
    cpu_gate_root,
    history_root,
    history_gate_filename="GATE_freshness_frontier_fork.json",
    hash_media=False,
    availability=AVAILABILITY,
    availability_sha256=AVAILABILITY_SHA,
    media=MEDIA,
    media_sha256=MEDIA_SHA,
    cache=CACHE / COCO_REVISION,
):
    root, history_root, cache = map(lambda p: Path(p).resolve(), (root, history_root, cache))
    if root.exists() or not root.is_relative_to(Path("/scratch/djy8hg").resolve()):
        raise ValueError("new write-once scratch audit root required")
    revision = clean_source(SOURCE)
    cpu = cpu_gate(cpu_gate_root, revision)
    available = read_json(read_bytes(availability, expected=availability_sha256)[0])
    if (available["dataset_id"], available["dataset_revision"], available["split"]) != (
        DATASET,
        COCO_REVISION,
        "val",
    ):
        raise ValueError("reviewed dataset route/revision/split differs")
    if history_gate_filename not in {
        "GATE_freshness_frontier_fork.json",
        "GATE_freshness_empty_parquet_frontier.json",
        "GATE_freshness_yaml_events.json",
    }:
        raise ValueError("reviewed history gate filename required")
    history_gate = history_root / history_gate_filename
    history_inventory = history_root / "combined_identity_inventory.json"
    gate = read_json(read_bytes(history_gate)[0])
    history, history_sha = historical_inventory(
        history_inventory, gate["identity_inventory_sha256"]
    )
    typed, typed_sha = metadata(media, media_sha256)
    arrow_names = {f"coco-caption-val-{i:05d}-of-00014.arrow" for i in range(14)}
    original_bindings = canonical_bindings(
        {
            name: value
            for name, value in available["bound_files"].items()
            if Path(name).name in arrow_names
        }
    )
    files, all_ids = [], []
    for index in range(14):
        path = (cache / f"coco-caption-val-{index:05d}-of-00014.arrow").resolve(strict=True)
        expected = original_bindings.get(str(path))
        if expected is None:
            raise ValueError("Arrow shard lacks reviewed pinned SHA256")
        before = signature(path)
        ids = [canonical_id(row) for row in arrow_rows(path)]
        if signature(path) != before:
            raise ValueError("cached Arrow changed during ID-only projection")
        files.append(
            dict(
                path=str(path),
                sha256=expected,
                signature=before,
                rows=len(ids),
                id_sha256=digest(ids),
            )
        )
        all_ids.extend(ids)
    if len(all_ids) != len(set(all_ids)) or len(all_ids) != available["validation_image_ids"]:
        raise ValueError("pinned ID-only cohort count or uniqueness differs")
    old_ids = set(history["identities"]["coco_ids"])
    plan = dict(
        source_sha=revision,
        measurement_type="metadata_only_availability_audit",
        libraries={name: version(name) for name in ("Pillow", "pyarrow")},
        common_lock_sha256=sha(SOURCE / "mosaickv/env/common/requirements.lock"),
        cpu_gate_root=str(cpu_gate_root),
        cpu_gate=cpu,
        dataset_id=DATASET,
        dataset_revision=COCO_REVISION,
        split="val",
        files=files,
        bindings={
            str(availability): availability_sha256,
            str(media): typed_sha,
            str(history_gate): sha(history_gate),
            str(history_inventory): history_sha,
        },
        history_root=str(history_root),
        history_gate_filename=history_gate_filename,
        typed_media_path=str(media),
        cached_validation_ids=len(all_ids),
        cached_full_file_hashes_reverified_during_optional_workers=False,
        projected_source_ids_sha256=digest(all_ids),
        observed_historical_exact_id_overlap=len(set(all_ids) & old_ids),
        exact_ids_not_observed_in_incomplete_history=len(set(all_ids) - old_ids),
        not_observed_ids_digest=digest(sorted(set(all_ids) - old_ids)),
        historical_frontier_exhausted=gate["frontier_exhausted"],
        historical_coverage_errors=gate["original_and_new_coverage_errors"],
        existing_typed_media_entries=len(typed.get("identities", {})),
        hash_only_image_decode_registered=hash_media is True,
        maximum_worker_seconds=WORK_SECONDS,
        maximum_image_bytes=MAX_IMAGE,
        maximum_image_pixels=MAX_PIXELS,
        shard_count=SHARDS,
        maximum_parallel_CPUs=8,
        image_copies_persisted=0,
        cohort_selected=False,
        materialization_authorized=False,
        fresh_confirmation_authorized=False,
    )
    write_once(root / "plan.json", plan)
    return dict(status="prepared_metadata_audit_only", plan_sha256=sha(root / "plan.json"))


run = prepare


def load(root):
    plan = read_json(read_bytes(Path(root) / "plan.json")[0])
    clean_source(SOURCE, plan["source_sha"])
    cpu_gate(Path(plan["cpu_gate_root"]), plan["source_sha"])
    for name, expected in plan["bindings"].items():
        if sha(Path(name)) != expected:
            raise ValueError("immutable availability-audit evidence changed")
    return plan


def worker(root, index):
    root = Path(root).resolve()
    plan = load(root)
    if (
        not plan["hash_only_image_decode_registered"]
        or type(index) is not int
        or not 0 <= index < 8
    ):
        raise ValueError("explicit hash-only decode registration and valid CPU shard required")
    write_once(root / "locks" / f"worker-{index}.json", dict(job_id=os.getenv("SLURM_JOB_ID")))
    deadline = time.monotonic() + plan["maximum_worker_seconds"]
    records, files = [], []
    for number, binding in enumerate(plan["files"]):
        if number % SHARDS != index:
            continue
        path, current = Path(binding["path"]), []
        observation = dict(path=str(path), expected_rows=binding["rows"], complete=False)
        try:
            observation["signature_before"] = signature(path)
            observation["whole_file_sha256_before"] = file_sha(path, deadline)
            if (
                observation["signature_before"] != binding["signature"]
                or observation["whole_file_sha256_before"] != binding["sha256"]
            ):
                raise ValueError("pinned Arrow signature or full-file SHA differs before decoding")
            ids = []
            for row_index, row in enumerate(arrow_rows(path, images=True)):
                if time.monotonic() >= deadline:
                    break
                sid = canonical_id(row)
                ids.append(sid)
                record = dict(sample_id=sid, source_shard_index=number, source_row_index=row_index)
                try:
                    record.update(image_identity(row["image"]["bytes"]))
                except (OSError, ValueError, TypeError, KeyError) as exc:
                    record["hash_error"] = f"{type(exc).__name__}: {exc}"
                current.append(record)
            # All partial rows remain reusable only after source immutability is
            # reverified. Thirty seconds is reserved for this final hash/check.
            observation["whole_file_sha256_after"] = file_sha(path, deadline + 30)
            observation["signature_after"] = signature(path)
            if (
                observation["whole_file_sha256_after"] != binding["sha256"]
                or observation["signature_after"] != binding["signature"]
            ):
                raise ValueError("pinned Arrow differs after hash-only decoding")
            complete = len(ids) == binding["rows"]
            if complete and digest(ids) != binding["id_sha256"]:
                raise ValueError("hash-stage source identity order differs from ID projection")
            records.extend(current)
            observation.update(complete=complete, verified_rows=len(current), next_row=len(current))
        except (OSError, ValueError, TimeoutError) as exc:
            observation["error"] = f"{type(exc).__name__}: {exc}"
        files.append(observation)
    output = dict(
        source_sha=plan["source_sha"],
        index=index,
        files=files,
        records=records,
        complete=all(f["complete"] for f in files),
        no_image_copies_or_labels_exported=True,
        fresh_confirmation_authorized=False,
    )
    path = root / "workers" / f"shard-{index}.json"
    write_once(path, output)
    return dict(index=index, complete=output["complete"], rows=len(records), sha256=sha(path))


def analyze(root, *, historical_media_root=None):
    root = Path(root).resolve()
    plan = load(root)
    history, _ = historical_inventory(
        Path(plan["history_root"]) / "combined_identity_inventory.json"
    )
    media_path = Path(plan["typed_media_path"])
    typed, _ = metadata(media_path)
    hashes = {kind: set() for kind in ("encoded_sha256", "pixel_sha256")}
    for row in typed.get("identities", {}).values():
        for kind in hashes:
            value = row.get(kind)
            if not isinstance(value, str) or not HASH.fullmatch(value):
                raise ValueError("typed historical media digest missing or malformed")
            hashes[kind].add(value)
    historical_status, historical_bindings, historical_unresolved = None, {}, None
    if historical_media_root is not None:
        historical_media_root = Path(historical_media_root).resolve()
        gate_path = historical_media_root / "GATE_historical_media_hashes.json"
        gate = read_json(read_bytes(gate_path)[0])
        if gate["source_sha"] != plan["source_sha"]:
            raise ValueError(
                "historical media audit source differs from registered availability source"
            )
        typed_path = historical_media_root / "typed_asset_inventory.json"
        extra = read_json(
            read_bytes(typed_path, bound=128 * 2**20, expected=gate["typed_inventory_sha256"])[0]
        )
        if extra.get("source_sha") != plan["source_sha"]:
            raise ValueError("historical typed asset producer differs from its registered gate")
        for record in extra["assets"]:
            for kind in hashes:
                value = record.get(kind)
                if not isinstance(value, str) or not HASH.fullmatch(value):
                    raise ValueError("historical typed asset identity is malformed")
                hashes[kind].add(value)
        historical_status = gate["status"]
        historical_unresolved = len(extra["unresolved"])
        historical_bindings = {
            str(gate_path): sha(gate_path),
            str(typed_path): gate["typed_inventory_sha256"],
        }
    records, missing, incomplete, bindings = [], [], [], {}
    for index in range(SHARDS):
        path = root / "workers" / f"shard-{index}.json"
        if not path.is_file():
            missing.append(index)
            continue
        result = read_json(read_bytes(path)[0])
        if result["source_sha"] != plan["source_sha"] or result["index"] != index:
            raise ValueError("hash worker source/index differs")
        if not result["complete"]:
            incomplete.append(index)
        if any(r["source_shard_index"] % SHARDS != index for r in result["records"]):
            raise ValueError("media-hash row belongs to a different registered shard")
        bindings[str(path)] = sha(path)
        records.extend(result["records"])
    ids = [r["sample_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate hash-stage COCO source identities")
    exact = set(history["identities"]["coco_ids"])
    untyped = set(history["identities"]["media_hashes"])
    usable = [r for r in records if "hash_error" not in r]
    observations = {}
    for kind in hashes:
        observations[kind] = sum(r[kind] in hashes[kind] for r in usable)
    report = dict(
        status="bounded_ID_and_media_availability_freshness_not_certified",
        source_sha=plan["source_sha"],
        dataset_id=DATASET,
        dataset_revision=COCO_REVISION,
        split="val",
        cached_validation_ids=plan["cached_validation_ids"],
        historical_frontier_exhausted=plan["historical_frontier_exhausted"],
        historical_coverage_errors=plan["historical_coverage_errors"],
        exact_ids_not_observed_in_incomplete_history=plan[
            "exact_ids_not_observed_in_incomplete_history"
        ],
        missing_worker_shards=missing,
        incomplete_worker_shards=incomplete,
        hash_rows=len(records),
        successfully_hashed_media=len(usable),
        missing_or_failed_media_hashes=plan["cached_validation_ids"] - len(usable),
        observed_encoded_or_pixel_alias_overlap=observations,
        media_comparator_scope="existing_audited_media_conservative_not_inference_only",
        historical_untyped_media_hashes=len(untyped),
        historical_media_audit_status=historical_status,
        historical_media_unresolved_entries=historical_unresolved,
        historical_media_bindings=historical_bindings,
        candidate_internal_pixel_duplicate_records=len(usable)
        - len({r["pixel_sha256"] for r in usable}),
        untyped_historical_hash_possible_overlap=sum(
            any(r[k] in untyped for k in hashes) for r in usable
        ),
        not_observed_ID_and_hash_overlap_count=sum(
            r["sample_id"] not in exact
            and all(r[k] not in hashes[k] and r[k] not in untyped for k in hashes)
            for r in usable
        ),
        remaining_transform_and_global_source_group_aliases_unknown=True,
        global_historical_media_inventory_complete=False,
        observed_hash_absence_is_not_certified_disjointness=True,
        source_pixels_EXIF_transposed_RGB_shape_prefixed=True,
        worker_bindings=bindings,
        cohort_selected=False,
        materialization_authorized=False,
        fresh_confirmation_authorized=False,
        confirmations_submitted=0,
    )
    write_once(root / "GATE_coco_media_availability.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "worker", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--history-root", type=Path)
    parser.add_argument("--history-gate-filename", default="GATE_freshness_frontier_fork.json")
    parser.add_argument("--historical-media-root", type=Path)
    parser.add_argument("--cpu-gate-root", type=Path)
    parser.add_argument("--hash-media", action="store_true")
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.mode == "prepare":
        result = prepare(
            args.root,
            cpu_gate_root=args.cpu_gate_root,
            history_root=args.history_root,
            history_gate_filename=args.history_gate_filename,
            hash_media=args.hash_media,
        )
    elif args.mode == "worker":
        result = worker(args.root, args.index)
    else:
        result = analyze(args.root, historical_media_root=args.historical_media_root)
    print(json.dumps(result))
