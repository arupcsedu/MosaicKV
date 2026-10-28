#!/usr/bin/env python3
"""CPU-only content audit of cached official MileBench media; no extraction."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once
from mosaickv.evaluation.fox_milebench_scope import (
    ORIGINAL_TASKS,
    PROJECTED_COLUMNS,
    PROPOSED_TASKS,
    REPOSITORY,
    REVISION,
    safe_relative,
)

SOURCE = Path(__file__).resolve().parents[2]
DATA = Path(f"/scratch/djy8hg/data/mosaickv/milebench-{REVISION}")
CACHE = (
    Path("/scratch/djy8hg/cache/mosaickv/huggingface/hub")
    / "datasets--FreedomIntelligence--MileBench"
    / "snapshots"
    / REVISION
)


def image_identity(raw):
    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(raw)) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        size = rgb.size
        pixel = hashlib.sha256(f"RGB:{size[0]}:{size[1]}:".encode() + rgb.tobytes()).hexdigest()
    return dict(
        encoded_sha256=hashlib.sha256(raw).hexdigest(),
        pixel_sha256=pixel,
        width=size[0],
        height=size[1],
    )


def question_type_only(raw):
    """Inspect the leading metadata object; never JSON-decode the data/answers."""
    match = re.match(rb'\s*\{\s*"meta_data"\s*:\s*\{', raw)
    if not match:
        raise ValueError("official annotation must start with meta_data")
    start, depth, quoted, escaped = match.end() - 1, 0, False, False
    end = None
    for i in range(start, len(raw)):
        char = raw[i]
        if quoted:
            if escaped:
                escaped = False
            elif char == 92:
                escaped = True
            elif char == 34:
                quoted = False
        elif char == 34:
            quoted = True
        elif char == 123:
            depth += 1
        elif char == 125:
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        raise ValueError("unterminated metadata")
    fields = re.findall(rb'"question_type"\s*:\s*"(open-ended|multi-choice)"', raw[start:end])
    if len(fields) != 1:
        raise ValueError("ambiguous or missing official question type")
    return fields[0].decode("ascii")


def scan_archive(path, required):
    """Read members in place. Never trust tar paths, symlinks, or extract files."""
    identities, metadata, errors = {}, {}, []
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            name = member.name.removeprefix("./")
            if name not in required and name not in {f"{t}/{t}.json" for t in PROPOSED_TASKS}:
                continue
            safe_relative(name)
            if not member.isfile() or member.size > 32 * 2**20:
                raise ValueError("requested media member is not a bounded regular file")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("missing requested archive member")
            raw = stream.read()
            if len(raw) != member.size:
                raise ValueError("truncated official archive member")
            if name.endswith(".json"):
                task = name.split("/")[0]
                metadata[task] = dict(
                    question_type=question_type_only(raw), sha256=hashlib.sha256(raw).hexdigest()
                )
            else:
                try:
                    identity = image_identity(raw)
                    if name in identities and identities[name] != identity:
                        raise ValueError("conflicting duplicate media member")
                    identities[name] = identity
                except Exception as e:
                    errors.append(dict(path=name, error=f"{type(e).__name__}: {e}"))
    return dict(
        path=str(path), sha256=sha(path), identities=identities, metadata=metadata, errors=errors
    )


def historical_catalogs():
    from huggingface_hub import HfFileSystem
    from pyarrow.parquet import ParquetFile

    fs, records = HfFileSystem(), []
    aliases = {"CLEVR-Change": "CLEVR_Change", "Spot-the-Diff": "Spot_the_Diff"}
    for task in ORIGINAL_TASKS:
        name = aliases.get(task, task)
        uri = f"datasets/{REPOSITORY}@{REVISION}/preview/{name}_test-00000-of-00001.parquet"
        with fs.open(uri, "rb", cache_type="none") as stream:
            rows = ParquetFile(stream).read(columns=list(PROJECTED_COLUMNS)).to_pylist()
        for r in rows:
            instance = r["task_instance"]
            records.append(
                dict(
                    task=task,
                    sample_id=f"{task}-{r['sample_id']:04d}",
                    source_image_paths=[safe_relative(p) for p in instance["images_path"]],
                    combined_1_images=safe_relative(instance["combined_1_images"][0]),
                )
            )
    return records


def audit(scope, output):
    from evicastkv_evidence import exclusion_inventory

    from mosaickv.evaluation.milebench import MILEBENCH_DATASET_TO_GROUP

    revision = clean_source(SOURCE)
    inventory = exclusion_inventory(
        Path("/scratch/djy8hg/runs/evicastkv/fox-review-20261028-v4"),
        output / "history_all_official_tasks.json",
        identity_tasks=tuple(MILEBENCH_DATASET_TO_GROUP),
    )
    consumed_ids = set(inventory["MileBench_consumed_ids"])
    availability = json.loads((scope / "availability.json").read_text())
    proposed = []
    bindings = {str(scope / "availability.json"): sha(scope / "availability.json")}
    for task, binding in availability["catalogs"].items():
        path = Path(binding["path"])
        if sha(path) != binding["sha256"]:
            raise ValueError("input catalog changed")
        bindings[str(path)] = sha(path)
        proposed.extend(dict(row, task=task) for row in json.loads(path.read_text())["rows"])
    history = historical_catalogs()
    all_rows = proposed + history
    required = set()
    for row in all_rows:
        required.update(f"{row['task']}/images/{p}" for p in row["source_image_paths"])
        required.add(f"{row['task']}/combined_1_images/{row['combined_1_images']}")
    identities, errors, metadata = {}, [], {}
    # Reuse already extracted history; do not create a duplicate media cache.
    for name in sorted(required):
        path = DATA / name
        if path.is_file():
            identities[name] = image_identity(path.read_bytes())
    archives = sorted(CACHE.glob("MileBench_part*.tar.gz"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda p: scan_archive(p, required - identities.keys()), archives))
    for result in results:
        cached_digest = Path(result["path"]).resolve(strict=True).name
        if len(cached_digest) != 64 or result["sha256"] != cached_digest:
            raise ValueError("official cached archive failed its content-addressed SHA check")
        bindings[result["path"]] = result["sha256"]
        errors.extend(result["errors"])
        for name, identity in result["identities"].items():
            if name in identities and identities[name] != identity:
                raise ValueError("conflicting media across official archives")
            identities[name] = identity
        metadata.update(result["metadata"])
    history_fingerprints = set()
    for row in history:
        names = [f"{row['task']}/images/{p}" for p in row["source_image_paths"]]
        names += [f"{row['task']}/combined_1_images/{row['combined_1_images']}"]
        for name in names:
            if name in identities:
                history_fingerprints.update(
                    (identities[name]["encoded_sha256"], identities[name]["pixel_sha256"])
                )
    summaries = []
    sample_audit = []
    for task in PROPOSED_TASKS:
        rows = [r for r in proposed if r["task"] == task]
        for row in rows:
            names = [f"{task}/images/{p}" for p in row["source_image_paths"]]
            names += [f"{task}/combined_1_images/{row['combined_1_images']}"]
            missing = [n for n in names if n not in identities]
            overlapping = [
                n
                for n in names
                if n in identities
                and any(
                    identities[n][k] in history_fingerprints
                    for k in ("encoded_sha256", "pixel_sha256")
                )
            ]
            sample_audit.append(
                dict(
                    sample_id=row["sample_id"],
                    task=task,
                    known_consumed_identity=row["sample_id"] in consumed_ids,
                    missing_assets=missing,
                    overlapping_historical_assets=overlapping,
                    all_assets_verified=not missing,
                    source_group_metadata_verified=False,
                )
            )
        subset = [r for r in sample_audit if r["task"] == task]
        summaries.append(
            dict(
                task=task,
                total=len(rows),
                complete_media_samples=sum(r["all_assets_verified"] for r in subset),
                historical_pixel_or_byte_overlap_samples=sum(
                    bool(r["overlapping_historical_assets"]) for r in subset
                ),
                observed_disjoint_complete_samples=sum(
                    r["all_assets_verified"]
                    and not r["overlapping_historical_assets"]
                    and not r["known_consumed_identity"]
                    for r in subset
                ),
                known_consumed_ids=sum(r["known_consumed_identity"] for r in subset),
                official_question_type_verified=metadata.get(task, {}).get("question_type")
                == PROPOSED_TASKS[task],
            )
        )
    write_once(output / "media_identities.json", dict(identities=identities, bound_files=bindings))
    write_once(output / "sample_audit.json", sample_audit)
    write_once(
        output / "report.json",
        dict(
            source_sha=revision,
            scope_sha256=sha(scope / "availability.json"),
            tasks=summaries,
            observed_media_audit_completed=True,
            no_images_extracted_or_downloaded=True,
            original_four_entire_pools_excluded_conservatively=True,
            errors=errors,
            historical_asset_missing_count=sum(
                n not in identities
                for r in history
                for n in [
                    *(f"{r['task']}/images/{p}" for p in r["source_image_paths"]),
                    f"{r['task']}/combined_1_images/{r['combined_1_images']}",
                ]
            ),
            available_archive_parts=[p.name for p in archives],
            source_video_document_identity_complete=False,
            exhaustive_historical_media_inventory_complete=False,
            historical_task_identity_taxonomy="all_29_official_tasks_not_only_the_original_four",
            unresolved_history_roots=inventory["unclassified_workload_roots"],
            missing_or_unverified_media_not_treated_as_fresh=True,
            fresh_confirmation_certified=False,
            confirmation_authorized=False,
            blockers=[
                "global historical/source video-document identities remain unverified",
                "missing cache archive parts cannot be silently ignored",
            ],
            media_identity_content_sha256=digest(identities),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("media audit output must be new")
    for p in (args.scope, args.output):
        if not p.resolve().is_relative_to(Path("/scratch/djy8hg").resolve()):
            raise ValueError("scratch-only media audit")
    audit(args.scope, args.output)
