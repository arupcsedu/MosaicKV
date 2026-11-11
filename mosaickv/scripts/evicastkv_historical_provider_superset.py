#!/usr/bin/env python3
"""Finite cached-image supersets, not legacy sample-route parity or freshness.

Dataset cards are read as bounded parser events; Parquet projection selects
only ID columns during preparation and only image during hash workers. Entire
pinned pools are charged when legacy row/split mapping is ambiguous. Missing
configurations, video exposure and partial workers remain explicit gaps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import evicastkv_freshness_yaml_events as yaml_events
from evicastkv_coco_media_availability import file_sha, image_identity, metadata
from evicastkv_freshness_producer_canary import read_bytes, read_json
from evicastkv_historical_media_hashes import RECOVERY, recovered_ledgers
from evicastkv_review_followups import cpu_gate
from evicastkv_review_freshness_stream import signature

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
HUB = Path("/scratch/djy8hg/cache/mosaickv/huggingface/hub")
SHARDS, SECONDS, MAX_FILES = 8, 210, 32
HASH = re.compile(r"[a-f0-9]{64}")
PROVIDERS = {
    "mmstar": dict(
        repository="Lin-Chen/MMStar",
        revision="bc98d668301da7b14f648724866e57302778ab27",
        card_git_oid="c80033654d246116cb2631e7dc219c25bb7a8589",
        configs=("val",),
        pools=(("val", "val"),),
        mode="entire_pinned_repository_image_pool",
        id_column="index",
    ),
    "textvqa": dict(
        repository="lmms-lab/textvqa",
        revision="9c0699cd19768ac5ab97568f6b3cbac4c0062884",
        card_git_oid="d1702416fd67874a5641e7424b6c9cf126c08d45",
        configs=("default",),
        pools=(("default", "validation"),),
        mode="entire_declared_split",
        id_column="question_id",
    ),
    "docvqa": dict(
        repository="lmms-lab/DocVQA",
        revision="539088ef8a8ada01ac8e2e6d4e372586748a265e",
        card_git_oid="67f107df5895fe445cccf053d7c299d90add0610",
        configs=("DocVQA", "InfographicVQA"),
        pools=(("DocVQA", "validation"), ("InfographicVQA", "validation")),
        mode="all_validation_configurations_no_legacy_config_guess",
        id_column="questionId",
    ),
    "chartqa": dict(
        repository="lmms-lab/ChartQA",
        revision="9e63b7df1592a1c2158e735cc1725454aef0d6d9",
        card_git_oid="21f4be8932d4f722c0d2efbaa74d79cd46a82152",
        configs=("default",),
        pools=(("default", "test"),),
        mode="entire_declared_split_no_row_ID_assertion",
        id_column=None,
    ),
}


def read(path, expected=None):
    return read_json(read_bytes(path, bound=128 * 2**20, expected=expected)[0])


def card(path, expected_git_oid=None):
    """No YAML constructors; num_examples is metadata, not a benchmark score."""
    raw, actual = read_bytes(path, bound=256 * 2**10)
    if expected_git_oid is not None:
        git_oid = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if git_oid != expected_git_oid:
            raise ValueError("pinned dataset-card Git object differs")
    if not raw.startswith(b"---\n"):
        raise ValueError("pinned dataset card lacks a metadata header")
    end = raw.find(b"\n---", 4)
    if end < 0:
        raise ValueError("dataset card metadata header is incomplete")
    original = yaml_events.safe_key
    with patch.object(yaml_events, "safe_key", lambda key: original(key) or key == "num_examples"):
        projected = yaml_events.projection(raw[4:end], deadline=time.monotonic() + 30)
    return projected, actual


def pool_description(info, config, split):
    configs = info.get("configs", [])
    match = [row for row in configs if row.get("config_name") == config]
    if len(match) != 1:
        raise ValueError("declared repository configuration is ambiguous")
    paths = [row["path"] for row in match[0]["data_files"] if row["split"] == split]
    if not paths:
        raise ValueError("declared split has no pinned data-file pattern")
    datasets = info.get("dataset_info")
    if isinstance(datasets, dict):
        datasets = [dict(datasets, config_name="default")]
    entries = [row for row in datasets if row.get("config_name") == config]
    if len(entries) != 1:
        raise ValueError("dataset-card configuration row counts are ambiguous")
    counts = [
        row["num_examples"] for row in entries[0].get("splits", []) if row.get("name") == split
    ]
    if len(counts) != 1 or type(counts[0]) is not int or counts[0] < 1:
        raise ValueError("positive exact split count missing from pinned dataset card")
    return paths, counts[0]


def cached_files(snapshot, patterns):
    # /scratch is a symlink to /sfs/weka/scratch on the cluster. Compare two
    # canonical roots, never a canonical LFS target with a lexical /scratch
    # prefix. Keep dataset/revision ownership exact; this is not permission
    # to follow a snapshot, repository or blobs directory into another repo.
    snapshot = Path(snapshot)
    repository_name, revision = snapshot.parent.parent.name, snapshot.name
    repository = snapshot.parent.parent.resolve(strict=True)
    actual_snapshot = snapshot.resolve(strict=True)
    if (
        snapshot.parent.name != "snapshots"
        or repository.parent != HUB.resolve(strict=True)
        or repository.name != repository_name
        or actual_snapshot != repository / "snapshots" / revision
    ):
        raise ValueError("pinned snapshot redirects outside its exact dataset/revision")
    snapshot = actual_snapshot
    result = {}
    for pattern in patterns:
        relative = Path(pattern)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or "**" in pattern
            or any("*" in part or "?" in part for part in relative.parts[:-1])
        ):
            raise ValueError("only finite one-directory pinned file patterns are supported")
        for path in (snapshot / relative.parent).glob(relative.name):
            if len(result) >= MAX_FILES:
                raise ValueError("pinned pool exceeds finite cache-file limit")
            actual = path.resolve(strict=True)
            if path.suffix != ".parquet" or actual.parent != repository / "blobs":
                raise ValueError("pinned cache file escapes its dataset repository")
            if not HASH.fullmatch(actual.name):
                raise ValueError("cached Parquet source lacks a content-addressed SHA256")
            result[str(actual)] = dict(
                path=str(actual),
                expected_sha256=actual.name,
                snapshot_alias=str(path),
                signature=signature(actual),
            )
    return list(result.values())


def prepare(root, *, cpu_gate_root=None):
    import pyarrow.parquet as pq

    root = Path(root).resolve()
    if root.exists() or not root.is_relative_to(Path("/scratch/djy8hg").resolve()):
        raise ValueError("new write-once scratch conservative-media root required")
    revision = clean_source(SOURCE)
    cpu_root = Path(cpu_gate_root) if cpu_gate_root else root.parent
    cpu = cpu_gate(cpu_root, revision)
    ledgers, bound = recovered_ledgers(RECOVERY)
    records = {}
    for path, expected in ledgers:
        value, actual = metadata(path, expected)
        bound[str(path)] = actual
        records[value["task"]] = dict(value, path=str(path), sha256=actual)
    if not set(PROVIDERS) <= set(records):
        raise ValueError("all four bound provider ID ledgers required")
    pools, files, unresolved, contradictions = [], [], [], []
    for task, spec in PROVIDERS.items():
        ledger = records[task]
        if (ledger["dataset_id"], ledger["dataset_revision"]) != (
            spec["repository"],
            spec["revision"],
        ):
            raise ValueError("historical provider dataset/revision differs from registered cache")
        snapshot = (
            HUB
            / f"datasets--{spec['repository'].replace('/', '--')}"
            / "snapshots"
            / spec["revision"]
        )
        info, card_sha = card(snapshot / "README.md", spec["card_git_oid"])
        bound[str(snapshot / "README.md")] = card_sha
        if {row["config_name"] for row in info["configs"]} != set(spec["configs"]):
            raise ValueError("pinned repository has unregistered possible configurations")
        if ledger["split"] not in {split for _config, split in spec["pools"]}:
            contradictions.append(
                dict(
                    task=task,
                    ledger_split=ledger["split"],
                    audited_pools=[list(p) for p in spec["pools"]],
                    legacy_split_equivalence_asserted=False,
                )
            )
        for config, split in spec["pools"]:
            patterns, expected = pool_description(info, config, split)
            pool = f"{task}/{config}/{split}"
            found = cached_files(snapshot, patterns)
            rows = 0
            ids = []
            for item in found:
                table = pq.ParquetFile(item["path"])
                if "image" not in table.schema_arrow.names:
                    raise ValueError("pinned image field missing")
                count = table.metadata.num_rows
                rows += count
                if spec["id_column"]:
                    if spec["id_column"] not in table.schema_arrow.names:
                        raise ValueError("pinned ID column missing")
                    for batch in table.iter_batches(columns=[spec["id_column"]], batch_size=512):
                        ids.extend(f"{task}:{r[spec['id_column']]}" for r in batch.to_pylist())
                files.append(
                    dict(
                        item,
                        pool=pool,
                        rows=count,
                        schema_sha256=digest(str(table.schema_arrow)),
                        index=len(files) % SHARDS,
                    )
                )
            complete = rows == expected and bool(found)
            if not complete:
                unresolved.append(
                    dict(
                        pool=pool,
                        reason="pinned_pool_cache_incomplete",
                        expected_rows=expected,
                        cached_rows=rows,
                        patterns=patterns,
                    )
                )
            pools.append(
                dict(
                    pool=pool,
                    task=task,
                    config=config,
                    split=split,
                    coverage_mode=spec["mode"],
                    expected_rows=expected,
                    cached_rows=rows,
                    files=len(found),
                    metadata_pool_complete=complete,
                    ledger_sha256=ledger["sha256"],
                    legacy_ledger_ID_count=len(ledger["sample_ids"]),
                    observed_ID_overlap=len(set(ids) & set(ledger["sample_ids"])) if ids else None,
                    unique_explicit_IDs=len(set(ids)) if ids else None,
                    legacy_sample_to_row_parity_asserted=False,
                )
            )
    video = records.get("videomme")
    if video:
        unresolved.append(
            dict(
                task="videomme",
                reason="actual_video_frame_exposure_or_status_journal_unresolved",
                ledger_sha256=video["sha256"],
                ID_references=len(video["sample_ids"]),
            )
        )
    plan = dict(
        source_sha=revision,
        cpu_gate_root=str(cpu_root),
        cpu_gate=cpu,
        bindings=bound,
        pools=pools,
        files=files,
        unresolved=unresolved,
        route_contradictions=contradictions,
        libraries={key: version(key) for key in ("Pillow", "pyarrow")},
        seconds_per_worker=SECONDS,
        shard_count=SHARDS,
        maximum_file_bytes=2 * 2**30,
        segment_count=1,
        automatic_resubmission=False,
        media_superset_not_sample_route_parity=True,
        downloaded_files=0,
        global_historical_media_inventory_complete=False,
        fresh_confirmation_authorized=False,
    )
    write_once(root / "plan.json", plan)
    return dict(
        status="prepared_finite_pinned_provider_supersets",
        files=len(files),
        pools=len(pools),
        unresolved=len(unresolved),
        plan_sha256=sha(root / "plan.json"),
    )


def load(root):
    plan = read(Path(root) / "plan.json")
    clean_source(SOURCE, plan["source_sha"])
    cpu_gate(Path(plan["cpu_gate_root"]), plan["source_sha"])
    for path, expected in plan["bindings"].items():
        if sha(Path(path)) != expected:
            raise ValueError("bound provider metadata changed")
    return plan


def worker(root, index=0):
    import pyarrow.parquet as pq

    root = Path(root).resolve()
    plan = load(root)
    if type(index) is not int or not 0 <= index < SHARDS:
        raise ValueError("registered CPU shard index required")
    write_once(root / "locks" / f"worker-{index}.json", dict(job_id=os.getenv("SLURM_JOB_ID")))
    deadline = time.monotonic() + plan["seconds_per_worker"]
    observations = []
    assets = {}
    for item in plan["files"]:
        if item["index"] != index:
            continue
        path = Path(item["path"])
        current = {}
        seen = 0
        result = dict(path=str(path), pool=item["pool"], rows_expected=item["rows"], complete=False)
        try:
            if path.resolve(strict=True) != path or signature(path) != item["signature"]:
                raise ValueError("frozen cached source identity changed")
            if item["signature"][2] > plan["maximum_file_bytes"]:
                raise ValueError("pinned source exceeds the registered file-byte bound")
            before = file_sha(path, deadline)
            if before != item["expected_sha256"]:
                raise ValueError("pinned LFS source SHA differs")
            table = pq.ParquetFile(path)
            stop = False
            for batch in table.iter_batches(columns=["image"], batch_size=128):
                for row in batch.to_pylist():
                    if time.monotonic() >= deadline:
                        stop = True
                        break
                    image = row["image"]
                    raw = image.get("bytes") if isinstance(image, dict) else image
                    if not isinstance(raw, bytes) or not 0 < len(raw) <= 32 * 2**20:
                        raise ValueError("embedded image outside the registered byte bound")
                    encoded = hashlib.sha256(raw).hexdigest()
                    if encoded not in current and encoded not in assets:
                        current[encoded] = dict(image_identity(raw), references=[])
                    target = current.setdefault(
                        encoded,
                        dict(assets[encoded], references=[])
                        if encoded in assets
                        else current[encoded],
                    )
                    target["references"].append(
                        dict(pool=item["pool"], source_sha256=before, source_row=seen)
                    )
                    seen += 1
                if stop:
                    break
            after = file_sha(path, deadline + 30)
            if (
                before != after
                or signature(path) != item["signature"]
                or path.resolve(strict=True) != path
            ):
                raise ValueError("cached source changed across pixel hashing")
            for key, value in current.items():
                old = assets.setdefault(key, dict(value, references=[]))
                old["references"].extend(value["references"])
            result.update(
                rows_hashed=seen,
                complete=seen == item["rows"],
                whole_file_sha256_before=before,
                whole_file_sha256_after=after,
                signature=item["signature"],
            )
        except (OSError, ValueError, TypeError, TimeoutError) as error:
            result.update(reason=type(error).__name__, rows_hashed=0)
        observations.append(result)
    output = dict(
        source_sha=plan["source_sha"],
        index=index,
        files=observations,
        assets=list(assets.values()),
        no_question_answer_or_text_columns_projected=True,
        fresh_confirmation_authorized=False,
    )
    path = root / "workers" / f"shard-{index}.json"
    write_once(path, output)
    return dict(index=index, files=len(observations), assets=len(assets), sha256=sha(path))


def analyze(root):
    root = Path(root).resolve()
    plan = load(root)
    missing = []
    bound = {}
    rows = defaultdict(int)
    assets = {}
    errors = []
    expected = {(item["path"], item["pool"]): item for item in plan["files"]}
    for index in range(SHARDS):
        path = root / "workers" / f"shard-{index}.json"
        if not path.exists():
            missing.append(index)
            continue
        result = read(path)
        if result["source_sha"] != plan["source_sha"] or result["index"] != index:
            raise ValueError("provider worker source/index differs")
        bound[str(path)] = sha(path)
        observed = {(r["path"], r["pool"]) for r in result["files"]}
        if len(observed) != len(result["files"]) or observed != {
            key for key, item in expected.items() if item["index"] == index
        }:
            raise ValueError("registered provider source files missing from worker summary")
        for observation in result["files"]:
            item = expected[observation["path"], observation["pool"]]
            if (observation["complete"] or observation["rows_hashed"] > 0) and (
                observation["whole_file_sha256_before"] != item["expected_sha256"]
                or observation["whole_file_sha256_after"] != item["expected_sha256"]
                or observation["signature"] != item["signature"]
            ):
                raise ValueError("provider source hash transaction differs")
            rows[item["pool"]] += observation["rows_hashed"]
            if not observation["complete"]:
                errors.append(observation)
        references = [
            (ref["pool"], ref["source_sha256"], ref["source_row"])
            for record in result["assets"]
            for ref in record["references"]
        ]
        expected_refs = {
            (o["pool"], expected[o["path"], o["pool"]]["expected_sha256"], i)
            for o in result["files"]
            for i in range(o["rows_hashed"])
        }
        if len(references) != len(set(references)) or set(references) != expected_refs:
            raise ValueError("verified source-image row references are not complete and unique")
        for record in result["assets"]:
            key = record["encoded_sha256"]
            if key in assets and assets[key]["pixel_sha256"] != record["pixel_sha256"]:
                raise ValueError("canonical pixel hash conflicts for one encoded body")
            old = assets.setdefault(key, dict(record, references=[]))
            old["references"].extend(record["references"])
    coverage = [
        dict(
            pool,
            hashed_rows=rows[pool["pool"]],
            conservative_media_superset_complete=pool["metadata_pool_complete"]
            and rows[pool["pool"]] == pool["expected_rows"],
        )
        for pool in plan["pools"]
    ]
    output = dict(
        source_sha=plan["source_sha"],
        assets=list(assets.values()),
        pool_coverage=coverage,
        unresolved=[*plan["unresolved"], *errors],
        route_contradictions=plan["route_contradictions"],
        worker_bindings=bound,
        global_historical_media_inventory_complete=False,
        fresh_confirmation_authorized=False,
    )
    write_once(root / "typed_asset_inventory.json", output)
    report = dict(
        status="finite_provider_superset_complete"
        if not (missing or errors or plan["unresolved"])
        else "finite_provider_superset_incomplete",
        source_sha=plan["source_sha"],
        missing_shards=missing,
        pool_coverage=coverage,
        typed_assets=len(assets),
        unresolved_providers=len(plan["unresolved"]),
        worker_errors=len(errors),
        route_contradictions=plan["route_contradictions"],
        worker_bindings=bound,
        typed_inventory_sha256=sha(root / "typed_asset_inventory.json"),
        legacy_sample_route_parity_certified=False,
        global_historical_media_inventory_complete=False,
        fresh_confirmation_authorized=False,
    )
    write_once(root / "GATE_historical_provider_superset.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "worker", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cpu-gate-root", type=Path)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    result = (
        prepare(args.root, cpu_gate_root=args.cpu_gate_root)
        if args.mode == "prepare"
        else worker(args.root, args.index)
        if args.mode == "worker"
        else analyze(args.root)
    )
    print(json.dumps(result))
