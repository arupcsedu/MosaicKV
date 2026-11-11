"""Bounded CPU recovery of already completed V12 COCO hash observations.

This reducer never loads a dataset, image, prediction or benchmark label.  It
does not turn incomplete historical metadata into a freshness certificate.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from mosaickv.evaluation.fox_campaign import clean_source, digest, write_once

PRODUCER = "76d4ced1a10c6746bad73f564e84210522355ef9"
PROVIDER_PRODUCER = "279c4f6336bd852d2ccd8408d519e1ddfa2a53b1"
HISTORY_PRODUCER = "84900a6a402e97c7d58499a722c159cfc86e46c7"
DATASET_REVISION = "c3213b54e173ae86517d481de17f31439d6f8c72"
PARENT = Path("/scratch/djy8hg/workdir/fox-review-v12/run-20261111-1")
SCRATCH_ROOT = Path("/scratch/djy8hg")
WORKER_BYTES = 4 * 2**20
METADATA_BYTES = 32 * 2**20
SHA = re.compile(r"[0-9a-f]{64}")
COCO = re.compile(r"COCO_val2014_\d{12}\.jpg")
WORKERS = (
    "39ee87517f4e3b6c1c0d518bb0b69e0722d8a7ce98d7ea3f485ba5a9ed5d3233",
    "f677f5748d20e757c47cb4ce742d7e97b433e68a58dbccd517e139028f1de66a",
    "7ae949981c07c48c94b59c87d6c825f0ffac931981fa13af86aad6c03421092f",
    "aba26739b4407516845cb26b403c260931e009cf0228a5add634965a1a537e46",
    "a9f1d06764d47382e77f3dce8d1fb3adefa778bdeecf2b06ddeb7360214a61e7",
    "150e091201f769e81399a0a84eec63bf83026e890224cce0926e545a7f71ace0",
    "aa0b9053a9873465339c5c04eeb28ccb243cb446abead43bde1dc60f1bb4691a",
    "440f67d8db5fc75c2724242d5658799739518a388a5f0ad8150310eae0febe62",
)
# Only explicit existing metadata inputs are readable.  Large metadata bounds
# are registered separately from the corrected 4-MiB worker-record bound.
INPUTS = {
    "plan": (
        PARENT / "media/plan.json",
        "3f49a8434dc247bf22bbdfa7757bd057af4a813210c6a799f4e7e4a4ca4e4fa7",
        WORKER_BYTES,
    ),
    "old_failure": (
        PARENT / "errors/analyze-media.json",
        "ebcf0dc2bef3cf4b148ee917d203c4633f972b8b25fc34dac3c45750eea90a7e",
        WORKER_BYTES,
    ),
    "typed_gate": (
        PARENT / "historical_media/GATE_historical_media_hashes.json",
        "3c0e7213c07d8286133088da0425b166be6289b81c9075f4f72254c9ee0cbcd8",
        WORKER_BYTES,
    ),
    "typed_history": (
        PARENT / "historical_media/typed_asset_inventory.json",
        "443e6731c843c992df08bcd4674d780fd6145f9a66dcf766e68b5920008fbe35",
        METADATA_BYTES,
    ),
    "provider_gate": (
        Path(
            "/scratch/djy8hg/workdir/fox-review-v15/run-20261111-1/providers/GATE_historical_provider_recovery.json"
        ),
        "85177b11d690e1c229f2a06fec91c60c6f901c24e35eafee9b50853791e86805",
        WORKER_BYTES,
    ),
    "provider_history": (
        Path(
            "/scratch/djy8hg/workdir/fox-review-v15/run-20261111-1/providers/typed_asset_inventory.json"
        ),
        "44b9399693fa44171124dc1cdb41f380e62c0a89d15cead6a75e20e22134dd13",
        8 * 2**20,
    ),
    "history_gate": (
        Path(
            "/scratch/djy8hg/workdir/fox-review-v16/run-20261111-2/history/GATE_history_closure_v16.json"
        ),
        "08c7897ffa7ad45be4eb7bbed13b8e3607056ebd16b0168928b18346d160ba34",
        WORKER_BYTES,
    ),
    "history": (
        Path(
            "/scratch/djy8hg/workdir/fox-review-v16/run-20261111-2/history/segment-4/combined_identity_inventory.json"
        ),
        "368c4fd8414e62af7cc4389eafb834d6d32988d857c8d9c23bb7763792e21102",
        METADATA_BYTES,
    ),
}


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} requires an integer >= {minimum}")
    return value


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate JSON metadata key")
        result[key] = value
    return result


def read_bound(path, expected, *, limit):
    """Read one explicit, immutable, strictly bounded JSON file; never recurse."""
    if not isinstance(expected, str) or not SHA.fullmatch(expected):
        raise ValueError("explicit SHA-256 metadata binding required")
    if type(limit) is not int or not 0 < limit <= METADATA_BYTES:
        raise ValueError("positive registered JSON byte bound required")
    path = Path(path).resolve(strict=True)
    if not path.is_relative_to(SCRATCH_ROOT.resolve(strict=True)):
        raise ValueError("metadata input lies outside the scratch-only scope")
    before = path.stat()
    if before.st_size > limit:
        raise ValueError("metadata exceeds its explicit registered byte bound")
    raw = path.read_bytes()
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or len(raw) != before.st_size:
        raise ValueError("metadata changed during its bound read")
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("immutable metadata SHA-256 mismatch")

    def constant(_value):
        raise ValueError("nonfinite JSON constant is not valid metadata")

    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=constant)


def aggregate_workers(plan, workers, *, expected_count=40504):
    """Validate saved worker transactions and return only identity/hash records."""
    integer(expected_count, "expected candidate count", 1)
    if (
        plan.get("source_sha") != PRODUCER
        or plan.get("dataset_id") != "lmms-lab/COCO-Caption"
        or plan.get("dataset_revision") != DATASET_REVISION
        or plan.get("split") != "val"
        or plan.get("cached_validation_ids") != expected_count
        or plan.get("shard_count") != 8
        or plan.get("hash_only_image_decode_registered") is not True
        or plan.get("cohort_selected") is not False
        or plan.get("materialization_authorized") is not False
        or plan.get("fresh_confirmation_authorized") is not False
    ):
        raise ValueError("original hash-only producer/dataset/permission contract differs")
    files = plan.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("complete original Arrow-file inventory required")
    expected_files = {}
    for index, row in enumerate(files):
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise ValueError("well-typed original source-file record required")
        if row["path"] in expected_files or not SHA.fullmatch(str(row.get("sha256", ""))):
            raise ValueError("duplicate or unbound original source file")
        integer(row.get("rows"), "original source rows", 1)
        if (
            not isinstance(row.get("signature"), list)
            or len(row["signature"]) != 4
            or any(type(value) is not int for value in row["signature"])
        ):
            raise ValueError("complete original source-file signature required")
        expected_files[row["path"]] = index, row
    if sum(row["rows"] for row in files) != expected_count:
        raise ValueError("original source rows differ from registered candidate count")
    if not isinstance(workers, (tuple, list)) or len(workers) != 8:
        raise ValueError("all eight original hash workers are required")
    records, seen_files, seen_rows = [], set(), set()
    allowed_record = {
        "encoded_sha256",
        "pixel_sha256",
        "sample_id",
        "source_row_index",
        "source_shard_index",
    }
    for index, worker in enumerate(workers):
        if (
            not isinstance(worker, dict)
            or set(worker)
            != {
                "complete",
                "files",
                "fresh_confirmation_authorized",
                "index",
                "no_image_copies_or_labels_exported",
                "records",
                "source_sha",
            }
            or type(worker.get("index")) is not int
            or worker["index"] != index
            or worker.get("source_sha") != PRODUCER
            or worker.get("complete") is not True
            or worker.get("fresh_confirmation_authorized") is not False
            or worker.get("no_image_copies_or_labels_exported") is not True
            or not isinstance(worker.get("files"), list)
            or not isinstance(worker.get("records"), list)
        ):
            raise ValueError("complete source-matched hash-only worker required")
        for observed in worker["files"]:
            path = observed.get("path")
            if path not in expected_files or path in seen_files:
                raise ValueError("unknown or repeated original file transaction")
            shard, original = expected_files[path]
            if (
                shard % 8 != index
                or observed.get("complete") is not True
                or any(
                    type(observed.get(k)) is not int or observed[k] != original["rows"]
                    for k in ("expected_rows", "next_row", "verified_rows")
                )
                or observed.get("signature_before") != original["signature"]
                or observed.get("signature_after") != original["signature"]
                or observed.get("whole_file_sha256_before") != original["sha256"]
                or observed.get("whole_file_sha256_after") != original["sha256"]
                or any(
                    not isinstance(observed.get(key), list)
                    or any(type(value) is not int for value in observed[key])
                    for key in ("signature_before", "signature_after")
                )
            ):
                raise ValueError("original source-file transaction is incomplete or changed")
            seen_files.add(path)
        for record in worker["records"]:
            if not isinstance(record, dict) or set(record) != allowed_record:
                raise ValueError(
                    "identity/hash-only record schema differs or contains a failed hash"
                )
            source_index = integer(record["source_shard_index"], "source shard index")
            row = integer(record["source_row_index"], "source row index")
            if (
                source_index >= len(files)
                or source_index % 8 != index
                or row >= files[source_index]["rows"]
            ):
                raise ValueError("hash record row/shard lies outside its registered source")
            if (source_index, row) in seen_rows:
                raise ValueError("duplicate original source row in hash observations")
            seen_rows.add((source_index, row))
            if not isinstance(record["sample_id"], str) or not COCO.fullmatch(record["sample_id"]):
                raise ValueError("canonical COCO validation filename required")
            if any(
                not isinstance(record[k], str) or not SHA.fullmatch(record[k])
                for k in ("encoded_sha256", "pixel_sha256")
            ):
                raise ValueError("complete encoded and canonical RGB SHA-256 hashes required")
            records.append(record)
    if len(records) != expected_count or seen_files != set(expected_files):
        raise ValueError("hash observations do not cover every registered source row and file")
    if len({r["sample_id"] for r in records}) != expected_count:
        raise ValueError("candidate canonical IDs are duplicated")
    if len({r["pixel_sha256"] for r in records}) != expected_count:
        raise ValueError("candidate canonical pixel hashes are duplicated")
    records.sort(key=lambda row: (row["source_shard_index"], row["source_row_index"]))
    if digest([r["sample_id"] for r in records]) != plan.get("projected_source_ids_sha256"):
        raise ValueError("complete source-order candidate ID digest differs from original plan")
    for source_index, original in enumerate(files):
        ids = [r["sample_id"] for r in records if r["source_shard_index"] == source_index]
        if digest(ids) != original.get("id_sha256"):
            raise ValueError("per-file candidate ID digest differs from original plan")
    return records


def historical_observations(history, history_gate, typed, typed_gate, provider, provider_gate):
    """Validate scoped metadata; unresolved sources remain explicitly unresolved."""
    if (
        history.get("source_sha") != HISTORY_PRODUCER
        or history_gate.get("source_sha") != HISTORY_PRODUCER
        or history.get("complete_historical_id_inventory") is not False
        or history_gate.get("frontier_exhausted") is not False
    ):
        raise ValueError("the preserved incomplete historical inventory contract differs")
    identities = history.get("identities")
    if not isinstance(identities, dict) or set(identities) != {
        "coco_ids",
        "media_hashes",
        "sample_ids",
        "scoped_identities",
        "scopes",
        "source_group_ids",
    }:
        raise ValueError("complete historical identity metadata schema required")
    for name, width in (("scoped_identities", 4), ("scopes", 3)):
        if not isinstance(identities[name], list) or any(
            not isinstance(row, list)
            or len(row) != width
            or any(not isinstance(x, str) for x in row)
            for row in identities[name]
        ):
            raise ValueError("well-typed historical scoped identities required")
    for name in ("coco_ids", "media_hashes", "sample_ids", "source_group_ids"):
        if not isinstance(identities[name], list) or any(
            not isinstance(x, str) for x in identities[name]
        ):
            raise ValueError("well-typed historical ID/hash lists required")
    if any(not COCO.fullmatch(x) for x in identities["coco_ids"]):
        raise ValueError("historical canonical COCO IDs differ from validation namespace")
    if any(not SHA.fullmatch(x) for x in identities["media_hashes"]):
        raise ValueError("historical untyped media hashes are malformed")
    hashes = {"encoded_sha256": set(), "pixel_sha256": set()}
    for inventory, gate, producer, gate_count in (
        (typed, typed_gate, PRODUCER, "typed_encoded_assets"),
        (provider, provider_gate, PROVIDER_PRODUCER, "typed_assets"),
    ):
        assets = inventory.get("assets")
        if (
            inventory.get("source_sha") != producer
            or gate.get("source_sha") != producer
            or inventory.get("fresh_confirmation_authorized") is not False
            or inventory.get("global_historical_media_inventory_complete") is not False
            or not isinstance(assets, list)
            or len(assets) != gate.get(gate_count)
        ):
            raise ValueError("hash-bound typed historical asset inventory differs")
        if not isinstance(inventory.get("unresolved"), list):
            raise ValueError("explicit unresolved historical provider records required")
        for asset in assets:
            if not isinstance(asset, dict) or any(
                not isinstance(asset.get(k), str) or not SHA.fullmatch(asset[k]) for k in hashes
            ):
                raise ValueError("typed historical encoded/RGB hashes are malformed")
            for key in hashes:
                hashes[key].add(asset[key])
    return dict(
        canonical_ids=set(identities["coco_ids"]),
        untyped_hashes=set(identities["media_hashes"]),
        typed_hashes=hashes,
        scoped_identity_records=len(identities["scoped_identities"]),
        unknown_scoped_record_count=sum(
            not row[0] or not row[2] for row in identities["scoped_identities"]
        ),
        unknown_scope_definition=(
            "namespace or split absent; configuration may legitimately be empty"
        ),
        source_group_ids=len(identities["source_group_ids"]),
        pending_frontier_items=integer(
            history_gate.get("pending_frontier_items"), "pending frontier"
        ),
        preserved_raw_coverage_errors=integer(
            history_gate.get("original_and_new_coverage_errors"), "raw coverage errors"
        ),
        typed_original_unresolved_entries=len(typed["unresolved"]),
        provider_unresolved_entries=len(provider["unresolved"]),
        video_exposure_resolved=provider_gate.get("video_exposure_resolved") is True,
    )


def availability_observations(records, historical):
    """Counts only: never export or select candidate identities for execution."""
    counts = dict(
        known_canonical_ID_overlap=0,
        typed_encoded_overlap=0,
        typed_RGB_overlap=0,
        conservative_untyped_hash_overlap=0,
        no_observed_overlap_with_incomplete_history=0,
    )
    for record in records:
        overlap = (
            record["sample_id"] in historical["canonical_ids"],
            record["encoded_sha256"] in historical["typed_hashes"]["encoded_sha256"],
            record["pixel_sha256"] in historical["typed_hashes"]["pixel_sha256"],
            any(
                record[key] in historical["untyped_hashes"]
                for key in ("encoded_sha256", "pixel_sha256")
            ),
        )
        for key, flag in zip(tuple(counts)[:4], overlap, strict=True):
            counts[key] += int(flag)
        counts["no_observed_overlap_with_incomplete_history"] += int(not any(overlap))
    return counts


def recover(output, *, reader_source_sha):
    """Canonical CPU-only recovery. Its result cannot release an experiment."""
    if not re.fullmatch(r"[0-9a-f]{40}", reader_source_sha):
        raise ValueError("immutable reader source SHA required")
    output = Path(output).resolve()
    if output.exists() or not output.is_relative_to(SCRATCH_ROOT.resolve(strict=True)):
        raise ValueError("new write-once scratch derived-analysis output required")
    if clean_source(Path(__file__).resolve().parents[4]) != reader_source_sha:
        raise ValueError("reader source SHA differs from its clean immutable worktree")
    inputs = dict(INPUTS)
    inputs.update(
        {
            f"worker_{i}": (PARENT / f"media/workers/shard-{i}.json", expected, WORKER_BYTES)
            for i, expected in enumerate(WORKERS)
        }
    )
    write_once(
        output / "registration.json",
        dict(
            reader_source_sha=reader_source_sha,
            original_producer_sha=PRODUCER,
            inputs={
                name: dict(path=str(path), sha256=expected, maximum_bytes=limit)
                for name, (path, expected, limit) in inputs.items()
            },
            expected_hash_rows=40504,
            worker_record_bound_bytes=WORKER_BYTES,
            separately_registered_historical_metadata_bound_bytes=METADATA_BYTES,
            models_loaded=False,
            dataset_rows_or_images_accessed=False,
            predictions_answers_or_scores_accessed=False,
            fresh_confirmation_authorized=False,
        ),
    )
    try:
        data = {
            name: read_bound(path, expected, limit=limit)
            for name, (path, expected, limit) in inputs.items()
        }
        failure = data["old_failure"]
        if (
            failure.get("status") != "blocked"
            or failure.get("error")
            != "ValueError: bound producer artifact exceeds registered byte limit"
        ):
            raise ValueError("the preserved original analysis-bound failure differs")
        for gate_name, inventory_name in (
            ("typed_gate", "typed_history"),
            ("provider_gate", "provider_history"),
        ):
            if data[gate_name].get("typed_inventory_sha256") != inputs[inventory_name][1]:
                raise ValueError("historical asset inventory/gate hash binding differs")
        if data["history_gate"].get("identity_inventory_sha256") != inputs["history"][1]:
            raise ValueError("historical identity inventory/gate hash binding differs")
        records = aggregate_workers(data["plan"], [data[f"worker_{i}"] for i in range(8)])
        historical = historical_observations(
            data["history"],
            data["history_gate"],
            data["typed_history"],
            data["typed_gate"],
            data["provider_history"],
            data["provider_gate"],
        )
        if len(historical["canonical_ids"]) != 5249:
            raise ValueError("reviewed current historical COCO-ID denominator differs from 5249")
        counts = availability_observations(records, historical)
        report = dict(
            status="completed_saved_COCO_hash_reduction_not_freshness_certification",
            complete_saved_hash_reduction=True,
            reader_source_sha=reader_source_sha,
            original_producer_sha=PRODUCER,
            original_failure_preserved=True,
            dataset_id="lmms-lab/COCO-Caption",
            dataset_revision=DATASET_REVISION,
            split="val",
            hash_rows=len(records),
            unique_canonical_ids=len({r["sample_id"] for r in records}),
            unique_RGB_hashes=len({r["pixel_sha256"] for r in records}),
            failed_hash_rows=0,
            complete_candidate_hash_records_sha256=digest(records),
            known_historical_canonical_COCO_ids=len(historical["canonical_ids"]),
            observations=counts,
            historical_diagnostics={
                key: value
                for key, value in historical.items()
                if key not in {"canonical_ids", "untyped_hashes", "typed_hashes"}
            },
            global_historical_metadata_complete=False,
            globally_fresh=False,
            observed_hash_absence_is_not_freshness=True,
            confirmation_blocked=True,
            fresh_confirmation_authorized=False,
            candidate_ids_selected_or_exported=False,
            materialization_authorized=False,
            GPU_jobs_submitted=0,
            remaining_blockers=[
                "incomplete historical frontier and unresolved metadata",
                "unresolved video exposure and source-group aliases",
            ],
        )
    except Exception as error:
        write_once(
            output / "FAILED_recovery.json",
            dict(
                complete_saved_hash_reduction=False,
                error=f"{type(error).__name__}: {error}",
                globally_fresh=False,
                confirmation_blocked=True,
                fresh_confirmation_authorized=False,
            ),
        )
        raise
    write_once(output / "GATE_saved_COCO_hash_recovery.json", report)
    return report
