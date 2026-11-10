#!/usr/bin/env python3
"""CPU-only assessment of saved inventory gaps; no historical traversal.

Read only an existing inventory and its exact hash-linked checkpoints. Optional
Parquet inspection reads schema/footer metadata for at most four error paths,
never rows, predictions, answers or references. Unknowns are not waived.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
DEFAULT_PARENT = Path(
    "/scratch/djy8hg/audits/fox-review-v8-consumed-diagnostics-20261110/freshness-continuation"
)
MAX_METADATA_BYTES = 2**30
MAX_METADATA_FILE = 256 * 2**20
MAX_CHECKPOINTS = 512
RULES = (
    ("YAML requires separately reviewed", "yaml_projection_unsupported"),
    ("JSONL record exceeds", "jsonl_record_oversize"),
    ("JSON exceeds bounded", "json_document_oversize"),
    ("Parquet has no safely projectable", "parquet_no_declared_identity_columns"),
    ("unprojectable nested/binary", "parquet_nested_projection_unsupported"),
    ("malformed historical JSONL", "jsonl_malformed"),
    ("ValueError: symlink target outside", "symlink_outside_scope"),
    ("ValueError: historical source changed", "source_mutation"),
    ("ValueError: truncated historical", "json_malformed"),
    ("ValueError: invalid historical", "json_malformed"),
)


def category(message):
    return next(
        (kind for prefix, kind in RULES if message.startswith(prefix)), "unknown_not_waived"
    )


def root_name(name):
    """Normalize known lexical root aliases without following error symlinks."""
    parts = PurePosixPath(name).parts
    for index in range(len(parts) - 2):
        if parts[index] == "runs" and parts[index + 1] in {"mosaickv", "evicastkv"}:
            return "/".join(parts[index + 1 : index + 3])
    return "unclassified_source_root_not_waived"


class MetadataReader:
    def __init__(self):
        self.bytes = 0
        self.bindings = {}

    def read(self, path, expected=None):
        path = Path(path)
        if not path.resolve(strict=True).is_relative_to(Path("/scratch/djy8hg").resolve()):
            raise ValueError("saved metadata resolves outside the registered scratch scope")
        before = path.stat()
        if before.st_size > MAX_METADATA_FILE or self.bytes + before.st_size > MAX_METADATA_BYTES:
            raise ValueError("saved-metadata byte budget exceeded; no historical scan attempted")
        raw = path.read_bytes()
        after = path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError("saved metadata changed during read")
        actual = hashlib.sha256(raw).hexdigest()
        if expected is not None and actual != expected:
            raise ValueError("saved inventory/checkpoint hash binding changed")
        self.bytes += len(raw)
        self.bindings[str(path)] = actual
        return json.loads(raw)


def checkpoint_chain(path, expected, reader, inventory_source):
    """Follow named parents only: never glob or discover new result sources."""
    path = Path(path)
    latest = reader.read(path, expected)
    registration = reader.read(path.parent / "registration.json", latest["registration_sha256"])
    if registration["source_sha"] != inventory_source:
        raise ValueError("checkpoint producer differs from inventory producer")
    count, current, current_path = 0, latest, path
    while True:
        count += 1
        if count > MAX_CHECKPOINTS or current["number"] < 0:
            raise ValueError("invalid or unbounded checkpoint chain")
        if current["registration_sha256"] != sha(path.parent / "registration.json"):
            raise ValueError("checkpoint registration binding changed")
        if digest(current["state"]) != current["state_sha256"]:
            raise ValueError("checkpoint state binding changed")
        number = current["number"]
        if current_path.name != f"checkpoint-{number:06d}.json":
            raise ValueError("checkpoint filename/number differs")
        if number == 0:
            if current["parent_checkpoint_sha256"] is not None:
                raise ValueError("initial checkpoint has an unexpected parent")
            break
        current_path = path.parent / f"checkpoint-{number - 1:06d}.json"
        current = reader.read(current_path, current["parent_checkpoint_sha256"])
    state = latest["state"]
    frontier = state["frontier"]
    if state["traversal_complete"] and frontier:
        raise ValueError("terminal checkpoint still has pending traversal work")
    return dict(
        index=registration["shard_index"],
        expected_shards=registration["shard_count"],
        latest_checkpoint=str(path),
        latest_checkpoint_number=latest["number"],
        checkpoint_files=count,
        traversal_complete=bool(state["traversal_complete"]),
        frontier_items=len(frontier),
        frontier_kinds=dict(sorted(Counter(item["kind"] for item in frontier).items())),
        frontier_roots=dict(sorted(Counter(root_name(item["path"]) for item in frontier).items())),
        frontier_files_are_not_total_remaining_files=True,
        processed_files=state["processed_files"],
        projected_records=state["projected_records"],
        coverage_errors=len(state["errors"]),
        projection_pauses=len(state.get("projection_pauses", [])),
    )


def error_groups(errors):
    groups = defaultdict(list)
    for row in errors:
        path = row.get("path", "")
        kind = category(str(row.get("error", "")))
        schema = (PurePosixPath(path).suffix or "unknown_suffix") + "/" + kind
        groups[(kind, root_name(path), schema)].append(row)
    return [
        dict(
            category=kind,
            root=root,
            schema=schema,
            occurrences=len(rows),
            unique_paths=len({row.get("path", "") for row in rows}),
            paths=sorted({row.get("path", "") for row in rows}),
            original_rows_sha256=digest(rows),
            waived=False,
        )
        for (kind, root, schema), rows in sorted(groups.items())
    ]


def parquet_schema_only(paths, allowed_roots):
    """Footer/schema observations cannot waive missing producer-ledger links."""
    import pyarrow.parquet as pq

    results = []
    roots = [Path(path).resolve(strict=True) for path in allowed_roots]
    for name in sorted(set(paths))[:4]:
        result = dict(path=name, rows_read=0, error_waived=False)
        try:
            path = Path(name).resolve(strict=True)
            if not any(path.is_relative_to(root) for root in roots):
                raise ValueError("schema path resolves outside registered historical roots")
            before = path.stat()
            with path.open("rb") as handle:
                handle.seek(-8, 2)
                tail = handle.read(8)
                length = int.from_bytes(tail[:4], "little")
                if (
                    tail[4:] != b"PAR1"
                    or not 0 < length <= 8 * 2**20
                    or length + 8 > before.st_size
                ):
                    raise ValueError("invalid or oversized Parquet footer")
                handle.seek(-8 - length, 2)
                footer = handle.read(length)
            parquet = pq.ParquetFile(path)
            fields = [dict(name=field.name, type=str(field.type)) for field in parquet.schema_arrow]
            after = path.stat()
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ):
                raise ValueError("Parquet metadata changed during inspection")
            result.update(
                schema=fields,
                schema_sha256=digest(fields),
                footer_sha256=hashlib.sha256(footer).hexdigest(),
                file_size=before.st_size,
                row_count_from_footer=parquet.metadata.num_rows,
                row_groups=parquet.metadata.num_row_groups,
                whole_file_content_bound=False,
                producer_identity_link_verified=False,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            result["metadata_error"] = f"{type(exc).__name__}: {exc}"
        results.append(result)
    return results


def assess(parent, *, inspect_parquet_schema=False):
    parent = Path(parent).resolve(strict=True)
    path = parent / "aggregate/inventory.json" if parent.is_dir() else parent
    reader = MetadataReader()
    inventory = reader.read(path)
    errors, chains, failures = inventory["errors"], [], []
    for checkpoint, expected in sorted(inventory["shard_bindings"].items()):
        try:
            chains.append(checkpoint_chain(checkpoint, expected, reader, inventory["source_sha"]))
        except (KeyError, ValueError, OSError) as exc:
            failures.append(dict(path=checkpoint, error=f"{type(exc).__name__}: {exc}"))
    expected_counts = {row["expected_shards"] for row in chains}
    ownership = [row["index"] for row in chains]
    if len(expected_counts) != 1 or sorted(ownership) != list(
        range(next(iter(expected_counts), 0))
    ):
        failures.append(dict(error="not every registered shard has one verified checkpoint chain"))
    if (
        chains
        and not failures
        and (
            sum(row["processed_files"] for row in chains) != inventory["processed_files"]
            or sum(row["projected_records"] for row in chains) != inventory["projected_records"]
            or sum(row["coverage_errors"] for row in chains) != len(errors)
        )
    ):
        failures.append(dict(error="inventory totals disagree with bound latest checkpoints"))
    groups = error_groups(errors)
    categories = dict(
        sorted(
            Counter(
                {
                    kind: sum(group["occurrences"] for group in groups if group["category"] == kind)
                    for kind in {group["category"] for group in groups}
                }
            ).items()
        )
    )
    inspected = (
        parquet_schema_only(
            [
                row["path"]
                for row in errors
                if category(row.get("error", "")).startswith("parquet_")
            ],
            inventory["roots"],
        )
        if inspect_parquet_schema
        else []
    )
    return dict(
        evidence_valid=not failures,
        input_bindings=reader.bindings,
        bounded_metadata_bytes=reader.bytes,
        inventory_source_sha=inventory["source_sha"],
        processed_files=inventory["processed_files"],
        projected_records=inventory["projected_records"],
        original_errors=len(errors),
        original_errors_sha256=digest(errors),
        category_counts=categories,
        schema_grouping_is_error_taxonomy_not_full_source_schema=True,
        groups=groups,
        checkpoint_chains=chains,
        evidence_failures=failures,
        completed_shards=sum(row["traversal_complete"] for row in chains),
        incomplete_shards=[row["index"] for row in chains if not row["traversal_complete"]],
        frontier_items=sum(row["frontier_items"] for row in chains),
        frontier_is_not_a_remaining_file_count=True,
        checkpoint_files_verified=sum(row["checkpoint_files"] for row in chains),
        parquet_schema_observations=inspected,
        parquet_rows_read=0,
        three_independent_blockers=dict(
            traversal_incomplete=bool(
                failures or any(not row["traversal_complete"] for row in chains)
            ),
            coverage_repair_required=bool(errors),
            media_alias_certification_missing=True,
        ),
        cheapest_targeted_recovery=dict(
            first="versioned_oversized_JSONL_identity_projection_overlay",
            reason="Reuse the already linear, deadline-aware JSON metadata projector on the exact "
            "registered error paths/offsets; do not repeat historical directory traversal.",
            prerequisites=[
                "Bind original checkpoint/source-segment hashes and every affected record offset.",
                "Project complete records without decoding benchmark values or "
                "truncating references.",
                "Publish a separate overlay and prove complete source coverage before "
                "resolving any error.",
            ],
            yaml="A separate reviewed lexical YAML projector is needed; ordinary YAML "
            "object loading "
            "would construct benchmark values before redaction.",
            parquet="Inspect bounded schemas; add reviewed identity aliases or verified "
            "producer-ledger "
            "links. No-column summaries are not blanket harmlessness certificates.",
            traversal="Continue the same immutable checkpoint frontier separately; "
            "projection repair "
            "cannot replace unfinished traversal.",
        ),
        original_errors_reclassified_as_harmless=0,
        historical_result_rows_or_answers_read=False,
        new_historical_scan_performed=False,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
        cohort_selected=False,
        models_loaded=False,
        complete_historical_id_inventory=False,
        cross_dataset_media_certified=False,
    )


def run(root, parent=DEFAULT_PARENT, *, inspect_parquet_schema=False):
    root = Path(root).resolve()
    if root.exists() or not root.is_relative_to(Path("/scratch/djy8hg").resolve()):
        raise ValueError("new immutable scratch gap-audit output required")
    revision = clean_source(SOURCE)
    report = assess(parent, inspect_parquet_schema=inspect_parquet_schema)
    report["source_sha"] = revision
    write_once(root / "report.json", report)
    gate = dict(
        status="passed_gap_assessment_confirmation_blocked"
        if report["evidence_valid"]
        else "blocked_invalid_saved_evidence",
        source_sha=revision,
        report_sha256=sha(root / "report.json"),
        original_errors=report["original_errors"],
        frontier_items=report["frontier_items"],
        completed_shards=report["completed_shards"],
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
        complete_historical_id_inventory=False,
        cross_dataset_media_certified=False,
        new_historical_scan_performed=False,
    )
    write_once(root / "GATE_freshness_gaps.json", gate)
    return gate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--inspect-parquet-schema", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(run(args.root, args.parent, inspect_parquet_schema=args.inspect_parquet_schema))
    )
