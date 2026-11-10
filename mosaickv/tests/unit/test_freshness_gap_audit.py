"""Saved-metadata gap assessment must never imply fresh cohort eligibility."""

import hashlib
import json
from pathlib import Path

import evicastkv_freshness_gap_audit as audit
import pytest


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def saved(tmp_path):
    parent = tmp_path / "parent"
    history = tmp_path / "runs" / "evicastkv"
    history.mkdir(parents=True)
    errors = [
        dict(
            path=str(history / "run-a" / "config.yaml"),
            error="YAML requires separately reviewed lexical identity projection",
        ),
        dict(
            path=str(history / "run-b" / "raw.jsonl"),
            offset=100,
            error="JSONL record exceeds identity-projection line bound",
        ),
        dict(
            path=str(history / "run-b" / "unknown.record"),
            error="unrecognized legacy identity alias",
        ),
    ]
    bindings = {}
    for index in range(2):
        root = parent / "shards" / f"shard-{index:03d}"
        registration = dict(source_sha="a" * 40, shard_index=index, shard_count=2)
        registration_sha = put(root / "registration.json", registration)
        current_errors = errors if index == 0 else []
        state = dict(
            frontier=[dict(kind="directory", path=str(history / f"pending-{index}"))],
            traversal_complete=False,
            processed_files=3,
            projected_records=7,
            errors=current_errors,
            projection_pauses=[],
        )
        previous = None
        for number in range(2):
            checkpoint = dict(
                number=number,
                registration_sha256=registration_sha,
                parent_checkpoint_sha256=previous,
                state=state,
                state_sha256=audit.digest(state),
            )
            path = root / f"checkpoint-{number:06d}.json"
            previous = put(path, checkpoint)
        bindings[str(path)] = previous
    inventory = dict(
        source_sha="a" * 40,
        errors=errors,
        shard_bindings=bindings,
        roots=[str(history)],
        processed_files=6,
        projected_records=14,
        traversal_complete=False,
        fresh_confirmation_authorized=False,
    )
    put(parent / "aggregate/inventory.json", inventory)
    return parent, history, inventory


def test_saved_chains_frontier_and_errors_are_reported_without_scanning(saved, monkeypatch):
    parent, _history, _inventory = saved

    def no_walk(*args, **kwargs):
        raise AssertionError("historical traversal is forbidden")

    monkeypatch.setattr(Path, "glob", no_walk)
    report = audit.assess(parent)
    assert report["evidence_valid"]
    assert report["processed_files"] == 6 and report["projected_records"] == 14
    assert report["frontier_items"] == 2 and report["completed_shards"] == 0
    assert report["checkpoint_files_verified"] == 4
    assert report["original_errors"] == 3
    assert sum(row["occurrences"] for row in report["groups"]) == 3
    assert not report["fresh_confirmation_authorized"]
    assert not report["new_historical_scan_performed"]


def test_unknown_errors_are_not_dropped_or_waived(saved):
    report = audit.assess(saved[0])
    assert report["category_counts"]["unknown_not_waived"] == 1
    assert all(group["waived"] is False for group in report["groups"])
    assert report["original_errors_reclassified_as_harmless"] == 0
    assert report["three_independent_blockers"] == dict(
        traversal_incomplete=True,
        coverage_repair_required=True,
        media_alias_certification_missing=True,
    )


def test_hash_tampering_is_evidence_failure_not_a_freshness_pass(saved):
    parent, _history, inventory = saved
    checkpoint = Path(next(iter(inventory["shard_bindings"])))
    checkpoint.write_text("{}")
    report = audit.assess(parent)
    assert not report["evidence_valid"]
    assert any("hash binding changed" in row["error"] for row in report["evidence_failures"])
    assert not report["fresh_confirmation_authorized"]


def test_missing_parent_checkpoint_is_explicit(saved):
    parent, _history, inventory = saved
    path = Path(next(iter(inventory["shard_bindings"]))).parent / "checkpoint-000000.json"
    # Keep original data recoverable in this synthetic fixture.
    path.rename(path.with_suffix(".saved"))
    report = audit.assess(parent)
    assert not report["evidence_valid"]
    assert report["evidence_failures"]


def test_contradictory_inventory_counts_cannot_pass(saved):
    parent, _history, inventory = saved
    inventory["processed_files"] = 999
    put(parent / "aggregate/inventory.json", inventory)
    report = audit.assess(parent)
    assert not report["evidence_valid"]
    assert any("totals disagree" in row["error"] for row in report["evidence_failures"])


def test_write_once_report_and_gate_keep_original_evidence_unchanged(saved, tmp_path, monkeypatch):
    parent, _history, _inventory = saved
    source = parent / "aggregate/inventory.json"
    original_sha = audit.sha(source)
    monkeypatch.setattr(audit, "clean_source", lambda source: "b" * 40)
    output = tmp_path / "derived"
    gate = audit.run(output, parent)
    assert gate["status"] == "passed_gap_assessment_confirmation_blocked"
    assert not gate["fresh_confirmation_authorized"]
    assert audit.sha(source) == original_sha
    assert audit.sha(output / "report.json") == gate["report_sha256"]
    assert (output / "GATE_freshness_gaps.json").is_file()
    with pytest.raises(ValueError, match="immutable"):
        audit.run(output, parent)


def test_schema_inspection_reads_at_most_four_footers_not_rows(tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    history = tmp_path / "historical"
    history.mkdir()
    paths = []
    for index in range(5):
        path = history / f"{index}.parquet"
        pq.write_table(pa.table(dict(metric=[1.0], answer=["synthetic secret"])), path)
        paths.append(str(path))

    def no_rows(*args, **kwargs):
        raise AssertionError("Parquet rows/benchmark values must not be read")

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", no_rows)
    monkeypatch.setattr(pq.ParquetFile, "read", no_rows)
    report = audit.parquet_schema_only(paths, [history])
    assert len(report) == 4
    assert all(row["rows_read"] == 0 and row["error_waived"] is False for row in report)
    assert all(row["footer_sha256"] and row["schema_sha256"] for row in report)
    assert "synthetic secret" not in json.dumps(report)


def test_schema_scope_symlink_escape_is_not_followed_or_waived(tmp_path):
    history = tmp_path / "history"
    history.mkdir()
    outside = tmp_path / "outside.parquet"
    outside.write_bytes(b"not_parquet")
    alias = history / "alias.parquet"
    alias.symlink_to(outside)
    result = audit.parquet_schema_only([str(alias)], [history])
    assert "outside registered" in result[0]["metadata_error"]
    assert not result[0]["error_waived"]


def test_metadata_budget_exhaustion_does_not_become_completion(saved, monkeypatch):
    parent, _history, _inventory = saved
    monkeypatch.setattr(audit, "MAX_CHECKPOINTS", 1)
    report = audit.assess(parent)
    assert not report["evidence_valid"]
    assert not report["complete_historical_id_inventory"]


@pytest.mark.parametrize(
    "path,expected",
    [
        ("/scratch/djy8hg/runs/mosaickv/run-a/raw.jsonl", "mosaickv/run-a"),
        ("/sfs/weka/scratch/djy8hg/runs/evicastkv/run-b/data.yaml", "evicastkv/run-b"),
        ("/unreviewed/location/file.json", "unclassified_source_root_not_waived"),
    ],
)
def test_root_grouping_is_lexical_not_a_new_filesystem_scan(path, expected):
    assert audit.root_name(path) == expected
