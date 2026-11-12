"""Synthetic selective scheduling and fail-closed wave closure."""

import builtins
import hashlib
import io
import json
import os
from copy import deepcopy
from pathlib import Path

import evicastkv_reader_continuation_v44 as driver
import pytest


def rows():
    return [
        dict(
            owner_index=0,
            historical_error_ordinal=i,
            original_error_sha256=str(i) * 64,
            path=f"/fake/{i}.json",
            operation="json_stream",
            signature=[1, 2, 30, 4],
        )
        for i in range(3)
    ]


def test_successful_old_occurrences_never_eligible():
    data = rows()
    parent = dict(unresolved_occurrences=[dict(data[1], error="old"), dict(data[2], error="old")])
    assert driver.eligible(parent, data) == [1, 2]


def test_changed_original_binding_is_not_silently_retried():
    data = rows()
    unresolved = deepcopy(data[1])
    unresolved["path"] = "/another/file.json"
    with pytest.raises(ValueError, match="binding"):
        driver.eligible(dict(unresolved_occurrences=[unresolved]), data)


def test_worker_cannot_invoke_completed_or_non_JSON_index(tmp_path, monkeypatch):
    monkeypatch.setattr(
        driver.common, "load", lambda *a, **k: dict(source_sha="a" * 40, retry_indices=[2])
    )
    monkeypatch.setattr(driver.common, "cpu_gate", lambda *a: {})
    with pytest.raises(ValueError, match="completed old proofs"):
        driver.worker(tmp_path, 0, "parse", "a" * 64)


def test_missing_hash_is_an_unqualified_wave_not_permission_to_continue(tmp_path, monkeypatch):
    phases = []
    monkeypatch.setattr(
        driver, "run_tasks", lambda root, plan, indices, phase, label: phases.append(phase)
    )
    monkeypatch.setattr(
        driver.common,
        "checked_task",
        lambda root, plan, i, phase: dict(status="passed") if phase == "parse" else None,
    )
    qualified, failed = driver.wave(tmp_path, {}, [0], "fixture")
    assert phases == ["parse", "hash"]
    assert qualified == [] and failed == [0]


def test_negative_canary_prevents_bulk_input_reads(tmp_path, monkeypatch):
    monkeypatch.setattr(driver.common, "load", lambda *a: dict(source_sha="a" * 40))
    monkeypatch.setattr(driver.common, "cpu_gate", lambda *a: {})
    monkeypatch.setattr(
        driver.common.history,
        "read",
        lambda *a: dict(source_sha="a" * 40, streaming_bulk_authorized=False),
    )
    monkeypatch.setattr(driver, "wave", lambda *a: pytest.fail("bulk started"))
    assert driver.bulk(tmp_path)["status"] == "blocked_canary_failed"


def test_closing_hash_failure_stops_the_next_wave(tmp_path, monkeypatch):
    plan = dict(
        source_sha="a" * 40,
        json_canary_indices=[],
        retry_indices=list(range(10)),
        bulk_wall_seconds=5400,
    )
    monkeypatch.setattr(driver.common, "load", lambda *a: plan)
    monkeypatch.setattr(driver.common, "cpu_gate", lambda *a: {})
    monkeypatch.setattr(
        driver.common.history,
        "read",
        lambda *a: dict(source_sha="a" * 40, streaming_bulk_authorized=True),
    )
    called = []

    def wave(*args):
        called.append(args[2])
        return [], [0]

    monkeypatch.setattr(driver, "wave", wave)
    result = driver.bulk(tmp_path)
    assert len(called) == 1
    assert result["status"] == "unqualified_pass_stopped_later_waves"


def test_failed_shard_does_not_pollute_exclusions(tmp_path, monkeypatch):
    known = {kind: set() for kind in driver.identities.KINDS}
    known["sample_ids"].add("already-known")
    before = deepcopy(known)

    def partial(*args):
        yield "sample_ids", "new"
        raise ValueError("corrupt later shard")

    monkeypatch.setattr(driver, "shard_values", partial)
    with pytest.raises(ValueError, match="later shard"):
        driver.exclusion_delta(tmp_path, 0, {}, known)
    assert known == before


def test_repeated_sample_in_new_scope_is_not_dropped(tmp_path, monkeypatch):
    known = {kind: set() for kind in driver.identities.KINDS}
    known["sample_ids"].add("id")
    known["scoped_identities"].add(("old", "", "train", "id"))
    monkeypatch.setattr(
        driver,
        "shard_values",
        lambda *args: iter(
            [
                ("sample_ids", "id"),
                ("scoped_identities", ("new", "", "train", "id")),
            ]
        ),
    )
    result = driver.exclusion_delta(tmp_path, 0, {}, known)
    assert result["sample_ids"] == set()
    assert result["scoped_identities"] == {("new", "", "train", "id")}


def test_reconciliation_rechecks_live_source_before_reading_shards(tmp_path, monkeypatch):
    path = tmp_path / "source.json"
    path.write_bytes(b" " * 30)
    signature = driver.bounded.geometry(path.lstat())
    row = dict(path=str(path), signature=signature)
    a = dict(
        path=str(path),
        whole_file_sha256="a" * 64,
        bytes_read=30,
        total_input_bytes=30,
        hash_passes=1,
        independent_second_pass_verified=False,
        parser_schema="evicastkv_bounded_external_JSON_v44",
        complete_document_validated=True,
        all_spill_replays_verified=True,
        forbidden_value_scalars_JSON_decoded=False,
        statistics=dict(
            maximum_live_projected_metadata_bytes=1, maximum_live_projected_metadata_nodes=1
        ),
    )
    for key in ("signature_before", "signature_after", "fd_signature_before", "fd_signature_after"):
        a[key] = signature
    b = dict(
        path=str(path),
        whole_file_sha256="a" * 64,
        bytes_read=30,
        signature=signature,
        independent_hash_pass=True,
    )
    monkeypatch.setattr(
        driver.common,
        "checked_task",
        lambda root, plan, i, phase: dict(
            status="passed", wall_seconds=1.0, proof=a if phase == "parse" else b
        ),
    )
    monkeypatch.setattr(driver, "shard_values", lambda *a: pytest.fail("stale source accepted"))
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="live source"):
        driver.pair(tmp_path, dict(rows=[row], roots=[str(tmp_path)]), 0)


@pytest.fixture
def saved_old_pairs(tmp_path, monkeypatch):
    """Thirty-two completed pairs plus one eligible source, all synthetic."""
    old = tmp_path / "old-proof-run"
    (old / "tasks").mkdir(parents=True)
    sources = tmp_path / "old-sources"
    sources.mkdir()
    data, bindings = [], {}
    for index in range(33):
        source = sources / f"{index:04d}.json"
        raw = b'{"sample_id":"synthetic-only"}'
        source.write_bytes(raw)
        row = dict(
            owner_index=0,
            historical_error_ordinal=index,
            original_error_sha256=f"{index:064x}",
            path=str(source),
            operation="json_stream",
            signature=driver.bounded.geometry(source.lstat()),
        )
        data.append(row)
        if index == 32:
            continue
        source_hash = hashlib.sha256(raw).hexdigest()
        for phase in ("parse", "hash"):
            proof = dict(path=str(source), bytes_read=len(raw), whole_file_sha256=source_hash)
            if phase == "hash":
                proof.update(signature=row["signature"], independent_hash_pass=True)
            else:
                proof.update(hash_passes=1, independent_second_pass_verified=False)
                for key in (
                    "signature_before",
                    "signature_after",
                    "fd_signature_before",
                    "fd_signature_after",
                ):
                    proof[key] = row["signature"]
            record = dict(
                source_sha="d37fc8daf75af02d66ce35cf76abe0d7c3a5446b",
                plan_sha256=driver.previous.OLD_PLAN_SHA,
                registration_sha256=driver.digest(row),
                status="passed",
                index=index,
                phase=phase,
                proof=proof,
                wall_seconds=1.0,
            )
            path = driver.common.artifact(old, index, phase)
            path.write_text(json.dumps(record))
            bindings[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(driver.previous, "OLD", old)
    return dict(source_bindings=bindings), data, [32], [str(sources)], old


def test_reuses_all_32_saved_pairs_without_opening_any_old_input_body(
    saved_old_pairs,
    monkeypatch,
):
    parent, data, targets, roots, _old = saved_old_pairs
    source_paths = {str(Path(row["path"]).resolve()) for row in data[:32]}
    opened_receipts = set()

    def guarded(original):
        def call(path, *args, **kwargs):
            if isinstance(path, (str, bytes, os.PathLike)):
                name = str(Path(os.fsdecode(path)).absolute())
                if name in source_paths:
                    pytest.fail("a completed historical input body was reopened")
                if name in parent["source_bindings"]:
                    opened_receipts.add(name)
            return original(path, *args, **kwargs)

        return call

    monkeypatch.setattr(builtins, "open", guarded(builtins.open))
    monkeypatch.setattr(io, "open", guarded(io.open))
    monkeypatch.setattr(os, "open", guarded(os.open))
    result = driver.verify_reused_pairs(parent, data, targets, roots)
    assert result == parent["source_bindings"]
    assert len(result) == len(opened_receipts) == 64


@pytest.mark.parametrize("field", ["status", "index", "phase", "registration_sha256"])
def test_bad_saved_receipt_fails_even_when_its_new_hash_is_bound(saved_old_pairs, field):
    parent, data, targets, roots, old = saved_old_pairs
    path = driver.common.artifact(old, 0, "parse")
    record = json.loads(path.read_bytes())
    record[field] = {
        "status": "blocked",
        "index": 1,
        "phase": "hash",
        "registration_sha256": "f" * 64,
    }[field]
    path.write_text(json.dumps(record))
    parent["source_bindings"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="receipt"):
        driver.verify_reused_pairs(parent, data, targets, roots)


def test_modified_saved_receipt_does_not_match_its_original_hash(saved_old_pairs):
    parent, data, targets, roots, old = saved_old_pairs
    path = driver.common.artifact(old, 0, "parse")
    with path.open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError):
        driver.verify_reused_pairs(parent, data, targets, roots)


def test_missing_old_receipt_binding_never_causes_reparsing(saved_old_pairs, monkeypatch):
    parent, data, targets, roots, old = saved_old_pairs
    path = driver.common.artifact(old, 0, "parse")
    parent["source_bindings"].pop(str(path))
    monkeypatch.setattr(
        driver.common.history, "read", lambda *a, **k: pytest.fail("unbound receipt was read")
    )
    with pytest.raises(ValueError, match="binding"):
        driver.verify_reused_pairs(parent, data, targets, roots)


def test_changed_old_source_signature_blocks_reuse_without_body_read(saved_old_pairs, monkeypatch):
    parent, data, targets, roots, _old = saved_old_pairs
    path = Path(data[0]["path"])
    path.write_bytes(b"changed-size")
    monkeypatch.setattr(
        driver.common.history,
        "read",
        lambda *a, **k: pytest.fail("stale source receipt was accepted"),
    )
    with pytest.raises(ValueError, match="source changed"):
        driver.verify_reused_pairs(parent, data, targets, roots)


def test_exactly_32_prior_pairs_must_remain_excluded(saved_old_pairs):
    parent, data, targets, roots, _old = saved_old_pairs
    with pytest.raises(ValueError, match="exactly 32"):
        driver.verify_reused_pairs(parent, data, [*targets, 0], roots)


def test_26_additional_record_resolutions_do_not_require_26_old_V41_pairs(
    saved_old_pairs,
    monkeypatch,
):
    parent, data, _targets, roots, old = saved_old_pairs
    expected = dict(parent["source_bindings"])
    qualified = data[:32]
    for index in range(32, 58):
        row = deepcopy(data[32])
        row.update(
            historical_error_ordinal=index,
            original_error_sha256=f"{index:064x}",
            path=f"/synthetic-record-resolved/source-{index}.json",
        )
        qualified.append(row)
        parent["source_bindings"][str(old.parent / f"record-run/tasks/{index:04d}-index.json")] = (
            "e" * 64
        )
    original = driver.bounded.common.canonical

    def old_32_only(path, allowed):
        assert not str(path).startswith("/synthetic-record-resolved/")
        return original(path, allowed)

    monkeypatch.setattr(driver.bounded.common, "canonical", old_32_only)
    assert driver.verify_reused_pairs(parent, qualified, [], roots) == expected


@pytest.fixture
def consumed_canary_failures(tmp_path, monkeypatch):
    old = tmp_path / "old-failure-run"
    (old / "tasks").mkdir(parents=True)
    categories = sorted(driver.previous.RESOURCE_ERRORS)
    sizes = [10, 100, 100, 70, 90, 500, 200, 1000, 2000]
    data, saved = [], {}
    for index, size in enumerate(sizes):
        row = dict(
            owner_index=0,
            historical_error_ordinal=index,
            original_error_sha256=f"{index:064x}",
            operation="json_stream",
            path=f"/synthetic-unopened/source-{index}.json",
            signature=[1, index + 1, size, 4],
        )
        data.append(row)
        if index == 8:  # No completed historical failure record exists.
            continue
        record = dict(
            source_sha="d37fc8daf75af02d66ce35cf76abe0d7c3a5446b",
            plan_sha256=driver.previous.OLD_PLAN_SHA,
            registration_sha256=driver.digest(row),
            index=index,
            phase="parse",
            status="blocked",
            error=categories[0] if index in {0, 1, 2, 6} else categories[1],
        )
        if index == 7:
            record["error"] = "invalid JSON scalar"
        path = driver.common.artifact(old, index, "parse")
        path.write_text(json.dumps(record))
        saved[str(path)] = record
    monkeypatch.setattr(driver.previous, "OLD", old)
    reads = []

    def read(path, *args, **kwargs):
        reads.append(str(path))
        return saved[str(path)]

    monkeypatch.setattr(driver.common.history, "read", read)
    return data, [0, 1, 2, 3, 4, 7, 8], saved, reads, old


def test_canaries_are_largest_unresolved_failures_with_low_index_ties(consumed_canary_failures):
    data, targets, _saved, reads, old = consumed_canary_failures
    assert driver.select_canaries(data, targets) == [1, 4]
    assert str(driver.common.artifact(old, 5, "parse")) not in reads
    assert str(driver.common.artifact(old, 6, "parse")) not in reads
    assert str(driver.common.artifact(old, 8, "parse")) not in reads


@pytest.mark.parametrize("field", ["source_sha", "plan_sha256", "registration_sha256"])
def test_canary_with_bad_original_provenance_is_rejected(consumed_canary_failures, field):
    data, targets, saved, _reads, old = consumed_canary_failures
    saved[str(driver.common.artifact(old, 1, "parse"))][field] = "f" * 64
    with pytest.raises(ValueError, match="original binding"):
        driver.select_canaries(data, targets)


@pytest.mark.parametrize("kind", ["missing_category", "not_blocked"])
def test_canaries_require_a_blocked_occurrence_from_each_resource_category(
    consumed_canary_failures,
    kind,
):
    data, targets, saved, _reads, old = consumed_canary_failures
    if kind == "missing_category":
        targets = [0, 1, 2, 7, 8]
    else:
        for index in (3, 4):
            saved[str(driver.common.artifact(old, index, "parse"))]["status"] = "passed"
    with pytest.raises(ValueError, match="per resource category"):
        driver.select_canaries(data, targets)


@pytest.mark.parametrize("phase", ["parse", "hash"])
@pytest.mark.parametrize("elapsed", [210.0001, 211.0, 334.0])
def test_either_whole_file_pass_over_210_seconds_is_rejected(
    tmp_path,
    monkeypatch,
    phase,
    elapsed,
):
    path = tmp_path / "source.json"
    path.write_bytes(b"{}")
    signature = driver.bounded.geometry(path.lstat())
    row = dict(path=str(path), signature=signature)
    a = dict(
        path=str(path),
        whole_file_sha256="a" * 64,
        bytes_read=2,
        total_input_bytes=2,
        hash_passes=1,
        independent_second_pass_verified=False,
        parser_schema="evicastkv_bounded_external_JSON_v44",
        complete_document_validated=True,
        all_spill_replays_verified=True,
        forbidden_value_scalars_JSON_decoded=False,
        statistics=dict(
            maximum_live_projected_metadata_bytes=1,
            maximum_live_projected_metadata_nodes=1,
        ),
    )
    for key in ("signature_before", "signature_after", "fd_signature_before", "fd_signature_after"):
        a[key] = signature
    b = dict(
        path=str(path),
        whole_file_sha256="a" * 64,
        bytes_read=2,
        signature=signature,
        independent_hash_pass=True,
    )
    monkeypatch.setattr(
        driver.common,
        "checked_task",
        lambda root, plan, i, current: dict(
            status="passed",
            wall_seconds=elapsed if current == phase else 1.0,
            proof=a if current == "parse" else b,
        ),
    )
    monkeypatch.setattr(
        driver, "shard_values", lambda *a: pytest.fail("over-time pass reached shard verification")
    )
    with pytest.raises(ValueError, match=r"210|time|bound"):
        driver.pair(tmp_path, dict(rows=[row], roots=[str(tmp_path)]), 0)
