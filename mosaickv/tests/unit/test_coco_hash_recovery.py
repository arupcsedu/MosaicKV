"""Synthetic hash-only fixtures; no real COCO rows, images or labels are read."""

import hashlib
from copy import deepcopy

import pytest

from mosaickv.evaluation import coco_hash_recovery as study


def sha(number):
    return f"{number:064x}"


def fixtures():
    workers, files, ids = [], [], []
    for index in range(8):
        sample_id = f"COCO_val2014_{index:012d}.jpg"
        ids.append(sample_id)
        source = dict(
            path=f"/synthetic/part-{index}.arrow",
            rows=1,
            signature=[1, index + 1, 100, 9],
            sha256=sha(1000 + index),
            id_sha256=study.digest([sample_id]),
        )
        files.append(source)
        observed = dict(
            path=source["path"],
            complete=True,
            expected_rows=1,
            next_row=1,
            verified_rows=1,
            signature_before=source["signature"],
            signature_after=source["signature"],
            whole_file_sha256_before=source["sha256"],
            whole_file_sha256_after=source["sha256"],
        )
        record = dict(
            sample_id=sample_id,
            encoded_sha256=sha(index + 1),
            pixel_sha256=sha(index + 101),
            source_shard_index=index,
            source_row_index=0,
        )
        workers.append(
            dict(
                complete=True,
                files=[observed],
                records=[record],
                fresh_confirmation_authorized=False,
                index=index,
                no_image_copies_or_labels_exported=True,
                source_sha=study.PRODUCER,
            )
        )
    plan = dict(
        source_sha=study.PRODUCER,
        dataset_id="lmms-lab/COCO-Caption",
        dataset_revision=study.DATASET_REVISION,
        split="val",
        cached_validation_ids=8,
        shard_count=8,
        hash_only_image_decode_registered=True,
        cohort_selected=False,
        materialization_authorized=False,
        fresh_confirmation_authorized=False,
        files=files,
        projected_source_ids_sha256=study.digest(ids),
    )
    return plan, workers


def history_fixtures():
    history = dict(
        source_sha=study.HISTORY_PRODUCER,
        complete_historical_id_inventory=False,
        identities=dict(
            coco_ids=["COCO_val2014_000000000000.jpg"],
            media_hashes=[sha(2)],
            sample_ids=["synthetic"],
            scoped_identities=[["dataset", "", "val", "a"], ["", "", "", "b"]],
            scopes=[["dataset", "", "val"]],
            source_group_ids=[],
        ),
    )
    gate = dict(
        source_sha=study.HISTORY_PRODUCER,
        frontier_exhausted=False,
        pending_frontier_items=10,
        original_and_new_coverage_errors=20,
    )
    typed = dict(
        source_sha=study.PRODUCER,
        fresh_confirmation_authorized=False,
        global_historical_media_inventory_complete=False,
        assets=[dict(encoded_sha256=sha(3), pixel_sha256=sha(103))],
        unresolved=[{}],
    )
    typed_gate = dict(source_sha=study.PRODUCER, typed_encoded_assets=1)
    provider = dict(
        source_sha=study.PROVIDER_PRODUCER,
        fresh_confirmation_authorized=False,
        global_historical_media_inventory_complete=False,
        assets=[dict(encoded_sha256=sha(4), pixel_sha256=sha(104))],
        unresolved=[],
    )
    provider_gate = dict(
        source_sha=study.PROVIDER_PRODUCER, typed_assets=1, video_exposure_resolved=False
    )
    return history, gate, typed, typed_gate, provider, provider_gate


def aggregate(plan, workers):
    return study.aggregate_workers(plan, workers, expected_count=8)


def test_complete_saved_hashes_recover_with_original_order_and_no_labels():
    plan, workers = fixtures()
    records = aggregate(plan, workers)
    assert len(records) == 8
    assert [r["source_shard_index"] for r in records] == list(range(8))
    assert all(
        set(r)
        == {"sample_id", "encoded_sha256", "pixel_sha256", "source_shard_index", "source_row_index"}
        for r in records
    )


def test_missing_worker_fails_closed():
    plan, workers = fixtures()
    with pytest.raises(ValueError, match="eight"):
        aggregate(plan, workers[:-1])


def test_duplicate_id_is_not_a_smaller_silent_denominator():
    plan, workers = fixtures()
    workers[1]["records"][0]["sample_id"] = workers[0]["records"][0]["sample_id"]
    with pytest.raises(ValueError, match="IDs are duplicated"):
        aggregate(plan, workers)


def test_duplicate_pixel_hashes_fail_registered_uniqueness_check():
    plan, workers = fixtures()
    workers[1]["records"][0]["pixel_sha256"] = workers[0]["records"][0]["pixel_sha256"]
    with pytest.raises(ValueError, match="pixel hashes are duplicated"):
        aggregate(plan, workers)


def test_wrong_worker_producer_is_not_reinterpreted_as_current_reader():
    plan, workers = fixtures()
    workers[0]["source_sha"] = "f" * 40
    with pytest.raises(ValueError, match="source-matched"):
        aggregate(plan, workers)


def test_boolean_worker_index_is_not_an_integer_identity():
    plan, workers = fixtures()
    workers[1]["index"] = True
    with pytest.raises(ValueError):
        aggregate(plan, workers)


def test_incomplete_worker_stays_failed():
    plan, workers = fixtures()
    workers[0]["complete"] = False
    with pytest.raises(ValueError):
        aggregate(plan, workers)


def test_prediction_or_answer_fields_are_forbidden_in_recovered_records():
    plan, workers = fixtures()
    workers[0]["records"][0]["answer"] = "synthetic-forbidden"
    with pytest.raises(ValueError, match="record schema"):
        aggregate(plan, workers)


def test_missing_or_invalid_media_hash_fails():
    plan, workers = fixtures()
    workers[0]["records"][0]["pixel_sha256"] = "not-a-hash"
    with pytest.raises(ValueError, match="SHA-256"):
        aggregate(plan, workers)


def test_row_cursor_outside_pinned_source_fails():
    plan, workers = fixtures()
    workers[0]["records"][0]["source_row_index"] = 1
    with pytest.raises(ValueError, match="outside"):
        aggregate(plan, workers)


def test_record_in_wrong_original_worker_fails():
    plan, workers = fixtures()
    workers[0]["records"][0]["source_shard_index"] = 1
    with pytest.raises(ValueError, match="outside"):
        aggregate(plan, workers)


def test_per_file_id_digest_must_match():
    plan, workers = fixtures()
    plan["files"][0]["id_sha256"] = sha(999)
    with pytest.raises(ValueError, match="per-file"):
        aggregate(plan, workers)


def test_source_file_transaction_must_be_complete_and_unchanged():
    plan, workers = fixtures()
    workers[0]["files"][0]["whole_file_sha256_after"] = sha(999)
    with pytest.raises(ValueError, match="transaction"):
        aggregate(plan, workers)


def test_dataset_revision_cannot_be_substituted():
    plan, workers = fixtures()
    plan["dataset_revision"] = "f" * 40
    with pytest.raises(ValueError, match="contract"):
        aggregate(plan, workers)


def test_old_authorization_flags_do_not_get_silently_reenabled():
    plan, workers = fixtures()
    plan["fresh_confirmation_authorized"] = True
    with pytest.raises(ValueError, match="contract"):
        aggregate(plan, workers)


def test_incomplete_historical_scope_is_reported_not_waived():
    result = study.historical_observations(*history_fixtures())
    assert result["unknown_scoped_record_count"] == 1
    assert result["pending_frontier_items"] == 10
    assert result["typed_original_unresolved_entries"] == 1
    assert result["source_group_ids"] == 0
    assert result["video_exposure_resolved"] is False


def test_malformed_scoped_identity_cannot_be_used_for_available_count():
    values = list(history_fixtures())
    values[0]["identities"]["scoped_identities"][0] = ["too-short"]
    with pytest.raises(ValueError, match="scoped"):
        study.historical_observations(*values)


def test_malformed_typed_historical_hash_blocks_reduction():
    values = list(history_fixtures())
    values[2]["assets"][0]["encoded_sha256"] = "bad"
    with pytest.raises(ValueError, match="malformed"):
        study.historical_observations(*values)


def test_historical_producer_is_validated_separately():
    values = list(history_fixtures())
    values[0]["source_sha"] = "0" * 40
    with pytest.raises(ValueError, match="inventory contract"):
        study.historical_observations(*values)


def test_no_observed_overlap_is_not_called_freshness():
    plan, workers = fixtures()
    records = aggregate(plan, workers)
    history = study.historical_observations(*history_fixtures())
    counts = study.availability_observations(records, history)
    assert counts == dict(
        known_canonical_ID_overlap=1,
        typed_encoded_overlap=2,
        typed_RGB_overlap=2,
        conservative_untyped_hash_overlap=1,
        no_observed_overlap_with_incomplete_history=4,
    )
    assert all("fresh" not in key for key in counts)


def test_bound_reader_rejects_changed_content(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "SCRATCH_ROOT", tmp_path)
    p = tmp_path / "metadata.json"
    p.write_text('{"id":1}')
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        study.read_bound(p, "0" * 64, limit=128)


def test_bound_reader_enforces_explicit_maximum_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "SCRATCH_ROOT", tmp_path)
    p = tmp_path / "metadata.json"
    p.write_text('{"id":1}')
    with pytest.raises(ValueError, match="byte bound"):
        study.read_bound(p, hashlib.sha256(p.read_bytes()).hexdigest(), limit=1)


def test_bound_reader_rejects_duplicate_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "SCRATCH_ROOT", tmp_path)
    p = tmp_path / "metadata.json"
    p.write_text('{"id":1,"id":2}')
    with pytest.raises(ValueError, match="duplicate"):
        study.read_bound(p, hashlib.sha256(p.read_bytes()).hexdigest(), limit=128)


def test_bound_reader_rejects_nonfinite_json(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "SCRATCH_ROOT", tmp_path)
    p = tmp_path / "metadata.json"
    p.write_text('{"id":NaN}')
    with pytest.raises(ValueError, match="nonfinite"):
        study.read_bound(p, hashlib.sha256(p.read_bytes()).hexdigest(), limit=128)


def test_recovery_cannot_overwrite_existing_output_or_use_dirty_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "SCRATCH_ROOT", tmp_path)
    with pytest.raises(ValueError, match="source SHA"):
        study.recover(tmp_path / "new", reader_source_sha="dirty")
    with pytest.raises(ValueError, match="write-once"):
        study.recover(tmp_path, reader_source_sha="a" * 40)


def test_synthetic_validator_does_not_mutate_original_metadata():
    plan, workers = fixtures()
    before = deepcopy((plan, workers))
    aggregate(plan, workers)
    assert (plan, workers) == before
