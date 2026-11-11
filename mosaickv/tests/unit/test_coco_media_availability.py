"""Synthetic cached pixels only: no real image/label access or submissions."""

import io
import json
from pathlib import Path
from types import SimpleNamespace

import evicastkv_coco_media_availability as audit
import pyarrow as pa
import pytest
from PIL import Image


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return audit.sha(path)


def png(level=0):
    stream = io.BytesIO()
    Image.new("RGB", (3, 2), (10, 20, 30)).save(stream, format="PNG", compress_level=level)
    return stream.getvalue()


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    cache, history = tmp_path / audit.COCO_REVISION, tmp_path / "history"
    cache.mkdir()
    files = {}
    for index in range(14):
        path = cache / f"coco-caption-val-{index:05d}-of-00014.arrow"
        sid = f"COCO_val2014_{index + 1:012d}.jpg"
        table = pa.Table.from_pylist(
            [
                dict(
                    question_id=sid,
                    file_name=sid,
                    id=1000 + index,
                    image=dict(bytes=png(), path=None),
                    question="FORBIDDEN_QUESTION",
                    answer=["FORBIDDEN_CAPTION"],
                )
            ]
        )
        with (
            pa.OSFile(str(path), "wb") as stream,
            pa.ipc.new_stream(stream, table.schema) as writer,
        ):
            writer.write_table(table)
        files[str(path)] = audit.sha(path)
    availability = tmp_path / "availability.json"
    availability_sha = put(
        availability,
        dict(
            dataset_id=audit.DATASET,
            dataset_revision=audit.COCO_REVISION,
            split="val",
            validation_image_ids=14,
            bound_files=files,
        ),
    )
    inventory = dict(
        identities=dict(
            sample_ids=[],
            coco_ids=["COCO_val2014_000000000001.jpg"],
            media_hashes=["b" * 64],
            source_group_ids=[],
            scopes=[],
            scoped_identities=[],
        ),
        complete_historical_id_inventory=False,
        cross_dataset_media_certified=False,
        fresh_confirmation_authorized=False,
    )
    inventory_sha = put(history / "combined_identity_inventory.json", inventory)
    put(
        history / "GATE_freshness_frontier_fork.json",
        dict(
            identity_inventory_sha256=inventory_sha,
            frontier_exhausted=False,
            original_and_new_coverage_errors=4,
        ),
    )
    media = tmp_path / "media_identities.json"
    media_sha = put(
        media, dict(identities={"other-dataset/picture.png": audit.image_identity(png())})
    )
    monkeypatch.setattr(audit, "clean_source", lambda *args: "f" * 40)
    monkeypatch.setattr(audit, "cpu_gate", lambda *args: dict(status="passed"))
    return dict(
        root=tmp_path / "output",
        cpu_gate_root=tmp_path,
        history_root=history,
        availability=availability,
        availability_sha256=availability_sha,
        media=media,
        media_sha256=media_sha,
        cache=cache,
    )


def prepare(fixture, **kwargs):
    return audit.prepare(**fixture, **kwargs)


def test_preparation_is_id_only_and_never_certifies_freshness(fixture, monkeypatch):
    monkeypatch.setattr(audit, "image_identity", lambda *args: pytest.fail("ID-only cannot decode"))
    prepare(fixture)
    plan = audit.load(fixture["root"])
    assert plan["cached_validation_ids"] == 14
    assert plan["observed_historical_exact_id_overlap"] == 1
    assert plan["exact_ids_not_observed_in_incomplete_history"] == 13
    assert not plan["hash_only_image_decode_registered"]
    assert not plan["fresh_confirmation_authorized"]
    assert not plan["cohort_selected"]


def test_hash_workers_require_explicit_registered_scope(fixture):
    prepare(fixture)
    with pytest.raises(ValueError, match="explicit hash-only"):
        audit.worker(fixture["root"], 0)
    assert not (fixture["root"] / "locks").exists()


def test_eight_shards_cover_ids_and_typed_hash_aliases_without_certification(fixture):
    prepare(fixture, hash_media=True)
    for index in range(8):
        result = audit.worker(fixture["root"], index)
        assert result["complete"]
    report = audit.analyze(fixture["root"])
    assert report["hash_rows"] == report["successfully_hashed_media"] == 14
    assert report["missing_or_failed_media_hashes"] == 0
    assert report["observed_encoded_or_pixel_alias_overlap"] == dict(
        encoded_sha256=14, pixel_sha256=14
    )
    assert report["candidate_internal_pixel_duplicate_records"] == 13
    assert report["not_observed_ID_and_hash_overlap_count"] == 0
    assert report["remaining_transform_and_global_source_group_aliases_unknown"]
    assert not report["fresh_confirmation_authorized"]
    assert sorted(fixture["root"].glob("**/*.png")) == []
    assert (
        "FORBIDDEN_CAPTION"
        not in (fixture["root"] / "GATE_coco_media_availability.json").read_text()
    )


def test_pixel_algorithm_matches_existing_manifest_and_reencoding(fixture):
    from evicastkv_media_audit import image_identity

    a, b = audit.image_identity(png(0)), audit.image_identity(png(9))
    assert a["pixel_sha256"] == b["pixel_sha256"]
    assert a["encoded_sha256"] != b["encoded_sha256"]
    previous = image_identity(png())
    assert all(a[key] == previous[key] for key in a)


@pytest.mark.parametrize("images", [False, True])
def test_projection_never_converts_label_columns(monkeypatch, images):
    selected = []

    class Batch:
        def select(self, columns):
            selected.append(columns)
            assert columns == list(audit.HASH_COLUMNS if images else audit.ID_COLUMNS)
            return SimpleNamespace(to_pylist=lambda: [dict(question_id="x", file_name="x", id=0)])

    class Context:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class Reader:
        schema = SimpleNamespace(names=[*audit.HASH_COLUMNS, "answer", "question"])

        def __iter__(self):
            yield Batch()

    monkeypatch.setattr(pa, "memory_map", lambda *args: Context())
    monkeypatch.setattr(pa.ipc, "open_stream", lambda *args: Reader())
    assert list(audit.arrow_rows(Path("not-a-real-file"), images=images)) == [
        dict(question_id="x", file_name="x", id=0)
    ]
    assert len(selected) == 1


def test_pinned_source_hash_failure_cannot_export_decoded_rows(fixture):
    prepare(fixture, hash_media=True)
    path = fixture["cache"] / "coco-caption-val-00000-of-00014.arrow"
    path.write_bytes(b"mutated")
    result = audit.worker(fixture["root"], 0)
    assert not result["complete"] and result["rows"] == 1  # Other registered file remains valid.
    output = json.loads((fixture["root"] / "workers/shard-0.json").read_text())
    assert "before decoding" in output["files"][0]["error"]
    assert all(r["sample_id"] != "COCO_val2014_000000000001.jpg" for r in output["records"])


def test_missing_workers_remain_missing_media_not_disjoint(fixture):
    prepare(fixture, hash_media=True)
    audit.worker(fixture["root"], 0)
    report = audit.analyze(fixture["root"])
    assert report["missing_worker_shards"] == list(range(1, 8))
    assert report["missing_or_failed_media_hashes"] == 12
    assert report["observed_hash_absence_is_not_certified_disjointness"]


def test_after_decode_source_hash_failure_discards_only_unverified_file(fixture, monkeypatch):
    prepare(fixture, hash_media=True)
    calls = []

    def hashing(path, deadline):
        calls.append(path)
        return "0" * 64 if len(calls) == 2 else audit.sha(path)

    monkeypatch.setattr(audit, "file_sha", hashing)
    result = audit.worker(fixture["root"], 0)
    assert not result["complete"] and result["rows"] == 1
    output = json.loads((fixture["root"] / "workers/shard-0.json").read_text())
    assert "after hash-only" in output["files"][0]["error"]
    assert output["records"][0]["source_shard_index"] == 8


def test_timeout_is_partial_reusable_metadata_not_completion(fixture, monkeypatch):
    prepare(fixture, hash_media=True)
    clock = iter([0, 241, 241])
    monkeypatch.setattr(audit.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(audit, "file_sha", lambda path, deadline: audit.sha(path))
    result = audit.worker(fixture["root"], 0)
    assert not result["complete"] and result["rows"] == 0


def test_auxiliary_numeric_id_is_never_source_identity():
    row = dict(
        question_id="COCO_val2014_000000000007.jpg", file_name="COCO_val2014_000000000007.jpg", id=0
    )
    assert audit.canonical_id(row).endswith("000000000007.jpg")
    row["question_id"] = "COCO_val2014_000000000000.jpg"
    with pytest.raises(ValueError, match="disagree"):
        audit.canonical_id(row)


def test_oversized_image_rejected_before_decode(monkeypatch):
    monkeypatch.setattr(audit, "MAX_IMAGE", 3)
    with pytest.raises(ValueError, match="bytes unavailable"):
        audit.image_identity(b"too big")


def test_new_continuation_gate_binds_merged_ids_and_cannot_be_mutated(fixture):
    inventory_path = fixture["history_root"] / "combined_identity_inventory.json"
    inventory = json.loads(inventory_path.read_text())
    inventory["identities"]["coco_ids"].append("COCO_val2014_000000000002.jpg")
    actual = put(inventory_path, inventory)
    new_gate = fixture["history_root"] / "GATE_freshness_empty_parquet_frontier.json"
    put(
        new_gate,
        dict(
            identity_inventory_sha256=actual,
            frontier_exhausted=False,
            original_and_new_coverage_errors=4,
        ),
    )
    prepare(fixture, hash_media=True, history_gate_filename=new_gate.name)
    plan = audit.load(fixture["root"])
    assert plan["observed_historical_exact_id_overlap"] == 2
    assert plan["history_gate_filename"] == new_gate.name
    put(new_gate, {})
    with pytest.raises(ValueError, match="evidence changed"):
        audit.worker(fixture["root"], 0)


def test_unregistered_history_gate_name_is_not_an_authorization_alias(fixture):
    with pytest.raises(ValueError, match="reviewed history gate"):
        prepare(fixture, history_gate_filename="../unrelated/GATE.json")


def test_historical_typed_asset_join_needs_no_new_pixel_decode(fixture, monkeypatch):
    fixture["media_sha256"] = put(
        fixture["media"],
        dict(identities={"old-nonmatching": dict(encoded_sha256="c" * 64, pixel_sha256="d" * 64)}),
    )
    prepare(fixture, hash_media=True)
    for index in range(8):
        audit.worker(fixture["root"], index)
    extra = fixture["root"].parent / "historical-media"
    known = audit.image_identity(png())
    actual = put(
        extra / "typed_asset_inventory.json",
        dict(source_sha="f" * 40, assets=[known], unresolved=[dict(reason="ID-only-provider")]),
    )
    put(
        extra / "GATE_historical_media_hashes.json",
        dict(
            source_sha="f" * 40,
            typed_inventory_sha256=actual,
            status="finite_historical_media_audit_incomplete",
        ),
    )
    monkeypatch.setattr(
        audit, "image_identity", lambda *args: pytest.fail("analysis cannot decode pixels")
    )
    report = audit.analyze(fixture["root"], historical_media_root=extra)
    assert report["historical_media_unresolved_entries"] == 1
    assert report["observed_encoded_or_pixel_alias_overlap"]["pixel_sha256"] == 14
    assert report["historical_media_bindings"]


def test_authoritative_YAML_gate_can_supply_merged_history(fixture):
    old = fixture["history_root"] / "GATE_freshness_frontier_fork.json"
    gate = json.loads(old.read_text())
    put(fixture["history_root"] / "GATE_freshness_yaml_events.json", gate)
    prepare(fixture, history_gate_filename="GATE_freshness_yaml_events.json")
    plan = audit.load(fixture["root"])
    assert plan["history_gate_filename"] == "GATE_freshness_yaml_events.json"
    assert plan["historical_coverage_errors"] == 4 and not plan["fresh_confirmation_authorized"]
