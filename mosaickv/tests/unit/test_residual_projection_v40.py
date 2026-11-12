"""Synthetic V40 metadata and exact-producer regressions, never real artifacts."""

import copy
import json
import time

import evicastkv_residual_projection_v40 as helper
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def row(path, kind):
    return dict(
        path=str(path),
        kind=kind,
        signature=helper.common.scanner.signature(path),
        original_error_sha256s=["f" * 64],
    )


@pytest.fixture
def yaml_case(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("dataset: synthetic-coco\nsample_id: synthetic-id\nprompt: PRIVATE_PROMPT\n")
    return row(path, "yaml_events"), [str(tmp_path)]


def test_unchanged_yaml_events_preserve_exclusions_and_redact_benchmark(yaml_case):
    item, roots = yaml_case
    item["file_sha256"] = helper.expected_file_sha(item, roots, time.monotonic() + 5)
    proof = helper.prove(item, roots, time.monotonic() + 5)
    assert helper.validate_proof(proof) is True
    assert proof["registered_current_file_sha256"] == item["file_sha256"]
    assert "synthetic-id" in proof["identities"]["sample_ids"]
    assert "PRIVATE" not in json.dumps(proof)
    assert proof["old_unbound_historical_version_byte_identity_claimed"] is False


def test_yaml_anchors_are_still_rejected(yaml_case):
    item, roots = yaml_case
    path = helper.Path(item["path"])
    path.write_text("dataset: &a synthetic\nother: *a\n")
    item = row(path, "yaml_events")
    item["file_sha256"] = helper.expected_file_sha(item, roots, time.monotonic() + 5)
    with pytest.raises(ValueError, match="anchors"):
        helper.prove(item, roots, time.monotonic() + 5)


def test_yaml_registered_hash_cannot_be_rebound(yaml_case):
    item, roots = yaml_case
    item["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="already registered"):
        helper.expected_file_sha(item, roots, time.monotonic() + 5)
    with pytest.raises(ValueError, match="registered current source"):
        helper.prove(item, roots, time.monotonic() + 5)


@pytest.fixture
def parquet_case(tmp_path, monkeypatch):
    original = helper.reviewed.original
    helper.reviewed._ATTESTED_SOURCES.clear()
    helper._WORKERS.clear()
    owner = tmp_path / "history/attempt"
    key = "000-method-40-r0"
    edge = owner / "mechanism-terms" / key / "edges.parquet"
    other = edge.with_name("raw_marginal_terms.parquet")
    edge.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(dict(source=[0], target=[1], weight=[0.5], edge_type=["internal_enum"])), edge
    )
    other.write_bytes(b"synthetic-other-artifact")

    def put(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return original.sha(path)

    audit, record, worker = (
        owner / "mechanisms" / f"{key}.json",
        owner / "records" / f"{key}.json",
        owner / "worker.json",
    )
    audit_sha = put(
        audit,
        dict(raw_terms={str(p): original.sha(p) for p in (edge, other)}, answer="PRIVATE_ANSWER"),
    )
    record_sha = put(
        record,
        dict(
            source_sha=helper.PRODUCER,
            mechanism_audit_path=str(audit),
            mechanism_audit_sha256=audit_sha,
        ),
    )
    put(
        worker,
        dict(source_sha=helper.PRODUCER, records=[dict(path=str(record), sha256=record_sha)]),
    )

    def producer(revision, deadline):
        assert revision == helper.PRODUCER
        assert original.TEMPLATES == helper.reviewed.BASE_TEMPLATES
        return dict(
            recorded_producer_commit=revision,
            reviewed_template_commit=revision,
            reviewed_AST_sha256s=copy.deepcopy(original.TEMPLATES),
            source_blob_sha256s={spec.split("::")[0]: "e" * 64 for spec in original.TEMPLATES},
            historical_source_executed=False,
        )

    monkeypatch.setattr(original, "producer_proof", producer)
    monkeypatch.setattr(
        pq, "read_table", lambda *args, **kwargs: pytest.fail("row decoding forbidden")
    )
    yield row(edge, "closed_parquet"), [str(tmp_path)], worker
    helper.reviewed._ATTESTED_SOURCES.clear()
    helper._WORKERS.clear()


def test_parquet_metadata_registration_never_body_hashes(parquet_case, monkeypatch):
    item, roots, _ = parquet_case
    monkeypatch.setattr(
        helper.reviewed.original,
        "hash_file",
        lambda *args: pytest.fail("body hashing forbidden at registration"),
    )
    expected = helper.expected_file_sha(item, roots, time.monotonic() + 5)
    assert expected == helper.reviewed.original.sha(helper.Path(item["path"]))


def test_exact_fourth_producer_is_retained_without_relabeling(parquet_case):
    item, roots, _ = parquet_case
    item["file_sha256"] = helper.expected_file_sha(item, roots, time.monotonic() + 5)
    proof = helper.prove(item, roots, time.monotonic() + 5, {})
    assert helper.validate_proof(proof) is True
    assert proof["source_proof"]["recorded_producer_commit"] == helper.PRODUCER
    assert proof["source_proof"]["reviewed_template_commit"] == helper.PRODUCER
    assert proof["source_proof"]["reviewed_AST_sha256s"] == helper.reviewed.BASE_TEMPLATES
    assert proof["reviewed_producer_registry"]["registry_sha256"] == helper.REGISTRY_SHA
    assert helper.PRODUCER not in helper.reviewed.WORKER_VARIANTS
    assert "PRIVATE" not in json.dumps(proof)


def test_old_graph_producer_not_reexecuted_accidentally(parquet_case):
    item, roots, worker = parquet_case
    value = json.loads(worker.read_text())
    value["source_sha"] = helper.reviewed.BASE
    worker.write_text(json.dumps(value))
    item["file_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="new V6 producer"):
        helper.prove(item, roots, time.monotonic() + 5, {})


def test_missing_audit_binding_fails_registration(parquet_case):
    item, roots, worker = parquet_case
    value = json.loads(worker.read_text())
    value["records"][0]["sha256"] = "a" * 64
    worker.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="hash changed"):
        helper.expected_file_sha(item, roots, time.monotonic() + 5)


@pytest.mark.parametrize("kind", ["single_json_document", "arbitrary_Parquet", "full_scan"])
def test_only_two_registered_projection_kinds(yaml_case, kind):
    item, roots = yaml_case
    item["kind"] = kind
    with pytest.raises(ValueError, match="only registered"):
        helper.expected_file_sha(item, roots, time.monotonic() + 5)


def test_invalid_outer_scope_is_rejected(yaml_case):
    item, roots = yaml_case
    item["file_sha256"] = helper.expected_file_sha(item, roots, time.monotonic() + 5)
    proof = helper.prove(item, roots, time.monotonic() + 5)
    proof["residual_projection"]["fresh_confirmation_authorized"] = True
    with pytest.raises(ValueError, match="scope"):
        helper.validate_proof(proof)
