"""Synthetic pinned-producer tests; no historical artifacts or model execution."""

import copy
import json
import time

import evicastkv_reviewed_producers_v39 as registry
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return registry.original.sha(path)


def synthetic_source(revision):
    return dict(
        recorded_producer_commit=revision,
        reviewed_template_commit=revision,
        reviewed_AST_sha256s=registry.reviewed_ast_map(revision),
        source_blob_sha256s={key.split("::")[0]: "e" * 64 for key in registry.BASE_TEMPLATES},
        historical_source_executed=False,
    )


@pytest.fixture
def chain(tmp_path, monkeypatch):
    registry._ATTESTED_SOURCES.clear()
    original = registry.original
    owner = tmp_path / "history/outputs/attempt"
    key = "000-method-40-r0"
    paths = [owner / "mechanism-terms" / key / name for name in sorted(original.NAMES)]
    kinds = {
        "int64": pa.int64(),
        "double": pa.float64(),
        "bool": pa.bool_(),
        "large_string": pa.large_string(),
    }
    values = {"int64": 0, "double": 0.5, "bool": True, "large_string": "internal_enum"}
    for path in paths:
        fields, arrays = [], []
        for name, accepted in original.SCHEMAS[path.name].items():
            kind = next(value for value in kinds if value in accepted)
            fields.append(pa.field(name, kinds[kind]))
            arrays.append(pa.array([values[kind]], type=kinds[kind]))
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_arrays(arrays, schema=pa.schema(fields)), path)
    worker, record, audit = (
        owner / "worker.json",
        owner / "records" / f"{key}.json",
        owner / "mechanisms" / f"{key}.json",
    )

    def bind(revision):
        audit_sha = put(
            audit,
            dict(
                raw_terms={str(path): original.sha(path) for path in paths}, prompt="PRIVATE_PROMPT"
            ),
        )
        record_sha = put(
            record,
            dict(
                source_sha=revision,
                mechanism_audit_path=str(audit),
                mechanism_audit_sha256=audit_sha,
                answer="PRIVATE_ANSWER",
            ),
        )
        put(worker, dict(source_sha=revision, records=[dict(path=str(record), sha256=record_sha)]))

    called = []

    def producer(revision, deadline):
        assert registry.reviewed_ast_map(revision) == original.TEMPLATES
        assert revision == original.REFERENCE_PRODUCER
        called.append(revision)
        return synthetic_source(revision)

    monkeypatch.setattr(original, "producer_proof", producer)
    monkeypatch.setattr(
        pq, "read_table", lambda *args, **kwargs: pytest.fail("row decoding forbidden")
    )
    yield dict(
        owner=owner,
        paths=paths,
        bind=bind,
        called=called,
        worker=worker,
        row=lambda path: dict(
            path=str(path),
            signature=original.scanner.signature(path),
            original_error_sha256s=["f" * 64],
        ),
    )
    registry._ATTESTED_SOURCES.clear()


@pytest.mark.parametrize("revision", list(registry.WORKER_VARIANTS))
def test_each_exact_source_uses_actual_AST_and_unchanged_transaction(chain, revision):
    chain["bind"](revision)
    before = copy.deepcopy(registry.original.TEMPLATES)
    cache = {}
    for path in chain["paths"]:
        proof = registry.prove(
            chain["row"](path), [chain["owner"].parents[1]], time.monotonic() + 10, cache
        )
        assert registry.validate_proof(proof) is True
        assert (
            proof["source_proof"]["reviewed_AST_sha256s"][registry.WORKER]
            == registry.WORKER_VARIANTS[revision]
        )
        assert proof["source_proof"]["reviewed_template_commit"] == revision
        assert proof["source_proof"]["recorded_producer_commit"] == revision
        assert proof["parquet_footer"]["rows_decoded"] == 0
        assert "PRIVATE" not in json.dumps(proof)
        assert proof["input_independence_or_global_freshness_claimed"] is False
    assert registry.original.TEMPLATES == before == registry.BASE_TEMPLATES
    assert chain["called"] == [revision]


def test_exact_sources_keep_other_seven_constraints():
    for revision in registry.WORKER_VARIANTS:
        actual = registry.reviewed_ast_map(revision)
        assert {key: value for key, value in actual.items() if key != registry.WORKER} == {
            key: value for key, value in registry.BASE_TEMPLATES.items() if key != registry.WORKER
        }
        assert len(actual) == 8


def test_unknown_source_rejected_before_artifact_hash(chain, monkeypatch):
    chain["bind"]("a" * 40)
    monkeypatch.setattr(
        registry.original, "hash_file", lambda *args: pytest.fail("artifact read forbidden")
    )
    with pytest.raises(ValueError, match="three reviewed"):
        registry.prove(
            chain["row"](chain["paths"][0]), [chain["owner"].parents[1]], time.monotonic() + 10, {}
        )


def good_proof(chain):
    revision = next(value for value in registry.WORKER_VARIANTS if value != registry.BASE)
    chain["bind"](revision)
    return registry.prove(
        chain["row"](chain["paths"][0]), [chain["owner"].parents[1]], time.monotonic() + 10, {}
    )


@pytest.mark.parametrize(
    "mutator",
    [
        lambda p: p["source_proof"]["reviewed_AST_sha256s"].update(
            {registry.WORKER: registry.WORKER_VARIANTS[registry.BASE]}
        ),
        lambda p: p["source_proof"].update(reviewed_template_commit=registry.BASE),
        lambda p: p["source_proof"].update(recorded_producer_commit="a" * 40),
        lambda p: p["source_proof"].update(historical_source_executed=True),
        lambda p: p["source_proof"]["reviewed_AST_sha256s"].pop(
            next(k for k in registry.BASE_TEMPLATES if k != registry.WORKER)
        ),
        lambda p: p["parquet_footer"].update(rows_decoded=1),
        lambda p: p["parquet_footer"].update(schema_sha256="a" * 64),
        lambda p: p.update(whole_file_sha256_after="a" * 64),
        lambda p: p.update(input_independence_or_global_freshness_claimed=True),
        lambda p: p["bindings"].clear(),
        lambda p: p.pop("reviewed_producer_registry"),
    ],
)
def test_analyzer_rejects_relabeling_or_weakened_guards(chain, mutator):
    proof = good_proof(chain)
    mutator(proof)
    with pytest.raises(ValueError):
        registry.validate_proof(proof)


def test_preexisting_fake_cache_is_not_trusted(chain):
    revision = registry.BASE
    chain["bind"](revision)
    bogus = synthetic_source(revision)
    bogus["source_blob_sha256s"] = {}
    with pytest.raises(ValueError, match="conflicts"):
        registry.prove(
            chain["row"](chain["paths"][0]),
            [chain["owner"].parents[1]],
            time.monotonic() + 10,
            {revision: bogus},
        )


def test_failure_restores_original_template_and_releases_dispatch_lock(chain):
    revision = next(value for value in registry.WORKER_VARIANTS if value != registry.BASE)
    chain["bind"](revision)
    row = chain["row"](chain["paths"][0])
    chain["paths"][0].write_bytes(b"changed")
    with pytest.raises(ValueError):
        registry.prove(row, [chain["owner"].parents[1]], time.monotonic() + 10, {})
    assert registry.original.TEMPLATES == registry.BASE_TEMPLATES
    assert registry.original.REFERENCE_PRODUCER == registry.BASE
    assert not registry._LOCK.locked()


def test_concurrent_dispatch_fails_closed(chain):
    registry._LOCK.acquire()
    try:
        with pytest.raises(RuntimeError, match="process-isolated"):
            registry.prove({}, [], time.monotonic() + 10, {})
    finally:
        registry._LOCK.release()
