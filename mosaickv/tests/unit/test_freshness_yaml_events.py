"""Synthetic event-only YAML projection; never parse historical YAML in tests."""

import json
import time
from pathlib import Path

import evicastkv_freshness_yaml_events as events
import pytest
import yaml
from evicastkv_review_freshness_stream import identity_projection


def project(raw):
    return events.projection(raw, deadline=time.monotonic() + 60)


def test_canonical_JSON_and_YAML_projections_have_identical_collector_semantics(monkeypatch):
    value = dict(
        dataset_id="lmms-lab/COCO-Caption",
        split="val",
        samples=[
            dict(
                sample_id="COCO_val2014_000000000001.jpg",
                media=[dict(path="image.png", sha256="a" * 64)],
                answer="PRIVATE_LABEL",
                prompt="PRIVATE_PROMPT",
                unknown_scalar="not-an-ID",
            )
        ],
        flags=dict(sample_ids=["001", "on", "42"], metadata=[True, False, None]),
    )
    rendered = yaml.safe_dump(value).encode()

    def forbidden(*args, **kwargs):
        pytest.fail("YAML object constructor/load API invoked")

    monkeypatch.setattr(yaml, "load", forbidden)
    monkeypatch.setattr(yaml, "safe_load", forbidden)
    monkeypatch.setattr(yaml.constructor.BaseConstructor, "construct_object", forbidden)
    monkeypatch.setattr(yaml.resolver.Resolver, "__init__", forbidden)
    observed = project(rendered)
    reference = identity_projection(json.dumps(value).encode())
    assert observed == reference
    left = {k: [] for k in events.IDENTITIES} | {"projected_records": 0}
    right = {k: [] for k in events.IDENTITIES} | {"projected_records": 0}
    events.collect(observed, left, Path("synthetic.yaml"))
    events.collect(reference, right, Path("synthetic.json"))
    assert left == right
    assert "PRIVATE_LABEL" not in json.dumps(observed)
    assert "PRIVATE_PROMPT" not in json.dumps(observed)


@pytest.mark.parametrize(
    "raw",
    [
        b"sample_id: !!str abc\n",
        b"answer: !!python/object/apply:os.system [echo SECRET]\n",
        b"sample_ids: &anchor [a, b]\nother: *anchor\n",
        b"cyclic: &cycle [*cycle]\n",
        b"sample_id: *missing\n",
        b"? [complex, key]\n: value\n",
        b"sample_id: a\nsample_id: b\n",
        b"sample_id: a\n---\nsample_id: b\n",
        b"%YAML 1.2\n---\nsample_id: a\n",
        b"sample_id: on\n",
        b"sample_id: 001\n",
        b"answer: [PRIVATE_LABEL\n",
        b"sample_id: \xff\n",
    ],
)
def test_unsupported_or_malicious_syntax_is_blocked_without_raw_error_snippets(raw):
    with pytest.raises(events.UnsupportedYAML) as raised:
        project(raw)
    assert "SECRET" not in str(raised.value)
    assert "PRIVATE_LABEL" not in str(raised.value)


def test_explicit_tag_like_quoted_string_is_not_executed():
    assert project(b"sample_id: '!!python/object:harmless-string'\n")["sample_id"].startswith(
        "!!python"
    )


def test_depth_event_size_and_parser_version_bounds(monkeypatch):
    monkeypatch.setattr(events, "MAX_DEPTH", 2)
    with pytest.raises(events.UnsupportedYAML, match="nesting_depth"):
        project(b"a: {b: {c: {sample_id: x}}}\n")
    monkeypatch.setattr(events, "MAX_EVENTS", 3)
    with pytest.raises(events.UnsupportedYAML, match="event_limit"):
        project(b"sample_id: x\n")
    monkeypatch.setattr(events, "MAX_BYTES", 1)
    with pytest.raises(events.UnsupportedYAML, match="file_size"):
        project(b"sample_id: x\n")


def test_unregistered_parser_version_is_rejected(monkeypatch):
    monkeypatch.setattr(yaml, "__version__", "unregistered")
    with pytest.raises(events.UnsupportedYAML, match="parser_version"):
        project(b"sample_id: known\n")


def test_deadline_is_checked_during_long_discarded_scalar_scan(monkeypatch):
    calls = []

    def clock():
        calls.append(1)
        return 0 if len(calls) < 40 else 10

    monkeypatch.setattr(events.time, "monotonic", clock)
    with pytest.raises(TimeoutError):
        events.projection(b"answer: '" + b"x" * 500000 + b"'\nsample_id: private\n", deadline=5)
    assert len(calls) <= 41


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return events.sha(path)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    data, parent, root = tmp_path / "old-data", tmp_path / "V11", tmp_path / "new-YAML"
    data.mkdir()
    files = []
    for index in range(8):
        path = data / f"config-{index}.yaml"
        path.write_text(f"sample_id: consumed-{index}\nanswer: PRIVATE_LABEL\n")
        files.append(path)
    put(parent / "plan.json", dict(source_sha="a" * 40, roots=[str(data)]))
    bound = {}
    for index, path in enumerate(files):
        state = dict(errors=[dict(path=str(path), error=events.YAML_ERROR)])
        target = parent / "workers" / f"shard-{index:03d}.json"
        bound[str(target)] = put(
            target, dict(source_sha="a" * 40, state=state, state_sha256=events.digest(state))
        )
    identities = {k: [] for k in events.IDENTITIES}
    identities["sample_ids"] = ["old-recovered-ID"]
    inventory_sha = put(parent / "combined_identity_inventory.json", dict(identities=identities))
    put(
        parent / "GATE_freshness_frontier_fork.json",
        dict(
            missing_shards=[],
            source_sha="a" * 40,
            frontier_exhausted=False,
            original_and_new_coverage_errors=8,
            worker_bindings=bound,
            identity_inventory_sha256=inventory_sha,
        ),
    )
    monkeypatch.setattr(events, "clean_source", lambda *args: "b" * 40)
    monkeypatch.setattr(events, "cpu_gate", lambda *args: dict(status="passed"))
    events.prepare(root, cpu_gate_root=tmp_path, parent=parent)
    return root, parent, files


def test_worker_hash_transactions_recover_metadata_without_mutating_original_errors(prepared):
    root, parent, _files = prepared
    old = events.sha(parent / "workers/shard-000.json")
    for index in range(8):
        events.worker(root, index)
    gate = events.analyze(root)
    assert gate["projected_files"] == 8 and gate["blocked_files"] == 0
    assert not gate["frontier_exhausted"] and gate["original_and_new_coverage_errors"] == 8
    assert gate["original_errors_preserved"] and gate["error_waivers"] == 0
    assert events.sha(parent / "workers/shard-000.json") == old
    inventory = events.read(root / "combined_identity_inventory.json")
    assert "old-recovered-ID" in inventory["identities"]["sample_ids"]
    assert not gate["fresh_confirmation_authorized"]
    for path in (root / "workers").iterdir():
        assert "PRIVATE_LABEL" not in path.read_text()


def test_source_mutation_blocks_file_without_identity_emission(prepared):
    root, _parent, files = prepared
    plan = events.load(root)
    row = next(r for r in plan["queue"] if r["path"] == str(files[0]))
    files[0].write_text("sample_id: changed\n")
    events.worker(root, row["index"])
    result = events.read(root / "workers" / f"shard-{row['index']}.json")
    assert any(r["row"]["path"] == str(files[0]) for r in result["blocked"])
    assert all(r["path"] != str(files[0]) for r in result["committed"])


def test_timeout_retains_exact_pending_queue_without_uncommitted_metadata(prepared, monkeypatch):
    root, _parent, _files = prepared
    row = events.load(root)["queue"][0]

    def pause(*args):
        raise TimeoutError("finite deadline")

    monkeypatch.setattr(events, "project_file", pause)
    events.worker(root, row["index"])
    result = events.read(root / "workers" / f"shard-{row['index']}.json")
    assert result["committed"] == [] and row in result["pending"]


def test_missing_workers_do_not_certify_projection_scope(prepared):
    gate = events.analyze(prepared[0])
    assert gate["missing_shards"] == list(range(8))
    assert gate["status"] == "bounded_YAML_event_projection_incomplete"


def test_post_projection_source_change_never_exports_a_transaction(prepared, monkeypatch):
    root, _parent, _files = prepared
    row = events.load(root)["queue"][0]
    original = events.read_bytes
    calls = []

    def reading(path, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            Path(path).write_text("sample_id: changed-after-projection\n")
        return original(path, **kwargs)

    monkeypatch.setattr(events, "read_bytes", reading)
    with pytest.raises(ValueError):
        events.project_file(row, time.monotonic() + 60)
    assert len(calls) == 2
