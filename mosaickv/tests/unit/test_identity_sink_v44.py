"""Bounded synthetic sink tests; no historical bodies or scheduler access."""

import hashlib
import io
import json
import time
from pathlib import Path

import evicastkv_identity_sink_v44 as sink
import evicastkv_spill_projection_v44 as projection
import pytest


class BudgetReader:
    def __init__(self, *, maximum=2 * 2**30, metadata=8 * 2**20, nodes=100000):
        self.maximum, self.metadata_limit, self.node_limit = maximum, metadata, nodes
        self.bytes = self.live = self.nodes = self.peak = self.peak_nodes = 0
        self.deadline = time.monotonic() + 30

    def acquire(self, amount, nodes=1):
        if self.live + amount > self.metadata_limit or self.nodes + nodes > self.node_limit:
            raise ValueError("bounded projected metadata capacity exceeded")
        self.live += amount
        self.nodes += nodes
        self.peak = max(self.peak, self.live)
        self.peak_nodes = max(self.peak_nodes, self.nodes)

    def release(self, amount, nodes=1):
        self.live -= amount
        self.nodes -= nodes
        assert self.live >= 0 and self.nodes >= 0

    def consume_aux_bytes(self, amount):
        if self.bytes + amount > self.maximum:
            raise ValueError("combined source/auxiliary input byte bound exceeded")
        self.bytes += amount

    def check_deadline(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("bounded streaming metadata deadline reached")

    def assert_spills_verified(self):
        pass


def _reference(node):
    state = dict.fromkeys(sink.KINDS)
    state.update({name: [] for name in sink.KINDS}, projected_records=0)
    sink.legacy.collect(node, state, Path("/synthetic"))
    return json.loads(json.dumps(state))


def _aggregate(result):
    values = {name: set() for name in sink.KINDS}
    for reference in result["identity_shards"]:
        shard = sink.read_shard(reference)
        for item in shard["values"]:
            values[shard["kind"]].add(tuple(item) if isinstance(item, list) else item)
    return {
        name: [list(item) if isinstance(item, tuple) else item for item in sorted(items)]
        for name, items in values.items()
    } | {"projected_records": result["projected_records"]}


def _run(node, tmp_path, reader=None, **kwargs):
    workspace, output = tmp_path / "work", tmp_path / "shards"
    workspace.mkdir()
    output.mkdir()
    reader = reader or BudgetReader()
    result = sink.collect_to_shards(node, reader, workspace, output, **kwargs)
    assert list(workspace.iterdir()) == []
    return result, reader


@pytest.mark.parametrize(
    "dataset",
    [
        {"id": "preferred", "name": "second", "path": "third"},
        {"id": 7, "name": "second", "path": "third"},
        {"id": False, "name": None, "path": "third"},
        "plain-dataset",
        True,
        17,
        None,
    ],
)
def test_exact_legacy_scope_priority_media_bool_and_integer_semantics(tmp_path, dataset):
    node = {
        "records": [
            {"sample_ids": ["one", "one", 1, True, False, None], "episode_id": "ep"},
            {"question_id": 12, "media": {"sha256": "input-media:" + "a" * 64}},
            {"id": "COCO_val2014_000000000009.jpg", "sha256": "b" * 64},
            {"media_group_ids": ["original-image:" + "c" * 64, "not-a-hash"]},
            {"dataset_id": "nested", "dataset_config": "n", "split": True, "id": "z"},
        ],
        # These appear after the array; replay must still scope earlier IDs.
        "dataset": dataset,
        "dataset_name": "fallback-name",
        "task": "fallback-task",
        "split": False,
        "dataset_config": "explicit",
    }
    if isinstance(dataset, dict):
        dataset.update(split="validation", configuration="native", config_name="not-config")
    result, reader = _run(node, tmp_path, target_bytes=128)
    assert _aggregate(result) == _reference(node)
    assert result["identity_counts"] == {name: len(_reference(node)[name]) for name in sink.KINDS}
    assert result["stats"]["duplicate_identities"] > 0
    assert reader.peak <= 8 * 2**20 and reader.peak_nodes <= 100000


def test_bounded_many_distinct_ids_and_typed_shard_hashes(tmp_path):
    node = {"ids": [f"identity-{number:06d}" for number in range(2000)], "dataset_id": "synthetic"}
    result, reader = _run(node, tmp_path, target_bytes=1024)
    assert _aggregate(result) == _reference(node)
    assert len(result["identity_shards"]) > 10
    assert reader.peak < 1024 * 1024
    assert reader.bytes == result["stats"]["sqlite_export_bytes"]
    for reference in result["identity_shards"]:
        assert reference["bytes"] <= 1024
        actual = hashlib.sha256(Path(reference["path"]).read_bytes()).hexdigest()
        assert actual == reference["sha256"]


@pytest.mark.parametrize("node", [None, {}, [], {"unknown": [False, None, 1.5]}])
def test_empty_projection_has_six_zero_counts_and_one_record(tmp_path, node):
    result, _ = _run(node, tmp_path)
    assert result["identity_shards"] == []
    assert result["identity_counts"] == dict.fromkeys(sink.KINDS, 0)
    assert _aggregate(result) == _reference(node)


def test_escaped_unicode_nul_and_sql_text_are_data(tmp_path):
    node = {"ids": ["a\x00b", '"; DROP TABLE ids; --', "é\U0001f643", "a\\b", "a\nb", "\ud800"]}
    result, _ = _run(node, tmp_path)
    assert _aggregate(result) == _reference(node)


@pytest.mark.parametrize("limit", [1, 500])
def test_auxiliary_read_bound_fails_closed_and_cleans_private_database(tmp_path, limit):
    workspace, output = tmp_path / "work", tmp_path / "shards"
    workspace.mkdir()
    output.mkdir()
    with pytest.raises(ValueError, match="byte bound"):
        sink.collect_to_shards(
            {"ids": [str(n) for n in range(100)]},
            BudgetReader(maximum=limit),
            workspace,
            output,
        )
    assert list(workspace.iterdir()) == []


def test_deadline_and_metadata_bounds_are_unchanged(tmp_path):
    for name, reader in (
        ("expired", BudgetReader()),
        ("memory", BudgetReader(metadata=sink.SQLITE_RESERVATION - 1)),
        ("nodes", BudgetReader(nodes=15)),
    ):
        workspace, output = tmp_path / name, tmp_path / (name + "-out")
        workspace.mkdir()
        output.mkdir()
        if name == "expired":
            reader.deadline = time.monotonic() - 1
        with pytest.raises((ValueError, TimeoutError), match=r"capacity|deadline"):
            sink.collect_to_shards({"id": "x"}, reader, workspace, output)
        assert list(workspace.iterdir()) == []


def test_tampered_shard_is_rejected_and_existing_output_never_overwritten(tmp_path):
    result, _ = _run({"id": "x"}, tmp_path)
    reference = result["identity_shards"][0]
    path = Path(reference["path"])
    original = path.read_bytes()
    with path.open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="byte/hash"):
        sink.read_shard(reference)
    assert path.read_bytes() == original + b" "
    with pytest.raises(FileExistsError):
        sink.collect_to_shards({"id": "x"}, BudgetReader(), tmp_path / "work", tmp_path / "shards")
    assert path.read_bytes() == original + b" "


@pytest.mark.parametrize("mutation", ["kind", "count", "typed", "version"])
def test_rehashed_but_invalid_typed_shard_fails(tmp_path, mutation):
    result, _ = _run({"id": "x"}, tmp_path)
    reference = dict(result["identity_shards"][0])
    path = Path(reference["path"])
    content = json.loads(path.read_bytes())
    if mutation == "kind":
        content["kind"] = "answers"
    elif mutation == "count":
        reference["count"] += 1
    elif mutation == "typed":
        content["values"] = [True]
    else:
        content["version"] = True
    raw = json.dumps(content).encode()
    path.unlink()
    with path.open("xb") as stream:
        stream.write(raw)
    reference.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError, match=r"typed|identity"):
        sink.read_shard(reference)


def test_spilled_arrays_replay_late_scopes_and_release_children(tmp_path, monkeypatch):
    monkeypatch.setattr(projection, "SPILL_NODES", 10)
    raw_node = {
        "records": [{"id": str(n), "media": {"sha256": "f" * 64}} for n in range(300)],
        "dataset": {"id": "late-parent", "split": "validation", "configuration": "c"},
    }
    raw = json.dumps(raw_node).encode()
    workspace, output = tmp_path / "work", tmp_path / "shards"
    workspace.mkdir()
    output.mkdir()
    with projection.ProjectionReader(
        io.BytesIO(raw),
        deadline=time.monotonic() + 30,
        maximum_bytes=2 * 2**30,
        chunk_bytes=4096,
        workspace=workspace,
    ) as reader:
        node = reader.value()
        reader.spaces()
        assert reader.peek() is None
        result = sink.collect_to_shards(node, reader, workspace, output)
        reader.assert_spills_verified()
        assert result["stats"]["sqlite_export_bytes"] > 0
        assert _aggregate(result) == _reference(raw_node)
        assert reader.stats["completed_spill_replays"] > 0
