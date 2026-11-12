"""File wrapper parity, source isolation and bounded output fixtures."""

import hashlib
import json
import time

import evicastkv_bounded_json_v44 as wrapper
import evicastkv_identity_sink_v44 as sink
import evicastkv_review_freshness_stream as original
import pytest


def run(tmp_path, raw):
    source = tmp_path / "synthetic.json"
    source.write_bytes(raw)
    output = tmp_path / "output"
    output.mkdir()
    row = dict(path=str(source), signature=wrapper.geometry(source.lstat()))
    proof = wrapper.project_file(row, [str(tmp_path)], time.monotonic() + 30, output)
    actual = {kind: set() for kind in sink.KINDS}
    for ref in proof["identity_shards"]:
        value = sink.read_shard(ref)
        for item in value["values"]:
            actual[value["kind"]].add(tuple(item) if value["kind"] in sink.WIDTHS else item)
    expected = {kind: [] for kind in sink.KINDS} | dict(projected_records=0)
    original.collect(original.identity_projection(raw), expected, source)
    assert actual == {
        kind: {tuple(x) if kind in sink.WIDTHS else x for x in expected[kind]}
        for kind in sink.KINDS
    }
    assert proof["whole_file_sha256"] == hashlib.sha256(raw).hexdigest()
    assert proof["statistics"]["maximum_live_projected_metadata_bytes"] <= 8 * 2**20
    assert proof["statistics"]["maximum_live_projected_metadata_nodes"] <= 100000
    assert len(list(output.iterdir())) == len(proof["identity_shards"])
    return proof


@pytest.mark.parametrize(
    "raw",
    [
        b"null",
        b"[]",
        b'{"answer":"PRIVATE"}',
        b'{"rows":[{"sample_id":"a"},{"sample_id":"b"}],"dataset":{"id":"late","split":"train"}}',
        b'{"media":{"sha256":"' + b"a" * 64 + b'"},"question":"PRIVATE"}',
    ],
)
def test_complete_file_hash_and_exact_metadata_only_identity_sets(tmp_path, raw):
    proof = run(tmp_path, raw)
    assert "PRIVATE" not in json.dumps(proof)
    assert proof["hash_passes"] == 1
    assert proof["independent_second_pass_verified"] is False
    stats = proof["statistics"]
    assert stats["complete_source_parse_seconds"] >= 0
    assert stats["identity_replay_and_export_seconds"] >= 0
    assert (
        stats["complete_source_parse_seconds"] + stats["identity_replay_and_export_seconds"]
        <= proof["wall_seconds"]
    )


def test_source_signature_failure_never_opens_output(tmp_path):
    source = tmp_path / "synthetic.json"
    source.write_bytes(b"{}")
    row = dict(path=str(source), signature=wrapper.geometry(source.lstat()))
    source.write_bytes(b"changed")
    with pytest.raises(wrapper.ProjectionFailure):
        wrapper.project_file(row, [str(tmp_path)], time.monotonic() + 5, tmp_path)


def test_malformed_tail_does_not_publish_complete_proof(tmp_path):
    with pytest.raises(wrapper.ProjectionFailure) as caught:
        run(tmp_path, b'{"sample_id":"x"} broken')
    assert caught.value.statistics["failure_phase"] == "source_parse"


def test_replay_failure_records_phase_without_publishing_a_proof(tmp_path, monkeypatch):
    def stop(*args, **kwargs):
        raise TimeoutError("synthetic bounded replay deadline")

    monkeypatch.setattr(wrapper.sink, "collect_to_shards", stop)
    with pytest.raises(wrapper.ProjectionFailure) as caught:
        run(tmp_path, b'{"sample_id":"x"}')
    assert caught.value.statistics["failure_phase"] == "identity_replay_and_export"
    assert caught.value.statistics["bytes_read"] == len(b'{"sample_id":"x"}')


def test_more_than_100000_expanded_nodes_use_bounded_external_records(tmp_path):
    raw = json.dumps(
        dict(
            events=[dict(id=i) for i in range(40000)],
            dataset=dict(id="synthetic-dataset", split="train"),
        )
    ).encode()
    proof = run(tmp_path, raw)
    assert proof["identity_counts"]["sample_ids"] == 40001
    assert proof["statistics"]["spill_files"] > 0
