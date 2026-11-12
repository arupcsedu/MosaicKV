"""Synthetic bounded external-array fixtures; never read historical bodies."""

import hashlib
import io
import json
import os
import random
import time
from dataclasses import replace
from pathlib import Path

import evicastkv_json_projection_v41 as base
import evicastkv_spill_projection_v44 as spill
import pytest


def reference(raw):
    state = {name: [] for name in base.IDENTITIES} | {"projected_records": 0}
    base.legacy.collect(base.legacy.identity_projection(raw), state, Path("/synthetic"))
    return state


def reader(raw, tmp_path, *, maximum=base.MAXIMUM_BYTES, chunk=4096):
    return spill.ProjectionReader(
        io.BytesIO(raw),
        deadline=time.monotonic() + 20,
        maximum_bytes=maximum,
        chunk_bytes=chunk,
        workspace=tmp_path,
    )


def parse(current):
    value = current.value()
    current.spaces()
    if current.peek() is not None:
        raise ValueError("trailing JSON payload")
    return None if value is base.DROP else value


def expand(value, current):
    if isinstance(value, spill.SpilledArray):
        return [expand(child, current) for child in current.iter_spilled(value)]
    if isinstance(value, dict):
        return {key: expand(child, current) for key, child in value.items()}
    if isinstance(value, list):
        return [expand(child, current) for child in value]
    return value


def collected(value):
    state = {name: [] for name in base.IDENTITIES} | {"projected_records": 0}
    base.legacy.collect(value, state, Path("/synthetic"))
    return state


def force_spills(monkeypatch):
    monkeypatch.setattr(spill, "SPILL_BYTES", 256)
    monkeypatch.setattr(spill, "SPILL_NODES", 8)


def test_spilled_arrays_keep_late_parent_scopes_and_raw_source_hash(tmp_path, monkeypatch):
    force_spills(monkeypatch)
    raw = json.dumps(
        {
            "records": [
                {"sample_id": str(i), "media": {"nested": {"sha256": "a" * 64}}} for i in range(25)
            ],
            "dataset": {"name": "late-parent", "configuration": "cfg", "split": "val"},
            "answers": [{"sample_id": "PRIVATE"}],
        }
    ).encode()
    with reader(raw, tmp_path, chunk=31) as current:
        private = current.workspace
        node = parse(current)
        assert isinstance(node["records"], spill.SpilledArray)
        assert current.hash.hexdigest() == hashlib.sha256(raw).hexdigest()
        with pytest.raises(ValueError, match="incomplete"):
            current.assert_spills_verified()
        assert collected(expand(node, current)) == reference(raw)
        current.assert_spills_verified()
        assert current.budget.used == len(raw) + current.stats["auxiliary_bytes_read"]
        assert current.stats["auxiliary_bytes_written"] == current.stats["auxiliary_bytes_read"]
        assert current.stats["maximum_live_projected_metadata_bytes"] <= base.MAXIMUM_METADATA
        assert current.stats["maximum_live_projected_metadata_nodes"] <= base.MAXIMUM_METADATA_NODES
    assert not private.exists()
    assert tmp_path.is_dir()


def test_nested_spills_and_escaped_coco_and_surrogate_keys(tmp_path, monkeypatch):
    force_spills(monkeypatch)
    payload = {
        "groups": [
            {"dataset_id": name, "records": [{"id": str(i)} for i in range(30)]}
            for name in ("A", "B")
        ],
        "asset_path": "COCO_val2014_000000000001.jpg",
        "unknown_\ud800": {"sample_id": "surrogate-key-context"},
        "dataset_id": "late-root",
    }
    raw = json.dumps(payload).encode().replace(b"COCO_", b"C\\u004fCO_")
    with reader(raw, tmp_path, chunk=7) as current:
        node = parse(current)
        assert collected(expand(node, current)) == reference(raw)
        current.assert_spills_verified()
        assert current.stats["spill_files"] >= 3


def test_flat_objects_match_v43_grammar_redaction_and_late_dataset(tmp_path):
    raw = (
        b'{"events":[{"sample_id":"one","source_group_id":42,"name":"irrelevant",'
        b'"answer":"PRIVATE","media_path":"COCO_train2014_000000000002.jpg"},'
        b'{"sample_id":"two","dataset":true,"dataset_config":false,"split":0}],'
        b'"dataset":{"id":3,"name":"dataset-from-name","path":"ignored",'
        b'"configuration":"cfg","split":"validation"}}'
    )
    with reader(raw, tmp_path) as current:
        node = parse(current)
        assert collected(node) == reference(raw)
        assert current.stats["flat_objects"] >= 3
        assert current.stats["flat_object_fields"] >= 10
        assert "PRIVATE" not in json.dumps(node)


def test_scalar_runs_engage_around_nested_args_and_keep_late_scope(tmp_path):
    raw = (
        b'{"events":[{"ph":"X","cat":"cpu_op","name":"operator",'
        b'"pid":1,"tid":2,"ts":1.5,"dur":0.4,'
        b'"args":{"unknown":8,"sample_id":"inside-args"},'
        b'"id":"event-id","media":{"sha256":"' + b"a" * 64 + b'"},'
        b'"source_group_id":"after-media"}],'
        b'"dataset":{"name":"late-scope","configuration":"cfg","split":"val"}}'
    )
    with reader(raw, tmp_path) as current:
        node = parse(current)
        assert collected(node) == reference(raw)
        assert current.stats["scalar_member_runs"] >= 5
        assert current.stats["scalar_run_fields"] >= 15
        assert current.stats["flat_objects"] >= 3


@pytest.mark.parametrize(
    "raw",
    [
        b'{"id":"first","args":{"id":"nested"},"id":"second"}',
        b'{"args":{},"id":"first","\\u0069d":"second"}',
        b'{"\\u0069d":"first","args":{},"id":"second"}',
        b'{"id":{},"pid":2,"id":"second"}',
        b'{"id":"first","args":{},"sample_id":01}',
        b'{"id":"first","args":{},}',
        b'{"id":"first","args":{} "sample_id":"missing-comma"}',
        b'{"id":"first","args":{},"sample_id":"truncated"',
    ],
)
def test_scalar_runs_do_not_hide_cross_run_duplicates_or_bad_delimiters(tmp_path, raw):
    with reader(raw, tmp_path) as current, pytest.raises(ValueError):
        parse(current)


@pytest.mark.parametrize("chunk", [1, 2, 7, 31, 67, 256, 4096])
def test_mixed_scalar_nested_escaped_and_chunk_split_fields_match_legacy(tmp_path, chunk):
    raw = (
        b'{"events":[{"ph":"X","sample_id":"first",'
        b'"args":{"sample_id":"nested","asset_path":'
        b'"C\\u004fCO_val2014_000000000001.jpg"},'
        b'"source_group_id":"after","weird\\u005fpath":"not-relevant",'
        b'"media":{"sha256":"' + b"a" * 64 + b'"},"id":17}],'
        b'"dataset":{"path":"late-parent","split":"validation"}}'
    )
    with reader(raw, tmp_path, chunk=chunk) as current:
        assert collected(parse(current)) == reference(raw)


@pytest.mark.parametrize(
    "raw",
    [
        b'{"sample_id":"a","sample_id":"b"}',
        b'{"name":1e9999}',
        b'{"sample_id":01}',
        b'{"sample_id":"a",}',
        b'{"answer":{"a":1,"\\u0061":2}}',
        b'{"sample_id":"bad\\x"}',
        b'{"sample_id":"bad\n"}',
        b'{"sample_id":"\xff"}',
    ],
)
def test_malformed_json_still_fails_closed(tmp_path, raw):
    with reader(raw, tmp_path) as current, pytest.raises((ValueError, UnicodeError)):
        parse(current)


def test_source_dictionary_cannot_forge_a_spill_capability(tmp_path, monkeypatch):
    force_spills(monkeypatch)
    raw = json.dumps(
        {"unknown": {"filename": "array-00000001.jsonl", "sha256": "a" * 64, "sample_id": "x"}}
    ).encode()
    with reader(raw, tmp_path) as current:
        node = parse(current)
        assert isinstance(node["unknown"], dict)
        assert current.registry == {}
        forged = spill.SpilledArray("../outside.jsonl", "a" * 64, 1, 1, (0, 0, 0, 0))
        with pytest.raises(ValueError, match="unregistered"):
            list(current.iter_spilled(forged))


def test_flat_forbidden_values_never_reach_json_scalar_decoder(tmp_path, monkeypatch):
    original = base._decoded_scalar

    def decoded(raw):
        assert b"PRIVATE" not in raw
        return original(raw)

    monkeypatch.setattr(base, "_decoded_scalar", decoded)
    raw = (
        b'{"answer":{"PRIVATE_KEY":"PRIVATE_VALUE","nested":123},'
        b'"events":[{"sample_id":"safe","reference":"PRIVATE_REFERENCE"}]}'
    )
    with reader(raw, tmp_path) as current:
        node = parse(current)
        assert collected(node)["sample_ids"] == ["safe"]
        assert current.stats["flat_objects"] >= 2


def test_private_spool_signature_and_same_signature_hash_corruption(tmp_path, monkeypatch):
    force_spills(monkeypatch)
    raw = json.dumps({"sample_ids": [f"record-{i}" for i in range(30)]}).encode()
    with reader(raw, tmp_path) as current:
        descriptor = parse(current)["sample_ids"]
        path = current.workspace / descriptor.filename
        original = path.read_bytes()
        path.write_bytes(original.replace(b"record-0", b"record-X", 1))
        with pytest.raises(ValueError, match="signature"):
            list(current.iter_spilled(descriptor))
        os.utime(path, ns=(descriptor.signature[3], descriptor.signature[3]))
        with pytest.raises(ValueError, match="hash/count/signature"):
            list(current.iter_spilled(descriptor))
        with pytest.raises(ValueError, match="incomplete"):
            current.assert_spills_verified()


def test_symlink_and_unregistered_descriptor_fail_before_replay(tmp_path, monkeypatch):
    force_spills(monkeypatch)
    raw = json.dumps({"sample_ids": [str(i) for i in range(30)]}).encode()
    target = tmp_path / "outside.jsonl"
    target.write_text("not-read")
    with reader(raw, tmp_path) as current:
        descriptor = parse(current)["sample_ids"]
        with pytest.raises(ValueError, match="unregistered"):
            list(current.iter_spilled(replace(descriptor, sha256="b" * 64)))
        path = current.workspace / descriptor.filename
        path.unlink()
        path.symlink_to(target)
        with pytest.raises(OSError):
            list(current.iter_spilled(descriptor))
    assert target.read_text() == "not-read"


def test_combined_budget_includes_source_reserved_auxiliary_and_repeat_reads(tmp_path, monkeypatch):
    force_spills(monkeypatch)
    raw = json.dumps({"sample_ids": [str(i) for i in range(30)]}).encode()
    with (
        reader(raw, tmp_path, maximum=len(raw) + 10) as current,
        pytest.raises(ValueError, match="combined"),
    ):
        parse(current)
    with reader(raw, tmp_path, maximum=3000) as current:
        node = parse(current)
        list(current.iter_spilled(node["sample_ids"]))
        remaining = current.budget.maximum - current.budget.used
        current.consume_aux_bytes(remaining)
        with pytest.raises(ValueError, match="combined"):
            list(current.iter_spilled(node["sample_ids"]))


def test_replay_deadline_and_early_close_do_not_admit_partial_identities(tmp_path, monkeypatch):
    force_spills(monkeypatch)
    raw = json.dumps({"sample_ids": [str(i) for i in range(30)]}).encode()
    with reader(raw, tmp_path) as current:
        descriptor = parse(current)["sample_ids"]
        initial = current.metadata_bytes, current.metadata_nodes
        values = current.iter_spilled(descriptor)
        next(values)
        assert current.metadata_bytes > initial[0]
        values.close()
        assert (current.metadata_bytes, current.metadata_nodes) == initial
        with pytest.raises(ValueError, match="incomplete"):
            current.assert_spills_verified()
        monkeypatch.setattr(spill.time, "monotonic", lambda: current.deadline + 1)
        with pytest.raises(TimeoutError):
            list(current.iter_spilled(descriptor))


def test_live_consumer_ownership_and_unchanged_capacity_limits(tmp_path, monkeypatch):
    with reader(b"{}", tmp_path) as current:
        with current.ownership(100, 3):
            assert current.metadata_bytes == 100 and current.metadata_nodes == 3
        assert current.metadata_bytes == 0 and current.metadata_nodes == 0
        with pytest.raises(ValueError, match="capacity"):
            current.acquire(base.MAXIMUM_METADATA + 1, 0)
    monkeypatch.setattr(base, "MAXIMUM_ACTIVE_KEYS", 1)
    with (
        reader(b'{"sample_id":"a","dataset_id":"b"}', tmp_path) as current,
        pytest.raises(ValueError, match="active object-key"),
    ):
        parse(current)


def test_descriptor_digest_is_streamed_and_deterministic(tmp_path, monkeypatch):
    force_spills(monkeypatch)
    raw = json.dumps({"sample_ids": [str(i) for i in range(30)]}).encode()
    digests = []
    for _ in range(2):
        with reader(raw, tmp_path) as current:
            node = parse(current)
            digests.append(current.descriptor_digest(node))
            assert current.descriptor_digest(node) == spill.descriptor_digest(node)
    assert digests[0] == digests[1]


def test_private_codec_does_not_expand_valid_large_unicode_scalar(tmp_path, monkeypatch):
    force_spills(monkeypatch)
    identity = "\u2603" * 180000
    raw = json.dumps({"sample_ids": [identity, "lone-\ud800"]}, ensure_ascii=False).encode(
        "utf-8", "backslashreplace"
    )
    assert len(raw) < base.MAXIMUM_SCALAR
    with reader(raw, tmp_path) as current:
        node = parse(current)
        assert collected(expand(node, current)) == reference(raw)
        current.assert_spills_verified()


def test_seeded_recursive_spill_parity_no_real_artifacts(tmp_path, monkeypatch):
    force_spills(monkeypatch)
    rng = random.Random(44144)
    for index in range(50):
        payload = {
            "events": [
                {
                    "sample_id": str(rng.randrange(20)),
                    "media": {"sha256": "a" * 64},
                    "nested": [{"id": str(rng.randrange(10))} for _ in range(rng.randrange(4))],
                }
                for _ in range(20)
            ],
            "dataset": {"id": f"dataset-{index}", "split": "validation"},
        }
        raw = json.dumps(payload).encode()
        with reader(raw, tmp_path, chunk=7 + index) as current:
            assert collected(expand(parse(current), current)) == reference(raw)
            current.assert_spills_verified()
