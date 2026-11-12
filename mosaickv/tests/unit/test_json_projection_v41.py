"""Strict streaming fixtures; no historical payload, dataset or GPU access."""

import hashlib
import io
import json
import random
import time
from pathlib import Path

import evicastkv_json_projection_v41 as parser
import pytest


def project(raw, *, chunk=31, maximum=None, statistics=None):
    return parser.identity_projection_stream(
        io.BytesIO(raw),
        deadline=time.monotonic() + 10,
        maximum_bytes=maximum or max(1, len(raw)),
        chunk_bytes=chunk,
        statistics=statistics,
    )


def reference(raw):
    value = parser.legacy.identity_projection(raw)
    state = {name: [] for name in parser.IDENTITIES} | {"projected_records": 0}
    parser.legacy.collect(value, state, Path("/synthetic"))
    return state


@pytest.mark.parametrize("chunk", [1, 2, 3, 7, 31, 64, 256])
def test_identity_parity_and_hash_across_all_boundaries(chunk):
    raw = json.dumps(
        {
            "records": [
                {"sample_id": "x", "media": {"sha256": "a" * 64}},
                {"question_id": 12, "source_group_id": "episode-1"},
            ],
            "dataset": {"id": "synthetic-coco", "split": "validation"},
            "unknown_tensor": [[0.5, 1, None, False], [True, -2.3e-8]],
            "answers": ["PRIVATE_ANSWER"],
            "prompt": "PRIVATE_PROMPT",
        },
        ensure_ascii=False,
    ).encode()
    output = project(raw, chunk=chunk)
    assert output["identities"] == reference(raw)
    assert output["whole_file_sha256"] == hashlib.sha256(raw).hexdigest()
    assert output["bytes_read"] == len(raw)
    assert output["hash_passes"] == 1
    assert output["independent_second_pass_verified"] is False
    assert output["forbidden_value_scalars_JSON_decoded"] is False
    assert output["statistics"]["maximum_buffer_bytes"] <= chunk + 6
    assert "PRIVATE" not in json.dumps(output)


def test_seeded_fixtures_match_existing_collector():
    rng = random.Random(20271112)
    for number in range(100):
        payload = {
            "dataset_id": "synthetic",
            "split": "calibration",
            "sample_ids": [str(number)],
            "records": [
                {"id": rng.randrange(100), "unknown": rng.random()},
                {"media_path": "COCO_val2014_000000000001.jpg"},
            ],
            "unknown": [[rng.randrange(100), False, None] for _ in range(3)],
            "reference": {"sample_id": "FORBIDDEN_ID", "caption": "PRIVATE"},
        }
        raw = json.dumps(payload, ensure_ascii=bool(number % 2)).encode()
        assert project(raw, chunk=1 + number % 43)["identities"] == reference(raw)


def test_unknown_numeric_array_does_not_construct_null_tree():
    raw = b'{"tensor":[' + b"1.25,-2e-3,true,false,null," * 100000 + b'0],"sample_id":"x"}'
    stats = {}
    output = project(raw, chunk=65536, statistics=stats)
    assert output["identities"]["sample_ids"] == ["x"]
    assert stats["numeric_run_values"] > 490000
    assert stats["projected_metadata_nodes"] < 10
    assert stats["work_units"] < 1000
    assert stats["maximum_buffer_bytes"] <= 65542


def test_forbidden_scalars_and_subtree_keys_never_reach_json_decoder(monkeypatch):
    raw = (
        b'{"answer":{"PRIVATE_KEY":"PRIVATE_VALUE","nested":["PRIVATE_ARRAY",123456789]},'
        b'"sample_id":"safe"}'
    )
    original = parser._decoded_scalar
    calls = []

    def check(value):
        assert b"PRIVATE" not in value and b"123456789" not in value
        calls.append(value)
        return original(value)

    monkeypatch.setattr(parser, "_decoded_scalar", check)
    output = project(raw, chunk=3)
    assert output["identities"]["sample_ids"] == ["safe"]
    assert calls


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"{",
        b"[",
        b'{"sample_id":}',
        b'{"a":1,}',
        b"[1,]",
        b"[1e,2]",
        b"[01,2]",
        b"[1.,2]",
        b"[+1]",
        b"[truefalse]",
        b"[NaN]",
        b"[Infinity]",
        b"{}{}",
        b'{"a" 1}',
        b'{"a":"bad\\x"}',
        b'{"a":"bad\\u12"}',
        b'{"a":"bad\n"}',
        b'{"a":1,"a":2}',
        b'{"answer":{"a":1,"\\u0061":2}}',
        b'{"sample_id":"\xff"}',
        b'{"sample_id":"\xe2\x82"}',
        b"\xef\xbb\xbf{}",
    ],
)
def test_malformed_duplicate_truncated_and_utf8_fail_closed(raw):
    with pytest.raises((ValueError, UnicodeError)):
        project(raw, chunk=2)


@pytest.mark.parametrize(
    "raw",
    [
        b'{"sample_id":"\\u00e9"}',
        '{"sample_id":"é 😀"}'.encode(),
        b'{"sample_id":"\\ud834\\udd1e"}',
        b'{"sample_id":"\\ud800"}',
        b'{"sample_id":"quote\\" slash\\\\"}',
    ],
)
def test_escaped_and_multibyte_metadata_matches_legacy(raw):
    for chunk in (1, 2, 5, 31):
        assert project(raw, chunk=chunk)["identities"] == reference(raw)


def test_equivalent_surrogate_and_utf8_object_keys_rejected():
    raw = ('{"answer":{"\\ud834\\udd1e":1,"𝄞":2}}').encode()
    with pytest.raises(ValueError, match="duplicate"):
        project(raw, chunk=3)


def test_expired_deadline_and_byte_limit_do_not_return_partial_ids():
    with pytest.raises(TimeoutError):
        parser.identity_projection_stream(
            io.BytesIO(b'{"id":"x"}'),
            deadline=time.monotonic() - 1,
            maximum_bytes=100,
            chunk_bytes=5,
        )
    with pytest.raises(ValueError, match="byte bound"):
        project(b'{"sample_id":"x"}', maximum=3)


@pytest.mark.parametrize(
    "bound",
    [
        "MAXIMUM_SCALAR",
        "MAXIMUM_METADATA",
        "MAXIMUM_ACTIVE_KEYS",
        "MAXIMUM_DEPTH",
        "MAXIMUM_METADATA_NODES",
    ],
)
def test_explicit_capacity_limits_fail_closed(monkeypatch, bound):
    monkeypatch.setattr(parser, bound, 1)
    raw = b'{"sample_id":"long-id","dataset":{"id":"nested"}}'
    with pytest.raises(ValueError, match=r"bound|capacity"):
        project(raw, chunk=3)


def test_retained_nonfinite_number_is_not_accepted():
    with pytest.raises(ValueError, match="nonfinite"):
        project(b'{"sample_id":1e9999}')
    assert project(b'{"answer":1e9999}')["identities"]["sample_ids"] == []


def test_project_file_regular_no_symlink_one_pass_and_signatures(tmp_path):
    path = tmp_path / "fixture.json"
    raw = b'{"dataset_id":"synthetic","sample_id":"x"}'
    path.write_bytes(raw)
    signature = parser.legacy.signature(path)
    output = parser.project_file(path, signature, time.monotonic() + 5)
    assert output["signature_before"] == output["signature_after"] == signature
    assert output["fd_signature_before"] == output["fd_signature_after"] == signature
    assert output["whole_file_sha256"] == hashlib.sha256(raw).hexdigest()
    assert output["fresh_confirmation_authorized"] is False
    alias = tmp_path / "alias.json"
    alias.symlink_to(path)
    with pytest.raises(ValueError, match="regular-file"):
        parser.project_file(alias, signature, time.monotonic() + 5)


def test_source_mutation_during_open_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "fixture.json"
    path.write_bytes(b'{"id":"x"}')
    signature = parser.legacy.signature(path)
    original = parser.identity_projection_stream

    def changed(stream, **kwargs):
        output = original(stream, **kwargs)
        path.write_bytes(b'{"id":"changed"}')
        return output

    monkeypatch.setattr(parser, "identity_projection_stream", changed)
    with pytest.raises(ValueError, match="changed"):
        parser.project_file(path, signature, time.monotonic() + 5)


def test_credentials_rejected_before_any_open(tmp_path, monkeypatch):
    monkeypatch.setattr(parser.os, "lstat", lambda *args: pytest.fail("credential stat forbidden"))
    with pytest.raises(ValueError, match="noncredential"):
        parser.project_file(tmp_path / ".codex/auth.json", [0, 0, 0, 0], time.monotonic() + 5)


def test_invalid_utf8_exception_does_not_include_source_payload():
    with pytest.raises(ValueError) as error:
        project(b'{"answer":"PRIVATE_PAYLOAD\xff"}')
    assert "PRIVATE" not in str(error.value)


def test_project_file_accepts_explicit_tuple_signature(tmp_path):
    path = tmp_path / "fixture.json"
    path.write_bytes(b'{"sample_id":"x"}')
    output = parser.project_file(path, tuple(parser.legacy.signature(path)), time.monotonic() + 5)
    assert output["identities"]["sample_ids"] == ["x"]


def test_repeated_profiler_events_deduplicate_with_later_parent_scope(monkeypatch):
    monkeypatch.setattr(parser, "MAXIMUM_METADATA_NODES", 64)
    events = [
        dict(
            name=f"operator-{i % 3}",
            args=dict(sample_id="same-request"),
            unknown_tensor=[0.5, 1, 2],
        )
        for i in range(12000)
    ]
    raw = json.dumps(
        dict(events=events, dataset_id="late-parent-dataset", split="calibration")
    ).encode()
    output = project(raw, chunk=65536)
    assert output["identities"] == reference(raw)
    stats = output["statistics"]
    assert stats["array_children_deduplicated"] == 11997
    assert stats["projected_metadata_nodes"] < 64
    assert stats["released_duplicate_metadata_nodes"] > 50000
    assert stats["work_units"] > 12000
    assert output["whole_file_sha256"] == hashlib.sha256(raw).hexdigest()


def test_dedup_never_crosses_array_contexts_or_loses_parent_namespaces():
    raw = json.dumps(
        dict(
            records=[
                dict(dataset_id="A", examples=[dict(sample_id="same"), dict(sample_id="same")]),
                dict(dataset_id="B", examples=[dict(sample_id="same"), dict(sample_id="same")]),
            ]
        )
    ).encode()
    output = project(raw, chunk=7)
    assert output["identities"] == reference(raw)
    scoped = {tuple(row) for row in output["identities"]["scoped_identities"]}
    assert ("A", "", "", "same") in scoped and ("B", "", "", "same") in scoped
    assert output["statistics"]["array_children_deduplicated"] == 2


def test_digest_collision_does_not_deduplicate_distinct_typed_projections(monkeypatch):
    raw = (
        b'{"sample_ids":[true,1,true,1],"records":['
        b'{"dataset":true,"id":"x"},{"dataset":1,"id":"x"}]}'
    )
    expected = reference(raw)
    monkeypatch.setattr(parser, "digest", lambda _value: "0" * 64)
    output = project(raw, chunk=5)
    assert output["identities"] == expected
    assert output["statistics"]["array_children_deduplicated"] == 2
    assert not parser._same_projection(0.0, -0.0)


def test_discarded_null_metadata_does_not_consume_live_capacity(monkeypatch):
    monkeypatch.setattr(parser, "MAXIMUM_METADATA_NODES", 8)
    raw = b'{"events":[' + b'{"name":null},' * 5000 + b'{"name":null}]}'
    output = project(raw, chunk=4096)
    assert output["identities"] == reference(raw)
    assert output["statistics"]["projected_metadata_nodes"] == 0
