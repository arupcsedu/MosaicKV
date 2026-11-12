"""Synthetic bounded metadata proofs; no historical data or credential reads."""

import hashlib
import json
import time
from pathlib import Path

import evicastkv_small_residuals_v41 as small
import pytest
import yaml


def project(raw, **kwargs):
    return small.projection(raw, deadline=time.monotonic() + 10, **kwargs)


def test_backward_aliases_use_contextual_redaction_and_preserve_query_identity():
    value = project(
        b"hidden: &a {sample_id: kept, answer: PRIVATE}\n"
        b"answer: *a\nmetadata: *a\ncopy: {sample_ids: &ids [one, two]}\nother: *ids\n"
    )
    assert value["answer"] is None
    assert value["metadata"]["sample_id"] == "kept"
    assert value["metadata"]["answer"] is None
    assert value["other"] == [None, None]  # Unknown scalar context stays redacted.
    assert "PRIVATE" not in json.dumps(value)


def test_yaml_merge_sequence_earlier_wins_and_explicit_key_always_overrides():
    result = project(
        b"a: &a {sample_id: first, split: train}\n"
        b"b: &b {sample_id: second, dataset_id: coco}\n"
        b"merged: {<<: [*a, *b]}\n"
        b"override: {sample_id: explicit, <<: [*a, *b]}\n"
        b"nested: {<<: {<<: *a, split: val}}\n"
    )
    assert result["merged"] == dict(sample_id="first", split="train", dataset_id="coco")
    assert result["override"]["sample_id"] == "explicit"
    assert result["nested"] == dict(sample_id="first", split="val")


@pytest.mark.parametrize(
    "raw",
    [
        b"x: &x [*x]\n",
        b"x: *future\ny: &future safe\n",
        b"x: &a one\ny: &a two\n",
        b"x: {sample_id: one, sample_id: two}\n",
        b"x: {<<: {}, <<: {}}\n",
        b"x: !!python/object/apply:os.system [PRIVATE]\n",
        b"x: &a scalar\ny: {<<: *a}\n",
        b"x: {<<: [1, 2]}\n",
        b"x: {<<: []}\nx: two\n",
        b"? [x, y]\n: value\n",
        b"x: one\n---\nx: two\n",
        b"%YAML 1.2\n---\nx: one\n",
        b"x: [unterminated\n",
        b"sample_id: on\n",
        b"sample_id: 001\n",
    ],
)
def test_unsafe_ambiguous_or_malformed_yaml_fails_without_raw_snippets(raw):
    with pytest.raises((ValueError, small.lexical.UnsupportedYAML)) as error:
        project(raw)
    assert "PRIVATE" not in str(error.value)


def test_no_yaml_constructors_or_loaders_are_used(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("YAML object construction forbidden")

    monkeypatch.setattr(yaml, "safe_load", forbidden)
    monkeypatch.setattr(yaml, "load", forbidden)
    monkeypatch.setattr(yaml.constructor.BaseConstructor, "construct_object", forbidden)
    monkeypatch.setattr(yaml.resolver.Resolver, "__init__", forbidden)
    assert project(b"x: &x {sample_id: one}\ny: *x\n")["y"]["sample_id"] == "one"


@pytest.mark.parametrize("limit", ["MAX_BYTES", "MAX_EVENTS", "MAX_DEPTH", "MAX_EXPANDED"])
def test_yaml_resource_limits(limit, monkeypatch):
    monkeypatch.setattr(small, limit, 1)
    with pytest.raises(ValueError):
        project(b"a: &a {sample_id: one}\nb: *a\n")


def test_yaml_deadline_and_version(monkeypatch):
    with pytest.raises(TimeoutError):
        small.projection(b"x: one", deadline=0)
    monkeypatch.setattr(yaml, "__version__", "unregistered")
    with pytest.raises(ValueError):
        project(b"x: one")


def test_alias_expansion_bomb_rejected_before_projection(monkeypatch):
    monkeypatch.setattr(small, "MAX_EXPANDED", 40)
    with pytest.raises(ValueError, match="expanded"):
        project(b"a: &a [one, two]\nb: &b [*a, *a, *a]\nc: [*b, *b, *b, *b]\n")


def test_nonfinite_is_accepted_only_for_discarded_metadata_values():
    raw = b'{"sample_id":"one","score":NaN,"unknown":Infinity,"answer":{"id":-Infinity}}'
    result, count = small.redact_nonfinite(raw, deadline=time.monotonic() + 10)
    assert count == 3 and result["sample_id"] == "one"
    assert result["score"] is None and result["answer"] is None
    assert "Infinity" not in json.dumps(result)


def test_reviewed_report_fields_do_not_decode_benchmark_values(monkeypatch):
    raw = (
        b'{"answer":{"PRIVATE_KEY":"PRIVATE_ANSWER"},"metrics":[{"score":0.5}],'
        b'"status":"passed_execution","completed_rows":64,"expected_rows":64,'
        b'"plan_sha256":"bound-plan","source_sha":"bound-source","sample_id":"consumed"}'
    )
    loads = json.loads

    def guarded(value, *args, **kwargs):
        assert "PRIVATE" not in str(value), "benchmark value was decoded"
        return loads(value, *args, **kwargs)

    monkeypatch.setattr(json, "loads", guarded)
    fields, projected = small.root_metadata(
        raw,
        {"status", "completed_rows", "expected_rows", "plan_sha256"},
        deadline=time.monotonic() + 10,
    )
    assert fields == dict(
        status="passed_execution", completed_rows=64, expected_rows=64, plan_sha256="bound-plan"
    )
    assert projected["sample_id"] == "consumed"
    assert "PRIVATE" not in str(projected)


def test_reviewed_report_projection_cannot_be_widened_to_answer_fields():
    with pytest.raises(ValueError, match="registered execution-control"):
        small.root_metadata(b'{"answer":"PRIVATE"}', {"answer"}, deadline=time.monotonic() + 10)


def test_reviewed_report_control_fields_must_be_scalar():
    with pytest.raises(ValueError, match="must be scalar"):
        small.root_metadata(
            b'{"status":{"answer":"PRIVATE"}}', {"status"}, deadline=time.monotonic() + 10
        )


@pytest.mark.parametrize(
    "raw",
    [
        b'{"sample_id":NaN}',
        b'{"dataset_id":Infinity}',
        b'{"media_path":-Infinity}',
        b'{"unknown":{"sample_id":NaN}}',
        b'{"x":NaN,"x":null}',
        b'{"x":NaN,}',
        b'{"x":NaNgarbage}',
        b'{"x":Infinity} garbage',
    ],
)
def test_nonfinite_identity_or_invalid_json_is_not_normalized(raw):
    with pytest.raises(ValueError):
        small.redact_nonfinite(raw, deadline=time.monotonic() + 10)


def test_json_strings_are_not_replaced():
    result, count = small.redact_nonfinite(
        b'{"sample_id":"NaN","answer":"Infinity"}', deadline=time.monotonic() + 10
    )
    assert count == 0 and result["sample_id"] == "NaN" and result["answer"] is None


def test_zero_byte_auth_proof_never_opens_body(tmp_path, monkeypatch):
    path = tmp_path / "auth.json"
    path.touch()
    monkeypatch.setattr(small, "EMPTY", {path})
    original_open = Path.open

    def forbidden(self, *args, **kwargs):
        if self == path:
            pytest.fail("credential body opened")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", forbidden)
    proof = small.prove(dict(path=str(path)), [tmp_path], time.monotonic() + 10)
    assert proof["current_file_sha256"] == hashlib.sha256(b"").hexdigest()
    assert proof["file_hash_basis"] == "stable_regular_size_zero"
    assert not proof["credential_bodies_read"]
    assert not proof["fresh_confirmation_authorized"]


def test_empty_credential_final_symlink_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "auth.json"
    path.touch()
    alias = tmp_path / "alias.json"
    alias.symlink_to(path)
    monkeypatch.setattr(small, "EMPTY", {path})
    with pytest.raises(ValueError):
        small.prove(dict(path=str(alias)), [tmp_path], time.monotonic() + 10)


def test_nonempty_auth_is_never_opened_even_if_registered_empty(tmp_path, monkeypatch):
    path = tmp_path / "auth.json"
    path.write_bytes(b"synthetic")
    monkeypatch.setattr(small, "EMPTY", {path})
    monkeypatch.setattr(Path, "open", lambda *a, **k: pytest.fail("auth body opened"))
    with pytest.raises(ValueError):
        small.prove(dict(path=str(path)), [tmp_path], time.monotonic() + 10)


def test_unknown_nonempty_credential_fixture_remains_blocked(tmp_path, monkeypatch):
    path = tmp_path / "auth.json"
    path.write_text("synthetic")
    monkeypatch.setattr(Path, "open", lambda *a, **k: pytest.fail("auth body opened"))
    with pytest.raises(ValueError, match="credential"):
        small.prove(dict(path=str(path)), [tmp_path], time.monotonic() + 10)


def test_yaml_proof_binds_both_hashes_and_retains_scope_flags(tmp_path):
    path = tmp_path / "sample.yaml"
    path.write_bytes(b"a: &a {sample_id: one}\ncopy: *a\nanswer: PRIVATE\n")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    proof = small.prove(
        dict(path=str(path), signature=small.signature(path), current_file_sha256=expected),
        [tmp_path],
        time.monotonic() + 10,
    )
    assert proof["full_file_sha256_before"] == proof["full_file_sha256_after"] == expected
    assert proof["identities"]["sample_ids"] == ["one"]
    assert "PRIVATE" not in json.dumps(proof)
    assert not proof["old_historical_byte_claim"] and not proof["error_waived"]


def test_mutation_between_reads_rejected(tmp_path, monkeypatch):
    path = tmp_path / "sample.yaml"
    path.write_text("sample_id: one")
    original = small.projection

    def mutate(raw, **kwargs):
        result = original(raw, **kwargs)
        path.write_text("sample_id: two")
        return result

    monkeypatch.setattr(small, "projection", mutate)
    with pytest.raises(ValueError, match="changed"):
        small.prove(dict(path=str(path)), [tmp_path], time.monotonic() + 10)


def test_small_read_rejects_final_link_swap_before_read(tmp_path, monkeypatch):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    path = allowed / "sample.yaml"
    path.write_text("sample_id: one")
    outside = tmp_path / "outside.yaml"
    outside.write_text("answer: PRIVATE")
    original_open = small.os.open

    def swap(name, flags, *args, **kwargs):
        assert flags & small.os.O_NOFOLLOW
        path.unlink()
        path.symlink_to(outside)
        return original_open(name, flags, *args, **kwargs)

    monkeypatch.setattr(small.os, "open", swap)
    with pytest.raises(OSError):
        small.read(path, [allowed], time.monotonic() + 10, {})


def test_small_read_checks_descriptor_before_body_read(tmp_path, monkeypatch):
    from types import SimpleNamespace

    path = tmp_path / "sample.yaml"
    path.write_text("sample_id: one")
    actual = path.stat()
    monkeypatch.setattr(
        small.os,
        "fstat",
        lambda fd: SimpleNamespace(
            st_mode=actual.st_mode,
            st_dev=actual.st_dev,
            st_ino=actual.st_ino,
            st_size=actual.st_size + 1,
            st_mtime_ns=actual.st_mtime_ns,
        ),
    )
    with pytest.raises(ValueError, match="descriptor differs"):
        small.read(path, [tmp_path], time.monotonic() + 10, {})


def test_outside_root_target_not_opened(tmp_path, monkeypatch):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    target = tmp_path / "other.yaml"
    target.write_text("sample_id: private")
    alias = allowed / "alias.yaml"
    alias.symlink_to(target)
    monkeypatch.setattr(Path, "open", lambda *a, **k: pytest.fail("outside target opened"))
    with pytest.raises(ValueError, match="outside"):
        small.prove(dict(path=str(alias)), [allowed], time.monotonic() + 10)


def test_wrong_registered_hash_and_signature_fail(tmp_path):
    path = tmp_path / "sample.yaml"
    path.write_text("sample_id: one")
    for addition in [dict(current_file_sha256="0" * 64), dict(signature=[0, 0, 0, 0])]:
        with pytest.raises(ValueError):
            small.prove(dict(path=str(path), **addition), [tmp_path], time.monotonic() + 10)


def test_unregistered_nan_file_is_not_admitted(tmp_path):
    path = tmp_path / "v38_analysis.json"
    path.write_text('{"score":NaN}')
    with pytest.raises(ValueError, match="no reviewed"):
        small.prove(dict(path=str(path)), [tmp_path], time.monotonic() + 10)


def test_registered_nan_metadata_proof_does_not_validate_numeric_results(tmp_path, monkeypatch):
    path = tmp_path / "aggregate.json"
    raw = b'{"sample_id":"consumed","score":NaN,"rows":[{"utility":NaN}]}'
    path.write_bytes(raw)
    expected = hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(small, "NAN_SUMMARIES", {path: (expected, 2)})
    proof = small.prove(dict(path=str(path)), [tmp_path], time.monotonic() + 10)
    assert proof["identities"]["sample_ids"] == ["consumed"]
    assert proof["current_file_sha256"] == expected
    assert not proof["historical_numeric_validity_claimed"]


def test_registered_nan_count_or_digest_change_blocks_proof(tmp_path, monkeypatch):
    path = tmp_path / "aggregate.json"
    raw = b'{"score":NaN}'
    path.write_bytes(raw)
    for expected, count in [(hashlib.sha256(raw).hexdigest(), 2), ("0" * 64, 1)]:
        monkeypatch.setattr(small, "NAN_SUMMARIES", {path: (expected, count)})
        with pytest.raises(ValueError):
            small.prove(dict(path=str(path)), [tmp_path], time.monotonic() + 10)


def test_no_recovery_is_inferred_from_a_report_filename(tmp_path):
    path = tmp_path / "report.json"
    path.write_text('{"status":"passed_execution"}')
    with pytest.raises(ValueError, match="no reviewed"):
        small.prove(dict(path=str(path)), [tmp_path], time.monotonic() + 10)


def test_root_controls_never_decode_unknown_or_answer_values(monkeypatch):
    original_loads = json.loads
    calls = []

    def tripwire(raw, *args, **kwargs):
        if "PRIVATE_ESCAPED" in str(raw):
            pytest.fail("benchmark value reached a JSON constructor")
        calls.append(raw)
        return original_loads(raw, *args, **kwargs)

    monkeypatch.setattr(json, "loads", tripwire)
    raw = (
        b'{"status":"passed_execution","completed_rows":64,"expected_rows":64,'
        b'"plan_sha256":"hash","answer":"PRIVATE_ESCAPED\\u0041",'
        b'"unknown":"PRIVATE_ESCAPED\\u0042",'
        b'"comparisons":[{"score":"PRIVATE_ESCAPED\\u0043"}]}'
    )
    controls, projected = small.root_metadata(
        raw,
        {"status", "completed_rows", "expected_rows", "plan_sha256"},
        deadline=time.monotonic() + 10,
    )
    assert controls == dict(
        status="passed_execution", completed_rows=64, expected_rows=64, plan_sha256="hash"
    )
    assert projected["answer"] is None and projected["unknown"] is None
    assert projected["comparisons"] == [{"score": None}]
    assert calls and "PRIVATE_ESCAPED" not in json.dumps(projected)


@pytest.mark.parametrize("raw", [b'{"status":[]}', b'{"status":{}}'])
def test_root_execution_controls_cannot_construct_nested_values(raw):
    with pytest.raises(ValueError, match="must be scalar"):
        small.root_metadata(raw, {"status"}, deadline=time.monotonic() + 10)


def test_root_metadata_whitelist_cannot_expand_to_benchmark_fields():
    with pytest.raises(ValueError, match="registered"):
        small.root_metadata(b'{"answer":"private"}', {"answer"}, deadline=time.monotonic() + 10)


def test_existing_v6_recovery_requires_original_plan_patch_and_equal_saved_data(
    tmp_path, monkeypatch
):
    parent, derived = tmp_path / "original", tmp_path / "derived"
    (parent / "analysis").mkdir(parents=True)
    derived.mkdir()

    def put(path, payload):
        raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        path.write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    original = parent / "analysis/report.json"
    original_sha = put(original, b'{"incomplete":')
    producer = "799ca3c6c3eccadadb52688ebc3823cfbe66522f"
    plan_sha = put(parent / "plan.json", dict(source_sha=producer))
    patch_sha = put(derived / "analysis-source.patch", b"synthetic patch, never executed")
    artifact_hashes = {str(original): original_sha}
    for name in ("samples.parquet", "summary.parquet"):
        artifact_hashes[str(parent / "analysis" / name)] = put(
            parent / "analysis" / name, b"synthetic saved-data bytes"
        )
        put(derived / name, b"synthetic saved-data bytes")
    report_sha = put(
        derived / "report.json",
        dict(
            status="passed_execution",
            completed_rows=64,
            expected_rows=64,
            plan_sha256=plan_sha,
            sample_id="consumed",
        ),
    )
    provenance_sha = put(
        derived / "analysis_provenance.json",
        dict(
            parent_root=str(parent),
            original_analysis_artifacts=artifact_hashes,
            gpu_reruns=0,
            original_analysis_preserved=True,
            producer_artifacts_verified=True,
            producer_git_sha=producer,
            parent_plan_sha256=plan_sha,
            analyzer_patch_sha256=patch_sha,
        ),
    )
    for key, value in dict(
        V6_ORIGINAL=original,
        V6_DERIVED=derived,
        V6_SHA=original_sha,
        V6_PROVENANCE_SHA=provenance_sha,
        V6_REPORT_SHA=report_sha,
    ).items():
        monkeypatch.setattr(small, key, value)
    proof = small.prove(dict(path=str(original)), [tmp_path], time.monotonic() + 10)
    assert proof["current_file_sha256"] == original_sha
    assert proof["identities"]["sample_ids"] == ["consumed"]
    assert not proof["historical_numeric_validity_claimed"]
    (derived / "samples.parquet").write_bytes(b"changed data")
    with pytest.raises(ValueError, match="hash changed"):
        small.prove(dict(path=str(original)), [tmp_path], time.monotonic() + 10)
