"""Lossless event metadata compression, bounded expansion and retry admission."""

import copy
import json

import evicastkv_source_ownership_v12 as owner
import evicastkv_source_trace_adjudication as adjudication
import evicastkv_source_trace_codec as codec
import pytest

from mosaickv.evaluation.fox_campaign import digest


def payload():
    stack = [
        dict(filename="/source/native.py", name="forward", line=17),
        dict(filename="/source/reader.py", name="reconstruct", line=9, extra="preserved"),
    ]
    events = [
        dict(action=kind, addr=10, size=64, sequence=i, time_us=i, frames=copy.deepcopy(stack))
        for i, kind in enumerate(
            ["alloc", "free_requested", "free_completed", "alloc", "snapshot", "free_completed"]
        )
    ]
    records = [
        dict(address=10, generation=i, role="unknown", evidence=[], bytes=64) for i in (0, 1)
    ]
    return dict(
        actual_events=events,
        allocator_records=records,
        capture_metadata=dict(records=records, events=len(events), complete_request=True),
        final_allocator_snapshot=dict(device_traces=[events], segments=[], extra="not dropped"),
    )


def test_nested_repeated_frames_aliases_frees_and_all_fields_roundtrip_exactly():
    value = payload()
    document = codec.encode(value)
    actual = codec.decode(json.loads(json.dumps(document)))
    assert actual == value
    assert actual["allocator_records"] is actual["capture_metadata"]["records"]
    assert actual["actual_events"] is actual["final_allocator_snapshot"]["device_traces"][0]
    assert actual["actual_events"][0]["frames"] is actual["actual_events"][1]["frames"]
    assert codec.document_digest(document) == digest(value)
    for field in value:
        assert codec.document_digest(document, (field,)) == digest(value[field])
    assert codec.semantic_digest(value) == digest(value)


def test_actual_repeated_nested_frame_shape_reduces_serialization_not_events():
    frames = [
        dict(filename="/scratch/source/" + "x" * 100, name="operator", line=i) for i in range(20)
    ]
    value = dict(
        actual_events=[
            dict(action="alloc", addr=i, size=8, frames=copy.deepcopy(frames)) for i in range(100)
        ]
    )
    document = codec.encode(value)
    assert len(json.dumps(document)) < len(json.dumps(value)) / 5
    assert codec.decode(document) == value
    assert len(codec.decode(document)["actual_events"]) == 100
    assert document["statistics"]["semantic_JSON_bytes"] == len(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    )


@pytest.mark.parametrize(
    "corruption", ["forward", "cycle", "badref", "unknownkind", "duplicatekey", "statistics"]
)
def test_adversarial_references_schema_and_size_lies_are_rejected(corruption):
    document = codec.encode(dict(a=[1], b=2))
    if corruption == "forward":
        document["nodes"][0][1].append({"$ref": len(document["nodes"]) - 1})
    elif corruption == "cycle":
        document["nodes"][0][1].append({"$ref": 0})
    elif corruption == "badref":
        document["nodes"][0][1].append({"$ref": True})
    elif corruption == "unknownkind":
        document["nodes"][0][0] = "execute"
    elif corruption == "duplicatekey":
        document["nodes"][-1][1].append(copy.deepcopy(document["nodes"][-1][1][0]))
    else:
        document["statistics"]["semantic_JSON_bytes"] -= 1
    with pytest.raises(ValueError):
        codec.decode(document)


@pytest.mark.parametrize("in_frames", [False, True])
def test_input_cycles_fail_closed_before_serialization(in_frames):
    value = []
    value.append(value)
    with pytest.raises(ValueError, match="cyclic"):
        codec.encode(dict(frames=value) if in_frames else value)


def test_DAG_bomb_is_rejected_before_expanding_shared_references(monkeypatch):
    # A physically tiny DAG may represent an exponential expanded JSON tree.
    value = [0]
    for _ in range(8):
        value = [value, value]
    monkeypatch.setitem(codec.LIMITS, "semantic_JSON_bytes", 256)
    with pytest.raises(ValueError, match="expansion"):
        codec.encode(value)


def test_orphan_intermediate_node_is_rejected_even_when_root_is_final():
    document = codec.encode(dict(a=[]))
    document["nodes"][-1][1] = []
    document["statistics"] = dict(containers=2, items=0, semantic_JSON_bytes=2)
    with pytest.raises(ValueError, match="reachable"):
        codec.decode(document)


def test_frame_interning_preserves_signed_zero_and_all_float_JSON_words():
    value = [dict(frames=[dict(extra=-0.0)]), dict(frames=[dict(extra=0.0)])]
    encoded = codec.encode(value)
    assert codec.document_digest(encoded) == digest(value)
    assert json.dumps(codec.decode(encoded)) == json.dumps(value)


@pytest.mark.parametrize("scalar", [float("nan"), float("inf"), object()])
def test_non_JSON_or_nonfinite_values_are_not_rewritten_or_silently_dropped(scalar):
    with pytest.raises(ValueError):
        codec.encode(dict(unknown_field=scalar))


def test_reader_supports_both_legacy_and_DAG_archives_without_changing_semantics(tmp_path):
    import gzip

    value = payload()
    for name, encoded in (("legacy.json.gz", value), ("DAG.json.gz", codec.encode(value))):
        with gzip.open(tmp_path / name, "wt") as stream:
            json.dump(encoded, stream)
        assert adjudication.load(tmp_path / name) == value


def failure_fixture(tmp_path, monkeypatch):
    (tmp_path / "worker").mkdir()
    revision = "76d4ced1a10c6746bad73f564e84210522355ef9"
    (tmp_path / "protocol.json").write_text(
        json.dumps(dict(source_sha=revision, source_root=str(tmp_path), bindings={}))
    )
    (tmp_path / "worker/manifest.json").write_text(
        json.dumps(dict(source_sha=revision, slurm_job_id="22800695"))
    )
    (tmp_path / "worker/raw.json").write_text(
        json.dumps(
            dict(
                arm=owner.original.ARM,
                counters=dict(updates=127),
                record=dict(output_tokens=128, storage=dict(history_updates_per_layer=[127] * 32)),
            )
        )
    )
    (tmp_path / "worker/logits.pt").write_bytes(b"hash-bound completed request")
    (tmp_path / "worker/declared_owners.json").write_text("{}")
    result = dict(
        source_sha=revision,
        status="failed",
        continuation_updates=127,
        failures=[
            dict(error="ownership reduction: bounded raw ownership trace archive exceeds 2 GiB")
        ],
        registered_guard_pass=False,
        repeat_all_tokens_agree=True,
        outputs={p.name: owner.sha(p) for p in (tmp_path / "worker").iterdir()},
    )
    (tmp_path / "worker/result.json").write_text(json.dumps(result))
    monkeypatch.setattr(owner.original, "clean_source", lambda *a: revision)
    return result


def test_only_exact_saved_archive_limit_failure_permits_registered_serialization_retry(
    tmp_path, monkeypatch
):
    failure_fixture(tmp_path, monkeypatch)
    decision = owner.classify_archive_failure(tmp_path)
    assert decision["one_retry_only"] and not decision["actual_GPU_retry_submitted"]
    assert decision["prior_registered_guard_pass"] is False
    assert decision["numerical_pass_not_claimed"]


@pytest.mark.parametrize("defect", ["generic_failure", "incomplete", "tamper", "existing_trace"])
def test_retry_proof_never_waives_other_failure_or_CPU_recovery_case(tmp_path, monkeypatch, defect):
    result = failure_fixture(tmp_path, monkeypatch)
    if defect == "generic_failure":
        result["failures"][0]["error"] = "CUDA_OOM"
    elif defect == "incomplete":
        result["continuation_updates"] = 10
    elif defect == "tamper":
        (tmp_path / "worker/raw.json").write_text("{}")
    else:
        (tmp_path / "worker/allocator_trace_pre_attribution.json.gz").write_bytes(
            b"existing archive"
        )
    (tmp_path / "worker/result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError):
        owner.classify_archive_failure(tmp_path)
