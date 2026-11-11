"""Synthetic compact DAGs only: never read the real historical archives."""

import copy
import gzip
import json
from dataclasses import replace

import evicastkv_snapshot_state_audit as audit
import pytest
from evicastkv_source_trace_codec import encode


def payload(*states, flags=None):
    frames = [{"filename": 'unicode-\u03b1-"-\\', "name": "line\nvalue", "line": 4}]
    blocks = [
        dict(address=1000 + i * 256, size=256, requested_size=31 + i, state=state, frames=frames)
        for i, state in enumerate(states or ("active_allocated", "inactive"))
    ]
    return dict(
        capture_metadata=dict(
            complete_request=True,
            allocator_history_complete=True,
            blocks_exact_certification=list(audit.ALLOWED_FLAGS) if flags is None else flags,
        ),
        final_allocator_snapshot=dict(segments=[dict(device=0, blocks=blocks)], device_traces=[[]]),
    )


def archive(tmp_path, value=None, *, document=None, raw=None):
    document = encode(payload() if value is None else value) if document is None else document
    body = (
        json.dumps(document, separators=(",", ":"), ensure_ascii=True).encode()
        if raw is None
        else raw
    )
    path = tmp_path / "archive.json.gz"
    path.write_bytes(gzip.compress(body, mtime=0))
    receipt = dict(
        sha256=audit.sha(path),
        encoded_JSON_bytes=len(body),
        codec_statistics=document["statistics"],
    )
    return path, receipt


@pytest.mark.parametrize("chunk", [1, 2, 7, 31, 4096])
def test_streaming_unicode_escapes_string_boundaries_and_shared_backrefs(tmp_path, chunk):
    value = payload(*audit.STATES)
    path, receipt = archive(tmp_path, value)
    result = audit.scan(path, receipt, chunk_bytes=chunk)
    assert result["state_counts"] == dict.fromkeys(audit.STATES, 1)
    assert result["requested_bytes_by_state"] == dict(zip(audit.STATES, range(31, 35), strict=True))
    assert result["compressed_sha256"] == receipt["sha256"]
    assert result["codec_statistics"] == receipt["codec_statistics"]
    assert result["whole_semantic_DAG_expanded"] is False
    assert result["baseline_state_histogram_observed"] is False


def test_empty_flags_are_observed_not_assumed(tmp_path):
    path, receipt = archive(tmp_path, payload(flags=[]))
    assert audit.scan(path, receipt)["capture_flags"] == []


def test_repeated_block_alias_counts_every_snapshot_occurrence(tmp_path):
    value = payload("active_allocated")
    blocks = value["final_allocator_snapshot"]["segments"][0]["blocks"]
    blocks.append(blocks[0])
    path, receipt = archive(tmp_path, value)
    result = audit.scan(path, receipt)
    assert result["state_counts"]["active_allocated"] == 2
    assert result["requested_bytes_by_state"]["active_allocated"] == 62


@pytest.mark.parametrize("state", ["active_future_state", "", None, 4])
def test_unreviewed_allocator_states_fail_closed(tmp_path, state):
    path, receipt = archive(tmp_path, payload(state))
    with pytest.raises(ValueError, match="unknown allocator state"):
        audit.scan(path, receipt)


def test_negative_requested_size_is_rejected(tmp_path):
    value = payload()
    value["final_allocator_snapshot"]["segments"][0]["blocks"][0]["requested_size"] = -1
    path, receipt = archive(tmp_path, value)
    with pytest.raises(ValueError, match="requested size"):
        audit.scan(path, receipt)


@pytest.mark.parametrize("field", ["requested_size", "frames"])
def test_incomplete_block_shape_cannot_silently_disappear(tmp_path, field):
    value = payload()
    del value["final_allocator_snapshot"]["segments"][0]["blocks"][0][field]
    path, receipt = archive(tmp_path, value)
    with pytest.raises(ValueError, match="BlockInfo fields"):
        audit.scan(path, receipt)


@pytest.mark.parametrize("key,value", [("sha256", "0" * 64), ("encoded_JSON_bytes", 1)])
def test_raw_compressed_hash_and_encoded_size_must_match(tmp_path, key, value):
    path, receipt = archive(tmp_path)
    receipt[key] = value
    with pytest.raises(ValueError, match="receipt mismatch"):
        audit.scan(path, receipt)


def test_receipt_statistics_cannot_be_fabricated(tmp_path):
    path, receipt = archive(tmp_path)
    receipt["codec_statistics"] = dict(receipt["codec_statistics"], semantic_JSON_bytes=1)
    with pytest.raises(ValueError, match="statistics"):
        audit.scan(path, receipt)


def test_orphan_nodes_are_not_ignored(tmp_path):
    document = encode(payload())
    document["nodes"].append(["l", []])
    # The root may not refer to an earlier node while hidden nodes follow it.
    path, receipt = archive(tmp_path, document=document)
    with pytest.raises(ValueError, match="complete rooted"):
        audit.scan(path, receipt)


@pytest.mark.parametrize("ref", [-1, 999999, True])
def test_invalid_or_forward_references_are_rejected(tmp_path, ref):
    document = encode(payload())
    document["nodes"][0] = ["l", [{"$ref": ref}]]
    path, receipt = archive(tmp_path, document=document)
    with pytest.raises(ValueError, match="backward"):
        audit.scan(path, receipt)


@pytest.mark.parametrize(
    "name,value", [("containers", 1), ("items", 1), ("depth", 2), ("semantic_bytes", 16)]
)
def test_node_item_depth_and_semantic_bounds(tmp_path, name, value):
    path, receipt = archive(tmp_path)
    with pytest.raises(ValueError, match="limit exceeded"):
        audit.scan(path, receipt, bounds=replace(audit.BOUNDS, **{name: value}))


def test_complete_node_larger_than_single_chunk_bound_is_rejected(tmp_path):
    value = payload()
    value["long_scalar"] = "x" * 2000
    path, receipt = archive(tmp_path, value)
    with pytest.raises(ValueError, match="node buffer"):
        audit.scan(path, receipt, bounds=replace(audit.BOUNDS, node_buffer_bytes=512))


@pytest.mark.parametrize("name", ["compressed_bytes", "encoded_bytes", "sparse_summaries"])
def test_other_physical_and_sparse_limits(tmp_path, name):
    path, receipt = archive(tmp_path)
    with pytest.raises(ValueError, match="limit exceeded"):
        audit.scan(path, receipt, bounds=replace(audit.BOUNDS, **{name: 1}))


def test_registered_limits_cannot_be_raised(tmp_path):
    path, receipt = archive(tmp_path)
    with pytest.raises(ValueError, match="smaller registered"):
        audit.scan(path, receipt, bounds=replace(audit.BOUNDS, depth=97))


def test_custom_node_stream_does_not_accept_trailing_separator(tmp_path):
    document = encode(payload())
    body = json.dumps(document, separators=(",", ":")).encode()
    split = body.index(b'],"root":')
    body = body[:split] + b"," + body[split:]
    path, receipt = archive(tmp_path, document=document, raw=body)
    with pytest.raises(ValueError, match="trailing node separator"):
        audit.scan(path, receipt)


def test_gibibyte_semantic_aliases_are_counted_without_expansion(tmp_path, monkeypatch):
    import evicastkv_source_trace_codec as codec

    value = payload()
    repeated = ["bounded"]
    for _ in range(28):
        repeated = [repeated, repeated]
    value["repeated_metadata"] = repeated
    path, receipt = archive(tmp_path, value)
    monkeypatch.setattr(
        codec, "decode", lambda *a, **k: pytest.fail("semantic expansion forbidden")
    )
    result = audit.scan(path, receipt)
    assert result["codec_statistics"]["semantic_JSON_bytes"] > 2**30
    assert result["encoded_JSON_bytes"] < 20_000
    assert result["state_counts"]["active_allocated"] == 1


def test_baseline_flush_proof_is_hashed_source_not_observed_histogram(monkeypatch):
    source = b"""
class CudaOwnershipSession:
    def __enter__(self):
        torch.cuda.synchronize()
        self._flush_cuda_events()
    def _flush_cuda_events(self):
        tensor = torch.empty(1, device=0)
        del tensor
class AllocationTimeline:
    def ingest(self, traces):
        self.errors.append("free_without_observed_allocation")
"""
    monkeypatch.setattr(audit.subprocess, "check_output", lambda *a, **k: source)
    proof = audit.baseline_flush_proof()
    assert proof["baseline_histogram_was_not_archived"] is True
    assert len(proof["reviewed_function_AST_sha256s"]) == 3
    assert proof["file_sha256"] == audit.hashlib.sha256(source).hexdigest()
    monkeypatch.setattr(audit.subprocess, "check_output", lambda *a, **k: b"class Other: pass")
    with pytest.raises(ValueError, match="baseline flush"):
        audit.baseline_flush_proof()


@pytest.mark.parametrize(
    "kind", ["truncated-gzip", "truncated-json", "trailing-json", "wrong-codec"]
)
def test_corrupt_or_unsupported_archives_are_rejected(tmp_path, kind):
    document = encode(payload())
    if kind == "wrong-codec":
        document["codec"] = "unsupported"
    path, receipt = archive(tmp_path, document=document)
    if kind == "truncated-gzip":
        path.write_bytes(path.read_bytes()[:-5])
    elif kind == "truncated-json":
        path, receipt = archive(tmp_path, document=document, raw=b'{"codec":')
    elif kind == "trailing-json":
        body = json.dumps(document).encode() + b" {}"
        path, receipt = archive(tmp_path, document=document, raw=body)
    with pytest.raises((ValueError, EOFError)):
        audit.scan(path, receipt, chunk_bytes=7)


def test_durable_raw_and_semantic_receipt_must_bind_same_archive(tmp_path):
    path, tiny = archive(tmp_path)
    actual = tmp_path / "allocator_trace.json.gz"
    path.rename(actual)
    limits = dict(
        compressed_bytes=audit.BOUNDS.compressed_bytes,
        encoded_JSON_bytes=audit.BOUNDS.encoded_bytes,
        semantic_JSON_bytes=audit.BOUNDS.semantic_bytes,
        containers=audit.BOUNDS.containers,
        items=audit.BOUNDS.items,
        depth=audit.BOUNDS.depth,
        hash_cache_bytes=32 * 2**20,
    )
    raw_path = tmp_path / "trace_raw_receipt.json"
    raw = dict(
        tiny,
        path=str(actual),
        schema="streaming_lossless_BF16_whole_request_ownership_v15",
        source_sha=audit.PRODUCER_SHA,
        metadata_codec_limits=limits,
        complete_final_snapshot_preserved=True,
        semantic_hashes_complete=False,
        compressed_bytes=actual.stat().st_size,
    )
    raw_path.write_text(json.dumps(raw))
    receipt = dict(
        raw,
        semantic_hashes_complete=True,
        raw_receipt_path=str(raw_path),
        raw_receipt_sha256=audit.sha(raw_path),
    )
    result = dict(outputs={raw_path.name: audit.sha(raw_path)})
    assert audit.validate_receipts(tmp_path, result, receipt) == raw_path
    bad = copy.deepcopy(receipt)
    bad["codec_statistics"]["items"] += 1
    with pytest.raises(ValueError, match="receipt binding"):
        audit.validate_receipts(tmp_path, result, bad)


def test_cpu_gate_is_required_before_any_parent_archive_access(tmp_path, monkeypatch):
    import evicastkv_review_followups as common

    from mosaickv.evaluation import fox_campaign

    monkeypatch.setattr(fox_campaign, "clean_source", lambda source: "synthetic-revision")

    def reject(root, revision):
        raise ValueError("synthetic CPU gate failed")

    monkeypatch.setattr(common, "cpu_gate", reject)
    monkeypatch.setattr(
        audit, "read", lambda path: pytest.fail("parent metadata read before CPU gate")
    )
    with pytest.raises(ValueError, match="CPU gate"):
        audit.run(tmp_path / "new-derived", cpu_gate_root=tmp_path)


@pytest.mark.parametrize("failure", [None, "pending", "history", "structural"])
def test_complete_run_never_promotes_pending_or_incomplete_capture(tmp_path, monkeypatch, failure):
    import evicastkv_review_followups as common

    from mosaickv.evaluation import fox_campaign

    monkeypatch.setattr(fox_campaign, "clean_source", lambda source: "synthetic-revision")
    monkeypatch.setattr(common, "cpu_gate", lambda root, revision: {"status": "passed"})
    parent = tmp_path / "original"
    parent.mkdir()
    gate = parent / "GATE_streaming_lossless_ownership.json"
    protocol = parent / "protocol.json"
    gate.write_text(
        json.dumps(
            dict(
                source_sha=audit.PRODUCER_SHA,
                complete_numerical_and_trace_controls=True,
                failures=[],
            )
        )
    )
    protocol.write_text(json.dumps(dict(source_sha=audit.PRODUCER_SHA)))
    case_sha = []
    for index in range(2):
        directory = parent / f"workers/{index}"
        directory.mkdir(parents=True)
        path = directory / "allocator_trace.json.gz"
        path.write_bytes(b"synthetic archive not opened by this orchestration test")
        receipt = directory / "trace_receipt.json"
        receipt.write_text(
            json.dumps(
                dict(
                    source_sha=audit.PRODUCER_SHA,
                    semantic_hashes_complete=True,
                    sha256=audit.sha(path),
                )
            )
        )
        raw_receipt = directory / "trace_raw_receipt.json"
        raw_receipt.write_text("{}")
        result = directory / "result.json"
        result.write_text(
            json.dumps(
                dict(
                    source_sha=audit.PRODUCER_SHA,
                    failures=[],
                    complete_trace_with_final_snapshot_archived=True,
                    all_32_128_guards_pass=True,
                    outputs={receipt.name: audit.sha(receipt), path.name: audit.sha(path)},
                )
            )
        )
        case_sha.append({p.name: audit.sha(p) for p in (result, receipt, raw_receipt)})
    monkeypatch.setattr(audit, "PARENT", parent)
    monkeypatch.setattr(audit, "PARENT_GATE_SHA", audit.sha(gate))
    monkeypatch.setattr(audit, "PROTOCOL_SHA", audit.sha(protocol))
    monkeypatch.setattr(audit, "CASE_SHA", case_sha)
    monkeypatch.setattr(
        audit, "baseline_flush_proof", lambda: {"baseline_histogram_was_not_archived": True}
    )
    monkeypatch.setattr(
        audit,
        "validate_receipts",
        lambda directory, result, receipt: directory / "trace_raw_receipt.json",
    )

    def controlled_scan(path, receipt):
        bad_case = path.parent.name == "1"
        return dict(
            compressed_sha256=receipt["sha256"],
            capture_complete=True,
            allocator_history_complete=not (bad_case and failure == "history"),
            state_counts=dict.fromkeys(audit.STATES, 0)
            | {
                "active_pending_free": int(bad_case and failure == "pending"),
            },
            capture_flags=["free_without_observed_allocation"]
            if bad_case and failure == "structural"
            else [],
        )

    monkeypatch.setattr(audit, "scan", controlled_scan)
    root = tmp_path / "new-derived"
    report = audit.run(root, cpu_gate_root=tmp_path)
    assert report["original_two_audited_capture_bounds_requalified"] is (failure is None)
    assert len(report["snapshots"]) == 2
    assert len(report["errors"]) == int(failure is not None)
    assert report["exact_KV_peak_certified"] is False
    assert report["superiority_established"] is False
    assert report["baseline_state_histogram_observed"] is False
    assert (root / audit.GATE).is_file()
    with pytest.raises(FileExistsError):
        audit.run(root, cpu_gate_root=tmp_path)
