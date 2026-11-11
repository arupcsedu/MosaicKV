"""Actual tiny HF construction and synthetic allocator replay, CPU only."""

import copy
import gc
from dataclasses import replace
from types import SimpleNamespace

import evicastkv_streaming_lossless_ownership as study
import pytest
import torch
from test_streaming_lossless_source_runner import TinyAdapter

from mosaickv.evaluation.streaming_lossless_source import StreamingLosslessSource
from mosaickv.evaluation.streaming_prefill import RetiringPrefillCache, SealAfterAttention
from mosaickv.measurements.kv_timeline import AllocationTimeline


@pytest.fixture(scope="module", autouse=True)
def bounded_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


class Marks:
    def __init__(self):
        self.records = []

    def mark(self, values, role, evidence):
        self.records.append(
            dict(
                role=role,
                evidence=evidence,
                owners=[
                    (str(t.device), t.untyped_storage().data_ptr(), t.untyped_storage().nbytes())
                    for t in values
                ],
            )
        )


def test_named_baseline_catalog_deduplicates_aliases_and_retains_no_tensor_values():
    model = torch.nn.Module()
    model.first = torch.nn.Parameter(torch.ones(4))
    model.second = model.first
    model.register_buffer("same_backing", model.first.detach())
    catalog = study.model_catalog(model)
    assert len(catalog["owners"]) == 1
    assert {row["name"] for row in catalog["owners"][0]["names"]} == {
        "first",
        "second",
        "same_backing",
    }
    assert catalog["total_backing_bytes"] == model.first.untyped_storage().nbytes()
    assert catalog["owner_catalog_sha256"] == study.digest(catalog["owners"])
    assert catalog["tensor_values_or_GPU_owners_retained"] is False


def test_only_exact_actual_seeded_model_backings_are_nonKV_not_all_baseline():
    timeline = AllocationTimeline()
    timeline.seed(100, 16)
    timeline.seed(200, 4096)
    session = SimpleNamespace(timeline=timeline)
    rows = [
        dict(
            device="cuda:0",
            address=100,
            storage_cdata=1,
            bytes=16,
            names=[dict(name="weight", family="parameter")],
        )
    ]
    catalog = dict(owners=rows, owner_catalog_sha256=study.digest(rows))
    assert len(study.mark_baseline_model(session, catalog)) == 1
    assert timeline.allocations[(100, 0)]["role"] == "model"
    assert timeline.allocations[(200, 0)]["role"] == "unknown"
    bad = copy.deepcopy(catalog)
    bad["owners"][0]["bytes"] = 17
    with pytest.raises(ValueError, match="prewindow"):
        study.mark_baseline_model(session, bad)


def test_typed_weak_observer_keeps_workspace_roles_without_retaining_owner():
    session = Marks()
    owners = study.TypedOwners(session)
    tensor = torch.ones(8)
    owners.track("source_validation", (tensor,), kind="kv_workspace")
    owners.observe("visible")
    assert session.records[0]["role"] == "kv_workspace"
    del tensor
    gc.collect()
    assert owners.observe("retired")["owners"] == []
    with pytest.raises(ValueError, match="explicit"):
        owners.track("bad", (torch.ones(1),), kind="non_kv")


def test_actual_tiny_HF_streaming_prefill_has_no_projection_hooks_and_preserves_logits():
    adapter = TinyAdapter()
    adapter.decode_one_token = lambda *args, **kwargs: None
    prepared = adapter.prepare_inputs({})
    with torch.inference_mode():
        ordinary = adapter.prefill(prepared, capture_queries=False, capture_attentions=False)
    reference = ordinary.logits.clone()
    cache = RetiringPrefillCache(2, audit=True)
    encoder = StreamingLosslessSource(adapter, prepared, cache)
    fresh = replace(prepared, model_inputs=dict(prepared.model_inputs, past_key_values=cache))
    marks = Marks()
    owners = study.TypedOwners(marks)
    with torch.inference_mode(), study.instrument(adapter, owners) as counts:
        for layer in adapter._language_layers():
            assert not layer.self_attn.k_proj._forward_hooks
            assert not layer.self_attn.v_proj._forward_hooks
        with SealAfterAttention(adapter, cache, encoder):
            streamed = adapter.prefill(fresh, capture_queries=False, capture_attentions=False)
        encoder.finish()
    assert torch.equal(reference, streamed.logits)
    assert counts["prefill_layers"] == counts["sealed_layers"] == 2
    assert counts["constructor_owner_notifications"] > 0
    assert all(ref() is None for ref in cache.original_refs)
    assert {row["role"] for row in marks.records} == {
        "kv_persistent",
        "kv_workspace",
        "kv_shared_workspace",
    }
    assert all(not layer.self_attn._forward_hooks for layer in adapter._language_layers())


def test_instrument_failure_removes_its_scopes_and_preserves_safety_guard():
    adapter = TinyAdapter()
    adapter.decode_one_token = lambda *args, **kwargs: None
    original_update = RetiringPrefillCache(1).layers[0].__class__.update
    owners = study.TypedOwners(Marks())
    with pytest.raises(RuntimeError, match="intentional"), study.instrument(adapter, owners):
        raise RuntimeError("intentional")
    assert RetiringPrefillCache(1).layers[0].__class__.update is original_update
    assert all(not layer.self_attn.k_proj._forward_hooks for layer in adapter._language_layers())


def synthetic_session():
    timeline = AllocationTimeline()
    raw = [
        dict(action="alloc", addr=50, size=64, time_us=10, frames=[]),
        dict(action="free_requested", addr=50, size=64, time_us=11),
        dict(action="free_completed", addr=50, size=64, time_us=12),
        dict(action="snapshot", time_us=13),
    ]
    timeline.ingest(raw)
    capture = timeline.report(
        complete=True, history_complete=True, opaque_workspace_classified=False
    )
    capture["exact_whole_request_KV_peak_certified"] = False
    return SimpleNamespace(
        snapshot=dict(segments=[], device_traces=[raw]),
        finished=True,
        device=0,
        baseline_trace_count=0,
        timeline=timeline,
        marks=[],
        report=lambda **kwargs: capture,
    )


def test_new_typed_DAG_roundtrip_preserves_all_free_events_and_final_snapshot(tmp_path):
    session = synthetic_session()
    owners = study.TypedOwners()
    catalog = study.model_catalog(torch.nn.Linear(2, 2))
    path = tmp_path / "trace.json.gz"
    receipt = study.archive(path, session, owners, catalog, source_sha="synthetic-source")
    value = study.load_bundle(path, receipt)
    assert value["schema"] == study.KIND
    assert value["actual_events"][-2]["action"] == "free_completed"
    assert value["final_allocator_snapshot"]["device_traces"][0][-1]["action"] == "snapshot"
    assert not value["capture_metadata"]["exact_whole_request_KV_peak_certified"]
    assert receipt["complete_final_snapshot_preserved"]
    assert study.validate_bundle(value)
    corrupted = copy.deepcopy(value)
    corrupted["final_allocator_snapshot"]["device_traces"][0].pop()
    with pytest.raises(ValueError, match="Every final"):
        study.validate_bundle(corrupted)
    missing = copy.deepcopy(value)
    del missing["final_allocator_snapshot"]
    with pytest.raises(ValueError, match="schema"):
        study.validate_bundle(missing)


def test_old_three_bit_ARM_trace_is_not_silently_accepted():
    with pytest.raises(ValueError, match="Distinct complete"):
        study.validate_bundle(dict(arm="source3_attention_optional", final_allocator_snapshot={}))
    assert study.POLICY["exact_KV_peak_certification_authorized"] is False
    assert study.POLICY["no_latency_or_quality_claim"]
    assert "source3" not in study.KIND


def test_same_private_cache_history_and_allword_guards_cannot_be_skipped():
    counts = dict(prefill_layers=32, sealed_layers=32, decode_updates=127, history_seal_calls=7)
    record = dict(
        arm=study.timing.ARMS[1],
        audited_precheck=True,
        original_native_KV_retired=True,
        private_source_objects_retired_after_response=True,
        constructor_all_word_validation=True,
        whole_request_KV_peak_certified=False,
        storage=dict(history_updates_per_layer=[127] * 32, outstanding_layer_leases=0),
    )
    assert study.verify_request(record, counts)
    for name in (
        "original_native_KV_retired",
        "private_source_objects_retired_after_response",
        "constructor_all_word_validation",
    ):
        bad = copy.deepcopy(record)
        bad[name] = False
        with pytest.raises(ValueError):
            study.verify_request(bad, counts)
    with pytest.raises(ValueError):
        study.verify_request(record, dict(counts, decode_updates=126))
