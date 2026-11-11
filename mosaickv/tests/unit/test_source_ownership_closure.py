from __future__ import annotations

import gc
from types import SimpleNamespace

import evicastkv_source_ownership_closure as subject
import pytest
import torch

from mosaickv.measurements.kv_timeline import AllocationTimeline


def captured():
    timeline = AllocationTimeline()
    timeline.seed(1, 100)
    timeline.allocations[(1, 0)]["role"] = "model"
    timeline.allocations[(1, 0)]["evidence"] = [
        "pinned immutable resident model parameters/buffers"
    ]
    timeline.ingest(
        [
            {"action": "alloc", "addr": 10, "size": 64, "time_us": 10},
            {"action": "free_completed", "addr": 10, "time_us": 20},
            {"action": "alloc", "addr": 10, "size": 64, "time_us": 30},
            {"action": "alloc", "addr": 20, "size": 32, "time_us": 31},
        ]
    )
    result = timeline.report(
        complete=True, history_complete=True, opaque_workspace_classified=False
    )
    result["clock_calibration"] = {"minimum_offset_us": 0, "maximum_offset_us": 0}
    return result, timeline.events


def point(begin, end, *, address=10, size=64):
    return dict(
        phase="observed",
        begin_us=begin,
        end_us=end,
        declared_cuda_bytes=size,
        owners=[dict(device="cuda:0", address=address, bytes=size, roles=["source"])],
    )


def test_pointer_reuse_links_exact_generation_not_latest():
    capture, events = captured()
    observed = [point(11, 12), point(32, 33)]
    result = subject.conservative_report(capture, observed, events=events)
    assert [item["owners"][0]["allocation_generation"] for item in observed] == [0, 1]
    assert result["known_KV_peak_lower_bound_bytes"] == 64
    assert result["KV_peak_upper_bound_including_unknown_bytes"] == 96
    assert not result["exact_whole_request_KV_peak_certified"]


def test_uncertain_time_interval_never_guesses_generation():
    capture, events = captured()
    observed = [point(19, 31)]
    result = subject.conservative_report(capture, observed, events=events)
    assert result["declared_owner_marks_linked"] == 0
    assert result["unresolved_owner_observations"]
    assert result["unknown_allocation_generations"] == 3


def test_unknown_is_not_non_KV_and_legacy_certificate_is_not_inherited():
    capture, events = captured()
    capture["exact_whole_request_KV_peak_certified"] = True
    capture["exact_KV_peak_bytes"] = 0
    capture["records"][1]["role"] = "non_kv"
    result = subject.conservative_report(capture, [], events=events)
    assert result["unknown_allocation_generations"] == 3
    assert result["KV_peak_upper_bound_including_unknown_bytes"] == 96
    assert result["exact_KV_peak_bytes"] is None
    assert not result["whole_device_memory_certified"]


def test_incomplete_capture_has_no_upper_certificate():
    capture, events = captured()
    capture["allocator_history_complete"] = False
    result = subject.conservative_report(capture, [], events=events)
    assert result["KV_peak_upper_bound_including_unknown_bytes"] is None
    assert not result["exact_whole_request_KV_peak_certified"]


def test_structural_history_failure_invalidates_interval_bounds_not_snapshot_lower():
    capture, events = captured()
    capture["blocks_exact_certification"].append("free_without_observed_allocation")
    result = subject.conservative_report(capture, [point(11, 12)], events=events)
    assert not result["allocator_interval_bounds_valid"]
    assert result["known_KV_peak_lower_bound_bytes"] is None
    assert result["KV_peak_upper_bound_including_unknown_bytes"] is None
    assert result["observed_declared_cuda_peak_lower_bound_bytes"] == 64


def test_forged_event_count_refused():
    capture, events = captured()
    with pytest.raises(ValueError, match="actual allocator event count"):
        subject.conservative_report(capture, [], events=events[:-1])


def test_missing_allocator_timestamps_stay_unlinked():
    capture, events = captured()
    capture["records"][1]["time_us"] = None
    result = subject.conservative_report(capture, [point(11, 12)], events=events)
    assert result["declared_owner_marks_linked"] == 0


def test_aliases_deduplicate_and_weak_observation_does_not_keep_storage_alive():
    ledger = subject.DeclaredOwners()
    value = torch.arange(8)
    alias = value[2:]
    ledger.track("original", [value])
    ledger.track("view", [alias])
    first = ledger.observe("both")
    assert len(first["owners"]) == 1
    assert first["declared_CPU_bytes"] == value.untyped_storage().nbytes()
    del value, alias
    gc.collect()
    assert ledger.observe("retired")["owners"] == []


def test_declared_KV_cannot_steal_model_owner():
    capture, events = captured()
    with pytest.raises(ValueError, match="aliases an immutable model"):
        subject.conservative_report(capture, [point(11, 12, address=1, size=100)], events=events)


def test_source_file_hash_binding_and_missing_JIT_frames(tmp_path):
    path = tmp_path / "operator.py"
    path.write_text("def source_operator():\n    return None\n")
    result = subject.stack_source_hashes(
        [[(str(path), "source_operator", 1)], [("<generated>", "forward", 1)]]
    )
    assert result[str(path)]["sha256"] == subject.sha(path)
    assert result["<generated>"]["status"] == "source_not_available_not_waived"


def test_unknown_cluster_total_is_not_reported_as_peak():
    capture, events = captured()
    result = subject.conservative_report(capture, [], events=events)
    groups = subject.unknown_clusters(result)
    assert groups[0]["total_allocated_bytes"] == 160
    assert groups[0]["simultaneous_peak_bytes"] == 96
    assert groups[0]["allocation_generations"] == 3
    assert not groups[0]["classified_non_KV"]


def test_nonstreaming_hooks_are_restored_on_failure(monkeypatch):
    import evicastkv_review_source_controls as controls
    import evicastkv_review_source_prefix as source

    from mosaickv.evaluation import shared_source_format, storage_runs
    from mosaickv.evaluation.graphed_storage import SharedSourceCache

    adapter = SimpleNamespace(prefill=lambda: None, decode_one_token=lambda: None)
    original = {
        "prefill": adapter.prefill,
        "decode": adapter.decode_one_token,
        "formats": controls._formats,
        "chosen": source._formats_with_ceiling,
        "construct": shared_source_format.construct,
        "fit": shared_source_format.fit_transport,
        "cache": storage_runs.cache_from_formats,
        "observe": SharedSourceCache.observe,
    }
    with (
        pytest.raises(RuntimeError, match="deliberate"),
        subject.instrument(adapter, subject.DeclaredOwners()),
    ):
        raise RuntimeError("deliberate")
    assert adapter.prefill is original["prefill"]
    assert adapter.decode_one_token is original["decode"]
    assert controls._formats is original["formats"]
    assert source._formats_with_ceiling is original["chosen"]
    assert shared_source_format.construct is original["construct"]
    assert shared_source_format.fit_transport is original["fit"]
    assert storage_runs.cache_from_formats is original["cache"]
    assert SharedSourceCache.observe is original["observe"]


def test_bounded_candidate_and_retirement_capture_uses_only_actual_tensors(monkeypatch):
    import evicastkv_review_source_controls as controls
    import evicastkv_review_source_prefix as source

    from mosaickv.evaluation import shared_source_format, storage_runs
    from mosaickv.evaluation.graphed_storage import SharedSourceCache

    pair = (torch.zeros(2), torch.ones(2))
    state = SimpleNamespace(past_key_values=(pair,))
    adapter = SimpleNamespace(
        prefill=lambda: SimpleNamespace(state=state),
        decode_one_token=lambda: None,
        _legacy_layers=lambda value: (value, None),
    )
    normalized = SimpleNamespace(records={0: torch.zeros(4)})
    attention = SimpleNamespace(records={}, windows={0: torch.zeros(3)}, priors={})
    rotary = SimpleNamespace(phases=(torch.ones(2), torch.zeros(2)))
    monkeypatch.setattr(
        shared_source_format,
        "construct",
        lambda: SimpleNamespace(tensors=lambda: (torch.zeros(3),)),
    )
    monkeypatch.setattr(shared_source_format, "fit_transport", lambda packet: (packet, {}))

    def builder(*args):
        packet = shared_source_format.construct()
        return (shared_source_format.fit_transport(packet)[0],), {}

    monkeypatch.setattr(controls, "_formats", builder)

    def chosen(*args):
        candidates = [controls._formats(*args) for _ in range(3)]
        source.choose_attainable([], 1)
        return candidates[-1]

    monkeypatch.setattr(source, "_formats_with_ceiling", chosen)
    monkeypatch.setattr(source, "choose_attainable", lambda *args: {})
    cache = SimpleNamespace(shared_tensors=pair, layers=[])
    monkeypatch.setattr(storage_runs, "cache_from_formats", lambda: cache)
    monkeypatch.setattr(SharedSourceCache, "observe", lambda *args: None)
    ledger = subject.DeclaredOwners()
    with subject.instrument(adapter, ledger) as counts:
        adapter.prefill()
        source._formats_with_ceiling(
            adapter, None, None, attention, normalized, rotary, subject.ARM
        )
        storage_runs.cache_from_formats()
        SharedSourceCache.observe(cache, ())
        for _ in range(127):
            adapter.decode_one_token()
    phases = [item["phase"] for item in ledger.observations]
    assert "finite_ladder_all_candidates_live_before_choice" in phases
    assert "sources_retired_before_cache_construction" in phases
    assert counts == dict(builders=3, constructs=3, fits=3, updates=127)
    assert all(item["checkpoint_not_exact_transient_peak"] for item in ledger.observations)


def test_no_silent_partial_decode_success():
    adapter = SimpleNamespace(prefill=lambda: None, decode_one_token=lambda: None)
    with (
        pytest.raises(ValueError, match="complete registered decode"),
        subject.instrument(adapter, subject.DeclaredOwners()),
    ):
        pass
