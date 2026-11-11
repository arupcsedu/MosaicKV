"""Synthetic native-history generation fixtures; not GPU free evidence."""

from copy import deepcopy

import pytest

from mosaickv.evaluation import native_retirement_receipts as module
from mosaickv.evaluation.native_retirement_receipts import (
    allocation_generations,
    bind_generations,
    validate_trace_prefix,
    verify_receipts,
    verify_retired,
)


def event(action, address=100, size=64, stream=0):
    return dict(action=action, addr=address, size=size, stream=stream)


def snapshot(address=100, requested=64, block_size=64):
    return [
        dict(
            address=address,
            total_size=block_size,
            device=0,
            is_expandable=True,
            blocks=[
                dict(
                    address=address,
                    size=block_size,
                    requested_size=requested,
                    state="active_allocated",
                )
            ],
        )
    ]


def fixture(label="layer_0"):
    before = [event("alloc"), event("snapshot")]
    token = bind_generations(before, snapshot(), [(100, 48)], device=0, label=label)
    after = [*deepcopy(before), event("free_requested"), event("free_completed")]
    return before, token, after


def synthetic_receipts():
    """Two generation-replayed CPU fixtures, never measured GPU evidence."""
    _, first, events = fixture()
    receipt0 = verify_retired(events, first)
    events = [*events, event("alloc", size=16)]
    second = bind_generations(
        events, snapshot(requested=16), [(100, 16)], device=0, label="layer_1"
    )
    events += [event("free_requested", size=16), event("free_completed", size=16)]
    return [receipt0, verify_retired(events, second)]


def test_actual_generation_free_completed_and_independent_receipt_replay():
    before, token, after = fixture()
    receipt = verify_retired(after, token)
    assert receipt["allocations"][0]["alloc_index"] == 0
    assert receipt["allocations"][0]["free_completed_index"] == 3
    assert receipt["allocations"][0]["owner_ranges"] == [[100, 48]]
    assert not receipt["whole_request_KV_peak_certified"]
    assert verify_receipts([receipt], expected_layers=1)["complete"]
    assert validate_trace_prefix(before, after, capacity=8)


def test_weak_death_or_free_requested_alone_never_proves_completed_free():
    before, token, after = fixture()
    for incomplete in (before, after[:-1]):
        with pytest.raises(ValueError, match="FREE_COMPLETED"):
            verify_retired(incomplete, token)


def test_pointer_reuse_does_not_satisfy_or_confuse_an_old_generation():
    _, token, after = fixture()
    reused = [*after, event("alloc", size=16)]
    first = verify_retired(reused, token)
    assert first["allocations"][0]["generation"] == 0
    second = bind_generations(
        reused, snapshot(requested=16), [(100, 16)], device=0, label="layer_1"
    )
    assert second["allocations"][0]["generation"] == 1
    assert second["allocations"][0]["alloc_index"] == 4
    with pytest.raises(ValueError, match="FREE_COMPLETED"):
        verify_retired(reused, second)
    complete = [*reused, event("free_requested", size=16), event("free_completed", size=16)]
    assert verify_receipts([first, verify_retired(complete, second)], expected_layers=2)["complete"]


def test_aliases_bind_one_physical_generation_and_do_not_add_bytes():
    before, _token, _after = fixture()
    token = bind_generations(
        before, snapshot(), [(100, 64), (100, 64), (116, 16)], device=0, label="layer_0"
    )
    assert len(token["allocations"]) == 1
    assert token["unique_backing_bytes"] == token["physical_block_bytes"] == 64


@pytest.mark.parametrize("alter", ("drop_prefix", "modify_prefix", "full_ring", "oom", "unknown"))
def test_bounded_trace_prefix_rejects_truncation_ring_full_and_unrecognized_events(alter):
    before, _token, after = fixture()
    capacity = 10
    if alter == "drop_prefix":
        after = after[1:]
    elif alter == "modify_prefix":
        after[0]["size"] += 1
    elif alter == "full_ring":
        capacity = len(after)
    elif alter == "oom":
        after.append(dict(action="oom"))
    else:
        after.append(dict(action="not_a_native_event"))
    with pytest.raises(ValueError):
        validate_trace_prefix(before, after, capacity=capacity)


def test_generation_reducer_rejects_reuse_before_free_or_inconsistent_free_sizes():
    for events in (
        [event("alloc"), event("alloc")],
        [event("alloc"), event("free_completed")],
        [event("alloc"), event("free_requested", size=63)],
        [event("alloc"), event("free_requested"), event("free_requested")],
    ):
        with pytest.raises(ValueError):
            allocation_generations(events)


def test_baseline_free_without_observed_alloc_cannot_be_bound():
    events = [event("free_requested"), event("free_completed")]
    with pytest.raises(ValueError, match="observed active ALLOC"):
        bind_generations(events, snapshot(), [(100, 48)], device=0, label="layer_0")


@pytest.mark.parametrize("change", ("missing", "label", "flag", "trace", "address", "range"))
def test_receipt_cpu_replay_rejects_forged_or_incomplete_evidence(change):
    _before, token, after = fixture()
    receipt = verify_retired(after, token)
    receipts = [deepcopy(receipt)]
    if change == "missing":
        receipts = []
    elif change == "label":
        receipts[0]["label"] = "layer_1"
    elif change == "flag":
        receipts[0]["actual_free_completed_verified"] = 1
    elif change == "trace":
        receipts[0]["trace_events"][-1]["action"] = "snapshot"
    elif change == "address":
        receipts[0]["source_token"]["allocations"][0]["address"] += 1
    else:
        receipts[0]["source_token"]["allocations"][0]["owner_ranges"][0][1] = 1024
    with pytest.raises(ValueError):
        verify_receipts(receipts, expected_layers=1)


def test_snapshot_request_and_unseen_or_cross_device_owner_fail_closed():
    before, _token, _after = fixture()
    with pytest.raises(ValueError, match="ALLOC size"):
        bind_generations(before, snapshot(requested=63), [(100, 48)], device=0, label="layer_0")
    with pytest.raises(ValueError):
        bind_generations(before, snapshot(), [(100, 65)], device=0, label="layer_0")
    with pytest.raises(ValueError):
        bind_generations(before, snapshot(), [(100, 48)], device=1, label="layer_0")


def test_two_layer_synthetic_receipts_have_consistent_trace_prefixes():
    receipts = synthetic_receipts()
    replay = verify_receipts(receipts, expected_layers=2)
    assert replay["complete"] is True
    assert replay["generations"] == 2
    assert receipts[1]["source_token"]["allocations"][0]["generation"] == 1
    assert all(len(r["source_token"]["captured_snapshot_sha256"]) == 64 for r in receipts)


@pytest.mark.parametrize(
    "field", ["address", "requested_bytes", "generation", "allocation_stream", "alloc_index"]
)
@pytest.mark.parametrize("value", [True, False, 1.0, "1", None])
def test_token_integer_fields_reject_boolean_float_or_missing(field, value):
    _, token, after = fixture()
    token["allocations"][0][field] = value
    with pytest.raises(ValueError):
        verify_retired(after, token)


@pytest.mark.parametrize(
    "field", ["device", "captured_trace_count", "unique_backing_bytes", "physical_block_bytes"]
)
@pytest.mark.parametrize("value", [True, False, 1.0, "1", None])
def test_token_top_level_integer_fields_are_strict(field, value):
    _, token, after = fixture()
    token[field] = value
    with pytest.raises(ValueError):
        verify_retired(after, token)


@pytest.mark.parametrize("field", ["unique_backing_bytes", "physical_block_bytes"])
def test_native_byte_totals_cannot_be_rewritten(field):
    _, token, after = fixture()
    token[field] += 1
    with pytest.raises(ValueError, match="totals"):
        verify_retired(after, token)


def test_same_generation_cannot_be_counted_twice_within_one_token():
    _, token, after = fixture()
    token["allocations"].append(deepcopy(token["allocations"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        verify_retired(after, token)


def test_layer_receipts_cannot_mix_independent_allocator_history_prefixes():
    receipts = synthetic_receipts()
    receipts[1]["trace_events"][1]["action"] = "segment_alloc"
    with pytest.raises(ValueError, match="prefix"):
        verify_receipts(receipts, expected_layers=2)


@pytest.mark.parametrize(
    "field", ["actual_free_completed_verified", "weakref_death_is_not_free_evidence"]
)
def test_receipt_true_flags_require_literal_booleans(field):
    receipts = synthetic_receipts()
    receipts[0][field] = 1
    with pytest.raises(ValueError):
        verify_receipts(receipts, expected_layers=2)


@pytest.mark.parametrize("layers", [True, False, 0, -1, 2.0, "2", None])
def test_receipt_expected_layer_count_is_strict(layers):
    with pytest.raises(ValueError):
        verify_receipts(synthetic_receipts(), expected_layers=layers)


class FakeHistory:
    """Explicit fake native events for context-boundary CPU tests."""

    def __init__(self):
        self.enabled, self.events, self.active = False, [], False
        self.calls = []

    def record(self, *, enabled="all", **kwargs):
        self.calls.append(dict(enabled=enabled, **kwargs))
        self.enabled = enabled is not None
        if kwargs.get("clear_history"):
            self.events = []

    def read(self):
        if self.enabled:
            self.events.append(event("snapshot", address=0, size=0))
        return dict(
            segments=snapshot() if self.active else [], device_traces=[deepcopy(self.events)]
        )

    def allocate(self):
        self.active = True
        self.events.append(event("alloc"))

    def free(self):
        self.active = False
        self.events.extend([event("free_requested"), event("free_completed")])


@pytest.fixture
def fake_history(monkeypatch):
    history = FakeHistory()
    monkeypatch.setattr(module.torch, "__version__", module._TORCH_VERSION)
    monkeypatch.setattr(module.torch.version, "git_version", module._TORCH_GIT)
    monkeypatch.setattr(module.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(module.torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(module.torch.cuda.memory, "get_allocator_backend", lambda: "native")
    monkeypatch.setattr(module.torch.cuda.memory, "_snapshot", history.read)
    monkeypatch.setattr(module.torch.cuda.memory, "_record_memory_history", history.record)
    monkeypatch.setattr(module.torch._C, "_cuda_isHistoryEnabled", lambda: history.enabled)
    monkeypatch.setattr(module, "tensor_owners", lambda tensors, *, device: tuple(tensors))
    monkeypatch.setattr(module.torch, "empty", lambda *args, **kwargs: object())
    return history


def test_owned_tracker_context_cpu_mock_restores_history_and_replays(fake_history):
    with module.CUDAFreeCompletedTracker(device="cuda:0") as tracker:
        fake_history.allocate()
        token = tracker.capture(((100, 48),), label="layer_0")
        fake_history.free()
        receipt = tracker.verify(token)
    assert tracker.summary()["complete"] is True
    assert verify_receipts([receipt], expected_layers=1)["complete"] is True
    assert fake_history.enabled is False
    assert fake_history.calls[0]["enabled"] == "all"
    assert fake_history.calls[0]["context"] is None
    assert fake_history.calls[0]["stacks"] == "python"
    assert fake_history.calls[0]["max_entries"] == 200_000
    assert fake_history.calls[-1]["enabled"] is None


def test_tracker_cannot_reenter_or_replace_another_active_history(fake_history):
    with (
        module.CUDAFreeCompletedTracker(device="cuda:0"),
        pytest.raises(RuntimeError, match="non-reentrant"),
        module.CUDAFreeCompletedTracker(device="cuda:0"),
    ):
        pass
    fake_history.enabled = True
    with (
        pytest.raises(RuntimeError, match="existing native memory tracing"),
        module.CUDAFreeCompletedTracker(device="cuda:0"),
    ):
        pass
    assert fake_history.enabled is True


def test_tracker_preserves_preexisting_inactive_events(fake_history):
    fake_history.events = [event("snapshot", address=0, size=0)]
    with (
        pytest.raises(RuntimeError, match="preexisting inactive history"),
        module.CUDAFreeCompletedTracker(device="cuda:0"),
    ):
        pass
    assert fake_history.calls == []


def test_tracker_restores_own_history_after_failure_without_claiming_success(fake_history):
    with (
        pytest.raises(RuntimeError, match="injected"),
        module.CUDAFreeCompletedTracker(device="cuda:0") as tracker,
    ):
        raise RuntimeError("injected bounded diagnostic failure")
    assert fake_history.enabled is False
    with pytest.raises(RuntimeError, match="successful"):
        tracker.summary()


def test_tracker_rejects_unverified_live_owner_on_context_exit(fake_history):
    with (
        pytest.raises(RuntimeError, match="not every"),
        module.CUDAFreeCompletedTracker(device="cuda:0") as tracker,
    ):
        fake_history.allocate()
        tracker.capture(((100, 48),), label="layer_0")
    assert fake_history.enabled is False


def test_tracker_rejects_copied_mutated_and_duplicate_tokens(fake_history):
    with module.CUDAFreeCompletedTracker(device="cuda:0") as tracker:
        fake_history.allocate()
        token = tracker.capture(((100, 48),), label="layer_0")
        with pytest.raises(ValueError, match="unique layer label"):
            tracker.capture(((100, 48),), label="layer_0")
        fake_history.free()
        with pytest.raises(ValueError, match="not captured"):
            tracker.verify(deepcopy(token))
        token["label"] = "layer_9"
        with pytest.raises(ValueError, match="mutated"):
            tracker.verify(token)
        token["label"] = "layer_0"
        tracker.verify(token)
        with pytest.raises(ValueError, match="one retirement"):
            tracker.verify(token)


def test_tracker_rejects_trace_rollover_even_if_target_free_is_visible(fake_history):
    with (
        pytest.raises(ValueError, match="prefix"),
        module.CUDAFreeCompletedTracker(device="cuda:0") as tracker,
    ):
        fake_history.allocate()
        token = tracker.capture(((100, 48),), label="layer_0")
        fake_history.free()
        fake_history.events.pop(0)
        tracker.verify(token)


@pytest.mark.parametrize("bound", [True, False, 1, 63, 1_000_001, 200_000.0, "200000"])
def test_tracker_capacity_is_bounded_strict_integer(bound):
    with pytest.raises(ValueError):
        module.CUDAFreeCompletedTracker(device="cuda:0", max_entries=bound)


def test_prefix_failure_evidence_preserves_every_raw_field_and_mismatching_key():
    previous = [dict(event("alloc"), time_us=100), dict(event("snapshot"), time_us=120)]
    current = deepcopy(previous)
    current[0]["time_us"] = 101
    current[1]["size"] = 11
    current[1]["new_optional_field"] = "untouched"
    captured = module.trace_prefix_failure_evidence(
        previous,
        current,
        capacity=200_000,
        error=ValueError("strict actual prefix changed"),
        snapshot_sha256="f" * 64,
    )
    assert captured["acceptance_rule_unchanged"] is True
    assert captured["source_history_fields_unmodified"] is True
    assert captured["raw_previous_events"] == previous
    assert captured["raw_current_events"] == current
    assert captured["all_mismatch_keys"] == ["new_optional_field", "size", "time_us"]
    assert captured["first_difference_index"] == 0
    assert captured["first_previous_event"]["time_us"] == 100
    assert captured["first_current_event"]["time_us"] == 101
    assert captured["mismatches"][1]["added_keys"] == ["new_optional_field"]
    current[0]["time_us"] = 999
    assert captured["raw_current_events"][0]["time_us"] == 101
    assert captured["whole_request_KV_peak_certified"] is False


def test_clock_field_prefix_failure_is_captured_without_relaxing_rejection(fake_history):
    tracker = module.CUDAFreeCompletedTracker(device="cuda:0")
    with pytest.raises(ValueError, match="prefix"), tracker:
        fake_history.events[0]["time_us"] = 1234
        tracker._read()
    assert fake_history.enabled is False
    captured = tracker.failure_evidence
    assert captured["all_mismatch_keys"] == ["time_us"]
    assert captured["raw_current_events"][0]["time_us"] == 1234
    assert "time_us" not in captured["raw_previous_events"][0]
    assert len(captured["current_snapshot_sha256"]) == 64
    with pytest.raises(RuntimeError, match="successful"):
        tracker.summary()


def test_first_failed_read_keeps_complete_evidence_across_context_cleanup(fake_history):
    tracker = module.CUDAFreeCompletedTracker(device="cuda:0")
    with pytest.raises(ValueError, match="prefix"), tracker:
        fake_history.events[0]["time_us"] = 100
        with pytest.raises(ValueError, match="prefix"):
            tracker._read()
        first_evidence = tracker.failure_evidence
        fake_history.events[0]["time_us"] = 200
        tracker._read()
    assert tracker.failure_evidence is first_evidence
    assert tracker.failure_evidence["raw_current_events"][0]["time_us"] == 100
    assert tracker.closed is True


def test_bounded_clock_diagnostic_reports_raw_failure_instead_of_claiming_pass(
    fake_history, monkeypatch
):
    monkeypatch.setattr(module.torch, "ones", lambda *args, **kwargs: object())
    monkeypatch.setattr(module.torch, "ones_like", lambda *args, **kwargs: object())
    monkeypatch.setattr(module.torch, "empty_like", lambda *args, **kwargs: object())
    monkeypatch.setattr(module.torch, "mm", lambda *args, **kwargs: None)

    def clock_changes_during_work(_seconds):
        fake_history.events[0]["time_us"] = 42

    monkeypatch.setattr(module.time, "sleep", clock_changes_during_work)
    report = module.trace_clock_diagnostic(device="cuda:0")
    assert report["strict_prefix_passed"] is False
    assert report["acceptance_rule_unchanged"] is True
    assert report["temporary_allocations_created"] == 20
    assert report["iterations_completed"] == 0
    assert report["history_restored_disabled"] is True
    assert report["failure_evidence"]["all_mismatch_keys"] == ["time_us"]
    assert report["failure_evidence"]["raw_current_events"][0]["time_us"] == 42


def observed_clock_pair():
    """Minimal factual pair from GPU 22826836, event 18; not a model result."""
    before = dict(
        action="alloc",
        addr=12918868992,
        size=1037,
        stream=0,
        time_us=1794426466675621,
    )
    after = dict(before, time_us=1794426466675622)
    return before, after


def test_only_observed_derived_clock_value_can_change_in_native_event_identity():
    before, after = observed_clock_pair()
    assert module.trace_identity([before]) == module.trace_identity([after])
    assert validate_trace_prefix([before], [after, event("snapshot")], capacity=100)
    assert before["time_us"] == 1794426466675621
    assert after["time_us"] == 1794426466675622


@pytest.mark.parametrize(
    "field,value",
    [
        ("action", "free_requested"),
        ("addr", 12918868993),
        ("size", 1038),
        ("stream", 1),
        ("frames", [{"filename": "changed.py", "line": 1, "name": "changed"}]),
        ("compile_context", "changed"),
        ("unknown_optional_field", "added"),
    ],
)
def test_every_nonclock_field_change_still_rejects_prefix(field, value):
    before, after = observed_clock_pair()
    after[field] = value
    with pytest.raises(ValueError, match="prefix"):
        validate_trace_prefix([before], [after], capacity=100)


@pytest.mark.parametrize("field", ["frames", "compile_context"])
def test_existing_numerical_or_nested_metadata_is_not_coerced_or_ignored(field):
    before, after = observed_clock_pair()
    before[field] = [{"line": 1}] if field == "frames" else 1
    after[field] = [{"line": True}] if field == "frames" else True
    with pytest.raises(ValueError, match="prefix"):
        validate_trace_prefix([before], [after], capacity=100)


@pytest.mark.parametrize("bad", [None, True, False, -1, 1794426466675621.0, "1794426466675621"])
def test_snapshot_time_value_requires_nonnegative_strict_integer(bad):
    before, after = observed_clock_pair()
    after["time_us"] = bad
    with pytest.raises(ValueError, match="time_us"):
        validate_trace_prefix([before], [after], capacity=100)


@pytest.mark.parametrize("side", ["before", "after"])
def test_clock_field_presence_is_still_part_of_native_identity(side):
    before, after = observed_clock_pair()
    del (before if side == "before" else after)["time_us"]
    with pytest.raises(ValueError, match="prefix"):
        validate_trace_prefix([before], [after], capacity=100)


def test_v2_capture_and_replay_preserve_raw_clock_histories_separately_from_identity():
    before = [dict(event("alloc"), time_us=100), dict(event("snapshot"), time_us=110)]
    token = bind_generations(before, snapshot(), [(100, 48)], device=0, label="layer_0")
    after = [
        dict(before[0], time_us=101),
        dict(before[1], time_us=111),
        dict(event("free_requested"), time_us=120),
        dict(event("free_completed"), time_us=125),
    ]
    receipt = verify_retired(after, token)
    assert token["schema"] == "native_cuda_generation_retirement_token_v2"
    assert receipt["schema"] == "native_cuda_generation_retirement_receipt_v2"
    assert token["captured_raw_trace_events"] == before
    assert receipt["trace_events"] == after
    assert token["captured_raw_trace_sha256"] != module._sha(after[:2])
    assert token["captured_trace_sha256"] == module._sha(module.trace_identity(after[:2]))
    assert receipt["raw_trace_sha256"] == module._sha(after)
    assert verify_receipts([receipt], expected_layers=1)["complete"] is True


@pytest.mark.parametrize("field", ["captured_raw_trace_sha256", "captured_trace_sha256"])
def test_v2_capture_cannot_rewrite_either_raw_or_identity_hash(field):
    _, token, after = fixture()
    token[field] = "a" * 64
    with pytest.raises(ValueError, match="hash"):
        verify_retired(after, token)


def test_v2_raw_verification_hash_is_independently_replayed():
    receipts = synthetic_receipts()
    receipts[0]["raw_trace_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="independent"):
        verify_receipts(receipts, expected_layers=2)


def test_v2_never_reinterprets_a_v1_raw_trace_token_or_receipt():
    _, token, after = fixture()
    token["schema"] = "native_cuda_generation_retirement_token_v1"
    with pytest.raises(ValueError, match="generation-bound"):
        verify_retired(after, token)
    receipts = synthetic_receipts()
    receipts[0]["schema"] = "native_cuda_generation_retirement_receipt_v1"
    with pytest.raises(ValueError, match="independent"):
        verify_receipts(receipts, expected_layers=2)
