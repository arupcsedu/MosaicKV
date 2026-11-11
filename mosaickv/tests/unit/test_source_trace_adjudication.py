from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import evicastkv_source_trace_adjudication as subject
import pytest
from evicastkv_source_ownership_closure import conservative_report, generation_for_observation

from mosaickv.measurements.kv_timeline import AllocationTimeline


def fixture(tmp_path, *, unknown_bytes=200, overlap=False, reuse=False, code=None):
    file = tmp_path / "operators.py"
    file.write_text(
        code or ("def cache_operator():\n    return 1\n\ndef pointwise_operator():\n    return 2\n")
    )
    frames = [dict(filename=str(file), name="cache_operator", line=2)]
    unknown = [dict(filename=str(file), name="pointwise_operator", line=5)]
    timeline = AllocationTimeline()
    timeline.seed(1, 100)
    timeline.allocations[(1, 0)]["evidence"] = [
        "pinned immutable resident model parameters/buffers"
    ]
    events = [dict(action="alloc", addr=10, size=100, time_us=10, frames=frames)]
    if not overlap:
        events.append(dict(action="free_completed", addr=10, time_us=20))
    events.append(
        dict(
            action="alloc", addr=10 if reuse else 20, size=unknown_bytes, time_us=30, frames=unknown
        )
    )
    events.append(dict(action="free_completed", addr=10 if reuse else 20, time_us=40))
    if overlap:
        events.append(dict(action="free_completed", addr=10, time_us=50))
    timeline.ingest(events)
    capture = timeline.report(
        complete=True, history_complete=True, opaque_workspace_classified=False
    )
    clock = dict(minimum_offset_us=0, maximum_offset_us=0)
    capture["clock_calibration"] = clock
    observations = [
        dict(
            phase="known_KV",
            begin_us=11,
            end_us=12,
            declared_cuda_bytes=100,
            owners=[dict(device="cuda:0", address=10, bytes=100)],
        )
    ]
    declared = dict(observations=observations)
    conservative = conservative_report(capture, deepcopy(observations), events=timeline.events)
    trace = dict(
        actual_events=timeline.events,
        allocator_records=capture["records"],
        stack_table=capture["stack_table"],
        clock_calibration=clock,
        installed_and_local_source_hashes={
            str(file): dict(sha256=subject.sha(file), canonical_path=str(file.resolve()))
        },
    )
    ledger = dict(
        schema_version=1,
        boundary=subject.BOUNDARY,
        review_status="explicit_source_inspection_complete",
        reviewed_by="CPU fixture review",
        rules=[],
    )
    return trace, conservative, declared, ledger, file


def rule_for(trace, file, *, stack_id=1, role="non_kv"):
    frame = trace["stack_table"][stack_id][0]
    return dict(
        stack_id=stack_id,
        stack_sha256=subject.stack_fingerprint(trace["stack_table"][stack_id]),
        role=role,
        reason="Specific fixture pointwise output has no cache ownership.",
        frame_index=0,
        filename=frame[0],
        function=frame[1],
        line=frame[2],
        source_sha256=subject.sha(file),
        source_line_sha256=hashlib.sha256(
            file.read_text().splitlines(keepends=True)[frame[2] - 1].encode()
        ).hexdigest(),
        qualified_function=frame[1],
        files={str(file): subject.sha(file)},
        reviewed_by="CPU fixture review",
        review_status="explicit_source_inspection_complete",
        match_mode="exact_full_stack",
    )


def test_unresolved_is_not_nonKV_or_automatically_certified(tmp_path):
    trace, conservative, declared, ledger, _ = fixture(tmp_path)
    before = deepcopy((trace, conservative, declared, ledger))
    result = subject.adjudicate(trace, conservative, declared, ledger)
    assert result["known_KV_peak_lower_bound_bytes"] == 100
    assert result["KV_peak_upper_bound_including_unknown_bytes"] == 200
    assert result["unknown_generations_preserved"] == 1
    assert not result["exact_PyTorch_boundary_peak_certified"]
    assert (trace, conservative, declared, ledger) == before


def test_indexed_conservative_replay_is_identical_to_V11_predicate(tmp_path):
    trace, conservative, declared, _, _ = fixture(tmp_path, reuse=True)
    capture = dict(
        records=trace["allocator_records"],
        events=len(trace["actual_events"]),
        stack_table=trace["stack_table"],
        clock_calibration=trace["clock_calibration"],
        complete_request=True,
        allocator_history_complete=True,
        blocks_exact_certification=conservative["blocks_exact_certification"],
    )
    reference = conservative_report(
        capture, deepcopy(declared["observations"]), events=trace["actual_events"]
    )
    indexed = subject.indexed_conservative_report(
        capture, deepcopy(declared["observations"]), events=trace["actual_events"]
    )
    assert indexed == reference


def test_index_preserves_ambiguous_and_missing_clock_semantics(tmp_path):
    trace, _, _, _, _ = fixture(tmp_path, reuse=True)
    rows = trace["allocator_records"]
    owner = dict(device="cuda:0", address=10, bytes=100)
    clock = trace["clock_calibration"]
    for observation in [
        dict(begin_us=11, end_us=12),
        dict(begin_us=19, end_us=31),
        dict(begin_us=31, end_us=32),
    ]:
        assert generation_for_observation(
            rows, owner, observation, clock
        ) == generation_for_observation(
            [row for row in rows if row["address"] == owner["address"]], owner, observation, clock
        )
    assert generation_for_observation(rows, owner, dict(begin_us=11, end_us=12), None) is None


def test_explicit_pinned_rule_closes_only_scoped_boundary(tmp_path):
    trace, conservative, declared, ledger, file = fixture(tmp_path)
    ledger["rules"] = [rule_for(trace, file)]
    result = subject.adjudicate(trace, conservative, declared, ledger)
    assert result["exact_PyTorch_boundary_peak_certified"]
    assert result["exact_KV_peak_bytes"] == 100
    assert result["complete_allocation_attribution"]
    assert result["exact_whole_request_KV_peak_certified"]
    assert result["exact_whole_request_source_KV_plus_attention_workspace_peak_certified"]
    assert not result["exact_whole_request_KV_only_peak_certified"]
    assert not result["whole_device_memory_certified"]


def test_peak_equality_can_certify_scalar_without_waiving_unknowns(tmp_path):
    trace, conservative, declared, ledger, _ = fixture(tmp_path, unknown_bytes=20)
    result = subject.adjudicate(trace, conservative, declared, ledger)
    assert result["exact_KV_peak_bytes"] == 100
    assert result["certificate_kind"] == "equal_interval_bounds"
    assert result["unknown_generations_preserved"] == 1
    assert not result["complete_allocation_attribution"]


def test_overlapping_unknown_cannot_pass_scalar_gate(tmp_path):
    trace, conservative, declared, ledger, _ = fixture(tmp_path, unknown_bytes=20, overlap=True)
    result = subject.adjudicate(trace, conservative, declared, ledger)
    assert result["KV_peak_upper_bound_including_unknown_bytes"] == 120
    assert result["exact_KV_peak_bytes"] is None


def test_pointer_reuse_remains_distinct_generations(tmp_path):
    trace, conservative, declared, ledger, file = fixture(tmp_path, reuse=True)
    ledger["rules"] = [rule_for(trace, file, role="kv_workspace")]
    result = subject.adjudicate(trace, conservative, declared, ledger)
    reused = [row for row in result["records"] if row["address"] == 10]
    assert [row["generation"] for row in reused] == [0, 1]
    assert result["exact_KV_peak_bytes"] == 200


def test_changed_source_file_refused(tmp_path):
    trace, conservative, declared, ledger, file = fixture(tmp_path)
    ledger["rules"] = [rule_for(trace, file)]
    file.write_text(file.read_text() + "# changed\n")
    with pytest.raises(ValueError, match="source file changed"):
        subject.adjudicate(trace, conservative, declared, ledger)


def test_changed_stack_or_incomplete_fingerprint_refused(tmp_path):
    trace, conservative, declared, ledger, file = fixture(tmp_path)
    ledger["rules"] = [rule_for(trace, file)]
    ledger["rules"][0]["stack_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="exact stack rule"):
        subject.adjudicate(trace, conservative, declared, ledger)


def test_ambiguous_or_duplicate_rules_refused(tmp_path):
    trace, conservative, declared, ledger, file = fixture(tmp_path)
    ledger["rules"] = [rule_for(trace, file), rule_for(trace, file, role="kv_workspace")]
    with pytest.raises(ValueError, match="ambiguous"):
        subject.adjudicate(trace, conservative, declared, ledger)


@pytest.mark.parametrize(
    "change",
    [
        {"match_mode": "suffix"},
        {"fallback": "non_kv"},
        {"role": "model"},
        {"review_status": "not_reviewed"},
    ],
)
def test_no_generic_or_unreviewed_rule_fallback(tmp_path, change):
    trace, conservative, declared, ledger, file = fixture(tmp_path)
    rule = rule_for(trace, file)
    rule.update(change)
    ledger["rules"] = [rule]
    with pytest.raises(ValueError):
        subject.adjudicate(trace, conservative, declared, ledger)


def test_claimed_flags_do_not_forge_a_certificate(tmp_path):
    trace, conservative, declared, ledger, _ = fixture(tmp_path)
    conservative.update(exact_KV_peak_bytes=1, exact_whole_request_KV_peak_certified=True)
    ledger["superiority_established"] = True
    result = subject.adjudicate(trace, conservative, declared, ledger)
    assert result["exact_KV_peak_bytes"] is None
    assert not result["superiority_established"]


def test_forged_interval_fails_independent_trace_replay(tmp_path):
    trace, conservative, declared, ledger, _ = fixture(tmp_path)
    trace["allocator_records"][1]["end"] = 1000
    with pytest.raises(ValueError, match="independently replayed"):
        subject.adjudicate(trace, conservative, declared, ledger)


def test_duplicate_raw_generation_is_not_hidden_by_dictionary_deduplication(tmp_path):
    trace, conservative, declared, ledger, _ = fixture(tmp_path)
    trace["allocator_records"][2] = deepcopy(trace["allocator_records"][1])
    with pytest.raises(ValueError, match="duplicate/malformed"):
        subject.adjudicate(trace, conservative, declared, ledger)


@pytest.mark.parametrize("value", ["true", 1, None])
def test_forged_completeness_types_do_not_certify(tmp_path, value):
    trace, conservative, declared, ledger, _ = fixture(tmp_path)
    conservative["complete_request"] = value
    with pytest.raises(ValueError, match="strict boolean"):
        subject.adjudicate(trace, conservative, declared, ledger)


def test_unknown_allocator_action_requires_review_not_automatic_coverage(tmp_path):
    trace, conservative, declared, ledger, _ = fixture(tmp_path)
    trace["actual_events"][0]["action"] = "unreviewed_allocator_event"
    with pytest.raises(ValueError, match="unrecognized"):
        subject.adjudicate(trace, conservative, declared, ledger)


def test_forged_saved_model_attribution_refused(tmp_path):
    trace, conservative, declared, ledger, _ = fixture(tmp_path)
    conservative["records"][2]["role"] = "model"
    with pytest.raises(ValueError, match="does not reproduce declared owners"):
        subject.adjudicate(trace, conservative, declared, ledger)


def test_review_cannot_override_declared_KV(tmp_path):
    trace, conservative, declared, ledger, file = fixture(tmp_path)
    ledger["rules"] = [rule_for(trace, file, stack_id=0)]
    with pytest.raises(ValueError, match="conflicts with an actual declared"):
        subject.adjudicate(trace, conservative, declared, ledger)


def test_broad_model_forward_nonKV_rule_refused(tmp_path):
    trace, conservative, declared, ledger, file = fixture(tmp_path)
    file.write_text("class LlamaModel:\n    def forward(self):\n        return 2\n")
    frame = [str(file), "forward", 3]
    trace["stack_table"][1] = [frame]
    for event in trace["actual_events"]:
        if event.get("addr") == 20 and event["action"] == "alloc":
            event["frames"] = [dict(filename=str(file), name="forward", line=3)]
    trace["installed_and_local_source_hashes"][str(file)]["sha256"] = subject.sha(file)
    rule = rule_for(trace, file)
    rule["qualified_function"] = "LlamaModel.forward"
    ledger["rules"] = [rule]
    with pytest.raises(ValueError, match="broad model-forward"):
        subject.adjudicate(trace, conservative, declared, ledger)


def test_incomplete_history_prevents_certification_even_after_rule(tmp_path):
    trace, conservative, declared, ledger, file = fixture(tmp_path)
    conservative["allocator_history_complete"] = False
    ledger["rules"] = [rule_for(trace, file)]
    result = subject.adjudicate(trace, conservative, declared, ledger)
    assert not result["exact_PyTorch_boundary_peak_certified"]
    assert result["exact_KV_peak_bytes"] is None
    assert result["KV_peak_upper_bound_including_unknown_bytes"] is None


def test_source_line_and_qualified_scope_are_both_checked(tmp_path):
    trace, conservative, declared, ledger, file = fixture(tmp_path)
    ledger["rules"] = [rule_for(trace, file)]
    ledger["rules"][0]["qualified_function"] = "Wrong.operator"
    with pytest.raises(ValueError, match="actual source scope"):
        subject.adjudicate(trace, conservative, declared, ledger)


def test_no_source_rule_inventory_from_unavailable_trace(tmp_path):
    with pytest.raises(FileNotFoundError):
        subject.inventory(tmp_path / "not_finished")


def test_metadata_reader_rejects_oversize(tmp_path, monkeypatch):
    path = tmp_path / "large.json"
    path.write_text(json.dumps({"x": "metadata"}))
    monkeypatch.setattr(subject, "MAX_COMPRESSED", 2)
    with pytest.raises(ValueError, match="bounded offline audit size"):
        subject.load(path)
