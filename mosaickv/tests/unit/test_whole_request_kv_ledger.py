"""Synthetic CPU lifecycle gates; these fixtures are NOT CUDA measurements."""

from copy import deepcopy

import pytest
import torch

from mosaickv.evaluation import kv_allocator_ledger as allocator
from mosaickv.evaluation import whole_request_kv_ledger as module


def _snapshot():
    return [
        dict(
            device=0,
            address=4096,
            total_size=2048,
            is_expandable=True,
            blocks=[
                dict(size=512, requested_size=128, state="active_allocated"),
                dict(size=1024, requested_size=900, state="active_allocated"),
                dict(size=512, requested_size=0, state="inactive"),
            ],
        )
    ]


@pytest.fixture
def synthetic_ledger(monkeypatch):
    """Exercise the REAL delta builder/replayer over explicitly fake blocks."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda.memory, "get_allocator_backend", lambda: "native")
    monkeypatch.setattr(torch.cuda, "memory_snapshot", _snapshot)
    monkeypatch.setattr(allocator, "tensor_owners", lambda tensors, *, device: tuple(tensors))
    policy = "backend:native,expandable_segments:True"
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", policy)
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", policy)

    def factory(*, source=True):
        return module.WholeRequestKVLedger(
            device="cuda:0",
            layer_count=2,
            source=source,
            decode_updates=1,
            ledger_factory=allocator.AllocatorLedger,
        )

    return factory


def _evidence():
    return dict(
        native_layers_retired=2,
        maximum_live_native_cache_layers=1,
        outstanding_native_owner_references=0,
        all_native_owner_weakrefs_released=True,
        uses_original_fullkv_snapshot=False,
        original_logical_positions_preserved=True,
        constructor_workspace_shared_with_decode=True,
        benchmark_labels_read=False,
        fallback_used=False,
    )


def _phases(*, source=True):
    phases = ["request_workspace_created"]
    for layer in range(2):
        if source:
            phases.append(f"prefill_layer_{layer}_source_captured")
        phases.extend(
            f"prefill_layer_{layer}_{name}"
            for name in ("KV_published", "packet_constructed", "native_retired")
        )
    return [
        *phases,
        "prefill_complete",
        "decode:initial_cache",
        "decode:decode_update_1_layer_0_growth_overlap",
        "decode:decode_update_1_layer_0_copied_history",
        "decode:decode_update_1_layer_1_growth_overlap",
        "decode:decode_update_1_layer_1_copied_history",
        "decode:final_cache",
    ]


def _populate(ledger, phases):
    for phase in phases:
        # Peak during each native-to-packet overlap must survive final summary.
        owners = ((4096, 128), (4608, 900)) if "packet_constructed" in phase else ((4096, 128),)
        ledger.observe(owners, phase)
    return ledger


@pytest.mark.parametrize("source", [False, True])
def test_two_layer_real_synthetic_delta_replay_covers_both_producers(synthetic_ledger, source):
    ledger = _populate(synthetic_ledger(source=source), _phases(source=source))
    report = ledger.finish(_evidence())
    assert report["whole_request_KV_peak_certified"] is True
    assert report["whole_request_GPU_peak_certified"] is False
    assert report["independent_quality_confirmation"] is False
    assert report["paper_campaign_authorized"] is False
    assert report["initial_allocator_block_bytes"] == 512
    assert report["final_allocator_block_bytes"] == 512
    assert report["peak_allocator_block_bytes"] == 1536
    assert report["initial_tensor_backing_bytes"] == 128
    assert report["final_tensor_backing_bytes"] == 128
    assert report["peak_tensor_backing_bytes"] == 1028
    assert report["replay"]["ownership_deltas_verified"] is True
    assert report["replay"]["observations"] == len(_phases(source=source))
    # Generic allocator ledger deliberately never certifies whole-request scope.
    assert report["allocator"]["whole_request_peak_certified"] is False


@pytest.mark.parametrize("source", [False, True])
@pytest.mark.parametrize("change", ["missing", "duplicate", "reordered"])
@pytest.mark.parametrize("phase", ["KV_published", "packet_constructed", "native_retired"])
def test_each_native_layer_transition_must_be_once_and_in_order(
    synthetic_ledger, source, change, phase
):
    phases = _phases(source=source)
    index = phases.index(f"prefill_layer_1_{phase}")
    if change == "missing":
        del phases[index]
    elif change == "duplicate":
        phases.insert(index, phases[index])
    else:
        phases[index], phases[index - 1] = phases[index - 1], phases[index]
    ledger = _populate(synthetic_ledger(source=source), phases)
    with pytest.raises(ValueError, match=r"phase|order"):
        ledger.finish(_evidence())


@pytest.mark.parametrize("layer", [0, 1])
@pytest.mark.parametrize("change", ["missing", "duplicate", "reordered"])
def test_source_capture_must_precede_its_own_native_publication(synthetic_ledger, layer, change):
    phases = _phases()
    index = phases.index(f"prefill_layer_{layer}_source_captured")
    if change == "missing":
        del phases[index]
    elif change == "duplicate":
        phases.insert(index, phases[index])
    else:
        phases[index], phases[index + 1] = phases[index + 1], phases[index]
    with pytest.raises(ValueError, match=r"phase|order"):
        _populate(synthetic_ledger(), phases).finish(_evidence())


@pytest.mark.parametrize(
    "key,bad",
    [
        ("native_layers_retired", 1),
        ("maximum_live_native_cache_layers", 2),
        ("outstanding_native_owner_references", 1),
        ("all_native_owner_weakrefs_released", False),
        ("uses_original_fullkv_snapshot", True),
        ("original_logical_positions_preserved", False),
        ("constructor_workspace_shared_with_decode", False),
        ("benchmark_labels_read", True),
        ("fallback_used", True),
    ],
)
def test_retirement_is_not_inferred_from_metadata_or_fallback(synthetic_ledger, key, bad):
    evidence = _evidence()
    evidence[key] = bad
    with pytest.raises(ValueError, match=r"retirement|labels|fallback"):
        _populate(synthetic_ledger(), _phases()).finish(evidence)


@pytest.mark.parametrize("key", list(_evidence()))
def test_every_lifecycle_evidence_field_is_required(synthetic_ledger, key):
    evidence = _evidence()
    del evidence[key]
    with pytest.raises(ValueError):
        _populate(synthetic_ledger(), _phases()).finish(evidence)


@pytest.mark.parametrize(
    "key",
    [
        "all_native_owner_weakrefs_released",
        "uses_original_fullkv_snapshot",
        "original_logical_positions_preserved",
        "constructor_workspace_shared_with_decode",
        "benchmark_labels_read",
        "fallback_used",
    ],
)
@pytest.mark.parametrize("bad", [None, "true", "false", 0, 1])
def test_boolean_evidence_never_accepts_truthy_or_falsy_substitutes(synthetic_ledger, key, bad):
    evidence = _evidence()
    evidence[key] = bad
    with pytest.raises(ValueError):
        _populate(synthetic_ledger(), _phases()).finish(evidence)


@pytest.mark.parametrize(
    "boundary",
    ["request_workspace_created", "prefill_complete", "decode:initial_cache", "decode:final_cache"],
)
def test_missing_request_boundaries_block_certification(synthetic_ledger, boundary):
    phases = _phases()
    phases.remove(boundary)
    with pytest.raises(ValueError):
        _populate(synthetic_ledger(), phases).finish(_evidence())


def test_decode_cannot_start_before_prefill_finishes(synthetic_ledger):
    phases = _phases()
    phases.remove("decode:initial_cache")
    phases.insert(1, "decode:initial_cache")
    with pytest.raises(ValueError, match="decode cannot precede"):
        _populate(synthetic_ledger(), phases).finish(_evidence())


@pytest.mark.parametrize("phase", [None, 1, "", "x" * 251])
def test_phase_labels_are_bounded_explicit_strings(synthetic_ledger, phase):
    ledger = synthetic_ledger()
    with pytest.raises(ValueError, match="phase"):
        ledger.observe(((4096, 128),), phase)
    assert ledger.phases == []


def test_successful_finish_and_later_observation_are_write_once(synthetic_ledger):
    ledger = _populate(synthetic_ledger(), _phases())
    ledger.finish(_evidence())
    with pytest.raises(RuntimeError, match="write-once"):
        ledger.finish(_evidence())
    with pytest.raises(RuntimeError, match="already closed"):
        ledger.observe(((4096, 128),), "decode:final_cache")


def test_actual_allocator_hash_tamper_blocks_whole_request_certification(synthetic_ledger):
    ledger = _populate(synthetic_ledger(), _phases())
    ledger.ledger.ownership_deltas[0]["current_tensor_backing_bytes"] += 1
    with pytest.raises(ValueError):
        ledger.finish(_evidence())
    assert ledger.closed is False


def test_phase_coverage_must_match_actual_allocator_events(synthetic_ledger):
    ledger = _populate(synthetic_ledger(), _phases())
    ledger.ledger.observe(((4096, 128),), phase="unregistered_extra_event")
    with pytest.raises(ValueError, match="lifecycle phases"):
        ledger.finish(_evidence())


def test_unresolved_allocator_owner_blocks_certification(synthetic_ledger, monkeypatch):
    ledger = _populate(synthetic_ledger(), _phases())
    summary = deepcopy(ledger.ledger.summary())
    summary["all_required_owners_resolved"] = False
    monkeypatch.setattr(ledger.ledger, "summary", lambda: summary)
    with pytest.raises(ValueError, match="unresolved"):
        ledger.finish(_evidence())


@pytest.mark.parametrize("count", [True, False, 0, -1, 1.0, "2", None])
def test_layer_count_is_strict_positive_integer(count):
    with pytest.raises(ValueError, match="layer count"):
        module.WholeRequestKVLedger(device="cpu", layer_count=count, source=True)


@pytest.mark.parametrize("source", [None, "true", 0, 1])
def test_source_mode_must_be_explicit_boolean(source):
    with pytest.raises(ValueError, match="boolean"):
        module.WholeRequestKVLedger(device="cpu", layer_count=2, source=source)


@pytest.mark.parametrize("count", [True, False, 0, -1, 1.0, "127", None])
def test_decode_updates_is_strict_positive_integer(count):
    with pytest.raises(ValueError, match="decode_updates"):
        module.WholeRequestKVLedger(device="cpu", layer_count=2, source=True, decode_updates=count)


@pytest.mark.parametrize(
    "key",
    [
        "native_layers_retired",
        "maximum_live_native_cache_layers",
        "outstanding_native_owner_references",
    ],
)
@pytest.mark.parametrize("value", [True, False, "0", "1", "2", 0.0, 1.0, 2.0, None])
def test_retirement_counts_reject_booleans_floats_and_strings(synthetic_ledger, key, value):
    evidence = _evidence()
    evidence[key] = value
    with pytest.raises(ValueError, match="strict integer"):
        _populate(synthetic_ledger(), _phases()).finish(evidence)


@pytest.mark.parametrize("boundary", ["decode:initial_cache", "decode:final_cache"])
def test_decode_boundaries_cannot_be_duplicated(synthetic_ledger, boundary):
    phases = _phases()
    phases.insert(phases.index(boundary), boundary)
    with pytest.raises(ValueError, match="duplicate"):
        _populate(synthetic_ledger(), phases).finish(_evidence())


@pytest.mark.parametrize("layer", [0, 1])
@pytest.mark.parametrize("change", ["missing", "duplicate", "reordered"])
def test_each_registered_decode_layer_update_is_covered_once(synthetic_ledger, layer, change):
    phases = _phases()
    phase = f"decode:decode_update_1_layer_{layer}_copied_history"
    index = phases.index(phase)
    if change == "missing":
        del phases[index]
    elif change == "duplicate":
        phases.insert(index, phase)
    else:
        phases.remove(phase)
        phases.insert(phases.index("decode:initial_cache"), phase)
    with pytest.raises(ValueError, match=r"phase|order"):
        _populate(synthetic_ledger(), phases).finish(_evidence())


def test_decode_layer_update_order_cannot_change_with_complete_counts(synthetic_ledger):
    phases = _phases()
    first = phases.index("decode:decode_update_1_layer_0_copied_history")
    last = phases.index("decode:decode_update_1_layer_1_copied_history")
    phases[first], phases[last] = phases[last], phases[first]
    with pytest.raises(ValueError, match="update/layer order"):
        _populate(synthetic_ledger(), phases).finish(_evidence())


def test_extra_unregistered_decode_updates_are_rejected(synthetic_ledger):
    phases = _phases()
    phases.insert(-1, "decode:decode_update_2_layer_0_copied_history")
    with pytest.raises(ValueError, match="unregistered"):
        _populate(synthetic_ledger(), phases).finish(_evidence())


@pytest.mark.parametrize("source", [False, True])
def test_saved_certificate_independently_replays_lifecycle_and_exact_boundaries(
    synthetic_ledger, source
):
    ledger = _populate(synthetic_ledger(source=source), _phases(source=source))
    record = ledger.finish(_evidence())
    deltas = record["allocator"]["ownership_deltas"]
    replay = module.replay_whole_request(
        record, deltas, layer_count=2, source=source, decode_updates=1
    )
    assert replay["whole_request_KV_peak_certified"] is True
    assert replay["whole_request_GPU_peak_certified"] is False
    assert replay["decode_updates"] == 1
    assert replay["initial_allocator_block_bytes"] == record["initial_allocator_block_bytes"]
    assert replay["final_allocator_block_bytes"] == record["final_allocator_block_bytes"]
    assert replay["peak_allocator_block_bytes"] == record["peak_allocator_block_bytes"]


@pytest.mark.parametrize(
    "field",
    [
        "initial_allocator_block_bytes",
        "final_allocator_block_bytes",
        "peak_allocator_block_bytes",
        "initial_tensor_backing_bytes",
        "final_tensor_backing_bytes",
        "peak_tensor_backing_bytes",
        "decode_updates",
    ],
)
def test_saved_certificate_rejects_tampered_boundary_bytes(synthetic_ledger, field):
    record = _populate(synthetic_ledger(), _phases()).finish(_evidence())
    record[field] += 1
    with pytest.raises(ValueError, match="independent replay"):
        module.replay_whole_request(
            record,
            record["allocator"]["ownership_deltas"],
            layer_count=2,
            source=True,
            decode_updates=1,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("whole_request_KV_peak_certified", False),
        ("whole_request_KV_peak_certified", 1),
        ("whole_request_GPU_peak_certified", True),
        ("whole_request_GPU_peak_certified", 0),
        ("initial_boundary", "request_workspace_created"),
        ("final_boundary", "prefill_complete"),
        ("scope", "allocator_envelope"),
        ("decode_updates", True),
    ],
)
def test_saved_certificate_rejects_mislabeled_boundaries_or_flags(synthetic_ledger, field, value):
    record = _populate(synthetic_ledger(), _phases()).finish(_evidence())
    record[field] = value
    with pytest.raises(ValueError, match="independent replay"):
        module.replay_whole_request(
            record,
            record["allocator"]["ownership_deltas"],
            layer_count=2,
            source=True,
            decode_updates=1,
        )


def test_saved_certificate_cannot_claim_default_127_updates_from_one_step(synthetic_ledger):
    record = _populate(synthetic_ledger(), _phases()).finish(_evidence())
    with pytest.raises(ValueError, match="decode_update_2"):
        module.replay_whole_request(
            record, record["allocator"]["ownership_deltas"], layer_count=2, source=True
        )
