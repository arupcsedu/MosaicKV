"""Synthetic CPU provenance tests; no inferred allocation is GPU evidence."""

import gc
import weakref
from copy import deepcopy

import pytest

from mosaickv.evaluation.kv_allocator_ledger import resolve_snapshot
from mosaickv.evaluation.live_owner_snapshot_cache import (
    LiveOwnerSnapshotCache,
    OwnerMetadata,
    resolution_mapping_sha256,
)
from mosaickv.evaluation.snapshot_reuse_replay import SnapshotReuseReplay

PHASE = "prefill_layer_0_constructor_workspace"


def mapping(pointer=4096, nbytes=126):
    return resolve_snapshot(
        [
            dict(
                address=pointer,
                total_size=512,
                device=0,
                is_expandable=True,
                blocks=[dict(size=512, requested_size=nbytes, state="active_allocated")],
            )
        ],
        [(pointer, nbytes)],
        device=0,
    )


def proof(index=0, *, phase=PHASE, reused=False, origin=0, calls=1, resolved=None):
    resolved = mapping() if resolved is None else resolved
    return dict(
        observation_index=index,
        phase=phase,
        enabled=True,
        scope="consecutive_constructor_observations_only",
        prior_weak_representatives_alive=reused,
        observation_skipped=False,
        allocator_bytes_inferred=False,
        outcome="resolved",
        mode="reused_actual_resolution" if reused else "fresh_native_snapshot",
        reason=(
            "identical_continuously_live_constructor_owners"
            if reused
            else "no_cached_snapshot"
            if phase == PHASE
            else "mandatory_lifecycle_snapshot"
        ),
        source_observation_index=origin,
        snapshot_calls=calls,
        owner_view_fingerprint_sha256="a" * 64,
        resolution_mapping_sha256=resolution_mapping_sha256(resolved),
        weak_representative_count=1,
    )


def first():
    replay = SnapshotReuseReplay()
    replay.observe(proof(), index=0, phase=PHASE, resolved=mapping())
    return replay


def test_fresh_then_multiple_constructor_reuses_preserve_all_observations():
    replay = first()
    for index in (1, 2, 3):
        replay.observe(proof(index, reused=True), index=index, phase=PHASE, resolved=mapping())
    result = replay.summary()
    assert result["observations"] == 4
    assert result["fresh"] == result["snapshot_calls"] == 1
    assert result["reused"] == 3
    assert result["every_observation_has_proof"] is True
    assert result["allocator_bytes_inferred"] is False
    assert result["whole_request_KV_peak_certified"] is False


@pytest.mark.parametrize(
    "phase",
    [
        "prefill_complete",
        "decode:initial_cache",
        "prefill_layer_0_KV_published",
        "prefill_layer_00_constructor_workspace",
    ],
)
def test_no_lifecycle_or_near_match_phase_can_reuse_a_snapshot(phase):
    replay = first()
    with pytest.raises(ValueError, match="consecutive constructor"):
        replay.observe(proof(1, phase=phase, reused=True), index=1, phase=phase, resolved=mapping())


def test_lifecycle_fresh_snapshot_interrupts_constructor_reuse():
    replay = first()
    phase = "prefill_complete"
    replay.observe(
        proof(1, phase=phase, origin=1, calls=2), index=1, phase=phase, resolved=mapping()
    )
    with pytest.raises(ValueError, match="consecutive constructor"):
        replay.observe(proof(2, reused=True, calls=2), index=2, phase=PHASE, resolved=mapping())


def test_changed_constructor_phase_cannot_reuse_the_previous_layer_mapping():
    replay = first()
    other = "prefill_layer_1_constructor_workspace"
    with pytest.raises(ValueError, match="consecutive constructor"):
        replay.observe(proof(1, phase=other, reused=True), index=1, phase=other, resolved=mapping())


@pytest.mark.parametrize(
    "field,value",
    [
        ("scope", "all_lifecycle_phases"),
        ("enabled", False),
        ("enabled", 1),
        ("observation_skipped", True),
        ("observation_skipped", 0),
        ("allocator_bytes_inferred", True),
        ("allocator_bytes_inferred", 0),
        ("outcome", "failed_closed"),
        ("phase", "other"),
        ("observation_index", 2),
        ("observation_index", True),
        ("source_observation_index", True),
        ("snapshot_calls", True),
        ("weak_representative_count", True),
        ("prior_weak_representatives_alive", 1),
    ],
)
def test_control_fields_and_indices_cannot_be_coerced_or_tampered(field, value):
    replay = first()
    bad = proof(1, reused=True)
    bad[field] = value
    with pytest.raises(ValueError):
        replay.observe(bad, index=1, phase=PHASE, resolved=mapping())


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_observation_index", 1),
        ("snapshot_calls", 2),
        ("prior_weak_representatives_alive", False),
        ("weak_representative_count", 2),
        ("owner_view_fingerprint_sha256", "b" * 64),
        ("reason", "owner_or_view_set_changed"),
    ],
)
def test_reuse_requires_unchanged_origin_calls_fingerprint_and_live_owners(field, value):
    replay = first()
    bad = proof(1, reused=True)
    bad[field] = value
    with pytest.raises(ValueError, match="changed origin"):
        replay.observe(bad, index=1, phase=PHASE, resolved=mapping())


def test_reused_mapping_cannot_change_actual_owners_even_if_its_new_hash_is_correct():
    replay = first()
    changed = mapping(pointer=8192)
    bad = proof(1, reused=True, resolved=changed)
    with pytest.raises(ValueError, match="changed origin"):
        replay.observe(bad, index=1, phase=PHASE, resolved=changed)


def test_forged_mapping_hash_cannot_hide_owner_transition():
    replay = first()
    with pytest.raises(ValueError, match="actual physical mapping"):
        replay.observe(proof(1, reused=True), index=1, phase=PHASE, resolved=mapping(8192))


@pytest.mark.parametrize("field", ["owner_view_fingerprint_sha256", "resolution_mapping_sha256"])
@pytest.mark.parametrize("value", [None, "a" * 63, "X" * 64, 1, False])
def test_hashes_require_canonical_sha256_strings(field, value):
    replay = first()
    bad = proof(1, reused=True)
    bad[field] = value
    with pytest.raises(ValueError, match="SHA-256"):
        replay.observe(bad, index=1, phase=PHASE, resolved=mapping())


@pytest.mark.parametrize("field", list(proof()))
def test_missing_any_proof_field_fails_closed(field):
    replay = first()
    bad = proof(1, reused=True)
    del bad[field]
    with pytest.raises(ValueError):
        replay.observe(bad, index=1, phase=PHASE, resolved=mapping())


def test_an_omitted_proof_cannot_be_hidden_by_later_success_or_summary():
    replay = first()
    with pytest.raises(ValueError, match="requires its explicit proof"):
        replay.observe(None, index=1, phase=PHASE, resolved=mapping())
    with pytest.raises(RuntimeError, match="failed observation"):
        replay.observe(proof(1, reused=True), index=1, phase=PHASE, resolved=mapping())
    with pytest.raises(RuntimeError, match="invalid or missing"):
        replay.summary()


@pytest.mark.parametrize("index", [0, 2, True, 1.0, "1", None])
def test_observation_indices_cannot_skip_or_repeat(index):
    replay = first()
    with pytest.raises(ValueError):
        replay.observe(proof(1, reused=True), index=index, phase=PHASE, resolved=mapping())


@pytest.mark.parametrize(
    "field,value", [("source_observation_index", 0), ("snapshot_calls", 1), ("snapshot_calls", 3)]
)
def test_fresh_snapshot_must_increment_counter_exactly_once_and_own_its_origin(field, value):
    replay = first()
    fresh = proof(1, origin=1, calls=2)
    fresh[field] = value
    with pytest.raises(ValueError, match="one new call"):
        replay.observe(fresh, index=1, phase=PHASE, resolved=mapping())


def test_reused_first_observation_has_no_actual_snapshot_origin():
    replay = SnapshotReuseReplay()
    with pytest.raises(ValueError, match="consecutive constructor"):
        replay.observe(proof(0, reused=True), index=0, phase=PHASE, resolved=mapping())


def test_empty_replay_cannot_be_summarized_as_complete():
    with pytest.raises(ValueError, match="complete observation"):
        SnapshotReuseReplay().summary()


def test_nonresolved_owner_mapping_cannot_support_a_valid_proof():
    replay = first()
    changed = mapping()
    changed["all_required_owners_resolved"] = False
    bad = proof(1, reused=True, resolved=changed)
    with pytest.raises(ValueError, match="actually resolved"):
        replay.observe(bad, index=1, phase=PHASE, resolved=changed)


class Owner:
    pass


def test_real_helper_provenance_replays_without_retaining_owner_or_mapping_objects():
    owner = Owner()
    snapshots = []

    def snapshot():
        snapshots.append(True)
        return [
            dict(
                address=4096,
                device=0,
                total_size=512,
                blocks=[dict(size=512, requested_size=126, state="active_allocated")],
            )
        ]

    helper = LiveOwnerSnapshotCache(
        device=0,
        enabled=True,
        owner_metadata_fn=lambda tensor: OwnerMetadata(
            tensor, 0, 1, 4096, 126, (0, (63,), (1,), "torch.bfloat16", 2)
        ),
        snapshot_fn=snapshot,
    )
    replay = SnapshotReuseReplay()
    phases = [PHASE, PHASE, "prefill_complete", PHASE, PHASE]
    for index, phase in enumerate(phases):
        resolved = helper.resolve([owner], phase)
        replay.observe(resolved["snapshot_cache"], index=index, phase=phase, resolved=resolved)
    assert len(snapshots) == replay.summary()["fresh"] == 3
    assert replay.summary()["reused"] == 2
    assert replay.summary()["observations"] == 5
    reference = weakref.ref(owner)
    del owner
    gc.collect()
    assert reference() is None
    # The checker retains scalar metadata only, never a mutable input mapping.
    original = deepcopy(replay.summary())
    resolved["owners"][0][0] = 0
    resolved["snapshot_cache"]["snapshot_calls"] = 900
    assert replay.summary() == original
