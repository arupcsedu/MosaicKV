import copy

import pytest

from mosaickv.measurements.native_event_ledger import NativeEventLedger


def event(action, address=100, size=1031, pool=(1, 2), device=0):
    return dict(action=action, addr=address, size=size, pool=list(pool), device=device)


def capture(events):
    return dict(
        events=[dict(item, sequence=index) for index, item in enumerate(events)],
        count=len(events),
        capacity=100,
        overflow=False,
        complete=True,
    )


def test_requested_not_rounded_bytes_and_delayed_free_generation():
    ledger = NativeEventLedger().ingest(
        capture(
            [
                event("alloc"),
                event("free_requested"),
                event("alloc", 200, 2051),
                event("free_completed"),
                event("alloc", size=5),
            ]
        )
    )
    result = ledger.verify_snapshot([event("alloc", size=5), event("alloc", 200, 2051)])
    assert result["observed_peak_requested_bytes"] == 3082
    assert result["final_requested_bytes"] == 2056
    assert [r["generation"] for r in ledger.records if r["addr"] == 100] == [0, 1]
    assert all(r["semantic_role"] == "unknown" for r in ledger.records)
    assert not result["exact_source_KV_peak_certified"]


def test_identical_device_addresses_are_distinct_and_aliases_not_new_allocations():
    events = [event("alloc"), event("alloc", device=1)]
    ledger = NativeEventLedger().ingest(capture(events))
    result = ledger.verify_snapshot(events)
    assert result["final_requested_bytes"] == 2062
    assert result["generations"] == 2


def test_unknown_baseline_is_retained_and_must_be_in_final_snapshot():
    base = event("alloc", 900, 17, pool=(0, 0))
    ledger = NativeEventLedger([base]).ingest(capture([event("alloc")]))
    assert ledger.verify_snapshot([base, event("alloc")])["final_requested_bytes"] == 1048
    assert ledger.records[0]["baseline"] is True
    with pytest.raises(ValueError, match="final requested backings"):
        ledger.verify_snapshot([event("alloc")])


def test_preexisting_pending_free_is_not_retired_before_completion():
    baseline = dict(event("alloc", size=9), state="active_awaiting_free")
    ledger = NativeEventLedger([baseline]).ingest(capture([event("free_completed", size=9)]))
    result = ledger.verify_snapshot([])
    assert result["observed_peak_requested_bytes"] == 9
    assert result["final_requested_bytes"] == 0


@pytest.mark.parametrize("change", ["drop", "overflow", "count", "capacity", "sequence", "bool"])
def test_incomplete_or_invalid_capture_fails_closed(change):
    value = capture([event("alloc")])
    if change == "drop":
        value["events"] = []
    elif change == "overflow":
        value["overflow"] = True
    elif change == "count":
        value["count"] += 1
    elif change == "capacity":
        value["capacity"] = 0
    elif change == "bool":
        value["events"][0]["sequence"] = False
    else:
        value["events"][0]["sequence"] += 1
    with pytest.raises(ValueError):
        NativeEventLedger().ingest(value)


@pytest.mark.parametrize(
    "events, message",
    [
        ([event("free_requested")], "free without"),
        ([event("alloc"), event("alloc")], "overlapping"),
        ([event("alloc"), event("free_completed")], "without free_requested"),
        ([event("alloc"), event("free_requested"), event("free_requested")], "duplicate"),
        ([event("alloc"), event("free_requested", pool=(3, 4))], "pool differs"),
        ([event("alloc"), event("free_requested", size=5)], "size differs"),
        ([event("alloc"), event("segment_free", size=4096)], "segment retired"),
        ([event("oom")], "OOM"),
        ([event("made_up")], "unknown"),
        ([event("alloc", size=True)], "integer"),
    ],
)
def test_invalid_generations_fail_closed(events, message):
    with pytest.raises(ValueError, match=message):
        NativeEventLedger().ingest(capture(events))


def test_snapshot_mutation_and_duplicate_owners_are_rejected():
    ledger = NativeEventLedger().ingest(capture([event("alloc")]))
    with pytest.raises(ValueError, match="duplicate"):
        ledger.verify_snapshot([event("alloc"), event("alloc")])
    altered = copy.deepcopy(event("alloc"))
    altered["pool"] = [2, 3]
    with pytest.raises(ValueError, match="final requested backings"):
        ledger.verify_snapshot([altered])
