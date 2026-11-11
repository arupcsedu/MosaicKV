import pytest

from mosaickv.measurements.native_event_ledger import NativeEventLedger
from mosaickv.measurements.snapshot_states import is_active_state, is_pending_state


@pytest.mark.parametrize("state", ["active_pending_free", "active_awaiting_free"])
def test_both_reviewed_pending_spellings_count_until_completed_free(state):
    assert is_active_state(state) and is_pending_state(state)
    baseline = dict(device=0, addr=1234, size=1031, pool=[0, 1], state=state)
    ledger = NativeEventLedger([baseline])
    result = ledger.ingest(
        dict(
            events=[dict(baseline, action="free_completed", sequence=0)],
            count=1,
            capacity=2,
            complete=True,
            overflow=False,
        )
    ).verify_snapshot([])
    assert result["observed_peak_requested_bytes"] == 1031
    assert result["final_requested_bytes"] == 0


def test_known_allocated_and_inactive_states_are_distinct():
    assert is_active_state("active_allocated") and not is_pending_state("active_allocated")
    assert not is_active_state("inactive") and not is_pending_state("inactive")
    with pytest.raises(ValueError, match="inactive"):
        NativeEventLedger([dict(device=0, addr=100, size=4, pool=[0, 0], state="inactive")])


@pytest.mark.parametrize("state", ["active_future_state", "", None, 0, [], {}])
def test_unknown_snapshot_state_cannot_silently_disappear(state):
    with pytest.raises(ValueError, match="unreviewed"):
        is_active_state(state)
    with pytest.raises(ValueError, match="unreviewed"):
        is_pending_state(state)
