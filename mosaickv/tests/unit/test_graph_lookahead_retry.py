from datetime import datetime
from zoneinfo import ZoneInfo

import evicastkv_graph_lookahead_retry as retry
import pytest

from mosaickv.evaluation.reservation_window import window

DESCRIPTION = (
    "ReservationName=bi_fox_dgx StartTime=2026-11-04T08:00:00 "
    "EndTime=2026-11-05T08:00:00 Flags=DAILY,SPEC_NODES"
)


@pytest.mark.parametrize("hour,minute,begin", [(7, 30, None), (7, 52, "2026-11-05T08:00:01")])
def test_daily_boundary_cannot_cut_off_the_registered_worker(hour, minute, begin):
    now = datetime(2026, 11, 5, hour, minute, tzinfo=ZoneInfo("America/New_York"))
    assert window(DESCRIPTION, now=now)["begin"] == begin


@pytest.mark.parametrize("fault", ["naive", "foreign", "expired", "missing", "no_renewal"])
def test_unknown_or_insufficient_reservation_fails_closed(fault):
    description = DESCRIPTION
    now = datetime(2026, 11, 5, 7, 52, tzinfo=ZoneInfo("America/New_York"))
    if fault == "naive":
        now = now.replace(tzinfo=None)
    elif fault == "foreign":
        description = description.replace("bi_fox_dgx", "foreign")
    elif fault == "expired":
        now = now.replace(hour=9)
    elif fault == "missing":
        description = "ReservationName=bi_fox_dgx"
    else:
        description = description.replace("DAILY,", "")
    with pytest.raises((RuntimeError, ValueError)):
        window(description, now=now)


def test_reuse_prohibits_any_runtime_or_numerical_guard_change():
    retry.compatible("fixture", changed=retry.ALLOWED)
    with pytest.raises(ValueError, match="runtime"):
        retry.compatible(
            "fixture", changed=["mosaickv/src/mosaickv/evaluation/prompt_graph_reader.py"]
        )
