"""Avoid launching bounded GPU work across the site's daily reservation end."""

import re
import subprocess
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


def window(description, *, now=None, seconds=1200, margin=60):
    if type(seconds) is not int or seconds < 1 or type(margin) is not int or margin < 0:
        raise ValueError("positive job duration and nonnegative reservation margin required")
    fields = dict(re.findall(r"(\w+)=(\S+)", description))
    if fields.get("ReservationName") != "bi_fox_dgx":
        raise ValueError("explicit bi_fox_dgx reservation required")
    timezone = ZoneInfo("America/New_York")
    try:
        start, end = (
            datetime.fromisoformat(fields[k]).replace(tzinfo=timezone)
            for k in ("StartTime", "EndTime")
        )
    except (KeyError, ValueError) as error:
        raise ValueError("finite explicit site-local reservation window required") from error
    now = datetime.now(timezone) if now is None else now
    if now.tzinfo is None or end <= start or now >= end:
        raise ValueError("active or future unexpired timezone-aware reservation required")
    remaining = (end - max(now, start)).total_seconds()
    if remaining >= seconds + margin:
        begin = (
            None
            if now >= start
            else (start + timedelta(seconds=1)).isoformat(timespec="seconds")[:19]
        )
    elif "DAILY" in fields.get("Flags", "").split(","):
        begin = (end + timedelta(seconds=1)).isoformat(timespec="seconds")[:19]
    else:
        raise RuntimeError("insufficient reservation time and no registered daily renewal")
    return dict(
        begin=begin,
        remaining_seconds=remaining,
        required_seconds=seconds + margin,
        timezone="America/New_York",
        reservation="bi_fox_dgx",
    )


def actual(*, at_worker_start=False):
    text = subprocess.check_output(
        ["scontrol", "show", "reservation", "bi_fox_dgx", "-o"], text=True
    )
    result = window(text)
    if at_worker_start and result["begin"] is not None:
        raise RuntimeError("reservation too short: refuse heavy inference before renewal")
    return result
