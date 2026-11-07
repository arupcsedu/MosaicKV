"""CPU-only tests: never invoke Slurm or change a real job."""

import runpy
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/control_a100_schedule.py"
API = runpy.run_path(str(SCRIPT))
EASTERN = ZoneInfo("America/New_York")


@pytest.mark.parametrize(
    ("clock", "expected"),
    [
        ("00:00", (4, 4)),
        ("04:27", (4, 4)),
        ("04:28", (4, 4)),
        ("06:29", (4, 4)),
        ("06:30", (4, 4)),
        ("07:00", (4, 4)),
        ("23:00", (4, 4)),
        ("23:59", (4, 4)),
    ],
)
def test_windows(clock, expected):
    now = datetime.fromisoformat(f"2026-10-13T{clock}:00").replace(tzinfo=EASTERN)
    assert API["limits"](now) == expected


def test_utc_and_daylight_saving():
    # Midnight and both repeated DST hours retain the same four-GPU limit.
    assert API["limits"](datetime.fromisoformat("2026-10-13T04:00:00+00:00")) == (4, 4)
    for fold in (0, 1):
        assert API["limits"](datetime(2026, 11, 1, 1, 30, tzinfo=EASTERN, fold=fold)) == (4, 4)
    with pytest.raises(ValueError, match="Timezone-aware"):
        API["limits"](datetime(2026, 10, 13))


def row(array, index, state="PENDING", held=True, gpus=1):
    return {
        "array": str(array),
        "id": f"{array}_{index}",
        "state": state,
        "name": "mk-test",
        "gres": f"gres/gpu:a100:{gpus}",
        "limit": "2:00:00",
        "reason": "(JobHeldUser)" if held else "(Resources)",
    }


def test_aggregate_not_per_array():
    rows = [row(1, i, "RUNNING", False) for i in range(4)]
    rows += [row(1, i) for i in range(4, 10)] + [row(2, i) for i in range(10)]
    assert API["choose_releases"](rows, {"1", "2"}, 4) == []
    with pytest.raises(ValueError, match="exceed four"):
        API["choose_releases"](rows, {"1", "2"}, 8)


def test_pending_admissions_and_other_own_jobs_reserve_slots():
    rows = [row(99, 0, "RUNNING", False, 2), row(1, 0, held=False)]
    rows += [row(2, i) for i in range(6)]
    assert len(API["choose_releases"](rows, {"1", "2"}, 4)) == 1


def test_no_releases_during_drain_and_no_active_task_selected():
    rows = [row(1, i, "RUNNING", False) for i in range(8)] + [row(2, 0)]
    assert API["choose_releases"](rows, {"1", "2"}, 4) == []
    rows = [row(1, 0, "COMPLETING", False)] + [row(2, i) for i in range(8)]
    assert len(API["choose_releases"](rows, {"1", "2"}, 4)) == 3


def test_single_remaining_array_never_exceeds_four():
    rows = [row(2, i) for i in range(20)]
    assert len(API["choose_releases"](rows, {"1", "2"}, 4)) == 4


def test_known_broken_array_stays_held():
    rows = [row(1, i) for i in range(10)] + [row(2, i) for i in range(10)]
    selected = API["choose_releases"](rows, {"1", "2"}, 4, {"2"})
    assert len(selected) == 4
    assert all(item["array"] == "1" for item in selected)


def test_job_scope_and_finite_duration():
    assert API["task_expression"]("123", ["123_2", "123_1"]) == "123_[1,2]"
    with pytest.raises(ValueError, match="Not an element"):
        API["task_expression"]("123", ["999_1"])
    assert API["duration_minutes"]("2:00:00") == 120
    assert API["duration_minutes"]("2:00:01") == 121
    assert API["duration_minutes"]("1-00:00:00") == 1440
    assert API["gpu_count"]("N/A") == 0


@pytest.mark.parametrize(
    ("value", "minutes"),
    [("00:00", 0), ("00:01", 1), ("45:00", 45), ("45:01", 46),
     ("59:59", 60), ("00:45:00", 45), ("2:00:01", 121), ("1-02:03:04", 1564)],
)
def test_squeue_short_duration_and_rounding(value, minutes):
    assert API["duration_minutes"](value) == minutes


@pytest.mark.parametrize(
    "value", ["UNLIMITED", "Partition_Limit", "N/A", "", "45", "-1:00",
              "1-45:00", "00:60", "1:60:00", "0:00:60"],
)
def test_invalid_duration_fails_closed(value):
    with pytest.raises(ValueError, match="finite Slurm"):
        API["duration_minutes"](value)


def test_short_quality_jobs_do_not_stall_other_arrays(tmp_path, monkeypatch):
    namespace = API["tick"].__globals__
    rows = [row(1, i) for i in range(3)] + [row(2, i) for i in range(2)]
    rows[3]["limit"] = rows[4]["limit"] = "45:00"
    calls = []
    monkeypatch.setitem(namespace, "read_queue", lambda: rows)
    monkeypatch.setitem(namespace, "command", lambda *args: calls.append(args) or "")
    policy = {"arrays": [1, 2], "adopt_existing_holds": [1, 2], "maximum_worker_minutes": 120}
    result = API["tick"](policy, {}, tmp_path / "state.json", True)
    assert len(result["released"]) == 4
    assert set(result["released"]) == {"1_0", "1_1", "2_0", "2_1"}
    assert not any(call[0] == "scancel" for call in calls)


def test_hold_barrier_precedes_throttle_and_release(tmp_path, monkeypatch):
    namespace = API["tick"].__globals__
    rows = [row(1, 0, held=False), row(2, 0)]
    calls = []

    def fake_command(*args):
        calls.append(args)
        if args[1] == "hold":
            for item in rows:
                if args[2].startswith(item["array"] + "_"):
                    item["reason"] = "(JobHeldUser)"
        return ""

    monkeypatch.setitem(namespace, "read_queue", lambda: rows)
    monkeypatch.setitem(namespace, "command", fake_command)
    policy = {"arrays": [1, 2], "adopt_existing_holds": [2], "maximum_worker_minutes": 120}
    API["tick"](policy, {}, tmp_path / "state.json", True)
    actions = [call[1] for call in calls]
    assert max(i for i, action in enumerate(actions) if action == "hold") < actions.index("update")
    last_update = max(i for i, action in enumerate(actions) if action == "update")
    assert last_update < actions.index("release")
    assert not any("cancel" in call or "suspend" in call for call in calls)


def test_failed_hold_is_retried_before_any_release(tmp_path, monkeypatch):
    namespace = API["tick"].__globals__
    rows = [row(1, 0, held=False)]
    calls = []
    failures = [True]

    def fake_command(*args):
        calls.append(args)
        if args[1] == "hold":
            if failures.pop() if failures else False:
                raise RuntimeError("temporary scheduler failure")
            rows[0]["reason"] = "(JobHeldUser)"
        return ""

    monkeypatch.setitem(namespace, "read_queue", lambda: rows)
    monkeypatch.setitem(namespace, "command", fake_command)
    monkeypatch.setitem(namespace, "limits", lambda *_: (4, 4))
    policy = {"arrays": [1], "maximum_worker_minutes": 120}
    state = {"initialized": ["1"], "admission_cap": 4}
    with pytest.raises(RuntimeError, match="temporary"):
        API["tick"](policy, state, tmp_path / "state.json", True)
    assert state["hold_intents"] == ["1_0"]
    assert not any(call[1] == "release" for call in calls)
    API["tick"](policy, state, tmp_path / "state.json", True)
    assert [call[1] for call in calls] == ["hold", "hold", "release"]
    assert not state["hold_intents"]


@pytest.mark.parametrize("throttle", [4, 8])
def test_finished_sibling_update_requires_readback(monkeypatch, throttle):
    namespace = API["set_array_throttle"].__globals__

    def fake_command(*args):
        if args[1] == "update":
            raise subprocess.CalledProcessError(1, args, stderr="123_0: Job has already finished\n")
        return f"ArrayJobId=123 JobState=PENDING ArrayTaskThrottle={throttle}\n"

    monkeypatch.setitem(namespace, "command", fake_command)
    if throttle == 4:
        API["set_array_throttle"]("123")
    else:
        with pytest.raises(RuntimeError, match="could not be verified"):
            API["set_array_throttle"]("123")
