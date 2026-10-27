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


def test_explicit_six_gpu_policy_remains_an_aggregate_limit(tmp_path, monkeypatch):
    rows = [row(99, 0, "RUNNING", False, 2)] + [row(1, i) for i in range(20)]
    assert len(API["choose_releases"](rows, {"1"}, 6, maximum=6)) == 4
    assert len(API["choose_releases"]([row(1, i) for i in range(20)], {"1"}, 6, maximum=6)) == 6
    with pytest.raises(ValueError, match="exceed four"):
        API["choose_releases"](rows, {"1"}, 6)
    namespace = API["tick"].__globals__
    monkeypatch.setitem(namespace, "read_queue", lambda: rows)
    policy = {"arrays": [1], "maximum_worker_minutes": 120, "maximum_a100s": 6}
    with pytest.raises(ValueError, match="explicit user-authorized"):
        API["tick"](policy, {}, tmp_path / "state.json", False)
    policy["six_a100s_explicitly_authorized"] = True
    result = API["tick"](policy, {}, tmp_path / "state.json", False)
    assert result["occupancy_cap"] == result["admission_cap"] == 6


def test_known_broken_array_stays_held():
    rows = [row(1, i) for i in range(10)] + [row(2, i) for i in range(10)]
    selected = API["choose_releases"](rows, {"1", "2"}, 4, {"2"})
    assert len(selected) == 4
    assert all(item["array"] == "1" for item in selected)


def test_dependency_array_does_not_reserve_its_prerequisites_slots(tmp_path, monkeypatch):
    namespace = API["tick"].__globals__
    rows = [row(1, i) for i in range(4)] + [row(2, i) for i in range(2)]
    rows += [row(3, i) for i in range(20)]
    calls = []
    monkeypatch.setitem(namespace, "read_queue", lambda: rows)
    monkeypatch.setitem(namespace, "command", lambda *args: calls.append(args) or "")
    policy = dict(
        arrays=[1, 2, 3],
        adopt_existing_holds=[1, 2, 3],
        maximum_worker_minutes=120,
        maximum_a100s=6,
        six_a100s_explicitly_authorized=True,
        release_after_arrays_finish={"3": [1, 2]},
    )
    result = API["tick"](policy, {}, tmp_path / "state.json", True)
    assert len(result["released"]) == 6
    assert not any(job.startswith("3_") for job in result["released"])


def test_job_scope_and_finite_duration():
    assert API["task_expression"]("123", ["123_2", "123_1"]) == "123_[1,2]"
    with pytest.raises(ValueError, match="Not an element"):
        API["task_expression"]("123", ["999_1"])
    assert API["duration_minutes"]("2:00:00") == 120
    assert API["duration_minutes"]("2:00:01") == 121
    assert API["duration_minutes"]("1-00:00:00") == 1440
    assert API["gpu_count"]("N/A") == 0


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("30:00", 30),
        ("00:30", 1),
        ("59:59", 60),
        ("120:01", 121),
        ("0:00", 0),
        ("00:30:00", 30),
        ("2:00:01", 121),
        ("1-00:00:00", 1440),
        ("2-03:04:05", 3065),
    ],
)
def test_subhour_and_day_duration_formats_preserve_conservative_rounding(value, expected):
    assert API["duration_minutes"](value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "UNLIMITED",
        "NOT_SET",
        "N/A",
        "",
        "30",
        "-30:00",
        "30:60",
        "1:60:00",
        "1-24:00:00",
        "1-30:00",
        "30:00;touch",
        "30:00 ",
    ],
)
def test_nonfinite_ambiguous_and_malformed_durations_fail_closed(value):
    with pytest.raises(ValueError):
        API["duration_minutes"](value)


def test_v9_subhour_workers_are_released_without_changing_aggregate_limit(tmp_path, monkeypatch):
    namespace = API["tick"].__globals__
    rows = [dict(row(22033772, i), limit="30:00") for i in range(2)]
    calls = []
    monkeypatch.setitem(namespace, "read_queue", lambda: rows)
    monkeypatch.setitem(namespace, "command", lambda *args: calls.append(args) or "")
    policy = dict(
        arrays=[22033772],
        adopt_existing_holds=[22033772],
        maximum_worker_minutes=120,
        maximum_a100s=6,
        six_a100s_explicitly_authorized=True,
    )
    result = API["tick"](policy, {}, tmp_path / "state.json", True)
    assert result["released"] == ["22033772_0", "22033772_1"]
    assert result["admission_cap"] == 6
    assert ("scontrol", "release", "22033772_[0,1]") in calls
    assert not any("cancel" in call or "suspend" in call for call in calls)


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
