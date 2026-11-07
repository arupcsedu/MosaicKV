#!/usr/bin/env python3
"""Admission control for explicitly registered, one-A100 Slurm array workers.

Hold only pending tasks; never cancel, suspend, or alter running workers.
Released-but-pending tasks reserve slots too, so independent array throttles
cannot multiply the aggregate GPU limit. Uses no model or third-party packages.
"""

from __future__ import annotations

import argparse
import fcntl
import getpass
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ZONE = ZoneInfo("America/New_York")
ACTIVE = {"RUNNING", "COMPLETING", "CONFIGURING", "SUSPENDED"}


def limits(now: datetime, max_worker_minutes: int = 120) -> tuple[int, int]:
    """Four A100s at all times; legacy overnight expansion is no longer permitted."""
    if now.tzinfo is None:
        raise ValueError("Timezone-aware time required")
    if max_worker_minutes < 1:
        raise ValueError("maximum worker duration must be positive")
    return 4, 4


def duration_minutes(value: str) -> int:
    """Parse finite squeue durations, rounding partial minutes upward.

    squeue displays sub-hour limits as MM:SS, even when sbatch received
    HH:MM:SS. Reject unlimited/malformed values before releasing any worker.
    """
    match = re.fullmatch(r"(?:(\d+)-)?(\d+):([0-5]\d)(?::([0-5]\d))?", value)
    if match is None or (match[1] is not None and match[4] is None):
        raise ValueError(f"Expected finite Slurm [D-]HH:MM:SS or MM:SS time, got {value!r}")
    days = int(match[1] or 0)
    if match[4] is None:
        hours, minutes, seconds = 0, int(match[2]), int(match[3])
    else:
        hours, minutes, seconds = int(match[2]), int(match[3]), int(match[4])
    return days * 1440 + hours * 60 + minutes + (seconds > 0)


def gpu_count(gres: str) -> int:
    matches = re.findall(r"(?:^|,)gres/gpu:a100:(\d+)(?:$|,)", gres)
    return sum(map(int, matches))


def command(*args: str) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True, timeout=30).stdout


def read_queue() -> list[dict[str, str]]:
    output = command(
        "squeue",
        "--user",
        getpass.getuser(),
        "--array",
        "--noheader",
        "--format",
        "%F|%i|%j|%T|%b|%l|%R",
    )
    keys = ("array", "id", "name", "state", "gres", "limit", "reason")
    return [dict(zip(keys, line.split("|", 6), strict=True)) for line in output.splitlines()]


def set_array_throttle(array: str) -> None:
    """Slurm can update live elements yet return an error for completed siblings."""
    try:
        command("scontrol", "update", f"JobId={array}", "ArrayTaskThrottle=4")
    except subprocess.CalledProcessError as exc:
        errors = (exc.stderr or "").strip().splitlines()
        if not errors or any("Job has already finished" not in line for line in errors):
            raise
        records = command("scontrol", "show", "job", "--oneliner", array).splitlines()
        live = [
            line
            for line in records
            if re.search(r"JobState=(?:PENDING|RUNNING|COMPLETING|CONFIGURING|SUSPENDED)\b", line)
        ]
        if not live or any(not re.search(r"\bArrayTaskThrottle=4\b", line) for line in live):
            raise RuntimeError(f"Array {array} throttle update could not be verified") from exc


def task_expression(array: str, ids: list[str]) -> str:
    indices = []
    for job_id in ids:
        if not re.fullmatch(rf"{re.escape(array)}_\d+", job_id):
            raise ValueError(f"Not an element of array {array}: {job_id}")
        indices.append(int(job_id.split("_")[1]))
    if not indices:
        raise ValueError("Empty job list")
    return f"{array}_[{','.join(map(str, sorted(set(indices))))}]"


def choose_releases(
    rows: list[dict[str, str]],
    arrays: set[str],
    admission: int,
    blocked: set[str] | None = None,
) -> list[dict[str, str]]:
    """Reserve slots for all own active A100 jobs and all already released targets."""
    if not 0 <= admission <= 4:
        raise ValueError("aggregate A100 admission cannot exceed four")
    occupied = sum(gpu_count(row["gres"]) for row in rows if row["state"] in ACTIVE)
    released = [
        row
        for row in rows
        if row["array"] in arrays
        and row["state"] == "PENDING"
        and "JobHeldUser" not in row["reason"]
    ]
    slots = max(0, admission - occupied - len(released))
    counts = {
        array: sum(
            row["array"] == array and (row["state"] in ACTIVE or row in released) for row in rows
        )
        for array in arrays
    }
    blocked = blocked or set()
    candidates = [
        row
        for row in rows
        if row["array"] in arrays
        and row["array"] not in blocked
        and row["state"] == "PENDING"
        and "JobHeldUser" in row["reason"]
    ]
    selected = []
    for _ in range(min(slots, len(candidates))):
        row = min(
            candidates,
            key=lambda item: (
                counts[item["array"]],
                int(item["array"]),
                int(item["id"].split("_")[1]),
            ),
        )
        candidates.remove(row)
        counts[row["array"]] += 1
        selected.append(row)
    return selected


def tick(policy: dict, state: dict, state_path: Path, apply: bool) -> dict:
    arrays = {str(value) for value in policy["arrays"]}
    blocked = {str(value) for value in policy.get("blocked_arrays", {})}
    now = datetime.now(ZONE)
    occupancy, admission = limits(now, policy["maximum_worker_minutes"])
    rows = read_queue()
    managed = [row for row in rows if row["array"] in arrays]
    for row in managed:
        if gpu_count(row["gres"]) != 1 or not row["name"].startswith("mk-"):
            raise ValueError(f"Unrecognized one-A100 MosaicKV worker: {row}")
        if duration_minutes(row["limit"]) > policy["maximum_worker_minutes"]:
            raise ValueError(f"Worker exceeds preregistered drain horizon: {row['id']}")
        if row["state"] == "PENDING" and "JobHeldAdmin" in row["reason"]:
            raise ValueError("Administrator-held workers require manual review")
    if not apply:
        return {
            "time": now.isoformat(),
            "occupancy_cap": occupancy,
            "admission_cap": admission,
            "managed_tasks": len(managed),
            "dry_run": True,
        }

    owned = set(state.get("owned", []))
    initialized = set(state.get("initialized", []))
    hold_intents = set(state.get("hold_intents", []))
    policy_owned = {str(value) for value in policy.get("adopt_existing_holds", [])}

    def save() -> None:
        state.update(
            owned=sorted(owned), initialized=sorted(initialized), hold_intents=sorted(hold_intents)
        )
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(state_path)

    new_arrays = {row["array"] for row in managed} - initialized
    # Never increase an array throttle until pending elements have been held.
    for array in sorted(new_arrays):
        for row in managed:
            if (
                row["array"] == array
                and "JobHeldUser" in row["reason"]
                and row["id"] not in owned
                and array not in policy_owned
            ):
                raise ValueError(f"Refusing to adopt an unrelated user hold: {row['id']}")
    # New tasks, requeued tasks, and boundary changes enter a safe pending-only barrier.
    boundary_changed = state.get("admission_cap") != admission
    for array in sorted(arrays):
        pending = [
            row
            for row in managed
            if row["array"] == array
            and row["state"] == "PENDING"
            and (
                row["id"] not in owned
                or row["id"] in hold_intents
                or boundary_changed
                or (array in blocked and "JobHeldUser" not in row["reason"])
            )
        ]
        if pending:
            ids = [row["id"] for row in pending]
            owned.update(ids)
            hold_intents.update(ids)
            save()  # Recovery knows ownership even if the RPC/response fails.
            command("scontrol", "hold", task_expression(array, ids))
            hold_intents.difference_update(ids)
            save()
    for array in sorted(new_arrays):
        set_array_throttle(array)
        initialized.add(array)
    rows = read_queue()  # After the barrier: a raced start is counted as occupied.
    released = choose_releases(rows, arrays, admission, blocked)
    if any(row["id"] not in owned for row in released):
        raise ValueError("Refusing to release a hold not owned by this controller")
    for array in sorted(arrays):
        ids = [row["id"] for row in released if row["array"] == array]
        if ids:
            command("scontrol", "release", task_expression(array, ids))
    state["admission_cap"] = admission
    save()
    return {
        "time": now.isoformat(),
        "occupancy_cap": occupancy,
        "admission_cap": admission,
        "managed_tasks": len(managed),
        "own_active_a100s": sum(gpu_count(row["gres"]) for row in rows if row["state"] in ACTIVE),
        "released": [row["id"] for row in released],
        "blocked_arrays": sorted(blocked),
        "dry_run": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    policy = json.loads(args.policy.read_text())
    required = {
        "timezone": "America/New_York",
        "maximum_worker_minutes": 120,
        "maximum_a100s": 4,
        "window": "all_times",
    }
    if any(policy.get(key) != value for key, value in required.items()):
        raise ValueError("This controller requires the four-A100 / two-hour worker policy")
    args.state.parent.mkdir(parents=True, exist_ok=True)
    with args.state.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads(args.state.read_text()) if args.state.exists() else {}
        while True:
            try:
                result = tick(policy, state, args.state, args.apply)
                print(json.dumps(result), flush=True)
                if not args.watch or result["managed_tasks"] == 0:
                    return
            except Exception as exc:
                print(
                    json.dumps({"time": datetime.now(ZONE).isoformat(), "error": str(exc)}),
                    flush=True,
                )
                if not args.watch:
                    raise
                # Do not release additional workers on an invalid queue/control response.
            time.sleep(30)


if __name__ == "__main__":
    main()
