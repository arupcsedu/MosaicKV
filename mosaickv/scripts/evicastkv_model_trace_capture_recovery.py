#!/usr/bin/env python3
"""CPU-only correction of a captured-trace schema key; no inference retry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once
from mosaickv.evaluation.native_retirement_receipts import _sha, validate_trace_prefix

SOURCE = Path(__file__).resolve().parents[2]
PRODUCER = "a30d05d29ffef176151289aed55537b686b0be55"
PARENT = Path("/scratch/djy8hg/workdir/fox-native-prefix-capture/run-20261111-1")


def read(path):
    return json.loads(Path(path).read_text())


def compare_capture(capture):
    previous, current = capture["previous_trace"], capture["current_trace"]
    capacity = capture["trace_capacity"]
    if (
        capture.get("accepted") is not False
        or capture.get("online_guard_modified") is not False
        or type(capacity) is not int
        or capacity != 100_000
        or capture.get("previous_count") != len(previous)
        or capture.get("current_count") != len(current)
        or not previous
        or len(current) < len(previous)
        or len(current) >= capacity
        or capture.get("previous_trace_sha256") != _sha(previous)
        or capture.get("current_trace_sha256") != _sha(current)
    ):
        raise ValueError("complete intact actual model failure capture required")
    validate_trace_prefix([], current, capacity=capacity)
    changed = []
    deltas = []
    for index, (left, right) in enumerate(zip(previous, current, strict=False)):
        keys = sorted(
            key
            for key in set(left) | set(right)
            if key not in left or key not in right or left[key] != right[key]
        )
        if keys:
            changed.append(dict(index=index, changed_keys=keys))
            if keys == ["time_us"]:
                for event in (left, right):
                    if type(event.get("time_us")) is not int or event["time_us"] < 0:
                        raise ValueError("unchanged timestamp presence/type required")
                deltas.append(right["time_us"] - left["time_us"])
    only_clock = bool(changed) and all(row["changed_keys"] == ["time_us"] for row in changed)
    stable = all(
        {k: v for k, v in left.items() if k != "time_us"}
        == {k: v for k, v in right.items() if k != "time_us"}
        for left, right in zip(previous, current, strict=False)
    )
    return dict(
        actual_model_clock_only_drift_observed=only_clock and stable,
        previous_count=len(previous),
        current_count=len(current),
        capacity=capacity,
        changed_prefix_events=changed,
        changed_keys=sorted({key for row in changed for key in row["changed_keys"]}),
        timestamp_deltas_microseconds=deltas,
        original_full_dict_prefix_rejected=previous != current[: len(previous)],
        all_other_event_fields_identical=stable,
        allocation_order_or_generation_guards_relaxed=False,
        GPU_inference_repeated=False,
        whole_request_peak_certified=False,
    )


def recover(output):
    revision = clean_source(SOURCE)
    output = Path(output).resolve()
    allowed = Path("/scratch/djy8hg/workdir").resolve()
    if not output.is_relative_to(allowed) or output == allowed or output.exists():
        raise ValueError("new immutable derived output root required")
    plan = read(PARENT / "plan.json")
    result = read(PARENT / "workers/0/result.json")
    if (
        plan.get("source_sha") != PRODUCER
        or result.get("source_sha") != PRODUCER
        or result.get("plan_sha256") != sha(PARENT / "plan.json")
    ):
        raise ValueError("original consumed model capture producer changed")
    if result.get("actual_rows") != 0 or not result.get("failures"):
        raise ValueError("retain the original negative execution; do not reinterpret success")
    bindings = {
        str(PARENT / "plan.json"): sha(PARENT / "plan.json"),
        str(PARENT / "workers/0/result.json"): sha(PARENT / "workers/0/result.json"),
    }
    for name, expected in result["outputs"].items():
        path = (PARENT / "workers/0" / name).resolve(strict=True)
        if not path.is_relative_to((PARENT / "workers/0").resolve()) or sha(path) != expected:
            raise ValueError("captured model artifact changed")
        bindings[str(path)] = expected
    capture = read(PARENT / "workers/0/traces/native_retirement_failure.json")
    report = compare_capture(capture)
    report.update(
        complete=True,
        execution_source_sha=PRODUCER,
        analyzer_source_sha=revision,
        input_bindings=bindings,
        original_incomplete_analysis_preserved=True,
        schema_correction="trace_capacity (not capacity)",
    )
    write_once(output / "GATE_trace_clock_capture.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = recover(args.output)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "complete",
                    "actual_model_clock_only_drift_observed",
                    "changed_keys",
                    "timestamp_deltas_microseconds",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
