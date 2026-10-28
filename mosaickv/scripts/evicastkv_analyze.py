"""Finite completion analysis; refuses incomplete cohorts and auto-promotes only gates."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from evicastkv_campaign import load_lock, submit

from mosaickv.evaluation.fox_campaign import digest, gate_payload, sha, write_once


def collect(root, stage):
    spec = json.loads((root / "jobs" / f"{stage}.json").read_text())
    workers = []
    rows = []
    errors = []
    for index, _job in enumerate(spec["jobs"]):
        paths = list((root / "outputs" / stage / f"{index:03d}").glob("*/worker.json"))
        if len(paths) != 1:
            errors.append(
                dict(
                    index=index,
                    error="missing or duplicate worker attempts",
                    paths=[str(p) for p in paths],
                )
            )
            continue
        worker = json.loads(paths[0].read_text())
        workers.append(worker)
        if worker["status"] != "passed":
            errors.append(dict(index=index, errors=worker["errors"]))
        for binding in worker["records"]:
            path = Path(binding["path"])
            if sha(path) != binding["sha256"]:
                raise ValueError("raw record hash mismatch")
            row = json.loads(path.read_text())
            if row["status"] != "completed":
                errors.append(dict(index=index, sample_id=row["sample_id"], error=row.get("error")))
            elif sha(Path(row["trace_path"])) != row["trace_sha256"]:
                raise ValueError("trace hash mismatch")
            rows.append(row)
    keys = [
        (
            r["workload"],
            r["sample_id"],
            r["method"],
            r.get("variant"),
            r["budget_bytes"],
            r["repeat"],
        )
        for r in rows
    ]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate experimental keys")
    return spec, workers, rows, errors


def pairing(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["status"] == "completed" and row["budget_bytes"] is not None:
            groups[(row["workload"], row["sample_id"], row["budget_bytes"], row["repeat"])].append(
                row
            )
    results = []
    for key, values in groups.items():
        controls = [
            tuple(r[k] for k in ("input_sha", "processor_sha", "scorer_sha", "forward_steps"))
            for r in values
        ]
        if len(set(controls)) != 1:
            raise ValueError(f"paired controls differ: {key}")
        for field in ("initial_kv_bytes", "final_kv_bytes", "peak_kv_bytes"):
            if len({r[field] for r in values}) != 1:
                raise ValueError(f"physical storage mismatch: {key}/{field}")
        results.append(
            dict(
                workload=key[0],
                sample_id=key[1],
                cap=key[2],
                repeat=key[3],
                methods=[r["method"] for r in values],
                initial_bytes=values[0]["initial_kv_bytes"],
                final_bytes=values[0]["final_kv_bytes"],
                peak_bytes=values[0]["peak_kv_bytes"],
            )
        )
    return results


def export_descriptive(root, stage, rows):
    output = root / "analysis" / stage
    output.mkdir(parents=True, exist_ok=True)
    completed = [r for r in rows if r["status"] == "completed"]
    groups = defaultdict(list)
    for row in completed:
        groups[
            (
                row["workload"],
                row["subtask"],
                row["method"],
                row.get("variant"),
                row["budget_bytes"],
            )
        ].append(row)
    summaries = []
    for key, values in sorted(groups.items(), key=str):
        timings = [v["e2e_wall_seconds"] for v in values]
        scores = [v["score"] for v in values if v["score"] is not None]
        summary = dict(
            workload=key[0],
            subtask=key[1],
            method=key[2],
            variant=key[3],
            budget_bytes=key[4],
            unique_inputs=len({v["sample_id"] for v in values}),
            observations=len(values),
            mean_e2e=statistics.mean(timings),
            p50_e2e=statistics.median(timings),
            p95_e2e=sorted(timings)[min(len(timings) - 1, math.ceil(0.95 * len(timings)) - 1)],
            p95_qualification=f"empirical_p95_n_observations={len(timings)}",
            mean_initial_kv_bytes=statistics.mean(v["initial_kv_bytes"] for v in values),
            mean_final_kv_bytes=statistics.mean(v["final_kv_bytes"] for v in values),
            mean_peak_kv_bytes=statistics.mean(v["peak_kv_bytes"] for v in values),
            mean_task_score=statistics.mean(scores) if scores else None,
        )
        summaries.append(summary)
    write_once(output / "descriptive.json", summaries)
    if summaries:
        with (output / "descriptive.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
    import pandas as pd

    # Nested dictionaries use JSON strings to avoid ambiguous Arrow inference.
    flat = [
        {
            k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
            for k, v in row.items()
        }
        for row in rows
    ]
    pd.DataFrame(flat).to_parquet(output / "observations.parquet", index=False)
    pd.DataFrame(summaries).to_parquet(output / "aggregate.parquet", index=False)
    with (output / "predictions.jsonl").open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    tables = output / "tables"
    tables.mkdir(exist_ok=True)
    with (tables / "development.tex").open("x") as stream:
        stream.write(
            "% DEVELOPMENT ONLY. CIDEr is computed separately at corpus level.\n"
            "\\begin{tabular}{llllrrr}\n"
            "Workload & Task & Method & Cap (MiB) & N & E2E (s) & KV (MiB) " + r"\\" + "\n"
        )
        for r in summaries:
            method = r["method"].replace("_", r"\_")
            cap = "Full" if r["budget_bytes"] is None else f"{r['budget_bytes'] / 2**20:g}"
            stream.write(
                f"{r['workload']} & {r['subtask'] or 'COCO'} & {method} & {cap} & "
                f"{r['unique_inputs']} & {r['mean_e2e']:.3f} & "
                f"{r['mean_initial_kv_bytes'] / 2**20:.3f} " + r"\\" + "\n"
            )
        stream.write("\\end{tabular}\n")
    write_once(
        output / "analysis_manifest.json",
        dict(
            measurement_type="measured_development",
            status="descriptive_not_confirmatory",
            input_rows_sha256=digest(rows),
            raw_count=len(rows),
            completed_count=len(completed),
            source_sha=rows[0]["source_sha"] if rows else None,
            bootstrap_status="not_run_until_complete_scoring",
            historical_outputs_reused=False,
        ),
    )
    return summaries


def analyze(root: Path, stage: str):
    lock = load_lock(root)
    spec, workers, rows, errors = collect(root, stage)
    output = root / "analysis" / stage
    output.mkdir(parents=True, exist_ok=True)
    paired = []
    try:
        paired = pairing(rows)
    except Exception as error:
        errors.append(dict(error=f"{type(error).__name__}: {error}"))
    expected = 0
    for job in spec["jobs"]:
        arms = 2 if stage in ("smoke", "correctness") else 18 if stage == "mechanisms" else 7
        repeats = lock["repeats"] if stage == "measurement" else 1
        expected += len(job["indices"]) * arms * repeats
    if len(rows) != expected:
        errors.append(dict(error="incomplete raw denominator", expected=expected, actual=len(rows)))
    write_once(output / "pairing.json", paired)
    write_once(output / "failures.json", errors)
    if rows:
        export_descriptive(root, stage, rows)
    gate = gate_payload(
        stage,
        source_sha=lock["source_sha"],
        protocol_sha=sha(root / "protocol_lock.json"),
        passed=not errors,
        tests=dict(
            expected=expected,
            actual=len(rows),
            completed=sum(r["status"] == "completed" for r in rows),
            workers_expected=len(spec["jobs"]),
            workers_completed=len(workers),
            pairing_count=len(paired),
        ),
        blockers=errors,
    )
    write_once(output / f"GATE_{stage}.json", gate)
    # Positive execution is not an instruction to manufacture a scientific claim.
    successor = {
        "smoke": "correctness",
        "correctness": "scoring",
        "scoring": "profiling",
        "profiling": "measurement",
    }.get(stage)
    if gate["status"] == "passed" and successor:
        submit(root, successor)
        action = dict(action="submitted_registered_successor", stage=successor)
    elif gate["status"] == "passed" and stage == "measurement":
        action = dict(
            action="stop_for_measured_hotspot_review",
            reason=(
                "No optimization is authorized without actual hotspot evidence; "
                "fresh confirmation is independently blocked"
            ),
        )
    else:
        action = dict(
            action="stop_failed_or_incomplete_gate", reason="No dependent GPU work submitted"
        )
    write_once(output / "completion_decision.json", action)
    print(json.dumps(dict(gate=gate, action=action), sort_keys=True))
