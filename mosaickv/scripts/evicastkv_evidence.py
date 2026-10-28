#!/usr/bin/env python3
"""Derived, CPU-only evidence from immutable Fox campaign observations."""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path

from mosaickv.evaluation.fox_campaign import TASKS, clean_source, digest, sha, write_once
from mosaickv.evaluation.fox_statistics import corpus_cider_statistics, paired_scalar_statistics
from mosaickv.evaluation.metrics.types import MetricSample

SOURCE = Path(__file__).resolve().parents[2]
MATERIALIZATION = Path(
    "/scratch/djy8hg/runs/mosaickv/milebench-representative-89ecd50-20261018/materialization.json"
)


def group_manifest(root):
    """Recover underlying frames/images, not just the combined-image montage."""
    lock = json.loads((root / "protocol_lock.json").read_text())
    source_meta = (
        json.loads(MATERIALIZATION.read_text()) if "milebench" in lock["cohorts"] else None
    )
    maps = {}
    bindings = {str(MATERIALIZATION): sha(MATERIALIZATION)} if source_meta else {}
    for workload in lock["cohorts"]:
        path = Path(lock["cohorts"][workload]["development"]["path"])
        if sha(path) != lock["cohorts"][workload]["development"]["sha256"]:
            raise ValueError("cohort changed")
        samples = json.loads(path.read_text())["samples"]
        maps[workload] = {}
        annotations = {}
        if workload == "milebench":
            for task in TASKS:
                dependency = source_meta["dependencies"][task]
                annotation = Path(dependency["annotation"])
                if sha(annotation) != dependency["annotation_sha256"]:
                    raise ValueError("official MileBench annotation changed")
                bindings[str(annotation)] = sha(annotation)
                annotations[task] = {
                    f"{task}-{s['sample_id']:04d}": s["task_instance"]
                    for s in json.loads(annotation.read_text())["data"]
                }
        for sample in samples:
            groups = {"input-media:" + m["sha256"] for m in sample["media"]}
            if workload == "milebench":
                task = sample["milebench_dataset"]
                instance = annotations[task][sample["sample_id"]]
                image_root = Path(source_meta["dependencies"][task]["annotation"]).parent / "images"
                for relative in instance["images_path"]:
                    image = (image_root / relative).resolve(strict=True)
                    image.relative_to(image_root.resolve())
                    image_sha = sha(image)
                    bindings[str(image)] = image_sha
                    groups.add("original-image:" + image_sha)
                    if task in ("ActionSequence", "ActionPrediction"):
                        parts = Path(relative).parts
                        if len(parts) < 2:
                            raise ValueError("missing video/clip grouping metadata")
                        groups.add("milebench-video:" + parts[0])
            maps[workload][sample["sample_id"]] = sorted(groups)
    return dict(
        schema_version=1,
        groups=maps,
        bound_files=bindings,
        construction="input-only original images and video clip IDs; no responses or scores used",
    )


def exclusion_inventory(root, output, max_seconds=90, *, run_root=None):
    """Bounded canonical-manifest inventory. Does not parse scheduler/raw logs."""
    started = time.monotonic()
    run_root = Path(run_root or "/scratch/djy8hg/runs/mosaickv")
    queue = []
    unclassified = []
    for entry in os.scandir(run_root):
        if not entry.is_dir(follow_symlinks=False):
            continue
        found = []
        for name in (
            "plan.json",
            "protocol.json",
            "materialization.json",
            "ledger.json",
            "registration.json",
        ):
            path = Path(entry.path) / name
            if path.is_file():
                found.append(path)
        queue.extend(found)
        if not found and ("coco" in entry.name.casefold() or "milebench" in entry.name.casefold()):
            unclassified.append(entry.path)
    seen = {}
    used_coco = set()
    used_mile = set()
    errors = []
    pattern = re.compile(r"(" + "|".join(map(re.escape, TASKS)) + r")[-:]0*(\d+)$")

    def identity(value):
        if not isinstance(value, str):
            return
        match = re.search(r"COCO_val2014_\d+\.jpg", value)
        if match:
            used_coco.add(match.group())
        match = pattern.search(value)
        if match:
            used_mile.add(f"{match[1]}-{int(match[2]):04d}")

    def visit(value, base):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in (
                    "references",
                    "reference",
                    "answers",
                    "answer",
                    "predictions",
                    "scores",
                    "score",
                    "prompt",
                    "choices",
                ):
                    continue
                if key in ("sample_id", "sample_ids", "excluded_ids", "selected_ids"):
                    if isinstance(item, list):
                        for sid in item:
                            identity(sid)
                    else:
                        identity(item)
                if (
                    key in ("ledger", "ledger_path")
                    and isinstance(item, str)
                    and item.endswith(".json")
                ):
                    # Relative links belong to the containing manifest, never
                    # the process CWD. Resolve strictly in the queue reader and
                    # retain missing/out-of-scratch links as errors, not guesses.
                    queue.append(base / Path(item))
                if key == "bound_files" and isinstance(item, dict):
                    queue.extend(
                        base / Path(p)
                        for p in item
                        if str(p).endswith(".json")
                        and ("/ledgers/" in str(p) or Path(p).name == "ledger.json")
                    )
                visit(item, base)
        elif isinstance(value, list):
            for item in value:
                visit(item, base)

    while queue and time.monotonic() - started < max_seconds:
        path = queue.pop()
        try:
            path = path.resolve(strict=True)
            path.relative_to(Path("/scratch/djy8hg").resolve())
            if str(path) in seen:
                continue
            if path.stat().st_size > 32 * 2**20:
                raise ValueError("canonical manifest exceeds bounded inventory size")
            value = json.loads(path.read_text())
            seen[str(path)] = sha(path)
            visit(value, path.parent)
        except Exception as error:
            errors.append(dict(path=str(path), error=f"{type(error).__name__}: {error}"))
    for workload in json.loads((root / "protocol_lock.json").read_text())["cohorts"]:
        p = Path(
            json.loads((root / "protocol_lock.json").read_text())["cohorts"][workload][
                "development"
            ]["path"]
        )
        visit(json.loads(p.read_text()), p.parent)
    per_task = {task: len({s for s in used_mile if s.startswith(task + "-")}) for task in TASKS}
    report = dict(
        status="freshness_unverified",
        complete_scan=False,
        canonical_files=len(seen),
        bound_files=seen,
        COCO_consumed_ids=sorted(used_coco),
        MileBench_consumed_ids=sorted(used_mile),
        MileBench_known_used_per_task=per_task,
        unclassified_workload_roots=unclassified,
        errors=errors,
        remaining_manifest_queue=len(queue),
        wall_seconds=time.monotonic() - started,
        confirmation_authorized=False,
        reason=(
            "Bounded canonical inventory is not proof of exhaustive historical ID/media "
            "coverage. No confirmation samples materialized or scored."
        ),
    )
    write_once(output, report)
    return report


def validated_rows(root, stage):
    lock = json.loads((root / "protocol_lock.json").read_text())
    clean_source(Path(lock["source_root"]), lock["source_sha"])
    gate = root / "analysis" / stage / f"GATE_{stage}.json"
    if json.loads(gate.read_text())["status"] != "passed":
        raise ValueError("incomplete stage cannot support full-cohort quality statistics")
    rows = []
    for worker_path in sorted((root / "outputs" / stage).glob("*/*/worker.json")):
        worker = json.loads(worker_path.read_text())
        for binding in worker["records"]:
            path = Path(binding["path"])
            if sha(path) != binding["sha256"]:
                raise ValueError("prediction changed")
            row = json.loads(path.read_text())
            if row["status"] != "completed":
                raise ValueError("failed row in passed cohort")
            if sha(Path(row["trace_path"])) != row["trace_sha256"]:
                raise ValueError("trace changed")
            rows.append(row)
    return lock, rows


def native_statistics(root, stage, output, workers):
    clean_source(SOURCE)
    lock, rows = validated_rows(root, stage)
    groups = group_manifest(root)
    write_once(output / "group_manifest.json", groups)
    # Repetitions are not independent quality observations.
    quality = {}
    for row in rows:
        key = (row["workload"], row["sample_id"], row["method"], row["budget_bytes"])
        if key in quality and quality[key]["token_ids"] != row["token_ids"]:
            raise ValueError("generation nondeterminism across repeats")
        quality[key] = row
    reports = []
    for workload in lock["cohorts"]:
        chosen = [r for r in quality.values() if r["workload"] == workload]
        point_rows = defaultdict(dict)
        for row in chosen:
            key = f"{row['method']}:{row['budget_bytes'] or 'full'}"
            point_rows[key][row["sample_id"]] = row
        expected = set(next(iter(point_rows.values())))
        if any(set(values) != expected for values in point_rows.values()):
            raise ValueError("cannot take a favorable surviving intersection")
        group_map = {sid: groups["groups"][workload][sid] for sid in expected}
        caps = lock["budgets_mib"][workload]
        comparisons = [
            (f"prompt_attention_topk:{int(cap * 2**20)}", f"mosaickv_exact:{int(cap * 2**20)}")
            for cap in caps
        ]
        port = "vl_cache_reimpl" if workload == "coco" else "lookm_reimpl"
        comparisons += [
            (f"{port}:{int(cap * 2**20)}", f"mosaickv_exact:{int(cap * 2**20)}") for cap in caps
        ]
        if workload == "coco":
            points = {
                key: tuple(
                    MetricSample(sid, row["answer"], tuple(row["references"]))
                    for sid, row in sorted(values.items())
                )
                for key, values in point_rows.items()
            }
            report = corpus_cider_statistics(
                points, comparisons, group_map, resamples=10000, seed=lock["seed"], workers=workers
            )
            reports.append(dict(workload=workload, **report))
        else:
            for family, tasks in (("accuracy", TASKS[:2]), ("rouge_l_f1", TASKS[2:])):
                ids = {
                    sid
                    for sid in expected
                    if next(iter(point_rows.values()))[sid]["subtask"] in tasks
                }
                task_map = {sid: next(iter(point_rows.values()))[sid]["subtask"] for sid in ids}
                values = {
                    key: {sid: point[sid]["score"] for sid in ids}
                    for key, point in point_rows.items()
                }
                contrasts = []
                for reference, candidate in comparisons:
                    contrasts.append(
                        dict(
                            reference=reference,
                            candidate=candidate,
                            **paired_scalar_statistics(
                                values[reference],
                                values[candidate],
                                {sid: group_map[sid] for sid in ids},
                                task_map,
                                resamples=10000,
                                seed=lock["seed"],
                                hierarchy=True,
                            ),
                        )
                    )
                scores = {
                    key: statistics.mean(
                        statistics.mean(v[sid] for sid in ids if task_map[sid] == task)
                        for task in tasks
                    )
                    for key, v in values.items()
                }
                reports.append(
                    dict(
                        workload=workload,
                        metric=family,
                        scores=scores,
                        contrasts=contrasts,
                        unique_inputs=len(ids),
                    )
                )
    for report in reports:
        for contrast in report["contrasts"]:
            contrast["statistical_family"] = (
                "primary_quality_six"
                if contrast["reference"].startswith("prompt_attention_topk:")
                else "secondary_local_ports_six"
            )
    write_once(output / "native_metrics.json", reports)
    write_once(
        output / "analysis_manifest.json",
        dict(
            runtime_source_sha=lock["source_sha"],
            analysis_source_sha=clean_source(SOURCE),
            protocol_sha256=sha(root / "protocol_lock.json"),
            raw_rows_sha256=digest(rows),
            development_only=True,
            resamples=10000,
            primary_contrasts=6,
            ordinary_confidence=0.95,
            simultaneous_confidence=1 - 0.05 / 6,
            source_scores_overwritten=False,
        ),
    )
    return reports


def mechanism_statistics(root, output, workers=4):
    """Sixty prespecified secondary contrasts; seeds never become extra inputs."""
    lock, rows = validated_rows(root, "mechanisms")
    groups = group_manifest(root)
    reports = []
    treatments = (
        "routed-prompt",
        "coverage-off",
        "value-off",
        "cross-modal-off",
        "route-0",
        "route-0.00390625",
        "route-0.25",
        "random-13",
        "random-29",
        "random-47",
    )
    for workload in lock["cohorts"]:
        selected = [r for r in rows if r["workload"] == workload]
        point_rows = defaultdict(dict)
        for row in selected:
            point_rows[f"{row['variant']}:{row['budget_bytes']}"][row["sample_id"]] = row
        expected = set(lock["cohorts"][workload]["development"]["sample_ids"])
        if any(set(v) != expected for v in point_rows.values()):
            raise ValueError("incomplete controlled mechanism cohort")
        comparisons = [
            (f"{variant}:{cap * 2**20}", f"mechanism-complete:{cap * 2**20}")
            for cap in lock["budgets_mib"][workload]
            for variant in treatments
        ]
        group_map = {sid: groups["groups"][workload][sid] for sid in expected}
        if workload == "coco":
            points = {
                key: tuple(
                    MetricSample(sid, row["answer"], tuple(row["references"]))
                    for sid, row in sorted(values.items())
                )
                for key, values in point_rows.items()
            }
            report = corpus_cider_statistics(
                points,
                comparisons,
                group_map,
                resamples=10000,
                seed=lock["seed"],
                workers=workers,
                family_size=60,
            )
            reports.append(
                dict(workload=workload, statistical_family="secondary_mechanisms_60", **report)
            )
        else:
            for metric, tasks in (("accuracy", TASKS[:2]), ("rouge_l_f1", TASKS[2:])):
                task_map = {r["sample_id"]: r["subtask"] for r in selected if r["subtask"] in tasks}
                values = {
                    key: {sid: row["score"] for sid, row in point.items() if sid in task_map}
                    for key, point in point_rows.items()
                }
                contrasts = [
                    dict(
                        reference=reference,
                        candidate=candidate,
                        **paired_scalar_statistics(
                            values[reference],
                            values[candidate],
                            {sid: group_map[sid] for sid in task_map},
                            task_map,
                            resamples=10000,
                            seed=lock["seed"],
                            hierarchy=True,
                            family_size=60,
                        ),
                    )
                    for reference, candidate in comparisons
                ]
                scores = {
                    key: statistics.mean(
                        statistics.mean(v[sid] for sid in task_map if task_map[sid] == task)
                        for task in tasks
                    )
                    for key, v in values.items()
                }
                reports.append(
                    dict(
                        workload=workload,
                        metric=metric,
                        scores=scores,
                        contrasts=contrasts,
                        statistical_family="secondary_mechanisms_60",
                        unique_inputs=len(task_map),
                    )
                )
    write_once(output / "mechanism_metrics.json", reports)
    write_once(
        output / "analysis_manifest.json",
        dict(
            source_sha=clean_source(SOURCE),
            runtime_source_sha=lock["source_sha"],
            raw_rows_sha256=digest(rows),
            resamples=10000,
            secondary_family=60,
            seeds_reported_separately=True,
            development_only=True,
        ),
    )
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inventory", "statistics"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", default="scoring")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.mode == "inventory":
        report = exclusion_inventory(args.root, args.output)
        print(
            json.dumps(
                {
                    k: v
                    for k, v in report.items()
                    if k
                    in (
                        "status",
                        "canonical_files",
                        "wall_seconds",
                        "MileBench_known_used_per_task",
                        "confirmation_authorized",
                    )
                }
            )
        )
    else:
        reports = native_statistics(args.root, args.stage, args.output, args.workers)
        print(json.dumps(reports, sort_keys=True))


if __name__ == "__main__":
    main()
