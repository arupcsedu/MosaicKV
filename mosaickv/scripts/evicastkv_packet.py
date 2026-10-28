#!/usr/bin/env python3
"""Read-only, write-once reporting and finite event-driven completion watch.

No model inference, no raw mutation, no threshold changes, no inference from
missing data. A watch schedules one successor callback, never polls a queue.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

from evicastkv_campaign import PYTHON, SOURCE
from evicastkv_evidence import group_manifest, validated_rows

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once
from mosaickv.evaluation.fox_statistics import paired_scalar_statistics


def read(path):
    return json.loads(Path(path).read_text())


def text_once(path, value):
    """Report source is generated, never edit or replace an existing report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(value)


def csv_once(path, rows):
    if not rows:
        return
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stages(root):
    return {p.parent.name: read(p) for p in sorted((root / "analysis").glob("*/GATE_*.json"))}


def all_records(root):
    rows, errors, bindings = [], [], {}
    for worker_path in sorted((root / "outputs").glob("*/*/*/worker.json")):
        worker = read(worker_path)
        bindings[str(worker_path)] = sha(worker_path)
        for item in worker["records"]:
            path = Path(item["path"])
            if sha(path) != item["sha256"]:
                raise ValueError("raw record changed")
            row = read(path)
            if (
                row.get("manifest_sha256")
                and sha(Path(row["manifest_path"])) != row["manifest_sha256"]
            ):
                raise ValueError("run manifest changed")
            if row["status"] == "completed":
                if sha(Path(row["trace_path"])) != row["trace_sha256"]:
                    raise ValueError("runtime trace changed")
            else:
                errors.append(dict(stage=worker_path.parts[-4], **row))
            rows.append(dict(stage=worker_path.parts[-4], **row))
            bindings[str(path)] = item["sha256"]
    return rows, errors, bindings


def timing_statistics(root, output):
    """Average repeats per input before paired clustered inference."""
    lock, rows = validated_rows(root, "measurement")
    groups = group_manifest(root)
    reports = []
    phase_rows = []
    for workload in ("coco", "milebench"):
        subset = [r for r in rows if r["workload"] == workload]
        observations = defaultdict(list)
        for row in subset:
            if row["timings"].get("profiler_contaminated"):
                raise ValueError("profiled timing cannot enter unprofiled results")
            observations[(row["sample_id"], row["method"], row["budget_bytes"])].append(row)
        expected_repeats = lock["repeats"]
        if any(len(v) != expected_repeats for v in observations.values()):
            raise ValueError("missing repeat denominator")
        all_ids = {r["sample_id"] for r in subset}
        tasks = {r["sample_id"]: r["subtask"] or "COCO" for r in subset}
        for cap in lock["budgets_mib"][workload]:
            cap = int(cap * 2**20)
            for baseline in lock["methods"][workload]:
                if baseline in ("full_kv", "mosaickv_exact"):
                    continue
                reference, candidate = {}, {}
                for sid in sorted(all_ids):
                    reference[sid] = statistics.mean(
                        r["e2e_wall_seconds"] for r in observations[(sid, baseline, cap)]
                    )
                    candidate[sid] = statistics.mean(
                        r["e2e_wall_seconds"] for r in observations[(sid, "mosaickv_exact", cap)]
                    )
                result = paired_scalar_statistics(
                    reference,
                    candidate,
                    {sid: groups["groups"][workload][sid] for sid in all_ids},
                    tasks,
                    resamples=lock["bootstrap_resamples"],
                    seed=lock["seed"],
                    family_size=8,
                )
                reports.append(
                    dict(
                        workload=workload,
                        baseline=baseline,
                        budget_bytes=cap,
                        statistical_family="secondary_timing_eight_comparisons",
                        **result,
                    )
                )
        for (sid, method, cap), values in sorted(observations.items(), key=str):
            phase_rows.append(
                dict(
                    workload=workload,
                    sample_id=sid,
                    method=method,
                    budget_bytes=cap,
                    repeats=len(values),
                    mean_e2e=statistics.mean(r["e2e_wall_seconds"] for r in values),
                    repeat_stddev=statistics.stdev(r["e2e_wall_seconds"] for r in values),
                    mean_compression=statistics.mean(
                        r["metrics"]["compression_time"] for r in values
                    )
                    if "compression_time" in values[0]["metrics"]
                    else None,
                    mean_first_emitted=statistics.mean(
                        r["timings"]["first_token_emitted_seconds"] for r in values
                    ),
                    mean_second_emitted=statistics.mean(
                        r["timings"]["second_token_emitted_seconds"] for r in values
                    ),
                    mean_second_gap=statistics.mean(
                        r["timings"]["second_token_emitted_seconds"]
                        - r["timings"]["first_token_emitted_seconds"]
                        for r in values
                    ),
                    mean_final_kv=statistics.mean(r["final_kv_bytes"] for r in values),
                    mean_peak_allocated=statistics.mean(r["peak_allocated_bytes"] for r in values),
                    mean_peak_reserved=statistics.mean(r["peak_reserved_bytes"] for r in values),
                )
            )
    write_once(output / "paired_latency.json", reports)
    csv_once(output / "input_timing.csv", phase_rows)
    return reports


def profile_summary(root, output):
    grouped = defaultdict(list)
    bindings = {}
    for path in sorted((root / "outputs/profiling").glob("*/*/profiler/*/operators.json")):
        bindings[str(path)] = sha(path)
        for item in read(path):
            if item["name"].startswith("evicastkv/"):
                grouped[item["name"]].append(item)
    values = []
    for name, items in sorted(grouped.items()):
        values.append(
            dict(
                name=name,
                observations=len(items),
                mean_inclusive_cpu_ms=statistics.mean(i["inclusive_cpu_us"] for i in items) / 1000,
                mean_self_cpu_ms=statistics.mean(i["self_cpu_us"] for i in items) / 1000,
                mean_self_device_ms=statistics.mean(i.get("self_device_us") or 0 for i in items)
                / 1000,
                qualification="diagnostic_profiler_contaminated_nested_ranges_not_additive",
            )
        )
    values.sort(key=lambda r: (-r["mean_inclusive_cpu_ms"], r["name"]))
    write_once(
        output / "profiler_hotspots.json",
        dict(
            bindings=bindings,
            ranges=values,
            optimization_selected=False,
            reason=(
                "Inspect measured ranges before choosing at most two semantics-preserving changes."
            ),
        ),
    )
    csv_once(output / "profiler_hotspots.csv", values)
    return values


def numerical_summary(root, output):
    reports, bindings = [], {}
    for path in sorted((root / "outputs").glob("*/*/*/correctness/*.json")):
        item = read(path)
        bindings[str(path)] = sha(path)
        reports.append(dict(path=str(path), **item))
    write_once(
        output / "numerical_error_summary.json",
        dict(
            records=reports,
            bound_files=bindings,
            qualified_contracts=sorted({r.get("numerical_contract", "legacy") for r in reports}),
            failed=sum(r.get("status") != "passed" for r in reports),
        ),
    )
    with (output / "correctness_cases.jsonl").open("x") as stream:
        for item in reports:
            stream.write(json.dumps(item, sort_keys=True, allow_nan=False) + "\n")
    return reports


def render_plots(root, output, rows):
    from xml.etree import ElementTree as ET

    plotted = []
    native = root / "analysis/scoring/native/native_metrics.json"
    if native.exists():
        for report in read(native):
            metric = report.get("metric", "cider")
            points = []
            for key, score in report["scores"].items():
                method, budget = key.split(":")
                if budget == "full":
                    continue
                relevant = [
                    r
                    for r in rows
                    if r["stage"] == "scoring"
                    and r["workload"] == report["workload"]
                    and r["method"] == method
                    and r["budget_bytes"] == int(budget)
                    and r["status"] == "completed"
                ]
                if not relevant:
                    continue
                points.append(
                    dict(
                        method=method,
                        active_mib=statistics.mean(r["initial_kv_bytes"] for r in relevant) / 2**20,
                        score=score,
                    )
                )
            if not points:
                continue
            xmin = min(p["active_mib"] for p in points)
            xmax = max(p["active_mib"] for p in points)
            ymin = min(0.0, min(p["score"] for p in points))
            ymax = max(p["score"] for p in points)
            width, height = 640, 400

            def xy(p, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax):
                return (
                    70 + 500 * (p["active_mib"] - xmin) / max(1.0, xmax - xmin),
                    315 - 250 * (p["score"] - ymin) / max(1.0, ymax - ymin),
                )

            svg = ET.Element(
                "svg",
                xmlns="http://www.w3.org/2000/svg",
                width=str(width),
                height=str(height),
                viewBox="0 0 640 400",
            )

            def label(x, y, text, *, svg=svg, **extra):
                node = ET.SubElement(
                    svg, "text", x=str(x), y=str(y), attrib={"font-size": "12", **extra}
                )
                node.text = text

            label(70, 22, f"{report['workload']}: {metric} (development only)")
            ET.SubElement(svg, "path", d="M 70 55 V 315 H 580", stroke="black", fill="none")
            label(205, 390, "Measured initial live KV (MiB)")
            for tick in range(5):
                value = ymin + (ymax - ymin) * tick / 4
                label(5, 319 - 250 * tick / 4, f"{value:.2f}")
            colors = ("#005b96", "#d55e00", "#009e73", "#6b3fa0")
            tex = [
                r"\begin{tikzpicture}",
                r"\begin{axis}[xlabel={Initial live KV (MiB)},ylabel={"
                + metric.replace("_", r"\_")
                + r"},legend pos=outer north east]",
            ]
            for index, method in enumerate(sorted({p["method"] for p in points})):
                values = sorted(
                    (p for p in points if p["method"] == method), key=lambda p: p["active_mib"]
                )
                color = colors[index % len(colors)]
                ET.SubElement(
                    svg,
                    "polyline",
                    points=" ".join(f"{x:.3f},{y:.3f}" for x, y in map(xy, values)),
                    fill="none",
                    stroke=color,
                )
                for p in values:
                    x, y = xy(p)
                    ET.SubElement(svg, "circle", cx=str(x), cy=str(y), r="4", fill=color)
                    label(x - 9, 335, f"{p['active_mib']:.1f}")
                label(70 + (index % 2) * 270, 350 + (index // 2) * 17, method, fill=color)
                tex += [
                    r"\addplot coordinates {"
                    + " ".join(f"({p['active_mib']:.12g},{p['score']:.12g})" for p in values)
                    + "};",
                    r"\addlegendentry{" + method.replace("_", r"\_") + "}",
                ]
            tex += [r"\end{axis}", r"\end{tikzpicture}"]
            filename = f"{report['workload']}-{metric}.svg"
            text_once(output / filename, ET.tostring(svg, encoding="unicode"))
            text_once(output / filename.replace(".svg", ".tex"), "\n".join(tex) + "\n")
            write_once(output / filename.replace(".svg", ".json"), points)
            plotted.append(filename)
    return plotted


def native_tables(root, output):
    """Keep three metric families separate and expose adjusted intervals."""
    path = root / "analysis/scoring/native/native_metrics.json"
    if not path.exists():
        return []
    tables = []
    for report in read(path):
        metric = report.get("metric", "cider")
        lines = [
            "% Complete development cohort only; not fresh confirmation.",
            r"\begin{tabular}{lrrr}",
            r"Method / initial cap & N & Score & Type \\",
        ]
        n = report["unique_inputs"]
        for name, score in sorted(report["scores"].items()):
            method, budget = name.split(":")
            label = method.replace("mosaickv_exact", "EviCastKV exact").replace("_", r"\_")
            cap = "Full" if budget == "full" else f"{int(budget) / 2**20:g} MiB"
            origin = "reimpl" if method.endswith("reimpl") else "local"
            lines.append(f"{label} / {cap} & {n} & {score:.6f} & {origin} " + r"\\")
        lines.append(r"\end{tabular}")
        name = f"{report['workload']}-{metric}-quality.tex"
        text_once(output / name, "\n".join(lines) + "\n")
        contrasts = []
        for item in report["contrasts"]:
            contrasts.append(
                dict(
                    workload=report["workload"],
                    metric=metric,
                    reference=item["reference"],
                    candidate=item["candidate"],
                    difference=item["difference"],
                    ordinary_95_ci=item["ordinary_95_ci"],
                    simultaneous_ci=item["simultaneous_ci"],
                    claim=(
                        "positive_in_development_scope_only"
                        if item["simultaneous_ci"][0] > 0
                        else "negative_in_development_scope_only"
                        if item["simultaneous_ci"][1] < 0
                        else "inconclusive_not_equivalent"
                    ),
                )
            )
        write_once(output / name.replace(".tex", "-contrasts.json"), contrasts)
        tables.append(name)
    return tables


def packet(root, output):
    analysis_sha = clean_source(SOURCE)
    lock = read(root / "protocol_lock.json")
    clean_source(Path(lock["source_root"]), lock["source_sha"])
    output.mkdir(parents=True, exist_ok=False)
    rows, failures, bindings = all_records(root)
    current = stages(root)
    numerics = numerical_summary(root, output)
    profiles = profile_summary(root, output)
    timing = None
    if current.get("measurement", {}).get("status") == "passed":
        timing = timing_statistics(root, output)
    plots = render_plots(root, output, rows)
    quality_tables = native_tables(root, output)
    counts = []
    for stage in (
        "smoke",
        "representation_canary",
        "correctness",
        "scoring",
        "profiling",
        "measurement",
        "mechanisms",
        "confirmation",
    ):
        selected = [r for r in rows if r["stage"] == stage]
        specpath = root / "jobs" / f"{stage}.json"
        planned = None
        if specpath.exists():
            spec = read(specpath)
            planned = sum(
                len(j["indices"])
                * len(j["arms"])
                * (lock["repeats"] if stage == "measurement" else 1)
                for j in spec["jobs"]
            )
        counts.append(
            dict(
                stage=stage,
                gate=current.get(stage, {}).get("status", "NOT_RUN"),
                planned=planned,
                attempted=len(selected),
                completed=sum(r["status"] == "completed" for r in selected),
                failed=sum(r["status"] != "completed" for r in selected),
            )
        )
    csv_once(output / "completeness.csv", counts)
    write_once(output / "failures.json", failures)
    archived = []
    for previous in sorted(root.parent.glob("fox-review-202610*-v[123]")):
        path = previous / "protocol_lock.json"
        if path.exists():
            old = read(path)
            archived.append(
                dict(
                    root=str(previous),
                    protocol_sha256=sha(path),
                    source_sha=old["source_sha"],
                    gates=stages(previous),
                    old_measurements_reused=False,
                )
            )
    write_once(output / "archived_attempts.json", archived)
    write_once(
        output / "capacity_cases.json",
        [
            dict(
                stage=r["stage"],
                sample_id=r["sample_id"],
                method=r["method"],
                budget_bytes=r["budget_bytes"],
                initial=r.get("initial_kv_bytes"),
                final=r.get("final_kv_bytes"),
                peak=r.get("peak_kv_bytes"),
                status=r["status"],
            )
            for r in rows
        ],
    )
    lines = [
        "# EviCastKV Fox-review writing packet",
        "",
        "This packet contains actual completed development artifacts, not fresh confirmation.",
        "The existing manuscript and historical raw results have not been overwritten.",
        "",
        f"Runtime SHA: `{lock['source_sha']}`. Analysis SHA: `{analysis_sha}`.",
        f"Protocol SHA256: `{sha(root / 'protocol_lock.json')}`.",
        "",
        "## Execution completeness",
        "",
        "| Stage | Gate | Planned | Attempted | Complete | Failed |",
        "|---|---|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {r['stage']} | {r['gate']} | "
        f"{r['planned'] if r['planned'] is not None else 'NOT_RUN'} | "
        f"{r['attempted']} | {r['completed']} | {r['failed']} |"
        for r in counts
    ]
    lines += [
        "",
        "## Permitted evidence and limitations",
        "",
        "- Fixed-budget quality differences do not establish lower memory at equal quality.",
        "- Six quality contrasts use ordinary 95% and Bonferroni-adjusted intervals; "
        "crossing zero is not equivalence.",
        "- COCO uses corpus CIDEr per bootstrap; MileBench Accuracy and ROUGE-L "
        "are separate equal-subtask means.",
        "- The local baselines are reimplementations, not official code. "
        "Shared-byte execution does not certify published operating-point reproduction.",
        "- Only exact selection is evaluated. Prototype, residual, repair and "
        "serving-backend claims remain unsupported.",
        "- Development and profiling rows are not independent confirmation "
        "or headline unprofiled timing.",
        "- Input-history and human-reviewed configuration requirements block confirmation. "
        "No statistical result overrides them.",
        "- Requested v1.4/Fox PDFs were unavailable; settings came from pinned "
        "source/configuration artifacts and the complete supplied prompt package.",
        "- Optional batching is NOT_RUN: the current complete adapter is batch-one.",
        "",
        "## Actual measurement contract",
        "",
        "FP16 HF eager; frozen models/data/processors; 64 decisions, 63 append forwards; "
        "prompt-only observations, unchanged retained KV and logical positions.",
        "Request wall time covers media loading through detokenization; post-output "
        "instrumentation is separate. Token times are local CPU handoffs, not network latency.",
        "Five warmups and three counterbalanced repeats apply to unprofiled measurement. "
        "Process HBM and allocator peaks are distinct from physical live KV.",
        "",
        "## Writing placement",
        "",
        "Main paper: passed validation and complete native metric/timing tables. "
        "Appendix: failed controls, boundary A/B, profiler ranges, inactive edges, "
        "full routing matrix and missingness ledger.",
        "No earlier table is silently superseded: v1-v3 remain archived failed/incomplete "
        "attempts with different contracts. This run's hashes identify its replacement scope.",
        "",
        f"Numerical reports: {len(numerics)}; profiler ranges: {len(profiles)}; "
        f"paired timing comparisons: {len(timing) if timing is not None else 'NOT_RUN'}.",
        f"Plots generated from measured saved points only: {', '.join(plots) or 'NOT_RUN'}.",
        f"Separate metric tables: {', '.join(quality_tables) or 'NOT_RUN'}.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "source mosaickv/scripts/cache_env.sh",
        'export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""',
        'export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"',
        f"{PYTHON} mosaickv/scripts/evicastkv_packet.py packet "
        f"--root {root} --output NEW_EMPTY_DIRECTORY",
        "```",
        "",
        "Inputs are read-only. Use a new directory per regeneration; PDF hashes may "
        "differ due to metadata, but numeric JSON/CSV must agree.",
    ]
    text_once(output / "final_writing_packet.md", "\n".join(lines) + "\n")
    write_once(
        output / "claim_evidence.json",
        dict(
            quality="see scoped complete native_metrics.json or NOT_RUN",
            lower_memory_equal_quality="not_tested_by_equal_budget_study",
            fresh_confirmation="NOT_RUN_BLOCKED",
            general_latency_advantage="not_established",
            prototype_residual_repair="NOT_TESTED",
            numerical_correctness="see controls and gates; no legacy failure relabeled",
            raw_bindings=bindings,
        ),
    )
    generated = {
        str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()
    }
    write_once(
        output / "analysis_manifest.json",
        dict(
            analysis_source_sha=analysis_sha,
            runtime_source_sha=lock["source_sha"],
            protocol_sha256=sha(root / "protocol_lock.json"),
            raw_bindings=bindings,
            generated_files=generated,
            row_content_sha256=digest(rows),
            measurement_type="derived_development_not_confirmation",
            raw_modified=False,
        ),
    )
    print(json.dumps(dict(output=str(output), raw_rows=len(rows), stage_counts=counts)))


def watch(root, stage):
    """Runs after a stage analyzer, then attaches once to its recorded successor."""
    clean_source(SOURCE)
    decision_path = root / "analysis" / stage / "completion_decision.json"
    if not decision_path.exists():
        packet(root, root / "writing" / f"after-{stage}-controller-failure")
        return
    decision = read(decision_path)
    if decision.get("action") != "submitted_registered_successor":
        packet(root, root / "writing" / f"after-{stage}")
        if (
            stage == "measurement"
            and decision.get("action") == "stop_for_measured_hotspot_review"
            and read(root / "analysis/measurement/GATE_measurement.json")["status"] == "passed"
        ):
            # No outcome-based choice and no configuration tuning: execute only
            # the already registered controlled-development matrix, with the
            # unchanged measured reference runtime. Confirmation stays blocked.
            from evicastkv_campaign import submit
            from evicastkv_development import prepare

            child = root.parent / (root.name + "-mechanisms")
            prepare(root, child)
            submit(child, "mechanisms")
            submission = read(child / "jobs/mechanisms-submission.json")
            schedule_watch(child, "mechanisms", submission["analysis"])
        return
    successor = decision["stage"]
    submission = read(root / "jobs" / f"{successor}-submission.json")
    schedule_watch(root, successor, submission["analysis"])


def schedule_watch(root, successor, dependency):
    command = [
        "sbatch",
        "--parsable",
        "--account=bii_dsc_community",
        "--partition=bii",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=4",
        "--mem=16G",
        "--time=00:20:00",
        "--job-name=evicast-packet",
        f"--dependency=afterany:{dependency}",
        f"--output={root}/slurm/{successor}-packet-%j.out",
        f"--error={root}/slurm/{successor}-packet-%j.err",
        str(SOURCE / "mosaickv/slurm/evicastkv_packet.sbatch"),
        str(SOURCE),
        str(root),
        successor,
        clean_source(SOURCE),
    ]
    jobid = subprocess.check_output(command, text=True).strip().split(";")[0]
    write_once(
        root / "jobs" / f"{successor}-packet-submission.json", dict(job=jobid, command=command)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("packet", "watch"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stage", default="smoke")
    args = parser.parse_args()
    if args.mode == "watch":
        watch(args.root, args.stage)
    elif args.output:
        packet(args.root, args.output)
    else:
        parser.error("packet mode requires --output")


if __name__ == "__main__":
    main()
