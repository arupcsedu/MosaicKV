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

from mosaickv.evaluation.fox_campaign import TASKS, clean_source, digest, sha, write_once
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
            if (
                row.get("resolved_config_path")
                and sha(Path(row["resolved_config_path"])) != row["resolved_config_file_sha256"]
            ):
                raise ValueError("resolved runtime configuration changed")
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
    for workload in lock["cohorts"]:
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
        per_request = defaultdict(list)
        for item in read(path):
            if item["name"].startswith("evicastkv/"):
                per_request[item["name"]].append(item)
        for name, range_rows in per_request.items():
            # Kineto exports CPU ranges and correlated CUDA-only annotations
            # with the SAME name. They are not independent request samples.
            cpu = [
                i
                for i in range_rows
                if i["inclusive_cpu_us"] or i["self_cpu_us"] or not i.get("inclusive_device_us", 0)
            ]
            device = [i for i in range_rows if i not in cpu]
            grouped[name].append(
                dict(
                    inclusive_cpu_us=sum(i["inclusive_cpu_us"] for i in cpu),
                    self_cpu_us=sum(i["self_cpu_us"] for i in cpu),
                    self_device_us=sum(
                        i["self_device_us"] for i in cpu if i.get("self_device_us") is not None
                    )
                    if any(i.get("self_device_us") is not None for i in cpu)
                    else None,
                    device_annotation_us=sum(
                        i["self_device_us"] for i in device if i.get("self_device_us") is not None
                    )
                    if any(i.get("self_device_us") is not None for i in device)
                    else None,
                    annotation_rows=len(range_rows),
                )
            )
    values = []
    for name, items in sorted(grouped.items()):
        values.append(
            dict(
                name=name,
                observations=len(items),
                annotation_rows=sum(i["annotation_rows"] for i in items),
                mean_inclusive_cpu_ms=statistics.mean(i["inclusive_cpu_us"] for i in items) / 1000,
                mean_self_cpu_ms=statistics.mean(i["self_cpu_us"] for i in items) / 1000,
                mean_self_device_ms=(
                    statistics.mean(
                        i["self_device_us"] for i in items if i.get("self_device_us") is not None
                    )
                    / 1000
                    if any(i.get("self_device_us") is not None for i in items)
                    else None
                ),
                mean_device_annotation_ms=(
                    statistics.mean(
                        i["device_annotation_us"]
                        for i in items
                        if i["device_annotation_us"] is not None
                    )
                    / 1000
                    if any(i["device_annotation_us"] is not None for i in items)
                    else None
                ),
                qualification="per_request_CPU_separate_from_device_annotations;"
                "diagnostic_profiler_contaminated_nested_ranges_not_additive",
            )
        )
    values.sort(key=lambda r: (-r["mean_inclusive_cpu_ms"], r["name"]))
    write_once(
        output / "profiler_hotspots.json",
        dict(
            bindings=bindings,
            ranges=values,
            aggregation_contract_version=2,
            supersedes="earlier annotation-row means, not raw profiles or unprofiled timings",
            optimization_selected=False,
            reason=(
                "Inspect measured ranges before choosing at most two semantics-preserving changes."
            ),
        ),
    )
    csv_once(output / "profiler_hotspots.csv", values)
    lines = [
        "# Complete-request diagnostic hotspots",
        "",
        "CPU and CUDA-only annotations are separated; observations count requests, "
        "not annotation rows. Inclusive ranges overlap and are not additive. "
        "These are profiler-contaminated diagnostics, never headline latency.",
        "",
        "| Range | Requests | CPU inclusive (ms) | CPU self (ms) |",
        "|---|---:|---:|---:|",
    ]
    lines += [
        f"| {r['name']} | {r['observations']} | {r['mean_inclusive_cpu_ms']:.6f} | "
        f"{r['mean_self_cpu_ms']:.6f} |"
        for r in values
    ]
    text_once(output / "hotspot_report.md", "\n".join(lines) + "\n")
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
            failed=sum(r.get("status") == "failed" for r in reports),
            legacy_without_explicit_status=sum("status" not in r for r in reports),
        ),
    )
    with (output / "correctness_cases.jsonl").open("x") as stream:
        for item in reports:
            stream.write(json.dumps(item, sort_keys=True, allow_nan=False) + "\n")
    with (output / "capacity_cases.jsonl").open("x") as stream:
        for item in reports:
            capacity = dict(
                path=item["path"],
                status=item.get("status", "legacy_result_without_explicit_status"),
                retained_head_position_entries=item.get("retained_entries"),
                physical_head_position_slots_including_padding=item.get("padded_entries"),
                unique_backing_bytes=item.get("actual_unique_backing_bytes"),
            )
            stream.write(json.dumps(capacity, sort_keys=True, allow_nan=False) + "\n")
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
            if report.get("contrasts"):
                plotted.append(contrast_plot(report, output))
    return plotted


def contrast_plot(report, output):
    """Draw the actual paired intervals; never infer pointwise error bars."""
    from xml.etree import ElementTree as ET

    values = []
    for item in report["contrasts"]:
        if not item.get("reference", "prompt_attention_topk:").startswith("prompt_attention_topk:"):
            continue
        values.append(
            dict(
                cap_mib=int(item["candidate"].rsplit(":", 1)[-1]) / 2**20,
                difference=item["difference"],
                ordinary=item["ordinary_95_ci"],
                adjusted=item["simultaneous_ci"],
            )
        )
    low = min(0.0, *(min(v["adjusted"][0], v["ordinary"][0], v["difference"]) for v in values))
    high = max(0.0, *(max(v["adjusted"][1], v["ordinary"][1], v["difference"]) for v in values))
    span = max(high - low, 1e-12)

    def y(value):
        return 310 - 240 * (value - low) / span

    svg = ET.Element("svg", xmlns="http://www.w3.org/2000/svg", width="640", height="390")

    def text(x, y_pos, value):
        ET.SubElement(svg, "text", x=str(x), y=str(y_pos), attrib={"font-size": "12"}).text = value

    text(60, 22, f"{report['workload']}: EviCastKV minus Top-K, {report.get('metric', 'cider')}")
    text(60, 40, "Development only: ordinary 95% (blue), Bonferroni-adjusted (orange)")
    ET.SubElement(
        svg,
        "line",
        x1="60",
        x2="600",
        y1=str(y(0)),
        y2=str(y(0)),
        stroke="gray",
        attrib={"stroke-dasharray": "4 3"},
    )
    tex = [
        r"\begin{tikzpicture}",
        r"\begin{axis}[xlabel={Initial cap (MiB)},ylabel={Paired score difference}]",
    ]
    for index, v in enumerate(values):
        x = 150 + 300 * index / max(1, len(values) - 1)
        for bounds, color, offset in (
            (v["adjusted"], "#d55e00", -3),
            (v["ordinary"], "#005b96", 3),
        ):
            ET.SubElement(
                svg,
                "line",
                x1=str(x + offset),
                x2=str(x + offset),
                y1=str(y(bounds[0])),
                y2=str(y(bounds[1])),
                stroke=color,
                attrib={"stroke-width": "3"},
            )
            for endpoint in bounds:
                ET.SubElement(
                    svg,
                    "line",
                    x1=str(x + offset - 7),
                    x2=str(x + offset + 7),
                    y1=str(y(endpoint)),
                    y2=str(y(endpoint)),
                    stroke=color,
                )
        ET.SubElement(svg, "circle", cx=str(x), cy=str(y(v["difference"])), r="4", fill="black")
        text(x - 15, 335, f"{v['cap_mib']:g} MiB")
        text(x - 55, 355, f"Delta {v['difference']:.4g}")
        lo, hi = v["adjusted"]
        tex += [
            r"\addplot[orange] coordinates {"
            + f"({v['cap_mib']},{lo}) ({v['cap_mib']},{hi})"
            + "};",
            r"\addplot[only marks] coordinates {" + f"({v['cap_mib']},{v['difference']})" + "};",
        ]
    text(5, 75, f"{high:.5g}")
    text(5, 310, f"{low:.5g}")
    tex += [r"\end{axis}", r"\end{tikzpicture}"]
    name = f"{report['workload']}-{report.get('metric', 'cider')}-paired-difference.svg"
    text_once(output / name, ET.tostring(svg, encoding="unicode"))
    text_once(output / name.replace(".svg", ".tex"), "\n".join(tex) + "\n")
    write_once(output / name.replace(".svg", ".json"), values)
    return name


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
        diff_lines = [
            r"\begin{tabular}{lrrr}",
            r"Baseline / cap & EviCastKV difference & 95\% CI & Adjusted CI \\",
        ]
        for item in contrasts:
            cap = int(item["candidate"].rsplit(":", 1)[-1]) / 2**20
            baseline = item["reference"].split(":")[0].replace("_", r"\_")
            lo, hi = item["ordinary_95_ci"]
            alo, ahi = item["simultaneous_ci"]
            diff_lines.append(
                f"{baseline} / {cap:g} MiB & {item['difference']:.6f} & "
                f"[{lo:.6f}, {hi:.6f}] & [{alo:.6f}, {ahi:.6f}] " + r"\\"
            )
        diff_lines.append(r"\end{tabular}")
        text_once(
            output / name.replace("quality.tex", "differences.tex"), "\n".join(diff_lines) + "\n"
        )
        tables.append(name)
    return tables


def supplemental_exports(root, output, rows, lock):
    """No scorer/runtime change: expose setup, complete phases and B controls."""
    from itertools import pairwise

    setup = {}
    write_once(
        output / "workload_scope.json",
        dict(
            originally_requested=lock.get("originally_requested_workloads", ["coco", "milebench"]),
            enabled=lock.get("enabled_workloads", list(lock.get("cohorts", {}))),
            unexecuted_primary_contrasts=lock.get("unexecuted_primary_contrasts", 0),
            scope_registration=lock.get("scope_registration"),
            primary_family_size_retained=6,
        ),
    )
    for workload, info in lock.get("templates", {}).items():
        if sha(Path(info["path"])) != info["sha256"]:
            raise ValueError("frozen template changed")
        config = read(info["path"])
        setup[workload] = {
            key: config.get(key)
            for key in (
                "model",
                "dataset",
                "execution",
                "generation",
                "cache",
                "forecasting",
                "graph",
                "utility",
                "selection",
                "prototypes",
                "residual",
                "repair",
            )
        }
    write_once(output / "experimental_setup.json", setup)
    phases = []
    for row in rows:
        if row["stage"] not in {"measurement", "optimization"} or row["status"] != "completed":
            continue
        metrics = row["metrics"]
        breakdown = read(row["trace_path"]).get("timing_breakdown", {})
        telemetry_path = Path(row["timings"].get("sampled_telemetry_path", "MISSING"))
        telemetry = read(telemetry_path) if telemetry_path.is_file() else {}
        emissions = row["timings"].get("token_emission_seconds", [])
        gaps = [b - a for a, b in pairwise(emissions)]
        phases.append(
            dict(
                model=setup[row["workload"]]["model"]["id"],
                stage=row["stage"],
                implementation_variant=row.get("variant"),
                workload=row["workload"],
                task=row["subtask"] or "COCO Caption",
                sample_id=row["sample_id"],
                method=row["method"],
                budget_bytes=row["budget_bytes"],
                repeat=row["repeat"],
                prefill_cuda_seconds=metrics.get("prefill_time"),
                image_video_encoder_seconds=breakdown.get("image_video_encoder"),
                projector_seconds=breakdown.get("projector"),
                language_model_prefill_seconds=breakdown.get("language_model_prefill"),
                cache_state_construction_seconds=breakdown.get("cache_state_construction"),
                forecast_seconds=breakdown.get("forecast"),
                planner_preparation_seconds=breakdown.get("planner_preparation"),
                graph_seconds=breakdown.get("graph"),
                utility_and_selection_seconds=breakdown.get("utility_and_selection"),
                tier_construction_seconds=breakdown.get("tier_construction"),
                cache_packing_seconds=breakdown.get("cache_packing"),
                compression_seconds=metrics.get("compression_time"),
                decode_cuda_seconds=metrics.get("decode_time"),
                e2e_wall_seconds=row["e2e_wall_seconds"],
                mean_local_token_handoff_gap=statistics.mean(gaps) if gaps else None,
                first_handoff_seconds=row["timings"].get("first_token_emitted_seconds"),
                first_logit_ready_seconds=row["timings"].get("first_logit_ready_seconds"),
                second_handoff_seconds=row["timings"].get("second_token_emitted_seconds"),
                initial_kv_bytes=row["initial_kv_bytes"],
                final_kv_bytes=row["final_kv_bytes"],
                peak_kv_bytes=row["peak_kv_bytes"],
                peak_allocated_bytes=row["peak_allocated_bytes"],
                peak_reserved_bytes=row["peak_reserved_bytes"],
                sampled_process_gpu_peak_bytes=telemetry.get("sampled_process_gpu_peak_bytes"),
                sampled_cpu_rss_peak_bytes=telemetry.get("sampled_cpu_rss_peak_bytes"),
                sampler_interval_seconds=telemetry.get("interval_seconds"),
                telemetry_sha256_at_analysis=sha(telemetry_path) if telemetry else None,
                telemetry_hash_scope="first bound at derived analysis, not original raw-write hash",
                process_peak_scope=telemetry.get("process_peak_scope", "NOT_AVAILABLE"),
                gpu_uuid=row.get("gpu_uuid"),
                source_sha=row["source_sha"],
            )
        )
    csv_once(output / "phase_breakdown.csv", phases)
    text_once(
        output / "measurement_contract.md",
        "# Measurement contract\n\nFP16 HF eager; 64 decisions and 63 append forwards; "
        "no draft/prototype/residual/repair. Each timed request recomputes media input, "
        "prefill, policy and decode. Five warmups and three counterbalanced repeats. "
        "Wall E2E ends after detokenization at the completion callback; full API return "
        "and post-output instrumentation are separate. Token times are local consumer "
        "handoffs, not network timings. CUDA phase events are synchronized. "
        "Profiler rows never enter headline latency. Physical active KV includes backing "
        "storage/padding/preallocation, and is not process HBM. Allocator allocated/reserved "
        "peaks cover the complete request. NVML/RSS peaks are sampled at the recorded "
        "interval and can miss peaks; their scope includes post-output instrumentation. "
        "Unavailable telemetry remains null, never replaced with KV bytes. "
        "Prefill includes vision/projector/LM subranges; nested ranges are not additive. "
        "Published active KV is not total resident KV: source/plan objects remain in the "
        "runtime frame through output construction. No total-resident-KV or process-HBM "
        "compression claim may be inferred from this active-cache ratio.\n",
    )
    csv_once(output / "fullkv_reference.csv", [r for r in phases if r["method"] == "full_kv"])
    csv_once(
        output / "memory_boundaries.csv",
        [
            {
                k: r[k]
                for k in (
                    "workload",
                    "task",
                    "sample_id",
                    "method",
                    "budget_bytes",
                    "repeat",
                    "initial_kv_bytes",
                    "final_kv_bytes",
                    "peak_kv_bytes",
                    "peak_allocated_bytes",
                    "peak_reserved_bytes",
                )
            }
            for r in phases
        ],
    )
    boundaries = []
    for path in sorted((root / "outputs/scoring").glob("*/*/boundary/*.json")):
        item = read(path)
        boundaries.append(dict(path=str(path), sha256=sha(path), **item))
    write_once(
        output / "boundary_A_B_diagnostics.json",
        dict(
            stratum="changed_boundary_not_primary_quality_or_headline_timing",
            records=boundaries,
            completed=sum(r.get("status") == "completed" for r in boundaries),
            blocked=sum(r.get("status") == "blocked" for r in boundaries),
            invalid_fullkv_control=sum(
                r.get("fullkv_chunked_logit_control_passed") is False for r in boundaries
            ),
            first_token_frozen_in_boundary_A=True,
        ),
    )
    inherited = lock.get("inherited_gates", {})
    write_once(
        output / "inherited_controls.json",
        dict(
            gates=inherited,
            note="Inherited gates are not new executions; parent/source hashes remain separate.",
        ),
    )


def claim_ledger(root, output, bindings, current, timing):
    """Human-readable, hash-bound claims; missing contrasts stay NOT_TESTED."""
    write_once(output / "raw_bindings.json", bindings)
    raw_sha = sha(output / "raw_bindings.json")
    native = root / "analysis/scoring/native/native_metrics.json"
    claims = []
    if native.exists():
        for report in read(native):
            for contrast in report["contrasts"]:
                low, high = contrast["simultaneous_ci"]
                claims.append(
                    dict(
                        statement="positive quality difference at matched physical bytes",
                        status=(
                            "supported_in_development_scope"
                            if low > 0
                            else "contradicted_in_development_scope"
                            if high < 0
                            else "inconclusive"
                        ),
                        workload=report["workload"],
                        metric=report.get("metric", "cider"),
                        unique_inputs=report["unique_inputs"],
                        reference=contrast["reference"],
                        candidate=contrast["candidate"],
                        row_filter="scoring; completed entire paired cohort; one record/input/arm",
                        test="10000 input-media-cluster paired draws; corpus/hierarchy reducer",
                        point_difference=contrast["difference"],
                        ordinary_95_ci=contrast["ordinary_95_ci"],
                        simultaneous_ci=contrast["simultaneous_ci"],
                        family_size=contrast["family_size"],
                        family=contrast.get("statistical_family", "registered_family"),
                        source=str(native),
                        source_sha256=sha(native),
                        raw_manifest_sha256=raw_sha,
                        confirmatory=False,
                    )
                )
    lines = [
        "# Claim-to-evidence ledger",
        "",
        "Every quality statement below is development-only. Zero-crossing or zero-width "
        "intervals do not establish population equivalence. Local ports are not official code.",
        "",
        f"Raw records: `raw_bindings.json`, SHA256 `{raw_sha}`.",
        "",
    ]
    for index, item in enumerate(claims, 1):
        lines.extend(
            [
                f"## Q{index}: {item['workload']} / {item['metric']}",
                "",
                f"- Classification: **{item['status']}**.",
                f"- Contrast: `{item['candidate']}` minus `{item['reference']}`; "
                f"N={item['unique_inputs']}.",
                f"- Difference: {item['point_difference']:.12g}; ordinary 95% CI "
                f"{item['ordinary_95_ci']}; adjusted CI {item['simultaneous_ci']}.",
                f"- Test: {item['test']}; family `{item['family']}`, size {item['family_size']}.",
                f"- Filter: {item['row_filter']}.",
                f"- Artifact: `{item['source']}`; SHA256 `{item['source_sha256']}`.",
                "",
            ]
        )
    lines.extend(
        [
            "## Unestablished claims",
            "",
            "- Lower memory at equal quality: NOT_TESTED by this equal-budget matrix.",
            "- Fresh confirmation: NOT_RUN/BLOCKED; development is not independent confirmation.",
            "- General latency superiority: NOT_ESTABLISHED; consult separate paired timing rows.",
            "- Prototype/residual/repair and serving: NOT_TESTED in this exact-only campaign.",
            "- Optimization speedup: NOT_TESTED until an explicit before/after equivalence run.",
            "- Missing workload contrasts remain NOT_TESTED; "
            "the original six-test family is retained.",
            "",
            f"Scoring gate: {current.get('scoring', {}).get('status', 'NOT_RUN')}. "
            f"Paired timing contrasts: {len(timing) if timing is not None else 'NOT_RUN'}.",
        ]
    )
    text_once(output / "claim_evidence.md", "\n".join(lines) + "\n")
    write_once(output / "quality_claims.json", claims)
    return claims


def mechanism_exports(root, output, rows):
    """Separate controlled-mechanism results from native deployment timing."""
    selected = [r for r in rows if r["stage"] == "mechanisms"]
    if not selected:
        return
    with (output / "development_predictions.jsonl").open("x") as stream:
        for row in selected:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    activity_path = root / "analysis/mechanisms/configuration_activity.json"
    activity = read(activity_path) if activity_path.exists() else []
    write_once(output / "configuration_activity.json", activity)
    by_key = {(r["workload"], r["sample_id"], r["variant"], r["budget_bytes"]): r for r in activity}
    native_path = root / "analysis/mechanisms/native/mechanism_metrics.json"
    scores = {}
    if native_path.exists():
        for report in read(native_path):
            for point, value in report["scores"].items():
                scores[(report["workload"], report.get("metric", "cider"), point)] = value
    groups = defaultdict(list)
    for row in selected:
        groups[(row["workload"], row["variant"], row["budget_bytes"])].append(row)
    summaries = []
    for (workload, variant, budget), values in sorted(groups.items()):
        for metric in ("cider",) if workload == "coco" else ("accuracy", "rouge_l_f1"):
            family_tasks = TASKS[:2] if metric == "accuracy" else TASKS[2:]
            applicable = (
                values
                if workload == "coco"
                else [r for r in values if r["subtask"] in family_tasks]
            )
            completed = [r for r in applicable if r["status"] == "completed"]
            measures = [
                by_key[(workload, r["sample_id"], variant, budget)]
                for r in completed
                if (workload, r["sample_id"], variant, budget) in by_key
            ]
            summaries.append(
                dict(
                    workload=workload,
                    variant=variant,
                    metric=metric,
                    budget_bytes=budget,
                    attempted=len(applicable),
                    completed=len(completed),
                    failed=len(applicable) - len(completed),
                    native_quality=scores.get((workload, metric, f"{variant}:{budget}")),
                    mean_initial_kv_bytes=statistics.mean(r["initial_kv_bytes"] for r in completed)
                    if completed
                    else None,
                    mean_graph_nodes=statistics.mean(r["graph_nodes"] for r in measures)
                    if measures
                    else None,
                    mean_graph_edges=statistics.mean(r["graph_edges"] for r in measures)
                    if measures
                    else None,
                    mean_selected_jaccard=statistics.mean(r["selected_jaccard"] for r in measures)
                    if measures
                    else None,
                    mean_controlled_e2e=statistics.mean(r["e2e_wall_seconds"] for r in completed)
                    if completed
                    else None,
                    timing_scope="fixed-quota mechanism diagnostic, not native deployment timing",
                )
            )
    fixed = {"mechanism-complete", "routed-prompt", "coverage-off", "value-off", "cross-modal-off"}
    csv_once(
        output / "routing_sweep.csv", [s for s in summaries if s["variant"].startswith("route-")]
    )
    csv_once(output / "component_ablation.csv", [s for s in summaries if s["variant"] in fixed])
    csv_once(
        output / "routing_controls.csv",
        [
            s
            for s in summaries
            if s["variant"].startswith("random-") or s["variant"] == "mechanism-complete"
        ],
    )
    write_once(output / "mechanism_summaries.json", summaries)
    text_once(
        output / "development_decision.md",
        "# Development decision\n\nNO_CONFIG_FROZEN. All registered controls and random seeds "
        "remain reported separately; no best-seed selection. Controlled diagnostic timings "
        "are not native deployment latency. No confirmation authorized.\n",
    )
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"Workload / metric & Variant & Cap (MiB) & N & Score & Nodes \\",
    ]
    for r in summaries:
        score = "NOT RUN" if r["native_quality"] is None else f"{r['native_quality']:.6f}"
        nodes = "NOT RUN" if r["mean_graph_nodes"] is None else f"{r['mean_graph_nodes']:.1f}"
        label = f"{r['workload']}/{r['metric']}".replace("_", r"\_")
        variant = r["variant"].replace("_", r"\_")
        lines.append(
            f"{label} & {variant} & {r['budget_bytes'] / 2**20:g} & "
            f"{r['completed']} & {score} & {nodes} " + r"\\"
        )
    lines.append(r"\end{tabular}")
    text_once(output / "mechanism_complete_matrix.tex", "\n".join(lines) + "\n")


def packet(root, output):
    analysis_sha = clean_source(SOURCE)
    lock = read(root / "protocol_lock.json")
    clean_source(Path(lock["source_root"]), lock["source_sha"])
    output.mkdir(parents=True, exist_ok=False)
    rows, failures, bindings = all_records(root)
    current = stages(root)
    if any(
        gate["protocol_sha256"] != sha(root / "protocol_lock.json") for gate in current.values()
    ):
        raise ValueError("gate belongs to a different protocol")
    supplemental_exports(root, output, rows, lock)
    mechanism_exports(root, output, rows)
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
        "optimization",
        "mechanisms",
        "confirmation",
    ):
        selected = [r for r in rows if r["stage"] == stage]
        specpath = root / "jobs" / f"{stage}.json"
        planned = None
        if specpath.exists():
            spec = read(specpath)
            planned = (
                sum(
                    len(j["indices"])
                    * len(j["arms"])
                    * (lock["repeats"] if stage == "measurement" else 1)
                    for j in spec["jobs"]
                )
                if all("arms" in j for j in spec["jobs"])
                else None
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
    notice = root / "writing/implementation_correction_notice.json"
    if notice.exists():
        correction = read(notice)
        write_once(output / "implementation_correction_notice.json", correction)
        lines[1:1] = [
            "", "## Materialization correction — read before using latency results", "",
            correction["summary"],
            "The original predictions and published-cache byte counts are preserved. "
            "Do not use the affected Top-K timings as uniform-materialization comparison evidence.",
            f"Notice SHA256: `{sha(notice)}`.", "",
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
    claims = claim_ledger(root, output, bindings, current, timing)
    required = ("correctness", "scoring", "profiling", "measurement", "mechanisms", "confirmation")
    missing = [name for name in required if current.get(name, {}).get("status") != "passed"]
    write_once(
        output / "GATE_analysis.json",
        dict(
            status="partial" if missing else "passed",
            source_sha=analysis_sha,
            protocol_sha256=sha(root / "protocol_lock.json"),
            bound_exports_validated=True,
            missing_or_blocked_required_stages=missing,
            actual_raw_rows=len(rows),
            failed_rows=len(failures),
            quality_contrasts_exported=len(claims),
            confirmation_claim_allowed=False,
            no_missing_results_invented=True,
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
