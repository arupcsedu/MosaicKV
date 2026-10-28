#!/usr/bin/env python3
"""Reproduce completed evidence and build a strictly scoped Fox-review packet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evicastkv_analyze import collect
from evicastkv_evidence import native_statistics
from evicastkv_optimization import factorial
from evicastkv_packet import csv_once, native_tables, render_plots, text_once
from evicastkv_statistics_recheck import compare_numeric
from evicastkv_transport_scores import analyze, prepare

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
BASE = Path("/scratch/djy8hg/runs/evicastkv")
COCO = BASE / "fox-review-20261028-v6-coco-scoped"
TIMING = BASE / "fox-review-20261028-v8-common-gather"
SCORES = BASE / "fox-review-20261028-common-transport-scores-v1"


def read(path):
    return json.loads(Path(path).read_text())


def verdict(low, high):
    if low > 0:
        return "positive_in_consumed_development_only"
    if high < 0:
        return "negative_in_consumed_development_only"
    return "inconclusive_not_equivalence_or_superiority"


def build(root, schema_root):
    import pandas as pd

    revision = clean_source(SOURCE)
    if root.exists():
        raise FileExistsError("writing packet must be regenerated into a new directory")
    root.mkdir(parents=True)
    bindings, comparisons = {}, {}

    def bind(path):
        bindings[str(path)] = sha(path)
        return read(path)

    schema_report = bind(schema_root / "report.json")
    provenance = bind(BASE / "fox-review-20261028-provenance-resolution-v1/report.json")
    # CIDEr is recomputed at corpus level and in every bootstrap replicate.
    actual_coco = native_statistics(COCO, "scoring", root / "coco_reproduced", workers=1)
    expected_coco = bind(COCO / "analysis/scoring/native/native_metrics.json")
    comparisons["coco_max_numeric_error"] = compare_numeric(expected_coco, actual_coco)

    # Re-score all 128 saved rows; never reopen a GPU or choose a new cohort.
    mile = root / "mile_reproduced"
    prepare(
        mile,
        BASE / "fox-review-20261028-v4",
        [
            BASE / "fox-review-20261028-mile-common-transport-v1",
            BASE / "fox-review-20261028-mile-common-transport-32-v1",
        ],
        BASE / "fox-review-20261028-consumed-regate-cpu-initialized-v1/GATE_cpu_suite.json",
    )
    analyze(mile)
    expected_mile = bind(SCORES / "paired_statistics.json")
    actual_mile = read(mile / "paired_statistics.json")
    comparisons["mile_max_numeric_error"] = compare_numeric(expected_mile, actual_mile)
    frame = pd.read_parquet(mile / "metric_aggregates.parquet")
    old_frame = pd.read_parquet(SCORES / "metric_aggregates.parquet")
    pd.testing.assert_frame_equal(frame, old_frame, check_exact=True)
    comparisons["mile_per_task_aggregates_bitwise_identical"] = True

    # Rebuild timing per input before resampling; do not treat repeats as N.
    lock = bind(TIMING / "protocol_lock.json")
    clean_source(Path(lock["source_root"]), lock["source_sha"])
    if bind(TIMING / "analysis/optimization/GATE_optimization.json")["status"] != "passed":
        raise ValueError("timing factorial gate failed")
    _, _, timing_rows, errors = collect(TIMING, "optimization")
    if errors or len(timing_rows) != 432:
        raise ValueError("incomplete timing cohort")
    summaries, timing, equivalence = factorial(TIMING, timing_rows)
    comparisons["timing_max_numeric_error"] = compare_numeric(
        bind(TIMING / "analysis/optimization/paired_factorial.json"), timing
    )
    csv_once(root / "timing_by_input.csv", summaries)
    write_once(root / "timing_factorial.json", timing)
    write_once(root / "timing_equivalence.json", equivalence)

    tables = root / "tables"
    plots = root / "plots"
    tables.mkdir()
    plots.mkdir()
    native_tables(COCO, tables)  # Numbers independently revalidated above.
    _, _, coco_rows, errors = collect(COCO, "scoring")
    if errors:
        raise ValueError("incomplete COCO scoring rows")
    render_plots(COCO, plots, [dict(r, stage="scoring") for r in coco_rows])
    for metric in ("accuracy", "rouge_l_f1"):
        selected = frame[frame.metric_name == metric]
        lines = [
            "% CONSUMED DEVELOPMENT; not fresh confirmation.",
            r"\begin{tabular}{llrrr}",
            r"Task & Method & Initial MiB & N & Score \\",
        ]
        for r in selected.to_dict("records"):
            method_label = r["method"].replace("_", r"\_")
            lines.append(
                f"{r['task']} & {method_label} & "
                f"{r['budget_bytes'] / 2**20:g} & {r['scored']} & {r['metric_value']:.8f} " + r"\\"
            )
        lines.append(r"\end{tabular}")
        text_once(tables / f"mile-{metric}-per-task.tex", "\n".join(lines) + "\n")
        data = []
        tex = [
            r"\begin{tikzpicture}",
            r"\begin{axis}[xlabel={Initial live KV (MiB)},ylabel={"
            + metric.replace("_", r"\_")
            + r"},legend pos=outer north east]",
        ]
        for method in sorted(selected.method.unique()):
            part = selected[selected.method == method]
            points = [
                (int(cap) / 2**20, float(values.metric_value.mean()))
                for cap, values in part.groupby("budget_bytes")
            ]
            data.extend(dict(method=method, initial_mib=cap, value=value) for cap, value in points)
            tex += [
                r"\addplot[only marks] coordinates {"
                + " ".join(f"({cap:g},{value:.12g})" for cap, value in points)
                + "};",
                r"\addlegendentry{" + method.replace("_", r"\_") + "}",
            ]
        tex += [r"\end{axis}", r"\end{tikzpicture}"]
        text_once(plots / f"mile-{metric}.tex", "\n".join(tex) + "\n")
        write_once(plots / f"mile-{metric}.json", data)

    claims = []
    for report in actual_coco:
        for item in report["contrasts"]:
            claims.append(
                dict(
                    metric="cider",
                    workload="coco",
                    scope="consumed_32_input_development",
                    **item,
                    verdict=verdict(*item["simultaneous_ci"]),
                )
            )
    for item in actual_mile:
        claims.append(
            dict(
                workload="milebench",
                scope="consumed_16_inputs_per_metric_family",
                **item,
                verdict=verdict(*item["simultaneous_ci"]),
            )
        )
    write_once(root / "claim_evidence.json", claims)
    cpu = bind(BASE / "fox-review-20261028-bounded-scoring-cpu-v1/GATE_cpu_suite.json")
    if cpu["status"] != "passed":
        raise ValueError("latest initialized CPU suite failed")
    write_once(
        root / "GATE_analysis.json",
        dict(
            source_sha=revision,
            status="passed_reproduction_limited_writing_only",
            comparisons=comparisons,
            source_bindings=bindings,
            CPU_suite=cpu["counts"],
            GPU_inference=0,
            fresh_confirmation=False,
            author_reviewed_config=False,
            superiority_claim_allowed=False,
            original_primary_matrix_expected=3584,
            fresh_confirmation_requests_executed=0,
            schemas_audited=len(schema_report["cases"]),
            source_alias_certified=False,
            roots_resolved_for_exclusion=provenance["root_classifications_resolved_for_exclusion"],
        ),
    )
    text_once(
        root / "final_writing_packet.md",
        """# Prof. Fox review: reproducible completed-evidence packet

Status: development reproduced; broad/fresh-confirmation claims BLOCKED.

## What can be written

- Exact packed-cache controls passed on the declared consumed inputs.
- The initialized predecessor CPU suite passed 1,760 tests.
- Complete 32-input COCO corpus CIDEr and paired intervals are reproduced.
- The 32 consumed MileBench inputs have separate per-task metric tables.
- The 16-input, three-repeat COCO implementation factorial is recomputed.
- Common packing gains benefit both methods, not just graph selection.

## Negative, inconclusive or not tested

- No strictly positive primary adjusted interval establishes superiority.
- Tied observed Accuracy or a zero-crossing CI is not population equivalence.
- No smaller-cache-at-equal-quality or three-tier/repair/serving claim passes.
- Instrumented correctness timings are not performance evidence.
- COCO boundary-B failures remain negative; thresholds were not relaxed.
- Official LOOK-M execution and transformed-cache fidelity remain unverified.
- Old Top-K timings ignoring gathering are superseded by the common factorial.

## Confirmation requires important author/data input

NO_CONFIG_FROZEN: no author-reviewed selected_config.json was manufactured.
The original four MileBench pools cannot supply the untouched cohort.
Replacement assets are verified, but global source aliases are uncertified.
Supply verifiable source mappings, including historical TextNeedle coverage,
or approve a new scope with independently verifiable fresh pools.
A schema audit is not a freshness certificate.
The original 3,584-request fresh matrix has executed zero requests.

## Files and reproduction

GATE_analysis.json: source/hash bindings and reproduction errors.
tables/: separate metric tables. plots/: measured SVG/TikZ/data sources.
coco_reproduced/: corpus reduction and cluster bootstrap.
mile_reproduced/: scorer fixtures, per-task metrics and paired intervals.
timing_by_input.csv and timing_factorial.json: repeat-aware comparison.
claim_evidence.json: every signed contrast and adjusted interval.
Reproduce into a NEW scratch output with evicastkv_review_closure.py.
Supply --schema-root and --output; do not overwrite previous artifacts.

Do not strengthen the manuscript or abstract from development results.
""",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    for p in (args.schema_root, args.output):
        if not p.resolve().is_relative_to(Path("/scratch/djy8hg").resolve()):
            raise ValueError("scratch-only writing packet")
    build(args.output, args.schema_root)
