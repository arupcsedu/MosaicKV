#!/usr/bin/env python3
"""CPU-only, provenance-bound export. No inference, scoring, or result mutation."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RUNS = Path("/scratch/djy8hg/runs")
PREFIX = RUNS / "mosaickv/prefixkv/full-campaign-e4de390-20261019-v6"
INTER = RUNS / "mosaickv/prefix-native-intermediate-20261029-v2-retry1"
FOX = RUNS / "evicastkv/fox-installed-guard-20261029-v2-retry1"
CLOSURE = RUNS / "evicastkv/fox-review-20261028-writing-closure-v2-report-retry1"
COMPACT = RUNS / "evicastkv/compact-exact-views-consumed-20261029-v1"
AUDIT = RUNS / "evicastkv/cpu-routing-prefix-audit-20261029-v1"
COCO = (
    RUNS
    / "mosaickv/coco-shared-32-edb6edf-retry1-20261014/analysis/lower-memory-frontier-21636391"
)
NAMES = {
    "full_kv": "FullKV",
    "prompt_attention_topk": "Prompt Top-K",
    "prefixkv_reimpl": "PrefixKV (R)",
    "vl_cache_reimpl": "VL-Cache (R)",
    "mosaickv_exact": r"\textbf{EviCastKV exact}",
}
STORAGE = [
    "initial_active_kv_storage_bytes",
    "final_active_kv_storage_bytes",
    "peak_active_kv_storage_bytes",
]
BOUND: dict[str, str] = {}
MIB = 2**20
ENGINEERING = RUNS / "evicastkv"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bind(path: Path, expected: str | None = None) -> Path:
    digest = sha(path)
    if expected is not None and digest != expected:
        raise ValueError(f"changed source artifact: {path}")
    BOUND[str(path)] = digest
    return path


def read(path: Path):
    return json.loads(bind(path).read_text())


def frame(path: Path) -> pd.DataFrame:
    return pd.read_parquet(bind(path))


def write(name: str, content: str) -> None:
    dest = HERE / "generated" / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content + "\n")


def table(name: str, spec: str, header: str, rows: list[str]) -> None:
    write(
        name + ".tex",
        "\n".join(
            [
                "% Derived from immutable measured artifacts by export_report.py.",
                r"\begin{tabular}{" + spec + "}",
                r"\toprule",
                header + r"\\\midrule",
                *[r + r"\\" for r in rows],
                r"\bottomrule",
                r"\end{tabular}",
            ]
        ),
    )


def data(name: str, columns: dict) -> None:
    pd.DataFrame(columns).to_csv(
        HERE / "generated" / (name + ".dat"), sep=" ", index=False
    )


def row(metrics, task, lane, method, metric):
    subset = metrics[
        (metrics.task == task)
        & (metrics.lane == lane)
        & (metrics.method == method)
        & (metrics.metric_name == metric)
    ]
    if len(subset) != 1:
        raise ValueError("ambiguous aggregate")
    return subset.iloc[0]


def export_storage_engineering() -> dict:
    """Export the sealed two-input diagnostic, never a new quality comparison."""
    seal = ENGINEERING / "storage-engineering-final-readiness-20261030-v1"
    readiness = json.loads(
        bind(
            seal / "readiness.json",
            "819e963fc690992e3383c8c08b93ca4bc38de0bc43dbfc3287ceb08766755335",
        ).read_text()
    )
    assert readiness["cpu_suite"] == {
        "tests": 2631,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
    }
    frames = {}
    for kind, directory in [
        ("runtime", "source-history-runtime-final-report-20261030-v1"),
        ("direct", "source-history-direct-final-report-20261030-v1"),
    ]:
        root = ENGINEERING / directory
        manifest_path = root / "manifest.json"
        manifest = json.loads(
            bind(manifest_path, readiness["inputs"][str(manifest_path)]).read_text()
        )
        assert manifest["complete_workers"] == manifest["expected_workers"] == 12
        assert not manifest["failures"]
        assert manifest["whole_request_peak_certified"] is False
        assert manifest["independent_quality_established"] is False
        for filename in ["controls.parquet", "report.md"]:
            bind(root / filename, manifest["outputs"][filename])
        frame_ = pd.read_parquet(root / "controls.parquet")
        assert len(frame_) == 12
        assert not frame_.duplicated(["task", "sample_id", "method"]).any()
        assert frame_.original_logits_bitwise.all()
        assert frame_.registered_memory_gate.all()
        assert frame_.actual_output_tokens.eq(128).all()
        frames[kind] = frame_
    runtime, direct = frames["runtime"], frames["direct"]
    names = {
        **NAMES,
        "full_kv": "All-position lossless control",
        "lookm_reimpl": "LOOK-M (R)",
        "prefixkv_reimpl": "PrefixKV adaptation (R)",
        "vl_cache_reimpl": "VL-Cache adaptation (R)",
    }
    storage_rows, latency_rows, quality_rows = [], [], []
    for task, label in [
        ("llava_description", "Description"),
        ("mmvet_prefixkv", "MM-Vet route"),
    ]:
        for method in [
            "full_kv",
            "prompt_attention_topk",
            "prefixkv_reimpl",
            "vl_cache_reimpl",
            "lookm_reimpl",
            "mosaickv_exact",
        ]:
            a = runtime[(runtime.task == task) & (runtime.method == method)].iloc[0]
            b = direct[(direct.task == task) & (direct.method == method)].iloc[0]
            assert a.sample_id == b.sample_id
            for boundary in ["initial", "final", "peak"]:
                assert (
                    a[f"candidate_{boundary}_bytes"] == b[f"reference_{boundary}_bytes"]
                )
                assert (
                    b[f"reference_{boundary}_bytes"] == b[f"candidate_{boundary}_bytes"]
                )
            storage_rows.append(
                f"{label} & {names[method]} & {a.reference_initial_MiB:.3f} & "
                f"{a.reference_final_MiB:.3f} & {a.candidate_final_MiB:.3f} & "
                f"{100 * a.final_reduction_fraction:.2f} & {a.reference_peak_MiB:.3f} & "
                f"{a.candidate_peak_MiB:.3f} & {100 * a.peak_reduction_fraction:.2f}"
            )
            quality_rows.append(
                f"{label} & {names[method]} & "
                f"{100 * b.reference_rouge_l_32_model_generated_only:.2f} & "
                f"{100 * b.reference_token_agreement_32_model_generated_only:.2f} & Yes"
            )
            if method != "full_kv":
                assert pd.notna(a.paired_latency_ratio)
                assert pd.notna(b.paired_latency_ratio) and b.latency_measured
                latency_rows.append(
                    f"{label} & {names[method]} & {a.paired_latency_ratio:.4f} & "
                    f"{b.reference_decode_p50_s:.3f} & {b.candidate_decode_p50_s:.3f} & "
                    f"{b.candidate_decode_p95_s:.3f} & {b.paired_latency_ratio:.4f} & "
                    f"[{b.paired_ratio_ci95_low:.4f}, {b.paired_ratio_ci95_high:.4f}]"
                )
        e = runtime[(runtime.task == task) & (runtime.method == "mosaickv_exact")].iloc[
            0
        ]
        data(
            "history_boundaries_" + task,
            {
                "boundary": [1, 2, 3],
                "ordinary": [
                    e.reference_initial_MiB,
                    e.reference_final_MiB,
                    e.reference_peak_MiB,
                ],
                "source": [
                    e.candidate_initial_MiB,
                    e.candidate_final_MiB,
                    e.candidate_peak_MiB,
                ],
            },
        )
    table(
        "history_storage",
        "llrrrrrrr",
        r"Task & Path & Init. & Final KV & Final source & Saving \% & Peak KV & Peak source & Saving \%",
        storage_rows,
    )
    table(
        "history_latency",
        "llrrrrrl",
        r"Task & Path & Source/KV & Ref. p50 & Direct p50 & Direct p95 & Direct/ref. & Paired 95\% CI",
        latency_rows,
    )
    table(
        "history_quality_controls",
        "llrrl",
        r"Task & Path & Model-ref. R-L & Token agreement \% & 128 logits bitwise",
        quality_rows,
    )
    return {
        "source_sha": readiness["source_git_sha"],
        "cpu_tests": readiness["cpu_suite"]["tests"],
        "unique_consumed_inputs": 2,
        "independent_quality_established": False,
        "whole_request_peak_certified": False,
    }


def export_historical_milebench() -> None:
    root = RUNS / "mosaickv/milebench-isoquality-e6-fdeca17-20261020/analysis/larger"
    report = json.loads(
        bind(
            root / "report.json",
            "60a7c44824ae616e1823a73a3234e6e83b5edb32d383ab5aa7da2313d1167a4c",
        ).read_text()
    )
    for path, digest in report["derived_artifact_hashes"].items():
        bind(Path(path), digest)
    assert report["lower_memory_equal_quality_demonstrated"] is False
    aggregates = frame(root / "aggregates.parquet")
    samples = frame(root / "derived_rows.parquet")
    assert len(samples) == 3072 and samples.error.isna().all()
    assert samples.status.eq("completed").all()
    assert not samples.duplicated(["sample_id", "method", "cap_mib"]).any()
    compressed = samples[samples.method != "full_kv"]
    assert (
        compressed.groupby(["sample_id", "cap_mib"])[STORAGE]
        .nunique()
        .eq(1)
        .all()
        .all()
    )
    rows = []
    for cap in [16, 20, 28, 40, 80]:
        for method in ["prompt_attention_topk", "lookm_reimpl", "mosaickv_exact"]:
            selected = aggregates[
                (aggregates.cap_mib == cap) & (aggregates.method == method)
            ]
            accuracy = selected[selected.metric_name.str.endswith("/accuracy")]
            rouge = selected[selected.metric_name.str.endswith("/rouge_l_f1")]
            assert len(accuracy) == len(rouge) == 1
            a, r = accuracy.iloc[0], rouge.iloc[0]
            name = "LOOK-M (R)" if method == "lookm_reimpl" else NAMES[method]
            rows.append(
                f"{cap} & {name} & {100 * a.metric_value:.2f} & "
                f"{100 * r.metric_value:.2f} & {a[STORAGE[1]] / MIB:.3f} & "
                f"{a.end_to_end_time:.2f}"
            )
    table(
        "mile192",
        "rlrrrr",
        r"Initial MiB & Method & Accuracy \% & R-L points & Final/peak MiB & Mean E2E s",
        rows,
    )


def main() -> None:
    (HERE / "generated").mkdir(exist_ok=True)
    plan = read(PREFIX / "plan.json")
    for path, digest in plan["bound_files"].items():
        bind(Path(path), digest)
    metrics = frame(PREFIX / "analysis/final/metrics.parquet")
    paired = frame(PREFIX / "analysis/final/paired_comparisons.parquet")
    samples = frame(PREFIX / "analysis/final/sample_rows.parquet")
    assert len(samples) == 8526 and samples.error.isna().all()
    assert (samples.status == "completed").all()
    assert not samples.duplicated(["task", "lane", "method", "sample_id"]).any()
    for task, n in [("llava_description", 1000), ("mmvet_prefixkv", 218)]:
        sub = samples[samples.task == task]
        assert sub.groupby(["lane", "method"]).size().eq(n).all()
        for lane, group in sub.groupby("lane"):
            assert (
                group.groupby("method")
                .sample_id.apply(set)
                .apply(lambda ids: ids == set(group.sample_id))
                .all()
            )
        shared = sub[(sub.lane == "shared_byte_40mib") & (sub.method != "full_kv")]
        assert shared[STORAGE[0]].eq(40 * MIB).all()
        assert shared.groupby("sample_id")[STORAGE].nunique().eq(1).all().all()
    for lane in ("paper_native", "shared_byte_40mib"):
        lines = []
        for task, label in [
            ("llava_description", "Description"),
            ("mmvet_prefixkv", "MM-Vet route"),
        ]:
            methods = ["full_kv", "prefixkv_reimpl", "mosaickv_exact"]
            if lane == "shared_byte_40mib":
                methods.insert(1, "prompt_attention_topk")
            for method in methods:
                p = row(metrics, task, lane, method, "teacher_forced_ppl")
                q = row(metrics, task, lane, method, "rouge_l_f1")
                lines.append(
                    " & ".join(
                        [
                            label,
                            NAMES[method],
                            f"{p.metric_value:.4f}",
                            f"{100 * q.metric_value:.2f}",
                            f"{p[STORAGE[0]] / MIB:.2f}",
                            f"{p[STORAGE[1]] / MIB:.2f}",
                            f"{p[STORAGE[2]] / MIB:.2f}",
                            "---"
                            if method == "full_kv"
                            else f"{p.mean_end_to_end_seconds:.2f}",
                        ]
                    )
                )
        table(
            "prefix_" + lane,
            "llrrrrrr",
            r"Task & Method & PPL $\downarrow$ & R-L $\uparrow$ & Init. & Final & Peak & E2E s",
            lines,
        )
    lines = []
    for task, label in [
        ("llava_description", "Description"),
        ("mmvet_prefixkv", "MM-Vet route"),
    ]:
        for lane, lane_label in [
            ("paper_native", "Native policies"),
            ("shared_byte_40mib", "40 MiB"),
        ]:
            qr = paired[
                (paired.task == task)
                & (paired.lane == lane)
                & (paired.candidate == "mosaickv_exact")
                & (paired.reference == "prefixkv_reimpl")
            ]
            for metric, metric_label, scale in [
                ("teacher_forced_ppl", "PPL", 1),
                ("rouge_l_f1", "R-L points", 100),
            ]:
                ref = row(metrics, task, lane, "prefixkv_reimpl", metric)
                cand = row(metrics, task, lane, "mosaickv_exact", metric)
                interval = qr[qr.metric_name == metric].iloc[0]
                lines.append(
                    f"{label} & {lane_label} & {metric_label} & "
                    f"{scale * (cand.metric_value - ref.metric_value):+.4f} & "
                    f"[{scale * interval.ci95_low:+.4f}, {scale * interval.ci95_high:+.4f}]"
                )
        for method, short in [
            ("prefixkv_reimpl", "prefix"),
            ("mosaickv_exact", "evicast"),
        ]:
            r = row(metrics, task, "paper_native", method, "teacher_forced_ppl")
            data(
                f"native_{task}_{short}",
                {"boundary": [1, 2, 3], "mib": [r[s] / MIB for s in STORAGE]},
            )
    table(
        "prefix_intervals",
        "lllrr",
        r"Task & Lane & Metric & $\Delta$ Evi--Prefix & Stored 95\% CI",
        lines,
    )

    # Freshness is not claimed for either historical or intermediate cohorts.
    gate = read(INTER / "analysis/GATE.json")
    assert not gate["quality_memory_gate"] and not gate["confirmation_authorized"]
    native = frame(INTER / "analysis/native_metrics.parquet")
    intervals = frame(INTER / "analysis/same_budget_intervals.parquet")
    cross = frame(INTER / "analysis/cross_budget_intervals.parquet")
    bind(INTER / "plan.json")
    lines = []
    for task, label in [
        ("llava_description", "Description"),
        ("mmvet_prefixkv", "MM-Vet route"),
    ]:
        for cap in (80, 120, 160):
            for method in (
                "prompt_attention_topk",
                "prefixkv_reimpl",
                "mosaickv_exact",
            ):
                sub = native[
                    (native.task == task)
                    & (native.cap_mib == cap)
                    & (native.method == method)
                ]
                p = sub[sub.metric == "teacher_forced_ppl"].iloc[0]
                q = sub[sub.metric == "rouge_l_f1"].iloc[0]
                lines.append(
                    f"{label} & {cap} & {NAMES[method]} & {p.value:.4f} & {100 * q.value:.2f}"
                )
        for method, short in [
            ("prompt_attention_topk", "topk"),
            ("prefixkv_reimpl", "prefix"),
            ("mosaickv_exact", "evicast"),
        ]:
            sub = native[(native.task == task) & (native.method == method)]
            data(
                f"intermediate_{task}_{short}",
                {
                    "cap": [80, 120, 160],
                    "rouge": [
                        100
                        * sub[(sub.cap_mib == c) & (sub.metric == "rouge_l_f1")]
                        .iloc[0]
                        .value
                        for c in (80, 120, 160)
                    ],
                    "ppl": [
                        sub[(sub.cap_mib == c) & (sub.metric == "teacher_forced_ppl")]
                        .iloc[0]
                        .value
                        for c in (80, 120, 160)
                    ],
                },
            )
    table(
        "prefix_intermediate",
        "lr lrr",
        r"Task & MiB & Method & PPL & R-L points",
        lines,
    )
    # Preserve all adjusted same-budget intervals in a separate inspectable export.
    intervals.to_csv(HERE / "generated/prefix_same_budget_intervals.csv", index=False)
    cross.to_csv(HERE / "generated/prefix_cross_budget_intervals.csv", index=False)

    claims = read(CLOSURE / "writing/claim_evidence.json")
    for path, digest in claims["bindings"].items():
        bind(Path(path), digest)
    scores = pd.read_csv(bind(CLOSURE / "writing/coco_native_scores.csv"))
    print("fox64 score columns:", scores.columns.tolist())
    differences = pd.read_csv(bind(CLOSURE / "writing/paired_differences.csv"))
    lines = []
    for study in ("r256", "r16"):
        for method, variant, label in [
            ("prompt_attention_topk", "complete", "Prompt Top-K"),
            ("vl_cache_reimpl", "complete", "VL-Cache (R)"),
            ("mosaickv_exact", "complete", "EviCastKV complete"),
            ("mosaickv_exact", "evidence-routing-v1", "Opportunity routing"),
            ("mosaickv_exact", "evidence-coverage-v1", "Demand coverage"),
            ("mosaickv_exact", "evidence-both-v1", "Both changes"),
        ]:
            sub = scores[
                (scores.study == study)
                & (scores.method == method)
                & (scores.variant == variant)
            ]
            vals = [
                float(sub[sub.cap_mib == cap].iloc[0].corpus_cider) for cap in (20, 40)
            ]
            lines.append(f"{study} & {label} & {vals[0]:.2f} & {vals[1]:.2f}")
    table(
        "fox64", "llrr", r"Study & Method/variant & 20 MiB CIDEr & 40 MiB CIDEr", lines
    )
    differences.to_csv(HERE / "generated/fox64_adjusted_intervals.csv", index=False)

    quality = read(FOX / "analysis/installed_guard_development/quality_statistics.json")
    read(FOX / "analysis/installed_guard_development/GATE.json")
    protocol = read(FOX / "protocol_lock.json")
    descript = frame(FOX / "analysis/installed_guard_development/aggregate.parquet")
    lines = []
    for cap in (20, 40):
        for method, variant, label in [
            ("prompt_attention_topk", "complete", "Prompt Top-K"),
            ("vl_cache_reimpl", "complete", "VL-Cache (R)"),
            ("mosaickv_exact", "complete", r"\textbf{EviCastKV complete}"),
            ("mosaickv_exact", "evidence-output-guarded-v2", "EviCastKV + guard"),
        ]:
            s = descript[
                (descript.method == method)
                & (descript.variant == variant)
                & (descript.budget_bytes == cap * MIB)
            ].iloc[0]
            value = quality["scores"][f"{method}/{variant}/{cap * MIB}"]
            lines.append(
                f"{cap} & {label} & {value:.2f} & {s.mean_final_kv_bytes / MIB:.3f} & "
                f"{s.mean_e2e:.2f} & {s.p95_e2e:.2f}"
            )
    table(
        "fox32",
        "rlrrrr",
        r"Initial MiB & Method & CIDEr & Final/peak MiB & Mean s & Empirical p95 s",
        lines,
    )
    lines = []
    for contrast in quality["contrasts"]:
        if "vl_cache_reimpl" not in contrast["reference"]:
            continue
        cap = int(contrast["candidate"].rsplit("/", 1)[-1]) // MIB
        lo, hi = contrast["ordinary_95_ci"]
        alo, ahi = contrast["simultaneous_ci"]
        lines.append(
            f"{cap} & {contrast['difference']:+.2f} & [{lo:+.2f}, {hi:+.2f}] & [{alo:+.2f}, {ahi:+.2f}]"
        )
    table(
        "fox32_intervals",
        "rrrr",
        r"Initial MiB & $\Delta$ guarded Evi--VL & Ordinary 95\% CI & Adjusted CI",
        lines,
    )

    # Completion and artifact-hash checks for the consumed-input optimization.
    compact_gate = read(COMPACT / "analysis/GATE.json")
    assert (
        compact_gate["status"] == "passed_diagnostic_only"
        and not compact_gate["failures"]
    )
    assert (
        not compact_gate["fresh_evaluation_authorized"]
        and not compact_gate["large_campaign_authorized"]
    )
    cm = read(COMPACT / "analysis/manifest.json")
    for path, digest in cm["artifact_sha256"].items():
        bind(COMPACT / "analysis" / path, digest)
    for i in (0, 1):
        result = read(COMPACT / f"workers/{i}/result.json")
        assert all(a["maximum_logit_difference"] == 0 for a in result["completed_arms"])
        parity = read(COMPACT / f"workers/{i}/fullkv_parity.json")
        assert all(
            v["maximum_logit_difference"] == 0 and v["token_agreement"] == 1
            for v in parity.values()
        )
        bind(COMPACT / f"workers/{i}/manifest.json")
    phases = frame(COMPACT / "analysis/representation_timing_pairs.parquet")
    lines = []
    for worker, arm, op, label in [
        (0, "mosaickv_exact", "tier.complete_construction", "EviCastKV tier, 120 MiB"),
        (
            0,
            "prompt_attention_topk",
            "exact.descriptor_gather_and_views",
            "Top-K views, 120 MiB",
        ),
        (1, "complete-20", "tier.complete_construction", "EviCastKV tier, 20 MiB"),
        (1, "complete-40", "tier.complete_construction", "EviCastKV tier, 40 MiB"),
        (
            1,
            "evidence-output-guarded-v2-20",
            "tier.complete_construction",
            "Guarded tier, 20 MiB",
        ),
        (
            1,
            "evidence-output-guarded-v2-40",
            "tier.complete_construction",
            "Guarded tier, 40 MiB",
        ),
    ]:
        r = phases[
            (phases.worker == worker) & (phases.arm == arm) & (phases.operation == op)
        ].iloc[0]
        lines.append(
            f"{label} & {r.reference_seconds:.3f} & {r.compact_seconds:.3f} & {100 * r.reduction_fraction:.1f}\\%"
        )
    table(
        "compact",
        "lrrr",
        r"Profiled operation & Eager s & Compact s & Reduction",
        lines,
    )
    chosen = phases[
        (phases.operation == "tier.complete_construction")
        & phases.arm.isin(["mosaickv_exact", "complete-20", "complete-40"])
    ]
    data(
        "compact_tier",
        {
            "x": [1, 2, 3],
            "reference": chosen.reference_seconds.tolist(),
            "compact": chosen.compact_seconds.tolist(),
        },
    )
    audit = frame(AUDIT / "compression_summary.parquet")
    lines = []
    for task, label in [
        ("llava_description", "Description"),
        ("mmvet_prefixkv", "MM-Vet route"),
    ]:
        for method in ("mosaickv_exact", "prompt_attention_topk", "prefixkv_reimpl"):
            r = audit[
                (audit.task == task) & (audit.method == method) & (audit.cap_mib == 120)
            ].iloc[0]
            lines.append(
                f"{label} & {NAMES[method]} & {r.cache_state_construction:.2f} & "
                f"{r.planner_preparation:.2f} & {r.graph:.2f} & {r.utility_and_selection:.2f} & "
                f"{r.tier_construction:.2f} & {r.cache_packing:.2f} & {r.compression_time:.2f}"
            )
    table(
        "compression",
        "llrrrrrrr",
        r"Task & Method & State & Prep. & Graph & Select & Tier & Pack & Total",
        lines,
    )

    # Larger historical same-byte CIDEr context; not the Fox confirmation cohort.
    coco = read(COCO / "report.json")
    for path, digest in coco["derived_artifact_hashes"].items():
        bind(Path(path), digest)
    c = pd.DataFrame(coco["aggregates"])
    lines = []
    for cap in (10, 20, 40, 80):
        for method in ("prompt_attention_topk", "vl_cache_reimpl", "mosaickv_exact"):
            r = c[(c.method == method) & (c.cap_mib == cap)].iloc[0]
            lines.append(
                f"{cap} & {NAMES[method]} & {r.metric_value:.2f} & {r[STORAGE[1]] / MIB:.3f} & {r.end_to_end_time:.2f}"
            )
    table(
        "coco256",
        "rlrrr",
        r"Initial MiB & Method & CIDEr & Final/peak MiB & Mean E2E s",
        lines,
    )
    for method, short in [
        ("prompt_attention_topk", "topk"),
        ("vl_cache_reimpl", "vl"),
        ("mosaickv_exact", "evicast"),
    ]:
        sub = c[c.method == method]
        data(
            "coco256_" + short,
            {
                "cap": [10, 20, 40, 80],
                "cider": [
                    sub[sub.cap_mib == cap].iloc[0].metric_value
                    for cap in (10, 20, 40, 80)
                ],
            },
        )

    keys = [
        "wang2025prefixkv",
        "tu2025vlcache",
        "wan2024lookm",
        "liu2024improved",
        "lin2014coco",
        "vedantam2015cider",
        "lin2004rouge",
        "song2024milebench",
    ]
    bib = bind(REPO / "paper/references.bib").read_text()
    entries = re.split(r"(?m)(?=^@)", bib)
    selected = [
        e for e in entries if any(re.match(r"@\w+\{" + k + ",", e) for k in keys)
    ]
    if len(selected) != len(keys):
        raise ValueError("missing bibliography record")
    (HERE / "references.bib").write_text("\n".join(selected))
    export_historical_milebench()
    engineering = export_storage_engineering()
    manifest = {
        "scope": "retrospective_report_only_no_inference",
        "system_names_are_identical": ["EviCastKV", "MosaicKV"],
        "raw_artifacts_modified": False,
        "fresh_evaluation_authorized": False,
        "large_campaign_authorized": False,
        "repository_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "report_written_in_user_dirty_worktree": True,
        "source_artifact_sha256": BOUND,
        "producer_shas": {
            "prefix_full": plan["execution_source_sha"],
            "fox_output_guard": protocol["source_sha"],
            "compact_views": cm["source_sha"],
            "source_history_direct": engineering["source_sha"],
        },
        "storage_engineering": engineering,
        "export_script_sha256": sha(Path(__file__)),
        "latex_source_sha256": sha(HERE / "main.tex"),
        "latex_sources_sha256": {
            str(p.relative_to(HERE)): sha(p) for p in HERE.glob("*.tex")
        },
        "derived_sha256": {
            str(p.relative_to(HERE)): sha(p) for p in (HERE / "generated").iterdir()
        },
        "interpretation": "Native-policy, shared-byte, development, and diagnostic results remain separate.",
    }
    (HERE / "provenance.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"Exported {len(manifest['derived_sha256'])} files; bound {len(BOUND)} source artifacts."
    )


if __name__ == "__main__":
    main()
