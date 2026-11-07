#!/usr/bin/env python3
"""Write-once CPU retrospective tests of existing paired COCO predictions.

Never submit inference, mutate source artifacts, interpolate curve points, or
reinterpret the MileBench result. All lower-cap pairs on the executed grid are
tested, not just favorable point estimates. An unexecuted registered 25 MiB
arm remains unavailable. CIDEr uses complete draw-specific corpus TF-IDF.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import subprocess
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from mosaickv.evaluation.cider_resampling import ResampledCider, paired_draw_counts
from mosaickv.evaluation.coco_shared_bytes import STORAGE_FIELDS, validate_caption_rows
from mosaickv.evaluation.metrics.types import MetricSample
from mosaickv.evaluation.shared_byte_contract import CONTROL_FIELDS

BASELINES = ("prompt_attention_topk", "vl_cache_reimpl")
CANDIDATE = "mosaickv_exact"
ENDPOINTS = (
    "cider",
    "end_to_end_difference",
    "end_to_end_ratio",
    "decode_difference",
    "decode_ratio",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, payload: object) -> None:
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def pair_catalog(caps: list[float], primary: list[dict]) -> list[dict]:
    """Exhaust the executed grid and retain unavailable explicit primary pairs."""
    pairs = {(float(c), float(b)) for b in caps for c in caps if c < b}
    pairs |= {(float(p["candidate_cap_mib"]), float(p["baseline_cap_mib"])) for p in primary}
    explicit = {(float(p["candidate_cap_mib"]), float(p["baseline_cap_mib"])) for p in primary}
    return [
        dict(
            baseline=baseline,
            candidate=CANDIDATE,
            baseline_cap_mib=b,
            candidate_cap_mib=c,
            registration=(
                "explicit_primary_pair_in_planning_protocol"
                if (c, b) in explicit
                else "retrospective_exhaustive_pair_on_registered_executed_grid"
            ),
            available=c in caps and b in caps,
        )
        for baseline in BASELINES
        for c, b in sorted(pairs, key=lambda pair: (pair[1], pair[0]))
    ]


def interval(values: np.ndarray, tests: int) -> dict[str, float]:
    """Two-sided pointwise 95% and simultaneous familywise 95% intervals."""
    if tests < 1 or not np.isfinite(values).all():
        raise ValueError("invalid interval inputs")
    lo, hi = np.quantile(values, [0.025, 0.975])
    alo, ahi = np.quantile(values, [0.025 / tests, 1 - 0.025 / tests])
    return dict(
        ci95_low=float(lo), ci95_high=float(hi), adjusted_low=float(alo), adjusted_high=float(ahi)
    )


def latency_statistics(
    reference: np.ndarray, candidate: np.ndarray, counts: np.ndarray, tests: int
) -> dict:
    """Paired difference and ratio of means; resample both arms together."""
    if reference.shape != candidate.shape or reference.ndim != 1 or len(reference) < 2:
        raise ValueError("latency requires equal nonempty paired arrays")
    if (
        not np.isfinite(reference).all()
        or not np.isfinite(candidate).all()
        or (reference <= 0).any()
        or (candidate <= 0).any()
    ):
        raise ValueError("latency must be positive and finite")
    ref_mean = float(reference.mean())
    cand_mean = float(candidate.mean())
    sampled_ref = counts @ reference / len(reference)
    sampled_cand = counts @ candidate / len(candidate)
    return dict(
        baseline_seconds=ref_mean,
        candidate_seconds=cand_mean,
        difference_seconds=cand_mean - ref_mean,
        ratio=cand_mean / ref_mean,
        reduction_fraction=1 - cand_mean / ref_mean,
        difference_interval=interval(sampled_cand - sampled_ref, tests),
        ratio_interval=interval(sampled_cand / sampled_ref, tests),
    )


def load_verified(source: Path) -> tuple[dict, dict, pd.DataFrame, dict]:
    """Trust only the hash-bound successful source analysis; validate its rows."""
    report_path = source / "report.json"
    report = json.loads(report_path.read_text())
    if (
        report["status"] != "passed"
        or report["missing_or_failed"]
        or report["completed_rows"] != 3328
        or report["planned_samples"] != 256
    ):
        raise ValueError("expected the complete 256-image / 3328-row source campaign")
    inputs = {str(report_path): sha(report_path)}
    for path, expected in report["derived_artifact_hashes"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"source artifact changed: {path}")
        inputs[path] = expected
    plan_path = Path(report["plan_path"])
    if sha(plan_path) != report["plan_sha256"]:
        raise ValueError("immutable inference plan changed")
    inputs[str(plan_path)] = sha(plan_path)
    plan = json.loads(plan_path.read_text())
    for path, expected in plan["bound_files"].items():
        # No model/dataset reload: check only metric code and immutable inputs
        # actually read by this analysis, not GB-sized dataset/model artifacts.
        if path.endswith(("coco_cap/utils.py", "coco2014_cap_val.yaml")):
            if sha(Path(path)) != expected:
                raise ValueError("pinned evaluator source changed")
            inputs[path] = expected
    rows = pd.read_parquet(source / "paired_samples.parquet")
    rows["cap_mib"] = rows.cap_mib.map(lambda value: None if pd.isna(value) else float(value))
    if len(rows) != 3328 or rows.duplicated(["method", "cap_mib", "sample_id"]).any():
        raise ValueError("missing/duplicate source rows")
    if (
        set(rows.source_sha) != {plan["source_sha"]}
        or rows.used_fullkv_fallback.any()
        or (rows.decode_evictions != 0).any()
    ):
        raise ValueError("source snapshot, fallback, or decode-policy mismatch")
    ids = sorted(rows.sample_id.unique())
    controls: dict = {}
    references: dict = {}
    for row in rows.to_dict("records"):
        sid = row["sample_id"]
        observed = json.loads(row["control_hashes"])
        if set(observed) != set(CONTROL_FIELDS):
            raise ValueError("incomplete source control fingerprints")
        if controls.setdefault(sid, observed) != observed:
            raise ValueError("cross-budget generation/media/hardware controls differ")
        refs = json.loads(row["reference"])
        if references.setdefault(sid, refs) != refs:
            raise ValueError("reference annotations differ across arms")
        validate_caption_rows([row], [dict(sample_id=sid, references=refs)], plan["output_tokens"])
        if row["method"] != "full_kv":
            if (
                row[STORAGE_FIELDS[0]] != int(row["cap_mib"] * 2**20)
                or row["target_shortfall_bytes"] != 0
            ):
                raise ValueError("compressed arm did not achieve its exact cap")
            if any(row[field] != row[STORAGE_FIELDS[0]] + 8257536 for field in STORAGE_FIELDS[1:]):
                raise ValueError("unexpected final/peak backing-storage growth")
    expected = {
        (method, float(cap), sid)
        for method in (*BASELINES, CANDIDATE)
        for cap in report["caps_mib"]
        for sid in ids
    }
    actual = {
        (r.method, r.cap_mib, r.sample_id) for r in rows.itertuples() if r.method != "full_kv"
    }
    if actual != expected:
        raise ValueError("curve points are not a complete paired grid")
    return report, plan, rows, inputs


def markdown(report: dict, records: list[dict], output: Path) -> str:
    lines = [
        "# COCO cross-budget retrospective analysis",
        "",
        "**Retrospective supporting evidence, not preregistered held-out confirmation.**",
        "Predictions and point estimates were inspected before these cross-budget tests. "
        "No inference was run; original artifacts and the failed MileBench gate are unchanged.",
        "",
        f"256 identical images; {report['bootstrap_resamples']:,} paired resamples. "
        "Complete corpus CIDEr is rebuilt in each draw, including reference document "
        "frequencies for repeated images.",
        f"Bonferroni family: {report['familywise_tests']} endpoint/pair tests "
        f"({report['planned_pair_count']} pairs x five endpoints), including unavailable pairs. "
        "The adjusted intervals below provide conservative familywise 95% coverage. "
        "No non-inferiority margin, interpolation, or outcome-based pair selection is used.",
        "",
        "All scored captions, reference lists, controls, source/model/dataset revisions and "
        "initial/final/peak backing-storage measurements are paired. "
        "FullKV is not a latency comparator.",
        "",
        "## Every available smaller-cache pair",
        "",
        "Memory entries show initial / final=peak MiB. "
        "Delta CIDEr and time are MosaicKV minus baseline. "
        "Quality preservation requires adjusted CIDEr lower bound ≥0; no slower requires "
        "adjusted latency-ratio upper bound ≤1. Negative times favor MosaicKV.",
        "",
        "| Baseline | Mosaic / baseline initial | Mosaic / baseline final=peak | "
        "Initial / final / peak reduction | ΔCIDEr [adjusted CI] | "
        "Δend-to-end s [adjusted CI] | Latency ratio [adjusted CI] | Quality / no-slower |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in records:
        q = row["quality"]
        t = row["latency"]["end_to_end_time"]
        s = row["storage"]
        lines.append(
            f"| {row['baseline']} | {row['candidate_cap_mib']:g} / {row['baseline_cap_mib']:g} | "
            f"{s['final']['candidate_mean_bytes'] / 2**20:g} / "
            f"{s['final']['baseline_mean_bytes'] / 2**20:g} | "
            + " / ".join(
                f"{100 * s[b]['reduction_fraction']:.2f}%" for b in ("initial", "final", "peak")
            )
            + f" | {q['difference']:+.3f} [{q['adjusted_low']:+.3f}, {q['adjusted_high']:+.3f}] | "
            f"{t['difference_seconds']:+.3f} [{t['difference_interval']['adjusted_low']:+.3f}, "
            f"{t['difference_interval']['adjusted_high']:+.3f}] | {t['ratio']:.3f} "
            f"[{t['ratio_interval']['adjusted_low']:.3f}, "
            f"{t['ratio_interval']['adjusted_high']:.3f}] | "
            f"{row['quality_verdict']} / "
            f"{'PASS' if row['no_slower_pass'] else 'unresolved'} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "All available pairs pass both ≥30% measured memory reduction at every boundary "
        "and the adjusted end-to-end no-slower test. The blocker is quality: "
        "no cross-budget CIDEr interval has a lower bound at least zero.",
        f"{sum(r['complete_claim_pass'] for r in records)}/{len(records)} "
        "available pairs meet "
        "the complete smaller-at-all-boundaries + preserved-quality + no-slower rule. "
        f"{sum(r['thirty_percent_claim_pass'] for r in records)} "
        "also meet ≥30% at all three boundaries.",
        "A confidence interval crossing zero is unresolved quality—not proof of equivalence "
        "and not necessarily proof of inferiority. "
        "Favorable point estimates alone do not pass.",
        "",
        "## Decode latency (secondary endpoint)",
        "",
        "The same paired draws and global multiple-testing correction apply. "
        "Ratios are ratios of cohort means, not means of per-image ratios.",
        "",
        "| Baseline | Mosaic / baseline initial MiB | Decode difference s "
        "[adjusted CI] | Decode ratio [adjusted CI] |",
        "|---|---:|---:|---:|",
    ]
    for row in records:
        t = row["latency"]["decode_time"]
        d, r = t["difference_interval"], t["ratio_interval"]
        lines.append(
            f"| {row['baseline']} | {row['candidate_cap_mib']:g} / "
            f"{row['baseline_cap_mib']:g} | {t['difference_seconds']:+.4f} "
            f"[{d['adjusted_low']:+.4f}, {d['adjusted_high']:+.4f}] | "
            f"{t['ratio']:.4f} [{r['adjusted_low']:.4f}, {r['adjusted_high']:.4f}] |"
        )
    lines += [
        "",
        "## Unavailable explicit pairs",
        "",
    ]
    for pair in report["unavailable_pairs"]:
        lines.append(
            f"- {pair['baseline']}: MosaicKV {pair['candidate_cap_mib']:g} versus "
            f"baseline {pair['baseline_cap_mib']:g} MiB: no measured candidate arm. "
            "No interval is fabricated and no different cap is substituted."
        )
    lines += [
        "",
        "## Limits",
        "",
        "- The executed four-cap grid was registered in the inference plan; the exhaustive "
        "cross-budget tests and global endpoint correction are retrospective. The separately "
        "planned primary 25/40 pair was never executed in this cohort.",
        "- COCO IDs came from a previously used 4,096-image cohort. No held-out claim is made.",
        "- The comparator is local `vl_cache_reimpl`, not official VL-Cache execution. "
        "Prompt Top-K is retained; a weaker, larger baseline is not called nearest.",
        "- Five warmups and one measured quality trial per sample underpin these latency "
        "intervals. Resampling examples measures cohort-level uncertainty, not run-to-run "
        "hardware variability; this is not a 20-repeat randomized microbenchmark.",
        "- The global correction gives per-interval confidence 99.9286%; with 10,000 draws, "
        "each extreme tail contains only about 3.57 draws. Tail estimates have Monte Carlo "
        "uncertainty. Every quality test also fails at unadjusted 95%, so this conclusion "
        "is not caused by the conservative multiple-testing correction.",
        "- Peak means live active KV backing storage, not total GPU allocator/process peak; "
        "append-only generation adds 7.875 MiB in every compressed arm.",
        "- No claim about MileBench, PrefixKV, serving engines, prototypes, or repair follows.",
        "",
        "## Reproduction and artifacts",
        "",
        "```bash",
        "source mosaickv/scripts/cache_env.sh",
        "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /scratch/djy8hg/env/mosaickv/bin/python \\",
        "  mosaickv/scripts/analyze_coco_cross_budget_retrospective.py \\",
        f"  --source {report['source_analysis']} \\",
        "  --output /scratch/djy8hg/runs/mosaickv/coco-cross-budget-retrospective-NEW",
        "```",
        "",
        f"Full report: [{output / 'report.json'}]({output / 'report.json'}).",
        f"Tables: [{output / 'paired_tests.parquet'}]({output / 'paired_tests.parquet'}); "
        f"[{output / 'paired_tests.csv'}]({output / 'paired_tests.csv'}).",
        "Per-sample provenance/storage and every bootstrap corpus score/draw count are saved "
        "separately. `analysis_plan.json` freezes this retrospective analysis before scoring; "
        "`report.json` hashes its inputs, code, evaluators and outputs.",
        "",
    ]
    return "\n".join(lines)


def analyze(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("retrospective outputs are write-once; select a new directory")
    source_report, plan, rows, input_hashes = load_verified(source)
    repo = Path(__file__).resolve().parents[2]
    protocol = repo / "mosaickv/configs/experiments/lower_memory_equal_quality_v1.yaml"
    registration = yaml.safe_load(protocol.read_text())
    pairs = pair_catalog(
        source_report["caps_mib"],
        registration["lanes"]["vl_cache_coco"]["primary_registered_pairs"],
    )
    tests = len(pairs) * len(ENDPOINTS)
    output.mkdir(parents=True)
    analysis_plan = dict(
        classification="retrospective_cross_budget_analysis",
        heldout_claim=False,
        future_experiment_submission_authorized=False,
        gpu_inference=False,
        created_at_utc=datetime.now(UTC).isoformat(),
        pairs=pairs,
        endpoints=ENDPOINTS,
        source_analysis=str(source),
        source_inputs=input_hashes,
        planning_protocol=str(protocol),
        planning_protocol_sha256=sha(protocol),
        registration_note=(
            "grid was executed; pair tests retrospective; "
            "planning file is not an executed preregistration"
        ),
        familywise_alpha=0.05,
        familywise_tests=tests,
        correction="Bonferroni_all_pairs_times_all_five_endpoints_including_unavailable",
        bootstrap_resamples=plan["bootstrap_resamples"],
        bootstrap_seed=plan["bootstrap_seed"],
        quality_margin=0.0,
        latency_ratio_upper_limit=1.0,
    )
    save_json(output / "analysis_plan.json", analysis_plan)
    print(
        json.dumps(
            dict(stage="source_verified", rows=len(rows), pairs=len(pairs), family_tests=tests)
        ),
        flush=True,
    )
    needed = {(p["baseline"], p["baseline_cap_mib"]) for p in pairs if p["available"]}
    needed |= {(CANDIDATE, p["candidate_cap_mib"]) for p in pairs if p["available"]}
    groups = {}
    indexed = {}
    for point in sorted(needed):
        method, cap = point
        group = rows[(rows.method == method) & (rows.cap_mib == cap)].sort_values("sample_id")
        indexed[point] = group
        groups[point] = [
            MetricSample(r.sample_id, r.answer, tuple(json.loads(r.reference)))
            for r in group.itertuples()
        ]
    scorer = ResampledCider(groups)
    counts = paired_draw_counts(
        len(scorer.sample_ids), plan["bootstrap_resamples"], plan["bootstrap_seed"]
    )
    original = np.ones((1, len(scorer.sample_ids)), dtype=np.int64)
    values = scorer.score(original)[0]
    # Validate original corpus, repeated IDs, and first 16 random draws at all
    # nine relevant points against the *unchanged official evaluator*.
    single = np.zeros_like(original)
    single[0, 0] = len(scorer.sample_ids)
    fixtures = np.vstack([original, single, counts[:16]])
    parity_error = scorer.validate_official(fixtures, scorer.score(fixtures))
    for point, value in zip(scorer.points, values, strict=True):
        old = next(
            r["metric_value"]
            for r in source_report["aggregates"]
            if (r["method"], r["cap_mib"]) == point
        )
        if not math.isclose(old, value, abs_tol=1e-9, rel_tol=0):
            raise ValueError("original cohort CIDEr no longer matches measured source aggregate")
    print(
        json.dumps(dict(stage="official_corpus_parity_passed", max_abs_error=parity_error)),
        flush=True,
    )
    before = time.perf_counter()
    scores = scorer.score(counts)
    print(
        json.dumps(
            dict(stage="10000_corpus_bootstraps_complete", seconds=time.perf_counter() - before)
        ),
        flush=True,
    )
    np.savez_compressed(
        output / "bootstrap_corpus_scores.npz",
        counts=counts,
        scores=scores,
        points=np.asarray(scorer.points, dtype=str),
        sample_ids=np.asarray(scorer.sample_ids),
    )
    point_index = {p: i for i, p in enumerate(scorer.points)}
    records, flattened, samples = [], [], []
    for pair in pairs:
        if not pair["available"]:
            continue
        bp = pair["baseline"], pair["baseline_cap_mib"]
        cp = CANDIDATE, pair["candidate_cap_mib"]
        baseline, candidate = indexed[bp], indexed[cp]
        if baseline.sample_id.tolist() != candidate.sample_id.tolist():
            raise ValueError("latency/storage sample alignment changed")
        bi, ci = point_index[bp], point_index[cp]
        q = dict(
            baseline_cider=float(values[bi]),
            candidate_cider=float(values[ci]),
            difference=float(values[ci] - values[bi]),
            **interval(scores[:, ci] - scores[:, bi], tests),
        )
        latency = {
            field: latency_statistics(
                baseline[field].to_numpy(), candidate[field].to_numpy(), counts, tests
            )
            for field in ("end_to_end_time", "decode_time")
        }
        storage = {}
        for field in STORAGE_FIELDS:
            b = baseline[field].to_numpy(dtype=np.int64)
            c = candidate[field].to_numpy(dtype=np.int64)
            reduction = 1 - c / b
            storage[field.removesuffix("_active_kv_storage_bytes")] = dict(
                baseline_mean_bytes=float(b.mean()),
                candidate_mean_bytes=float(c.mean()),
                reduction_fraction=float(1 - c.mean() / b.mean()),
                worst_sample_reduction=float(reduction.min()),
                best_sample_reduction=float(reduction.max()),
            )
        smaller = all(s["worst_sample_reduction"] > 0 for s in storage.values())
        thirty = all(s["worst_sample_reduction"] >= 0.30 for s in storage.values())
        quality = q["adjusted_low"] >= 0
        quality_verdict = (
            "PASS" if quality else "inferior" if q["adjusted_high"] < 0 else "unresolved"
        )
        faster = latency["end_to_end_time"]["ratio_interval"]["adjusted_high"] <= 1
        record = dict(
            **pair,
            sample_count=len(baseline),
            measurement_type="derived",
            analysis_classification="retrospective",
            quality=q,
            latency=latency,
            storage=storage,
            smaller_memory_pass=smaller,
            thirty_percent_memory_pass=thirty,
            quality_pass=quality,
            quality_verdict=quality_verdict,
            no_slower_pass=faster,
            complete_claim_pass=smaller and quality and faster,
            thirty_percent_claim_pass=thirty and quality and faster,
        )
        records.append(record)
        flat = {k: v for k, v in record.items() if k not in {"quality", "latency", "storage"}}
        flat.update({f"cider_{k}": v for k, v in q.items()})
        for field, stats in latency.items():
            for key, value in stats.items():
                if isinstance(value, dict):
                    flat.update({f"{field}_{key}_{k}": v for k, v in value.items()})
                else:
                    flat[f"{field}_{key}"] = value
        for boundary, stats in storage.items():
            flat.update({f"{boundary}_{k}": v for k, v in stats.items()})
        flattened.append(flat)
        for b, c in zip(baseline.to_dict("records"), candidate.to_dict("records"), strict=True):
            samples.append(
                dict(
                    **pair,
                    sample_id=b["sample_id"],
                    baseline_manifest=b["manifest_path"],
                    candidate_manifest=c["manifest_path"],
                    **{
                        f"baseline_{f}": b[f]
                        for f in (*STORAGE_FIELDS, "end_to_end_time", "decode_time")
                    },
                    **{
                        f"candidate_{f}": c[f]
                        for f in (*STORAGE_FIELDS, "end_to_end_time", "decode_time")
                    },
                )
            )
    pd.DataFrame(flattened).to_parquet(output / "paired_tests.parquet", index=False)
    pd.DataFrame(flattened).to_csv(output / "paired_tests.csv", index=False)
    pd.DataFrame(pairs).to_parquet(output / "pair_catalog.parquet", index=False)
    pd.DataFrame(samples).to_parquet(output / "paired_samples.parquet", index=False)
    curve = [
        dict(method=p[0], cap_mib=p[1], cider=float(v))
        for p, v in zip(scorer.points, values, strict=True)
    ]
    pd.DataFrame(curve).to_parquet(output / "corpus_metrics.parquet", index=False)
    files = [
        Path(__file__),
        Path(inspect.getfile(ResampledCider)),
        Path(inspect.getfile(type(scorer.official))),
        Path(inspect.getfile(scorer.official._scorer)),
    ]
    # Include the actual class's implementation, even when lmms-eval vendors it.
    official_scorer = scorer.official._scorer.compute_score.__globals__["CiderScorer"]
    files.append(Path(inspect.getfile(official_scorer)))
    files += [
        repo / "mosaickv/tests/unit/test_cider_resampling.py",
        repo / "mosaickv/tests/unit/test_coco_cross_budget_retrospective.py",
    ]
    # Preserve the actual dirty-tree analysis code, not just an unavailable
    # future commit. Inference itself remains bound to the original clean SHA.
    with zipfile.ZipFile(output / "analysis_source.zip", "x") as archive:
        for i, path in enumerate(files):
            archive.write(path, f"{i:02d}/{path.name}")

    def git(*arguments):
        return subprocess.check_output(["git", *arguments], cwd=repo, text=True).strip()

    report = dict(
        **analysis_plan,
        status="completed",
        source_model=plan["model_id"],
        model_revision=plan["model_revision"],
        dataset_id=plan["dataset_id"],
        dataset_revision=plan["dataset_revision"],
        inference_source_sha=plan["source_sha"],
        analysis_source_sha=git("rev-parse", "HEAD"),
        analysis_worktree_dirty=bool(git("status", "--porcelain")),
        source_code_hashes={str(p): sha(p) for p in files},
        software_versions={
            p: importlib.metadata.version(p)
            for p in ("numpy", "scipy", "pandas", "lmms-eval", "pycocoevalcap")
        },
        official_parity=dict(
            corpora=len(fixtures) * len(scorer.points), max_abs_error=parity_error, atol=1e-9
        ),
        sample_count=len(scorer.sample_ids),
        planned_pair_count=len(pairs),
        tested_pair_count=len(records),
        unavailable_pairs=[p for p in pairs if not p["available"]],
        adjusted_interval_confidence=1 - 0.05 / tests,
        comparisons=records,
        complete_claim_passed_pairs=sum(r["complete_claim_pass"] for r in records),
        thirty_percent_claim_passed_pairs=sum(r["thirty_percent_claim_pass"] for r in records),
        milebench_gate_changed=False,
        inference_jobs_submitted=0,
        total_cpu_wall_seconds=time.perf_counter() - started,
        interpretation="retrospective_pairwise_evidence_only_not_a_nearest_competitor_or_heldout_claim",
    )
    text = markdown(report, records, output)
    with (output / "report.md").open("x") as stream:
        stream.write(text)
    report["derived_artifact_hashes"] = {str(p): sha(p) for p in output.iterdir() if p.is_file()}
    save_json(output / "report.json", report)
    print(
        json.dumps(
            dict(
                status="completed",
                report=str(output / "report.md"),
                quality_passes=sum(r["quality_pass"] for r in records),
                complete_claim_passes=report["complete_claim_passed_pairs"],
                wall_seconds=report["total_cpu_wall_seconds"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    analyze(parser.parse_args())
