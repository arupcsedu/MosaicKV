#!/usr/bin/env python3
"""Reproduce negative development evidence and assess COCO confirmation feasibility.

No algorithm, evaluator, tolerance, or benchmark outcome is changed. Freshness
is not manufactured from an ID-only inventory. CPU callbacks include PrefixKV
when available and follow its finite timing successor without polling logs.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import pandas as pd
from evicastkv_analyze import collect
from evicastkv_opportunity_development import (
    STAGE,
    analyze,
    load_analysis_lock,
    validate_rows,
)

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
BASE = Path("/scratch/djy8hg/runs/evicastkv")
STUDIES = {
    "r256": BASE / "fox-review-20261028-opportunity-development-reanalysis-v1",
    "r16": BASE / "fox-review-20261028-opportunity-followup-r16-recovery-v1/r16-development",
}
PREFIX = Path("/scratch/djy8hg/runs/mosaickv/prefix-isoquality-development-20261028-retry1")
POOL = Path("/scratch/djy8hg/runs/mosaickv/coco-shared-pool-edb6edf-retry1-20261014/ledger.json")
HISTORY = BASE / "fox-review-20261028-opportunity-history-refresh-v1"
COCO_CACHE = Path(
    "/scratch/djy8hg/cache/mosaickv/datasets/lmms-lab___coco-caption/default/0.0.0/"
    "c3213b54e173ae86517d481de17f31439d6f8c72"
)


def read(path):
    return json.loads(Path(path).read_text())


def strict_positive(lower):
    return lower > 0


def summarize_activity(activity):
    grouped = defaultdict(list)
    for row in activity:
        if row.get("policy"):
            grouped[row["cap"], row["variant"]].append(row["policy"])
    result = []
    for (cap, variant), policies in sorted(grouped.items()):
        routes = [r for p in policies for r in p.get("query_guard", {}).get("routes", [])]
        result.append(
            dict(
                cap_mib=cap,
                variant=variant,
                inputs=len(policies),
                grouped_query_heads_averaged=all(
                    p["grouped_query_heads_averaged"] for p in policies
                ),
                individual_prompt_positions=all(p["individual_prompt_positions"] for p in policies),
                guarded_routes=len(routes),
                rolled_back_routes=sum(r["rolled_back"] for r in routes),
                floor_is_a_query_mass_constraint_not_a_quality_guarantee=True,
                mean_evidence_preparation_seconds=sum(
                    p["evidence_preparation_seconds"] for p in policies
                )
                / len(policies),
            )
        )
    return result


def reproduce(root, label):
    parent = STUDIES[label]
    lock = load_analysis_lock(parent)
    spec, workers, rows, errors = collect(parent, STAGE)
    if errors or len(workers) != 16:
        raise ValueError("complete 16-worker development study required")
    validate_rows(rows, lock["cohorts"]["coco"]["development"]["sample_ids"])
    target = root / "reproduced" / label
    write_once(target / "protocol_lock.json", lock)
    revised = copy.deepcopy(spec)
    revised["protocol_sha256"] = sha(target / "protocol_lock.json")
    revised["imported_workers"] = worker_bindings(spec, parent, lock["source_sha"])
    write_once(target / f"jobs/{STAGE}.json", revised)
    analyze(target, recorded_source=True)
    old = read(parent / f"analysis/{STAGE}/quality_statistics.json")
    new = read(target / f"analysis/{STAGE}/quality_statistics.json")
    if old != new:
        raise ValueError("independent regeneration changed registered CIDEr/bootstrap statistics")
    write_once(
        target / "reproduction_gate.json",
        dict(
            status="passed",
            producer_source_sha=lock["source_sha"],
            analysis_source_sha=clean_source(SOURCE),
            original_statistics_sha256=sha(parent / f"analysis/{STAGE}/quality_statistics.json"),
            regenerated_statistics_sha256=sha(target / f"analysis/{STAGE}/quality_statistics.json"),
            numerically_identical=True,
            new_gpu_requests=0,
            rows=832,
            unique_inputs=64,
        ),
    )


def projected_ids(path):
    """Memory-map IPC buffers; convert only ID columns, never images/answers."""
    import pyarrow as pa

    result = []
    with pa.memory_map(str(path), "r") as source:
        reader = pa.ipc.open_stream(source)
        required = ["question_id", "file_name", "id"]
        if not set(required) <= set(reader.schema.names):
            raise ValueError("pinned COCO ID schema changed")
        for batch in reader:
            result.extend(batch.select(required).to_pylist())
    return result


def worker_bindings(spec, parent, producer_sha):
    """A prior derived analysis has no GPU outputs; retain its actual imports."""
    return {
        str(i): copy.deepcopy(
            spec.get("imported_workers", {}).get(
                str(i), dict(parent_root=str(parent), source_sha=producer_sha)
            )
        )
        for i in range(16)
    }


def coco_identity(record):
    filename = record["file_name"]
    match = re.fullmatch(r"COCO_val2014_(\d{12})\.jpg", filename)
    # The release's auxiliary `id` is a dataset row index, NOT the numeric
    # COCO image ID. Native evaluation keys are question_id/file_name.
    if not match or record["question_id"] != filename:
        raise ValueError("COCO ID/file/source-image fields disagree")
    return filename


def freshness_gate(*, available_ids, unresolved_roots, global_media_complete):
    # Availability never certifies source/media independence on its own.
    return bool(available_ids >= 256 and not unresolved_roots and global_media_complete)


def interval_plot(comparisons, path):
    """Use the existing dependency-free vector renderer, not a new package."""
    from analyze_component_ablations import _PdfCanvas

    chosen = [c for c in comparisons if c["reference"].startswith("prompt_attention_topk/")]
    if not chosen:
        raise ValueError("missing registered Top-K contrasts")
    minimum = min(0.0, *(c["low"] for c in chosen))
    maximum = max(0.0, *(c["high"] for c in chosen))
    padding = max(0.1, (maximum - minimum) * 0.08)
    minimum -= padding
    maximum += padding

    def x(value):
        return 270 + 490 * (value - minimum) / (maximum - minimum)

    canvas = _PdfCanvas(810, 500)
    canvas.text(25, 474, "Same-budget CIDEr differences against Prompt Top-K", size=13)
    canvas.text(
        25, 455, "Adjusted 95% intervals; same 64 consumed inputs; not confirmation", size=10
    )
    canvas.line(x(0), 70, x(0), 433, color=(0.5, 0.5, 0.5))
    for tick in range(6):
        value = minimum + (maximum - minimum) * tick / 5
        canvas.line(x(value), 70, x(value), 65)
        canvas.text(x(value) - 12, 50, f"{value:.2f}")
    for i, c in enumerate(chosen):
        _method, variant, budget = c["candidate"].split("/")
        label = variant.removeprefix("evidence-").removesuffix("-v1")
        y = 420 - i * 29
        color = (0.12, 0.47, 0.71) if c["study"] == "r256" else (0.95, 0.45, 0.05)
        canvas.text(25, y - 3, f"{c['study']} / {label} / {budget} MiB", size=10)
        canvas.line(x(c["low"]), y, x(c["high"]), y, color=color, width=1.5)
        for end in (c["low"], c["high"]):
            canvas.line(x(end), y - 3, x(end), y + 3, color=color)
        canvas.marker(x(c["difference"]), y, color=color)
    canvas.line(270, 70, 760, 70)
    canvas.text(290, 25, "CIDEr difference (score points); zero is not strict superiority", size=10)
    canvas.save(path)


def audit(root):
    revision = clean_source(SOURCE)
    output = root / "coco_availability"
    pool = read(POOL)
    reports = [read(HISTORY / f"{name}-inventory.json") for name in ("mosaickv", "evicastkv")]
    used = {sid for report in reports for sid in report["COCO_consumed_ids"]}
    used.update(s["sample_id"].split(":", 1)[-1] for s in pool["samples"])
    bound = {str(POOL): sha(POOL)}
    bound.update(
        {
            str(HISTORY / f"{n}-inventory.json"): sha(HISTORY / f"{n}-inventory.json")
            for n in ("mosaickv", "evicastkv")
        }
    )
    all_ids = []
    for i in range(14):
        path = COCO_CACHE / f"coco-caption-val-{i:05d}-of-00014.arrow"
        expected = pool["bound_files"].get(str(path.resolve()))
        if expected is None:
            raise ValueError("pinned Arrow shard lacks a recorded content hash")
        if sha(path) != expected:
            raise ValueError("pinned Arrow shard changed")
        bound[str(path)] = expected
        all_ids.extend(coco_identity(record) for record in projected_ids(path))
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("duplicate source COCO image IDs")
    available = sorted(set(all_ids) - used)
    review = read(HISTORY / "review.json")
    unresolved = list(review["unresolved_roots"])
    classifications = []
    # Resolve the old COCO launch alias from its sealed original execution plan,
    # not from directory spelling or assumptions about what was run.
    alias = Path("/scratch/djy8hg/runs/mosaickv/coco-cublas-retry-edb6edf-20261014")
    seal = read(alias / "seal.json")
    for path, expected in seal["bound_files"].items():
        if sha(Path(path)) != expected:
            raise ValueError("historical launch seal changed")
        bound[path] = expected
    bound[str(alias / "seal.json")] = sha(alias / "seal.json")
    if seal["ledger_reused"] is not True or str(POOL) not in seal["bound_files"]:
        raise ValueError("old launch alias lacks immutable parent-pool evidence")
    classifications.append(
        dict(
            root=str(alias),
            classification="COCO retry of already excluded pool",
            evidence=str(alias / "seal.json"),
        )
    )
    unresolved = [p for p in unresolved if Path(p).resolve() != alias.resolve()]
    # Saved image-audit outputs contain no model inference; their negative media
    # certification is retained, not converted into a positive freshness claim.
    media_root = BASE / "fox-review-20261028-milebench-media-content-v1"
    media_report = read(media_root / "report.json")
    if media_report["no_images_extracted_or_downloaded"] is not True:
        raise ValueError("unexpected historical media-audit behavior")
    bound[str(media_root / "report.json")] = sha(media_root / "report.json")
    classifications.append(
        dict(
            root=str(media_root),
            classification="MileBench media-only diagnostic",
            evidence=str(media_root / "report.json"),
        )
    )
    unresolved = [p for p in unresolved if Path(p).resolve() != media_root.resolve()]
    extract_root = Path("/scratch/djy8hg/runs/mosaickv/milebench-actionprediction-extract-20261018")
    command = subprocess.check_output(
        ["sacct", "-j", "21564774", "-X", "-n", "-P", "--format=JobID,SubmitLine%4000"],
        text=True,
        timeout=30,
    ).strip()
    if "tar -xzf" not in command or "./ActionPrediction/*" not in command or "python" in command:
        raise ValueError("original extraction command is unavailable or not extraction-only")
    write_once(output / "extraction_command.json", dict(job="21564774", accounting_command=command))
    classifications.append(
        dict(
            root=str(extract_root),
            classification="tar extraction only; no inference",
            evidence=str(output / "extraction_command.json"),
        )
    )
    unresolved = [p for p in unresolved if Path(p).resolve() != extract_root.resolve()]
    # The bounded manifest inventory explicitly does not certify exhaustive
    # historical/source-media coverage. Resolving three directory aliases does
    # not license flipping that separate scientific control to true.
    global_media_complete = review["exhaustive_historical_media_inventory_complete"]
    eligible = freshness_gate(
        available_ids=len(available),
        unresolved_roots=unresolved,
        global_media_complete=global_media_complete,
    )
    write_once(
        output / "report.json",
        dict(
            status="eligible_for_frozen_confirmation"
            if eligible
            else "blocked_global_media_provenance",
            source_sha=revision,
            dataset_id=pool["dataset_id"],
            dataset_revision=pool["dataset_revision"],
            split=pool["split"],
            source_id_columns=["question_id", "file_name", "id"],
            identity_key="question_id_equal_to_canonical_COCO_file_name",
            auxiliary_id_not_used_as_source_image_id=True,
            validation_image_ids=len(all_ids),
            recorded_consumed_coco_ids=len(used),
            id_only_available=len(available),
            original_4096_pool_available=len(
                set(s["sample_id"].split(":", 1)[-1] for s in pool["samples"]) - used
            ),
            classifications=classifications,
            unresolved_workload_roots=unresolved,
            exhaustive_historical_media_inventory_complete=global_media_complete,
            confirmation_authorized=eligible,
            materialization_executed=False,
            responses_images_predictions_decoded=0,
            cohort_selected=False,
            no_absence_of_detected_reuse_claim=True,
            bound_files=bound,
        ),
    )


def packet(root):
    revision = clean_source(SOURCE)
    if read(root / "cpu-tests.json")["status"] != "passed":
        raise ValueError("complete CPU unit suite must pass before closing the packet")
    corrective_tests = root / "verification/cpu-tests.json"
    if (root / "analysis_reuse.json").exists() and (
        not corrective_tests.exists() or read(corrective_tests)["status"] != "passed"
    ):
        raise ValueError("corrected report requires its own clean-source CPU verification")
    all_scores, comparisons, activities, descriptive = [], [], [], []
    evidence = {}
    for label, parent in STUDIES.items():
        base = parent / f"analysis/{STAGE}"
        regenerated = root / f"reproduced/{label}/reproduction_gate.json"
        if not regenerated.exists() or read(regenerated)["status"] != "passed":
            raise ValueError("both independent statistical regenerations must pass")
        gate = read(base / "GATE_evidence_development.json")
        if gate["actual_rows"] != 832 or gate["status"] != "passed_execution_development_only":
            raise ValueError("incomplete source study")
        statistics = read(base / "quality_statistics.json")
        for key, value in statistics["scores"].items():
            method, variant, budget = key.split("/")
            all_scores.append(
                dict(
                    study=label,
                    method=method,
                    variant=variant,
                    cap_mib=None if budget == "None" else int(budget),
                    corpus_cider=value,
                    unique_inputs=64,
                    scope="consumed_development_not_confirmation",
                )
            )
        for c in statistics["contrasts"]:
            comparisons.append(
                dict(
                    study=label,
                    candidate=c["candidate"],
                    reference=c["reference"],
                    difference=c["difference"],
                    low=c["simultaneous_ci"][0],
                    high=c["simultaneous_ci"][1],
                    family_size=c["family_size"],
                    strictly_positive=strict_positive(c["simultaneous_ci"][0]),
                )
            )
        activities.extend(
            dict(study=label, **a)
            for a in summarize_activity(read(base / "configuration_activity.json"))
        )
        descriptive.extend(dict(study=label, **r) for r in read(base / "descriptive.json"))
        write_once(
            root / f"seals/{label}.json",
            dict(
                status="negative_or_inconclusive_development",
                source_gate_sha256=sha(base / "GATE_evidence_development.json"),
                statistics_sha256=sha(base / "quality_statistics.json"),
                unique_inputs=64,
                identities_consumed=True,
                new_confirmation_or_gpu_campaign_authorized=False,
                stricter_positive_lower_bound_required=True,
            ),
        )
        for filename in (
            "GATE_evidence_development.json",
            "quality_statistics.json",
            "analysis_manifest.json",
            "descriptive.json",
        ):
            evidence[str(base / filename)] = sha(base / filename)
    out = root / "writing"
    out.mkdir(parents=True, exist_ok=False)
    scores = pd.DataFrame(all_scores)
    scores.to_parquet(out / "coco_native_scores.parquet", index=False)
    pd.DataFrame(comparisons).to_parquet(out / "paired_differences.parquet", index=False)
    pd.DataFrame(activities).to_parquet(out / "mechanism_activity.parquet", index=False)
    pd.DataFrame(descriptive).to_parquet(out / "descriptive_storage_timing.parquet", index=False)
    for name, rows in (
        ("coco_native_scores", all_scores),
        ("paired_differences", comparisons),
        ("mechanism_activity", activities),
    ):
        pd.DataFrame(rows).to_csv(out / f"{name}.csv", index=False)
    # Complete matrices, including every null/negative contrast, are retained.
    scores.to_latex(
        out / "development_quality.tex", index=False, float_format="%.4f", na_rep="--", escape=True
    )
    pd.DataFrame(comparisons).to_latex(
        out / "development_differences.tex", index=False, float_format="%.4f", escape=True
    )
    lines = [
        r"\documentclass{standalone}",
        r"\usepackage{pgfplots}",
        r"\pgfplotsset{compat=1.18}",
        r"\begin{document}",
        r"\begin{tikzpicture}",
        r"\begin{axis}[xlabel={Initial active KV (MiB)},ylabel={Corpus CIDEr},"
        r"legend pos=outer north east]",
    ]
    for (study, method, variant), group in scores[scores.cap_mib.notna()].groupby(
        ["study", "method", "variant"], sort=True
    ):
        points = " ".join(
            f"({r.cap_mib:g},{r.corpus_cider:.9g})"
            for r in group.sort_values("cap_mib").itertuples()
        )
        label = f"{study}/{method}/{variant}".replace("_", r"\_")
        lines += [
            f"\\addplot+[only marks] coordinates {{{points}}};",
            f"\\addlegendentry{{{label}}}",
        ]
    lines += [r"\end{axis}", r"\end{tikzpicture}", r"\end{document}"]
    (out / "development_quality_plot.tex").write_text("\n".join(lines) + "\n")
    # Plot differences with their registered simultaneous intervals rather than
    # suggesting that isolated CIDEr point estimates establish superiority.
    interval_plot(comparisons, out / "paired_quality_intervals.pdf")
    availability = read(root / "coco_availability/report.json")
    prefix = read(root / "prefix-summary.json")
    claims = dict(
        complete_real_model_development_evidence=(
            "supported: two studies, 832 rows each, same 64 inputs"
        ),
        independent_128_examples="false: do not sum reused development identities",
        opportunity_or_coverage_superiority=(
            "not established; preserve both studies as negative/inconclusive"
        ),
        coverage_preserves_individual_query_heads="unsupported: GQA query heads were averaged",
        exact_retained_kv_and_matched_physical_caps=(
            "supported by separately hash-bound numerical/storage gates"
        ),
        general_latency_win=(
            "unsupported; quality-only timings are descriptive and variants add overhead"
        ),
        lower_memory_at_equal_quality="not tested by these equal-capacity experiments",
        three_tier_or_native_serving_advantage="not tested",
        fresh_confirmation=availability["status"],
    )
    write_once(
        out / "claim_evidence.json", dict(source_sha=revision, claims=claims, bindings=evidence)
    )
    text = [
        "# Fox-review evidence closure",
        "",
        "These are two consumed-development studies, not fresh confirmation.",
        "All 1,664 request observations cover the same 64 unique inputs; do not report N=128.",
        "The 1/16 comparator is entropy routing at 1/16, not the original frozen 1/256 policy.",
        "Both routing/coverage revisions are sealed as negative or inconclusive. No more",
        "routing, swap, precision, budget, or coefficient search is authorized by this packet.",
        "",
        "## Measured scores",
        "",
        "| Study | Method | Variant | MiB | CIDEr |",
        "|---|---|---|---:|---:|",
    ]
    for row in all_scores:
        text.append(
            f"| {row['study']} | {row['method']} | {row['variant']} | "
            f"{row['cap_mib']} | {row['corpus_cider']:.4f} |"
        )
    text += ["", "## Interpretation and paper scope", ""]
    text += [f"- {k}: {v}." for k, v in claims.items()]
    text += [
        "",
        "FullKV is a quality reference, not a compressed-method latency competitor.",
        "Retained entries remain exact; eviction still loses evidence. Numerical cache",
        "equivalence does not imply FullKV task-quality equivalence. No inactive prototype",
        "or repair mechanism may support a performance claim.",
        "",
        "The remaining independent-confirmation blocker must not be waived because",
        "the writing deadline is near. ID availability alone is not a media freshness certificate.",
        "MileBench remains development-only; its exhausted/uncertified sources are not replaced",
        "silently. CIDEr, Accuracy, ROUGE-L, and PrefixKV PPL stay in separate tables.",
        "",
        "## PrefixKV parallel development result",
        "",
        f"The separate PrefixKV gate is `{prefix.get('gate', {}).get('status', 'unavailable')}`; "
        "quality/memory acceptance is "
        f"`{prefix.get('gate', {}).get('quality_memory_gate', False)}`.",
        "See prefix-summary.json and its hash-bound PPL/ROUGE-L tables. Cross-budget",
        "hypotheses are explicitly unequal-budget, not byte-matched comparisons.",
        "The local byte-rescaled append-only PrefixKV adaptation is not an official",
        "paper-native reproduction. No new timing campaign follows a failed quality gate.",
        "",
        "## Paper and appendix mapping",
        "",
        "Main text: tested exact-cache representation contract and explicitly scoped evidence.",
        "Appendix: complete routing/coverage matrix, adjusted intervals, mechanism activity,",
        "descriptive timings, negative-result seals, and independent-confirmation blockers.",
        "Do not replace validated historical paper tables with these development numbers.",
    ]
    (out / "final_writing_packet.md").write_text("\n".join(text) + "\n")
    write_once(
        out / "GATE_analysis.json",
        dict(
            status="passed_development_reporting_only",
            source_sha=revision,
            independent_statistical_regenerations=2,
            raw_request_rows=1664,
            unique_quality_inputs=64,
            no_new_gpu_inference=True,
            paper_superiority_claim_established=False,
            confirmation_executed=False,
            confirmation_blocker=availability["status"],
            cpu_suite_sha256=sha(root / "cpu-tests.json"),
            corrective_cpu_suite_sha256=sha(corrective_tests)
            if corrective_tests.exists()
            else None,
            artifacts={str(p): sha(p) for p in out.iterdir() if p.is_file()},
        ),
    )


def prefix_summary(root):
    path = PREFIX / "analysis/GATE.json"
    if not path.exists():
        write_once(
            root / "prefix-summary.json", dict(status="missing_prefix_analysis", root=str(PREFIX))
        )
        return
    gate = read(path)
    record = dict(
        root=str(PREFIX),
        analysis_sha256=sha(path),
        gate=gate,
        separate_metrics=["teacher_forced_ppl", "rouge_l_f1"],
        fullkv_excluded_from_compressed_latency=True,
    )
    for name in (
        "native_metrics.parquet",
        "paired_cross_budget.parquet",
        "paired_same_budget.parquet",
    ):
        artifact = PREFIX / "analysis" / name
        if artifact.exists():
            record[name] = dict(path=str(artifact), sha256=sha(artifact))
    timing = PREFIX / "decisions/timing.json"
    if timing.exists():
        record["conditional_timing_submission"] = read(timing)
    write_once(root / "prefix-summary.json", record)


def cpu_suite(root, *, timeout_seconds=540, capture_mode="fd"):
    if type(timeout_seconds) is not int or not 540 <= timeout_seconds <= 1800:
        raise ValueError("bounded complete CPU-suite timeout must be 540..1800 seconds")
    if type(capture_mode) is not str or capture_mode not in ("fd", "sys"):
        raise ValueError("CPU capture mode must be fd (legacy default) or sys")
    root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "mosaickv/tests/unit",
        "-o",
        f"cache_dir={os.environ['MOSAICKV_CACHE_ROOT']}/pytest-fox-closure-full",
        f"--junitxml={root}/cpu-tests.xml",
    ]
    # Keep the historical default command byte-for-byte unchanged. The
    # explicitly registered infrastructure retry replaces only global pytest
    # capture's repeated shared-filesystem seek/read/truncate operations.
    # Native stdout remains retained by the outer immutable CPU log.
    if capture_mode == "sys":
        command.append("--capture=sys")
    with (root / "cpu-tests.log").open("x") as stream:
        try:
            run = subprocess.run(
                command,
                env=env,
                cwd=SOURCE,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            stream.flush()
            write_once(
                root / "cpu-tests.json",
                dict(
                    source_sha=clean_source(SOURCE),
                    status="failed",
                    returncode="timeout",
                    command=command,
                    timeout_seconds=timeout_seconds,
                    capture_mode=capture_mode,
                    all_tests_completed=False,
                    models_or_data_downloaded=False,
                    log_sha256=sha(root / "cpu-tests.log"),
                    junit_sha256=sha(root / "cpu-tests.xml")
                    if (root / "cpu-tests.xml").exists()
                    else None,
                ),
            )
            raise ValueError(
                "complete CPU suite timed out; no empirical work may proceed"
            ) from error
    write_once(
        root / "cpu-tests.json",
        dict(
            source_sha=clean_source(SOURCE),
            status="passed" if run.returncode == 0 else "failed",
            returncode=run.returncode,
            command=command,
            timeout_seconds=timeout_seconds,
            capture_mode=capture_mode,
            models_or_data_downloaded=False,
            log_sha256=sha(root / "cpu-tests.log"),
            junit_sha256=sha(root / "cpu-tests.xml"),
        ),
    )
    if run.returncode:
        raise ValueError("CPU unit suite failed; see immutable cpu-tests.log")


def submit(root):
    """Finite afterany graph: errors also reach analysis; no GPU admission."""
    revision = clean_source(SOURCE)
    root.mkdir(parents=True, exist_ok=False)
    (root / "slurm").mkdir()
    jobs = {}
    wrapper = SOURCE / "mosaickv/slurm/evicastkv_review_closure_v2.sbatch"
    for mode in ("r256", "r16", "audit", "prefix", "tests", "packet"):
        command = [
            "sbatch",
            "--parsable",
            "--account=bii_dsc_community",
            f"--output={root}/slurm/{mode}-%j.out",
            f"--error={root}/slurm/{mode}-%j.err",
        ]
        if mode == "packet":
            command += ["--dependency=afterany:" + ":".join(jobs.values())]
        command += [str(wrapper), str(SOURCE), str(root), mode]
        job = subprocess.check_output(command, text=True, timeout=30).strip().split(";", 1)[0]
        if not job.isdigit():
            raise ValueError("scheduler did not return a job ID")
        jobs[mode] = job
        write_once(
            root / f"jobs/{mode}.json", dict(source_sha=revision, command=command, job_id=job)
        )
    write_once(root / "submission.json", dict(source_sha=revision, jobs=jobs, gpu_jobs=0))
    print(json.dumps(jobs, sort_keys=True))


def reuse_report_inputs(root, parent):
    """New derived root, verified producer evidence; never relabel its source."""
    if not parent.is_dir() or root.exists():
        raise ValueError("require completed parent and a new report root")
    cpu = read(parent / "cpu-tests.json")
    if cpu["status"] != "passed" or any(
        sha(parent / name) != cpu[key]
        for name, key in (("cpu-tests.log", "log_sha256"), ("cpu-tests.xml", "junit_sha256"))
    ):
        raise ValueError("parent CPU evidence changed")
    files = [
        "cpu-tests.json",
        "cpu-tests.log",
        "cpu-tests.xml",
        "coco_availability/report.json",
        "prefix-summary.json",
    ]
    for label, original in STUDIES.items():
        name = f"reproduced/{label}/reproduction_gate.json"
        gate = read(parent / name)
        stats = original / f"analysis/{STAGE}/quality_statistics.json"
        if (
            gate["status"] != "passed"
            or not gate["numerically_identical"]
            or gate["original_statistics_sha256"] != sha(stats)
            or gate["regenerated_statistics_sha256"] != sha(stats)
        ):
            raise ValueError("parent numerical reproduction is missing or changed")
        files.append(name)
    audit_record = read(parent / "coco_availability/report.json")
    bindings = dict(audit_record["bound_files"])
    prefix_record = read(parent / "prefix-summary.json")
    bindings[str(PREFIX / "analysis/GATE.json")] = prefix_record["analysis_sha256"]
    for name in (
        "native_metrics.parquet",
        "paired_cross_budget.parquet",
        "paired_same_budget.parquet",
    ):
        entry = prefix_record[name]
        bindings[entry["path"]] = entry["sha256"]
    if any(sha(Path(path)) != expected for path, expected in bindings.items()):
        raise ValueError("a reused immutable input changed")
    root.mkdir()
    for name in files:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(parent / name, target)
    write_once(
        root / "analysis_reuse.json",
        dict(
            analysis_source_sha=clean_source(SOURCE),
            parent_root=str(parent),
            parent_cpu_source_sha=cpu["source_sha"],
            reused_inputs={str(parent / name): sha(parent / name) for name in files},
            new_gpu_requests=0,
            new_bootstrap_replicates=0,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=["r256", "r16", "audit", "packet", "prefix", "tests", "submit", "reuse"]
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--cpu-timeout-seconds", type=int, default=540)
    parser.add_argument("--cpu-capture-mode", choices=("fd", "sys"), default="fd")
    args = parser.parse_args()
    if not args.root.resolve().is_relative_to(Path("/scratch/djy8hg").resolve()):
        parser.error("scratch-only output")
    clean_source(SOURCE)
    try:
        if args.mode in STUDIES:
            reproduce(args.root, args.mode)
        elif args.mode == "audit":
            audit(args.root)
        elif args.mode == "packet":
            packet(args.root)
        elif args.mode == "submit":
            submit(args.root)
        elif args.mode == "tests":
            if args.cpu_capture_mode == "fd":
                cpu_suite(args.root, timeout_seconds=args.cpu_timeout_seconds)
            else:
                cpu_suite(
                    args.root,
                    timeout_seconds=args.cpu_timeout_seconds,
                    capture_mode=args.cpu_capture_mode,
                )
        elif args.mode == "reuse":
            if args.parent is None:
                parser.error("reuse requires --parent")
            reuse_report_inputs(args.root, args.parent)
        else:
            prefix_summary(args.root)
    except Exception as exc:
        write_once(
            args.root / f"failures/{args.mode}.json",
            dict(
                status="failed",
                error_type=type(exc).__name__,
                error=str(exc),
                source_sha=clean_source(SOURCE),
                traceback=traceback.format_exc(),
            ),
        )
        sys.exit(1)
