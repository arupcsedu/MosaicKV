#!/usr/bin/env python3
"""Registered CPU scoring of consumed, common-transport correctness predictions.

This does not enable the GPU worker's scoring/timing paths or reclassify its
instrumented timings as performance measurements. No model inference is needed.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from evicastkv_analyze import collect, pairing
from evicastkv_consumed_regate import TRANSPORT_FILES
from evicastkv_evidence import group_manifest

from mosaickv.evaluation.fox_campaign import TASKS, clean_source, sha, write_once
from mosaickv.evaluation.fox_statistics import paired_scalar_statistics
from mosaickv.evaluation.milebench import route_milebench_dataset, score_milebench_sample

SOURCE = Path(__file__).resolve().parents[2]
ARITHMETIC = dict(
    allow_fp16_reduced_precision_reduction=True, attention_layout_policy="valid_slot_decode"
)
METHODS = ("prompt_attention_topk", "mosaickv_exact")
CAPS = (16 * 2**20, 40 * 2**20)


def quality_statistics(metric_name, statistics):
    """Keep quality effects, never latency labels from the generic reducer."""
    if metric_name not in {"accuracy", "rouge_l_f1"}:
        raise ValueError("only the registered higher-is-better quality metrics are supported")
    reference = statistics["reference_value"]
    candidate = statistics["candidate_value"]
    delta = statistics["difference"]
    if (
        any(
            type(v) not in {int, float} or not math.isfinite(v)
            for v in (reference, candidate, delta)
        )
        or not 0 <= reference <= 1
        or not 0 <= candidate <= 1
        or not math.isclose(delta, candidate - reference, rel_tol=1e-12, abs_tol=1e-12)
    ):
        raise ValueError("invalid quality point estimates or difference")
    for name in ("ordinary_95_ci", "simultaneous_ci"):
        bounds = statistics[name]
        if (
            len(bounds) != 2
            or any(type(v) not in {int, float} or not math.isfinite(v) for v in bounds)
            or bounds[0] > bounds[1]
        ):
            raise ValueError("invalid quality confidence interval")
    result = {
        key: value
        for key, value in statistics.items()
        if not key.startswith("speedup_")
        and key not in {"percentage_reduction", "relative_difference"}
    }
    result.update(
        metric_name=metric_name,
        metric_direction="higher",
        effect_quantity="candidate_minus_reference_quality_score",
        relative_quality_difference=delta / abs(reference) if reference else None,
        timing_inference_allowed=False,
    )
    return result


def reexport(root, output):
    """Correct labels on saved quality statistics without changing any result."""
    revision = clean_source(SOURCE)
    if output.exists():
        raise FileExistsError("derived quality-statistics output must be new")
    gate_path = root / "GATE_scoring.json"
    registration_path = root / "registration.json"
    statistics_path = root / "paired_statistics.json"
    gate = json.loads(gate_path.read_text())
    registration = json.loads(registration_path.read_text())
    if (
        gate["status"] != "passed_bounded_retrospective_scoring_not_confirmation"
        or gate["registration_sha256"] != sha(registration_path)
        or gate["completed"] != 128
        or gate["expected"] != 128
        or gate["failed"] != 0
        or gate["official_scorer_fixtures_passed"] is not True
        or gate["timing_inference_allowed"] is not False
        or gate["fresh_confirmation_authorized"] is not False
        or gate["superiority_claim_allowed"] is not False
        or registration["expected_rows"] != 128
    ):
        raise ValueError("complete, passed, development-only quality scoring is required")
    bindings = dict(registration["bindings"])
    for path, expected in bindings.items():
        if sha(Path(path)) != expected:
            raise ValueError("registered original scoring evidence changed")
    source = json.loads(statistics_path.read_text())
    expected = {(metric, cap) for metric in ("accuracy", "rouge_l_f1") for cap in CAPS}
    seen, corrected = set(), []
    for contrast in source:
        key = (contrast["metric_name"], contrast["budget_bytes"])
        if (
            key not in expected
            or key in seen
            or contrast["reference"] != METHODS[0]
            or contrast["candidate"] != METHODS[1]
            or contrast["unique_inputs"] != 16
            or contrast["family_size"] != registration["statistical_family_size"]
            or contrast["resamples"] != registration["bootstrap_resamples"]
            or contrast["seed"] != registration["seed"]
        ):
            raise ValueError("unexpected, duplicated or misbound quality contrast")
        seen.add(key)
        corrected.append(quality_statistics(contrast["metric_name"], contrast))
    if seen != expected:
        raise ValueError("complete registered quality-contrast grid is required")
    for path in (gate_path, registration_path, statistics_path):
        bindings[str(path)] = sha(path)
    write_once(output / "paired_quality_statistics.json", corrected)
    write_once(
        output / "GATE_quality_statistics.json",
        dict(
            status="passed_metadata_correction_consumed_development_only",
            source_sha=revision,
            source_scoring_root=str(root),
            source_scoring_sha=gate["source_sha"],
            bindings=bindings,
            correction_script_sha256=sha(Path(__file__)),
            contrasts=len(corrected),
            quality_values_and_confidence_intervals_unchanged=True,
            removed_fields="speedup_*; percentage_reduction; ambiguous relative_difference",
            relative_effect="relative_quality_difference=(candidate-reference)/abs(reference)",
            new_GPU_inference=0,
            bootstrap_recomputed=False,
            original_artifacts_preserved=True,
            timing_inference_allowed=False,
            fresh_confirmation_authorized=False,
            superiority_claim_allowed=False,
        ),
    )


def comparison_controls(config):
    """Method-specific ranking is excluded; execution/storage controls are not."""
    cache = config["cache"]
    return dict(
        model=config["model"],
        dataset=config["dataset"],
        generation=config["generation"],
        execution=config["execution"],
        cache={
            k: cache[k]
            for k in (
                "accounting_spec_sha",
                "block_size",
                "budget_unit",
                "budget_value",
                "exact_gather_mode",
                "packing_mode",
                "physical_budget_block_size",
                "physical_budget_mandatory_policy",
                "total_storage_budget_bytes",
            )
        },
    )


def validate_rows(rows, samples):
    """Reject incomplete/misbound input grids without looking at task outcomes."""
    ids = {s["sample_id"] for s in samples}
    expected = {(sid, method, cap) for sid in ids for method in METHODS for cap in CAPS}
    seen, controls = set(), {}
    by_id = {s["sample_id"]: s for s in samples}
    if (
        len(ids) != 32
        or len(samples) != 32
        or any(sum(s["milebench_dataset"] == task for s in samples) != 8 for task in TASKS)
    ):
        raise ValueError("the consumed cohort must contain exactly 32 unique inputs")
    for row in rows:
        key = (row["sample_id"], row["method"], row["budget_bytes"])
        if key in seen or key not in expected:
            raise ValueError("duplicate or unexpected scoring key")
        seen.add(key)
        sample = by_id[row["sample_id"]]
        if (
            row["status"] != "completed"
            or row["error"] is not None
            or row["arithmetic_policy"] != ARITHMETIC
            or row["effective_method"] != row["method"]
            or row["actual_method"] != row["method"]
            or row["references"] != sample["references"]
            or row["choices"] != sample.get("choices", [])
            or row["subtask"] != sample["milebench_dataset"]
            or row["question_type"] != sample["milebench_question_type"]
            or row["question_type"] != sample["milebench_question_type"]
            or row["emitted_tokens"] != 64
            or row["forward_steps"] != 63
            or len(row["token_ids"]) != 64
        ):
            raise ValueError("failed, fallback, input, schedule or arithmetic mismatch")
        path = Path(row["resolved_config_path"])
        if sha(path) != row["resolved_config_file_sha256"]:
            raise ValueError("resolved configuration changed")
        control = comparison_controls(json.loads(path.read_text()))
        paired_key = (row["sample_id"], row["budget_bytes"])
        if control != controls.setdefault(paired_key, control):
            raise ValueError("paired generation/model/cache controls differ")
    if seen != expected:
        raise ValueError("incomplete scoring denominator; do not take an intersection")
    pairs = pairing(rows)
    if len(pairs) != 64 or any(p["initial_bytes"] != p["cap"] for p in pairs):
        raise ValueError("all 64 paired initial byte caps must match exactly")
    return pairs


def gated_rows(roots):
    rows, bindings = [], {}
    for root in roots:
        lock_path = root / "protocol_lock.json"
        lock = json.loads(lock_path.read_text())
        clean_source(Path(lock["source_root"]), lock["source_sha"])
        for name in TRANSPORT_FILES:
            if sha(SOURCE / name) != sha(Path(lock["source_root"]) / name):
                raise ValueError("saved prediction transport/runtime source differs")
        gate_path = root / "analysis/correctness/GATE_correctness.json"
        gate = json.loads(gate_path.read_text())
        if gate["status"] != "passed" or gate["protocol_sha256"] != sha(lock_path):
            raise ValueError("representation gate failed or belongs to another protocol")
        spec, workers, part, errors = collect(root, "correctness")
        if errors or len(workers) != len(spec["jobs"]):
            raise ValueError("source predictions are incomplete or failed")
        rows.extend(part)
        for path in (lock_path, gate_path):
            bindings[str(path)] = sha(path)
        for path in (root / "outputs/correctness").glob("*/*/records/*.json"):
            bindings[str(path)] = sha(path)
    return rows, bindings


def prepare(root, parent, roots, cpu_gate):
    revision = clean_source(SOURCE)
    if root.exists():
        raise FileExistsError("scoring registration root must be new")
    cpu = json.loads(cpu_gate.read_text())
    if (
        cpu["status"] != "passed"
        or cpu["counts"].get("tests", 0) < 1
        or any(cpu["counts"][k] for k in ("errors", "failures", "skipped"))
    ):
        raise ValueError("complete initialized-submodule CPU suite has not passed")
    rows, bindings = gated_rows(roots)
    parent_lock = json.loads((parent / "protocol_lock.json").read_text())
    info = parent_lock["cohorts"]["milebench"]["validation"]
    ledger = Path(info["path"])
    if sha(ledger) != info["sha256"]:
        raise ValueError("consumed input ledger changed")
    samples = json.loads(ledger.read_text())["samples"]
    validate_rows(rows, samples)
    groups = group_manifest(parent)
    group_path = root / "group_manifest.json"
    write_once(group_path, groups)
    bindings.update(groups["bound_files"])
    for p in (cpu_gate, ledger, parent / "protocol_lock.json", group_path):
        bindings[str(p)] = sha(p)
    write_once(
        root / "registration.json",
        dict(
            source_sha=revision,
            parent=str(parent),
            prediction_roots=list(map(str, roots)),
            bindings=bindings,
            input_ledger=str(ledger),
            group_manifest=str(group_path),
            arithmetic_policy=ARITHMETIC,
            methods=METHODS,
            budgets_bytes=CAPS,
            sample_ids=[s["sample_id"] for s in samples],
            expected_rows=128,
            bootstrap_resamples=10000,
            seed=20271014,
            statistical_family_size=6,
            metrics=["accuracy", "rouge_l_f1"],
            scope="retrospective_consumed_exact_method_scoring_only",
            fullkv_quality_reference_available=False,
            baseline_fidelity_claim_allowed=False,
            superiority_claim_allowed=False,
            boundary_B="not_executed_original_failures_preserved",
            lookm_transformed_cache_fidelity="not_tested",
            gpu_inference_authorized=False,
            timing_inference_allowed=False,
            fresh_confirmation_authorized=False,
            automatic_successor_authorized=False,
        ),
    )


def analyze(root):
    import pandas as pd

    reg = json.loads((root / "registration.json").read_text())
    clean_source(SOURCE, reg["source_sha"])
    for path, expected in reg["bindings"].items():
        if sha(Path(path)) != expected:
            raise ValueError("registered scoring evidence changed")
    rows, _ = gated_rows([Path(p) for p in reg["prediction_roots"]])
    samples = json.loads(Path(reg["input_ledger"]).read_text())["samples"]
    validate_rows(rows, samples)
    fixture = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "mosaickv/tests/unit/test_paper_native_metrics.py",
            "-q",
            "-p",
            "no:cacheprovider",
            "-k",
            "milebench",
        ],
        cwd=SOURCE,
        capture_output=True,
        text=True,
        timeout=180,
    )
    write_once(
        root / "scorer_fixture.json",
        dict(
            returncode=fixture.returncode,
            stdout=fixture.stdout,
            stderr=fixture.stderr,
            source_sha=reg["source_sha"],
        ),
    )
    if fixture.returncode:
        raise ValueError("pinned official scorer fixtures failed")
    observations, per_task = [], defaultdict(list)
    for row in rows:
        route = route_milebench_dataset(row["subtask"], row["question_type"])
        if len(row["references"]) != 1:
            raise ValueError("this MileBench route requires one declared reference")
        score = score_milebench_sample(
            route, row["answer"], row["references"][0], choices=tuple(row["choices"])
        )
        observation = dict(
            run_id=root.name,
            source_run_id=row["run_id"],
            sample_id=row["sample_id"],
            task=row["subtask"],
            method=row["method"],
            budget_bytes=row["budget_bytes"],
            metric_name=route.metric.value,
            metric_value=score,
            metric_direction="higher",
            metric_scope="sample",
            metric_implementation="pinned_LOOK-M_compatible_MileBench",
            metric_version=sha(SOURCE / "mosaickv/src/mosaickv/evaluation/milebench.py"),
            evaluator_commit=reg["source_sha"],
            reference_artifact_sha=reg["bindings"][reg["input_ledger"]],
            evaluated_token_count=row["quality_decoding"]["scored_token_count"],
            initial_kv_bytes=row["initial_kv_bytes"],
            final_kv_bytes=row["final_kv_bytes"],
            peak_kv_bytes=row["peak_kv_bytes"],
            manifest_path=row["manifest_path"],
            source_sha=row["source_sha"],
            config_sha=row["config_sha"],
            measurement_type="derived_from_consumed_instrumented_predictions_not_latency",
        )
        observations.append(observation)
        per_task[(row["subtask"], row["method"], row["budget_bytes"], route.metric.value)].append(
            score
        )
    aggregates = [
        dict(
            task=k[0],
            method=k[1],
            budget_bytes=k[2],
            metric_name=k[3],
            metric_value=sum(v) / len(v),
            scored=len(v),
            attempted=8,
            failed=0,
        )
        for k, v in sorted(per_task.items())
    ]
    group_map = json.loads(Path(reg["group_manifest"]).read_text())["groups"]["milebench"]
    contrasts = []
    for metric, tasks in (("accuracy", TASKS[:2]), ("rouge_l_f1", TASKS[2:])):
        for cap in CAPS:
            selected = [
                o
                for o in observations
                if o["metric_name"] == metric and o["task"] in tasks and o["budget_bytes"] == cap
            ]
            values = {
                m: {o["sample_id"]: o["metric_value"] for o in selected if o["method"] == m}
                for m in METHODS
            }
            task_map = {o["sample_id"]: o["task"] for o in selected}
            contrasts.append(
                dict(
                    metric_name=metric,
                    budget_bytes=cap,
                    reference=METHODS[0],
                    candidate=METHODS[1],
                    **quality_statistics(
                        metric,
                        paired_scalar_statistics(
                            values[METHODS[0]],
                            values[METHODS[1]],
                            {s: group_map[s] for s in task_map},
                            task_map,
                            resamples=reg["bootstrap_resamples"],
                            seed=reg["seed"],
                            hierarchy=True,
                            family_size=reg["statistical_family_size"],
                        ),
                    ),
                )
            )
    pd.DataFrame(observations).to_parquet(root / "metric_observations.parquet", index=False)
    pd.DataFrame(aggregates).to_parquet(root / "metric_aggregates.parquet", index=False)
    write_once(root / "paired_statistics.json", contrasts)
    write_once(
        root / "GATE_scoring.json",
        dict(
            source_sha=reg["source_sha"],
            registration_sha256=sha(root / "registration.json"),
            status="passed_bounded_retrospective_scoring_not_confirmation",
            completed=len(observations),
            expected=reg["expected_rows"],
            failed=0,
            official_scorer_fixtures_passed=True,
            metrics_not_combined=True,
            GPU_inference=0,
            timing_inference_allowed=False,
            fullkv_reference_not_measured=True,
            lookm_fidelity_not_tested=True,
            boundary_B_not_tested=True,
            fresh_confirmation_authorized=False,
            superiority_claim_allowed=False,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "analyze", "reexport"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--predictions", nargs="+", type=Path)
    parser.add_argument("--cpu-gate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    for path in (args.root, args.parent, args.cpu_gate, args.output, *(args.predictions or [])):
        if path and not path.resolve().is_relative_to(Path("/scratch/djy8hg").resolve()):
            raise ValueError("scratch-only scoring")
    if args.mode == "prepare":
        if not args.parent or not args.predictions or not args.cpu_gate:
            parser.error("prepare requires parent, predictions and cpu-gate")
        prepare(args.root, args.parent, args.predictions, args.cpu_gate)
    elif args.mode == "reexport":
        if args.output is None:
            parser.error("reexport requires a new --output directory")
        reexport(args.root, args.output)
    else:
        analyze(args.root)
