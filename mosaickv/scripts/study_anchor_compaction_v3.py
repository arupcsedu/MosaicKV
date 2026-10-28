#!/usr/bin/env python3
"""Finite CPU-only anchor-preserving fit study; no automatic GPU successor."""

from __future__ import annotations

import argparse
import json
import os
import platform
import runpy
import subprocess
import time
from pathlib import Path

import study_constrained_compaction_v2 as v2
import yaml
from bounded_history_pilot import SOURCE, digest, write_once

from mosaickv.prototypes.attention_compaction import (
    AttentionCompactionPolicy,
    fit_batched_mass_bias,
)
from mosaickv.prototypes.constrained_compaction import (
    ConstrainedValuePolicy,
    attention_output,
    fit_component,
)

PROTOCOL = SOURCE / "mosaickv/configs/experiments/anchor_compaction_cpu_v3.yaml"
BLOCKED = dict(**v2.BLOCKED, gpu_execution_authorized=False)
NEW_FILES = (
    "mosaickv/configs/experiments/anchor_compaction_cpu_v3.yaml",
    "mosaickv/scripts/study_anchor_compaction_v3.py",
    "mosaickv/tests/unit/test_anchor_compaction_v3.py",
    "mosaickv/docs/anchor_compaction_v3.md",
    "mosaickv/slurm/anchor_compaction_cpu_v3.sbatch",
    "mosaickv/slurm/anchor_compaction_analysis_v3.sbatch",
)
BOUND_SCHEDULER_FILES = (
    "mosaickv/scripts/control_a100_schedule.py",
    "mosaickv/tests/unit/test_a100_schedule.py",
)
CASES = [
    dict(
        name="ridge_only",
        policy=dict(iterations=256, ridge_multiplier=10000.0, maximum_delta_norm_ratio=None),
        role="mechanism_ablation_not_promotable",
    ),
    dict(
        name="delta_only",
        policy=dict(iterations=256, ridge_multiplier=1.0, maximum_delta_norm_ratio=0.1),
        role="mechanism_ablation_not_promotable",
    ),
    dict(
        name="anchor_preserving",
        policy=dict(iterations=256, ridge_multiplier=10000.0, maximum_delta_norm_ratio=0.1),
        role="fixed_candidate",
    ),
]


def validate_protocol(config):
    old = yaml.safe_load(v2.PROTOCOL.read_text())
    v2.validate_protocol(old)
    if (
        any(config[key] is not False for key in BLOCKED)
        or config["classification"]
        != "exploratory_CPU_consumed_capture_anchor_study_NOT_model_quality"
        or config["effective_method"] != "attention_matching_anchor_preserving_reimpl"
        or config["cases"] != CASES
        or config["useful_screen"] != old["useful_screen"]
        or config["prefix_folds"] != old["prefix_folds"]
        or config["components"] != old["components"]
        or config["holdout_scope"] != old["holdout_scope"]
        or config["dictionary_scope"] != old["dictionary_scope"]
        or config["unchanged_value_norm_limit"] != 4.0
        or config["unchanged_mass_multiplier_limit"] != 64.0
        or config["workers"] != 6
        or config["maximum_concurrent_cpu_workers"] != 4
        or config["cpus_per_worker"] != 1
        or config["torch_threads"] != 1
        or config["chunk_routes"] != 64
        or config["maximum_worker_seconds"] != 1200
        or config["memory_gib_per_worker"] != 4
        or config["future_guard_unchanged"]
        != dict(
            same_consumed_inputs_only=True,
            cap_bytes=28 * 2**20,
            horizon=32,
            checkpoints=[1, 4, 16, 32],
            absolute_tolerance=1e-7,
            relative_tolerance=1e-6,
        )
    ):
        raise ValueError("V3 CPU scope, fixed cases, prefix screen or unchanged guard changed")


def scheduler_preflight():
    path = SOURCE / BOUND_SCHEDULER_FILES[0]
    parse = runpy.run_path(str(path))["duration_minutes"]
    expected = {"30:00": 30, "59:59": 60, "2:00:01": 121, "1-00:00:00": 1440}
    for text, value in expected.items():
        if parse(text) != value:
            raise ValueError("bound scheduler failed finite Slurm-time preflight")
    return dict(
        source=str(path), sha256=digest(path), finite_duration_cases=expected, no_slurm_actions=True
    )


def source_overlay():
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True).strip()
    if subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"], cwd=SOURCE, text=True
    ):
        raise ValueError("stage only the scoped V2/V3 files before preserving the overlay")
    changed = set(
        subprocess.check_output(
            ["git", "diff", "--name-only", "HEAD"], cwd=SOURCE, text=True
        ).splitlines()
    )
    if changed != set(v2.SOURCE_FILES + NEW_FILES):
        raise ValueError("unrelated or missing source changes in V3 overlay")
    return sha, subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=SOURCE)


def prepare(root):
    if root.exists() or Path("/scratch/djy8hg").resolve() not in root.resolve().parents:
        raise ValueError("require a new immutable scratch root")
    config = yaml.safe_load(PROTOCOL.read_text())
    validate_protocol(config)
    sha, patch = source_overlay()
    if sha != config["base_source_sha"]:
        raise ValueError("V3 requires the registered Git base")
    parent = Path(config["negative_v2_root"])
    bound = {}
    for name, key in (
        ("plan.json", "negative_v2_plan_sha256"),
        ("analysis/report.json", "negative_v2_report_sha256"),
        ("analysis/decision.json", "negative_v2_decision_sha256"),
    ):
        path = parent / name
        if digest(path) != config[key]:
            raise ValueError("negative V2 seal changed")
        bound[str(path)] = config[key]
    negative = json.loads((parent / "analysis/report.json").read_text())
    decision = json.loads((parent / "analysis/decision.json").read_text())
    if (
        negative["status"] != "completed"
        or negative["errors"]
        or decision["useful_bounded_fit"] is not False
    ):
        raise ValueError("V2 must remain a complete preserved negative result")
    old = json.loads((parent / "plan.json").read_text())
    bound.update(old["bound_files"])
    for path, expected in bound.items():
        if digest(path) != expected:
            raise ValueError("negative V2 source/input changed")
    bound.update(
        {
            str(SOURCE / name): digest(SOURCE / name)
            for name in v2.SOURCE_FILES + NEW_FILES + BOUND_SCHEDULER_FILES
        }
    )
    scheduler = scheduler_preflight()
    root.mkdir(parents=True)
    with (root / "source-overlay.patch").open("xb") as handle:
        handle.write(patch)
    bound[str(root / "source-overlay.patch")] = digest(root / "source-overlay.patch")
    jobs = [
        dict(
            index=i * 2 + input_index,
            case=case,
            input_index=input_index,
            task=old["tasks"][input_index],
        )
        for i, case in enumerate(CASES)
        for input_index in (0, 1)
    ]
    plan = dict(
        config=config,
        base_git_sha=sha,
        source_root=str(SOURCE),
        source_dirty=True,
        source_patch_sha256=digest(root / "source-overlay.patch"),
        bound_files=bound,
        core=old["core"],
        capture_root=old["config"]["parent_root"],
        jobs=jobs,
        scheduler_preflight=scheduler,
        new_identities=0,
        **BLOCKED,
    )
    write_once(root / "plan.json", plan)
    return plan


def validate(root, plan_sha):
    if digest(root / "plan.json") != plan_sha:
        raise ValueError("V3 plan changed")
    plan = json.loads((root / "plan.json").read_text())
    validate_protocol(plan["config"])
    sha, patch = source_overlay()
    if sha != plan["base_git_sha"] or patch != (root / "source-overlay.patch").read_bytes():
        raise ValueError("V3 exploratory source overlay changed")
    for path, expected in plan["bound_files"].items():
        if digest(path) != expected:
            raise ValueError("V3 bound source/input artifact changed")
    if scheduler_preflight() != plan["scheduler_preflight"]:
        raise ValueError("scheduler preflight changed")
    return plan


def evaluate_fold(arrays, fold, core, policy, chunk=64):
    import torch

    a, v, y, h, n, protected = arrays
    t = fold["train_prefixes"]
    lo = fold["holdout_start"] - 1
    hi = fold["holdout_end"]
    if type(chunk) is not int or chunk < 1 or not 0 < t <= lo < hi <= a.shape[1]:
        raise ValueError("coefficient training and later-prefix holdouts must be disjoint")
    records = []
    for start in range(0, a.shape[0], chunk):
        part = slice(start, min(start + chunk, a.shape[0]))
        count = v[part].shape[0]
        training = (a[part, :t], v[part], y[part, :t], h[part, :t], n[part, :t], protected[part])
        ho = (a[part, lo:hi], y[part, lo:hi], h[part, lo:hi], n[part, lo:hi])
        zero = torch.zeros((count, v.shape[1]), dtype=torch.float32)
        before = (
            (attention_output(ho[0], v[part], zero, ho[2], ho[3]) - ho[1].double())
            .square()
            .mean((1, 2))
        )
        bias = fit_batched_mass_bias(training[0], training[3], training[5], core)
        for mode in ("anchor", "mass_only", "value_only", "joint"):
            fit = fit_component(*training, mode, core, policy, mass_bias=bias)
            after = (
                (attention_output(ho[0], fit.values, fit.log_bias, ho[2], ho[3]) - ho[1].double())
                .square()
                .mean((1, 2))
            )
            attempted = (
                (
                    attention_output(
                        ho[0], fit.attempted_values, fit.attempted_log_bias, ho[2], ho[3]
                    )
                    - ho[1].double()
                )
                .square()
                .mean((1, 2))
            )
            delta = (fit.values.double() - v[part].double()).norm(dim=-1)
            source_norm = v[part].double().norm(dim=-1)
            delta_ratio = (delta / source_norm.clamp_min(1e-12)).amax(-1)
            norm_ratio = (fit.values.double().norm(dim=-1) / source_norm.clamp_min(1e-12)).amax(-1)
            feasible = torch.ones(count, dtype=torch.bool)
            if policy.maximum_delta_norm_ratio is not None:
                feasible = (delta <= policy.maximum_delta_norm_ratio * source_norm).all(-1)
            for j in range(count):
                records.append(
                    dict(
                        route=start + j,
                        component=mode,
                        train_prefixes=t,
                        holdout_start=lo + 1,
                        holdout_end=hi,
                        admitted=bool(fit.admitted[j]),
                        train_mse_before=float(fit.diagnostics["training_mse_before"][j]),
                        train_mse_after=float(fit.diagnostics["returned_training_mse"][j]),
                        attempted_train_mse=float(
                            fit.diagnostics["attempted_training_mse_after"][j]
                        ),
                        holdout_mse_before=float(before[j]),
                        holdout_mse_after=float(after[j]),
                        attempted_holdout_mse=float(attempted[j]),
                        maximum_value_norm_ratio=float(norm_ratio[j]),
                        maximum_value_delta_norm_ratio=float(delta_ratio[j]),
                        delta_constraint_valid=bool(feasible[j]),
                        protected_values_unchanged=bool(
                            torch.equal(
                                fit.values[j][protected[part][j]], v[part][j][protected[part][j]]
                            )
                        ),
                        protected_bias_zero=bool((fit.log_bias[j][protected[part][j]] == 0).all()),
                    )
                )
    return records


def summarize(records, worker, task, expected_routes=1024):
    summary = v2.summarize(records, worker, task, expected_routes)
    for row in summary:
        matching = [
            r
            for r in records
            if r["component"] == row["component"] and r["train_prefixes"] == row["train_prefixes"]
        ]
        row["maximum_value_delta_norm_ratio"] = max(
            r["maximum_value_delta_norm_ratio"] for r in matching
        )
        row["constraints_valid"] &= all(r["delta_constraint_valid"] for r in matching)
    return summary


def choose_candidate(summaries, screen):
    assessments = {}
    for case in CASES:
        rows = [r for r in summaries if r["case"] == case["name"]]
        assessments[case["name"]] = v2.choose_component(rows, screen)
    candidate = assessments["anchor_preserving"]
    complete = all(item["complete"] for item in candidate["assessments"].values())
    useful = complete and candidate["useful_bounded_fit"]
    return dict(
        useful_bounded_fit=useful,
        fixed_case_complete=complete,
        selected_component=candidate["selected_component"] if useful else None,
        selected_case="anchor_preserving" if useful else None,
        assessments=assessments,
    )


def worker(root, plan_sha, index):
    import torch

    plan = validate(root, plan_sha)
    if type(index) is not int or not 0 <= index < 6 or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU-only V3 requires an exact registered worker index")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    job = plan["jobs"][index]
    output = root / f"workers/{index}"
    if output.exists():
        raise FileExistsError("V3 workers are write-once")
    output.mkdir(parents=True)
    started = time.perf_counter()
    records = []
    try:
        arrays = v2.load_arrays(Path(plan["capture_root"]) / f"workers/{job['input_index']}")
        policy = ConstrainedValuePolicy(**job["case"]["policy"])
        core = AttentionCompactionPolicy(**plan["core"]["policy"])
        if (
            core.maximum_value_norm_ratio != 4
            or core.maximum_mass_multiplier != 64
            or core.value_ridge != 1e-6
        ):
            raise ValueError("original safety limits and base ridge cannot change")
        for fold in plan["config"]["prefix_folds"]:
            start = time.perf_counter()
            rows = evaluate_fold(arrays, fold, core, policy)
            records.extend(rows)
            with (output / f"heads-{fold['train_prefixes']}.jsonl").open("x") as handle:
                for row in rows:
                    handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            write_once(
                output / f"fold-{fold['train_prefixes']}.json",
                dict(fold=fold, elapsed_seconds=time.perf_counter() - start),
            )
            print(
                json.dumps(
                    dict(
                        worker=index,
                        completed_fold=fold["train_prefixes"],
                        elapsed_seconds=time.perf_counter() - start,
                    )
                ),
                flush=True,
            )
        summary = summarize(records, job["input_index"], job["task"])
        for row in summary:
            row["case"] = job["case"]["name"]
        validate(root, plan_sha)
        producer = Path(plan["capture_root"]) / f"workers/{job['input_index']}/manifest.json"
        parent = json.loads(producer.read_text())
        report = dict(
            status="completed",
            summaries=summary,
            job=job,
            elapsed_seconds=time.perf_counter() - started,
            plan_sha256=plan_sha,
            source_base_sha=plan["base_git_sha"],
            source_dirty=True,
            source_patch_sha256=plan["source_patch_sha256"],
            measurement_type=plan["config"]["classification"],
            effective_method=plan["config"]["effective_method"],
            model=parent["model"],
            dataset=parent["dataset"],
            producer_manifest=str(producer),
            producer_manifest_sha256=digest(producer),
            software=dict(
                torch=torch.__version__,
                python=platform.python_version(),
                cuda="not_used",
                driver="not_used",
                transformers="not_used",
                vllm="not_used",
                sglang="not_used",
                gpu_type="not_used",
                gpu_count=0,
                seed=0,
                backend="cpu_surrogate_reconstruction",
                attention_implementation="not_used",
            ),
            no_benchmark_labels=True,
            no_model_inference=True,
            new_identities=0,
            **BLOCKED,
        )
    except Exception as error:
        report = dict(
            status="failed",
            job=job,
            error=f"{type(error).__name__}: {error}",
            elapsed_seconds=time.perf_counter() - started,
            **BLOCKED,
        )
    write_once(output / "report.json", report)
    write_once(
        output / "artifact-hashes.json",
        {str(p): digest(p) for p in output.iterdir() if p.is_file()},
    )
    if report["status"] != "completed":
        raise RuntimeError(report["error"])


def analyze(root, plan_sha):
    plan = validate(root, plan_sha)
    summaries = []
    errors = []
    for job in plan["jobs"]:
        index = job["index"]
        folder = root / f"workers/{index}"
        try:
            hashes = json.loads((folder / "artifact-hashes.json").read_text())
            required = {str(folder / "report.json")} | {
                str(folder / f"heads-{t}.jsonl") for t in (8, 16, 24)
            }
            if not required.issubset(hashes):
                raise ValueError("missing required V3 artifact hashes")
            for path, expected in hashes.items():
                if digest(path) != expected:
                    raise ValueError("V3 output artifact changed")
            report = json.loads((folder / "report.json").read_text())
            if report["status"] != "completed" or report["plan_sha256"] != plan_sha:
                raise ValueError(report.get("error", "invalid V3 worker provenance"))
            records = [
                json.loads(line)
                for t in (8, 16, 24)
                for line in (folder / f"heads-{t}.jsonl").read_text().splitlines()
            ]
            actual = summarize(records, job["input_index"], job["task"])
            for row in actual:
                row["case"] = job["case"]["name"]
            if actual != report["summaries"]:
                raise ValueError("V3 summary differs from head observations")
            summaries.extend(actual)
        except (ValueError, KeyError, OSError) as error:
            errors.append(dict(worker=index, error=f"{type(error).__name__}: {error}"))
    decision = choose_candidate(summaries, plan["config"]["useful_screen"])
    if errors:
        decision.update(useful_bounded_fit=False, selected_case=None, selected_component=None)
    report = dict(
        status="incomplete" if errors else "completed",
        summaries=summaries,
        errors=errors,
        measurement_type=plan["config"]["classification"],
        plan_sha256=plan_sha,
        source_base_sha=plan["base_git_sha"],
        source_dirty=True,
        source_patch_sha256=plan["source_patch_sha256"],
        holdout_limit=plan["config"]["holdout_scope"],
        new_identities=0,
        **decision,
        **BLOCKED,
    )
    write_once(root / "analysis/report.json", report)
    write_once(
        root / "analysis/decision.json",
        dict(
            **decision,
            same_consumed_guard_recommended=decision["useful_bounded_fit"],
            gpu_jobs_submitted=False,
            requires_clean_source_for_GPU=True,
            report_sha256=digest(root / "analysis/report.json"),
            **BLOCKED,
        ),
    )
    print(json.dumps(dict(status=report["status"], **decision), allow_nan=False))


def submit(root):
    prepare(root)
    sha = digest(root / "plan.json")
    logs = root / "logs"
    logs.mkdir()

    def sbatch(args):
        return (
            subprocess.check_output(
                ["sbatch", "--parsable", "--account=bii_dsc_community", *args], text=True
            )
            .strip()
            .split(";")[0]
        )

    array = sbatch(
        [
            "--array=0-5%4",
            f"--output={logs}/cpu-%A_%a.out",
            f"--error={logs}/cpu-%A_%a.err",
            str(SOURCE / "mosaickv/slurm/anchor_compaction_cpu_v3.sbatch"),
            str(SOURCE),
            str(root),
            sha,
        ]
    )
    analysis = sbatch(
        [
            f"--dependency=afterany:{array}",
            f"--output={logs}/analysis-%j.out",
            f"--error={logs}/analysis-%j.err",
            str(SOURCE / "mosaickv/slurm/anchor_compaction_analysis_v3.sbatch"),
            str(SOURCE),
            str(root),
            sha,
        ]
    )
    write_once(
        root / "submission.json",
        dict(
            cpu_array=array,
            analysis=analysis,
            plan_sha256=sha,
            gpu_jobs_submitted=False,
            source_dirty_exploratory=True,
            **BLOCKED,
        ),
    )
    print(json.dumps(dict(cpu_array=array, analysis=analysis, root=str(root))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "submit", "worker", "analyze"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha")
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.mode == "worker":
        worker(args.root.resolve(), args.plan_sha, args.index)
    elif args.mode == "analyze":
        analyze(args.root.resolve(), args.plan_sha)
    else:
        globals()[args.mode](args.root.resolve())
