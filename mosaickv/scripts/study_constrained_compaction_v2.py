#!/usr/bin/env python3
"""Finite CPU-only, hash-bound coefficient holdouts on two consumed captures."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

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

PROTOCOL = SOURCE / "mosaickv/configs/experiments/constrained_compaction_cpu_v2.yaml"
BLOCKED = dict(
    fresh_cohort_authorized=False, calibration_authorized=False, paper_campaign_authorized=False
)
SOURCE_FILES = (
    "mosaickv/src/mosaickv/prototypes/constrained_compaction.py",
    "mosaickv/scripts/study_constrained_compaction_v2.py",
    "mosaickv/configs/experiments/constrained_compaction_cpu_v2.yaml",
    "mosaickv/tests/unit/test_constrained_compaction.py",
    "mosaickv/tests/unit/test_constrained_compaction_study.py",
    "mosaickv/docs/constrained_compaction_v2.md",
    "mosaickv/slurm/constrained_compaction_cpu_v2.sbatch",
    "mosaickv/slurm/constrained_compaction_analysis_v2.sbatch",
)


def validate_protocol(config):
    if (
        any(config[k] is not False for k in BLOCKED)
        or config["gpu_execution_authorized"] is not False
        or config["same_consumed_GPU_followup_authorized_only_if_useful"] is not True
        or config["effective_method"] != "attention_matching_constrained_reimpl"
        or config["classification"]
        != "exploratory_CPU_saved_capture_surrogate_study_not_model_quality"
        or config["dictionary_scope"]
        != "frozen_39_keys_selected_using_all_32_parent_draft_prefixes"
        or config["holdout_scope"]
        != "coefficient_fit_only_NOT_independent_key_selection_or_benchmark_evidence"
        or config["components"] != ["anchor", "mass_only", "value_only", "joint"]
        or config["prefix_folds"]
        != [dict(train_prefixes=t, holdout_start=t + 1, holdout_end=t + 8) for t in (8, 16, 24)]
        or config["source_policy"]
        != "unchanged_parent_ridge_mass_limit_and_four_x_value_norm_bound"
        or config["value_solver"] != dict(iterations=256)
        or config["chunk_routes"] != 64
        or config["torch_threads"] != 1
        or config["workers"] != 2
        or config["cpus_per_worker"] != 1
        or config["memory_gib_per_worker"] != 4
        or config["maximum_worker_seconds"] != 1200
        or config["useful_screen"]
        != dict(
            maximum_each_fold_holdout_mse_ratio=1.000001,
            maximum_each_input_mean_holdout_mse_ratio=0.95,
            minimum_changed_heads_each_fold=1,
            require_all_stored_value_norms_within_four_x=True,
            require_mandatory_values_and_bias_unchanged=True,
            candidate_rule="lowest_worst_fold_ratio_then_mean_then_component_name",
        )
        or config["GPU_followup_if_useful"]
        != dict(
            coefficient_fit_prefixes=16,
            same_dictionary_and_consumed_inputs_only=True,
            cap_bytes=28 * 2**20,
            horizon=32,
            checkpoints=[1, 4, 16, 32],
            absolute_tolerance=1e-7,
            relative_tolerance=1e-6,
        )
    ):
        raise ValueError("CPU study scope, constraint, holdout or eligibility policy changed")


def prepare(root):
    if root.exists() or Path("/scratch/djy8hg").resolve() not in root.resolve().parents:
        raise ValueError("require a NEW immutable scratch study root")
    config = yaml.safe_load(PROTOCOL.read_text())
    validate_protocol(config)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True).strip()
    if sha != config["base_source_sha"]:
        raise ValueError("exploratory overlay requires the specified real Git base")
    if subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"], cwd=SOURCE, text=True
    ):
        raise ValueError("stage only the scoped CPU study files to preserve its complete overlay")
    changed = set(
        subprocess.check_output(
            ["git", "diff", "--name-only", "HEAD"], cwd=SOURCE, text=True
        ).splitlines()
    )
    if changed != set(SOURCE_FILES):
        raise ValueError("CPU study overlay contains missing or unrelated source edits")
    patch = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=SOURCE)
    parent = Path(config["parent_root"])
    bound = {str(SOURCE / name): digest(SOURCE / name) for name in SOURCE_FILES}
    for name, expected in (
        ("plan.json", config["parent_plan_sha256"]),
        ("analysis/report.json", config["negative_report_sha256"]),
        ("analysis/decision.json", config["negative_decision_sha256"]),
    ):
        path = parent / name
        if digest(path) != expected:
            raise ValueError("preserved negative parent changed")
        bound[str(path)] = expected
    previous = json.loads((parent / "plan.json").read_text())
    for index in (0, 1):
        folder = parent / f"workers/{index}"
        completed = json.loads((folder / "completed.json").read_text())
        if (
            completed["status"] != "completed"
            or digest(completed["manifest"]) != completed["manifest_sha256"]
        ):
            raise ValueError("invalid consumed capture provenance")
        bound[completed["manifest"]] = completed["manifest_sha256"]
        artifact_map = folder / "artifact-hashes.json"
        bound[str(artifact_map)] = digest(artifact_map)
        for path, expected in json.loads(artifact_map.read_text()).items():
            if digest(path) != expected:
                raise ValueError("consumed GPU capture changed")
            bound[path] = expected
    root.mkdir(parents=True)
    with (root / "source-overlay.patch").open("xb") as handle:
        handle.write(patch)
    bound[str(root / "source-overlay.patch")] = digest(root / "source-overlay.patch")
    result = dict(
        source_root=str(SOURCE),
        base_git_sha=sha,
        source_dirty=True,
        source_patch_sha256=digest(root / "source-overlay.patch"),
        config=config,
        bound_files=bound,
        core=previous["core"],
        tasks=[j["task"] for j in previous["jobs"]],
        classification=config["classification"],
        new_identities=0,
        **BLOCKED,
    )
    write_once(root / "plan.json", result)
    return result


def validate(root, plan_sha):
    if digest(root / "plan.json") != plan_sha:
        raise ValueError("CPU study plan changed")
    plan = json.loads((root / "plan.json").read_text())
    validate_protocol(plan["config"])
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True).strip()
    if sha != plan["base_git_sha"]:
        raise ValueError("CPU overlay base changed")
    if subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"], cwd=SOURCE, text=True
    ):
        raise ValueError("untracked source appeared during the CPU study")
    patch = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=SOURCE)
    if patch != (root / "source-overlay.patch").read_bytes():
        raise ValueError("CPU source overlay changed")
    for path, expected in plan["bound_files"].items():
        if digest(path) != expected:
            raise ValueError("CPU source/input overlay changed")
    return plan


def load_arrays(folder):
    import torch

    rows = torch.load(folder / "constructed.pt", map_location="cpu", weights_only=True)
    evidence = torch.load(folder / "evidence.pt", map_location="cpu", weights_only=True)
    if len(rows) != 1024 or len(evidence) != 32:
        raise ValueError("complete 32-layer/32-head captures required")
    aa, vv, yy, hh, nn = [], [], [], [], []
    for index, row in enumerate(rows):
        layer, head = divmod(index, 32)
        data = evidence[layer]
        p = row["positions"]
        if (
            row["layer"] != layer
            or row["head"] != head
            or len(p) != 39
            or sorted(set(p)) != list(p)
            or p[0] != 0
            or p[-1] != data["prompt_probabilities"].shape[-1] - 1
            or not torch.equal(row["keys"], row["original_keys"])
            or row["original_values"].shape != (39, 128)
        ):
            raise ValueError("invalid fixed dictionary or source values")
        aa.append(data["prompt_probabilities"][head][:, list(p)])
        vv.append(row["original_values"])
        yy.append(data["target"][head])
        hh.append(data["generated_mass"][head])
        nn.append(data["generated_numerator"][head])
    a, v, y, h, n = [torch.stack(items).contiguous() for items in (aa, vv, yy, hh, nn)]
    protected = torch.zeros((1024, 39), dtype=torch.bool)
    protected[:, [0, -1]] = True
    return a, v, y, h, n, protected


def evaluate_fold(arrays, fold, core, policy, chunk=64):
    import torch

    a, v, y, h, n, protected = arrays
    t = fold["train_prefixes"]
    lo = fold["holdout_start"] - 1
    hi = fold["holdout_end"]
    if type(chunk) is not int or chunk < 1 or not 0 < t <= lo < hi <= a.shape[1]:
        raise ValueError("training prefixes and later coefficient holdouts must be disjoint")
    records = []
    saved = {mode: [] for mode in ("mass_only", "value_only", "joint")}
    for start in range(0, a.shape[0], chunk):
        end = min(start + chunk, a.shape[0])
        part = slice(start, end)
        tr = (a[part, :t], v[part], y[part, :t], h[part, :t], n[part, :t], protected[part])
        ho = (a[part, lo:hi], y[part, lo:hi], h[part, lo:hi], n[part, lo:hi])
        zero = torch.zeros((end - start, v.shape[1]), dtype=torch.float32)
        baseline = (
            (attention_output(ho[0], v[part], zero, ho[2], ho[3]) - ho[1].double())
            .square()
            .mean((1, 2))
        )
        mass_bias = fit_batched_mass_bias(tr[0], tr[3], tr[5], core)
        for mode in ("anchor", "mass_only", "value_only", "joint"):
            fit = fit_component(*tr, mode, core, policy, mass_bias=mass_bias)
            returned = (
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
            for j in range(end - start):
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
                        holdout_mse_before=float(baseline[j]),
                        holdout_mse_after=float(returned[j]),
                        attempted_holdout_mse=float(attempted[j]),
                        maximum_value_norm_ratio=float(
                            fit.diagnostics["maximum_value_norm_ratio"][j]
                        ),
                        protected_values_unchanged=fit.diagnostics["protected_values_unchanged"],
                        protected_bias_zero=fit.diagnostics["protected_bias_zero"],
                    )
                )
            if t == 16 and mode != "anchor":
                saved[mode].append(
                    dict(values=fit.values, log_bias=fit.log_bias, admitted=fit.admitted)
                )
    return records, saved


def error_ratio(after, before):
    if before == 0:
        if after == 0:
            return 1.0
        raise ValueError("nonzero reconstruction error relative to a zero-error anchor")
    return after / before


def summarize_fold(records, worker, task, count, expected_routes=1024):
    summaries = []
    if count not in (8, 16, 24) or len(records) != 4 * expected_routes:
        raise ValueError("incomplete or extra coefficient-study rows")
    for mode in ("anchor", "mass_only", "value_only", "joint"):
        rows = [r for r in records if r["train_prefixes"] == count and r["component"] == mode]
        if len(rows) != expected_routes or {r["route"] for r in rows} != set(
            range(expected_routes)
        ):
            raise ValueError("incomplete or duplicate coefficient-study rows")
        for row in rows:
            if (
                row["holdout_start"] != count + 1
                or row["holdout_end"] != count + 8
                or type(row["admitted"]) is not bool
                or type(row["protected_values_unchanged"]) is not bool
                or type(row["protected_bias_zero"]) is not bool
                or any(
                    not math.isfinite(row[key]) or row[key] < 0
                    for key in (
                        "train_mse_before",
                        "train_mse_after",
                        "attempted_train_mse",
                        "holdout_mse_before",
                        "holdout_mse_after",
                        "attempted_holdout_mse",
                        "maximum_value_norm_ratio",
                    )
                )
            ):
                raise ValueError("invalid or nonfinite coefficient observation")
        before = sum(r["holdout_mse_before"] for r in rows)
        after = sum(r["holdout_mse_after"] for r in rows)
        train_before = sum(r["train_mse_before"] for r in rows)
        train_after = sum(r["train_mse_after"] for r in rows)
        summaries.append(
            dict(
                worker=worker,
                task=task,
                component=mode,
                train_prefixes=count,
                routes=len(rows),
                admitted_heads=sum(r["admitted"] for r in rows),
                holdout_mse_before=before / len(rows),
                holdout_mse_after=after / len(rows),
                holdout_mse_ratio=error_ratio(after, before),
                training_mse_ratio=error_ratio(train_after, train_before),
                attempted_holdout_mse=sum(r["attempted_holdout_mse"] for r in rows) / len(rows),
                nonworse_holdout_head_fraction=sum(
                    r["holdout_mse_after"] <= r["holdout_mse_before"] + 1e-12 for r in rows
                )
                / len(rows),
                maximum_value_norm_ratio=max(r["maximum_value_norm_ratio"] for r in rows),
                constraints_valid=all(
                    r["maximum_value_norm_ratio"] <= 4
                    and r["protected_values_unchanged"]
                    and r["protected_bias_zero"]
                    for r in rows
                ),
            )
        )
    return summaries


def summarize(records, worker, task, expected_routes=1024):
    if len(records) != 12 * expected_routes or {r["train_prefixes"] for r in records} != {
        8,
        16,
        24,
    }:
        raise ValueError("incomplete or extra coefficient-study folds")
    return [
        summary
        for count in (8, 16, 24)
        for summary in summarize_fold(
            [r for r in records if r["train_prefixes"] == count],
            worker,
            task,
            count,
            expected_routes,
        )
    ]


def choose_component(summaries, screen):
    eligible = []
    assessments = {}
    for mode in ("mass_only", "value_only", "joint"):
        rows = [r for r in summaries if r["component"] == mode]
        complete = len(rows) == 6 and {(r["worker"], r["train_prefixes"]) for r in rows} == {
            (worker, t) for worker in (0, 1) for t in (8, 16, 24)
        }
        inputs = {worker: [r for r in rows if r["worker"] == worker] for worker in (0, 1)}
        passed = complete and all(
            r["constraints_valid"]
            and math.isfinite(r["holdout_mse_ratio"])
            and r["admitted_heads"] >= screen["minimum_changed_heads_each_fold"]
            and r["holdout_mse_ratio"] <= screen["maximum_each_fold_holdout_mse_ratio"]
            for r in rows
        )
        passed = passed and all(
            sum(r["holdout_mse_ratio"] for r in per_input) / 3
            <= screen["maximum_each_input_mean_holdout_mse_ratio"]
            for per_input in inputs.values()
        )
        assessments[mode] = dict(
            eligible=bool(passed),
            complete=complete,
            worst_fold_ratio=max((r["holdout_mse_ratio"] for r in rows), default=None),
            per_input_mean_ratio={
                worker: sum(r["holdout_mse_ratio"] for r in rr) / len(rr) if rr else None
                for worker, rr in inputs.items()
            },
        )
        if passed:
            eligible.append(
                (
                    max(r["holdout_mse_ratio"] for r in rows),
                    sum(r["holdout_mse_ratio"] for r in rows) / 6,
                    mode,
                )
            )
    return dict(
        selected_component=min(eligible)[-1] if eligible else None,
        useful_bounded_fit=bool(eligible),
        assessments=assessments,
    )


def worker(root, plan_sha, index):
    import platform

    import torch

    plan = validate(root, plan_sha)
    if (
        type(index) is not int
        or index not in (0, 1)
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
    ):
        raise ValueError("CPU study must hide GPUs and use only the two consumed captures")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    output = root / f"workers/{index}"
    if output.exists():
        raise FileExistsError("CPU worker is write-once")
    output.mkdir(parents=True)
    started = time.perf_counter()
    records = []
    try:
        arrays = load_arrays(Path(plan["config"]["parent_root"]) / f"workers/{index}")
        core = AttentionCompactionPolicy(**plan["core"]["policy"])
        if core.maximum_value_norm_ratio != 4 or core.maximum_mass_multiplier != 64:
            raise ValueError("original norm/mass limits must not change")
        policy = ConstrainedValuePolicy(**plan["config"]["value_solver"])
        for fold in plan["config"]["prefix_folds"]:
            t0 = time.perf_counter()
            rows, saved = evaluate_fold(arrays, fold, core, policy)
            records.extend(rows)
            if fold["train_prefixes"] == 16:
                torch.save(saved, output / "train16-fits.pt")
            write_once(
                output / f"fold-{fold['train_prefixes']}.json",
                dict(
                    elapsed_seconds=time.perf_counter() - t0,
                    fold=fold,
                    summaries=summarize_fold(
                        rows, index, plan["tasks"][index], fold["train_prefixes"]
                    ),
                ),
            )
            print(
                json.dumps(
                    dict(
                        worker=index,
                        completed_fold=fold["train_prefixes"],
                        elapsed_seconds=time.perf_counter() - t0,
                    )
                ),
                flush=True,
            )
        with (output / "heads.jsonl").open("x") as handle:
            for row in records:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        validate(root, plan_sha)
        producer = Path(plan["config"]["parent_root"]) / f"workers/{index}/manifest.json"
        producer_manifest = json.loads(producer.read_text())
        report = dict(
            status="completed",
            summaries=summarize(records, index, plan["tasks"][index]),
            elapsed_seconds=time.perf_counter() - started,
            measurement_type=plan["classification"],
            source_base_sha=plan["base_git_sha"],
            source_dirty=True,
            source_patch_sha256=plan["source_patch_sha256"],
            plan_sha256=plan_sha,
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
            model=producer_manifest["model"],
            dataset=producer_manifest["dataset"],
            effective_method=plan["config"]["effective_method"],
            producer_manifest=str(producer),
            producer_manifest_sha256=digest(producer),
            no_model_inference=True,
            no_benchmark_labels=True,
            new_identities=0,
            **BLOCKED,
        )
    except Exception as error:
        report = dict(
            status="failed",
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
    for index in (0, 1):
        folder = root / f"workers/{index}"
        try:
            for path, expected in json.loads((folder / "artifact-hashes.json").read_text()).items():
                if digest(path) != expected:
                    raise ValueError("CPU result artifact changed")
            report = json.loads((folder / "report.json").read_text())
            if report["status"] != "completed" or report["plan_sha256"] != plan_sha:
                raise ValueError(report.get("error", "invalid worker provenance"))
            rows = [json.loads(line) for line in (folder / "heads.jsonl").read_text().splitlines()]
            actual = summarize(rows, index, plan["tasks"][index])
            if actual != report["summaries"]:
                raise ValueError("CPU summary differs from head observations")
            summaries.extend(actual)
        except (OSError, ValueError, KeyError) as error:
            errors.append(dict(worker=index, error=f"{type(error).__name__}: {error}"))
    decision = choose_component(summaries, plan["config"]["useful_screen"])
    if errors:
        decision.update(selected_component=None, useful_bounded_fit=False)
    report = dict(
        status="incomplete" if errors else "completed",
        summaries=summaries,
        errors=errors,
        source_base_sha=plan["base_git_sha"],
        source_dirty=True,
        source_patch_sha256=plan["source_patch_sha256"],
        plan_sha256=plan_sha,
        measurement_type=plan["classification"],
        holdout_limit=plan["config"]["holdout_scope"],
        no_new_inference=True,
        **decision,
        **BLOCKED,
    )
    write_once(root / "analysis/report.json", report)
    write_once(
        root / "analysis/decision.json",
        dict(
            **decision,
            same_consumed_GPU_followup_eligible=decision["useful_bounded_fit"],
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
            "--array=0-1%2",
            f"--output={logs}/cpu-%A_%a.out",
            f"--error={logs}/cpu-%A_%a.err",
            str(SOURCE / "mosaickv/slurm/constrained_compaction_cpu_v2.sbatch"),
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
            str(SOURCE / "mosaickv/slurm/constrained_compaction_analysis_v2.sbatch"),
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
            gpu_jobs_submitted=False,
            plan_sha256=sha,
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
