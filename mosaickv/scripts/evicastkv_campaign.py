#!/usr/bin/env python3
"""Finite, hash-bound Fox-review campaign. See --help for executable modes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

from mosaickv.evaluation.fox_campaign import (
    BUDGETS,
    COCO_POOL,
    METHODS,
    MILE_POOL,
    REFERENCE_CONFIG_ROOT,
    SEED,
    STAGES,
    TASKS,
    clean_source,
    development_subset,
    registered_arms,
    sha,
    source_config,
    validate_locked_policy,
    write_once,
)

SOURCE = Path(__file__).resolve().parents[2]
PYTHON = "/scratch/djy8hg/env/mosaickv/bin/python"


def environment():
    packages = {}
    for name in (
        "torch",
        "transformers",
        "accelerate",
        "flash_attn",
        "vllm",
        "sglang",
        "lmms-eval",
        "datasets",
        "pycocoevalcap",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not_installed"
    return dict(
        python=sys.version,
        executable=sys.executable,
        packages=packages,
        platform=platform.platform(),
        caches={
            k: os.environ.get(k)
            for k in ("HF_HOME", "HF_DATASETS_CACHE", "PIP_CACHE_DIR", "TORCH_HOME", "TMPDIR")
        },
    )


def prepare(root: Path, *, strict_fp16_reductions=False, workloads=None, scope_registration=None):
    """Read previously materialized development pools; no HF access/downloads."""
    source_sha = clean_source(SOURCE)
    workloads = tuple(workloads or BUDGETS)
    if not workloads or set(workloads) - set(BUDGETS):
        raise ValueError("unknown workload scope")
    if set(workloads) != set(BUDGETS):
        if scope_registration is None:
            raise ValueError("narrowed execution requires a pre-quality scope registration")
        scope = json.loads(Path(scope_registration).read_text())
        if set(scope["enabled_workloads"]) != set(workloads):
            raise ValueError("scope registration differs from enabled workloads")
        if (
            scope.get("basis") != "model_scoped_numerical_failure_only"
            or scope.get("task_scores_used") is not False
        ):
            raise ValueError(
                "scope must be registered from numerical support, not quality outcomes"
            )
        if sha(Path(scope["parent_gate_path"])) != scope["parent_gate_sha256"]:
            raise ValueError("scope evidence changed")
    arithmetic = dict(allow_fp16_reduced_precision_reduction=not strict_fp16_reductions)
    templates, cohorts, bindings = {}, {}, {}
    for workload in workloads:
        template_path = REFERENCE_CONFIG_ROOT / workload / "mosaickv_exact.json"
        templates[workload] = source_config(workload)
        validate_locked_policy(templates[workload])
        bindings[str(template_path)] = sha(template_path)
        for method in METHODS[workload]:
            method_path = REFERENCE_CONFIG_ROOT / workload / f"{method}.json"
            bindings[str(method_path)] = sha(method_path)
        if workload == "coco":
            parent = json.loads(COCO_POOL.read_text())
            bindings[str(COCO_POOL)] = sha(COCO_POOL)
            samples = development_subset(parent["samples"], count=64, task=workload)
            parent = {
                k: parent[k]
                for k in ("schema_version", "task", "dataset_id", "dataset_revision", "split")
            }
        else:
            samples = []
            parent = None
            for task in TASKS:
                path = MILE_POOL / f"{task}.json"
                payload = json.loads(path.read_text())
                bindings[str(path)] = sha(path)
                samples.extend(development_subset(payload["samples"], count=16, task=task))
                parent = {
                    k: payload[k]
                    for k in ("schema_version", "task", "dataset_id", "dataset_revision", "split")
                }
        for sample in samples:
            for media in sample["media"]:
                if sha(Path(media["path"])) != media["sha256"]:
                    raise ValueError("development media changed")
        cohorts[workload] = {}
        for subset, count in (
            ("development", 64),
            ("validation", 32),
            ("timing", 16),
            ("profiler", 8),
            ("smoke", 1),
        ):
            if workload == "coco":
                selected = samples[:count]
            elif subset == "smoke":
                selected = samples[:1]
            else:
                selected = [
                    s
                    for task in TASKS
                    for s in [x for x in samples if x["milebench_dataset"] == task][: count // 4]
                ]
            payload = dict(parent, samples=selected)
            path = root / "manifests" / f"{workload}-{subset}.json"
            write_once(path, payload)
            cohorts[workload][subset] = dict(
                path=str(path), sha256=sha(path), sample_ids=[s["sample_id"] for s in selected]
            )
        write_once(root / "configs" / f"{workload}-historical.json", templates[workload])
    lock = dict(
        schema_version=1,
        name="evicastkv_fox_v2_controlled_numerics",
        source_sha=source_sha,
        source_root=str(SOURCE),
        source_patch_sha256=hashlib.sha256(b"").hexdigest(),
        source_patch_encoding="raw_git_diff_bytes_empty_clean_worktree",
        seed=SEED,
        arithmetic=arithmetic,
        historical_base_sha="edb6edfaf50f4539c9c1969371fefecf0eabccbf",
        stages=list(STAGES),
        resources=dict(
            maximum_a100s=4,
            cpus_per_worker=12,
            memory_gib_per_worker=64,
            wall_minutes=120,
            partition="bii-gpu",
            reservation="bi_fox_dgx",
            account="bii_dsc_community",
        ),
        budgets_mib={w: BUDGETS[w] for w in workloads},
        methods={w: METHODS[w] for w in workloads},
        originally_requested_workloads=list(BUDGETS),
        enabled_workloads=list(workloads),
        unexecuted_primary_contrasts=0 if set(workloads) == set(BUDGETS) else 4,
        cohorts=cohorts,
        templates={
            w: dict(
                path=str(root / "configs" / f"{w}-historical.json"),
                sha256=sha(root / "configs" / f"{w}-historical.json"),
            )
            for w in workloads
        },
        historical_bindings=bindings,
        environment=environment(),
        output_policy=dict(
            max_new_tokens=64,
            emitted_schedule=64,
            forward_steps=63,
            eos="execute_all_64_then_quality_decode_at_first_terminal_token",
        ),
        tolerances=dict(
            retention_one_max_logit=1e-4,
            synthetic_atol=1e-10,
            synthetic_rtol=1e-8,
            compressed_logit_atol=0.0078125,
            compressed_logit_rtol=0.0078125,
            compressed_contract="independent_compact_strict_plus_unpruned_controlled_normwise_FP16_v2",
            compressed_independent_compact_max_absolute=1e-4,
            original_masked_metric="Linf_difference / max(1,Linf_reference)",
            unpruned_padding_control_required=True,
            legacy_elementwise_8epsilon_results_retained=True,
        ),
        authorization=dict(
            automatic_stage_progression=True,
            current_user_explicit=True,
            confirmation_requires_reviewed_configuration=True,
        ),
        repeats=3,
        warmups=5,
        bootstrap_resamples=10000,
        primary_contrasts=6,
    )
    if scope_registration is not None:
        lock["scope_registration"] = dict(
            path=str(scope_registration), sha256=sha(Path(scope_registration))
        )
        bindings[str(scope_registration)] = sha(Path(scope_registration))
    # Avoid asserting a fresh cohort from an incomplete historical inventory.
    write_once(
        root / "exclusion_report.json",
        dict(
            status="freshness_unverified",
            fresh_confirmation_authorized=False,
            reason=(
                "Historical sample AND reusable-media-group inventory is not exhaustively "
                "certified. Development manifests use already consumed pools."
            ),
            known_bound_development_pools=bindings,
            reserved_confirmation_count=0,
            requested_confirmation={"coco": 256, "milebench": 256},
        ),
    )
    write_once(root / "protocol_lock.json", lock)
    import yaml

    text = yaml.safe_dump(lock, sort_keys=True)
    path = root / "protocol_lock.yaml"
    with path.open("x") as stream:
        stream.write(text)
    write_once(
        root / "readiness.json",
        dict(
            implementation="staged_validation",
            execution="prepared_not_run",
            source_sha=source_sha,
            protocol_sha256=sha(root / "protocol_lock.json"),
            fresh_confirmation="blocked_freshness_and_configuration_review",
            batch="unsupported_adapter_batch_one",
        ),
    )
    print(json.dumps(dict(root=str(root), protocol_sha256=sha(root / "protocol_lock.json"))))


def load_lock(root):
    lock = json.loads((root / "protocol_lock.json").read_text())
    clean_source(SOURCE, lock["source_sha"])
    import mosaickv

    if not Path(mosaickv.__file__).resolve().is_relative_to(SOURCE / "mosaickv/src"):
        raise ValueError("imported package is not the registered source worktree")
    for path, expected in lock["historical_bindings"].items():
        if sha(Path(path)) != expected:
            raise ValueError("historical input/configuration changed")
    for path, expected in lock.get("inherited_gates", {}).items():
        if sha(Path(path)) != expected or json.loads(Path(path).read_text())["status"] != "passed":
            raise ValueError("inherited execution control changed or failed")
    for info in lock["templates"].values():
        if sha(Path(info["path"])) != info["sha256"]:
            raise ValueError("template hash mismatch")
    for cohort in lock["cohorts"].values():
        for info in cohort.values():
            if sha(Path(info["path"])) != info["sha256"]:
                raise ValueError("cohort hash mismatch")
    return lock


def resolved_config(lock, workload, method, cap, *, variant="complete"):
    from mosaickv.config import RunConfig

    template = json.loads(Path(lock["templates"][workload]["path"]).read_text())
    if method != "mosaickv_exact":
        historical = REFERENCE_CONFIG_ROOT / workload / f"{method}.json"
        template = json.loads(historical.read_text())
    template["cache"]["budget_value"] = int(cap * 2**20) if cap else 2**63 - 1
    # Keep each pinned port's validated nominal convention. These runs are
    # driven by physical bytes; changing LOOK-M's nominal ratio independently
    # of its recent/important split is both invalid and a policy change.
    if method == "full_kv":
        template["cache"]["retention_ratio"] = 1.0
    template["diagnostics"]["enabled"] = False
    template["diagnostics"]["capture_full_trace"] = False
    if method == "mosaickv_exact" and variant.startswith("route-"):
        fraction = float(variant.split("-", 1)[1])
        if fraction == 0:
            # Explicit route-zero control is Prompt Top-K, not silently a graph run.
            return resolved_config(lock, workload, "prompt_attention_topk", cap)
        template["graph"]["uncertain_head_fraction"] = fraction
        # Exact-mode planner routing still uses this historical configuration
        # namespace even though prototypes themselves are disabled.
        template["prototypes"]["uncertain_head_fraction"] = fraction
    if variant in (
        "routed-prompt",
        "coverage-off",
        "value-off",
        "cross-modal-off",
        "mechanism-complete",
        "random-13",
        "random-29",
        "random-47",
    ):
        template["graph"]["uncertain_head_fraction"] = 1 / 16
        template["prototypes"]["uncertain_head_fraction"] = 1 / 16
    if variant in ("coverage-off", "routed-prompt"):
        template["selection"]["lambda_g"] = 0.0
        template["selection"]["lambda_m"] = 0.0
    if variant in ("value-off", "routed-prompt"):
        template["utility"]["lambda_v"] = 0.0
        template["utility"]["lambda_o"] = 0.0
    if variant == "cross-modal-off":
        template["graph"]["cross_modal_weight"] = 0.0
    return RunConfig.from_mapping(template)


def jobs_for(lock, stage):
    jobs = []
    if stage in {"smoke", "representation_canary"}:
        return [
            dict(
                workload=w, subset="smoke", indices=[0], stage=stage, arms=registered_arms(w, stage)
            )
            for w in lock["budgets_mib"]
        ]
    subset = {
        "correctness": "validation",
        "scoring": "validation",
        "profiling": "profiler",
        "measurement": "timing",
        "mechanisms": "development",
        "full_graph_pilot": "profiler",
    }[stage]
    for workload in lock["budgets_mib"]:
        count = len(lock["cohorts"][workload][subset]["sample_ids"])
        chunk = (
            2 if stage in {"profiling", "full_graph_pilot"} else 4 if stage == "measurement" else 8
        )
        for start in range(0, count, chunk):
            jobs.append(
                dict(
                    workload=workload,
                    subset=subset,
                    indices=list(range(start, min(count, start + chunk))),
                    stage=stage,
                    arms=registered_arms(workload, stage),
                )
            )
    return jobs


def submit(root, stage, dependency=None):
    lock = load_lock(root)
    jobs = jobs_for(lock, stage)
    spec = root / "jobs" / f"{stage}.json"
    imported = {}
    extra_analysis_dependencies = []
    import_path = root / "imported_correctness.json"
    if stage == "correctness" and import_path.exists():
        from mosaickv.evaluation.fox_campaign import production_runtime_sha

        registration = json.loads(import_path.read_text())
        parent = Path(registration["parent_root"])
        old_lock = json.loads((parent / "protocol_lock.json").read_text())
        if (
            old_lock["tolerances"]["compressed_contract"]
            != lock["tolerances"]["compressed_contract"]
        ):
            raise ValueError("cannot reuse a different numerical validation contract")
        clean_source(Path(old_lock["source_root"]), old_lock["source_sha"])
        if production_runtime_sha(SOURCE) != production_runtime_sha(Path(old_lock["source_root"])):
            raise ValueError("cannot reuse correctness across production-runtime changes")
        if sha(parent / "analysis/smoke/GATE_smoke.json") != registration["smoke_gate_sha256"]:
            raise ValueError("imported smoke gate changed")
        if (
            json.loads((parent / "analysis/smoke/GATE_smoke.json").read_text())["status"]
            != "passed"
        ):
            raise ValueError("imported smoke gate did not pass")
        for index in registration["worker_indices"]:
            job = jobs[index]
            if (
                old_lock["cohorts"][job["workload"]][job["subset"]]["sha256"]
                != lock["cohorts"][job["workload"]][job["subset"]]["sha256"]
            ):
                raise ValueError("imported correctness inputs changed")
            imported[str(index)] = dict(
                parent_root=str(parent), source_sha=old_lock["source_sha"], index=index
            )
        extra_analysis_dependencies = registration["wait_for_jobs"]
    write_once(
        spec,
        dict(
            stage=stage,
            protocol_sha256=sha(root / "protocol_lock.json"),
            jobs=jobs,
            imported_workers=imported,
        ),
    )
    common = ["sbatch", "--parsable", "--account=bii_dsc_community"]
    active = subprocess.check_output(
        ["squeue", "-h", "-u", os.environ["USER"], "-t", "RUNNING,COMPLETING", "-o", "%i|%b"],
        text=True,
    )
    active_gpus = 0
    gpu_jobs = []
    for line in active.splitlines():
        identity, gres = line.split("|", 1)
        match = re.search(r"gpu(?::[a-zA-Z0-9_-]+)?:(\d+)", gres)
        if match:
            active_gpus += int(match[1])
            gpu_jobs.append(identity)
    available = max(0, lock["resources"]["maximum_a100s"] - active_gpus)
    if available == 0:
        dependency = ":".join(([dependency] if dependency else []) + gpu_jobs)
        available = lock["resources"]["maximum_a100s"]
    command = [
        *common,
        "--array="
        + ",".join(str(i) for i in range(len(jobs)) if str(i) not in imported)
        + f"%{available}",
        f"--output={root}/slurm/{stage}-%A_%a.out",
        f"--error={root}/slurm/{stage}-%A_%a.err",
    ]
    if dependency:
        command += [f"--dependency=afterany:{dependency}"]
    command += [
        str(SOURCE / "mosaickv/slurm/evicastkv_worker.sbatch"),
        str(SOURCE),
        str(root),
        stage,
        sha(spec),
    ]
    (root / "slurm").mkdir(parents=True, exist_ok=True)
    gpu = subprocess.check_output(command, text=True).strip().split(";")[0]
    callback = [
        *common,
        "--partition=bii",
        "--cpus-per-task=4",
        "--mem=16G",
        "--time=01:00:00" if stage == "mechanisms" else "--time=00:20:00",
        "--dependency=afterany:" + ":".join([gpu, *extra_analysis_dependencies]),
        f"--output={root}/slurm/{stage}-analysis-%j.out",
        f"--error={root}/slurm/{stage}-analysis-%j.err",
        str(SOURCE / "mosaickv/slurm/evicastkv_completion.sbatch"),
        str(SOURCE),
        str(root),
        stage,
    ]
    cpu = subprocess.check_output(callback, text=True).strip().split(";")[0]
    write_once(
        root / "jobs" / f"{stage}-submission.json",
        dict(array=gpu, analysis=cpu, command=command, callback=callback),
    )
    print(json.dumps(dict(stage=stage, array=gpu, analysis=cpu)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "prepare",
            "dry-run",
            "validate",
            "pilot",
            "development",
            "confirmation",
            "worker",
            "analyze",
            "resume",
        ),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES, default="smoke")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--spec-sha")
    parser.add_argument("--strict-fp16-reductions", action="store_true")
    parser.add_argument("--workloads", nargs="+", choices=tuple(BUDGETS))
    parser.add_argument("--scope-registration", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(
            args.root,
            strict_fp16_reductions=args.strict_fp16_reductions,
            workloads=args.workloads,
            scope_registration=args.scope_registration,
        )
    elif args.mode == "worker":
        from evicastkv_worker import run_worker

        run_worker(args.root, args.stage, args.index, args.spec_sha)
    elif args.mode == "analyze":
        from evicastkv_analyze import analyze

        analyze(args.root, args.stage)
    elif args.mode == "confirmation":
        raise SystemExit(
            "BLOCKED: exhaustive freshness inventory and human-reviewed selected_config.json "
            "required"
        )
    elif args.mode in ("dry-run", "validate"):
        lock = load_lock(args.root)
        jobs = jobs_for(lock, args.stage)
        for job in jobs:
            for method, cap, variant in job["arms"]:
                resolved_config(lock, job["workload"], method, cap, variant=variant)
        print(json.dumps(dict(status="validated", jobs=jobs, GPU_launched=False)))
    else:
        # Resume is selective: a new retry spec is needed rather than overwrite.
        if args.mode == "resume":
            raise SystemExit(
                "Use preserved stage gate failures to create a new immutable retry plan; "
                "no blind resubmission"
            )
        submit(args.root, args.stage)


if __name__ == "__main__":
    main()
