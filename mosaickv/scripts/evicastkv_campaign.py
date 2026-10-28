#!/usr/bin/env python3
"""Finite, hash-bound Fox-review campaign. See --help for executable modes."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from mosaickv.evaluation.fox_campaign import (
    BUDGETS,
    COCO_POOL,
    HISTORICAL,
    METHODS,
    MILE_POOL,
    SEED,
    STAGES,
    TASKS,
    clean_source,
    development_subset,
    digest,
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


def prepare(root: Path):
    """Read previously materialized development pools; no HF access/downloads."""
    source_sha = clean_source(SOURCE)
    templates, cohorts, bindings = {}, {}, {}
    for workload in BUDGETS:
        template_path = HISTORICAL[workload] / "configs/validation/mosaickv_exact-40.json"
        templates[workload] = source_config(workload)
        validate_locked_policy(templates[workload])
        bindings[str(template_path)] = sha(template_path)
        for method in METHODS[workload]:
            method_path = HISTORICAL[workload] / "configs/validation" / f"{method}-40.json"
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
        name="evicastkv_fox_v1",
        source_sha=source_sha,
        source_root=str(SOURCE),
        source_patch_sha256=digest(""),
        seed=SEED,
        historical_base_sha="edb6edfaf50f4539c9c1969371fefecf0eabccbf",
        stages=list(STAGES),
        resources=dict(
            maximum_a100s=4,
            cpus_per_worker=12,
            wall_minutes=120,
            partition="bii-gpu",
            reservation="bi_fox_dgx",
            account="bii_dsc_community",
        ),
        budgets_mib=BUDGETS,
        methods=METHODS,
        cohorts=cohorts,
        templates={
            w: dict(
                path=str(root / "configs" / f"{w}-historical.json"),
                sha256=sha(root / "configs" / f"{w}-historical.json"),
            )
            for w in BUDGETS
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
            compressed_contract="independently_preregistered_fp16_8epsilon_not_retention_one",
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
    for path, expected in lock["historical_bindings"].items():
        if sha(Path(path)) != expected:
            raise ValueError("historical input/configuration changed")
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
        historical = HISTORICAL[workload] / "configs/validation" / f"{method}-40.json"
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
    if variant in (
        "routed-prompt",
        "coverage-off",
        "value-off",
        "cross-modal-off",
        "mechanism-complete",
    ):
        template["graph"]["uncertain_head_fraction"] = 1 / 16
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
    if stage == "smoke":
        return [dict(workload=w, subset="smoke", indices=[0], stage=stage) for w in BUDGETS]
    subset = {
        "correctness": "validation",
        "scoring": "validation",
        "measurement": "timing",
        "mechanisms": "development",
    }[stage]
    for workload in BUDGETS:
        count = len(lock["cohorts"][workload][subset]["sample_ids"])
        chunk = 4 if stage == "measurement" else 8
        for start in range(0, count, chunk):
            jobs.append(
                dict(
                    workload=workload,
                    subset=subset,
                    indices=list(range(start, min(count, start + chunk))),
                    stage=stage,
                )
            )
    return jobs


def submit(root, stage, dependency=None):
    lock = load_lock(root)
    jobs = jobs_for(lock, stage)
    spec = root / "jobs" / f"{stage}.json"
    write_once(spec, dict(stage=stage, protocol_sha256=sha(root / "protocol_lock.json"), jobs=jobs))
    common = ["sbatch", "--parsable", "--account=bii_dsc_community"]
    command = [
        *common,
        f"--array=0-{len(jobs) - 1}%4",
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
        "--time=00:20:00",
        f"--dependency=afterany:{gpu}",
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
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.root)
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
        print(
            json.dumps(
                dict(status="validated", jobs=jobs_for(lock, args.stage), GPU_launched=False)
            )
        )
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
