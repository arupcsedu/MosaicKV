#!/usr/bin/env python3
"""Two consumed-input diagnostics only. No scoring, tuning, or fresh cohorts."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path

from evicastkv_campaign import environment, resolved_config

from mosaickv.config import RunConfig, canonical_config, config_sha256, load_config
from mosaickv.evaluation.fox_campaign import clean_source, write_once
from mosaickv.evaluation.fox_campaign import sha as file_sha

SOURCE = Path(__file__).resolve().parents[2]
PREFIX = Path("/scratch/djy8hg/runs/mosaickv/prefix-native-intermediate-20261029-v2-retry1")
FOX = Path("/scratch/djy8hg/runs/evicastkv/fox-installed-guard-20261029-v2-retry1")
STORAGE = (
    "initial_active_kv_storage_bytes",
    "final_active_kv_storage_bytes",
    "peak_active_kv_storage_bytes",
)
CATALOGS = ("selected_blocks", "packed_slots", "packed_padding_kv_bytes")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return file_sha(Path(path))


def prepare(root, *, compact_source_views=False):
    revision = clean_source(SOURCE)
    if root.exists():
        raise FileExistsError("new immutable diagnostic root required")
    prefix = read(PREFIX / "plan.json")
    fox = read(FOX / "protocol_lock.json")
    bindings = {
        str(PREFIX / "plan.json"): sha(PREFIX / "plan.json"),
        str(FOX / "protocol_lock.json"): sha(FOX / "protocol_lock.json"),
    }
    jobs = []
    # Selection rule is input-only: first ledger entry, never an outcome/failure.
    first = next(j for j in prefix["jobs"] if j["task"] == "llava_description")
    ledger_path = Path(first["ledger"])
    ledger = read(ledger_path)
    sample = ledger["samples"][0]
    arms = []
    for method in ("prompt_attention_topk", "prefixkv_reimpl", "mosaickv_exact"):
        j = next(
            j
            for j in prefix["jobs"]
            if j["task"] == "llava_description" and j["method"] == method and j["cap_mib"] == 120
        )
        cfg = load_config(j["config"])
        bindings[j["config"]] = sha(j["config"])
        arms.append(dict(label=method, config=canonical_config(cfg), config_sha=config_sha256(cfg)))
    jobs.append(
        dict(
            index=0,
            scope="prefix_consumed_description",
            task="llava_description",
            ledger=str(ledger_path),
            sample_id=str(sample["sample_id"]),
            arms=arms,
            logit_tolerance=0.001,
            expected_tokens=1024,
            arithmetic=dict(allow_fp16_reduced_precision_reduction=True),
        )
    )
    consumed = fox["cohorts"]["coco"]["development"]
    ledger_path = Path(consumed["path"])
    if sha(ledger_path) != consumed["sha256"]:
        raise ValueError("consumed COCO ledger changed")
    ledger = read(ledger_path)
    sample = ledger["samples"][0]
    arms = []
    for cap in (20, 40):
        for variant in ("complete", "evidence-output-guarded-v2"):
            cfg = resolved_config(fox, "coco", "mosaickv_exact", cap, variant=variant)
            if not isinstance(cfg, RunConfig):
                cfg = RunConfig.from_mapping(cfg)
            arms.append(
                dict(
                    label=f"{variant}-{cap}",
                    config=canonical_config(cfg),
                    config_sha=config_sha256(cfg),
                )
            )
    jobs.append(
        dict(
            index=1,
            scope="fox_consumed_coco",
            task=ledger["task"],
            ledger=str(ledger_path),
            sample_id=str(sample["sample_id"]),
            arms=arms,
            logit_tolerance=0.0001,
            expected_tokens=64,
            arithmetic=fox["arithmetic"],
        )
    )
    for job in jobs:
        ledger_path = Path(job["ledger"])
        bindings[str(ledger_path)] = sha(ledger_path)
        for media in next(
            s for s in read(ledger_path)["samples"] if str(s["sample_id"]) == job["sample_id"]
        )["media"]:
            path = Path(media["path"])
            if sha(path) != media["sha256"]:
                raise ValueError("consumed media changed")
            bindings[str(path)] = media["sha256"]
    plan = dict(
        source_sha=revision,
        source_root=str(SOURCE),
        jobs=jobs,
        bound_files=bindings,
        sampling_rule="first entry of each already consumed ledger; no outcome selection",
        fresh_evaluation_authorized=False,
        large_campaign_authorized=False,
        frozen_algorithms_and_budgets_unchanged=True,
        maximum_a100s=6,
        requested_workers=2,
        cpus_per_worker=12,
        wall_minutes=30,
        profiler_contaminated=True,
        no_benchmark_scoring=True,
        comparison=(
            "reference_vs_compact_source_views" if compact_source_views else "audit_off_on"
        ),
    )
    write_once(root / "plan.json", plan)
    return sha(root / "plan.json")


def inputs(root, expected):
    if sha(root / "plan.json") != expected:
        raise ValueError("diagnostic plan changed")
    plan = read(root / "plan.json")
    if clean_source(SOURCE) != plan["source_sha"]:
        raise ValueError("immutable source changed")
    for path, value in plan["bound_files"].items():
        if sha(Path(path)) != value:
            raise ValueError(f"bound input changed: {path}")
    return plan


def check_equivalence(control, instrumented, tolerance):
    if not math.isfinite(instrumented["maximum_logit_difference"]):
        raise ValueError("nonfinite logit difference")
    if any(
        type(row["metrics"][key]) is not int or row["metrics"][key] < 1
        for row in (control, instrumented)
        for key in STORAGE
    ):
        raise ValueError("missing physical cache accounting")
    if control["tokens"] != instrumented["tokens"]:
        raise ValueError("instrumentation changed greedy tokens")
    if control["catalogs"] != instrumented["catalogs"]:
        raise ValueError("instrumentation changed selected blocks or packing")
    if any(control["metrics"][k] != instrumented["metrics"][k] for k in STORAGE):
        raise ValueError("instrumentation changed physical cache bytes")
    if instrumented["maximum_logit_difference"] > tolerance:
        raise ValueError("instrumentation failed unchanged logit tolerance")
    if instrumented["metrics"]["residual_kv_bytes"] or instrumented["metrics"]["repair_count"]:
        raise ValueError("exact diagnostic unexpectedly used residuals or repair")


def replay_routing(row):
    ordered = row["all_scores"]
    if (
        ordered != sorted(ordered, key=lambda r: (-r["score"], r["layer"], r["kv_head"]))
        or len(ordered) != row["head_count"]
        or not ordered
        or any(not math.isfinite(r["score"]) or not 0 <= r["score"] <= 1 for r in ordered)
        or len({(r["layer"], r["kv_head"]) for r in ordered}) != len(ordered)
        or [r["rank"] for r in ordered] != list(range(1, len(ordered) + 1))
    ):
        raise ValueError("full routing scores/ranks are inconsistent")
    count = max(1, math.ceil(len(ordered) * row["fraction"]))
    selected = [
        [r["layer"], r["kv_head"]] for r in ordered[:count] if r["score"] >= row["threshold"]
    ]
    selected = selected or [[ordered[0]["layer"], ordered[0]["kv_head"]]]
    if row["selected"] != selected:
        raise ValueError("recorded routing decision does not replay from saved scores")
    return selected


def logit_capture(destination):
    def capture(_module, _args, value):
        if value.ndim != 3 or value.shape[0] != 1:
            raise ValueError("diagnostic LM head must return [1,positions,vocabulary]")
        destination.append(value.detach()[0, -1].float().cpu().numpy().copy())

    return capture


def comparison_config(cfg, *, compact_comparison, candidate):
    if not compact_comparison:
        return cfg
    if cfg.cache.exact_gather_mode != "source_views":
        raise ValueError("compact comparison requires frozen source_views configuration")
    return replace(
        cfg,
        cache=replace(
            cfg.cache,
            exact_gather_mode="source_views" if candidate else "source_views_reference",
        ),
    )


def worker(root, expected, index):
    import numpy as np
    import torch

    from mosaickv.adapters.huggingface import load_hf_adapter
    from mosaickv.adapters.huggingface.parity import (
        compare_cache_reinjection,
        compare_with_generate,
    )
    from mosaickv.backends.hf_runtime import HuggingFaceMosaicKVModel
    from mosaickv.evaluation.messages import MediaItem, build_multimodal_messages
    from mosaickv.evaluation.model import EvaluationRequest
    from mosaickv.evaluation.paper_native_lanes import _media_payload, load_paper_native_ledger
    from mosaickv.evaluation.tasks import default_task_registry
    from mosaickv.evaluation.worker_support import settle
    from mosaickv.execution_audit import collect_execution_audit
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    plan = inputs(root, expected)
    compact_comparison = plan.get("comparison") == "reference_vs_compact_source_views"
    job = plan["jobs"][index]
    output = root / f"workers/{index}"
    if output.exists():
        raise FileExistsError("worker attempts are immutable; no blind retry")
    output.mkdir(parents=True)
    result = dict(
        status="failed",
        scope=job["scope"],
        sample_id=job["sample_id"],
        source_sha=plan["source_sha"],
        plan_sha256=expected,
        completed_arms=[],
        fresh_evaluation_authorized=False,
        large_campaign_authorized=False,
    )
    started = time.monotonic()
    audit = None
    try:
        if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name(0):
            raise ValueError("one allocated A100 required")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ValueError("cuBLAS workspace must be set before Python starts")
        torch.set_num_threads(12)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = job["arithmetic"][
            "allow_fp16_reduced_precision_reduction"
        ]
        cfg = RunConfig.from_mapping(job["arms"][0]["config"])
        torch.manual_seed(cfg.execution.seed)
        ledger = load_paper_native_ledger(
            job["ledger"],
            expected_sha256=sha(job["ledger"]),
            task=default_task_registry().resolve(job["task"]),
            dataset_revision=cfg.dataset.revision,
        )
        sample = next(s for s in ledger.samples if s.sample_id == job["sample_id"])
        messages = build_multimodal_messages(
            sample.prompt, tuple(MediaItem(m.kind, _media_payload(m)) for m in sample.media)
        )
        adapter = load_hf_adapter(
            cfg.model.id,
            revision=cfg.model.revision,
            model_kwargs=_model_kwargs(cfg, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        write_once(
            output / "manifest.json",
            dict(
                source_sha=plan["source_sha"],
                plan_sha256=expected,
                environment=environment(),
                cuda_runtime=torch.version.cuda,
                gpu=asdict(capture_gpu_environment(torch)),
                model=asdict(cfg.model),
                dataset=asdict(cfg.dataset),
                execution=asdict(cfg.execution),
                arithmetic=job["arithmetic"],
                seed=cfg.execution.seed,
                measurement_type="diagnostic_profile",
                profiler_contaminated=True,
                benchmark_references_used=False,
                slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                scope=job["scope"],
                arms=job["arms"],
            ),
        )
        prepared = adapter.prepare_inputs(messages)
        parity = {}
        for label, fn in (
            ("generate", compare_with_generate),
            ("reinjection", compare_cache_reinjection),
        ):
            report = fn(adapter, prepared, max_new_tokens=16)
            parity[label] = asdict(report)
            if (
                report.token_agreement != 1
                or report.maximum_logit_difference > job["logit_tolerance"]
            ):
                raise ValueError(f"FullKV {label} parity failed")
        write_once(output / "fullkv_parity.json", parity)
        head = adapter.model.get_output_embeddings()
        for arm in job["arms"]:
            if time.monotonic() - started > 1500:
                raise TimeoutError("bounded diagnostic wall budget exhausted")
            cfg = RunConfig.from_mapping(arm["config"])
            if config_sha256(cfg) != arm["config_sha"]:
                raise ValueError("arm configuration changed")
            pair = []
            reference_logits = None
            for enabled in (False, True):
                suffix = (
                    ("-compact" if enabled else "-reference")
                    if compact_comparison
                    else ("-instrumented" if enabled else "-control")
                )
                label = arm["label"] + suffix
                execution_cfg = comparison_config(
                    cfg, compact_comparison=compact_comparison, candidate=enabled
                )
                profiling = enabled or compact_comparison
                settle(adapter)
                logits = []

                hook = head.register_forward_hook(logit_capture(logits))
                runtime = HuggingFaceMosaicKVModel(
                    adapter, execution_cfg, trace_directory=output / "traces" / label
                )
                try:
                    if profiling:
                        with collect_execution_audit(torch_module=torch) as audit:
                            generation = runtime.generate(
                                EvaluationRequest(
                                    label, sample.sample_id, job["task"], messages, {}
                                )
                            )
                    else:
                        generation = runtime.generate(
                            EvaluationRequest(label, sample.sample_id, job["task"], messages, {})
                        )
                finally:
                    hook.remove()
                    if profiling and audit is not None:
                        payload = audit.payload()
                        write_once(output / f"profiles/{label}.json", payload)
                        audit = None
                        if cfg.method.value == "mosaickv_exact" and (
                            not payload["routing"] or not payload["phases"]
                        ):
                            raise ValueError(
                                "EviCastKV routing/tier instrumentation was not exercised"
                            )
                traces = list((output / "traces" / label).rglob("*.json"))
                if len(traces) != 1:
                    raise ValueError("one complete trace per diagnostic request required")
                trace = read(traces[0])
                array = np.stack(logits)
                if array.shape[0] != job["expected_tokens"] or not np.isfinite(array).all():
                    raise ValueError("missing or nonfinite diagnostic logits")
                difference = (
                    0.0
                    if reference_logits is None
                    else float(np.max(np.abs(array - reference_logits)))
                )
                if reference_logits is None:
                    reference_logits = array
                row = dict(
                    label=label,
                    metrics=asdict(generation.metrics),
                    tokens=trace["generated_token_ids"],
                    catalogs={k: trace[k] for k in CATALOGS},
                    maximum_logit_difference=difference,
                    trace_path=str(traces[0]),
                    trace_sha256=sha(traces[0]),
                    profiler_contaminated=True,
                    execution_config=canonical_config(execution_cfg),
                    execution_config_sha=config_sha256(execution_cfg),
                    frozen_config_sha=arm["config_sha"],
                )
                if len(row["tokens"]) != job["expected_tokens"]:
                    raise ValueError("truncated diagnostic output")
                if row["metrics"][STORAGE[0]] != cfg.cache.budget_value:
                    raise ValueError("physical initial budget changed")
                write_once(output / f"requests/{label}.json", row)
                pair.append(row)
                del runtime, generation, logits
            check_equivalence(*pair, job["logit_tolerance"])
            result["completed_arms"].append(
                dict(
                    label=arm["label"],
                    equivalence="passed",
                    maximum_logit_difference=pair[1]["maximum_logit_difference"],
                )
            )
            del pair, reference_logits
        result["status"] = "passed_instrumentation_equivalence_only"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result["wall_seconds"] = time.monotonic() - started
        write_once(output / "result.json", result)


def analyze(root, expected):
    inputs(root, expected)
    plan = read(root / "plan.json")
    failures, workers, phases, routes, representations = [], [], [], [], []
    for index in (0, 1):
        output = root / f"workers/{index}"
        path = output / "result.json"
        if not path.exists():
            failures.append(dict(worker=index, error="missing worker result"))
            continue
        result = read(path)
        workers.append(result)
        if result["status"] != "passed_instrumentation_equivalence_only":
            failures.append(result)
        for profile_path in sorted((output / "profiles").glob("*.json")):
            profile = read(profile_path)
            representations.extend(
                dict(worker=index, arm=profile_path.stem, **row)
                for row in profile.get("exact_representations", [])
            )
            if plan.get("comparison") == "reference_vs_compact_source_views":
                rows = profile.get("exact_representations", [])
                is_compact = profile_path.stem.endswith("-compact")
                compact_rows = [r for r in rows if r["sequence_type"] == "_ExactViewSequence"]
                if is_compact and any(r["materialized_block_views"] for r in compact_rows):
                    failures.append(
                        dict(worker=index, error="compact path materialized block views")
                    )
                if "prefixkv_reimpl" not in profile_path.stem and (
                    not rows or (is_compact and not compact_rows)
                ):
                    failures.append(
                        dict(worker=index, error="missing exact representation evidence")
                    )
            phases.extend(dict(worker=index, arm=profile_path.stem, **r) for r in profile["phases"])
            for row in profile["routing"]:
                try:
                    replay_routing(row)
                except ValueError as exc:
                    failures.append(dict(worker=index, error=str(exc)))
                routes.append(dict(worker=index, arm=profile_path.stem, **row))
    import pandas as pd

    out = root / "analysis"
    if out.exists():
        raise FileExistsError("write-once derived analysis required")
    out.mkdir()
    pd.DataFrame(phases).to_parquet(out / "tier_operations.parquet", index=False)
    write_once(out / "routing_scores.json", routes)
    write_once(out / "exact_representations.json", representations)
    gate = dict(
        status="passed_diagnostic_only" if not failures and len(workers) == 2 else "failed",
        workers=workers,
        failures=failures,
        fresh_evaluation_authorized=False,
        large_campaign_authorized=False,
        superiority_established=False,
        profiler_contaminated=True,
    )
    write_once(out / "GATE.json", gate)
    table = pd.DataFrame(phases)
    lines = [
        "# EviCastKV consumed-input instrumentation diagnostic",
        "",
        "Not benchmark-quality or production-latency evidence. No fresh identities.",
        "",
        json.dumps(gate, indent=2),
        "",
        "## Tier operations (exclusive synchronized wall seconds)",
        "",
    ]
    if len(table):
        lines += [
            "```text",
            table.sort_values("exclusive_wall_seconds", ascending=False)[
                ["worker", "arm", "operation", "exclusive_wall_seconds", "inclusive_wall_seconds"]
            ]
            .head(35)
            .to_string(index=False),
            "```",
        ]
    if plan.get("comparison") == "reference_vs_compact_source_views" and len(table):
        paired = []
        operations = (
            "exact.descriptor_gather_and_views",
            "exact.tier_invariants",
            "tier.complete_construction",
            "state.create_and_account_storage",
        )
        for worker_index in (0, 1):
            for arm in plan["jobs"][worker_index]["arms"]:
                for operation in operations:
                    subset = table[(table.worker == worker_index) & (table.operation == operation)]
                    reference = subset[subset.arm == arm["label"] + "-reference"]
                    compact = subset[subset.arm == arm["label"] + "-compact"]
                    if len(reference) and len(compact):
                        a = float(reference.inclusive_wall_seconds.sum())
                        b = float(compact.inclusive_wall_seconds.sum())
                        paired.append(
                            dict(
                                worker=worker_index,
                                arm=arm["label"],
                                operation=operation,
                                reference_seconds=a,
                                compact_seconds=b,
                                reduction_fraction=1 - b / a if a > 0 else None,
                            )
                        )
        pd.DataFrame(paired).to_parquet(out / "representation_timing_pairs.parquet", index=False)
        lines += [
            "",
            "## Reference versus compact representation",
            "",
            "Same consumed inputs; synchronized, single-pair profiling, not a latency claim.",
            "Operation rows are nested and must not be added together.",
            "",
            "```text",
            pd.DataFrame(paired).to_string(index=False),
            "```",
            "",
            "Exact view counts are recorded in exact_representations.json. Selected payload "
            "sizes are not the full aliased source allocation; request records separately "
            "validate actual initial/final/peak packed backing storage.",
        ]
    lines += [
        "",
        "Full routing scores and ranks are preserved in routing_scores.json.",
        "Fresh evaluation and larger campaigns remain blocked regardless of this diagnostic.",
    ]
    lines += ["", "## Deterministic route replay", ""]
    for route in routes:
        top = route["all_scores"][:3]
        lines.append(
            json.dumps(
                dict(
                    worker=route["worker"],
                    arm=route["arm"],
                    call=route["label"],
                    selected=route["selected"],
                    top_three=top,
                    winner_runner_up_margin=top[0]["score"] - top[1]["score"]
                    if len(top) > 1
                    else None,
                )
            )
        )
    if failures:
        lines += [
            "",
            "Next: preserve failures and inspect instrumentation-equivalence "
            "mismatches before any optimization. Do not retry blindly.",
        ]
    elif len(table):
        dominant = table.sort_values("exclusive_wall_seconds", ascending=False).iloc[0]
        lines += [
            "",
            "Next: review the dominant exclusive operation "
            f"`{dominant.operation}` using these consumed-input captures. "
            "An execution optimization must preserve exact selection, tensors, logical "
            "positions and bytes; this gate does not authorize fresh evaluation.",
        ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    write_once(
        out / "manifest.json",
        dict(
            source_sha=clean_source(SOURCE),
            plan_sha256=expected,
            input_sha256={str(p): sha(p) for p in (root / "workers").rglob("*.json")},
            artifact_sha256={p.name: sha(p) for p in out.iterdir()},
        ),
    )


def submit(root, expected):
    inputs(root, expected)
    command = [
        "sbatch",
        "--parsable",
        "--account=bii_dsc_community",
        "--job-name=mk-evicast-consumed-profile",
        "--partition=bii-gpu",
        "--reservation=bi_fox_dgx",
        "--nodes=1",
        "--ntasks=1",
        "--gres=gpu:a100:1",
        "--cpus-per-task=12",
        "--mem=64G",
        "--time=00:30:00",
        "--array=0-1%2",
        f"--output={root}/slurm/worker-%A_%a.out",
        f"--error={root}/slurm/worker-%A_%a.err",
        str(SOURCE / "mosaickv/slurm/evicastkv_consumed_instrumentation.sbatch"),
        str(SOURCE),
        str(root),
        expected,
        "worker",
    ]
    (root / "slurm").mkdir(exist_ok=True)
    gpu = subprocess.check_output(command, text=True, timeout=30).strip().split(";")[0]
    write_once(root / "submissions/gpu.json", dict(job_id=gpu, command=command))
    callback = [
        "sbatch",
        "--parsable",
        "--account=bii_dsc_community",
        "--partition=bii",
        "--job-name=mk-evicast-profile-analysis",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=2",
        "--mem=8G",
        "--time=00:10:00",
        f"--dependency=afterany:{gpu}",
        f"--output={root}/slurm/analysis-%j.out",
        f"--error={root}/slurm/analysis-%j.err",
        str(SOURCE / "mosaickv/slurm/evicastkv_consumed_instrumentation.sbatch"),
        str(SOURCE),
        str(root),
        expected,
        "analyze",
    ]
    cpu = subprocess.check_output(callback, text=True, timeout=30).strip().split(";")[0]
    write_once(
        root / "submission.json",
        dict(gpu_array=gpu, cpu_analysis=cpu, command=command, callback=callback),
    )
    # Attach/preserve the afterany analyzer before admission registration can
    # fail, so an orchestration error cannot silently strand completed results.
    from control_a100_schedule import tick

    policy = dict(
        timezone="America/New_York",
        maximum_worker_minutes=120,
        maximum_a100s=6,
        six_a100s_explicitly_authorized=True,
        arrays=[gpu],
        per_lane_static_throttle=2,
        other_lane_reserved_slots=4,
    )
    write_once(root / "scheduler/policy.json", policy)
    write_once(
        root / "scheduler/registration.json", tick(policy, {}, root / "scheduler/state.json", False)
    )
    print(json.dumps(dict(gpu_array=gpu, cpu_analysis=cpu)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "submit", "worker", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha")
    parser.add_argument("--index", type=int, choices=(0, 1))
    parser.add_argument("--compact-source-views", action="store_true")
    args = parser.parse_args()
    if args.mode == "prepare":
        print(prepare(args.root, compact_source_views=args.compact_source_views))
    elif args.mode == "submit":
        submit(args.root, args.plan_sha)
    elif args.mode == "worker":
        worker(args.root, args.plan_sha, args.index)
    else:
        analyze(args.root, args.plan_sha)


if __name__ == "__main__":
    main()
