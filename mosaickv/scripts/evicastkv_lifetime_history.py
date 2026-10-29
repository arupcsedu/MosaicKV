#!/usr/bin/env python3
"""Two consumed native-metric inputs. No automatic fresh/paper promotion."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path

from evicastkv_campaign import environment

from mosaickv.config import (
    BoundedDecodeConfig,
    IncrementalHistoryConfig,
    RunConfig,
    canonical_config,
    config_sha256,
    load_config,
)
from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once
from mosaickv.types import BudgetUnit, MosaicKVMethod

SOURCE = Path(__file__).resolve().parents[2]
PARENT = Path("/scratch/djy8hg/runs/mosaickv/prefixkv/full-campaign-e4de390-20261019-v6")
CONSUMED = Path("/scratch/djy8hg/runs/mosaickv/prefix-native-intermediate-20261029-v2-retry1")
PYTHON = "/scratch/djy8hg/env/mosaickv/bin/python"
STORAGE = tuple(f"{x}_active_kv_storage_bytes" for x in ("initial", "final", "peak"))


def read(path):
    return json.loads(Path(path).read_text())


def prepare(root):
    source_sha = clean_source(SOURCE)
    if root.exists():
        raise FileExistsError("write-once run root required")
    parent = read(PARENT / "plan.json")
    consumed = read(CONSUMED / "plan.json")
    bound = {str(p): sha(p) for p in (PARENT / "plan.json", CONSUMED / "plan.json")}
    jobs = []
    for index, task in enumerate(("llava_description", "mmvet_prefixkv")):
        old = next(j for j in consumed["jobs"] if j["task"] == task)
        ledger = Path(old["ledger"])
        bound[str(ledger)] = sha(ledger)
        sample = read(ledger)["samples"][0]
        arms = []
        bases = {}
        for method in ("full_kv", "prefixkv_reimpl", "mosaickv_exact"):
            j = next(
                j
                for j in parent["jobs"]
                if j["task"] == task and j["lane"] == "paper_native" and j["method"] == method
            )
            path = Path(j["config"])
            bound[str(path)] = sha(path)
            bases[method] = load_config(path)
            if method != "mosaickv_exact":
                arms.append(
                    {"label": method + "-native-control", "config": canonical_config(bases[method])}
                )
            if method == "prefixkv_reimpl":
                profile = Path(bases[method].prefixkv.profile_path)
                bound[str(profile)] = sha(profile)
        base = bases["mosaickv_exact"]
        if base.forecasting.draft_steps != 0:
            raise ValueError("do not alter frozen prompt forecasting for this transport diagnostic")
        common = replace(
            base,
            cache=replace(
                base.cache,
                budget_value=90 * 2**20,
                budget_unit=BudgetUnit.BYTES,
                retention_ratio=math.nextafter(1.0, 0.0),
                block_size=1,
                physical_budget_block_size=1,
            ),
            evidence_policy=None,
            output_preserving=None,
        )
        arms.append({"label": "evicast-append-anchor-90mib", "config": canonical_config(common)})
        for method in (MosaicKVMethod.PROMPT_ATTENTION_TOPK, MosaicKVMethod.MOSAICKV_EXACT):
            for policy in ("sink_recent", "recent_importance"):
                method_base = common
                if method is MosaicKVMethod.PROMPT_ATTENTION_TOPK:
                    method_base = replace(
                        bases["full_kv"],
                        method=method,
                        cache=common.cache,
                        forecasting=replace(common.forecasting, enabled=False),
                    )
                cfg = replace(
                    method_base,
                    method=method,
                    cache=replace(common.cache, budget_value=192 * 2**20),
                    bounded_decode=BoundedDecodeConfig(
                        192, 4, incremental_history=IncrementalHistoryConfig(16, 64, policy)
                    ),
                )
                arms.append(
                    {"label": f"{method.value}-{policy}-192mib", "config": canonical_config(cfg)}
                )
        for arm in arms:
            cfg = RunConfig.from_mapping(arm["config"])
            if any(
                getattr(cfg, k) != getattr(base, k)
                for k in ("model", "dataset", "execution", "generation")
            ):
                raise ValueError("model, data, arithmetic, or generation changed")
            arm["config_sha"] = config_sha256(cfg)
        for media in sample["media"]:
            path = Path(media["path"])
            if sha(path) != media["sha256"]:
                raise ValueError("consumed media changed")
            bound[str(path)] = media["sha256"]
        jobs.append(
            dict(
                index=index,
                task=task,
                sample_id=str(sample["sample_id"]),
                ledger=str(ledger),
                arms=arms,
            )
        )
    write_once(
        root / "plan.json",
        dict(
            source_sha=source_sha,
            jobs=jobs,
            bound_files=bound,
            sample_rule="first already-consumed ledger entry, input-only, no outcome selection",
            scope="consumed_development_diagnostic_not_confirmation",
            history_capacity=192,
            recent_tokens=64,
            sinks=4,
            chunk_tokens=16,
            importance_decay=0.95,
            model_forward_tokens=1024,
            quality_warmups=0,
            quality_repetitions=1,
            generation_quality_and_teacher_forcing_are_separate=True,
            quality_margin_ppl=0.0,
            quality_margin_rouge=0.0,
            quality_not_statistically_established_by_two_inputs=True,
            maximum_a100s=6,
            requested_workers=2,
            cpus_per_worker=12,
            wall_minutes=60,
            fresh_evaluation_authorized=False,
            large_campaign_authorized=False,
            whole_request_peak_requires_separate_complete_prefill_instrumentation=True,
        ),
    )
    return sha(root / "plan.json")


def inputs(root, expected):
    if sha(root / "plan.json") != expected:
        raise ValueError("plan changed")
    plan = read(root / "plan.json")
    if clean_source(SOURCE) != plan["source_sha"]:
        raise ValueError("source changed")
    for path, digest in plan["bound_files"].items():
        if sha(Path(path)) != digest:
            raise ValueError(f"bound file changed: {path}")
    return plan


def worker(root, expected, index):
    import torch

    from mosaickv.adapters.huggingface import load_hf_adapter
    from mosaickv.adapters.huggingface.parity import (
        compare_cache_reinjection,
        compare_with_generate,
    )
    from mosaickv.backends.hf_runtime import HuggingFaceMosaicKVModel, compare_runtime_retention_one
    from mosaickv.evaluation.messages import MediaItem, build_multimodal_messages
    from mosaickv.evaluation.metrics.scorers import rouge_l_f1
    from mosaickv.evaluation.model import EvaluationRequest
    from mosaickv.evaluation.paper_native_lanes import _media_payload, load_paper_native_ledger
    from mosaickv.evaluation.tasks import default_task_registry
    from mosaickv.evaluation.worker_support import settle
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    plan = inputs(root, expected)
    job = plan["jobs"][index]
    out = root / f"workers/{index}"
    if out.exists():
        raise FileExistsError("immutable attempts; use a new retry root")
    out.mkdir(parents=True)
    result = dict(
        status="failed",
        source_sha=plan["source_sha"],
        plan_sha256=expected,
        task=job["task"],
        sample_id=job["sample_id"],
        completed_arms=[],
        failures=[],
    )
    started = time.monotonic()
    rows = []
    try:
        if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name(0):
            raise ValueError("one A100 required")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ValueError("workspace must be set before Python starts")
        torch.set_num_threads(12)
        torch.use_deterministic_algorithms(True)
        first = RunConfig.from_mapping(job["arms"][0]["config"])
        torch.manual_seed(first.execution.seed)
        adapter = load_hf_adapter(
            first.model.id,
            revision=first.model.revision,
            model_kwargs=_model_kwargs(first, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        ledger = load_paper_native_ledger(
            job["ledger"],
            expected_sha256=sha(Path(job["ledger"])),
            task=default_task_registry().resolve(job["task"]),
            dataset_revision=first.dataset.revision,
        )
        sample = next(s for s in ledger.samples if s.sample_id == job["sample_id"])
        messages = build_multimodal_messages(
            sample.prompt, tuple(MediaItem(m.kind, _media_payload(m)) for m in sample.media)
        )
        write_once(
            out / "manifest.json",
            dict(
                source_sha=plan["source_sha"],
                plan_sha256=expected,
                environment=environment(),
                cuda_runtime=torch.version.cuda,
                gpu=asdict(capture_gpu_environment(torch)),
                model=asdict(first.model),
                dataset=asdict(first.dataset),
                execution=asdict(first.execution),
                measurement_type="consumed_diagnostic",
                slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                all_arms=job["arms"],
                no_paper_or_confirmation_claim=True,
            ),
        )
        prepared = adapter.prepare_inputs(messages)
        parity = {}
        for label, fn in [
            ("generate", compare_with_generate),
            ("reinjection", compare_cache_reinjection),
        ]:
            check = fn(adapter, prepared, max_new_tokens=16)
            parity[label] = asdict(check)
            if check.token_agreement != 1 or check.maximum_logit_difference > 0.001:
                raise ValueError("FullKV parity failed")
        identity = RunConfig.from_mapping(
            job.get("retention_one_config", job["arms"][-1]["config"])
        )
        identity = replace(
            identity,
            cache=replace(identity.cache, retention_ratio=1.0, budget_value=2**63 - 1),
            generation=replace(identity.generation, max_new_tokens=16),
        )
        check = compare_runtime_retention_one(adapter, prepared, identity)
        parity["incremental_retention_one"] = asdict(check)
        if check.token_agreement != 1 or check.maximum_logit_difference > 0.001:
            raise ValueError("incremental retention-one parity failed")
        write_once(out / "parity.json", parity)
        reference = None
        if job.get("frozen_fullkv_reference"):
            reference_path = Path(job["frozen_fullkv_reference"])
            record = read(reference_path)
            if (
                record["sample_id"] != sample.sample_id
                or record["model"] != asdict(first.model)
                or record["generation"] != asdict(first.generation)
            ):
                raise ValueError("reused FullKV reference controls differ")
            reference = record["answer"]
            write_once(out / "fullkv_reference.json", record)
            if sha(out / "fullkv_reference.json") != sha(reference_path):
                raise ValueError("reference copy changed content")
        for arm in job["arms"]:
            if time.monotonic() - started > 3300:
                raise TimeoutError("bounded worker wall budget")
            cfg = RunConfig.from_mapping(arm["config"])
            row = dict(
                sample_id=sample.sample_id,
                task=job["task"],
                label=arm["label"],
                method=cfg.method.value,
                config_sha=arm["config_sha"],
                status="failed",
                error=None,
                manifest_path=str(out / "manifest.json"),
            )
            runtime = HuggingFaceMosaicKVModel(
                adapter, cfg, trace_directory=out / "traces" / arm["label"]
            )
            try:
                settle(adapter)
                request = EvaluationRequest(
                    arm["label"] + "-generation",
                    sample.sample_id,
                    job["task"],
                    messages,
                    {"_mosaickv_capture_kv_ownership": True},
                )
                generation = runtime.generate(request)
                if cfg.method.is_full_cache:
                    reference = generation.answer
                    write_once(
                        out / "fullkv_reference.json",
                        dict(
                            sample_id=sample.sample_id,
                            answer=reference,
                            source_sha=plan["source_sha"],
                            model=asdict(cfg.model),
                            generation=asdict(cfg.generation),
                        ),
                    )
                if reference is None:
                    raise ValueError("FullKV reference must precede compressed arms")
                target = sample.teacher_forcing_target
                if not target:
                    raise ValueError("missing native teacher-forcing target")
                loss = runtime.teacher_forced_nll(
                    EvaluationRequest(
                        arm["label"] + "-ppl",
                        sample.sample_id,
                        job["task"],
                        messages,
                        {"_mosaickv_capture_kv_ownership": True},
                    ),
                    target,
                )
                traces = list((out / "traces" / arm["label"]).rglob("*.json"))
                if len(traces) != 2:
                    raise ValueError("both generation and PPL traces required")
                phase = {read(p)["execution_phase"]: read(p) for p in traces}
                memory_audit = None
                memory_paths = []
                if plan.get("separate_memory_replay", False):
                    # Do not interpret per-concatenation observer overhead as
                    # algorithm latency. Replay privately for memory; demand
                    # identical tokens before admitting its accounting.
                    settle(adapter)
                    memory_runtime = HuggingFaceMosaicKVModel(
                        adapter, cfg, trace_directory=out / "memory_audit" / arm["label"]
                    )
                    memory_generation = memory_runtime.generate(
                        EvaluationRequest(
                            arm["label"] + "-memory-audit",
                            sample.sample_id,
                            job["task"],
                            messages,
                            {
                                "_mosaickv_capture_kv_ownership": True,
                                "_mosaickv_capture_kv_transactions": True,
                            },
                        )
                    )
                    memory_paths = list((out / "memory_audit" / arm["label"]).rglob("*.json"))
                    if len(memory_paths) != 1:
                        raise ValueError("one private memory-replay trace required")
                    memory_audit = read(memory_paths[0])
                    if (
                        memory_audit["generated_token_ids"]
                        != phase["generation"]["generated_token_ids"]
                        or memory_generation.answer != generation.answer
                    ):
                        raise ValueError("memory observer changed generated tokens")
                    audited_peak = memory_audit["request_KV_transactions"][
                        "decode_transaction_storage"
                    ]["peak_storage_bytes"]
                    if audited_peak < generation.metrics.peak_active_kv_storage_bytes:
                        raise ValueError("transaction peak is below sampled active-cache peak")
                    if cfg.bounded_decode is not None and audited_peak > cfg.cache.budget_value:
                        raise ValueError("audited decode transaction exceeded lifetime cap")
                    del memory_runtime, memory_generation
                if cfg.bounded_decode is not None and (
                    cfg.bounded_decode.incremental_history is not None
                    and cfg.bounded_decode.incremental_history.fixed_prompt_reservation
                ):
                    a, b = phase["generation"], phase["teacher_forced"]
                    if (
                        a["packed_prefill_active_kv_bytes"] != b["packed_prefill_active_kv_bytes"]
                        or a["selected_blocks"] != b["selected_blocks"]
                    ):
                        raise ValueError("generation and PPL did not use the same prompt selection")
                if generation.metrics.generated_tokens != 1024:
                    raise ValueError("truncated generation")
                if (
                    phase["teacher_forced"]["teacher_forced_evaluation"]["evaluated_tokens"]
                    != loss.evaluated_tokens
                ):
                    raise ValueError("incomplete NLL/token record")
                if cfg.bounded_decode is not None:
                    for trace in phase.values():
                        if trace["source_KV_retirement"]["remaining_source_tensor_objects"]:
                            raise ValueError("inactive source KV still retained")
                        bounded = trace["bounded_decode"]
                        peak = (
                            bounded["owned_decode_KV_including_growth_peak"]["peak_storage_bytes"]
                            if "owned_decode_KV_including_growth_peak" in bounded
                            else trace["cache_lifecycle"]["peak_active_kv_storage_bytes"]
                        )
                        if peak > cfg.cache.budget_value:
                            raise ValueError("hard transient cap exceeded")
                eos = getattr(getattr(adapter.processor, "tokenizer", None), "eos_token_id", None)
                eos_ids = set(
                    eos if isinstance(eos, (list, tuple)) else (() if eos is None else (eos,))
                )
                eos_positions = [
                    i + 1
                    for i, t in enumerate(phase["generation"]["generated_token_ids"])
                    if t in eos_ids
                ]
                first_eos = min(eos_positions) if eos_positions else None
                row.update(
                    status="completed",
                    answer=generation.answer,
                    effective_method=generation.effective_method,
                    metrics=asdict(generation.metrics),
                    rouge_l_f1=rouge_l_f1(generation.answer, [reference]),
                    nll_sum=loss.negative_log_likelihood_sum,
                    evaluated_tokens=loss.evaluated_tokens,
                    teacher_forced_ppl=math.exp(
                        loss.negative_log_likelihood_sum / loss.evaluated_tokens
                    ),
                    fullkv_reference_sha256=sha(out / "fullkv_reference.json"),
                    traces={str(p): sha(p) for p in traces},
                    request_owned_peak_bytes=phase["generation"]["request_KV_ownership"][
                        "peak_storage_bytes"
                    ],
                    request_peak_not_certified_complete_prefill=True,
                    memory_audit_traces={str(p): sha(p) for p in memory_paths},
                    audited_request_owned_peak_bytes=(
                        None
                        if memory_audit is None
                        else memory_audit["request_KV_ownership"]["peak_storage_bytes"]
                    ),
                    audited_decode_transaction_peak_bytes=(
                        None
                        if memory_audit is None
                        else memory_audit["request_KV_transactions"]["decode_transaction_storage"][
                            "peak_storage_bytes"
                        ]
                    ),
                    pytorch_request_peak_allocated_bytes=(
                        None if memory_audit is None else memory_audit["peak_gpu_memory_allocated"]
                    ),
                    memory_observer_token_parity=(None if memory_audit is None else True),
                    first_eos_generated_position=first_eos,
                    forced_tokens_after_first_eos=(
                        None
                        if first_eos is None
                        else generation.metrics.generated_tokens - first_eos
                    ),
                    first_eviction_can_affect_pre_eos_output=(
                        None
                        if cfg.bounded_decode is None
                        else (
                            first_eos is None
                            or first_eos > min(phase["generation"]["bounded_decode"].get(
                                "generated_capacities_per_layer",
                                [cfg.bounded_decode.generated_capacity])) + 1
                        )
                    ),
                )
                result["completed_arms"].append(arm["label"])
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                result["failures"].append(dict(label=arm["label"], error=row["error"]))
            finally:
                rows.append(row)
                write_once(out / f"rows/{arm['label']}.json", row)
                del runtime
                settle(adapter)
        result["status"] = "completed_consumed_diagnostic" if not result["failures"] else "failed"
    except Exception as exc:
        result["failures"].append(dict(error=f"{type(exc).__name__}: {exc}"))
        raise
    finally:
        result["wall_seconds"] = time.monotonic() - started
        (out / "raw.jsonl").write_text(
            "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows)
        )
        write_once(out / "result.json", result)


def analyze(root, expected):
    import pandas as pd

    plan = inputs(root, expected)
    results = []
    rows = []
    failures = []
    for i in (0, 1):
        out = root / f"workers/{i}"
        if not (out / "result.json").exists():
            failures.append(dict(worker=i, error="missing result"))
            continue
        result = read(out / "result.json")
        results.append(result)
        failures.extend(result["failures"])
        for p in sorted((out / "rows").glob("*.json")):
            rows.append(read(p))
    out = root / "analysis"
    if out.exists():
        raise FileExistsError("write-once analysis")
    out.mkdir()
    summary = []
    for r in rows:
        if r["status"] != "completed":
            continue
        m = r["metrics"]
        summary.append(
            {k: r[k] for k in ("task", "sample_id", "label", "teacher_forced_ppl", "rouge_l_f1")}
            | {k: m[k] for k in (*STORAGE, "end_to_end_time")}
            | {"owned_request_peak": r["request_owned_peak_bytes"]}
        )
    table = pd.DataFrame(summary)
    table.to_parquet(out / "summary.parquet", index=False)
    checks = []
    if len(summary) == 14:
        for task, g in table.groupby("task"):
            baseline = g[g.label == "prefixkv_reimpl-native-control"].iloc[0]
            for r in g[g.label.str.startswith("mosaickv_exact-")].itertuples():
                checks.append(
                    dict(
                        task=task,
                        label=r.label,
                        smaller_all_decode_boundaries=all(
                            getattr(r, k) < baseline[k] for k in STORAGE
                        ),
                        ppl_point_not_worse=bool(
                            r.teacher_forced_ppl <= baseline.teacher_forced_ppl
                        ),
                        rouge_point_not_worse=bool(r.rouge_l_f1 >= baseline.rouge_l_f1),
                        no_statistical_claim_n_one_per_task=True,
                    )
                )
    gate = dict(
        status="complete_consumed_diagnostic" if len(summary) == 14 and not failures else "failed",
        checks=checks,
        failures=failures,
        worker_results=results,
        quality_preservation_statistically_established=False,
        whole_request_peak_certified=False,
        fresh_evaluation_authorized=False,
        large_campaign_authorized=False,
        superiority_established=False,
    )
    write_once(out / "GATE.json", gate)
    (out / "report.md").write_text(
        "# Incremental lifetime-history consumed diagnostic\n\n"
        "One consumed input per task. Cross-budget native controls are not byte-matched. "
        "No confidence interval or paper superiority claim is established.\n\n"
        + table.to_markdown(index=False)
        + "\n\n```json\n"
        + json.dumps(gate, indent=2)
        + "\n```\n"
    )
    write_once(
        out / "manifest.json",
        dict(
            source_sha=plan["source_sha"],
            plan_sha256=expected,
            input_sha256={str(p): sha(p) for p in (root / "workers").rglob("*.json")},
            output_sha256={p.name: sha(p) for p in out.iterdir()},
        ),
    )


def submit(root, expected):
    inputs(root, expected)
    (root / "slurm").mkdir()
    script = str(SOURCE / "mosaickv/slurm/evicastkv_lifetime_history.sbatch")
    args = [str(SOURCE), str(root), expected]
    gpu = (
        subprocess.check_output(
            [
                "sbatch",
                "--parsable",
                "--account=bii_dsc_community",
                "--partition=bii-gpu",
                "--reservation=bi_fox_dgx",
                "--job-name=mk-evicast-lifetime",
                "--nodes=1",
                "--ntasks=1",
                "--gres=gpu:a100:1",
                "--cpus-per-task=12",
                "--mem=64G",
                "--time=01:00:00",
                "--array=0-1%2",
                f"--output={root}/slurm/gpu-%A_%a.out",
                f"--error={root}/slurm/gpu-%A_%a.err",
                script,
                *args,
                "worker",
            ],
            text=True,
        )
        .strip()
        .split(";")[0]
    )
    write_once(root / "gpu_submission.json", dict(array=gpu))
    cpu = (
        subprocess.check_output(
            [
                "sbatch",
                "--parsable",
                "--account=bii_dsc_community",
                "--partition=bii",
                "--job-name=mk-evicast-lifetime-analysis",
                "--nodes=1",
                "--ntasks=1",
                "--cpus-per-task=2",
                "--mem=8G",
                "--time=00:10:00",
                f"--dependency=afterany:{gpu}",
                f"--output={root}/slurm/analysis-%j.out",
                f"--error={root}/slurm/analysis-%j.err",
                script,
                *args,
                "analyze",
            ],
            text=True,
        )
        .strip()
        .split(";")[0]
    )
    write_once(root / "submission.json", dict(gpu_array=gpu, cpu_analysis=cpu))
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare", "submit", "worker", "analyze"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--plan-sha")
    p.add_argument("--index", type=int, choices=(0, 1))
    a = p.parse_args()
    if a.mode == "prepare":
        print(prepare(a.root))
    elif a.mode == "submit":
        submit(a.root, a.plan_sha)
    elif a.mode == "worker":
        worker(a.root, a.plan_sha, a.index)
    else:
        analyze(a.root, a.plan_sha)


if __name__ == "__main__":
    main()
