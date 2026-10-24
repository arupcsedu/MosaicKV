#!/usr/bin/env python3
"""Recover saved ROUGE on CPU, retry invalid PPL only, audit frozen V4 logits.

This cannot materialize samples, change V4, or authorize a larger campaign.
All outputs are new; original predictions and failed artifacts are immutable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

from bounded_history_pilot import SOURCE, clean_source, digest, inputs, progress, write_once

from mosaickv.config import load_config
from mosaickv.evaluation.cache_phase_audit import audit_phase_trace

PARENT = Path("/scratch/djy8hg/runs/mosaickv/bounded-history-memory-v4-20261024-v1").resolve()
PARENT_SHA = "caa4616930ff505958577224a5704f1f5aed68401609dc6068aac592a0acbd62"


def object_sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def phase_inventory(paths, sample_ids, cap):
    """Reuse only one complete, storage-valid observation for each phase/ID."""
    valid, invalid = {}, {}
    for path in sorted(map(Path, paths)):
        trace = json.loads(path.read_text())
        sample_id = trace["sample_id"]
        if sample_id not in sample_ids:
            raise ValueError("unregistered sample in phase traces")
        phase = (
            "teacher_forced"
            if trace.get("teacher_forced_evaluation", {}).get("enabled")
            else "generation"
        )
        key = sample_id, phase
        try:
            record = audit_phase_trace(trace, cap)
        except (ValueError, KeyError) as error:
            invalid.setdefault(key, []).append(
                dict(path=str(path), sha256=digest(path), error=str(error))
            )
            continue
        if key in valid:
            raise ValueError("multiple admissible observations for one sample/phase")
        valid[key] = dict(**record, path=str(path), sha256=digest(path))
    retries = []
    for sample_id in sample_ids:
        if (sample_id, "generation") not in valid:
            raise ValueError("generation is not reusable; this repair cannot rerun predictions")
        if (sample_id, "teacher_forced") not in valid:
            retries.append(
                dict(
                    sample_id=sample_id,
                    rejected=invalid.get((sample_id, "teacher_forced"), []),
                    reason="missing_or_storage_invalid_teacher_forced_phase",
                )
            )
    return valid, retries


def prepare(root):
    import pandas as pd

    from mosaickv.evaluation.metrics.registry import default_metric_registry
    from mosaickv.evaluation.prefixkv_metrics import load_fullkv_reference_artifact, rouge_samples
    from mosaickv.evaluation.storage import load_jsonl

    sha = clean_source()
    root = root.resolve()
    if root.exists() or Path("/scratch/djy8hg").resolve() not in root.parents:
        raise ValueError("require a new immutable scratch recovery directory")
    parent = inputs(PARENT, PARENT_SHA, check_clean=False)
    if parent["protocol"]["experiment_id"] != "bounded_history_memory_v4":
        raise ValueError("only frozen negative V4 may be recovered")
    bound = dict(parent["bound_files"])
    bound[str(PARENT / "plan.json")] = PARENT_SHA
    # Seal all existing execution evidence, including raw failures and manifests.
    for folder in ("outputs", "workers", "references", "attempts", "analysis"):
        for path in (PARENT / folder).rglob("*"):
            if path.is_file():
                bound[str(path)] = digest(path)
    recovered, reused, jobs, rejected = [], [], [], []
    definition = default_metric_registry().resolve("rouge_l_f1")
    for job in parent["jobs"]:
        cfg = load_config(job["config"])
        output = Path(job["output"])
        paths = list(output.rglob("results.jsonl"))
        if len(paths) != 1:
            raise ValueError("require exactly one immutable raw prediction file")
        rows = load_jsonl(paths[0])
        if {row.sample_id for row in rows} != set(job["sample_ids"]) or len(rows) != 8:
            raise ValueError("incomplete or duplicated saved predictions")
        if any(
            row.error
            or row.status.value != "completed"
            or row.generated_tokens != 1024
            or row.model != cfg.model.id
            or row.backend != "huggingface"
            for row in rows
        ):
            raise ValueError("saved generation failed controlled-comparison checks")
        reference = load_fullkv_reference_artifact(job["reference"])
        if reference.model_id != cfg.model.id or reference.model_revision != cfg.model.revision:
            raise ValueError("FullKV reference checkpoint differs")
        for metric in rouge_samples(rows, reference):
            row = next(row for row in rows if row.sample_id == metric.sample_id)
            recovered.append(
                dict(
                    parent_index=job["index"],
                    task=job["task"],
                    method=job["method"],
                    cap_mib=job["cap_mib"],
                    run_id=row.run_id,
                    sample_id=row.sample_id,
                    metric_name=definition.name,
                    metric_value=metric.value,
                    metric_direction=definition.direction.value,
                    metric_scope="sample",
                    metric_implementation=definition.implementation,
                    metric_version=definition.version,
                    evaluator_commit=sha,
                    reference_artifact_sha=digest(job["reference"]),
                    producer_git_sha=parent["source_sha"],
                    config_sha=job["config_sha"],
                    manifest_path=row.manifest_path,
                    parent_manifest_exists=Path(row.manifest_path).is_file(),
                    raw_output_path=str(paths[0]),
                    raw_output_sha256=digest(paths[0]),
                    measurement_type="derived_retrospective_development_not_paper_evidence",
                    **{
                        f"{b}_active_kv_storage_bytes": getattr(row, f"{b}_active_kv_storage_bytes")
                        for b in ("initial", "final", "peak")
                    },
                )
            )
        cap = None if job["method"] == "full_kv" else cfg.cache.budget_value
        valid, retries = phase_inventory(
            (output / "traces").rglob("*.json"), job["sample_ids"], cap
        )
        for row in rows:
            generation = valid[row.sample_id, "generation"]
            if row.method != generation["effective_method"] or any(
                getattr(row, f"{b}_active_kv_storage_bytes") != generation["storage_bytes"][b]
                for b in ("initial", "final", "peak")
            ):
                raise ValueError("saved prediction and measured generation trace disagree")
        for (_sample_id, phase), record in valid.items():
            if phase == "teacher_forced":
                reused.append(
                    dict(
                        parent_index=job["index"],
                        producer_git_sha=parent["source_sha"],
                        config_sha=job["config_sha"],
                        **record,
                    )
                )
        rejected.extend(dict(parent_index=job["index"], **item) for item in retries)
        if retries:
            if job["method"] == "full_kv":
                raise ValueError("unexpected FullKV metric failure; cannot expand repair scope")
            jobs.append(
                dict(
                    index=len(jobs),
                    kind="ppl",
                    parent_index=job["index"],
                    sample_ids=[item["sample_id"] for item in retries],
                )
            )
    # Two fixed consumed identities: first ledger entry of each task at 40 MiB.
    # Chosen by protocol order, NOT by answers, errors, or observed quality.
    for job in parent["jobs"]:
        if job["method"] == "mosaickv_direct_error_v4" and job["cap_mib"] == 40:
            jobs.append(
                dict(
                    index=len(jobs),
                    kind="diagnostic",
                    parent_index=job["index"],
                    sample_ids=job["sample_ids"][:1],
                    horizon=4,
                )
            )
    if len(jobs) > 11 or sum(len(j["sample_ids"]) for j in jobs if j["kind"] == "ppl") > 30:
        raise ValueError("unexpected repair scope; refuse broader GPU work")
    write_once(
        root / "negative-v4-seal.json",
        dict(
            classification="negative_development_result",
            all_planned_identities_consumed=True,
            cohorts={
                task: next(j["sample_ids"] for j in parent["jobs"] if j["task"] == task)
                for task in parent["protocol"]["tasks"]
            },
            parent_plan_sha256=PARENT_SHA,
            existing_report_sha256=digest(PARENT / "analysis/report.json"),
            algorithm_changed=False,
            exchange_limits_increased=False,
            paper_promotion_authorized=False,
            reason="prompt reconstruction improvement did not establish answer preservation",
        ),
    )
    pd.DataFrame(recovered).to_parquet(root / "recovered_rouge.parquet", index=False)
    write_once(root / "reused-ppl.json", reused)
    write_once(root / "rejected-ppl.json", rejected)
    for name in (
        "negative-v4-seal.json",
        "recovered_rouge.parquet",
        "reused-ppl.json",
        "rejected-ppl.json",
    ):
        bound[str(root / name)] = digest(root / name)
    plan = dict(
        source_sha=sha,
        source_root=str(SOURCE),
        parent_root=str(PARENT),
        parent_plan_sha256=PARENT_SHA,
        bound_files=bound,
        jobs=jobs,
        prediction_reruns=0,
        larger_campaign_authorized=False,
        algorithm_selection_feedback=False,
    )
    write_once(root / "plan.json", plan)
    progress(
        "recovery_prepared",
        rouge_rows=len(recovered),
        reused_ppl=len(reused),
        retry_ppl=len(rejected),
        workers=len(jobs),
        plan_sha256=digest(root / "plan.json"),
    )


def validate(root, plan_sha, index=None):
    if digest(root / "plan.json") != plan_sha:
        raise ValueError("recovery plan hash changed")
    plan = json.loads((root / "plan.json").read_text())
    clean_source(plan["source_sha"])
    if (
        Path(plan["source_root"]).resolve() != SOURCE.resolve()
        or plan["parent_plan_sha256"] != PARENT_SHA
    ):
        raise ValueError("wrong source or parent")
    for path, sha in plan["bound_files"].items():
        if digest(path) != sha:
            raise ValueError(f"bound recovery input changed: {path}")
    if index is None:
        return plan
    job = plan["jobs"][index]
    parent, original, cfg = inputs(PARENT, PARENT_SHA, job["parent_index"], check_clean=False)
    return plan, job, parent, original, cfg


def worker(root, plan_sha, index):
    import torch

    from mosaickv.adapters.huggingface import (
        compare_cache_reinjection,
        compare_with_generate,
        load_hf_adapter,
    )
    from mosaickv.backends.hf_runtime import HuggingFaceMosaicKVModel
    from mosaickv.evaluation.direct_error_diagnostic import DirectErrorLogitDiagnostic
    from mosaickv.evaluation.messages import MediaItem, build_multimodal_messages
    from mosaickv.evaluation.metrics.registry import default_metric_registry
    from mosaickv.evaluation.model import EvaluationRequest
    from mosaickv.evaluation.paper_native_lanes import _media_payload, load_paper_native_ledger
    from mosaickv.evaluation.prefixkv_metrics import target_artifact_sha256
    from mosaickv.evaluation.tasks import default_task_registry
    from mosaickv.evaluation.worker_support import settle
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.manifest import (
        ArtifactProvenance,
        EvaluationProvenance,
        InputProvenance,
        RunManifestWriter,
    )
    from mosaickv.types import MeasurementType

    plan, job, _parent, original, cfg = validate(root, plan_sha, index)
    destination = root / "workers" / f"{index:02d}"
    if destination.exists():
        raise FileExistsError(
            "recovery worker outputs are write-once; inspect failures before retry"
        )
    destination.mkdir(parents=True)
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("deterministic CUBLAS setting must precede Python")
    if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("exactly one allocated A100 required")
    torch.set_num_threads(12)
    torch.manual_seed(cfg.execution.seed)
    torch.use_deterministic_algorithms(True)
    ledger = load_paper_native_ledger(
        original["ledger"],
        expected_sha256=original["ledger_sha"],
        task=default_task_registry().resolve(original["task"]),
        dataset_revision=cfg.dataset.revision,
    )
    samples = [s for s in ledger.samples if s.sample_id in job["sample_ids"]]
    if len(samples) != len(job["sample_ids"]):
        raise ValueError("recovery subset does not match immutable ledger")
    adapter = load_hf_adapter(
        cfg.model.id,
        revision=cfg.model.revision,
        model_kwargs=_model_kwargs(cfg, local_files_only=True),
        processor_kwargs={"local_files_only": True},
    )
    progress("recovery_model_loaded", index=index, kind=job["kind"], samples=len(samples))

    def messages(sample):
        return build_multimodal_messages(
            sample.prompt, tuple(MediaItem(m.kind, _media_payload(m)) for m in sample.media)
        )

    prepared = adapter.prepare_inputs(messages(samples[0]))
    parity = {}
    for name, check in (
        ("generate", compare_with_generate),
        ("reinjection", compare_cache_reinjection),
    ):
        result = check(adapter, prepared, max_new_tokens=16)
        parity[name] = asdict(result)
        if result.token_agreement != 1.0 or result.maximum_logit_difference > 0.001:
            raise RuntimeError("repair-source FullKV parity failed")
    write_once(destination / "parity.json", parity)
    runtime_type = (
        DirectErrorLogitDiagnostic if job["kind"] == "diagnostic" else HuggingFaceMosaicKVModel
    )
    runtime = runtime_type(
        adapter,
        cfg,
        trace_directory=destination / "traces",
        **({"horizon": job["horizon"]} if job["kind"] == "diagnostic" else {}),
    )
    records, tokenization = [], []
    # Targets are accessed ONLY in the teacher-forced PPL branch, never diagnostic.
    for sample in samples:
        request = EvaluationRequest(
            f"recovery-{index:02d}", sample.sample_id, original["task"], messages(sample), {}
        )
        tokenization.append(
            adapter.prepare_inputs(request.messages).model_inputs["input_ids"].cpu().tolist()
        )
        settle(adapter)
        try:
            if job["kind"] == "ppl":
                loss = runtime.teacher_forced_nll(request, sample.teacher_forcing_target)
                paths = list((destination / "traces").rglob("*.json"))
                matching = [
                    p for p in paths if json.loads(p.read_text())["sample_id"] == sample.sample_id
                ]
                if len(matching) != 1:
                    raise ValueError("PPL recovery requires one complete trace per sample")
                result = audit_phase_trace(
                    json.loads(matching[0].read_text()), cfg.cache.budget_value
                )
                if (
                    result["nll_sum"] != loss.negative_log_likelihood_sum
                    or result["tokens"] != loss.evaluated_tokens
                ):
                    raise ValueError("PPL result disagrees with audited trace")
                record = dict(**result, path=str(matching[0]), sha256=digest(matching[0]))
            else:
                record = runtime.diagnose(request)
            row = dict(
                parent_index=original["index"],
                sample_id=sample.sample_id,
                status="completed",
                producer_git_sha=plan["source_sha"],
                config_sha=original["config_sha"],
                result=record,
            )
        except Exception as error:
            row = dict(
                parent_index=original["index"],
                sample_id=sample.sample_id,
                status="failed",
                error=f"{type(error).__name__}: {error}",
            )
        records.append(row)
        write_once(destination / "samples" / f"{len(records):02d}.json", row)
        progress(
            "recovery_sample_recorded",
            index=index,
            sample_id=sample.sample_id,
            status=row["status"],
        )
    write_once(destination / "raw.json", records)
    artifacts = {str(p): digest(p) for p in destination.rglob("*") if p.is_file()}
    write_once(destination / "artifact-hashes.json", artifacts)
    definition = default_metric_registry().resolve("teacher_forced_ppl")
    reference_sha = (
        target_artifact_sha256({s.sample_id: s.teacher_forcing_target for s in samples})
        if job["kind"] == "ppl"
        else digest(original["reference"])
    )
    preprocessing = dict(
        processor=cfg.model.id,
        revision=cfg.model.revision,
        class_name=type(adapter.processor).__name__,
        config=getattr(adapter.processor, "to_dict", lambda: {})(),
    )
    manifest = RunManifestWriter(SOURCE).write(
        destination / "manifest.json",
        cfg,
        MeasurementType.VALIDATION_SMOKE,
        InputProvenance(
            object_sha([s.prompt for s in samples]),
            object_sha(
                [
                    [dict(kind=m.kind.value, path=str(m.path), sha256=m.sha256) for m in s.media]
                    for s in samples
                ]
            ),
            object_sha(preprocessing),
            object_sha(tokenization),
        ),
        ArtifactProvenance(
            digest(destination / "raw.json"),
            "not_applicable",
            digest(destination / "artifact-hashes.json"),
        ),
        run_id=f"v4-recovery-{index:02d}",
        execution_metadata=dict(
            parent_plan_sha256=PARENT_SHA,
            recovery_plan_sha256=plan_sha,
            phase=job["kind"],
            slurm_job_id=os.environ.get("SLURM_JOB_ID"),
            generation_predictions_rerun=False,
            new_timing_comparison=False,
            diagnostic_selection_feedback=False,
            sample_ids=job["sample_ids"],
        ),
        evaluation_provenance=EvaluationProvenance(
            evaluator_name=definition.implementation
            if job["kind"] == "ppl"
            else "isolated_FullKV_draft_logit_audit",
            evaluator_version=definition.version if job["kind"] == "ppl" else "1",
            evaluator_commit=plan["source_sha"],
            reference_output_sha256=reference_sha,
            baseline_name=original["method"],
            baseline_commit=plan["source_sha"],
        ),
    )
    write_once(
        destination / "completed.json",
        dict(
            status="passed" if all(r["status"] == "completed" for r in records) else "failed",
            source_sha=plan["source_sha"],
            plan_sha256=plan_sha,
            index=index,
            raw_sha256=digest(destination / "raw.json"),
            manifest_path=str(manifest),
            manifest_sha256=digest(manifest),
        ),
    )
    if any(row["status"] != "completed" for row in records):
        raise RuntimeError("recovery contains explicit failed samples; see immutable raw records")


def analyze(root, plan_sha):
    import numpy as np
    import pandas as pd

    plan = validate(root, plan_sha)
    parent = inputs(PARENT, PARENT_SHA, check_clean=False)
    ppl = {
        (r["parent_index"], r["sample_id"]): r
        for r in json.loads((root / "reused-ppl.json").read_text())
    }
    failures, diagnostics = [], []
    for job in plan["jobs"]:
        folder = root / "workers" / f"{job['index']:02d}"
        raw = folder / "raw.json"
        complete = folder / "completed.json"
        if not raw.exists() or not complete.exists():
            failures.append(
                dict(
                    index=job["index"],
                    error="incomplete_recovery_worker",
                    sample_ids=job["sample_ids"],
                )
            )
            continue
        completion = json.loads(complete.read_text())
        if completion["plan_sha256"] != plan_sha or completion["source_sha"] != plan["source_sha"]:
            raise ValueError("recovery worker source/plan mismatch")
        if (
            digest(raw) != completion["raw_sha256"]
            or digest(completion["manifest_path"]) != completion["manifest_sha256"]
        ):
            raise ValueError("recovery output changed")
        for path, sha in json.loads((folder / "artifact-hashes.json").read_text()).items():
            if digest(path) != sha:
                raise ValueError("recovery trace/parity artifact changed")
        records = json.loads(raw.read_text())
        if len(records) != len(job["sample_ids"]) or {r["sample_id"] for r in records} != set(
            job["sample_ids"]
        ):
            raise ValueError("recovery lost or duplicated a planned sample")
        for row in records:
            if row["status"] != "completed":
                failures.append(row)
            elif job["kind"] == "diagnostic":
                diagnostics.append(row)
            else:
                key = row["parent_index"], row["sample_id"]
                if key in ppl:
                    raise ValueError("attempted to replace already valid PPL")
                ppl[key] = dict(
                    **row["result"],
                    parent_index=row["parent_index"],
                    producer_git_sha=row["producer_git_sha"],
                    manifest_path=completion["manifest_path"],
                )
    rows = pd.read_parquet(root / "recovered_rouge.parquet").to_dict("records")
    for row in rows:
        record = ppl.get((row["parent_index"], row["sample_id"]))
        row.update(
            nll_sum=record["nll_sum"] if record else None,
            tokens=record["tokens"] if record else None,
            ppl_status="completed" if record else "missing_or_invalid",
            ppl_producer_git_sha=record.get("producer_git_sha") if record else None,
            ppl_trace=record["path"] if record else None,
        )
        for boundary in ("initial", "final", "peak"):
            row[f"ppl_{boundary}_active_kv_storage_bytes"] = (
                record["storage_bytes"][boundary] if record else None
            )
    summaries, pairs = [], []
    for job in parent["jobs"]:
        subset = [r for r in rows if r["parent_index"] == job["index"]]
        valid = [r for r in subset if r["ppl_status"] == "completed"]
        summaries.append(
            dict(
                parent_index=job["index"],
                task=job["task"],
                method=job["method"],
                cap_mib=job["cap_mib"],
                planned=8,
                rouge_completed=len(subset),
                ppl_completed=len(valid),
                rouge_l_f1=sum(r["metric_value"] for r in subset) / len(subset),
                corpus_ppl=math.exp(
                    sum(r["nll_sum"] for r in valid) / sum(r["tokens"] for r in valid)
                )
                if len(valid) == 8
                else None,
            )
        )
        if job["method"] != "mosaickv_direct_error_v4" or len(valid) != 8:
            continue
        baseline = next(
            j
            for j in parent["jobs"]
            if j["task"] == job["task"]
            and j["cap_mib"] == job["cap_mib"]
            and j["method"] == "prompt_attention_topk"
        )
        b = {r["sample_id"]: r for r in rows if r["parent_index"] == baseline["index"]}
        ids = job["sample_ids"]
        if any(b[s]["ppl_status"] != "completed" for s in ids):
            continue
        a = {r["sample_id"]: r for r in valid}
        nll_a, nll_b, count_a, count_b = [
            np.array([source[s][key] for s in ids])
            for source, key in ((a, "nll_sum"), (b, "nll_sum"), (a, "tokens"), (b, "tokens"))
        ]
        if not np.array_equal(count_a, count_b):
            raise ValueError("paired teacher-forcing target lengths differ")
        draw = np.random.default_rng(20271014).integers(0, len(ids), size=(10000, len(ids)))
        ppl_diff = np.exp(nll_a[draw].sum(1) / count_a[draw].sum(1)) - np.exp(
            nll_b[draw].sum(1) / count_b[draw].sum(1)
        )
        rouge_diff = np.array([a[s]["metric_value"] - b[s]["metric_value"] for s in ids])
        pairs.append(
            dict(
                task=job["task"],
                cap_mib=job["cap_mib"],
                candidate="mosaickv_direct_error_v4",
                baseline="prompt_attention_topk",
                ppl_delta_ci95=np.quantile(ppl_diff, [0.025, 0.975]).tolist(),
                rouge_delta=float(rouge_diff.mean()),
                rouge_delta_ci95=np.quantile(rouge_diff[draw].mean(1), [0.025, 0.975]).tolist(),
                bootstrap_replicates=10000,
                multiplicity_adjusted=False,
                retrospective_development_only=True,
            )
        )
    destination = root / "analysis"
    destination.mkdir(exist_ok=False)
    pd.DataFrame(rows).to_parquet(destination / "samples.parquet", index=False)
    pd.DataFrame(summaries).to_parquet(destination / "summary.parquet", index=False)
    write_once(
        destination / "report.json",
        dict(
            source_sha=plan["source_sha"],
            plan_sha256=plan_sha,
            summaries=summaries,
            pairs=pairs,
            failures=failures,
            diagnostics=diagnostics,
            saved_predictions_reused=len(rows),
            complete_ppl=sum(r["ppl_status"] == "completed" for r in rows),
            expected_ppl=160,
            v4_retained_as_negative=True,
            larger_campaign_authorized=False,
            caveats=[
                "repaired and reused PPL producer SHAs are explicitly separated",
                "missing original manifests are disclosed, not retrospectively fabricated",
                "diagnostic is evaluation-only and is not a quality/latency row",
                "all outcomes are development-only, not paper or held-out evidence",
            ],
        ),
    )
    lines = [
        "# V4 metric recovery (negative development result)",
        "",
        f"Saved predictions reused: {len(rows)} / 160.",
        f"Complete PPL observations: {sum(r['ppl_status'] == 'completed' for r in rows)} / 160.",
        f"Recorded failures: {len(failures)}; isolated diagnostics: {len(diagnostics)}.",
        "",
        "No generation predictions were rerun. Initial/final/peak storage is audited",
        "separately for generation and PPL. FullKV remains an uncapped quality reference.",
        "V4 remains negative; these retrospective diagnostics do not authorize promotion.",
        "",
        "| Task | Method | Cap MiB | ROUGE-L | Corpus PPL | PPL n/8 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for item in summaries:
        ppl_value = f"{item['corpus_ppl']:.6f}" if item["corpus_ppl"] is not None else "incomplete"
        lines.append(
            f"| {item['task']} | {item['method']} | {item['cap_mib']} | "
            f"{item['rouge_l_f1']:.6f} | {ppl_value} | {item['ppl_completed']}/8 |"
        )
    with (destination / "report.md").open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    progress(
        "recovery_analysis_completed",
        failures=len(failures),
        rows=len(rows),
        diagnostics=len(diagnostics),
    )


def submit(root, plan_sha):
    plan = validate(root, plan_sha)
    logs = root / "logs"
    logs.mkdir()

    def sbatch(script, extra, args):
        return (
            subprocess.check_output(
                [
                    "sbatch",
                    "--parsable",
                    "--account=bii_dsc_community",
                    f"--output={logs}/%x-%A_%a.out",
                    f"--error={logs}/%x-%A_%a.err",
                    *extra,
                    str(SOURCE / script),
                    *args,
                ],
                text=True,
            )
            .strip()
            .split(";", 1)[0]
        )

    args = [str(SOURCE), str(root), plan_sha]
    gpu = sbatch(
        "mosaickv/slurm/bounded_history_recovery.sbatch",
        ["--hold", f"--array=0-{len(plan['jobs']) - 1}%6"],
        args,
    )
    write_once(root / "gpu-submission.json", dict(array=gpu, source_sha=plan["source_sha"]))
    analysis = sbatch(
        "mosaickv/slurm/bounded_history_recovery_analysis.sbatch",
        [f"--dependency=afterany:{gpu}"],
        args,
    )
    snapshot = root / "scheduler"
    snapshot.mkdir()
    for filename in ("control_a100_schedule.py", "cache_env.sh"):
        shutil.copy2(SOURCE / "mosaickv/scripts" / filename, snapshot / filename)
    write_once(
        snapshot / "a100_schedule.json",
        dict(
            timezone="America/New_York",
            maximum_worker_minutes=120,
            maximum_a100s=6,
            six_a100s_explicitly_authorized=True,
            window="all_times",
            arrays=[gpu],
            adopt_existing_holds=[gpu],
            source_sha=plan["source_sha"],
            authorization="scoped_V4_invalid_PPL_recovery_and_two_consumed_identity_diagnostics_only",
        ),
    )
    write_once(
        snapshot / "snapshot.json",
        dict(
            source_sha=plan["source_sha"],
            files={p.name: digest(p) for p in snapshot.iterdir() if p.is_file()},
        ),
    )
    controller = sbatch(
        "mosaickv/slurm/a100_schedule_controller.sbatch",
        ["--time=04:00:00", "--export=ALL,MOSAICKV_POLICY_ROOT=" + str(snapshot)],
        [],
    )
    result = dict(
        source_sha=plan["source_sha"],
        plan_sha256=plan_sha,
        array=gpu,
        analysis=analysis,
        controller=controller,
        maximum_a100s=6,
        cpus_per_worker=12,
        worker_time_limit="02:00:00",
        gpu_workers=len(plan["jobs"]),
        generation_reruns=0,
        paper_campaign_submitted=False,
    )
    write_once(root / "submission.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "worker", "analyze", "submit"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha")
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root)
    elif args.action == "worker":
        worker(args.root, args.plan_sha, args.index)
    else:
        globals()[args.action](args.root, args.plan_sha)
