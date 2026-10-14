#!/usr/bin/env python3
"""Small frozen, paired 40-MiB PrefixKV allocation diagnostic; no full sweep."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from run_byte_capped_published_baseline_pilot import _require_clean_source
from run_prefixkv_decode_preflight import (
    CAP,
    MODEL,
    REVISION,
    allocation_configs,
    check_plan,
    configs,
    digest,
    write_once,
)

CLASSIFICATION = "retrospective_disjoint_allocation_development_only"
ARMS = (
    "full_kv",
    "prompt_attention_topk",
    "prefixkv_upstream_shape_reimpl",
    "prefixkv_byte_calibrated_reimpl",
    "prefixkv_capacity_floor_reimpl",
    "prefixkv_uniform_layer_reimpl",
    "mosaickv_exact",
)


def split_ids(ids):
    if len(ids) < 50 or len(set(ids)) != len(ids):
        raise ValueError("need at least 50 distinct, ordered parent identities")
    return tuple(ids[:32]), tuple(ids[32:42]), tuple(ids[42:50])


def validate_ledgers(calibration, evaluation, consumed):
    for key in ("task", "dataset_id", "dataset_revision", "split"):
        if not calibration.get(key) or calibration[key] != evaluation.get(key):
            raise ValueError(f"calibration/evaluation provenance mismatch: {key}")
    cal, ev = calibration["sample_ids"], evaluation["sample_ids"]
    if len(cal) != 10 or len(ev) != 8 or len(set(cal + ev)) != 18:
        raise ValueError("require ten calibration and eight disjoint evaluation IDs")
    if set(cal + ev).intersection(consumed):
        raise ValueError("new diagnostic identities overlap prior consumed IDs")


def prepare(root, sha, parent_root):
    source = Path(__file__).resolve().parents[2]
    _require_clean_source(source, sha)
    jobs = []
    bound = {}
    for task in ("textvqa", "docvqa"):
        paths = [parent_root / task / f"{i:02d}.json" for i in range(4)]
        parents = [json.loads(p.read_text()) for p in paths]
        for key in ("task", "dataset_id", "dataset_revision", "split"):
            if any(p[key] != parents[0][key] for p in parents):
                raise ValueError(f"parent provenance differs: {key}")
        consumed, cal, ev = split_ids([sid for p in parents for sid in p["sample_ids"]])
        job = {"task": task}
        for kind, ids in (("calibration", cal), ("evaluation", ev)):
            payload = {
                k: parents[0][k] for k in ("task", "dataset_id", "dataset_revision", "split")
            }
            payload.update(
                schema_version=1,
                sample_ids=list(ids),
                sample_count=len(ids),
                source_git_sha=sha,
                source_git_dirty=False,
                classification=CLASSIFICATION,
                selection_rule="ID-only parent order: prior[0:32], calibration[32:42], eval[42:50]",
                parent_ledgers={str(p): digest(p) for p in paths},
            )
            job[kind] = f"{task}_{kind}.json"
            write_once(root / job[kind], payload)
        job["consumed"] = f"{task}_consumed.json"
        write_once(root / job["consumed"], {"sample_ids": list(consumed)})
        validate_ledgers(
            json.loads((root / job["calibration"]).read_text()),
            json.loads((root / job["evaluation"]).read_text()),
            consumed,
        )
        jobs.append(job)
        bound.update({str(p): digest(p) for p in paths})
        bound.update(
            {job[k]: digest(root / job[k]) for k in ("calibration", "evaluation", "consumed")}
        )
    bound.update({name: digest(root / name) for name in ("worker.sbatch", "analyze.sbatch")})
    plan = dict(
        schema_version=1,
        source_sha=sha,
        source_root=str(source),
        classification=CLASSIFICATION,
        full_campaign_authorized=False,
        max_a100s_global=4,
        gpus_per_worker=1,
        jobs=jobs,
        bound_files=bound,
        arms=list(ARMS),
        byte_cap=CAP,
        output_tokens=64,
        seed=20271001,
        block_size=1,
        mandatory_policy="first_and_last",
        warmups=0,
        repetitions=1,
        timing_claim="informational_only",
        quality_gate="per_task_score_not_below_Prompt_TopK; report paired bootstrap CI",
        minimum_quality_delta=0.0,
        floor_fraction=0.5,
        calibration_samples_per_task=10,
        evaluation_samples_per_task=8,
        absolute_logit_tolerance=1e-4,
        differential_steps=64,
        fullkv_use="quality_reference_only_no_latency_comparison",
        heldout_claim=False,
    )
    write_once(root / "plan.json", plan)
    print(digest(root / "plan.json"))


def collect(adapter, ledger, output):
    import torch

    from mosaickv.baselines.prefixkv import prefixkv_calibration_observation
    from mosaickv.evaluation.lmms_adapter import (
        _flatten_visuals,
        _media_kind,
        prepare_seeded_lmms_tasks,
    )
    from mosaickv.evaluation.messages import MediaItem, build_multimodal_messages
    from mosaickv.evaluation.tasks import default_task_registry

    spec = default_task_registry().resolve(ledger["task"])
    tasks, _, selected, _ = prepare_seeded_lmms_tasks(
        (spec,),
        seed=20271001,
        subset_size=None,
        explicit_sample_ids=tuple(ledger["sample_ids"]),
        model_name=MODEL,
        dataset_revisions={ledger["dataset_id"]: ledger["dataset_revision"]},
        prepare_samples=False,
    )
    if set(selected) != set(ledger["sample_ids"]) or len(tasks) != 1:
        raise RuntimeError("calibration task routing changed")
    task = tasks[0]
    task.config.num_fewshot = 0
    task.build_all_requests(cache_requests=False)
    observations, files = [], {}
    for instance in task.instances:
        context, _, to_visual, doc_id, _, split = instance.args[:6]
        doc = task.dataset[split][doc_id]
        sid = doc["_mosaickv_sample_id"]
        if sid in files or sid not in ledger["sample_ids"]:
            raise ValueError("duplicate or unexpected calibration request")
        visuals = _flatten_visuals(to_visual(doc))
        media = tuple(MediaItem(_media_kind(v), v) for v in visuals)
        prepared = adapter.prepare_inputs(build_multimodal_messages(context, media))
        with torch.inference_mode():
            prefill = adapter.prefill(prepared, capture_queries=False, capture_attentions=True)
            observation = prefixkv_calibration_observation(sid, prefill.attention_weights)
            snapshot = adapter.extract_past_key_values(prefill.state.past_key_values)
        widths = [
            (
                layer.key.numel() * layer.key.element_size()
                + layer.value.numel() * layer.value.element_size()
            )
            // layer.key.shape[-2]
            for layer in snapshot.layers
        ]
        if len(widths) != 32 or set(widths) != {16384}:
            raise RuntimeError("unexpected cache geometry for frozen 40 MiB calibration")
        path = output / "calibration" / f"{len(observations):02d}.json"
        # NO labels, references, decoded answers, or task scores enter the profile.
        write_once(
            path,
            dict(
                sample_id=sid,
                layer_scores=observation.layer_scores,
                bytes_per_layer_position=16384,
                prompt_tokens=prepared.logical_sequence_length,
            ),
        )
        files[sid] = {"path": str(path), "sha256": digest(path)}
        observations.append(observation)
        torch.cuda.synchronize()
        print(json.dumps({"phase": "calibration", "completed": len(observations)}), flush=True)
    if set(files) != set(ledger["sample_ids"]):
        raise RuntimeError("incomplete calibration captures")
    return tuple(observations), files, prepared, prefill


def run(args):
    import torch

    from mosaickv.adapters.huggingface import (
        compare_cache_reinjection,
        compare_with_generate,
        load_hf_adapter,
    )
    from mosaickv.baselines.prefixkv import (
        build_prefixkv_reimpl_plan,
        generate_prefixkv_byte_profile,
    )
    from mosaickv.cache_state import FullKVState
    from mosaickv.evaluation.prefixkv_differential import compare_upstream_decode
    from mosaickv.evaluation.worker_support import record_hf_run, settle
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    plan, source = check_plan(args.root, args.plan_sha)
    if plan["arms"] != list(ARMS) or plan["byte_cap"] != CAP or plan["floor_fraction"] != 0.5:
        raise RuntimeError("allocation protocol changed")
    job = plan["jobs"][args.index]
    task = job["task"]
    cal = json.loads((args.root / job["calibration"]).read_text())
    ev = json.loads((args.root / job["evaluation"]).read_text())
    consumed = json.loads((args.root / job["consumed"]).read_text())["sample_ids"]
    validate_ledgers(cal, ev, consumed)
    output = args.root / "outputs" / task
    output.mkdir(parents=True, exist_ok=False)
    summary = dict(
        status="running",
        records=[],
        failures=[],
        source_sha=plan["source_sha"],
        source_dirty=False,
        plan_sha256=args.plan_sha,
        slurm_job_id=os.getenv("SLURM_JOB_ID"),
        classification=CLASSIFICATION,
        full_campaign_authorized=False,
        task=task,
        calibration_ledger_sha256=digest(args.root / job["calibration"]),
        evaluation_ledger_sha256=digest(args.root / job["evaluation"]),
        planned_denominator=len(ev["sample_ids"]),
    )
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("requires one allocated GPU")
        torch.manual_seed(plan["seed"])
        torch.use_deterministic_algorithms(True)
        base = configs(source, task)
        adapter = load_hf_adapter(
            MODEL,
            revision=REVISION,
            model_kwargs=_model_kwargs(base["full_kv"], local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        summary["gpu_environment"] = asdict(capture_gpu_environment(torch))
        start = time.perf_counter()
        observations, captures, prepared, prefill = collect(adapter, cal, output)
        profile = generate_prefixkv_byte_profile(
            observations,
            byte_cap=CAP,
            bytes_per_layer_position=16384,
            model_id=MODEL,
            model_revision=REVISION,
            dataset_id=cal["dataset_id"],
            dataset_revision=cal["dataset_revision"],
            calibration_split=cal["split"],
            evaluation_sample_ids=tuple(ev["sample_ids"]),
            seed=plan["seed"],
        )
        profile_path = output / "frozen_profile.json"
        write_once(profile_path, profile.to_json_object())
        torch.cuda.synchronize()
        summary.update(
            offline_calibration_seconds=time.perf_counter() - start,
            calibration_captures=captures,
            profile_sha256=profile.profile_sha256,
            profile_file_sha256=digest(profile_path),
            profile_frozen_before_evaluation=True,
        )
        arms = allocation_configs(source, task, profile_path, profile)
        if tuple(arms) != ARMS:
            raise RuntimeError("unexpected experiment arms")
        parity = dict(status="running", points={}, source_sha=plan["source_sha"])
        try:
            for name, function in (
                ("ordinary_generate", compare_with_generate),
                ("retention_one", compare_cache_reinjection),
            ):
                result = function(adapter, prepared, max_new_tokens=16)
                parity[name] = asdict(result)
                if result.token_agreement != 1 or result.maximum_logit_difference > 1e-4:
                    raise RuntimeError(f"{name} parity failed")
            full = FullKVState.from_cache_snapshot(
                adapter.extract_past_key_values(prefill.state.past_key_values),
                block_size=1,
                mandatory_logical_positions=(0, prepared.logical_sequence_length - 1),
                modality_spans=prepared.modality_map,
                token_ids=prepared.model_inputs["input_ids"],
            )
            for name in ARMS[2:6]:
                cfg = arms[name]
                candidate = build_prefixkv_reimpl_plan(
                    full,
                    prefill.attention_weights,
                    cfg.prefixkv,
                    cfg.cache,
                    model_id=MODEL,
                    model_revision=REVISION,
                )
                if candidate.retained_bytes != CAP:
                    raise RuntimeError("calibration parity plan did not fill exactly 40 MiB")
                parity["points"][name] = compare_upstream_decode(
                    adapter,
                    prefill,
                    candidate,
                    source,
                    steps=64,
                    atol=1e-4,
                )
            parity["status"] = "passed"
        finally:
            write_once(output / "parity.json", parity)
        del full, prefill, prepared, candidate
        settle(adapter)
        summary["parity_sha256"] = digest(output / "parity.json")
        for order, (name, config) in enumerate(arms.items()):
            try:
                record = record_hf_run(
                    config,
                    adapter,
                    task=task,
                    run_id=f"prefix-allocation-{task}-{name}",
                    output_root=output,
                    ledger_path=args.root / job["evaluation"],
                    ledger_sha=digest(args.root / job["evaluation"]),
                    sample_ids=tuple(ev["sample_ids"]),
                    phase=CLASSIFICATION,
                    repetition=0,
                    order=order,
                )
                record["variant"] = name
                if name != "full_kv" and any(r["active_kv_bytes"] != CAP for r in record["rows"]):
                    raise RuntimeError("evaluation failed exact-byte matching")
                summary["records"].append(record)
            except Exception as exc:
                summary["failures"].append(
                    {"variant": name, "error": f"{type(exc).__name__}: {exc}"}
                )
            settle(adapter)
            print(
                json.dumps(
                    {"phase": "quality", "variant": name, "failure_count": len(summary["failures"])}
                ),
                flush=True,
            )
        summary["status"] = "completed" if not summary["failures"] else "failed"
    except Exception as exc:
        summary.update(status="failed", fatal_error=f"{type(exc).__name__}: {exc}")
    finally:
        write_once(output / "completion.json", summary)
    return int(summary["status"] != "completed")


def summarize_task(completion, ids):
    import numpy as np

    records = {r["variant"]: r for r in completion["records"]}
    result = {
        "status": completion["status"],
        "failures": completion["failures"],
        "fatal_error": completion.get("fatal_error"),
        "planned_denominator": len(ids),
        "arms": {},
    }
    pairs = {}
    for name in ARMS:
        rows = records.get(name, {}).get("rows", [])
        completed = [r for r in rows if r["status"] == "completed"]
        complete = len(completed) == len(ids) and {r["sample_id"] for r in completed} == set(ids)
        out = {
            "attempted": len(ids),
            "completed": len(completed),
            "failed_or_unavailable": len(ids) - len(completed),
            "complete": complete,
        }
        result["arms"][name] = out
        if not complete:
            out["quality_gate"] = False
            continue
        pairs[name] = {r["sample_id"]: r for r in rows}
        values = np.array([pairs[name][sid]["task_score"] for sid in ids])
        out.update(
            task_score=float(values.mean()),
            active_bytes=sorted({r["active_kv_bytes"] for r in rows}),
        )
        if name != "full_kv":
            out.update(
                byte_gate=all(r["active_kv_bytes"] == CAP for r in rows),
                mean_end_to_end_seconds=float(np.mean([r["end_to_end_time"] for r in rows])),
            )
    top = pairs.get("prompt_attention_topk")
    for name, rows in pairs.items():
        if name == "full_kv" or top is None:
            continue
        delta = np.array([rows[s]["task_score"] - top[s]["task_score"] for s in ids])
        rng = np.random.default_rng(20271001)
        boot = delta[rng.integers(0, len(ids), size=(10000, len(ids)))].mean(1)
        out = result["arms"][name]
        out.update(
            paired_score_delta=float(delta.mean()),
            paired_score_delta_ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
            answer_agreement=float(np.mean([rows[s]["answer"] == top[s]["answer"] for s in ids])),
            quality_gate=bool(out["byte_gate"] and delta.mean() >= -1e-12),
        )
    return result


def analyze(args):
    plan, _ = check_plan(args.root, args.plan_sha)
    report = dict(
        source_sha=plan["source_sha"],
        plan_sha256=args.plan_sha,
        classification=CLASSIFICATION,
        full_campaign_authorized=False,
        timing_claim="informational_0_warmups_1_trial",
        fullkv_latency_comparator=False,
        tasks={},
        inputs={},
    )
    for job in plan["jobs"]:
        path = args.root / "outputs" / job["task"] / "completion.json"
        if not path.exists():
            report["tasks"][job["task"]] = {"status": "missing", "quality_gate": False}
            continue
        summary = json.loads(path.read_text())
        if summary["source_sha"] != plan["source_sha"] or summary["plan_sha256"] != args.plan_sha:
            raise RuntimeError("worker source/plan mismatch")
        ids = json.loads((args.root / job["evaluation"]).read_text())["sample_ids"]
        report["tasks"][job["task"]] = summarize_task(summary, ids)
        report["inputs"][str(path)] = digest(path)
    write_once(args.root / "analysis/report.json", report)
    print(json.dumps(report, indent=2))
    return 0 if all(t["status"] == "completed" for t in report["tasks"].values()) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha")
    parser.add_argument("--index", type=int, choices=(0, 1))
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--expected-git-sha")
    parser.add_argument("--parent-root", type=Path)
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        if not args.expected_git_sha or not args.parent_root:
            parser.error("--prepare requires --expected-git-sha and --parent-root")
        prepare(args.root, args.expected_git_sha, args.parent_root)
    elif not args.plan_sha or (not args.analyze and args.index is None):
        parser.error("execution requires --plan-sha and --index or --analyze")
    else:
        raise SystemExit(analyze(args) if args.analyze else run(args))
