#!/usr/bin/env python3
"""Clean HF C-only/LOOK-M validation; frozen profiles, independent stage ledgers.

Preparation and execution never submit jobs. The larger-stage worker fails closed
unless the complete validation report passed for both tasks with this plan hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

from run_byte_capped_published_baseline_pilot import _require_clean_source
from run_prefixkv_decode_preflight import CAP, MODEL, REVISION, configs, digest, write_once

from mosaickv.config import canonical_config, config_sha256
from mosaickv.evaluation.hf_matched_validation import (
    CANDIDATE,
    CONTROL,
    LOOKM_METHODS,
    PREFIX_METHODS,
    matched_configs,
    select_disjoint_ids,
    summarize,
)

TASKS = ("textvqa", "docvqa")
CAPS_MIB = (10, 20, 40, 80)
CLASSIFICATION = "canonical_retrospective_hf_matched_validation_v1"


def freeze_cap_profiles(profile_root, task, output, bound):
    """Re-use label-free captures, never new task answers, for fixed cap curves."""
    from mosaickv.baselines.prefixkv import (
        PrefixKVCalibrationObservation,
        generate_prefixkv_byte_profile,
    )

    frozen_path = profile_root / "outputs" / task / "frozen_ablations.json"
    frozen = json.loads(frozen_path.read_text())
    bound[str(frozen_path.resolve())] = digest(frozen_path)
    observations = []
    for sid, info in sorted(frozen["calibration_captures"].items()):
        path = Path(info["path"])
        if digest(path) != info["sha256"]:
            raise RuntimeError("frozen prompt-only calibration capture changed")
        d = json.loads(path.read_text())
        if d["sample_id"] != sid or d["bytes_per_layer_position"] != 16384:
            raise RuntimeError("calibration identity/geometry mismatch")
        observations.append(
            PrefixKVCalibrationObservation(
                sample_id=sid,
                layer_scores=tuple(tuple(x) for x in d["layer_scores"]),
                question_positions=tuple(d["question_positions"]),
                question_layer_scores=tuple(tuple(x) for x in d["question_layer_scores"]),
                question_mapping_sha256=d["question_mapping_sha256"],
            )
        )
        bound[str(path.resolve())] = digest(path)
    profiles, excluded = {}, set()
    for cap in CAPS_MIB:
        profiles[str(cap)] = {}
        for arm, alpha in [("control", 0.0), ("c_rank", 0.5)]:
            path = profile_root / "outputs" / task / "profiles" / f"{arm}.json"
            data = json.loads(path.read_text())
            cal = data["calibration"]
            if set(cal["sample_ids"]) != {o.sample_id for o in observations}:
                raise RuntimeError("profile/capture identity mismatch")
            excluded.update(cal["sample_ids"])
            target = output / "profiles" / task / f"cap{cap}MiB" / path.name
            if cap == 40:
                # Preserve the exact previously frozen operating point.
                write_once(target, data)
            else:
                profile = generate_prefixkv_byte_profile(
                    tuple(observations),
                    byte_cap=cap * 2**20,
                    bytes_per_layer_position=16384,
                    model_id=MODEL,
                    model_revision=REVISION,
                    dataset_id=cal["dataset_id"],
                    dataset_revision=cal["dataset_revision"],
                    calibration_split=cal["split"],
                    evaluation_sample_ids=(),
                    seed=cal["seed"],
                    exclude_protected_from_calibration=False,
                    prompt_reservation_policy="boundaries",
                    question_score_weight=alpha,
                )
                profile.write(target)
            profiles[str(cap)][arm] = str(target.resolve())
            bound[str(path.resolve())] = digest(path)
            bound[str(target.resolve())] = digest(target)
    return profiles, excluded


def prepare(args):
    source = Path(__file__).resolve().parents[2]
    _require_clean_source(source, args.source_sha)
    if args.root.exists():
        raise FileExistsError("new immutable root required")
    if args.validation_count % args.shard_size or args.larger_count % args.shard_size:
        raise ValueError("stage counts must divide into complete fixed shards")
    plan = dict(
        schema_version=1,
        classification=CLASSIFICATION,
        source_sha=args.source_sha,
        source_root=str(source),
        heldout_claim=False,
        paper_claim_authorized=False,
        validation_authorized=True,
        larger_authorized_if_validation_passes=True,
        lookm_validation_authorized=True,
        curves_authorized_if_validation_passes=True,
        larger_caps_mib=list(CAPS_MIB),
        max_global_a100s=4,
        cap_bytes=CAP,
        output_tokens=64,
        warmups=5,
        quality_repetitions=1,
        latency="paired per-example, informational; not repeated microbenchmark",
        seed=20271001,
        parity_tokens=16,
        parity_atol=1e-4,
        quality_gate="observed parity vs Top-K and 95% paired lower bound >= -0.05, both tasks",
        no_30_percent_claim_from_equal_cap=True,
        decoder_policy="append_only_for_every_compressed_method",
        profile_selection="C-only chosen from prior development ablations; frozen, no new fitting",
        validation_count=args.validation_count,
        larger_count=args.larger_count,
        jobs=[],
        profiles={},
        bound_files={},
    )
    for task in TASKS:
        parents = sorted((args.parent_root / task).glob("*.json"))
        if not parents:
            raise ValueError("no parent ID ledgers")
        payloads = [json.loads(p.read_text()) for p in parents]
        keys = ("task", "dataset_id", "dataset_revision", "split")
        meta = {k: payloads[0][k] for k in keys}
        if meta["task"] != task or any(any(p[k] != meta[k] for k in keys) for p in payloads):
            raise ValueError("parent task/dataset provenance differs")
        excluded = set()
        for root in args.exclude_root:
            paths = sorted(root.glob(f"{task}_*.json"))
            if not paths:
                raise ValueError(f"missing declared exclusion ledgers: {root}/{task}")
            for path in paths:
                data = json.loads(path.read_text())
                excluded.update(data["sample_ids"])
                plan["bound_files"][str(path.resolve())] = digest(path)
        plan["profiles"][task], cal_ids = freeze_cap_profiles(
            args.profile_root, task, args.root, plan["bound_files"]
        )
        excluded.update(cal_ids)
        prior_plan = args.profile_root / "plan.json"
        plan["bound_files"][str(prior_plan.resolve())] = digest(prior_plan)
        ids = [sid for p in payloads for sid in p["sample_ids"]]
        small, large = select_disjoint_ids(
            ids, excluded, validation_count=args.validation_count, larger_count=args.larger_count
        )
        for p in parents:
            plan["bound_files"][str(p.resolve())] = digest(p)
        for stage, selected in (("validation", small), ("larger", large)):
            for shard, offset in enumerate(range(0, len(selected), args.shard_size)):
                path = args.root / "ledgers" / stage / task / f"{shard:02d}.json"
                subset = selected[offset : offset + args.shard_size]
                write_once(
                    path,
                    dict(
                        schema_version=1,
                        **meta,
                        sample_ids=subset,
                        sample_count=len(subset),
                        heldout_claim=False,
                        classification=CLASSIFICATION,
                        selection_rule="parent ID order minus declared development/calibration IDs",
                    ),
                )
                plan["bound_files"][str(path.resolve())] = digest(path)
                for cap in (40,) if stage == "validation" else CAPS_MIB:
                    plan["jobs"].append(
                        dict(
                            task=task,
                            stage=stage,
                            shard=shard,
                            cap_mib=cap,
                            ledger=str(path.resolve()),
                            ledger_sha256=digest(path),
                        )
                    )
    write_once(args.root / "plan.json", plan)
    print(
        json.dumps(
            {
                "root": str(args.root),
                "plan_sha256": digest(args.root / "plan.json"),
                "jobs": len(plan["jobs"]),
                "submitted": False,
            }
        )
    )


def check_plan(root, expected_hash):
    if digest(root / "plan.json") != expected_hash:
        raise RuntimeError("plan changed")
    plan = json.loads((root / "plan.json").read_text())
    source = Path(__file__).resolve().parents[2]
    _require_clean_source(source, plan["source_sha"])
    if (
        source.resolve() != Path(plan["source_root"]).resolve()
        or plan["classification"] != CLASSIFICATION
        or plan["max_global_a100s"] != 4
        or plan["cap_bytes"] != CAP
        or plan["output_tokens"] != 64
        or plan["larger_caps_mib"] != list(CAPS_MIB)
        or plan["decoder_policy"] != "append_only_for_every_compressed_method"
    ):
        raise RuntimeError("unexpected protocol scope")
    for name, sha in plan["bound_files"].items():
        if digest(name) != sha:
            raise RuntimeError(f"bound artifact changed: {name}")
    return plan, source


def job_configurations(source, plan, task, lane, cap_mib=40):
    base = configs(source, task)
    controls = {k: base[k] for k in ("full_kv", "prompt_attention_topk", "mosaickv_exact")}
    controls["full_kv"] = replace(
        controls["full_kv"],
        cache=replace(
            controls["full_kv"].cache,
            block_size=1,
            physical_budget_block_size=1,
            physical_budget_mandatory_policy="first_and_last",
        ),
    )
    for name in controls:
        if name != "full_kv":
            controls[name] = replace(
                controls[name], cache=replace(controls[name].cache, budget_value=cap_mib * 2**20)
            )
    return matched_configs(
        controls, plan["profiles"][task][str(cap_mib)] if lane == "prefixkv" else None, lane=lane
    )


def require_larger_gate(root, plan_sha):
    path = root / "analysis/prefixkv-validation.json"
    data = json.loads(path.read_text())
    if (
        data.get("plan_sha256") != plan_sha
        or not data.get("all_gates_passed")
        or set(data.get("tasks", {})) != set(TASKS)
        or not all(t.get("gates_passed") for t in data.get("tasks", {}).values())
    ):
        raise RuntimeError("larger campaign blocked: paired validation has not passed")
    for p, sha in data["inputs"].items():
        if digest(p) != sha:
            raise RuntimeError("gate input changed")


def enrich_record(record, output, variant, source_sha, cap, config):
    traces = {}
    for path in (output / "traces" / record["run_id"]).glob("*.json"):
        trace = json.loads(path.read_text())
        if trace["sample_id"] in traces:
            raise RuntimeError("duplicate sample trace")
        traces[trace["sample_id"]] = (trace, path)
    for row in record["rows"]:
        trace, path = traces[row["sample_id"]]
        manifest = json.loads(Path(row["manifest_path"]).read_text())
        if (
            manifest["source"]["git_sha"] != source_sha
            or manifest["source"]["git_dirty"]
            or not manifest["source"]["canonical_eligible"]
            or manifest["source"]["config_sha"] != config_sha256(config)
        ):
            raise RuntimeError("noncanonical source provenance")
        if trace.get("status") != "completed" or len(trace.get("generated_token_ids", [])) != 64:
            raise RuntimeError("incomplete trace/output")
        if row["sample_id"].startswith("textvqa:"):
            references = json.loads(row["reference"])
            if (
                not isinstance(references, list)
                or not references
                or not all(isinstance(r, str) for r in references)
            ):
                raise RuntimeError("TextVQA requires its annotation list, not a template field")
        events = trace.get("prefixkv_decode_events", [])
        evictions = sum(e.get("removed_logical_position") is not None for e in events)
        if variant != "full_kv":
            if (
                trace["active_kv_bytes"] != cap
                or trace.get("residual_kv_bytes") != 0
                or trace.get("repair_events")
                or evictions
            ):
                raise RuntimeError("shared-byte append-only contract failed")
            if variant in (CANDIDATE, CONTROL) and trace["effective_method"] != variant:
                raise RuntimeError("PrefixKV variant label differs from execution")
        row.update(
            final_active_kv_bytes=trace["final_active_kv_bytes"],
            decode_evictions=evictions,
            trace_validated=True,
            trace_path=str(path),
            trace_sha256=digest(path),
            effective_method=trace.get("effective_method"),
            token_ids=trace["generated_token_ids"],
            input_controls_sha256=hashlib.sha256(
                json.dumps(
                    {
                        k: manifest[k]
                        for k in ("model", "dataset", "execution", "generation", "inputs")
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest(),
            cache_controls_sha256=hashlib.sha256(
                json.dumps(manifest["resolved_config"]["cache"], sort_keys=True).encode()
            ).hexdigest(),
        )
    record["variant"] = variant
    return record


def run(args):
    import torch
    from run_prefixkv_allocation_preflight import collect

    from mosaickv.adapters.huggingface import (
        compare_cache_reinjection,
        compare_with_generate,
        load_hf_adapter,
    )
    from mosaickv.evaluation.worker_support import record_hf_run, settle
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    plan, source = check_plan(args.root, args.plan_sha)
    jobs = [j for j in plan["jobs"] if j["stage"] == args.stage]
    if not 0 <= args.index < len(jobs):
        raise ValueError("array index outside registered stage")
    job = jobs[args.index]
    if args.stage == "larger":
        if args.lane != "prefixkv":
            raise RuntimeError("only PrefixKV larger stage is registered")
        require_larger_gate(args.root, args.plan_sha)
    output = (
        args.root
        / "outputs"
        / args.lane
        / args.stage
        / job["task"]
        / f"cap{job['cap_mib']}MiB"
        / f"{job['shard']:02d}"
    )
    output.mkdir(parents=True, exist_ok=False)
    ledger = json.loads(Path(job["ledger"]).read_text())
    cfgs = job_configurations(source, plan, job["task"], args.lane, job["cap_mib"])
    summary = dict(
        status="running",
        records=[],
        failures=[],
        plan_sha256=args.plan_sha,
        task=job["task"],
        stage=args.stage,
        lane=args.lane,
        source_sha=plan["source_sha"],
        source_dirty=False,
        slurm_job_id=os.getenv("SLURM_JOB_ID"),
        classification=CLASSIFICATION,
        heldout_claim=False,
        paper_claim_authorized=False,
        planned_ids=ledger["sample_ids"],
        cap_mib=job["cap_mib"],
        warmups=plan["warmups"],
    )
    namespace = (
        f"matched-{args.lane}-{args.stage}-{job['task']}-"
        f"cap{job['cap_mib']}MiB-shard{job['shard']:02d}"
    )
    started = time.perf_counter()
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("exactly one allocated GPU required")
        torch.manual_seed(plan["seed"])
        torch.use_deterministic_algorithms(True)
        adapter = load_hf_adapter(
            MODEL,
            revision=REVISION,
            model_kwargs=_model_kwargs(cfgs["full_kv"], local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        summary["gpu_environment"] = asdict(capture_gpu_environment(torch))
        warm = output / "warmup_ledger.json"
        write_once(warm, {**ledger, "sample_ids": ledger["sample_ids"][:1], "sample_count": 1})
        # Prepare exactly one real input without using its annotation for selection.
        _, _, prepared, prefill = collect(
            adapter, json.loads(warm.read_text()), output / "parity_input"
        )
        parity = {}
        for label, fn in (
            ("ordinary_generate", compare_with_generate),
            ("reinjection", compare_cache_reinjection),
        ):
            result = fn(adapter, prepared, max_new_tokens=plan["parity_tokens"])
            parity[label] = asdict(result)
            if (
                result.token_agreement != 1
                or not math.isfinite(result.maximum_logit_difference)
                or result.maximum_logit_difference > plan["parity_atol"]
            ):
                raise RuntimeError("FullKV parity failed")
        write_once(output / "parity.json", dict(status="passed", **parity))
        del prepared, prefill
        # Rotate order deterministically across shards, identically for both tasks.
        order = list(cfgs)
        shift = job["shard"] % len(order)
        order = order[shift:] + order[:shift]
        write_once(
            output / "resolved_configs.json",
            {
                k: {"config": canonical_config(v), "sha256": config_sha256(v)}
                for k, v in cfgs.items()
            },
        )
        for rank, variant in enumerate(order):
            check_plan(args.root, args.plan_sha)
            cfg = cfgs[variant]
            for rep in range(plan["warmups"]):
                record_hf_run(
                    cfg,
                    adapter,
                    task=job["task"],
                    run_id=f"{namespace}-warmup-{variant}-{rep}",
                    output_root=output,
                    ledger_path=warm,
                    ledger_sha=digest(warm),
                    sample_ids=tuple(ledger["sample_ids"][:1]),
                    phase="warmup_excluded_from_analysis",
                    repetition=rep,
                    order=rank,
                )
                settle(adapter)
            run_id = f"{namespace}-{variant}"
            try:
                record = record_hf_run(
                    cfg,
                    adapter,
                    task=job["task"],
                    run_id=run_id,
                    output_root=output,
                    ledger_path=Path(job["ledger"]),
                    ledger_sha=job["ledger_sha256"],
                    sample_ids=tuple(ledger["sample_ids"]),
                    phase=CLASSIFICATION,
                    repetition=0,
                    order=rank,
                )
                summary["records"].append(
                    enrich_record(
                        record, output, variant, plan["source_sha"], job["cap_mib"] * 2**20, cfg
                    )
                )
            except Exception as exc:
                summary["failures"].append(
                    dict(
                        variant=variant,
                        error=f"{type(exc).__name__}: {exc}",
                        planned_sample_ids=ledger["sample_ids"],
                        raw_path=str(output / run_id / "raw.jsonl"),
                    )
                )
            settle(adapter)
            write_once(output / "progress" / f"{rank:02d}.json", summary)
            print(
                json.dumps({"variant": variant, "failures": len(summary["failures"])}), flush=True
            )
        summary["status"] = "completed" if not summary["failures"] else "failed"
    except Exception as exc:
        summary.update(status="failed", fatal_error=f"{type(exc).__name__}: {exc}")
    finally:
        summary["wall_seconds"] = time.perf_counter() - started
        write_once(output / "completion.json", summary)
    return int(summary["status"] != "completed")


def analyze(args):
    plan, _ = check_plan(args.root, args.plan_sha)
    report = dict(
        plan_sha256=args.plan_sha,
        inputs={},
        tasks={},
        all_gates_passed=False,
        classification=CLASSIFICATION,
        paper_claim_authorized=False,
    )
    methods = PREFIX_METHODS if args.lane == "prefixkv" else LOOKM_METHODS
    for task, cap in (
        (t, c) for t in TASKS for c in ((40,) if args.stage == "validation" else CAPS_MIB)
    ):
        records, ids, failures = [], [], []
        for job in plan["jobs"]:
            if job["stage"] != args.stage or job["task"] != task or job["cap_mib"] != cap:
                continue
            ids.extend(json.loads(Path(job["ledger"]).read_text())["sample_ids"])
            path = (
                args.root
                / "outputs"
                / args.lane
                / args.stage
                / task
                / f"cap{cap}MiB"
                / f"{job['shard']:02d}"
                / "completion.json"
            )
            if not path.exists():
                failures.append(f"missing: {path}")
                continue
            d = json.loads(path.read_text())
            if d["plan_sha256"] != args.plan_sha:
                raise RuntimeError("worker plan mismatch")
            report["inputs"][str(path)] = digest(path)
            records.extend(d["records"])
            if d["status"] != "completed":
                failures.append(f"failed: {path}")
        result = summarize(records, ids, methods=methods, cap=cap * 2**20)
        result["worker_failures"] = failures
        result["gates_passed"] = result["gates_passed"] and not failures
        key = task if args.stage == "validation" else f"{task}/cap{cap}MiB"
        report["tasks"][key] = result
    report["all_gates_passed"] = all(t["gates_passed"] for t in report["tasks"].values())
    path = args.root / "analysis" / f"{args.lane}-{args.stage}.json"
    write_once(path, report)
    print(json.dumps({"report": str(path), "all_gates_passed": report["all_gates_passed"]}))
    return int(not report["all_gates_passed"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--source-sha")
    parser.add_argument("--parent-root", type=Path)
    parser.add_argument("--profile-root", type=Path)
    parser.add_argument("--exclude-root", type=Path, action="append", default=[])
    parser.add_argument("--validation-count", type=int, default=64)
    parser.add_argument("--larger-count", type=int, default=256)
    parser.add_argument("--shard-size", type=int, default=32)
    parser.add_argument("--plan-sha")
    parser.add_argument("--lane", choices=("prefixkv", "lookm"), default="prefixkv")
    parser.add_argument("--stage", choices=("validation", "larger"), default="validation")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--index", type=int)
    mode.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        if not all((args.source_sha, args.parent_root, args.profile_root, args.exclude_root)):
            parser.error(
                "preparation needs clean source, parent pool, frozen profiles and exclusions"
            )
        prepare(args)
    elif args.plan_sha and (args.analyze or args.index is not None):
        raise SystemExit(analyze(args) if args.analyze else run(args))
    else:
        parser.error("select --prepare, --analyze or --index with --plan-sha")
