#!/usr/bin/env python3
"""Write-once two-input HF gate and finite afterany analysis; never a cohort."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import yaml
from bounded_history_pilot import SOURCE, clean_source, digest, write_once
from head_substitution_v7 import object_sha

from mosaickv.config import DraftAllocationGuardConfig, canonical_config, config_sha256, load_config
from mosaickv.prototypes.attention_compaction import (
    AttentionCompactionPolicy,
    plan_compaction_capacity,
)

PROTOCOL = SOURCE / "mosaickv/configs/experiments/attention_compaction_hf_gate_v1.yaml"
BLOCKED = dict(
    fresh_cohort_authorized=False, calibration_authorized=False, paper_campaign_authorized=False
)
METHOD = "attention_matching_protected_reimpl"


def manifest_control_config(config, core):
    """Keep frozen model controls; never misreport the parent's 40 MiB cap."""
    accounting_sha = object_sha(
        dict(
            representation=METHOD,
            boundary="unique_KV_bias_masks_and_decode_metadata_backing_storage",
            original_kv_accounting_spec_sha=config.cache.accounting_spec_sha,
            history_positions=core["capacity_planning"]["history_positions"],
        )
    )
    return replace(
        config,
        cache=replace(
            config.cache,
            budget_value=core["capacity_planning"]["cap_bytes"],
            accounting_spec_sha=accounting_sha,
        ),
    )


def validate_protocol(protocol):
    if (
        any(protocol[k] is not False for k in BLOCKED)
        or protocol["gpu_execution_authorized"] is not True
        or protocol["effective_method"] != METHOD
        or protocol["guard"]
        != dict(
            horizon=32, checkpoints=[1, 4, 16, 32], absolute_tolerance=1e-7, relative_tolerance=1e-6
        )
        or protocol["retention_one"]
        != dict(horizon=32, maximum_logit_difference=0.0, token_agreement=1.0)
        or protocol["history"] != dict(capacity=16, sink_tokens=4)
        or protocol["required_mandatory_policy"] != "first_and_last"
        or protocol["dictionary"] != "stable_mean_draft_attention_topk"
        or protocol["maximum_joint_candidates"] != 1
        or protocol["controls"] != ["clone", "repack_after_candidate"]
        or tuple(
            protocol[k]
            for k in (
                "workers",
                "gpus_per_worker",
                "cpus_per_worker",
                "max_global_a100s",
                "time_limit_seconds",
            )
        )
        != (2, 1, 12, 6, 1800)
    ):
        raise ValueError("HF compaction scope, thresholds, or bounded controls changed")
    DraftAllocationGuardConfig.from_mapping(protocol["guard"])


def prepare(root):
    sha = clean_source()
    if root.exists() or Path("/scratch/djy8hg").resolve() not in root.resolve().parents:
        raise ValueError("require a NEW immutable scratch output root")
    protocol = yaml.safe_load(PROTOCOL.read_text())
    validate_protocol(protocol)
    core_path = SOURCE / protocol["core_protocol"]
    report_path = Path(protocol["cpu_feasibility_report"])
    parent_path = Path(protocol["consumed_parent"]) / "plan.json"
    bound = {str(PROTOCOL): digest(PROTOCOL)}
    for path, expected in (
        (core_path, protocol["core_protocol_sha256"]),
        (report_path, protocol["cpu_feasibility_sha256"]),
        (parent_path, protocol["consumed_parent_plan_sha256"]),
    ):
        if digest(path) != expected:
            raise ValueError("compaction core, feasibility evidence, or consumed inputs changed")
        bound[str(path)] = expected
    core = yaml.safe_load(core_path.read_text())
    capacity = plan_compaction_capacity(**core["capacity_planning"])
    parent = json.loads(parent_path.read_text())
    jobs = []
    for original in parent["jobs"]:
        index = original["index"]
        config_path, payload_path = Path(original["config"]), Path(original["input"])
        for path in (config_path, payload_path):
            if digest(path) != parent["bound_files"][str(path)]:
                raise ValueError("consumed parent input/control configuration changed")
            bound[str(path)] = digest(path)
        cfg = load_config(config_path)
        payload = json.loads(payload_path.read_text())
        if (
            config_sha256(cfg) != original["config_sha"]
            or cfg.cache.physical_budget_mandatory_policy != "first_and_last"
            or cfg.model.id != "llava-hf/llava-1.5-7b-hf"
            or cfg.model.revision != "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
            or cfg.model.precision.value != "fp16"
            or cfg.execution.attention_implementation != "eager"
            or set(payload)
            != {
                "task",
                "dataset_id",
                "dataset_revision",
                "split",
                "sample_id",
                "prompt",
                "media",
                "draft_conditioned",
            }
        ):
            raise ValueError("gate must reuse the fixed consumed, label-redacted LLaVA controls")
        payload.pop("draft_conditioned")
        path = root / f"input-{index}.json"
        write_once(path, payload)
        bound[str(path)] = digest(path)
        jobs.append(
            dict(
                index=index,
                task=original["task"],
                input=str(path),
                config=str(config_path),
                control_config_sha=config_sha256(cfg),
            )
        )
    if [job["index"] for job in jobs] != [0, 1]:
        raise ValueError("only the same two consumed inputs are allowed")
    result = dict(
        source_sha=sha,
        protocol=protocol,
        core=core,
        capacity_planning=asdict(capacity),
        bound_files=bound,
        jobs=jobs,
        new_identities=0,
        **BLOCKED,
    )
    write_once(root / "plan.json", result)
    return result


def validate(root, plan_sha):
    if digest(root / "plan.json") != plan_sha:
        raise ValueError("HF gate plan changed")
    plan = json.loads((root / "plan.json").read_text())
    clean_source(plan["source_sha"])
    validate_protocol(plan["protocol"])
    if any(plan[k] is not False for k in BLOCKED):
        raise ValueError("larger campaigns remain blocked")
    for path, expected in plan["bound_files"].items():
        if digest(path) != expected:
            raise ValueError("HF gate bound file changed")
    return plan


def worker(root, plan_sha, index):
    import torch

    from mosaickv.adapters.huggingface import load_hf_adapter
    from mosaickv.evaluation.attention_compaction_diagnostic import diagnose_bias_transport
    from mosaickv.evaluation.messages import MediaItem, MediaKind, build_multimodal_messages
    from mosaickv.evaluation.paper_native_lanes import MaterializedMedia, _media_payload
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.manifest import (
        ArtifactProvenance,
        EvaluationProvenance,
        InputProvenance,
        RunManifestWriter,
    )
    from mosaickv.types import MeasurementType

    plan = validate(root, plan_sha)
    if type(index) is not int or not 0 <= index < 2:
        raise ValueError("invalid consumed gate worker index")
    job = plan["jobs"][index]
    cfg, payload = load_config(job["config"]), json.loads(Path(job["input"]).read_text())
    output = root / "workers" / str(index)
    if output.exists():
        raise FileExistsError("GPU gate outputs are immutable")
    output.mkdir(parents=True)
    events = output / "phase_events.jsonl"
    prepared, adapter, artifacts = None, None, {}
    try:
        if (
            os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
            or torch.cuda.device_count() != 1
            or "A100" not in torch.cuda.get_device_name(0)
        ):
            raise RuntimeError("gate requires one A100 and deterministic CUBLAS before Python")
        torch.set_num_threads(12)
        torch.manual_seed(cfg.execution.seed)
        torch.use_deterministic_algorithms(True)
        media = [
            MaterializedMedia(MediaKind(m["kind"]), Path(m["path"]), m["sha256"])
            for m in payload["media"]
        ]
        messages = build_multimodal_messages(
            payload["prompt"], tuple(MediaItem(m.kind, _media_payload(m)) for m in media)
        )
        adapter = load_hf_adapter(
            cfg.model.id,
            revision=cfg.model.revision,
            model_kwargs=_model_kwargs(cfg, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        prepared = adapter.prepare_inputs(messages)
        torch.cuda.reset_peak_memory_stats()
        with events.open("x") as handle:

            def event(row):
                handle.write(json.dumps(row, allow_nan=False) + "\n")
                handle.flush()

            result, artifacts = diagnose_bias_transport(
                adapter,
                prepared,
                prompt_positions=plan["capacity_planning"]["prompt_positions"],
                cap_bytes=plan["core"]["capacity_planning"]["cap_bytes"],
                policy=AttentionCompactionPolicy(**plan["core"]["policy"]),
                event=event,
            )
        record = dict(status="completed", result=result)
    except Exception as error:
        record = dict(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            effective_method=METHOD,
            phase_events=str(events),
            **BLOCKED,
        )
    write_once(output / "diagnostic.json", record)
    for name, value in artifacts.items():
        torch.save(value, output / f"{name}.pt")
    files = {str(p): digest(p) for p in output.rglob("*") if p.is_file()}
    write_once(output / "artifact-hashes.json", files)
    preprocessing = dict(
        processor=type(adapter.processor).__name__ if adapter else "load_failed",
        model=cfg.model.id,
        revision=cfg.model.revision,
        config=getattr(adapter.processor, "to_dict", lambda: {})() if adapter else {},
    )
    manifest_cfg = manifest_control_config(cfg, plan["core"])
    effective_config_sha = object_sha(
        dict(
            model_controls=canonical_config(manifest_cfg),
            protocol=plan["protocol"],
            compaction_core=plan["core"],
        )
    )
    manifest = RunManifestWriter(SOURCE).write(
        output / "manifest.json",
        manifest_cfg,
        MeasurementType.VALIDATION_SMOKE,
        InputProvenance(
            object_sha(payload["prompt"]),
            object_sha(payload["media"]),
            object_sha(preprocessing),
            object_sha(
                prepared.model_inputs["input_ids"].cpu().tolist() if prepared else "prepare_failed"
            ),
        ),
        ArtifactProvenance(
            digest(output / "diagnostic.json"),
            "not_applicable",
            digest(output / "artifact-hashes.json"),
        ),
        run_id=f"attention-compaction-hf-v1-{index}",
        execution_metadata=dict(
            phase="consumed_input_compaction_engineering_not_benchmark_quality",
            effective_method=METHOD,
            effective_config_sha=effective_config_sha,
            frozen_parent_control_config_sha=job["control_config_sha"],
            retention_ratio_semantics="unused_sentinel_selection_is_physical_byte_capped",
            control_config_purpose="model/processor controls only; mosaickv_exact not executed",
            experimental_resolved_config=dict(protocol=plan["protocol"], compaction=plan["core"]),
            plan_sha256=plan_sha,
            slurm_job_id=os.environ.get("SLURM_JOB_ID"),
            benchmark_labels_accessed=False,
            **BLOCKED,
        ),
        evaluation_provenance=EvaluationProvenance(
            evaluator_name="unchanged_32_prefix_guard",
            evaluator_version="compaction-v1",
            evaluator_commit=plan["source_sha"],
            reference_output_sha256=digest(output / "logits.pt") if artifacts else "not_used",
            baseline_name="same_dictionary_exact_zero_bias_anchor",
            baseline_commit=plan["source_sha"],
        ),
    )
    write_once(
        output / "completed.json",
        dict(
            status=record["status"],
            manifest=str(manifest),
            manifest_sha256=digest(manifest),
            diagnostic_sha256=digest(output / "diagnostic.json"),
            effective_config_sha=effective_config_sha,
            **BLOCKED,
        ),
    )
    if record["status"] != "completed":
        raise RuntimeError("GPU gate failed; structured failure, events and manifest preserved")


def review_result(result, logits, protocol):
    """Replay authoritative GPU-FP32 decisions; FP64 remains sideband only."""
    import torch

    from mosaickv.backends.hf_allocation_guard import decide_draft_stability, draft_row_causes
    from mosaickv.selection.draft_shortlist import meaningful_improvement

    guard = DraftAllocationGuardConfig.from_mapping(protocol["guard"])
    parity = (
        len(logits["fullkv"]) == 32
        and len(logits["zero_bias_retention_one"]) == 32
        and all(
            torch.equal(logits["fullkv"][step], logits["zero_bias_retention_one"][step])
            for step in range(1, 33)
        )
    )
    controls = all(
        len(logits[role]) == 32
        and all(torch.equal(logits["anchor"][step], logits[role][step]) for step in range(1, 33))
        for role in ("null_clone", "null_repack_after_candidate")
    )
    rows = result["guard_steps"]
    decision = decide_draft_stability(rows, guard, allow_early_rejection=True)
    if decision != result["same_byte_guard"]:
        raise ValueError("serialized guard decision changed")
    failures = [r["step"] for r in rows if draft_row_causes(r, guard)]
    if failures and failures != [len(rows)]:
        raise ValueError("failed proposal did not stop at first violation")
    meaningful = decision["accepted"] and meaningful_improvement(rows, guard)
    if meaningful != result["meaningful_model_improvement"]:
        raise ValueError("serialized meaningfulness changed")
    for row in rows:
        step = row["step"]
        for role, metric in (("candidate", row["candidate"]), ("anchor", row["anchor"])):
            z, r = logits[role][step].float(), logits["fullkv"][step].float()
            if (
                float((z - r).abs().max()) != metric["maximum_logit_difference"]
                or bool(torch.equal(z.argmax(-1), r.argmax(-1))) != metric["greedy_token_agreement"]
            ):
                raise ValueError("saved logits contradict guard telemetry")
    storage_ok = bool(result["storage"]) and all(
        row["active_kv_backing_bytes"] + row["auxiliary_backing_bytes"]
        == row["total_persistent_backing_bytes"]
        <= row["cap_bytes"]
        for row in result["storage"]
    )
    # Every compressed role must have identical KV allocation, without omitting aux.
    storage_ok &= len({row["active_kv_backing_bytes"] for row in result["storage"]}) == 1
    for step in {row["step"] for row in result["storage"]}:
        storage_ok &= (
            len(
                {
                    row["total_persistent_backing_bytes"]
                    for row in result["storage"]
                    if row["step"] == step
                }
            )
            == 1
        )
    flags = (
        "source_cache_unchanged",
        "source_weights_unchanged",
        "source_logical_positions_unchanged",
        "mask_scope_closed",
    )
    safe = all(result[k] is True for k in flags) and not result["used_fullkv_fallback"]
    safe &= not result["benchmark_labels_accessed"] and result["fullkv_prefills"] == 1
    safe &= all(result[k] is False for k in BLOCKED)
    roles = ("anchor", "null_clone", "candidate", "null_repack_after_candidate")
    for role in roles:
        expected_steps = set(range(1 + (len(rows) if role == "candidate" else 32)))
        observed = [r["step"] for r in result["storage"] if r["role"] == role]
        storage_ok &= set(observed) == expected_steps and len(observed) == len(expected_steps)
    required_phases = {
        "prefill",
        "source_d2h",
        "retention_one_state_construction",
        "fullkv_draft_and_zero_bias_retention_one",
        "cpu_value_and_mass_construction",
    }
    required_phases.update(
        f"{role}_{suffix}" for role in roles for suffix in ("state_h2d_construction", "decode")
    )
    timings_ok = required_phases == set(result["phase_timing"]) and all(
        math.isfinite(r["wall_seconds"])
        and r["wall_seconds"] >= 0
        and (
            r["cuda_event_ms"] is None
            or (math.isfinite(r["cuda_event_ms"]) and r["cuda_event_ms"] >= 0)
        )
        for r in result["phase_timing"].values()
    )
    construction_bounded = (
        result["phase_timing"]["cpu_value_and_mass_construction"]["wall_seconds"]
        <= protocol["maximum_cpu_construction_seconds"]
    )
    engineering = bool(
        parity and controls and storage_ok and safe and construction_bounded and timings_ok
    )
    return dict(
        engineering_passed=engineering,
        parity_passed=bool(parity),
        controls_passed=bool(controls),
        storage_passed=bool(storage_ok),
        timing_complete=bool(timings_ok),
        guard=decision,
        meaningful_model_improvement=bool(meaningful),
        diagnostic_passed=bool(engineering and meaningful),
        construction_accepted_heads=result["construction_accepted_heads"],
        constructed_heads=result["constructed_heads"],
        candidate_prefixes=len(rows),
        **BLOCKED,
    )


def analyze(root, plan_sha):
    import torch

    plan = validate(root, plan_sha)
    tasks, errors = [], []
    for job in plan["jobs"]:
        folder = root / "workers" / str(job["index"])
        try:
            completion = json.loads((folder / "completed.json").read_text())
            if completion["status"] != "completed":
                raise ValueError("producer failed; inspect diagnostic.json and phase_events.jsonl")
            if digest(completion["manifest"]) != completion["manifest_sha256"]:
                raise ValueError("producer manifest changed")
            manifest = json.loads(Path(completion["manifest"]).read_text())
            if (
                manifest["source"]["git_sha"] != plan["source_sha"]
                or manifest["source"]["git_dirty"]
                or manifest["execution"]["effective_method"] != METHOD
                or manifest["execution"]["plan_sha256"] != plan_sha
                or manifest["source"]["config_sha"]
                != config_sha256(manifest_control_config(load_config(job["config"]), plan["core"]))
                or manifest["execution"]["frozen_parent_control_config_sha"]
                != job["control_config_sha"]
                or manifest["artifacts"]["raw_output_sha"] != digest(folder / "diagnostic.json")
                or manifest["artifacts"]["log_sha"] != digest(folder / "artifact-hashes.json")
                or completion["diagnostic_sha256"] != digest(folder / "diagnostic.json")
            ):
                raise ValueError("incorrect source, effective method or plan provenance")
            for path, sha in json.loads((folder / "artifact-hashes.json").read_text()).items():
                if digest(path) != sha:
                    raise ValueError("GPU artifact changed")
            raw = json.loads((folder / "diagnostic.json").read_text())
            if raw["status"] != "completed":
                raise ValueError(raw.get("error", "producer incomplete"))
            logits = torch.load(folder / "logits.pt", map_location="cpu", weights_only=True)
            result = raw["result"]
            if (
                result["prompt_positions"] != plan["capacity_planning"]["prompt_positions"]
                or result["effective_method"] != METHOD
                or result["constructed_heads"] != 1024
                or any(
                    row["cap_bytes"] != plan["core"]["capacity_planning"]["cap_bytes"]
                    for row in result["storage"]
                )
            ):
                raise ValueError("unregistered physical capacity or cache geometry")
            constructed = torch.load(
                folder / "constructed.pt", map_location="cpu", weights_only=True
            )
            if len(constructed) != 1024:
                raise ValueError("missing constructed head records")
            for row in constructed:
                if (
                    not torch.equal(row["keys"], row["original_keys"])
                    or row["positions"][0] != 0
                    or row["positions"][-1] != result["source_prompt_positions"] - 1
                    or not torch.equal(row["values"][[0, -1]], row["original_values"][[0, -1]])
                    or not (row["log_bias"][[0, -1]] == 0).all()
                    or not torch.isfinite(row["values"]).all()
                ):
                    raise ValueError("constructed keys, mandatory values or biases changed")
            reviewed = review_result(raw["result"], logits, plan["protocol"])
            tasks.append(dict(worker=job["index"], task=job["task"], **reviewed))
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            errors.append(
                dict(
                    worker=job["index"],
                    error=f"{type(error).__name__}: {error}",
                    diagnostic_path=str(folder / "diagnostic.json"),
                    phase_events=str(folder / "phase_events.jsonl"),
                )
            )
    passed = len(tasks) == 2 and not errors and all(r["diagnostic_passed"] for r in tasks)
    report = dict(
        status="completed" if len(tasks) == 2 and not errors else "incomplete",
        source_sha=plan["source_sha"],
        plan_sha256=plan_sha,
        tasks=tasks,
        errors=errors,
        diagnostic_passed=passed,
        no_new_inference=True,
        benchmark_quality_established=False,
        action="design_review_only_no_automatic_promotion"
        if passed
        else "negative_diagnostic_no_automatic_successor",
        **BLOCKED,
    )
    write_once(root / "analysis/report.json", report)
    write_once(
        root / "analysis/decision.json",
        dict(
            action=report["action"],
            report_sha256=digest(root / "analysis/report.json"),
            gpu_submission_authorized=False,
            **BLOCKED,
        ),
    )
    print(json.dumps(report, allow_nan=False))


def submit(root):
    plan = prepare(root)
    plan_sha = digest(root / "plan.json")
    validate(root, plan_sha)
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
            "--hold",
            "--array=0-1%2",
            f"--output={logs}/gpu-%A_%a.out",
            f"--error={logs}/gpu-%A_%a.err",
            str(SOURCE / "mosaickv/slurm/attention_compaction_hf_gate.sbatch"),
            str(SOURCE),
            str(root),
            plan_sha,
        ]
    )
    scheduler = root / "scheduler"
    scheduler.mkdir()
    for name in ("control_a100_schedule.py", "cache_env.sh"):
        shutil.copy2(SOURCE / "mosaickv/scripts" / name, scheduler / name)
    write_once(
        scheduler / "a100_schedule.json",
        dict(
            timezone="America/New_York",
            maximum_worker_minutes=120,
            maximum_a100s=6,
            six_a100s_explicitly_authorized=True,
            window="all_times",
            arrays=[array],
            adopt_existing_holds=[array],
            source_sha=plan["source_sha"],
            authorization="only_two_consumed_compaction_engineering_workers",
        ),
    )
    analysis = sbatch(
        [
            f"--dependency=afterany:{array}",
            f"--output={logs}/analysis-%j.out",
            f"--error={logs}/analysis-%j.err",
            str(SOURCE / "mosaickv/slurm/attention_compaction_analysis.sbatch"),
            str(SOURCE),
            str(root),
            plan_sha,
        ]
    )
    write_once(
        root / "submission.json",
        dict(
            array=array,
            analysis=analysis,
            plan_sha256=plan_sha,
            maximum_a100s=6,
            new_identities=0,
            **BLOCKED,
        ),
    )
    # One admission pass is enough for this two-task array: no watch/log polling.
    subprocess.run(
        [
            "/scratch/djy8hg/env/mosaickv/bin/python",
            str(scheduler / "control_a100_schedule.py"),
            "--policy",
            str(scheduler / "a100_schedule.json"),
            "--state",
            str(scheduler / "state.json"),
            "--apply",
        ],
        check=True,
    )
    print(json.dumps(dict(array=array, analysis=analysis, root=str(root))))


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
