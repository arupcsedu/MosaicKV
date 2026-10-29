#!/usr/bin/env python3
"""Same consumed MM-Vet selected prompt; append-only teacher-forcing diagnostic."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import evicastkv_lifetime_frontier as frontier
import evicastkv_lifetime_history as common
import evicastkv_lifetime_v2 as launch
from evicastkv_campaign import environment

from mosaickv.config import RunConfig, canonical_config, config_sha256
from mosaickv.evaluation.append_counterfactual import append_config, verify_intervention
from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
PARENT_SHA = "c74c12fb58ecf38cad5d6a2cec7dfe70e13e16d4327c5662aca5a3516fe24c6a"
PARENT = Path("/scratch/djy8hg/runs/evicastkv/token-eviction-consumed-20261029-v1")


def prepare(root):
    source = clean_source(SOURCE)
    if root.exists() or sha(PARENT / "plan.json") != PARENT_SHA:
        raise ValueError("new immutable root and pinned consumed parent required")
    old = common.read(PARENT / "plan.json")
    gate_path = PARENT / "analysis_cpu_recovery_v1/GATE.json"
    gate = common.read(gate_path)
    if gate["status"] != "complete_consumed_diagnostic" or gate["failures"]:
        raise ValueError("complete prior capture required")
    bound = dict(old["bound_files"])
    for p in (PARENT / "plan.json", gate_path):
        bound[str(p)] = sha(p)
    references = {}
    for role, parent_index in (("full_kv_reference", 5), ("native_prefixkv_reference", 6)):
        folder = PARENT / f"workers/{parent_index}"
        result = common.read(folder / "result.json")
        if result["status"] != "completed" or result["evaluated_tokens"] != 593:
            raise ValueError("complete original teacher references required")
        if sha(folder / "per_token.json") != result["per_token_sha256"]:
            raise ValueError("reference teacher capture changed")
        references[role] = str(folder)
        for p in (folder / "result.json", folder / "per_token.json", folder / "manifest.json"):
            bound[str(p)] = sha(p)
    jobs = []
    for parent_index in (7, 8):
        old_job = old["jobs"][parent_index]
        if old_job["task"] != "mmvet_prefixkv" or old_job["sample_id"] != "v1_197":
            raise ValueError("only the already consumed MM-Vet diagnostic is authorized")
        worker = PARENT / f"workers/{parent_index}"
        result = common.read(worker / "result.json")
        if result["status"] != "completed" or result["failures"]:
            raise ValueError("complete FIFO source required")
        trace_path = Path(result["teacher_trace"])
        if (
            sha(trace_path) != result["teacher_trace_sha256"]
            or sha(worker / "per_token.json") != result["per_token_sha256"]
        ):
            raise ValueError("FIFO capture changed")
        for p in (
            worker / "result.json",
            worker / "manifest.json",
            worker / "per_token.json",
            trace_path,
        ):
            bound[str(p)] = sha(p)
        trace = common.read(trace_path)
        _lengths, costs = frontier.geometry(trace)
        cfg = RunConfig.from_mapping(old_job["arms"][0]["config"])
        target_count = result["evaluated_tokens"]
        candidate = append_config(cfg, target_count, costs)
        jobs.append(
            dict(
                index=len(jobs),
                task=old_job["task"],
                sample_id=old_job["sample_id"],
                ledger=old_job["ledger"],
                role=old_job["diagnostic_role"],
                fifo_worker=str(worker),
                fifo_config=old_job["arms"][0]["config"],
                target_count=target_count,
                source_position_costs=costs,
                arms=[
                    dict(
                        label=old_job["diagnostic_role"] + "-append-only-teacher",
                        config=canonical_config(candidate),
                        config_sha=config_sha256(candidate),
                    )
                ],
            )
        )
    for path, digest in bound.items():
        if sha(Path(path)) != digest:
            raise ValueError(f"bound evidence changed: {path}")
    write_once(
        root / "plan.json",
        dict(
            source_sha=source,
            bound_files=bound,
            jobs=jobs,
            parent_plan_sha256=PARENT_SHA,
            protocol="same_selected_prompt_append_only_teacher_v1",
            reference_workers=references,
            scope="consumed_development_history_loss_intervention",
            sample_rule="same consumed MM-Vet input as the preceding token-eviction capture",
            measurement_type="evaluation_only_append_history_counterfactual",
            output_generation_reruns_authorized=False,
            no_prior_or_selector_changes=True,
            byte_matched_comparison=False,
            extra_memory_must_be_disclosed=True,
            fresh_evaluation_authorized=False,
            paper_campaign_authorized=False,
            maximum_a100s=6,
            requested_workers=2,
            cpus_per_worker=12,
            wall_minutes=20,
            sum_nll_absolute_tolerance=1e-8,
            pre_eviction_nll_tolerance=0.0,
            no_automatic_fresh_followup=True,
        ),
    )
    return sha(root / "plan.json")


def worker(root, expected, index):
    import torch

    import mosaickv.backends.hf_runtime as runtime_module
    from mosaickv.adapters.huggingface import load_hf_adapter
    from mosaickv.adapters.huggingface.parity import compare_with_generate
    from mosaickv.evaluation.messages import MediaItem, build_multimodal_messages
    from mosaickv.evaluation.model import EvaluationRequest
    from mosaickv.evaluation.paper_native_lanes import _media_payload, load_paper_native_ledger
    from mosaickv.evaluation.tasks import default_task_registry
    from mosaickv.evaluation.token_eviction import TeacherTokenCapture
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    plan = common.inputs(root, expected)
    job = plan["jobs"][index]
    out = root / f"workers/{index}"
    if out.exists():
        raise FileExistsError("write-once worker output required")
    out.mkdir(parents=True)
    started, capture = time.monotonic(), None
    result = dict(
        status="failed",
        failures=[],
        source_sha=plan["source_sha"],
        plan_sha256=expected,
        task=job["task"],
        sample_id=job["sample_id"],
        role=job["role"],
    )
    try:
        if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name(0):
            raise ValueError("one A100 required")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ValueError("deterministic workspace required before Python")
        torch.set_num_threads(12)
        torch.use_deterministic_algorithms(True)
        cfg = RunConfig.from_mapping(job["arms"][0]["config"])
        original = RunConfig.from_mapping(job["fifo_config"])
        if cfg != append_config(original, job["target_count"], tuple(job["source_position_costs"])):
            raise ValueError("unregistered configuration change")
        torch.manual_seed(cfg.execution.seed)
        adapter = load_hf_adapter(
            cfg.model.id,
            revision=cfg.model.revision,
            model_kwargs=_model_kwargs(cfg, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        ledger = load_paper_native_ledger(
            job["ledger"],
            expected_sha256=sha(Path(job["ledger"])),
            task=default_task_registry().resolve(job["task"]),
            dataset_revision=cfg.dataset.revision,
        )
        sample = next(s for s in ledger.samples if s.sample_id == job["sample_id"])
        messages = build_multimodal_messages(
            sample.prompt, tuple(MediaItem(m.kind, _media_payload(m)) for m in sample.media)
        )
        write_once(
            out / "manifest.json",
            dict(
                source_sha=plan["source_sha"],
                source_dirty=False,
                plan_sha256=expected,
                config_sha=config_sha256(cfg),
                config=canonical_config(cfg),
                model=asdict(cfg.model),
                dataset=asdict(cfg.dataset),
                execution=asdict(cfg.execution),
                environment=environment(),
                cuda_runtime=torch.version.cuda,
                gpu=asdict(capture_gpu_environment(torch)),
                slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                measurement_type=plan["measurement_type"],
                result_category="unified_comparison",
                result_origin="locally_measured",
                byte_matched_comparison=False,
                environment_lock_sha256=sha(SOURCE / "mosaickv/env/common/requirements.lock"),
                baseline_commit=plan["source_sha"],
                parent_fifo_worker=job["fifo_worker"],
                reference_artifact_sha=sha(Path(job["fifo_worker"]) / "per_token.json"),
                evaluator_commit=plan["source_sha"],
                ledger_sha256=sha(Path(job["ledger"])),
                no_paper_or_confirmation_claim=True,
                timings_diagnostic_only=True,
            ),
        )
        parity = compare_with_generate(adapter, adapter.prepare_inputs(messages), max_new_tokens=16)
        write_once(out / "parity.json", asdict(parity))
        if parity.token_agreement != 1 or parity.maximum_logit_difference > 0.001:
            raise ValueError("shared FullKV parity failed")
        runtime = runtime_module.HuggingFaceMosaicKVModel(
            adapter, cfg, trace_directory=out / "traces"
        )
        targets = runtime._teacher_forced_target_ids(sample.teacher_forcing_target)
        fifo_root = Path(job["fifo_worker"])
        fifo_result = common.read(fifo_root / "result.json")
        fifo_trace = common.read(fifo_result["teacher_trace"])
        fifo_detail = common.read(fifo_root / "per_token.json")
        if (
            targets != tuple(fifo_trace["generated_token_ids"])
            or len(targets) != job["target_count"]
        ):
            raise ValueError("registered teacher target changed")
        request = EvaluationRequest(
            "append-counterfactual-" + str(index),
            sample.sample_id,
            job["task"],
            messages,
            {"_mosaickv_capture_kv_ownership": True},
        )
        with TeacherTokenCapture(adapter, targets, runtime_module) as capture:
            loss = runtime.teacher_forced_nll(request, sample.teacher_forcing_target)
        detail = capture.record()
        write_once(out / "per_token.json", detail)
        traces = list((out / "traces").rglob("*.json"))
        if len(traces) != 1:
            raise ValueError("one teacher trace required")
        trace = common.read(traces[0])
        if abs(
            detail["sum_nll"] - loss.negative_log_likelihood_sum
        ) > 1e-8 or loss.evaluated_tokens != len(targets):
            raise ValueError("NLL/token count accounting mismatch")
        checks = verify_intervention(
            detail,
            fifo_detail,
            trace,
            fifo_trace,
            history_capacity=cfg.bounded_decode.generated_capacity,
        )
        result.update(
            status="completed",
            config_sha=config_sha256(cfg),
            checks=checks,
            nll_sum=loss.negative_log_likelihood_sum,
            evaluated_tokens=loss.evaluated_tokens,
            per_token_sha256=sha(out / "per_token.json"),
            teacher_trace=str(traces[0]),
            teacher_trace_sha256=sha(traces[0]),
            extra_lifetime_cap_bytes=cfg.cache.budget_value - original.cache.budget_value,
        )
    except Exception as exc:
        result["failures"].append(dict(error=f"{type(exc).__name__}: {exc}"))
        if capture is not None:
            write_once(
                out / "partial_capture.json",
                dict(
                    status="failed_partial_diagnostic",
                    records=capture.records,
                    complete_target_count=len(capture.target_ids),
                ),
            )
        raise
    finally:
        result["wall_seconds"] = time.monotonic() - started
        write_once(out / "result.json", result)


def analyze(root, expected):
    import pandas as pd

    plan = common.inputs(root, expected)
    out = root / "analysis"
    if out.exists():
        raise FileExistsError("write-once analysis required")
    out.mkdir()
    failures, rows, token_rows, checks = [], [], [], []
    reference_nll = {
        role: common.read(Path(folder) / "result.json")["nll_sum"]
        for role, folder in plan["reference_workers"].items()
    }
    for job in plan["jobs"]:
        folder = root / f"workers/{job['index']}"
        if not (folder / "result.json").exists():
            failures.append(dict(index=job["index"], error="missing worker result"))
            continue
        result = common.read(folder / "result.json")
        if result["status"] != "completed" or result["failures"]:
            failures.append(dict(index=job["index"], errors=result["failures"]))
            continue
        if result["source_sha"] != plan["source_sha"] or result["plan_sha256"] != expected:
            raise ValueError("worker provenance mismatch")
        if (
            sha(folder / "per_token.json") != result["per_token_sha256"]
            or sha(Path(result["teacher_trace"])) != result["teacher_trace_sha256"]
        ):
            raise ValueError("counterfactual capture changed")
        detail, trace = common.read(folder / "per_token.json"), common.read(result["teacher_trace"])
        fifo_root = Path(job["fifo_worker"])
        fifo_result = common.read(fifo_root / "result.json")
        fifo_detail, fifo_trace = (
            common.read(fifo_root / "per_token.json"),
            common.read(fifo_result["teacher_trace"]),
        )
        check = verify_intervention(
            detail, fifo_detail, trace, fifo_trace, history_capacity=job["target_count"] - 1
        )
        checks.append(dict(role=job["role"], **check))
        boundary = check["first_fifo_affected_prediction"]
        for mode, d, t in (("fifo256", fifo_detail, fifo_trace), ("append592", detail, trace)):
            rows.append(
                dict(
                    task=job["task"],
                    sample_id=job["sample_id"],
                    role=job["role"],
                    history=mode,
                    tokens=len(d["records"]),
                    nll_sum=d["sum_nll"],
                    teacher_forced_ppl=math.exp(d["sum_nll"] / len(d["records"])),
                    nll_minus_fullkv=d["sum_nll"] - reference_nll["full_kv_reference"],
                    nll_minus_native_prefixkv=d["sum_nll"]
                    - reference_nll["native_prefixkv_reference"],
                    nll_before_fifo_boundary=sum(r["nll"] for r in d["records"][: boundary - 1]),
                    nll_from_fifo_boundary=sum(r["nll"] for r in d["records"][boundary - 1 :]),
                    **{k: t["cache_lifecycle"][k] for k in common.STORAGE},
                )
            )
        for a, b in zip(detail["records"], fifo_detail["records"], strict=True):
            token_rows.append(
                dict(
                    role=job["role"],
                    prediction_position=a["prediction_position"],
                    target_token_id=a["target_token_id"],
                    append_nll=a["nll"],
                    fifo_nll=b["nll"],
                    append_minus_fifo_nll=a["nll"] - b["nll"],
                    teacher_argmax_changed=a["greedy_token_id"] != b["greedy_token_id"],
                )
            )
    pd.DataFrame(rows).to_parquet(out / "summary.parquet", index=False)
    pd.DataFrame(token_rows).to_parquet(out / "per_token.parquet", index=False)
    gate = dict(
        status="complete_consumed_diagnostic" if len(checks) == 2 and not failures else "failed",
        completed=len(checks),
        failures=failures,
        checks=checks,
        byte_matched_comparison=False,
        fresh_evaluation_authorized=False,
        paper_campaign_authorized=False,
        superiority_established=False,
        whole_request_peak_certified=False,
    )
    write_once(out / "GATE.json", gate)
    write_once(out / "measurements.json", dict(rows=rows, checks=checks))
    text = (
        "# Same-selected-prompt append-only teacher-forcing counterfactual\n\n"
        "Consumed MM-Vet v1_197 only; no new prior mixing or generation sweep. "
        "G=592 appends all teacher-forced updates; G=256 replaces generated history. "
        "The prompt selection, model, precision, inputs, targets and logical positions are fixed. "
        "Storage is intentionally higher: this is NOT a byte-matched performance comparison. "
        "The reported peak is live post-compression decode backing/transaction storage, "
        "not certified whole-request KV peak. Timing includes observer synchronizations "
        "and is not a latency benchmark.\n\n"
    )
    if rows:
        display = pd.DataFrame(rows).copy()
        for k in common.STORAGE:
            display[k.replace("_bytes", "_MiB")] = display.pop(k) / 2**20
        text += display.to_markdown(index=False) + "\n\n"
        text += "## Conditional effect of retaining history\n\n"
        for job in plan["jobs"]:
            pair = [r for r in rows if r["role"] == job["role"]]
            if len(pair) == 2:
                before, after = pair
                before_delta = (
                    after["nll_before_fifo_boundary"] - before["nll_before_fifo_boundary"]
                )
                after_delta = after["nll_from_fifo_boundary"] - before["nll_from_fifo_boundary"]
                text += (
                    f"- {job['role']}: append minus FIFO total NLL "
                    f"{after['nll_sum'] - before['nll_sum']:+.9f} nats; before prediction 258 "
                    f"{before_delta:+.9f}; "
                    f"from prediction 258 {after_delta:+.9f}. "
                    "Negative means retaining history helped on this fixed teacher trajectory.\n"
                )
        text += (
            "\nThese are token-level conditional-loss effects on one consumed input, not "
            "benchmark-quality estimates or statistical superiority evidence. The saved free "
            "greedy answers diverged before history replacement (mixed-prior EviCastKV at "
            "prediction 23); this counterfactual cannot explain or repair that early divergence. "
            "No fresh cohort, calibration, or larger campaign is authorized.\n"
        )
    (out / "report.md").write_text(text + "\n```json\n" + json.dumps(gate, indent=2) + "\n```\n")
    write_once(
        out / "manifest.json",
        dict(
            source_sha=plan["source_sha"],
            plan_sha256=expected,
            inputs={str(p): sha(p) for p in (root / "workers").rglob("*.json")},
            outputs={p.name: sha(p) for p in out.iterdir()},
            measurement_type="derived_consumed_diagnostic",
        ),
    )
    write_once(
        root / "recursive_decision.json",
        dict(
            decision="stop_after_same_input_counterfactual",
            no_automatic_gpu_followup=True,
            reason="one teacher trajectory cannot authorize a new candidate or quality campaign",
            fresh_evaluation_authorized=False,
            paper_campaign_authorized=False,
        ),
    )
    print(json.dumps(gate))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare", "submit", "worker", "analyze"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--plan-sha")
    p.add_argument("--index", type=int)
    args = p.parse_args()
    if args.mode == "prepare":
        print(prepare(args.root))
    elif args.mode == "submit":
        launch.submit(
            args.root,
            args.plan_sha,
            script_name="evicastkv_append_counterfactual.sbatch",
            worker_minutes=20,
            memory_gb=16,
        )
    elif args.mode == "worker":
        worker(args.root, args.plan_sha, args.index)
    else:
        analyze(args.root, args.plan_sha)


if __name__ == "__main__":
    main()
