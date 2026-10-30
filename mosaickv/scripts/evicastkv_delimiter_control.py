#!/usr/bin/env python3
"""Same-byte delimiter restore, null repack and anchor on consumed inputs only."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import evicastkv_early_prefix as prior
import evicastkv_lifetime_history as common
import evicastkv_lifetime_v2 as launch
from evicastkv_campaign import environment

from mosaickv.config import DraftAllocationGuardConfig, RunConfig, config_sha256
from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
PARENT = Path("/scratch/djy8hg/runs/evicastkv/early-prefix-consumed-20261029-v1-retry1")
PARENT_SHA = "ad153ffe6e625c991e9acde8f46c27b168cc8d05bc8e7cafd4324fde9cb4e489"
MODES = ("anchor", "null_repack", "restore_delimiter")


def prepare(root):
    source = clean_source(SOURCE)
    if root.exists() or sha(PARENT / "plan.json") != PARENT_SHA:
        raise ValueError("new immutable root and pinned consumed capture required")
    gate = common.read(PARENT / "analysis/GATE.json")
    if (
        gate["status"] != "complete_consumed_diagnostic"
        or gate["failures"]
        or gate["completed_roles"] != 6
    ):
        raise ValueError("complete read-only controls required before causal intervention")
    prior.prepare(root / "inputs")
    plan = common.read(root / "inputs/plan.json")
    bound = dict(plan["bound_files"])
    for p in (root / "inputs/plan.json", PARENT / "plan.json", PARENT / "analysis/GATE.json"):
        bound[str(p)] = sha(p)
    for job in plan["jobs"]:
        folder = PARENT / f"workers/{job['index']}"
        result = common.read(folder / "result.json")
        if result["status"] != "completed" or result["failures"]:
            raise ValueError("original capture incomplete")
        for name, digest in result["outputs"].items():
            if sha(folder / name) != digest:
                raise ValueError("original evidence changed")
            bound[str(folder / name)] = digest
        bound[str(folder / "result.json")] = sha(folder / "result.json")
        job["parent_worker"] = str(folder)
        job["arms"] = job["arms"][1:]
        for arm in job["arms"]:
            cfg = RunConfig.from_mapping(arm["config"])
            if (
                cfg.method.value not in {"prompt_attention_topk", "mosaickv_exact"}
                or cfg.cache.physical_budget_mandatory_policy != "method"
            ):
                raise ValueError(
                    "only the frozen last-token mandatory exact configurations are audited"
                )
    plan.update(
        source_sha=source,
        bound_files=bound,
        protocol="consumed_same_byte_complete_delimiter_control_v1",
        parent_plan_sha256=PARENT_SHA,
        modes=MODES,
        prompt_only_donor_window=16,
        donor_score="mean_raw_prompt_attention_times_source_value_L2_norm",
        delimiter_rule="shortest_exact_tokenized_ASSISTANT_colon_suffix_at_most_16_tokens",
        guard=asdict(DraftAllocationGuardConfig()),
        maximum_reference_kl=0.02,
        maximum_logit_difference=1.0,
        meaningful_improvement="changed_positions_and_every_prefix_guard_and_absolute_limits_and_strict_KL_gain_or_recovered_token",
        production_selector_changes_authorized=False,
        evaluation_only_delimiter_intervention_authorized=True,
        selector_changes_authorized=False,
        fresh_evaluation_authorized=False,
        calibration_authorized=False,
        paper_campaign_authorized=False,
        no_prior_or_budget_sweep=True,
        automatic_promotion=False,
    )
    write_once(root / "plan.json", plan)
    return sha(root / "plan.json")


def worker(root, expected, index):
    import torch

    import mosaickv.backends.hf_runtime as runtime_module
    from mosaickv.adapters.huggingface import load_hf_adapter
    from mosaickv.adapters.huggingface.parity import (
        compare_cache_reinjection,
        compare_with_generate,
    )
    from mosaickv.evaluation.delimiter_control import DelimiterTransaction
    from mosaickv.evaluation.early_prefix import EarlyPrefixCapture
    from mosaickv.evaluation.model import EvaluationRequest
    from mosaickv.evaluation.worker_support import settle
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    plan = common.inputs(root, expected)
    job = plan["jobs"][index]
    out = root / f"workers/{index}"
    if out.exists():
        raise FileExistsError("write-once worker required")
    out.mkdir(parents=True)
    started = time.monotonic()
    result = dict(
        status="failed",
        failures=[],
        source_sha=plan["source_sha"],
        plan_sha256=expected,
        task=job["task"],
        sample_id=job["sample_id"],
    )
    saved, rows, cap, tx = {}, {}, None, None
    try:
        if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name(0):
            raise ValueError("one A100 required")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ValueError("deterministic workspace required")
        torch.set_num_threads(12)
        torch.use_deterministic_algorithms(True)
        payload = common.read(job["input"])
        messages = prior.redacted_messages(payload)
        targets = tuple(payload["fullkv_prefix_tokens"])
        if len(targets) != 32:
            raise ValueError("unchanged complete 32-token FullKV prefix required")
        cfg = RunConfig.from_mapping(job["arms"][0]["config"])
        torch.manual_seed(cfg.execution.seed)
        adapter = load_hf_adapter(
            cfg.model.id,
            revision=cfg.model.revision,
            model_kwargs=_model_kwargs(cfg, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        parent = torch.load(
            Path(job["parent_worker"]) / "captures.pt", map_location="cpu", weights_only=True
        )
        write_once(
            out / "manifest.json",
            dict(
                source_sha=plan["source_sha"],
                source_dirty=False,
                plan_sha256=expected,
                model=asdict(cfg.model),
                dataset=asdict(cfg.dataset),
                execution=asdict(cfg.execution),
                arms=job["arms"],
                modes=plan["modes"],
                guard=plan["guard"],
                environment=environment(),
                cuda_runtime=torch.version.cuda,
                gpu=asdict(capture_gpu_environment(torch)),
                slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                measurement_type="evaluation_only_same_byte_causal_delimiter_control",
                result_category="unified_comparison",
                result_origin="locally_measured",
                study_track=plan["study_track"],
                parent_capture_sha256=sha(Path(job["parent_worker"]) / "captures.pt"),
                environment_lock_sha256=sha(SOURCE / "mosaickv/env/common/requirements.lock"),
                evaluator_commit=plan["source_sha"],
                baseline_commit=plan["source_sha"],
                annotation_targets_used=False,
                decode_timing_not_performance_evidence=True,
                no_production_selector_change=True,
            ),
        )
        parity = {}
        for name, fn in (
            ("generate", compare_with_generate),
            ("reinjection", compare_cache_reinjection),
        ):
            report = fn(adapter, adapter.prepare_inputs(messages), max_new_tokens=16)
            parity[name] = asdict(report)
            if report.token_agreement != 1 or report.maximum_logit_difference > 0.001:
                raise ValueError("FullKV control failed")
        write_once(out / "parity.json", parity)
        for arm in job["arms"]:
            role = arm["role"]
            c = RunConfig.from_mapping(arm["config"])
            saved[role] = {}
            rows[role] = {}
            for mode in MODES:
                folder = out / "traces" / role / mode
                runtime = runtime_module.HuggingFaceMosaicKVModel(
                    adapter, c, trace_directory=folder
                )
                cap = EarlyPrefixCapture(
                    adapter, runtime_module, targets, role=f"{role}-{mode}", observed=False
                )
                tx = DelimiterTransaction(
                    adapter,
                    runtime_module,
                    mode=mode,
                    prompt_window=16,
                    mandatory_policy=c.cache.physical_budget_mandatory_policy,
                )
                # The intervention wraps the capture's pack hook, so the
                # captured positions describe the cache actually injected.
                with (
                    patch.object(runtime, "_teacher_forced_target_ids", return_value=targets),
                    cap,
                    tx,
                ):
                    runtime.teacher_forced_nll(
                        EvaluationRequest(
                            f"delimiter-{index}-{role}-{mode}",
                            job["sample_id"],
                            job["task"],
                            messages,
                            {},
                        ),
                        "__saved_FullKV_model_tokens_not_annotation_targets__",
                    )
                cap.validate_complete()
                if tx.record is None:
                    raise ValueError("prompt transaction did not execute")
                paths = list(folder.rglob("*.json"))
                if len(paths) != 1:
                    raise ValueError("one trace per causal replay required")
                trace = common.read(paths[0])
                logits = torch.stack(cap.logits)
                if mode == "anchor":
                    if (
                        not torch.equal(logits, parent["logits"][role])
                        or cap.positions != parent["retained_positions"][role]
                    ):
                        raise ValueError("unmodified replay changed prior logits or selections")
                    baseline = (logits, cap.positions, trace["cache_lifecycle"])
                else:
                    if trace["cache_lifecycle"] != baseline[2]:
                        raise ValueError("same-byte control changed initial/final/peak storage")
                    if mode == "null_repack" and (
                        not torch.equal(logits, baseline[0]) or cap.positions != baseline[1]
                    ):
                        raise ValueError("null repacking changed logits or positions")
                    if mode == "restore_delimiter" and any(
                        not set(tx.delimiter).issubset(head)
                        for layer in cap.positions
                        for head in layer
                    ):
                        raise ValueError("injected cache does not contain complete delimiter")
                    if mode == "restore_delimiter" and any(
                        tx.record[key] != rows[role]["null_repack"]["transaction"][key]
                        for key in (
                            "source_KV_sha256",
                            "before_positions_sha256",
                            "delimiter_positions",
                        )
                    ):
                        raise ValueError(
                            "restoration and null control have different source or anchor"
                        )
                saved[role][mode] = dict(logits=logits, positions=cap.positions)
                rows[role][mode] = dict(
                    config_sha=config_sha256(c),
                    transaction=tx.record,
                    storage=trace["cache_lifecycle"],
                    trace_path=str(paths[0]),
                    trace_sha256=sha(paths[0]),
                    trace_selection_metadata_role="original_anchor_only_see_transaction_for_actual_selection",
                )
                del runtime
                settle(adapter)
        torch.save(saved, out / "captures.pt")
        write_once(out / "transactions.json", rows)
        result.update(
            status="completed",
            completed_replays=6,
            outputs={p.name: sha(p) for p in out.iterdir() if p.is_file()},
            paper_campaign_authorized=False,
            fresh_evaluation_authorized=False,
        )
    except Exception as exc:
        result["failures"].append(dict(error=f"{type(exc).__name__}: {exc}"))
        torch.save(
            dict(
                completed=saved,
                current_logits=[] if cap is None else cap.logits,
                current_transaction=None if tx is None else tx.record,
            ),
            out / "partial_capture.pt",
        )
        raise
    finally:
        result["wall_seconds"] = time.monotonic() - started
        write_once(out / "result.json", result)


def analyze(root, expected):
    import pandas as pd
    import torch

    from mosaickv.backends.hf_allocation_guard import decide_draft_stability
    from mosaickv.evaluation.delimiter_control import compare_prefix_logits

    plan = common.inputs(root, expected)
    policy = DraftAllocationGuardConfig.from_mapping(plan["guard"])
    out = root / "analysis"
    if out.exists():
        raise FileExistsError("write-once analysis required")
    out.mkdir()
    failures, summaries, per_prefix, guards = [], [], [], []
    for job in plan["jobs"]:
        folder = root / f"workers/{job['index']}"
        if not (folder / "result.json").exists():
            failures.append(dict(index=job["index"], error="missing worker"))
            continue
        result = common.read(folder / "result.json")
        if result["status"] != "completed" or result["failures"]:
            failures.append(dict(index=job["index"], errors=result["failures"]))
            continue
        if result["source_sha"] != plan["source_sha"] or result["plan_sha256"] != expected:
            raise ValueError("worker provenance mismatch")
        for name, digest in result["outputs"].items():
            if sha(folder / name) != digest:
                raise ValueError("worker capture changed")
        captured = torch.load(folder / "captures.pt", weights_only=True, map_location="cpu")
        original = torch.load(
            Path(job["parent_worker"]) / "captures.pt", weights_only=True, map_location="cpu"
        )
        transactions = common.read(folder / "transactions.json")
        for role, modes in captured.items():
            records = compare_prefix_logits(
                original["logits"]["full_kv_reference"],
                modes["anchor"]["logits"],
                modes["restore_delimiter"]["logits"],
            )
            verdict = decide_draft_stability(records, policy)
            changed = transactions[role]["restore_delimiter"]["transaction"]["changed_routes"]
            safe = all(
                r["candidate"]["logit_kl"] <= plan["maximum_reference_kl"]
                and r["candidate"]["maximum_logit_difference"] <= plan["maximum_logit_difference"]
                for r in records
            )
            gains = sum(
                r["candidate"]["logit_kl"]
                < r["anchor"]["logit_kl"]
                - policy.absolute_tolerance
                - policy.relative_tolerance * abs(r["anchor"]["logit_kl"])
                for r in records
            )
            recovered = sum(
                not r["anchor"]["greedy_token_agreement"]
                and r["candidate"]["greedy_token_agreement"]
                for r in records
            )
            row = dict(
                task=job["task"],
                role=role,
                changed_routes=changed,
                restored_bytes=transactions[role]["restore_delimiter"]["transaction"][
                    "total_replaced_bytes"
                ],
                anchor_mean_kl=sum(r["anchor"]["logit_kl"] for r in records) / 32,
                restored_mean_kl=sum(r["candidate"]["logit_kl"] for r in records) / 32,
                anchor_token_agreement=sum(r["anchor"]["greedy_token_agreement"] for r in records)
                / 32,
                restored_token_agreement=sum(
                    r["candidate"]["greedy_token_agreement"] for r in records
                )
                / 32,
                recovered_predictions=recovered,
                newly_divergent_predictions=sum(
                    r["anchor"]["greedy_token_agreement"]
                    and not r["candidate"]["greedy_token_agreement"]
                    for r in records
                ),
                first_failed_prefix=verdict["rejected_steps"][0]["step"]
                if verdict["rejected_steps"]
                else None,
                strict_kl_improvement_prefixes=gains,
                guard_passed=verdict["accepted"],
                absolute_limits_passed=safe,
                meaningful_safe_improvement=bool(
                    changed and verdict["accepted"] and safe and (gains or recovered)
                ),
                storage_identical=True,
                **{
                    key: transactions[role]["restore_delimiter"]["storage"][key]
                    for key in common.STORAGE
                },
            )
            summaries.append(row)
            per_prefix.extend(dict(task=job["task"], role=role, **r) for r in records)
            guards.append(dict(task=job["task"], role=role, **verdict))
    write_once(out / "prefix_metrics.json", dict(records=per_prefix))
    write_once(out / "guards.json", dict(records=guards))
    pd.DataFrame(summaries).to_parquet(out / "summary.parquet", index=False)
    gate = dict(
        status="complete_consumed_diagnostic" if len(summaries) == 4 and not failures else "failed",
        failures=failures,
        completed_comparisons=len(summaries),
        checks=summaries,
        meaningful_safe_improvement_all_cases=bool(
            len(summaries) == 4 and all(r["meaningful_safe_improvement"] for r in summaries)
        ),
        superiority_established=False,
        production_selector_modified=False,
        fresh_evaluation_authorized=False,
        calibration_authorized=False,
        paper_campaign_authorized=False,
    )
    write_once(out / "GATE.json", gate)
    text = (
        "# Same-byte complete generation-delimiter causal control\n\n"
        "Consumed Description and MM-Vet inputs only; not benchmark-quality confirmation. "
        "FullKV-prefix replay is 32 tokens with unchanged BF16 eager arithmetic and argmax. "
        "Top-K and EviCastKV receive the same intervention and prompt-only donor rule. "
        "FullKV and null/replay controls are references, not latency competitors. "
        "The original selector trace describes its anchor; transactions.json and captures.pt "
        "record the actual injected selection. No production selector change, prior sweep "
        "or fresh cohort was made.\n\n"
    )
    if summaries:
        display = pd.DataFrame(summaries)
        display["first_failed_prefix"] = display["first_failed_prefix"].fillna("none")
        text += display.to_markdown(index=False) + "\n\n"
    text += (
        "Per-prefix non-regression uses the existing 32-token guard (absolute tolerance 1e-7, "
        "relative tolerance 1e-6), including later prefixes. Means are descriptive and cannot "
        "override a failed prefix. FullKV ties are recorded, never used to waive token agreement. "
        "No latency claim is made: extra capture, source hashing and repacking contaminate timing. "
        "Storage is post-compression active decode backing, not whole-request KV peak.\n\n"
    )
    (out / "report.md").write_text(text + "```json\n" + json.dumps(gate, indent=2) + "\n```\n")
    write_once(
        out / "manifest.json",
        dict(
            source_sha=plan["source_sha"],
            plan_sha256=expected,
            inputs={str(p): sha(p) for p in (root / "workers").rglob("result.json")},
            outputs={p.name: sha(p) for p in out.iterdir()},
        ),
    )
    write_once(
        root / "recursive_decision.json",
        dict(
            decision="review_consumed_causal_control_only",
            automatic_selector_promotion=False,
            larger_campaign_submitted=False,
            meaningful_safe_improvement_all_cases=gate["meaningful_safe_improvement_all_cases"],
        ),
    )
    print(json.dumps(gate))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "submit", "worker", "analyze"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--plan-sha")
    parser.add_argument("--index", type=int)
    a = parser.parse_args()
    if a.mode == "prepare":
        print(prepare(a.root))
    elif a.mode == "submit":
        launch.submit(
            a.root,
            a.plan_sha,
            script_name="evicastkv_delimiter_control.sbatch",
            worker_minutes=20,
            memory_gb=16,
        )
    elif a.mode == "worker":
        worker(a.root, a.plan_sha, a.index)
    else:
        analyze(a.root, a.plan_sha)


if __name__ == "__main__":
    main()
