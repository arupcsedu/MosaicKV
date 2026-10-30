#!/usr/bin/env python3
"""Frozen donor-only attention ablation on two already-consumed inputs."""

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
PARENT = Path("/scratch/djy8hg/runs/evicastkv/delimiter-control-consumed-20261029-v1")
PARENT_SHA = "fdb29db8bb61ee49b97b52762d33a59ed1b36b23bb348fbe52352c1a69ac89e0"
MODES = ("zero_mask", "donor_only")


def prepare(root):
    source = clean_source(SOURCE)
    if root.exists() or sha(PARENT / "plan.json") != PARENT_SHA:
        raise ValueError("new immutable output root and pinned delimiter study required")
    plan = common.read(PARENT / "plan.json")
    gate = common.read(PARENT / "analysis/GATE.json")
    if (
        gate["status"] != "complete_consumed_diagnostic"
        or gate["failures"]
        or gate["completed_comparisons"] != 4
    ):
        raise ValueError("complete previous causal control required")
    bound = dict(plan["bound_files"])
    for path in (PARENT / "plan.json", PARENT / "analysis/GATE.json"):
        bound[str(path)] = sha(path)
    for job in plan["jobs"]:
        folder = PARENT / f"workers/{job['index']}"
        result = common.read(folder / "result.json")
        if result["status"] != "completed" or result["failures"]:
            raise ValueError("incomplete parent worker")
        for name, digest in result["outputs"].items():
            path = folder / name
            if sha(path) != digest:
                raise ValueError("parent evidence changed")
            bound[str(path)] = digest
        bound[str(folder / "result.json")] = sha(folder / "result.json")
        job["delimiter_worker"] = str(folder)
        if len(job["arms"]) != 2:
            raise ValueError("same two frozen compressed roles required")
    plan.update(
        source_sha=source,
        bound_files=bound,
        protocol="consumed_frozen_donor_attention_availability_isolation_v1",
        parent_plan_sha256=PARENT_SHA,
        modes=MODES,
        donor_selection="reuse_saved_removed_positions_verbatim_no_reranking",
        intervention="per_head_negative_infinity_attention_mask_only_no_KV_removal_no_restore",
        interpretation="causal_attention_access_ablation_NOT_physical_eviction_or_memory_savings",
        mandatory_policy_unchanged=True,
        delimiter_reservation_unchanged=True,
        guard_unchanged=True,
        automatic_promotion=False,
        evaluation_only_delimiter_intervention_authorized=False,
        evaluation_only_frozen_donor_mask_authorized=True,
        fresh_evaluation_authorized=False,
        calibration_authorized=False,
        paper_campaign_authorized=False,
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
    from mosaickv.evaluation.donor_isolation import DonorMaskControl
    from mosaickv.evaluation.early_prefix import EarlyPrefixCapture
    from mosaickv.evaluation.model import EvaluationRequest
    from mosaickv.evaluation.worker_support import settle
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    plan = common.inputs(root, expected)
    job = plan["jobs"][index]
    out = root / f"workers/{index}"
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = dict(
        status="failed",
        failures=[],
        source_sha=plan["source_sha"],
        plan_sha256=expected,
        task=job["task"],
        sample_id=job["sample_id"],
    )
    saved, rows, cap, control = {}, {}, None, None
    try:
        if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name(0):
            raise ValueError("one A100 required")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ValueError("registered deterministic workspace required")
        torch.set_num_threads(12)
        torch.use_deterministic_algorithms(True)
        payload = common.read(job["input"])
        messages = prior.redacted_messages(payload)
        targets = tuple(payload["fullkv_prefix_tokens"])
        if len(targets) != 32:
            raise ValueError("unchanged 32-token model-generated trajectory required")
        cfg = RunConfig.from_mapping(job["arms"][0]["config"])
        torch.manual_seed(cfg.execution.seed)
        adapter = load_hf_adapter(
            cfg.model.id,
            revision=cfg.model.revision,
            model_kwargs=_model_kwargs(cfg, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        parent_folder = Path(job["delimiter_worker"])
        parent = torch.load(parent_folder / "captures.pt", map_location="cpu", weights_only=True)
        transactions = common.read(parent_folder / "transactions.json")
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
                modes=MODES,
                guard=plan["guard"],
                environment=environment(),
                cuda_runtime=torch.version.cuda,
                gpu=asdict(capture_gpu_environment(torch)),
                slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                measurement_type="evaluation_only_frozen_donor_attention_availability_ablation",
                result_category="unified_comparison",
                result_origin="locally_measured",
                study_track=plan["study_track"],
                evaluator_commit=plan["source_sha"],
                baseline_commit=plan["source_sha"],
                annotation_targets_used=False,
                parent_capture_sha256=sha(parent_folder / "captures.pt"),
                parent_transactions_sha256=sha(parent_folder / "transactions.json"),
                environment_lock_sha256=sha(SOURCE / "mosaickv/env/common/requirements.lock"),
                interpretation=plan["interpretation"],
                decode_timing_not_performance_evidence=True,
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
                raise ValueError("FullKV correctness control failed")
        write_once(out / "parity.json", parity)
        for arm in job["arms"]:
            role = arm["role"]
            c = RunConfig.from_mapping(arm["config"])
            frozen = transactions[role]["restore_delimiter"]["transaction"]
            saved[role], rows[role] = {}, {}
            for mode in MODES:
                folder = out / "traces" / role / mode
                runtime = runtime_module.HuggingFaceMosaicKVModel(
                    adapter, c, trace_directory=folder
                )
                cap = EarlyPrefixCapture(
                    adapter, runtime_module, targets, role=role, observed=False
                )
                tx = DelimiterTransaction(adapter, runtime_module, mode="anchor")
                control = DonorMaskControl(adapter, runtime_module, frozen, mode=mode)
                with (
                    patch.object(runtime, "_teacher_forced_target_ids", return_value=targets),
                    cap,
                    tx,
                    control,
                ):
                    runtime.teacher_forced_nll(
                        EvaluationRequest(
                            f"donor-{index}-{role}-{mode}",
                            job["sample_id"],
                            job["task"],
                            messages,
                            {},
                        ),
                        "__saved_FullKV_tokens_not_annotation_answers__",
                    )
                cap.validate_complete()
                control.validate_complete()
                logits = torch.stack(cap.logits)
                paths = list(folder.rglob("*.json"))
                if len(paths) != 1:
                    raise ValueError("exactly one trace per replay required")
                storage = common.read(paths[0])["cache_lifecycle"]
                if (
                    cap.positions != parent[role]["anchor"]["positions"]
                    or storage != transactions[role]["anchor"]["storage"]
                ):
                    raise ValueError("donor mask changed selected positions or physical storage")
                if mode == "zero_mask" and not torch.equal(
                    logits, parent[role]["anchor"]["logits"]
                ):
                    raise ValueError("zero-mask/capture control changed saved anchor logits")
                if not torch.equal(logits[0], parent[role]["anchor"]["logits"][0]):
                    raise ValueError("prefill prediction changed before donor masking")
                saved[role][mode] = dict(logits=logits, positions=cap.positions)
                rows[role][mode] = dict(
                    config_sha=config_sha256(c),
                    storage=storage,
                    intervention=control.record(),
                    trace_path=str(paths[0]),
                    trace_sha256=sha(paths[0]),
                    elapsed_seconds=time.monotonic() - started,
                )
                write_once(out / f"{role}-{mode}.json", rows[role][mode])
                del runtime
                settle(adapter)
        torch.save(saved, out / "captures.pt")
        write_once(out / "controls.json", rows)
        result.update(
            status="completed",
            completed_replays=4,
            outputs={p.name: sha(p) for p in out.iterdir() if p.is_file()},
        )
    except Exception as exc:
        result["failures"].append(dict(error=f"{type(exc).__name__}: {exc}"))
        torch.save(
            dict(
                completed=saved,
                current_logits=[] if cap is None else cap.logits,
                current_control=None if control is None else control.record(),
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
    out.mkdir(exist_ok=False)
    failures, summaries, prefixes, verdicts = [], [], [], []
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
        for name, digest in result["outputs"].items():
            if sha(folder / name) != digest:
                raise ValueError("worker evidence changed")
        new = torch.load(folder / "captures.pt", map_location="cpu", weights_only=True)
        parent = torch.load(
            Path(job["delimiter_worker"]) / "captures.pt", map_location="cpu", weights_only=True
        )
        full = torch.load(
            Path(job["parent_worker"]) / "captures.pt", map_location="cpu", weights_only=True
        )["logits"]["full_kv_reference"]
        controls = common.read(folder / "controls.json")
        for role, modes in new.items():
            for mode in MODES:
                record = controls[role][mode]
                trace_path = Path(record["trace_path"])
                intervention = record["intervention"]
                if (
                    sha(trace_path) != record["trace_sha256"]
                    or common.read(trace_path)["cache_lifecycle"] != record["storage"]
                    or [r["prediction"] for r in intervention["attention"]] != list(range(2, 33))
                    or intervention["physical_KV_bytes_saved"] != 0
                    or not intervention["availability_only_not_physical_eviction"]
                    or modes[mode]["positions"] != parent[role]["anchor"]["positions"]
                ):
                    raise ValueError("incomplete mask, trace, position or storage control")
                if mode == "donor_only" and any(
                    r["maximum_donor_attention_probability"] != 0 for r in intervention["attention"]
                ):
                    raise ValueError("saved donor mask has nonzero attention")
            if controls[role]["zero_mask"]["storage"] != controls[role]["donor_only"]["storage"]:
                raise ValueError("mask intervention changed physical storage")
            anchor = modes["zero_mask"]["logits"]
            donor = modes["donor_only"]["logits"]
            if not torch.equal(anchor, parent[role]["anchor"]["logits"]):
                raise ValueError("saved zero control no longer matches")
            rows = compare_prefix_logits(full, anchor, donor)
            guard = decide_draft_stability(rows, policy)
            joint = compare_prefix_logits(full, anchor, parent[role]["restore_delimiter"]["logits"])
            # The second effect is restoration CONDITIONAL on donors being
            # unavailable. It is not an additive attribution or a restore-only test.
            row = dict(
                task=job["task"],
                role=role,
                anchor_mean_kl=sum(r["anchor"]["logit_kl"] for r in rows) / 32,
                donor_only_mean_kl=sum(r["candidate"]["logit_kl"] for r in rows) / 32,
                joint_restore_donor_mean_kl=sum(r["candidate"]["logit_kl"] for r in joint) / 32,
                anchor_tokens=sum(r["anchor"]["greedy_token_agreement"] for r in rows),
                donor_only_tokens=sum(r["candidate"]["greedy_token_agreement"] for r in rows),
                joint_tokens=sum(r["candidate"]["greedy_token_agreement"] for r in joint),
                donor_only_guard_passed=guard["accepted"],
                donor_only_maximum_kl=max(r["candidate"]["logit_kl"] for r in rows),
                donor_only_maximum_logit_difference=max(
                    r["candidate"]["maximum_logit_difference"] for r in rows
                ),
                donor_only_absolute_limits_passed=all(
                    r["candidate"]["logit_kl"] <= plan["maximum_reference_kl"]
                    and r["candidate"]["maximum_logit_difference"]
                    <= plan["maximum_logit_difference"]
                    for r in rows
                ),
                donor_only_first_failed_prefix=guard["rejected_steps"][0]["step"]
                if guard["rejected_steps"]
                else None,
                donor_worsened_kl_prefixes=sum(
                    r["candidate"]["logit_kl"]
                    > r["anchor"]["logit_kl"]
                    + policy.absolute_tolerance
                    + policy.relative_tolerance * abs(r["anchor"]["logit_kl"])
                    for r in rows
                ),
                donor_recovered_tokens=sum(
                    not r["anchor"]["greedy_token_agreement"]
                    and r["candidate"]["greedy_token_agreement"]
                    for r in rows
                ),
                donor_newly_divergent_tokens=sum(
                    r["anchor"]["greedy_token_agreement"]
                    and not r["candidate"]["greedy_token_agreement"]
                    for r in rows
                ),
                masked_donor_payload_bytes=controls[role]["donor_only"]["intervention"][
                    "frozen_masked_donor_payload_bytes"
                ],
                physical_KV_bytes_saved=0,
                additional_mask_backing_bytes=controls[role]["donor_only"]["intervention"][
                    "additional_mask_backing_bytes"
                ],
                zero_mask_bitwise_parity=True,
                **{k: controls[role]["donor_only"]["storage"][k] for k in common.STORAGE},
            )
            row["mean_donor_access_removal_kl_change"] = (
                row["donor_only_mean_kl"] - row["anchor_mean_kl"]
            )
            row["mean_restore_conditional_on_donor_removal_kl_change"] = (
                row["joint_restore_donor_mean_kl"] - row["donor_only_mean_kl"]
            )
            summaries.append(row)
            for r, j in zip(rows, joint, strict=True):
                prefixes.append(
                    dict(task=job["task"], role=role, **r, joint_restore_donor=j["candidate"])
                )
            verdicts.append(dict(task=job["task"], role=role, **guard))
    pd.DataFrame(summaries).to_parquet(out / "summary.parquet", index=False)
    write_once(out / "prefix_metrics.json", dict(records=prefixes))
    write_once(out / "guards.json", dict(records=verdicts))
    gate = dict(
        status="complete_consumed_diagnostic" if len(summaries) == 4 and not failures else "failed",
        failures=failures,
        completed_comparisons=len(summaries),
        checks=summaries,
        superiority_established=False,
        fresh_evaluation_authorized=False,
        calibration_authorized=False,
        paper_campaign_authorized=False,
        production_selector_modified=False,
        delimiter_reservation_increased=False,
        guard_weakened=False,
        physical_eviction_equivalence_claimed=False,
    )
    write_once(out / "GATE.json", gate)
    text = (
        "# Frozen donor-removal availability diagnostic\n\n"
        "Same two consumed inputs and saved donor/restore sets; no donor reranking, new "
        "reservation, threshold change, or annotation use. Zero-mask controls must reproduce "
        "saved anchor logits bitwise; donor attention must be exactly zero. Masks block access "
        "but retain donor tensors physically: **zero KV bytes saved**. Extra mask backing is "
        "reported separately; this is neither a compressed operating point nor latency evidence. "
        "The 32-token per-prefix guard and absolute KL/logit limits are unchanged.\n\n"
    )
    if summaries:
        text += pd.DataFrame(summaries).to_markdown(index=False) + "\n\n"
    text += (
        "`donor_only - anchor` isolates loss of donor attention access in the same physical "
        "layout. `joint - donor_only` describes restoration conditional on that loss, using "
        "the prior physically repacked joint intervention. These nonlinear autoregressive "
        "effects are not an additive attribution, and mask/repacking layout differences remain "
        "a limitation of the second comparison. Every mean is descriptive: a failed prefix "
        "cannot be overridden by an improved average. FullKV argmax ties do not waive the guard. "
        "Post-compression active decode backing is not a whole-request/HBM peak. "
        "No benchmark quality or fresh-confirmation claim is established.\n"
    )
    (out / "report.md").write_text(text)
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
            decision="stop_after_bounded_consumed_donor_isolation_review",
            larger_campaign_submitted=False,
            selector_changed=False,
            guard_weakened=False,
        ),
    )
    print(json.dumps(gate))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "submit", "worker", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha")
    parser.add_argument("--index", type=int)
    a = parser.parse_args()
    if a.mode == "prepare":
        print(prepare(a.root))
    elif a.mode == "submit":
        launch.submit(
            a.root,
            a.plan_sha,
            script_name="evicastkv_donor_isolation.sbatch",
            worker_minutes=20,
            memory_gb=16,
        )
    elif a.mode == "worker":
        worker(a.root, a.plan_sha, a.index)
    else:
        analyze(a.root, a.plan_sha)


if __name__ == "__main__":
    main()
