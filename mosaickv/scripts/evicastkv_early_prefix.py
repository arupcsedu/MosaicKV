#!/usr/bin/env python3
"""Consumed-input early prompt-loss audit; separate from Fox confirmation."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import evicastkv_layerwise_prior as previous
import evicastkv_lifetime_history as common
import evicastkv_lifetime_v2 as launch
import evicastkv_token_eviction_audit as parent_audit
from evicastkv_campaign import environment

from mosaickv.config import RunConfig, config_sha256
from mosaickv.evaluation.early_prefix import (
    EarlyPrefixCapture,
    compare_layer_outputs,
    position_metadata,
)
from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
PARENT = Path("/scratch/djy8hg/runs/evicastkv/token-eviction-consumed-20261029-v1")
PARENT_SHA = "c74c12fb58ecf38cad5d6a2cec7dfe70e13e16d4327c5662aca5a3516fe24c6a"
HORIZON = 32


def prepare(root):
    source = clean_source(SOURCE)
    if root.exists() or sha(PARENT / "plan.json") != PARENT_SHA:
        raise ValueError("new immutable root and pinned consumed parent required")
    old = common.read(PARENT / "plan.json")
    bound, jobs = dict(old["bound_files"]), []
    bound[str(PARENT / "plan.json")] = PARENT_SHA
    for offset in (0, 5):
        arms = []
        full_row = common.read(old["jobs"][offset]["frozen_measurement_row"])
        full_trace = previous.phase_trace(full_row)
        targets = full_trace["generated_token_ids"][:HORIZON]
        if len(targets) != HORIZON:
            raise ValueError("complete saved FullKV greedy prefix required")
        for index in (offset, offset + 2, offset + 3):
            j = old["jobs"][index]
            row = parent_audit.bind_row(bound, Path(j["frozen_measurement_row"]))
            cfg = RunConfig.from_mapping(j["arms"][0]["config"])
            if cfg.prototypes.enabled or cfg.repair.enabled or cfg.forecasting.draft_steps:
                raise ValueError("frozen prompt-only exact configurations required")
            arms.append(
                dict(
                    role=j["diagnostic_role"],
                    config=j["arms"][0]["config"],
                    config_sha=config_sha256(cfg),
                    frozen_row=j["frozen_measurement_row"],
                    saved_free_greedy=previous.phase_trace(row)["generated_token_ids"][:HORIZON],
                )
            )
        baseline = RunConfig.from_mapping(arms[0]["config"])
        for arm in arms[1:]:
            cfg = RunConfig.from_mapping(arm["config"])
            if any(
                getattr(cfg, f) != getattr(baseline, f)
                for f in ("model", "dataset", "generation", "execution")
            ):
                raise ValueError("arithmetic/input configuration differs across roles")
        template = old["jobs"][offset]
        ledger = common.read(template["ledger"])
        sample = next(s for s in ledger["samples"] if str(s["sample_id"]) == template["sample_id"])
        # Only prompt/media/model-generated tokens go to the worker. No
        # annotation answers, references, or benchmark scores are copied.
        payload = dict(
            task=template["task"],
            sample_id=template["sample_id"],
            prompt=sample["prompt"],
            media=sample["media"],
            fullkv_prefix_tokens=targets,
        )
        path = root / f"inputs/{len(jobs)}.json"
        write_once(path, payload)
        bound[str(path)] = sha(path)
        jobs.append(
            dict(
                index=len(jobs),
                task=template["task"],
                sample_id=template["sample_id"],
                input=str(path),
                arms=arms,
            )
        )
    for path, digest in bound.items():
        if sha(Path(path)) != digest:
            raise ValueError("bound evidence changed")
    write_once(
        root / "plan.json",
        dict(
            source_sha=source,
            bound_files=bound,
            jobs=jobs,
            protocol="consumed_saved_FullKV_greedy_prefix_actual_layer_outputs_v1",
            trajectory="saved_FullKV_greedy_tokens_not_annotation_answers",
            horizon=HORIZON,
            study_track="PrefixKV_bounded_history_consumed_diagnostic_NOT_Fox_confirmation",
            fox_track="frozen_exact_prompt_selection_append_only_no_pooling_with_this_diagnostic",
            capture_predictions=[2, 4, 8, 16, 20, 21, 22, 23, 24, 32],
            decomposition_prediction=23,
            capture_control="bitwise_logits_selected_prompt_and_storage_vs_unobserved_replay",
            source_configs_unchanged=True,
            selector_changes_authorized=False,
            fresh_evaluation_authorized=False,
            calibration_authorized=False,
            paper_campaign_authorized=False,
            maximum_a100s=6,
            requested_workers=2,
            cpus_per_worker=12,
            wall_minutes=20,
            decomposition_is_not_causal_intervention_evidence=True,
        ),
    )
    return sha(root / "plan.json")


def worker(root, expected, index):
    import torch
    from PIL import Image

    import mosaickv.backends.hf_runtime as runtime_module
    from mosaickv.adapters.huggingface import load_hf_adapter
    from mosaickv.adapters.huggingface.parity import (
        compare_cache_reinjection,
        compare_with_generate,
    )
    from mosaickv.evaluation.messages import MediaItem, build_multimodal_messages
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
    captures, logits, positions, rows = {}, {}, {}, []
    capture = None
    try:
        if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name(0):
            raise ValueError("one A100 required")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ValueError("deterministic workspace required")
        torch.set_num_threads(12)
        torch.use_deterministic_algorithms(True)
        payload = common.read(job["input"])
        if set(payload) != {"task", "sample_id", "prompt", "media", "fullkv_prefix_tokens"}:
            raise ValueError("only label-redacted diagnostic input allowed")
        targets = tuple(payload["fullkv_prefix_tokens"])
        cfg = RunConfig.from_mapping(job["arms"][0]["config"])
        torch.manual_seed(cfg.execution.seed)
        adapter = load_hf_adapter(
            cfg.model.id,
            revision=cfg.model.revision,
            model_kwargs=_model_kwargs(cfg, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        media = []
        for m in payload["media"]:
            if m["kind"] != "image" or sha(Path(m["path"])) != m["sha256"]:
                raise ValueError("pinned image input required")
            with Image.open(m["path"]) as image:
                media.append(MediaItem("image", image.convert("RGB").copy()))
        messages = build_multimodal_messages(payload["prompt"], tuple(media))
        write_once(
            out / "manifest.json",
            dict(
                source_sha=plan["source_sha"],
                source_dirty=False,
                plan_sha256=expected,
                arms=job["arms"],
                model=asdict(cfg.model),
                dataset=asdict(cfg.dataset),
                execution=asdict(cfg.execution),
                environment=environment(),
                cuda_runtime=torch.version.cuda,
                gpu=asdict(capture_gpu_environment(torch)),
                slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                measurement_type="evaluation_only_early_prefix_layer_capture",
                result_origin="locally_measured",
                result_category="unified_comparison",
                study_track=plan["study_track"],
                environment_lock_sha256=sha(SOURCE / "mosaickv/env/common/requirements.lock"),
                input_sha256=sha(Path(job["input"])),
                evaluator_commit=plan["source_sha"],
                baseline_commit=plan["source_sha"],
                annotation_targets_used=False,
                timing_not_valid_for_latency_claims=True,
                fresh_evaluation_authorized=False,
            ),
        )
        controls = {}
        for name, fn in (
            ("generate", compare_with_generate),
            ("reinjection", compare_cache_reinjection),
        ):
            report = fn(adapter, adapter.prepare_inputs(messages), max_new_tokens=16)
            controls[name] = asdict(report)
            if report.token_agreement != 1 or report.maximum_logit_difference > 0.001:
                raise ValueError("FullKV numerical control failed")
        write_once(out / "parity.json", controls)
        # Compressed selections are known before FullKV's algebraic audit.
        for arm in (*job["arms"][1:], job["arms"][0]):
            role = arm["role"]
            c = RunConfig.from_mapping(arm["config"])
            old_trace = previous.phase_trace(common.read(arm["frozen_row"]))
            control = None
            for observed in (False, True):
                folder = out / "traces" / role / ("observed" if observed else "control")
                runtime = runtime_module.HuggingFaceMosaicKVModel(
                    adapter, c, trace_directory=folder
                )
                capture = EarlyPrefixCapture(
                    adapter,
                    runtime_module,
                    targets,
                    role=role,
                    observed=observed,
                    reference_positions=positions
                    if role == "full_kv_reference" and observed
                    else None,
                )
                with (
                    patch.object(runtime, "_teacher_forced_target_ids", return_value=targets),
                    capture,
                ):
                    runtime.teacher_forced_nll(
                        EvaluationRequest(
                            f"early-{index}-{role}-{observed}",
                            job["sample_id"],
                            job["task"],
                            messages,
                            {},
                        ),
                        "__saved_FullKV_model_tokens_not_annotation_targets__",
                    )
                capture.validate_complete()
                paths = list(folder.rglob("*.json"))
                if len(paths) != 1:
                    raise ValueError("one runtime trace per replay required")
                trace = common.read(paths[0])
                for key in (
                    "selected_blocks",
                    "packed_prefill_active_kv_bytes",
                    "packed_padding_kv_bytes",
                ):
                    if trace[key] != old_trace[key]:
                        raise ValueError(f"frozen prompt changed: {key}")
                x = torch.stack(capture.logits)
                if not observed:
                    control = (x, capture.positions, trace)
                    continue
                if (
                    not torch.equal(x, control[0])
                    or capture.positions != control[1]
                    or trace["cache_lifecycle"] != control[2]["cache_lifecycle"]
                ):
                    raise ValueError(
                        "read-only instrumentation changed logits, positions or storage"
                    )
                if role == "full_kv_reference" and tuple(x.argmax(-1).tolist()) != targets:
                    raise ValueError("FullKV replay differs from saved model-generated trajectory")
                greedy = tuple(x.argmax(-1).tolist())
                divergence = next(
                    (i + 1 for i, (a, b) in enumerate(zip(greedy, targets, strict=True)) if a != b),
                    None,
                )
                saved_divergence = next(
                    (
                        i + 1
                        for i, (a, b) in enumerate(
                            zip(arm["saved_free_greedy"], targets, strict=True)
                        )
                        if a != b
                    ),
                    None,
                )
                if divergence != saved_divergence:
                    raise ValueError("earliest divergence changed from saved free-generation run")
                captures[role] = capture.recorder.records[role]
                logits[role] = x
                if capture.positions is not None:
                    positions[role] = capture.positions
                if capture.decompositions:
                    torch.save(capture.decompositions, out / "omission_decompositions.pt")
                    write_once(
                        out / "source_modality.json",
                        dict(
                            spans=[asdict(s) for s in capture.modality_map],
                            source_length=capture.source_length,
                            source_contract=capture.recorder.source_contract,
                        ),
                    )
                rows.append(
                    dict(
                        role=role,
                        first_greedy_divergence=divergence,
                        instrumented_logits_bitwise_equal=True,
                        selected_prompt_unchanged=True,
                        initial_final_peak_storage=trace["cache_lifecycle"],
                        config_sha=arm["config_sha"],
                    )
                )
                del runtime, control
                settle(adapter)
        layer_rows = []
        for role in positions:
            layer_rows.extend(
                dict(role=role, **r)
                for r in compare_layer_outputs(captures["full_kv_reference"], captures[role])
            )
        torch.save(
            dict(actual_layer_outputs=captures, logits=logits, retained_positions=positions),
            out / "captures.pt",
        )
        write_once(out / "layer_errors.json", dict(records=layer_rows))
        write_once(out / "roles.json", dict(records=rows))
        result.update(
            status="completed",
            roles=rows,
            model_label_access=False,
            outputs={p.name: sha(p) for p in out.iterdir() if p.is_file()},
            fresh_evaluation_authorized=False,
            calibration_authorized=False,
            paper_campaign_authorized=False,
        )
    except Exception as exc:
        result["failures"].append(dict(error=f"{type(exc).__name__}: {exc}"))
        if capture is not None:
            torch.save(
                dict(
                    logits=capture.logits,
                    actual_layer_outputs=capture.recorder.records,
                    retained_positions=capture.positions,
                    completed_roles=rows,
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

    plan = common.inputs(root, expected)
    out = root / "analysis"
    if out.exists():
        raise FileExistsError("write-once analysis")
    out.mkdir()
    failures, layers, omitted, roles = [], [], [], []
    for j in plan["jobs"]:
        folder = root / f"workers/{j['index']}"
        if not (folder / "result.json").exists():
            failures.append(dict(index=j["index"], error="missing worker"))
            continue
        r = common.read(folder / "result.json")
        if r["status"] != "completed" or r["failures"]:
            failures.append(dict(index=j["index"], failures=r["failures"]))
            continue
        if r["source_sha"] != plan["source_sha"] or r["plan_sha256"] != expected:
            raise ValueError("worker provenance mismatch")
        for name, digest in r["outputs"].items():
            if sha(folder / name) != digest:
                raise ValueError("worker evidence changed")
        roles.extend(dict(task=j["task"], **row) for row in r["roles"])
        layers.extend(
            dict(task=j["task"], **row)
            for row in common.read(folder / "layer_errors.json")["records"]
        )
        data = torch.load(
            folder / "omission_decompositions.pt", map_location="cpu", weights_only=True
        )
        spans = common.read(folder / "source_modality.json")["spans"]
        for role, records in data.items():
            for layer, row in enumerate(records):
                signed = row["omitted_signed_output_contribution"]
                for head in range(signed.shape[0]):
                    available = (~row["retained_prompt_mask"][head]).nonzero().flatten().tolist()
                    chosen = sorted(available, key=lambda p: (-float(signed[head, p]), p))[:4]
                    for p in chosen:
                        omitted.append(
                            dict(
                                task=j["task"],
                                role=role,
                                prediction=23,
                                layer=layer,
                                query_head=head,
                                logical_position=p,
                                signed_output_contribution=float(signed[head, p]),
                                attention_mass=float(row["prompt_probabilities"][head, p]),
                                route_output_error=float(row["output_error_norm"][head]),
                                **position_metadata(p, spans),
                                interpretation=row["interpretation"],
                            )
                        )
    pd.DataFrame(layers).to_parquet(out / "actual_layer_errors.parquet", index=False)
    pd.DataFrame(omitted).to_parquet(out / "omitted_block_contributions.parquet", index=False)
    write_once(out / "roles.json", dict(records=roles))
    gate = dict(
        status="complete_consumed_diagnostic" if len(roles) == 6 and not failures else "failed",
        failures=failures,
        completed_roles=len(roles),
        tracks_separated=True,
        new_selector_or_quality_improvement_tested=False,
        superiority_established=False,
        fresh_evaluation_authorized=False,
        calibration_authorized=False,
        paper_campaign_authorized=False,
    )
    write_once(out / "GATE.json", gate)
    text = (
        "# Early prompt-loss diagnostic\n\n"
        "Two consumed PrefixKV-development inputs; NOT Prof. Fox confirmation. "
        "All arms replay the same saved 32 FullKV greedy tokens, not annotation answers. "
        "The bounded transport never evicts history during this short replay. "
        "Fox's exact-selection/append-only campaign remains a separate track.\n\n"
    )
    if roles:
        text += (
            pd.DataFrame(
                [
                    {
                        k: r[k]
                        for k in (
                            "task",
                            "role",
                            "first_greedy_divergence",
                            "instrumented_logits_bitwise_equal",
                            "selected_prompt_unchanged",
                        )
                    }
                    for r in roles
                ]
            ).to_markdown(index=False)
            + "\n\n"
        )
    if layers:
        df = pd.DataFrame(layers)
        focus = df[
            (df.prediction == 23)
            & df.field.isin(
                ["residual_input", "projected_attention", "post_attention_residual", "layer_output"]
            )
        ]
        text += "## Actual model differences at prediction 23\n\n"
        text += (
            focus.sort_values("l2_error", ascending=False)
            .groupby(["task", "role", "field"], sort=False)
            .head(1)
            .to_markdown(index=False)
            + "\n\n"
        )
    text += (
        "## Interpretation and stopping boundary\n\n"
        "Layer errors above are actual captured forwards. Omitted-block scores are a separate "
        "FP32 algebraic decomposition under frozen FullKV queries and values, with observed "
        "softmax rounding retained. They do not predict changed queries, later-layer propagation, "
        "or generated-value drift. A high score is NOT causal evidence that restoring that block "
        "will improve the final answer. No selector change or larger cohort is promoted. "
        "The next permissible intervention is a bounded same-byte causal restore/donor test "
        "on consumed inputs only if these captures identify a coherent omission. "
        "No prior-mixing sweep is justified. Capture timings are contaminated; "
        "no latency or benchmark-quality superiority claim is made.\n"
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
            decision="stop_at_actual_layer_and_omission_diagnostic",
            selector_modified=False,
            larger_campaign_submitted=False,
            no_unverified_intervention_promotion=True,
        ),
    )
    print(json.dumps(gate))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare", "submit", "worker", "analyze"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--plan-sha")
    p.add_argument("--index", type=int)
    a = p.parse_args()
    if a.mode == "prepare":
        print(prepare(a.root))
    elif a.mode == "submit":
        launch.submit(
            a.root,
            a.plan_sha,
            script_name="evicastkv_early_prefix.sbatch",
            worker_minutes=20,
            memory_gb=16,
        )
    elif a.mode == "worker":
        worker(a.root, a.plan_sha, a.index)
    else:
        analyze(a.root, a.plan_sha)


if __name__ == "__main__":
    main()
