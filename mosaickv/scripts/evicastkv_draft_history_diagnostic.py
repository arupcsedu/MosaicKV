#!/usr/bin/env python3
"""Consumed-input online-draft generated-history causal repair; no paper authorization."""

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import evicastkv_funded_delimiter_diagnostic as parent
import evicastkv_request_validation as previous
from evicastkv_campaign import environment

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
OLD = Path("/scratch/djy8hg/runs/evicastkv/funded-delimiter-consumed-20261031-v2")
METHODS = (
    "mosaickv_exact",
    "lookm_reimpl",
    "prompt_attention_topk",
    "vl_cache_reimpl",
    "prefixkv_reimpl",
)
VARIANTS = {
    method + "--" + variant: (method, variant == "repair")
    for method in METHODS
    for variant in ("shadow", "repair")
}


def prepare(root):
    if root.exists():
        raise ValueError("new write-once output root required")
    old = previous.read(OLD / "plan.json")
    bound = dict(old["bound_files"])
    bound[str(OLD / "plan.json")] = sha(OLD / "plan.json")
    for i in range(2):
        folder = OLD / f"workers/{i}"
        result = previous.read(folder / "result.json")
        if result["status"] != "completed" or result["failures"]:
            raise ValueError("complete consumed source/completion controls required")
        bound[str(folder / "result.json")] = sha(folder / "result.json")
        for name, digest in result["outputs"].items():
            if sha(folder / name) != digest:
                raise ValueError("old native artifact changed")
            bound[str(folder / name)] = digest
    protocol = SOURCE / "mosaickv/docs/draft_history_causal_20261031.md"
    bound[str(protocol)] = sha(protocol)
    negative = Path(
        "/scratch/djy8hg/runs/evicastkv/coherent-token-consumed-20261031-v1/analysis/decision.json"
    )
    outcome = previous.read(negative)
    if not outcome["complete"] or outcome["failures"]:
        raise ValueError("completed prior causal diagnostic required")
    bound[str(negative)] = sha(negative)
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=clean_source(SOURCE),
            bound_files=bound,
            stage="consumed_generated_history_drift_causal_control",
            variants=VARIANTS,
            future_draft_steps=32,
            explicit_cpu_history_residual=True,
            common_history_policy_for_all_five_methods=True,
            greedy_predictions=128,
            no_donor_removal=True,
            attention_bias_disabled=True,
            identical_selected_source_memberships=True,
            no_untouched_or_paper_claim=True,
            paper_campaign_authorized=False,
            independent_quality_N=0,
            no_fresh_or_large_campaign_authorized=True,
            original_source_root=str(OLD),
            selection_policy="head3_boundary8_balanced",
            error_metric="actual_model_32_prefix_KL_logit_greedy_guard",
        ),
    )
    print(sha(root / "plan.json"))


def worker(root, expected, index):
    import torch

    from mosaickv.adapters.huggingface.registry import load_hf_adapter
    from mosaickv.config import RunConfig, config_sha256
    from mosaickv.evaluation import storage_runs
    from mosaickv.evaluation.draft_history_causal import HistoryReference
    from mosaickv.evaluation.exact_storage import ExactFormat
    from mosaickv.evaluation.head_local_precision import bundle
    from mosaickv.evaluation.lookm_merge_ablation import comparison
    from mosaickv.evaluation.lossless_bf16 import PackedBF16
    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
    from mosaickv.evaluation.metrics.scorers import rouge_l_f1
    from mosaickv.evaluation.request_precision import PromptQueryCapture
    from mosaickv.evaluation.text_source_storage import TextSourceCapture, optimize
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    plan = previous.common.inputs(root, expected)
    job = plan["jobs"][index]
    out = root / f"workers/{index}"
    out.mkdir(parents=True, exist_ok=False)
    result = dict(status="failed", failures=[], source_sha=plan["source_sha"], plan_sha256=expected)
    (out / "raw.jsonl").touch(exist_ok=False)
    try:
        if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name():
            raise ValueError("one allocated A100 required")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ValueError("deterministic CUDA workspace required")
        torch.set_num_threads(12)
        torch.manual_seed(plan["seed"])
        torch.use_deterministic_algorithms(True)
        cfg = RunConfig.from_mapping(previous.resolved_mapping(plan, job))
        if (
            cfg.model.id != "llava-hf/llava-1.5-7b-hf"
            or cfg.model.revision != "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
            or cfg.model.precision.value != "bf16"
            or cfg.execution.attention_implementation != "eager"
        ):
            raise ValueError("pinned LLaVA/BF16/eager configuration required")
        adapter = load_hf_adapter(
            cfg.model.id,
            revision=cfg.model.revision,
            model_kwargs=_model_kwargs(cfg, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        payload = previous.read(job["inputs"][0])
        ref_folder = Path(plan["parent"]) / f"workers/{index}"
        full = torch.load(ref_folder / "logits.pt", map_location="cpu", weights_only=True)[
            "0-full_kv-ordinary"
        ]
        prepared = adapter.prepare_inputs(previous.load_messages(payload))
        if not bool(prepared.model_inputs["attention_mask"].bool().all()):
            raise NotImplementedError("unpadded single-image causal prompt required")
        with (
            PromptQueryCapture(adapter, summaries=False, normalized=True, prior=False) as capture,
            TextSourceCapture(adapter, prepared) as text_capture,
        ):
            prefill = adapter.prefill(prepared, capture_queries=False, capture_attentions=True)
        source, _ = adapter._legacy_layers(prefill.state.past_key_values)
        before = parent.causal.tensor_hash(source)
        logical_start = prefill.state.next_decode_position
        raw = tuple(
            ExactFormat(
                "kv",
                tuple(k.shape),
                pair=(
                    PackedBF16(tuple(k.shape), True, (k,)),
                    PackedBF16(tuple(v.shape), True, (v,)),
                ),
            )
            for k, v in source
        )
        layers, heads, dim = len(source), source[0][0].shape[1], source[0][0].shape[-1]
        history = torch.empty(
            (32, layers, 2, heads, 1, dim), dtype=torch.bfloat16, device="cpu", pin_memory=True
        )
        tokens = torch.empty(32, dtype=torch.long, device="cpu")
        counter = 0
        original_decode = adapter.decode_one_token

        def capture_history(token, state, **kwargs):
            nonlocal counter
            output = original_decode(token, state, **kwargs)
            if counter >= 32 or any(
                layer.history_length != counter + 1 for layer in state.past_key_values.layers
            ):
                raise RuntimeError("private FullKV forecast history is not consecutive")
            current = torch.stack(
                [
                    torch.stack(
                        (
                            layer.history[0][0, :, counter : counter + 1],
                            layer.history[1][0, :, counter : counter + 1],
                        )
                    )
                    for layer in state.past_key_values.layers
                ]
            )
            history[counter].copy_(current, non_blocking=False)
            tokens[counter] = int(token.item())
            counter += 1
            return output

        torch.cuda.synchronize()
        started = time.monotonic()
        with patch.object(adapter, "decode_one_token", capture_history):
            draft_logits, forecast_storage = storage_runs.run(
                adapter, prefill, raw, tokens=33, sealed=False, audit=False
            )
        torch.cuda.synchronize()
        draft_seconds = time.monotonic() - started
        forecast_control = comparison(full[:33], draft_logits)
        if (
            counter != 32
            or forecast_control["maximum_logit_difference"] > 0.001
            or forecast_control["token_agreement"] != 1
            or before != parent.causal.tensor_hash(source)
            or prefill.state.next_decode_position != logical_start
        ):
            raise RuntimeError("shared-prefix isolated FullKV forecast control failed")
        reference = HistoryReference(history, tokens, logical_start)
        torch.save(
            dict(kv=history, tokens=tokens, logical_start=reference.logical_start),
            out / "online-full-history.pt",
        )
        write_once(
            out / "forecast.json",
            dict(
                forecast_control=forecast_control,
                forecast_seconds=draft_seconds,
                private_history=True,
                prompt_owners_shared_read_only_not_deep_copied=True,
                source_prefill_unchanged=True,
                original_prefill_reused_no_second_prefill=True,
                online_reference_not_benchmark_annotation=True,
                cpu_residual_bytes=reference.cpu_bytes,
                storage=forecast_storage["storage"],
            ),
        )
        manifest = previous.read(ref_folder / "manifest.json")
        manifest.update(
            source_sha=plan["source_sha"],
            source_dirty=False,
            plan_sha256=expected,
            config_sha256=hashlib.sha256(
                json.dumps(
                    dict(
                        base=config_sha256(cfg),
                        plan=expected,
                        variants=VARIANTS,
                        online_history_horizon=32,
                        raw_source_ratio=plan["ratio"],
                    ),
                    sort_keys=True,
                ).encode()
            ).hexdigest(),
            environment=environment(),
            gpu=asdict(capture_gpu_environment(torch)),
            cuda_runtime=torch.version.cuda,
            baseline_commit=plan["source_sha"],
            evaluator_commit=plan["source_sha"],
            measurement_type="ablation_measured",
            online_history_reference_sha256=sha(out / "online-full-history.pt"),
            diagnostic_only=True,
            no_shared_byte_or_superiority_claim=True,
            independent_quality_N=0,
            variant="common_source_storage_with_online_forecast_history_control",
            generated_history_is_raw_lossless_BF16_for_every_method=True,
            CPU_residuals_and_GPU_staging_fully_charged=True,
            full_three_tier_runtime_used=False,
            disabled_components=[
                "future_query_forecasting_for_selection",
                "evidence_graph",
                "value_aware_block_utility",
                "submodular_selector",
                "prototype",
            ],
            selection_variant="frozen_shared_prior_not_full_graph_compressor",
            history_repair_experimental_not_main_adapter_capability=True,
            whole_request_KV_peak_certified=False,
        )
        write_once(out / "manifest.json", manifest)
        forced = tuple(full.argmax(-1)[:33].tolist())

        def fixed_run(formats, *, enabled):
            return storage_runs.run(
                adapter,
                prefill,
                formats,
                tokens=33,
                forced=forced,
                sealed=False,
                audit=False,
                history_reference=reference,
                history_repair_enabled=enabled,
            )

        name = "fullcache_history_numerical_control"
        repaired_full, full_details = fixed_run(raw, enabled=True)
        numerical = comparison(full[:33], repaired_full)
        write_once(
            out / "fullcache_control.json",
            dict(numerical, repair=full_details["storage"]["draft_history_repair"]),
        )
        if numerical["maximum_logit_difference"] > 0.001 or numerical["token_agreement"] != 1:
            raise RuntimeError("FullKV repaired-history control differs from ordinary FullKV")
        del raw
        constructions = {}
        for method in METHODS:
            name = method + "--construction"
            built_at = time.monotonic()
            packets, proof = bundle(
                adapter,
                prepared,
                prefill,
                capture,
                cfg,
                method,
                plan["profile_path"],
                plan["ratio"],
                policies=("head3_boundary8",),
                allocation_mode="balanced"
                if method in ("mosaickv_exact", "prompt_attention_topk")
                else "profile",
            )
            originals = packets["head3_boundary8"]
            original_cap = backing_bytes(tuple(t for f in originals for t in f.tensors()))
            source_only, records = [], []
            for layer, (packet, positions, block) in enumerate(
                zip(originals, proof["source_memberships"], adapter._language_layers(), strict=True)
            ):
                optimized, record = optimize(
                    packet,
                    text_capture.records[layer],
                    positions,
                    text_capture.positions,
                    block.self_attn,
                )
                source_only.append(optimized)
                records.append(record)
            source_only = tuple(source_only)
            construction_seconds = time.monotonic() - built_at
            constructions[method] = dict(
                source_storage=records,
                construction_seconds=construction_seconds,
                original_source_cap_bytes=original_cap,
            )
            old_native = torch.load(
                OLD / f"workers/{index}" / (method + "--source-logits.pt"),
                map_location="cpu",
                weights_only=True,
            )
            for enabled in (False, True):
                name = method + ("--repair" if enabled else "--shadow")
                fixed, fixed_info = fixed_run(source_only, enabled=enabled)
                greedy, details = storage_runs.run(
                    adapter,
                    prefill,
                    source_only,
                    tokens=128,
                    sealed=False,
                    audit=False,
                    history_reference=reference,
                    history_repair_enabled=enabled,
                )
                torch.save(dict(fixed=fixed, greedy=greedy), out / (name + "-logits.pt"))
                if not enabled and (
                    not torch.equal(fixed, old_native["fixed"])
                    or not torch.equal(greedy, old_native["greedy"])
                ):
                    raise RuntimeError("inactive history control changed original 128-token logits")
                repair = details["storage"]["draft_history_repair"]
                if not enabled and repair["repaired_token_steps"]:
                    raise RuntimeError("disabled history policy repaired a token")
                if details["storage"]["initial_active_kv_storage_bytes"] > original_cap:
                    raise RuntimeError("source and staging exceed original own source cap")
                eos = adapter.model.generation_config.eos_token_id
                eos_ids = {eos} if type(eos) is int else set(eos or ())
                scored = parent.scored_text(
                    adapter.processor.tokenizer, greedy.argmax(-1).tolist(), eos_ids
                )
                reference_text = parent.scored_text(
                    adapter.processor.tokenizer, full.argmax(-1).tolist(), eos_ids
                )
                row = dict(
                    status="completed",
                    variant=name,
                    method=method,
                    task=payload["task"],
                    sample_id=payload["sample_id"],
                    source_sha=plan["source_sha"],
                    plan_sha256=expected,
                    manifest_path=str(out / "manifest.json"),
                    config_sha256=hashlib.sha256(
                        json.dumps(
                            dict(parent=manifest["config_sha256"], variant=name), sort_keys=True
                        ).encode()
                    ).hexdigest(),
                    effective_variant="common_codec_" + name + "_online_history_diagnostic",
                    measurement_type="ablation_measured",
                    history_repair_enabled=enabled,
                    fullkv_fixed_prefix=comparison(full[1:33], fixed[1:]),
                    fullkv_reference_rouge_l=rouge_l_f1(
                        scored["answer"], (reference_text["answer"],)
                    ),
                    scored_answer=scored,
                    storage=details["storage"],
                    fixed_prefix_repair=fixed_info["storage"]["draft_history_repair"],
                    decode_wall_seconds=details["decode_wall_seconds"],
                    decode_cuda_ms=details["decode_cuda_ms"],
                    forecast_seconds=draft_seconds,
                    construction_seconds=construction_seconds,
                    cpu_residual_bytes=reference.cpu_bytes,
                    original_source_unchanged=before == parent.causal.tensor_hash(source),
                    no_forced_draft_token_emission_or_fullkv_fallback=True,
                    original_prompt_KV_and_selection_unchanged=True,
                    metrics_are_consumed_diagnostics_not_official_MM_Vet_or_PPL=True,
                    different_encoded_memory_points_not_byte_matched=True,
                    whole_request_peak_certified=False,
                    no_independent_quality_or_superiority_claim=True,
                )
                if not row["original_source_unchanged"]:
                    raise RuntimeError("history repair altered original prefill")
                with (out / "raw.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
            del originals, packets, source_only
        write_once(out / "constructions.json", constructions)
        result["status"] = "completed"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        result["failures"].append(error)
        with (out / "raw.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    dict(
                        status="failed",
                        error=error,
                        sample_id=payload["sample_id"] if "payload" in locals() else None,
                        task=payload["task"] if "payload" in locals() else None,
                        method="mosaickv_exact",
                        manifest_path=str(out / "manifest.json")
                        if (out / "manifest.json").exists()
                        else None,
                        source_sha=plan["source_sha"],
                        plan_sha256=expected,
                        variant=locals().get("name", "setup"),
                    )
                )
                + "\n"
            )
    finally:
        result["outputs"] = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
        write_once(out / "result.json", result)


def analyze(root, expected):
    plan = previous.common.inputs(root, expected)
    rows, failures = [], []
    for index in range(2):
        folder = root / f"workers/{index}"
        try:
            result = previous.read(folder / "result.json")
            if result["source_sha"] != plan["source_sha"] or result["plan_sha256"] != expected:
                raise ValueError("wrong producer provenance")
            for name, digest in result["outputs"].items():
                if sha(folder / name) != digest:
                    raise ValueError("producer changed")
            failures.extend(result["failures"])
            rows.extend(
                json.loads(line) for line in (folder / "raw.jsonl").read_text().splitlines()
            )
        except (ValueError, OSError) as exc:
            failures.append(str(exc))
    paired = []
    for job in plan["jobs"]:
        sample = previous.read(job["inputs"][0])["sample_id"]
        group = {
            r["variant"]: r
            for r in rows
            if r.get("sample_id") == sample and r.get("status") == "completed"
        }
        if set(group) == set(VARIANTS):
            anchor = group["mosaickv_exact--shadow"]
            candidate = group["mosaickv_exact--repair"]
            repaired = [
                r
                for r in group.values()
                if r["history_repair_enabled"] and r["method"] != "mosaickv_exact"
            ]
            paired.append(
                dict(
                    sample_id=sample,
                    task=candidate["task"],
                    guard=candidate["fullkv_fixed_prefix"]["registered_guard_pass"],
                    meaningful_repaired_steps=candidate["storage"]["draft_history_repair"][
                        "repaired_token_steps"
                    ],
                    ROUGE_not_worse_than_anchor=candidate["fullkv_reference_rouge_l"]
                    >= anchor["fullkv_reference_rouge_l"],
                    ROUGE_not_worse_than_all_repaired_baselines=all(
                        candidate["fullkv_reference_rouge_l"] >= r["fullkv_reference_rouge_l"]
                        for r in repaired
                    ),
                    smaller_active_storage_than_LOOKM_at_all_boundaries=all(
                        candidate["storage"][field]
                        < group["lookm_reimpl--repair"]["storage"][field]
                        for field in (
                            "initial_active_kv_storage_bytes",
                            "final_active_kv_storage_bytes",
                            "peak_active_kv_storage_bytes",
                        )
                    ),
                    note="different measured points, development-only, not byte matched",
                )
            )
    diagnostic_pass = (
        len(paired) == 2
        and not failures
        and all(
            p["guard"] and p["meaningful_repaired_steps"] > 0 and p["ROUGE_not_worse_than_anchor"]
            for p in paired
        )
    )
    write_once(
        root / "analysis/decision.json",
        dict(
            complete=len(rows) == 20 and not failures,
            failures=failures,
            rows=rows,
            paired_diagnostic_gates=paired,
            model_level_diagnostic_pass=diagnostic_pass,
            source_sha=plan["source_sha"],
            plan_sha256=expected,
            independent_quality_N=0,
            superiority_established=False,
            no_fresh_or_large_campaign_authorized=True,
        ),
    )


def submit(root, expected):
    # Reuse the tested finite two-worker scheduler, but bind OUR wrapper exactly.
    original = parent.subprocess.check_output

    def scoped(command, **kwargs):
        if command and command[0] == "sbatch":
            command = [
                str(SOURCE / "mosaickv/slurm" / (Path(__file__).stem + ".sbatch"))
                if x.endswith("/evicastkv_funded_delimiter_diagnostic.sbatch")
                else x
                for x in command
            ]
        return original(command, **kwargs)

    with patch.object(parent.subprocess, "check_output", scoped):
        parent.submit(root, expected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "cpu", "submit", "worker", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.root)
    elif args.mode == "cpu":
        previous.cpu(args.root, args.plan_sha)
    elif args.mode == "worker":
        worker(args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    else:
        globals()[args.mode](args.root, args.plan_sha)
