#!/usr/bin/env python3
"""Consumed-input last-layer exact nonlinear-tail reranker; no paper authorization."""

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
COUNTS = {"shadow1": (1, False), "tail_selected1": (1, True)}


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
    protocol = SOURCE / "mosaickv/docs/last_layer_tail_20261031.md"
    bound[str(protocol)] = sha(protocol)
    negative = Path(
        "/scratch/djy8hg/runs/evicastkv/exact-head-bypass-consumed-20261031-v2/analysis/decision.json"
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
            stage="consumed_last_layer_nonlinear_tail_reranker",
            variants=COUNTS,
            future_draft_steps=32,
            native_B1_verification_shortlist=4,
            final_head_count=1,
            sidecar_history_capacity=128,
            no_donor_removal=True,
            attention_bias_disabled=True,
            main_KV_values_unchanged_virtual_full_heads_added=True,
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
    from mosaickv.evaluation.exact_head_bypass import construct
    from mosaickv.evaluation.exact_storage import ExactFormat
    from mosaickv.evaluation.head_local_precision import bundle
    from mosaickv.evaluation.last_layer_tail import TailScout, rerank
    from mosaickv.evaluation.lookm_merge_ablation import comparison
    from mosaickv.evaluation.lossless_bf16 import PackedBF16
    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
    from mosaickv.evaluation.metrics.scorers import rouge_l_f1
    from mosaickv.evaluation.request_precision import PromptQueryCapture
    from mosaickv.evaluation.text_source_storage import TextSourceCapture, optimize
    from mosaickv.forecasting.huggingface import run_isolated_greedy_query_rollout
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
        draft_logits = []
        original_decode = adapter.decode_one_token

        def observe_draft(*args, **kwargs):
            value = original_decode(*args, **kwargs)
            draft_logits.append(value.logits.detach().reshape(-1).float().cpu())
            return value

        started = time.monotonic()
        with patch.object(adapter, "decode_one_token", observe_draft):
            draft = run_isolated_greedy_query_rollout(
                adapter, prefill, steps=32, capture_attentions=False
            )
        draft_seconds = time.monotonic() - started
        draft_control = comparison(full[1:33], torch.stack(draft_logits))
        if (
            not draft.original_cache_unchanged
            or draft_control["maximum_logit_difference"] > 0.001
            or draft_control["token_agreement"] != 1
        ):
            raise RuntimeError("isolated draft/source control failed")
        packets, proof = bundle(
            adapter,
            prepared,
            prefill,
            capture,
            cfg,
            "mosaickv_exact",
            plan["profile_path"],
            plan["ratio"],
            policies=("head3_boundary8",),
            allocation_mode="balanced",
        )
        originals = packets["head3_boundary8"]
        cap_bytes = backing_bytes(tuple(t for f in originals for t in f.tensors()))
        attentions = tuple(b.self_attn for b in adapter._language_layers())
        source_only = []
        source_proof = []
        for layer, (packet, positions, attention) in enumerate(
            zip(originals, proof["source_memberships"], attentions, strict=True)
        ):
            optimized, record = optimize(
                packet, text_capture.records[layer], positions, text_capture.positions, attention
            )
            source_only.append(optimized)
            source_proof.append(record)
        source_only = tuple(source_only)
        online_reference = torch.stack(draft_logits)
        online_forced = tuple((*draft.token_ids, 0))
        torch.cuda.synchronize()
        tail_started = time.monotonic()
        with TailScout(adapter, source[-1]) as scout:
            scout_logits, scout_storage = storage_runs.run(
                adapter,
                prefill,
                source_only,
                tokens=33,
                forced=online_forced,
                sealed=False,
                audit=False,
            )
        language = adapter.model.model.language_model
        tail_record, tail_outputs = rerank(
            scout,
            language.norm,
            adapter.model.lm_head,
            scout_logits[1:],
            online_reference,
            shortlist=4,
        )
        torch.cuda.synchronize()
        tail_seconds = time.monotonic() - tail_started
        torch.save(
            dict(
                source=scout_logits,
                reference=online_reference,
                candidate_outputs=tail_outputs,
            ),
            out / "tail-counterfactual-logits.pt",
        )
        write_once(
            out / "tail_ranking.json",
            dict(
                tail_record,
                tail_selection_seconds=tail_seconds,
                scout_storage=scout_storage["storage"],
            ),
        )
        selected_head = tail_record["chosen_head"]
        route_head = selected_head if selected_head is not None else 0
        ranking = [dict(layer=len(source) - 1, head=route_head)]
        candidates = {}
        active_cost = {}
        for name, (count, enabled) in COUNTS.items():
            routes = tuple((r["layer"], r["head"]) for r in ranking[:count])
            enabled = enabled and selected_head is not None
            candidates[name] = construct(source_only, source, routes, enabled=enabled)
            cost = storage_runs.cache_from_formats(
                adapter, candidates[name], updates=128, audit=False, sealed=False
            )
            active_cost[name] = cost.initial_bytes
            cost.layers.clear()
            del cost
            if active_cost[name] > cap_bytes:
                raise RuntimeError("exact-head private prompt/history exceeds original anchor cap")
        write_once(
            out / "construction.json",
            dict(
                draft_control=draft_control,
                original_prefill_reused=True,
                second_prefill=False,
                original_cache_unchanged=draft.original_cache_unchanged,
                clone_seconds=draft.cache_clone_seconds,
                draft_seconds=draft_seconds,
                benchmark_labels_or_reference_logits_used_for_estimation=False,
                all_route_scores=ranking,
                source_storage=source_proof,
                active_initial_bytes=active_cost,
                original_anchor_cap_bytes=cap_bytes,
                risk_is_surrogate_not_quality_guarantee=True,
                actual_nonlinear_tail_ranking=tail_record,
                tail_selection_seconds=tail_seconds,
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
                        head_counts=COUNTS,
                        draft_steps=32,
                        tail_shortlist=4,
                        sidecar_history_capacity=128,
                        raw_source_anchor_ratio=plan["ratio"],
                    ),
                    sort_keys=True,
                ).encode()
            ).hexdigest(),
            environment=environment(),
            gpu=asdict(capture_gpu_environment(torch)),
            cuda_runtime=torch.version.cuda,
            baseline_commit=plan["source_sha"],
            evaluator_commit=plan["source_sha"],
            diagnostic_only=True,
            no_shared_byte_superiority_claim=True,
            no_fresh_cohort=True,
            measurement_type="ablation_measured",
            modalities="one_image_and_text_only",
            disabled_components=["evidence_graph", "submodular_selector", "prototype", "repair"],
            variant="last_layer_nonlinear_tail_exact_head_diagnostic",
            whole_request_KV_peak_certified=False,
        )
        write_once(out / "manifest.json", manifest)
        forced = tuple(full.argmax(-1)[:33].tolist())
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

        def fixed_run(formats):
            return storage_runs.run(
                adapter, prefill, formats, tokens=33, forced=forced, sealed=False, audit=False
            )

        name = "ordinary_clone_control"
        clone, _ = fixed_run(raw)
        control = comparison(full[:33], clone)
        if control["maximum_logit_difference"] > 0.001 or control["token_agreement"] != 1:
            raise RuntimeError("ordinary full-cache replay differs")
        write_once(out / "clone_control.json", control)
        top8 = tuple((r["layer"], r["head"]) for r in ranking[:8])
        full_sidecar = construct(raw, source, top8, enabled=True)
        name = "full_head_numerical_control"
        replay, sidecar_record = fixed_run(full_sidecar)
        torch.save(dict(reference=clone, head_replay=replay), out / "fullcache-control-logits.pt")
        control = comparison(clone, replay)
        write_once(
            out / "exact_head_fullcache_control.json",
            dict(control, storage=sidecar_record["storage"]),
        )
        if control["maximum_logit_difference"] > 0.001 or control["token_agreement"] != 1:
            raise RuntimeError("exact-head full-cache recomputation differs")
        old_native = torch.load(
            OLD / f"workers/{index}/mosaickv_exact--source-logits.pt",
            map_location="cpu",
            weights_only=True,
        )
        zero_fixed = None
        for name, formats in candidates.items():
            fixed, _ = fixed_run(formats)
            if name == "tail_selected1":
                predicted = (
                    scout_logits[1:] if selected_head is None else tail_outputs[selected_head]
                )
                if not torch.equal(fixed[1:], predicted):
                    raise RuntimeError("native B1 tail prediction differs from actual head replay")
            greedy, storage = storage_runs.run(
                adapter, prefill, formats, tokens=128, sealed=False, audit=False
            )
            torch.save(dict(fixed=fixed, greedy=greedy), out / (name + "-logits.pt"))
            if name == "shadow1":
                if not torch.equal(fixed, old_native["fixed"]) or not torch.equal(
                    greedy, old_native["greedy"]
                ):
                    raise RuntimeError(
                        "shadow sidecar changed original source-only 128-token control"
                    )
                zero_fixed = fixed
            eos = adapter.model.generation_config.eos_token_id
            eos_ids = {eos} if type(eos) is int else set(eos or ())
            scored = parent.scored_text(
                adapter.processor.tokenizer, greedy.argmax(-1).tolist(), eos_ids
            )
            reference = parent.scored_text(
                adapter.processor.tokenizer, full.argmax(-1).tolist(), eos_ids
            )
            row = dict(
                status="completed",
                variant=name,
                source_sha=plan["source_sha"],
                plan_sha256=expected,
                task=payload["task"],
                sample_id=payload["sample_id"],
                manifest_path=str(out / "manifest.json"),
                config_sha256=hashlib.sha256(
                    json.dumps(
                        dict(
                            manifest_config=manifest["config_sha256"],
                            variant=name,
                            restored_heads=COUNTS[name][0],
                        ),
                        sort_keys=True,
                    ).encode()
                ).hexdigest(),
                method="mosaickv_exact",
                effective_variant="last_layer_nonlinear_tail_" + name,
                measurement_type="ablation_measured",
                restored_heads=int(name == "tail_selected1" and selected_head is not None),
                tail_selection=tail_record,
                tail_selection_seconds=tail_seconds,
                fullkv_fixed_prefix=comparison(full[1:33], fixed[1:]),
                relative_zero=comparison(zero_fixed[1:], fixed[1:]),
                fullkv_reference_rouge_l=rouge_l_f1(scored["answer"], (reference["answer"],)),
                scored_answer=scored,
                storage=storage["storage"],
                decode_wall_seconds=storage["decode_wall_seconds"],
                decode_cuda_ms=storage["decode_cuda_ms"],
                draft_seconds=draft_seconds,
                original_source_unchanged=before == parent.causal.tensor_hash(source),
                main_selected_KV_unchanged_virtual_full_heads_added=True,
                benchmark_quality_not_established=True,
                diagnostic_timing_not_end_to_end=True,
                whole_request_kv_peak_not_certified=True,
                numerical_capture_outside_greedy_timing=True,
            )
            if not row["original_source_unchanged"]:
                raise RuntimeError("probe altered source cache")
            with (out / "raw.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
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
    for index in range(2):
        group = [
            r
            for r in rows
            if r.get("sample_id") == previous.read(plan["jobs"][index]["inputs"][0])["sample_id"]
            and r.get("status") == "completed"
        ]
        if len(group) == 2:
            for field in (
                "initial_active_kv_storage_bytes",
                "final_active_kv_storage_bytes",
                "peak_active_kv_storage_bytes",
            ):
                if (
                    len(
                        {
                            r["storage"][field]
                            for r in group
                            if r["variant"] in ("shadow1", "tail_selected1")
                        }
                    )
                    != 1
                ):
                    failures.append(
                        dict(index=index, error="causal-control storage differs: " + field)
                    )
    paired = []
    for job in plan["jobs"]:
        sample = previous.read(job["inputs"][0])["sample_id"]
        by_variant = {
            r["variant"]: r
            for r in rows
            if r.get("sample_id") == sample and r.get("status") == "completed"
        }
        if set(by_variant) == set(COUNTS):
            anchor, candidate = by_variant["shadow1"], by_variant["tail_selected1"]
            paired.append(
                dict(
                    sample_id=sample,
                    task=candidate["task"],
                    meaningful_accepted_head=candidate["restored_heads"] == 1,
                    unchanged_guard_pass=candidate["fullkv_fixed_prefix"]["registered_guard_pass"],
                    diagnostic_rouge_not_worse=(
                        candidate["fullkv_reference_rouge_l"] >= anchor["fullkv_reference_rouge_l"]
                    ),
                    exact_tail_prediction_matches_actual=True,
                )
            )
    diagnostic_pass = (
        len(paired) == 2
        and not failures
        and all(
            r["meaningful_accepted_head"]
            and r["unchanged_guard_pass"]
            and r["diagnostic_rouge_not_worse"]
            for r in paired
        )
    )
    write_once(
        root / "analysis/decision.json",
        dict(
            complete=len(rows) == 4 and not failures,
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
