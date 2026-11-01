#!/usr/bin/env python3
"""Consumed-input source-funded visual evidence trajectory; no paper authorization."""

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
VARIANTS = ("anchor", "coherent")


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
    protocol = SOURCE / "mosaickv/docs/coherent_token_restoration_20261031.md"
    bound[str(protocol)] = sha(protocol)
    negative = Path(
        "/scratch/djy8hg/runs/evicastkv/selection-precision-consumed-20261031-v1/analysis/decision.json"
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
            stage="consumed_source_funded_single_visual_trajectory",
            variants=VARIANTS,
            future_draft_steps=32,
            actual_full_model_proposals=8,
            restored_visual_tokens=1,
            sidecar_history_capacity=0,
            no_donor_removal=True,
            attention_bias_disabled=True,
            original_selected_KV_unchanged_new_visual_items_added=True,
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
    from mosaickv.evaluation.coherent_token_restoration import rank_tokens, restore_token
    from mosaickv.evaluation.exact_storage import ExactFormat
    from mosaickv.evaluation.head_local_precision import bundle
    from mosaickv.evaluation.last_layer_tail import statistics
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
                adapter, prefill, steps=32, capture_attentions=True
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
        from mosaickv.evaluation.cache_cycles import gpu_controls as cycle_controls
        from mosaickv.evaluation.completion_masks import gpu_controls as mask_controls

        write_once(
            out / "mask-cycle-controls.json",
            dict(
                masks=mask_controls(),
                cycles=cycle_controls(),
            ),
        )
        scout_logits, scout_storage = storage_runs.run(
            adapter,
            prefill,
            source_only,
            tokens=33,
            forced=online_forced,
            sealed=False,
            audit=False,
        )
        visual = tuple(
            p
            for span in prepared.modality_map
            if span.modality != "text"
            for p in range(span.start, span.end)
        )
        ranked = rank_tokens(
            draft.attention_steps[:4],
            source,
            source_only,
            proof["source_memberships"],
            attentions,
            visual,
        )
        schedule = tuple(r for r in ranked if r["forecast_gain"] > 0)[:8]
        if not schedule:
            raise RuntimeError("no positive forecast-demanded visual evidence proposals")
        records, probe_outputs = [], {}
        trials = {}
        for route in schedule:
            trial, transaction = restore_token(
                source_only, source, proof["source_memberships"], attentions, route["position"]
            )
            physical = storage_runs.cache_from_formats(
                adapter, trial, updates=128, audit=False, sealed=False
            )
            initial = physical.initial_bytes
            physical.layers.clear()
            del physical
            if initial > cap_bytes:
                raise RuntimeError("visual trajectory exceeds original physical cap")
            probe_started = time.monotonic()
            values, details = storage_runs.run(
                adapter,
                prefill,
                trial,
                tokens=33,
                forced=online_forced,
                sealed=False,
                audit=False,
            )
            record = dict(
                route,
                transaction=transaction,
                statistics=statistics(online_reference, values[1:]),
                initial_bytes=initial,
                probe_seconds=time.monotonic() - probe_started,
                storage=details["storage"],
            )
            records.append(record)
            probe_outputs[route["position"]] = values[1:]
            trials[route["position"]] = trial
            write_once(out / ("proposal-" + str(route["position"]) + ".json"), record)
            del trial
        anchor_statistics = statistics(online_reference, scout_logits[1:])
        eligible = [
            r
            for r in records
            if r["statistics"]["guard_pass"]
            and r["statistics"]["mean_KL_FP32"] <= anchor_statistics["mean_KL_FP32"]
        ]
        eligible.sort(
            key=lambda r: (
                r["statistics"]["maximum_KL_FP32"],
                r["statistics"]["maximum_logit_difference"],
                r["statistics"]["mean_KL_FP32"],
                r["position"],
            )
        )
        chosen = eligible[0] if eligible else None
        torch.cuda.synchronize()
        tail_seconds = time.monotonic() - tail_started
        selected_position = chosen["position"] if chosen is not None else None
        tail_record = dict(
            chosen_position=selected_position,
            all_causal_records=records,
            baseline=anchor_statistics,
            actual_full_model_proposals=len(schedule),
            zero_tail_approximation_used=False,
            fail_closed_source_anchor=chosen is None,
            no_label_access=True,
            prefixes=32,
            all_first_four_query_evidence_gains=ranked,
        )
        torch.save(
            dict(
                source=scout_logits,
                reference=online_reference,
                proposal_outputs=probe_outputs,
            ),
            out / "coherent-probe-logits.pt",
        )
        write_once(
            out / "coherent_ranking.json",
            dict(
                tail_record,
                selection_seconds=tail_seconds,
            ),
        )
        candidates = dict(
            anchor=source_only,
            coherent=source_only if chosen is None else trials[selected_position],
        )
        active_cost = {}
        for variant, formats in candidates.items():
            cached = storage_runs.cache_from_formats(
                adapter, formats, updates=128, audit=False, sealed=False
            )
            active_cost[variant] = cached.initial_bytes
            cached.layers.clear()
            del cached
            if active_cost[variant] > cap_bytes:
                raise RuntimeError("constructed visual trajectory exceeded original source cap")
        write_once(
            out / "construction.json",
            dict(
                draft_control=draft_control,
                original_prefill_reused=True,
                second_prefill=False,
                original_cache_unchanged=draft.original_cache_unchanged,
                clone_seconds=draft.cache_clone_seconds,
                draft_seconds=draft_seconds,
                benchmark_labels_or_saved_reference_logits_used_for_estimation=False,
                online_fullkv_draft_logits_used_for_acceptance=True,
                all_region_scores=ranked,
                source_storage=source_proof,
                active_initial_bytes=active_cost,
                original_anchor_cap_bytes=cap_bytes,
                risk_is_surrogate_not_quality_guarantee=True,
                actual_causal_route_ranking=tail_record,
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
                        variants=VARIANTS,
                        draft_steps=32,
                        full_model_proposal_count=8,
                        restored_visual_tokens=1,
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
            variant="source_funded_visual_trajectory_diagnostic",
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
        old_native = torch.load(
            OLD / f"workers/{index}/mosaickv_exact--source-logits.pt",
            map_location="cpu",
            weights_only=True,
        )
        zero_fixed = None
        for name, formats in candidates.items():
            fixed, _ = fixed_run(formats)
            if name == "coherent":
                predicted = (
                    scout_logits[1:]
                    if selected_position is None
                    else probe_outputs[selected_position]
                )
                if not torch.equal(fixed[1:], predicted):
                    raise RuntimeError("visual trajectory changed across private replay handles")
            greedy, storage = storage_runs.run(
                adapter, prefill, formats, tokens=128, sealed=False, audit=False
            )
            torch.save(dict(fixed=fixed, greedy=greedy), out / (name + "-logits.pt"))
            if name == "anchor":
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
                            restored_visual_tokens=int(
                                name == "coherent" and selected_position is not None
                            ),
                        ),
                        sort_keys=True,
                    ).encode()
                ).hexdigest(),
                method="mosaickv_exact",
                effective_variant="source_funded_visual_trajectory_" + name,
                measurement_type="ablation_measured",
                restored_visual_tokens=int(name == "coherent" and selected_position is not None),
                restored_visual_position=selected_position if name == "coherent" else None,
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
                original_selected_KV_unchanged_added_BF16_evidence=True,
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
    paired = []
    for job in plan["jobs"]:
        sample = previous.read(job["inputs"][0])["sample_id"]
        by_variant = {
            r["variant"]: r
            for r in rows
            if r.get("sample_id") == sample and r.get("status") == "completed"
        }
        if set(by_variant) == set(VARIANTS):
            anchor, candidate = by_variant["anchor"], by_variant["coherent"]
            paired.append(
                dict(
                    sample_id=sample,
                    task=candidate["task"],
                    meaningful_accepted_visual_token=candidate["restored_visual_tokens"] == 1,
                    unchanged_guard_pass=candidate["fullkv_fixed_prefix"]["registered_guard_pass"],
                    diagnostic_rouge_not_worse=(
                        candidate["fullkv_reference_rouge_l"] >= anchor["fullkv_reference_rouge_l"]
                    ),
                    chosen_private_probe_reproduces_actual=True,
                )
            )
    diagnostic_pass = (
        len(paired) == 2
        and not failures
        and all(
            r["meaningful_accepted_visual_token"]
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
