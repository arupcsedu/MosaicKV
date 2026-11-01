#!/usr/bin/env python3
"""Consumed-input causal diagnostic, not new-cohort or paper authorization."""

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
STRENGTHS = {"zero": 0.0, "half": 0.5, "full": 1.0}


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
    protocol = SOURCE / "mosaickv/docs/modality_normalizer_20261031.md"
    bound[str(protocol)] = sha(protocol)
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=clean_source(SOURCE),
            bound_files=bound,
            stage="consumed_draft_visual_text_odds_causal_diagnostic",
            variants=STRENGTHS,
            future_draft_steps=4,
            maximum_visual_multiplier=2.0,
            maximum_visual_downweight=0.5,
            text_and_history_bias_zero=True,
            key_value_values_and_memberships_unchanged_relative_to_completed_cache=True,
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


def modal_capture(attentions, memberships, text):
    import torch

    rows = []
    for attention, heads in zip(attentions, memberships, strict=True):
        n = len(heads[0])
        text_mask = torch.tensor([[p in text for p in h] for h in heads], device=attention.device)
        visual_mask = torch.tensor(
            [[p is not None and p not in text for p in h] for h in heads], device=attention.device
        )
        p = attention[0, :, 0].float()
        rows.append(
            torch.stack(
                (
                    (p[:, :n] * visual_mask).sum(-1),
                    (p[:, :n] * text_mask).sum(-1),
                    p[:, n:].sum(-1),
                ),
                -1,
            )
        )
    return torch.stack(rows).cpu()


def worker(root, expected, index):
    import torch

    from mosaickv.adapters.huggingface.registry import load_hf_adapter
    from mosaickv.config import RunConfig, config_sha256
    from mosaickv.evaluation import storage_runs
    from mosaickv.evaluation.exact_storage import ExactFormat
    from mosaickv.evaluation.funded_text_completion import generation_delimiter, transaction
    from mosaickv.evaluation.head_local_precision import bundle
    from mosaickv.evaluation.lookm_merge_ablation import comparison
    from mosaickv.evaluation.lossless_bf16 import PackedBF16
    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
    from mosaickv.evaluation.metrics.scorers import rouge_l_f1
    from mosaickv.evaluation.modality_mass import estimate, gpu_controls, wrap
    from mosaickv.evaluation.request_precision import PromptQueryCapture
    from mosaickv.evaluation.text_source_storage import TextSourceCapture
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
        write_once(out / "mask_controls.json", gpu_controls())
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
                adapter, prefill, steps=4, capture_attentions=True
            )
        draft_seconds = time.monotonic() - started
        draft_control = comparison(full[1:5], torch.stack(draft_logits))
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
        completed, complete_proof = transaction(
            packets["head3_boundary8"],
            text_capture.records,
            proof["source_memberships"],
            text_capture.positions,
            generation_delimiter(prepared, adapter.processor.tokenizer),
            tuple(b.self_attn for b in adapter._language_layers()),
            source,
        )
        memberships = complete_proof["final_source_memberships"]
        candidates, corrections = {}, {}
        for name, strength in STRENGTHS.items():
            biases, records = estimate(
                draft.attention_steps, memberships, text_capture.positions, strength=strength
            )
            candidates[name] = wrap(completed, biases, memberships)
            corrections[name] = records
            actual = backing_bytes(tuple(t for p in candidates[name] for t in p.tensors()))
            if actual > complete_proof["original_physical_cap_bytes"]:
                raise RuntimeError("bias metadata exceeds original measured anchor cap")
        # Estimation is frozen before the 32-prefix evaluation-only oracle runs.
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
                completion=complete_proof,
                corrections=corrections,
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
                        strengths=STRENGTHS,
                        maximum_visual_multiplier=2.0,
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
            variant="source_funded_completion_with_draft_visual_text_log_odds",
            whole_request_KV_peak_certified=False,
        )
        write_once(out / "manifest.json", manifest)
        forced = tuple(full.argmax(-1)[:33].tolist())
        text = set(text_capture.positions)
        n = source[0][0].shape[-2]
        full_memberships = tuple(
            tuple(tuple(range(n)) for _ in range(k.shape[1])) for k, v in source
        )
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

        def fixed_run(formats, positions):
            observed = []

            def capture_modal(*args, **kwargs):
                value = original_decode(*args, **dict(kwargs, capture_attentions=True))
                observed.append(modal_capture(value.attention_weights, positions, text))
                return value

            with patch.object(adapter, "decode_one_token", capture_modal):
                logits, record = storage_runs.run(
                    adapter, prefill, formats, tokens=33, forced=forced, sealed=False, audit=False
                )
            return logits, record, torch.stack(observed)

        clone, _, full_mass = fixed_run(raw, full_memberships)
        control = comparison(full[:33], clone)
        if control["maximum_logit_difference"] > 0.001 or control["token_agreement"] != 1:
            raise RuntimeError("ordinary full-cache replay differs")
        write_once(out / "clone_control.json", control)
        torch.save(full_mass, out / "full_modal_mass.pt")
        old_native = torch.load(
            OLD / f"workers/{index}/mosaickv_exact--completion-logits.pt",
            map_location="cpu",
            weights_only=True,
        )
        zero_fixed = None
        for name, formats in candidates.items():
            fixed, _, masses = fixed_run(formats, memberships)
            greedy, storage = storage_runs.run(
                adapter, prefill, formats, tokens=128, sealed=False, audit=False
            )
            torch.save(dict(fixed=fixed, greedy=greedy, masses=masses), out / (name + "-logits.pt"))
            if name == "zero":
                if not torch.equal(fixed, old_native["fixed"]) or not torch.equal(
                    greedy, old_native["greedy"]
                ):
                    raise RuntimeError(
                        "zero-bias completion differs from original 128-token control"
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
                            strength=STRENGTHS[name],
                        ),
                        sort_keys=True,
                    ).encode()
                ).hexdigest(),
                method="mosaickv_exact",
                effective_variant="draft_modality_odds_" + name,
                measurement_type="ablation_measured",
                strength=STRENGTHS[name],
                fullkv_fixed_prefix=comparison(full[1:33], fixed[1:]),
                relative_zero=comparison(zero_fixed[1:], fixed[1:]),
                fullkv_reference_rouge_l=rouge_l_f1(scored["answer"], (reference["answer"],)),
                scored_answer=scored,
                storage=storage["storage"],
                decode_wall_seconds=storage["decode_wall_seconds"],
                decode_cuda_ms=storage["decode_cuda_ms"],
                draft_seconds=draft_seconds,
                actual_group_mass_mean_absolute_error=float((masses - full_mass).abs().mean()),
                actual_group_mass_max_absolute_error=float((masses - full_mass).abs().max()),
                original_source_unchanged=before == parent.causal.tensor_hash(source),
                original_selected_KV_and_memberships_unchanged=True,
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
        if len(group) == 3:
            for field in (
                "initial_active_kv_storage_bytes",
                "final_active_kv_storage_bytes",
                "peak_active_kv_storage_bytes",
            ):
                if len({r["storage"][field] for r in group}) != 1:
                    failures.append(
                        dict(index=index, error="causal-control storage differs: " + field)
                    )
    write_once(
        root / "analysis/decision.json",
        dict(
            complete=len(rows) == 6 and not failures,
            failures=failures,
            rows=rows,
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
