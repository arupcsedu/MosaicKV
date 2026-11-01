#!/usr/bin/env python3
"""Consumed-input precision-versus-selection causal control; higher-memory, not competitive."""

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import evicastkv_funded_delimiter_diagnostic as parent
import evicastkv_request_validation as previous
from evicastkv_campaign import environment

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
OLD = Path("/scratch/djy8hg/runs/evicastkv/funded-delimiter-consumed-20261031-v2")
VARIANTS = ("quantized", "selected_bf16")


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
    protocol = SOURCE / "mosaickv/docs/selection_precision_causal_20261031.md"
    bound[str(protocol)] = sha(protocol)
    negative = Path(
        "/scratch/djy8hg/runs/evicastkv/causal-head-routing-consumed-20261031-v1/analysis/decision.json"
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
            stage="consumed_selection_precision_decomposition",
            variants=VARIANTS,
            future_draft_steps=0,
            higher_memory_causal_control_not_cap_eligible=True,
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
    from mosaickv.evaluation.exact_storage import ExactFormat
    from mosaickv.evaluation.head_local_precision import bundle
    from mosaickv.evaluation.lookm_merge_ablation import comparison
    from mosaickv.evaluation.lossless_bf16 import PackedBF16
    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
    from mosaickv.evaluation.metrics.scorers import rouge_l_f1
    from mosaickv.evaluation.request_precision import PromptQueryCapture
    from mosaickv.evaluation.selection_precision_causal import selected_original
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
        draft_seconds = 0.0
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
        candidates = dict(
            quantized=source_only,
            selected_bf16=selected_original(source, proof["source_memberships"]),
        )
        write_once(
            out / "construction.json",
            dict(
                source_storage=source_proof,
                original_source_cap_bytes=cap_bytes,
                selection_memberships_identical=True,
                no_draft_or_second_prefill=True,
                higher_memory_selected_bf16_is_not_cap_eligible=True,
                not_a_new_compression_candidate=True,
                raw_KV_storage_bytes=sum(
                    t.numel() * t.element_size()
                    for packet in candidates["selected_bf16"]
                    for t in packet.private()
                ),
                preserved_negative_causal_diagnostic=True,
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
                        draft_steps=0,
                        greedy_predictions=128,
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
            disabled_components=[
                "future_query_forecast",
                "evidence_graph",
                "submodular_selector",
                "prototype",
                "repair",
            ],
            variant="same_selection_original_precision_causal_control",
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
            greedy, storage = storage_runs.run(
                adapter, prefill, formats, tokens=128, sealed=False, audit=False
            )
            torch.save(dict(fixed=fixed, greedy=greedy), out / (name + "-logits.pt"))
            if name == "quantized":
                if not torch.equal(fixed, old_native["fixed"]) or not torch.equal(
                    greedy, old_native["greedy"]
                ):
                    raise RuntimeError(
                        "quantized control changed original source-only 128-token trajectory"
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
                            precision_variant=name,
                        ),
                        sort_keys=True,
                    ).encode()
                ).hexdigest(),
                method="mosaickv_exact",
                effective_variant="selection_precision_causal_" + name,
                measurement_type="ablation_measured",
                same_selected_positions=True,
                cap_eligible=name == "quantized",
                higher_memory_causal_control=name == "selected_bf16",
                no_cap_or_competitive_claim=True,
                fullkv_fixed_prefix=comparison(full[1:33], fixed[1:]),
                relative_zero=comparison(zero_fixed[1:], fixed[1:]),
                fullkv_reference_rouge_l=rouge_l_f1(scored["answer"], (reference["answer"],)),
                scored_answer=scored,
                storage=storage["storage"],
                decode_wall_seconds=storage["decode_wall_seconds"],
                decode_cuda_ms=storage["decode_cuda_ms"],
                draft_seconds=draft_seconds,
                original_source_unchanged=before == parent.causal.tensor_hash(source),
                selected_KV_original_bf16_only=name == "selected_bf16",
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
    decomposition = []
    for job in plan["jobs"]:
        sample = previous.read(job["inputs"][0])["sample_id"]
        group = {
            r["variant"]: r
            for r in rows
            if r.get("sample_id") == sample and r.get("status") == "completed"
        }
        if set(group) == set(VARIANTS):
            q, exact = group["quantized"], group["selected_bf16"]
            decomposition.append(
                dict(
                    sample_id=sample,
                    task=q["task"],
                    quantized_guard=q["fullkv_fixed_prefix"]["registered_guard_pass"],
                    original_selected_guard=exact["fullkv_fixed_prefix"]["registered_guard_pass"],
                    quantized_rouge=q["fullkv_reference_rouge_l"],
                    original_selected_rouge=exact["fullkv_reference_rouge_l"],
                    selected_original_initial_bytes=exact["storage"][
                        "initial_active_kv_storage_bytes"
                    ],
                    quantized_initial_bytes=q["storage"]["initial_active_kv_storage_bytes"],
                    not_a_byte_matched_or_competitive_comparison=True,
                )
            )
    write_once(
        root / "analysis/decision.json",
        dict(
            complete=len(rows) == 4 and not failures,
            failures=failures,
            rows=rows,
            selection_precision_decomposition=decomposition,
            no_new_compression_candidate=True,
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
