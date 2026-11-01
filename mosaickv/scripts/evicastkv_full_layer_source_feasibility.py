#!/usr/bin/env python3
"""Higher-memory causal ceiling before implementing another source codec."""

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import evicastkv_funded_delimiter_diagnostic as parent
import evicastkv_phase_capacity_diagnostic as phase
import evicastkv_request_validation as previous
from evicastkv_campaign import environment

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
OLD = phase.OLD
NEGATIVE = Path("/scratch/djy8hg/runs/evicastkv/phase-capacity-consumed-20261031-v1")


def prepare(root):
    if root.exists():
        raise ValueError("new write-once causal diagnostic root required")
    old = previous.read(NEGATIVE / "plan.json")
    bound = dict(old["bound_files"])
    bound[str(NEGATIVE / "plan.json")] = sha(NEGATIVE / "plan.json")
    decision = NEGATIVE / "analysis/decision.json"
    status = previous.read(decision)
    if not status["complete"] or status["failures"] or status["model_level_diagnostic_pass"]:
        raise ValueError("completed negative phase-capacity study required")
    bound[str(decision)] = sha(decision)
    for i in range(2):
        folder = NEGATIVE / f"workers/{i}"
        record = previous.read(folder / "result.json")
        if record["status"] != "completed" or record["failures"]:
            raise ValueError("complete prior workers required")
        bound[str(folder / "result.json")] = sha(folder / "result.json")
        for name, digest in record["outputs"].items():
            if sha(folder / name) != digest:
                raise ValueError("prior artifact changed")
            bound[str(folder / name)] = digest
    protocol = SOURCE / "mosaickv/docs/full_layer_source_feasibility_20261101.md"
    bound[str(protocol)] = sha(protocol)
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=clean_source(SOURCE),
            bound_files=bound,
            stage="consumed_full_layer_restoration_causal_ceiling",
            variants=("shadow", "selected_higher_memory_control"),
            original_cap_not_relaxed_for_any_method_claim=True,
            restored_controls_not_cap_eligible=True,
            full_single_layer_controls=32,
            composed_layer_counts=(2, 4, 8),
            source_codec_not_implemented_or_validated_by_payload_lower_bounds=True,
            independent_quality_N=0,
            no_fresh_or_large_campaign_authorized=True,
            paper_campaign_authorized=False,
        ),
    )
    print(sha(root / "plan.json"))


def worker(root, expected, index):
    import torch

    from mosaickv.adapters.huggingface.registry import load_hf_adapter
    from mosaickv.config import RunConfig, config_sha256
    from mosaickv.evaluation import storage_runs
    from mosaickv.evaluation.exact_storage import ExactFormat
    from mosaickv.evaluation.full_layer_causal import ProjectionInputCapture, compositions, restore
    from mosaickv.evaluation.head_local_precision import bundle
    from mosaickv.evaluation.last_layer_tail import statistics
    from mosaickv.evaluation.lookm_merge_ablation import comparison
    from mosaickv.evaluation.lossless_bf16 import PackedBF16
    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
    from mosaickv.evaluation.metrics.scorers import rouge_l_f1
    from mosaickv.evaluation.phase_capacity import PhaseCapture
    from mosaickv.evaluation.precision_tier import pack
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
    name = "setup"
    try:
        if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name():
            raise ValueError("one A100 required")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ValueError("deterministic CUDA workspace required")
        torch.set_num_threads(12)
        torch.manual_seed(plan["seed"])
        torch.use_deterministic_algorithms(True)
        cfg = RunConfig.from_mapping(previous.resolved_mapping(plan, job))
        if (
            cfg.model.id,
            cfg.model.revision,
            cfg.model.precision.value,
            cfg.execution.attention_implementation,
        ) != (
            "llava-hf/llava-1.5-7b-hf",
            "b234b804b114d9e37bb655e11cbbb5f5e971b7a9",
            "bf16",
            "eager",
        ):
            raise ValueError("pinned LLaVA BF16 eager required")
        adapter = load_hf_adapter(
            cfg.model.id,
            revision=cfg.model.revision,
            model_kwargs=_model_kwargs(cfg, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        payload = previous.read(job["inputs"][0])
        folder = Path(plan["parent"]) / f"workers/{index}"
        full = torch.load(folder / "logits.pt", map_location="cpu", weights_only=True)[
            "0-full_kv-ordinary"
        ]
        prepared = adapter.prepare_inputs(previous.load_messages(payload))
        if not prepared.model_inputs["attention_mask"].bool().all():
            raise ValueError("unpadded consumed input required")
        with (
            PromptQueryCapture(adapter, summaries=False, normalized=True, prior=False) as capture,
            TextSourceCapture(adapter, prepared) as text_capture,
            PhaseCapture(adapter) as phases,
            ProjectionInputCapture(adapter) as inputs,
        ):
            prefill = adapter.prefill(prepared, capture_queries=False, capture_attentions=True)
        source, _ = adapter._legacy_layers(prefill.state.past_key_values)
        if len(source) != 32 or len(inputs.records) != 32:
            raise ValueError("32 complete LLaVA projection layers required")
        before = parent.causal.tensor_hash(source)
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
        forced = tuple(full.argmax(-1)[:33].tolist())

        def fixed_run(formats):
            return storage_runs.run(
                adapter, prefill, formats, tokens=33, forced=forced, sealed=False, audit=False
            )

        clone, _ = fixed_run(raw)
        control = comparison(full[:33], clone)
        if control["maximum_logit_difference"] > 0.001 or control["token_agreement"] != 1:
            raise RuntimeError("original FullKV replay failed")
        write_once(out / "fullcache_control.json", control)
        packets, proof = bundle(
            adapter,
            prepared,
            prefill,
            capture,
            cfg,
            "mosaickv_exact",
            plan["profile_path"],
            0.5,
            policies=("head3_boundary8",),
            allocation_mode="balanced",
        )
        original = packets["head3_boundary8"]
        original_cap = backing_bytes(tuple(t for p in original for t in p.tensors()))
        base = tuple(
            optimize(
                p,
                text_capture.records[i],
                proof["source_memberships"][i],
                text_capture.positions,
                b.self_attn,
            )[0]
            for i, (p, b) in enumerate(zip(original, adapter._language_layers(), strict=True))
        )
        shadow, _ = fixed_run(base)
        old = torch.load(
            OLD / f"workers/{index}/mosaickv_exact--source-logits.pt",
            map_location="cpu",
            weights_only=True,
        )
        if not torch.equal(shadow, old["fixed"]):
            raise RuntimeError("anchor changed")
        manifest = previous.read(folder / "manifest.json")
        manifest.update(
            source_sha=plan["source_sha"],
            source_dirty=False,
            plan_sha256=expected,
            config_sha256=hashlib.sha256(
                json.dumps(dict(config=config_sha256(cfg), plan=expected), sort_keys=True).encode()
            ).hexdigest(),
            environment=environment(),
            gpu=asdict(capture_gpu_environment(torch)),
            cuda_runtime=torch.version.cuda,
            evaluator_commit=plan["source_sha"],
            baseline_commit=plan["source_sha"],
            measurement_type="ablation_measured",
            diagnostic_only=True,
            higher_memory_controls_not_cap_eligible=True,
            no_superiority_claim=True,
            independent_quality_N=0,
            disabled_components=[
                "forecast_selector",
                "evidence_graph",
                "value_aware_utility",
                "submodular_selector",
                "prototypes",
                "repair",
            ],
            fixed_prefix_is_evaluation_only=True,
            normalized_projection_capture_CPU_bytes=backing_bytes(tuple(inputs.records.values())),
            actual_timing_protocol=dict(
                warmups=0, repetitions=1, order="fixed", not_paper_latency=True
            ),
            whole_request_peak_certified=False,
        )
        write_once(out / "manifest.json", manifest)
        torch.save(
            dict(
                normalized=inputs.records,
                phases=tuple(t.cpu() for t in phases.phases),
                source_memberships=proof["source_memberships"],
                original_cap_bytes=original_cap,
            ),
            out / "projection-inputs.pt",
        )
        probes, candidates, probe_outputs = [], [], {}
        started = time.monotonic()
        for i in range(32):
            name = "single-layer-" + str(i)
            trial = restore(base, raw, (i,))
            logits, details = fixed_run(trial)
            values = statistics(full[1:33], logits[1:])
            probes.append(dict(layer=i, statistics=values))
            candidates.append(dict(layers=(i,), statistics=values, storage=details["storage"]))
            probe_outputs[name] = logits[1:]
        for selected in compositions(probes):
            name = "composed-" + str(len(selected))
            trial = restore(base, raw, selected)
            logits, details = fixed_run(trial)
            values = statistics(full[1:33], logits[1:])
            candidates.append(dict(layers=selected, statistics=values, storage=details["storage"]))
            probe_outputs[name] = logits[1:]
        eligible = [r for r in candidates if r["statistics"]["guard_pass"]]
        chosen = (
            min(
                eligible,
                key=lambda r: (len(r["layers"]), r["statistics"]["mean_KL_FP32"], r["layers"]),
            )
            if eligible
            else None
        )
        write_once(
            out / "causal_controls.json",
            dict(
                singles=probes,
                candidates=candidates,
                chosen=chosen,
                probe_seconds=time.monotonic() - started,
                original_own_cap_bytes=original_cap,
                controls_are_higher_memory_not_cap_eligible=True,
                benchmark_labels_or_task_scores_used=False,
            ),
        )
        torch.save(
            dict(anchor=shadow[1:], reference=full[1:33], proposal_outputs=probe_outputs),
            out / "causal-logits.pt",
        )
        payload_costs = []
        for i, x in sorted(inputs.records.items()):
            for bits in (2, 3, 4):
                body = pack(x[:, None], bits=bits, axis=3)
                payload_costs.append(
                    dict(
                        layer=i,
                        bits=bits,
                        source_quant_payload_lower_bound_bytes=backing_bytes(body.tensors()),
                        not_an_implemented_cache_or_total_storage_measurement=True,
                    )
                )
        write_once(
            out / "source_payload_lower_bounds.json",
            dict(
                rows=payload_costs,
                excludes="exact_protections,RoPE,positions,maps,leases,history,construction_workspace,allocator_reserved",
                source_codec_not_implemented=True,
            ),
        )
        for name, formats in (
            ("shadow", base),
            (
                "selected_higher_memory_control",
                restore(base, raw, tuple(chosen["layers"])) if chosen else base,
            ),
        ):
            greedy, details = storage_runs.run(
                adapter, prefill, formats, tokens=128, sealed=False, audit=False
            )
            fixed, _ = fixed_run(formats)
            if name == "shadow" and not torch.equal(greedy, old["greedy"]):
                raise RuntimeError("anchor greedy trajectory changed")
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
                task=payload["task"],
                sample_id=payload["sample_id"],
                source_sha=plan["source_sha"],
                plan_sha256=expected,
                manifest_path=str(out / "manifest.json"),
                config_sha256=manifest["config_sha256"],
                measurement_type="ablation_measured",
                fullkv_fixed_prefix=comparison(full[1:33], fixed[1:]),
                fullkv_reference_rouge_l=rouge_l_f1(scored["answer"], (reference["answer"],)),
                scored_answer=scored,
                restored_layers=chosen["layers"] if chosen and name != "shadow" else (),
                source_anchor_fallback=name != "shadow" and chosen is None,
                storage=details["storage"],
                original_cap_bytes=original_cap,
                higher_memory_control_not_cap_eligible=name != "shadow",
                decode_wall_seconds=details["decode_wall_seconds"],
                original_source_unchanged=before == parent.causal.tensor_hash(source),
                cpu_diagnostic_capture_bytes=manifest["normalized_projection_capture_CPU_bytes"],
                not_an_official_MM_Vet_score_or_PPL=True,
                independent_quality_N=0,
                whole_request_peak_certified=False,
            )
            if not row["original_source_unchanged"]:
                raise RuntimeError("causal intervention mutated original source")
            torch.save(dict(fixed=fixed, greedy=greedy), out / (name + "-logits.pt"))
            with (out / "raw.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        result["status"] = "completed"
    except Exception as exc:
        result["failures"].append(f"{type(exc).__name__}: {exc}")
        with (out / "raw.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    dict(
                        status="failed",
                        error=result["failures"][-1],
                        variant=name,
                        source_sha=plan["source_sha"],
                        plan_sha256=expected,
                    )
                )
                + "\n"
            )
    finally:
        result["outputs"] = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
        write_once(out / "result.json", result)
    if result["status"] != "completed":
        raise SystemExit(1)


def analyze(root, expected):
    plan = previous.common.inputs(root, expected)
    rows, failures = [], []
    for i in range(2):
        folder = root / f"workers/{i}"
        try:
            result = previous.read(folder / "result.json")
            if result["source_sha"] != plan["source_sha"] or result["plan_sha256"] != expected:
                raise ValueError("producer provenance changed")
            for name, digest in result["outputs"].items():
                if sha(folder / name) != digest:
                    raise ValueError("producer artifacts changed")
            failures.extend(result["failures"])
            rows.extend(
                json.loads(line) for line in (folder / "raw.jsonl").read_text().splitlines()
            )
        except (ValueError, OSError) as exc:
            failures.append(str(exc))
    opportunity = len(rows) == 4 and not failures
    for job in plan["jobs"]:
        sample = previous.read(job["inputs"][0])["sample_id"]
        group = {
            r["variant"]: r
            for r in rows
            if r.get("sample_id") == sample and r["status"] == "completed"
        }
        a, b = group.get("shadow"), group.get("selected_higher_memory_control")
        opportunity = opportunity and bool(
            a
            and b
            and b["restored_layers"]
            and b["fullkv_fixed_prefix"]["registered_guard_pass"]
            and b["fullkv_reference_rouge_l"] >= a["fullkv_reference_rouge_l"]
        )
    write_once(
        root / "analysis/decision.json",
        dict(
            complete=len(rows) == 4 and not failures,
            failures=failures,
            rows=rows,
            model_level_restoration_opportunity=opportunity,
            cap_eligible_candidate_produced=False,
            source_sha=plan["source_sha"],
            plan_sha256=expected,
            independent_quality_N=0,
            superiority_established=False,
            no_fresh_or_large_campaign_authorized=True,
        ),
    )


def submit(root, expected):
    original = parent.subprocess.check_output

    def scoped(command, **kwargs):
        if command and command[0] == "sbatch":
            command = [
                str(SOURCE / "mosaickv/slurm/evicastkv_full_layer_source_feasibility.sbatch")
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
