#!/usr/bin/env python3
"""Common lossless text-source representation on consumed native inputs."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from unittest.mock import patch

import evicastkv_lookm_selection_merge as causal
import evicastkv_request_validation as previous
from evicastkv_campaign import environment

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
ANCHOR_REFERENCE = Path(
    "/scratch/djy8hg/runs/evicastkv/isolated-draft-evidence-consumed-20261031-v1"
)
METHODS = (
    "mosaickv_exact",
    "lookm_reimpl",
    "prompt_attention_topk",
    "vl_cache_reimpl",
    "prefixkv_reimpl",
)
VARIANTS = {
    method + "--" + representation: (method, representation)
    for method in METHODS
    for representation in ("anchor", "source")
}


def scored_text(tokenizer, ids, eos_ids):
    """Score only through EOS; keep the complete trajectory for timing."""
    end = next((i for i, token in enumerate(ids) if token in eos_ids), len(ids))
    return dict(
        answer=tokenizer.decode(ids[:end], skip_special_tokens=True),
        scored_tokens=end,
        timed_tokens=len(ids),
        first_terminal_index=end if end < len(ids) else None,
        terminal_ids=sorted(eos_ids),
    )


def materialize_packets(packets, adapter):
    """Projection-backed packets require the actual layer's frozen weights."""
    return tuple(
        packet.materialize(block.self_attn)
        for packet, block in zip(packets, adapter._language_layers(), strict=True)
    )


def checkpoint_variant(out, row, fixed, greedy):
    import torch

    variant = row["variant"]
    if variant not in VARIANTS:
        raise ValueError("unregistered diagnostic variant")
    path = out / (variant + "-logits.pt")
    if path.exists():
        raise FileExistsError(path)
    torch.save(dict(fixed=fixed, greedy=greedy), path)
    with (out / "raw.jsonl").open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")


def prepare(root):
    if root.exists():
        raise ValueError("new immutable diagnostic root required")
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", "68dac21", "HEAD", "--", "mosaickv/src"],
        cwd=SOURCE,
        text=True,
    ).splitlines()
    if changed != ["mosaickv/src/mosaickv/evaluation/text_source_storage.py"]:
        raise ValueError("only the registered lossless text source module may change")
    parent = causal.parent.PARENT
    old = previous.read(parent / "plan.json")
    if sha(parent / "plan.json") != causal.parent.PARENT_SHA:
        raise ValueError("consumed parent changed")
    bound = dict(old["bound_files"])
    bound[str(parent / "plan.json")] = sha(parent / "plan.json")
    for job in old["jobs"]:
        folder = parent / f"workers/{job['index']}"
        result = previous.read(folder / "result.json")
        if result["status"] != "completed" or result["failures"]:
            raise ValueError("complete consumed FullKV reference required")
        bound[str(folder / "result.json")] = sha(folder / "result.json")
        for name, digest in result["outputs"].items():
            if sha(folder / name) != digest:
                raise ValueError("reference changed")
            bound[str(folder / name)] = digest
    protocol = SOURCE / "mosaickv/docs/text_source_storage_20261031.md"
    bound[str(protocol)] = sha(protocol)
    for index in range(2):
        folder = ANCHOR_REFERENCE / f"workers/{index}"
        result = previous.read(folder / "result.json")
        if result["status"] != "completed" or result["failures"]:
            raise ValueError("unchanged anchor-control reference must be complete")
        bound[str(folder / "result.json")] = sha(folder / "result.json")
        for name, digest in result["outputs"].items():
            if sha(folder / name) != digest:
                raise ValueError("old anchor reference changed")
            bound[str(folder / name)] = digest
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=clean_source(SOURCE),
            bound_files=bound,
            parent=str(parent),
            stage="consumed_common_lossless_text_source_diagnostic",
            variants=VARIANTS,
            anchor_reference_root=str(ANCHOR_REFERENCE),
            no_fresh_or_large_campaign_authorized=True,
            paper_campaign_authorized=False,
            unchanged_guard=dict(
                horizon=32, maximum_KL_FP32=0.02, maximum_logit_difference=1.0, token_agreement=1.0
            ),
            future_draft_steps=0,
            no_selected_kv_or_logical_position_change=True,
            whole_request_kv_peak_not_certified=True,
            bitwise_same_decoded_cache_and_128_token_logits_required=True,
        ),
    )
    print(sha(root / "plan.json"))


def worker(root, expected, index):
    from dataclasses import asdict

    import torch

    from mosaickv.adapters.huggingface.registry import load_hf_adapter
    from mosaickv.config import RunConfig, config_sha256
    from mosaickv.evaluation import storage_runs
    from mosaickv.evaluation.head_local_precision import bundle, gpu_controls
    from mosaickv.evaluation.lookm_merge_ablation import comparison
    from mosaickv.evaluation.request_precision import PromptQueryCapture
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
            raise ValueError("one A100 required")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ValueError("deterministic workspace required")
        torch.set_num_threads(12)
        torch.manual_seed(plan["seed"])
        torch.use_deterministic_algorithms(True)
        write_once(out / "codec_controls.json", gpu_controls())
        cfg = RunConfig.from_mapping(previous.resolved_mapping(plan, job))
        if (
            cfg.model.id != "llava-hf/llava-1.5-7b-hf"
            or cfg.model.revision != "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
            or cfg.model.precision.value != "bf16"
            or cfg.execution.attention_implementation != "eager"
        ):
            raise ValueError("pinned model/precision/backend changed")
        adapter = load_hf_adapter(
            cfg.model.id,
            revision=cfg.model.revision,
            model_kwargs=_model_kwargs(cfg, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        payload = previous.read(job["inputs"][0])
        parent = Path(plan["parent"]) / f"workers/{index}"
        full = torch.load(parent / "logits.pt", map_location="cpu", weights_only=True)[
            "0-full_kv-ordinary"
        ]
        manifest = previous.read(parent / "manifest.json")
        configurations = {
            name: dict(
                base_config_sha256=config_sha256(cfg),
                plan_sha256=expected,
                method=method,
                representation=representation,
                policy="head3_boundary8",
                allocation_mode="balanced_for_mosaickv_topk_profile_otherwise",
                bitwise_decoded_kv_identity=True,
                baseline_merged_text_preserved=True,
                prompt_window=16,
            )
            for name, (method, representation) in VARIANTS.items()
        }

        def digest(value):
            return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

        manifest.update(
            source_sha=plan["source_sha"],
            plan_sha256=expected,
            diagnostic_only=True,
            common_lossless_text_variants=VARIANTS,
            future_forecast_disabled=True,
            measurement_type="ablation_measured",
            prototype_or_repair=False,
            shared_source_storage_only_ablation=True,
            full_three_tier_runtime_used=False,
            residual_repair=False,
            no_benchmark_reference_input_to_constructor=True,
            environment=environment(),
            gpu=asdict(capture_gpu_environment(torch)),
            cuda_runtime=torch.version.cuda,
            config_sha256=digest(configurations),
            resolved_variant_configurations=configurations,
            baseline_commit=plan["source_sha"],
            evaluator_commit=plan["source_sha"],
        )
        write_once(out / "manifest.json", manifest)
        prepared = adapter.prepare_inputs(previous.load_messages(payload))
        from mosaickv.evaluation.text_source_storage import TextSourceCapture, optimize

        with (
            PromptQueryCapture(adapter, summaries=False, normalized=True, prior=False) as capture,
            TextSourceCapture(adapter, prepared) as text_capture,
        ):
            prefill = adapter.prefill(prepared, capture_queries=False, capture_attentions=True)
        if comparison(full[:1], prefill.logits.reshape(1, -1).float().cpu())[
            "maximum_logit_difference"
        ]:
            raise RuntimeError("ordinary prefill changed")
        source, _ = adapter._legacy_layers(prefill.state.past_key_values)
        before = causal.tensor_hash(source)
        forced = tuple(full.argmax(-1)[:33].tolist())
        from mosaickv.evaluation.exact_storage import ExactFormat
        from mosaickv.evaluation.lossless_bf16 import PackedBF16

        raw_formats = tuple(
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
        cloned, _ = storage_runs.run(
            adapter, prefill, raw_formats, tokens=33, forced=forced, sealed=False, audit=False
        )
        clone_control = comparison(full[:33], cloned)
        write_once(out / "clone_repack_control.json", clone_control)
        if (
            clone_control["maximum_logit_difference"] > 0.001
            or clone_control["token_agreement"] != 1
        ):
            raise RuntimeError("FullKV clone/repack transport control failed")
        del raw_formats, cloned
        rows, saved, anchors, anchor_keys, anchor_values = [], {}, {}, {}, {}
        built, greedy_anchors, build_seconds = {}, {}, {}
        for name, (method, representation) in VARIANTS.items():
            policy = "head3_boundary8"
            steps = 0
            draft_logits = []
            original_decode = adapter.decode_one_token

            def observed(*args, decode=original_decode, captured=draft_logits, **kwargs):
                output = decode(*args, **kwargs)
                captured.append(output.logits.detach().reshape(-1).float().cpu())
                return output

            torch.cuda.synchronize()
            started = time.monotonic()
            if representation == "anchor":
                with patch.object(adapter, "decode_one_token", observed):
                    packets, proof = bundle(
                        adapter,
                        prepared,
                        prefill,
                        capture,
                        cfg,
                        method,
                        plan["profile_path"],
                        plan["ratio"],
                        policies=(policy,),
                        allocation_mode="balanced"
                        if method in ("mosaickv_exact", "prompt_attention_topk")
                        else "profile",
                    )
                built[method] = packets, proof
            else:
                original_packets, original_proof = built[method]
                converted, records = [], []
                for layer_index, (packet, positions) in enumerate(
                    zip(original_packets[policy], original_proof["source_memberships"], strict=True)
                ):
                    candidate, record = optimize(
                        packet,
                        text_capture.records[layer_index],
                        positions,
                        text_capture.positions,
                        adapter._language_layers()[layer_index].self_attn,
                    )
                    converted.append(candidate)
                    records.append(dict(layer=layer_index, **record))
                packets = {policy: tuple(converted)}
                proof = dict(
                    original_proof,
                    common_lossless_text_source=records,
                    original_logical_positions_unchanged=True,
                )
            torch.cuda.synchronize()
            conversion_seconds = time.monotonic() - started
            if representation == "anchor":
                build_seconds[method] = conversion_seconds
            construction = conversion_seconds + (
                build_seconds[method] if representation == "source" else 0
            )
            if len(draft_logits) != steps:
                raise RuntimeError("draft did not reuse the original prefill exactly once")
            draft_control = comparison(full[1:5], torch.stack(draft_logits)) if steps else None
            if steps and (
                draft_control["maximum_logit_difference"] > 0.001
                or draft_control["token_agreement"] != 1
            ):
                raise RuntimeError("isolated FullKV draft differs from ordinary reference")
            write_once(out / f"{name}-construction.json", proof)
            decoded = materialize_packets(packets[policy], adapter)
            if representation == "anchor":
                anchor_keys[method] = tuple(pair[0] for pair in decoded)
                anchor_values[method] = tuple(pair[1] for pair in decoded)
            keys_unchanged = all(
                torch.equal(k.view(torch.int16), a.view(torch.int16))
                for (k, _), a in zip(decoded, anchor_keys[method], strict=True)
            )
            if not keys_unchanged:
                raise RuntimeError("value-only construction altered encoded cache keys")
            changed_scalars = sum(
                int((v.view(torch.int16) != a.view(torch.int16)).sum())
                for (_, v), a in zip(decoded, anchor_values[method], strict=True)
            )
            del decoded
            fixed, _ = storage_runs.run(
                adapter,
                prefill,
                packets[policy],
                tokens=33,
                forced=forced,
                sealed=False,
                audit=False,
            )
            greedy, storage = storage_runs.run(
                adapter, prefill, packets[policy], tokens=128, sealed=False, audit=False
            )
            if method == "mosaickv_exact" and representation == "anchor":
                reference = torch.load(
                    Path(plan["anchor_reference_root"]) / f"workers/{index}/logits.pt",
                    weights_only=True,
                    map_location="cpu",
                )
                if not torch.equal(fixed, reference["anchor:fixed"]) or not torch.equal(
                    greedy, reference["anchor:greedy"]
                ):
                    raise RuntimeError("zero-strength anchor drifted from previous immutable run")
            if representation == "anchor":
                anchors[method] = fixed
            if representation == "anchor":
                greedy_anchors[method] = greedy
            elif (
                changed_scalars
                or not torch.equal(fixed, anchors[method])
                or not torch.equal(greedy, greedy_anchors[method])
            ):
                raise RuntimeError(
                    "lossless source transport changed cache values or 128-token logits"
                )
            row = dict(
                variant=name,
                policy=policy,
                task=payload["task"],
                sample_id=payload["sample_id"],
                source_sha=plan["source_sha"],
                plan_sha256=expected,
                manifest_path=str(out / "manifest.json"),
                config_sha256=digest(configurations[name]),
                status="completed",
                construction_seconds=construction,
                fullkv_fixed_prefix=comparison(full[1:33], fixed[1:]),
                relative_anchor_fixed=comparison(anchors[method][1:], fixed[1:]),
                fullkv_draft_control=draft_control,
                prompt_component_bytes=sum(f.component_bytes for f in packets[policy]),
                greedy_ids=greedy.argmax(-1).tolist(),
                storage=storage["storage"],
                original_source_unchanged=before == causal.tensor_hash(source),
                encoded_keys_bitwise_unchanged=keys_unchanged,
                encoded_value_scalars_changed=changed_scalars,
                representation=representation,
                method=method,
                conversion_seconds=conversion_seconds,
                construction_includes_anchor_build=True,
                inactive_reference_formats_excluded_from_decode_owned_cache_bytes=True,
                single_trial_fixed_method_order_exploratory_timing_only=True,
                measurement_type="ablation_measured",
                prototype_or_repair=False,
            )
            from mosaickv.evaluation.metrics.scorers import rouge_l_f1

            eos = adapter.model.generation_config.eos_token_id
            eos_ids = {eos} if isinstance(eos, int) else set(eos or ())
            if not eos_ids:
                raise RuntimeError("explicit terminal-token policy required")
            scored = scored_text(adapter.processor.tokenizer, row["greedy_ids"], eos_ids)
            full_answer = scored_text(
                adapter.processor.tokenizer, full.argmax(-1).tolist(), eos_ids
            )
            row.update(
                scored_answer=scored,
                fullkv_reference_rouge_l=rouge_l_f1(scored["answer"], (full_answer["answer"],)),
                fullkv_reference_artifact_sha256=sha(parent / "logits.pt"),
                rouge_is_consumed_two_input_diagnostic_not_benchmark_confirmation=True,
            )
            rows.append(row)
            saved[name + ":fixed"] = fixed
            saved[name + ":greedy"] = greedy
            checkpoint_variant(out, row, fixed, greedy)
            del packets
        torch.save(saved, out / "logits.pt")
        result["status"] = "completed"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        result["failures"].append(error)
        payload = previous.read(job["inputs"][0])
        with (out / "raw.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    dict(
                        status="failed",
                        error=error,
                        variant=locals().get("name", "setup"),
                        sample_id=payload["sample_id"],
                        task=payload["task"],
                        source_sha=plan["source_sha"],
                        plan_sha256=expected,
                        measurement_type="ablation_measured",
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
                raise ValueError("producer provenance mismatch")
            for name, digest in result["outputs"].items():
                if sha(folder / name) != digest:
                    raise ValueError("producer changed")
            failures.extend(result["failures"])
            rows.extend(
                json.loads(line) for line in (folder / "raw.jsonl").read_text().splitlines()
            )
        except (OSError, ValueError) as exc:
            failures.append(dict(index=index, error=str(exc)))
    write_once(
        root / "analysis/decision.json",
        dict(
            complete=len(rows) == 2 * len(VARIANTS) and not failures,
            failures=failures,
            rows=rows,
            source_sha=plan["source_sha"],
            plan_sha256=expected,
            superiority_established=False,
            independent_quality_N=0,
            no_fresh_or_large_campaign_authorized=True,
        ),
    )


def submit(root, expected):
    plan = previous.common.inputs(root, expected)
    proof = previous.read(root / "cpu.json")
    if (
        proof["source_sha"] != plan["source_sha"]
        or proof["returncode"]
        or proof["failed_or_skipped"]
        or proof["test_count"] < 2892
        or proof["xml_sha256"] != sha(root / "cpu.xml")
    ):
        raise ValueError("clean complete CPU proof required")
    queue = subprocess.check_output(
        ["squeue", "-h", "-r", "-u", os.environ["USER"], "-o", "%i|%b"], text=True
    )
    if any("gpu" in row.lower() for row in queue.splitlines()):
        raise ValueError("wait for preceding owned GPU arrays before bounded diagnostic")
    (root / "slurm").mkdir(exist_ok=True)
    script = str(SOURCE / "mosaickv/slurm/evicastkv_text_source_diagnostic.sbatch")
    args = [str(SOURCE), str(root), expected]
    gpu = subprocess.check_output(
        [
            "sbatch",
            "--parsable",
            "--partition=bii-gpu",
            "--account=bii_dsc_community",
            "--reservation=bi_fox_dgx",
            "--gres=gpu:a100:1",
            "--nodes=1",
            "--ntasks=1",
            "--cpus-per-task=12",
            "--mem=48G",
            "--time=00:30:00",
            "--array=0-1%2",
            "--job-name=mk-text-source",
            f"--output={root}/slurm/gpu-%A_%a.out",
            f"--error={root}/slurm/gpu-%A_%a.err",
            script,
            *args,
            "worker",
        ],
        text=True,
    ).strip()
    cpu_job = subprocess.check_output(
        [
            "sbatch",
            "--parsable",
            "--partition=bii",
            "--account=bii_dsc_community",
            "--cpus-per-task=2",
            "--mem=16G",
            "--time=00:10:00",
            f"--dependency=afterany:{gpu}",
            "--job-name=mk-text-source-score",
            f"--output={root}/slurm/analysis-%j.out",
            f"--error={root}/slurm/analysis-%j.err",
            script,
            *args,
            "analyze",
        ],
        text=True,
    ).strip()
    from control_a100_schedule import tick

    policy = dict(
        timezone="America/New_York",
        maximum_worker_minutes=30,
        maximum_a100s=6,
        six_a100s_explicitly_authorized=True,
        arrays=[gpu],
    )
    write_once(root / "scheduler/policy.json", policy)
    write_once(
        root / "scheduler/registration.json", tick(policy, {}, root / "scheduler/state.json", False)
    )
    write_once(root / "submission.json", dict(gpu_array=gpu, cpu_analysis=cpu_job))
    print(json.dumps(dict(gpu_array=gpu, cpu_analysis=cpu_job)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("prepare", "bootstrap", "cpu", "worker", "analyze", "submit")
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha")
    args = parser.parse_args()
    if args.mode == "bootstrap":
        prepare(args.root)
        digest = sha(args.root / "plan.json")
        previous.cpu(args.root, digest)
        submit(args.root, digest)
    elif args.mode == "cpu":
        previous.cpu(args.root, args.plan_sha)
    elif args.mode == "worker":
        worker(args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    else:
        globals()[args.mode](args.root, args.plan_sha) if args.mode != "prepare" else prepare(
            args.root
        )
