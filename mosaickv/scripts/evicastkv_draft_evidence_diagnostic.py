#!/usr/bin/env python3
"""Two consumed-input draft-evidence controls; never benchmark confirmation."""

import argparse
import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import evicastkv_lookm_selection_merge as causal
import evicastkv_request_validation as previous
from evicastkv_campaign import environment

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
VARIANTS = {
    "anchor": ("head3_boundary8", 0),
    "hybrid4_boundary8": ("head3_boundary8", 4),
    "hybrid4_k2v": ("head3k2v_mse", 4),
}


def prepare(root):
    if root.exists():
        raise ValueError("new immutable diagnostic root required")
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
    protocol = SOURCE / "mosaickv/docs/isolated_draft_evidence_20261031.md"
    bound[str(protocol)] = sha(protocol)
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=clean_source(SOURCE),
            bound_files=bound,
            parent=str(parent),
            stage="consumed_four_step_future_attention_diagnostic",
            variants=VARIANTS,
            no_fresh_or_large_campaign_authorized=True,
            paper_campaign_authorized=False,
            unchanged_guard=dict(
                horizon=32, maximum_KL_FP32=0.02, maximum_logit_difference=1.0, token_agreement=1.0
            ),
            fixed_prompt_to_draft_weight=0.5,
        ),
    )
    print(sha(root / "plan.json"))


def worker(root, expected, index):
    from dataclasses import asdict

    import torch

    from mosaickv.adapters.huggingface.registry import load_hf_adapter
    from mosaickv.config import RunConfig
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
        manifest.update(
            source_sha=plan["source_sha"],
            plan_sha256=expected,
            diagnostic_only=True,
            frozen_four_step_weight=0.5,
            no_benchmark_reference_input_to_constructor=True,
            environment=environment(),
            gpu=asdict(capture_gpu_environment(torch)),
            cuda_runtime=torch.version.cuda,
        )
        write_once(out / "manifest.json", manifest)
        prepared = adapter.prepare_inputs(previous.load_messages(payload))
        with PromptQueryCapture(adapter, summaries=False, normalized=True, prior=False) as capture:
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
        rows, saved, anchor = [], {}, None
        for name, (policy, steps) in VARIANTS.items():
            draft_logits = []
            original_decode = adapter.decode_one_token

            def observed(*args, decode=original_decode, captured=draft_logits, **kwargs):
                output = decode(*args, **kwargs)
                captured.append(output.logits.detach().reshape(-1).float().cpu())
                return output

            torch.cuda.synchronize()
            started = time.monotonic()
            with patch.object(adapter, "decode_one_token", observed):
                packets, proof = bundle(
                    adapter,
                    prepared,
                    prefill,
                    capture,
                    cfg,
                    "mosaickv_exact",
                    plan["profile_path"],
                    plan["ratio"],
                    policies=(policy,),
                    allocation_mode="balanced",
                    future_steps=steps,
                )
            torch.cuda.synchronize()
            construction = time.monotonic() - started
            if len(draft_logits) != steps:
                raise RuntimeError("draft did not reuse the original prefill exactly once")
            draft_control = comparison(full[1:5], torch.stack(draft_logits)) if steps else None
            if steps and (
                draft_control["maximum_logit_difference"] > 0.001
                or draft_control["token_agreement"] != 1
            ):
                raise RuntimeError("isolated FullKV draft differs from ordinary reference")
            write_once(out / f"{name}-construction.json", proof)
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
            if anchor is None:
                anchor = fixed
            row = dict(
                variant=name,
                policy=policy,
                task=payload["task"],
                sample_id=payload["sample_id"],
                source_sha=plan["source_sha"],
                plan_sha256=expected,
                manifest_path=str(out / "manifest.json"),
                status="completed",
                construction_seconds=construction,
                fullkv_fixed_prefix=comparison(full[1:33], fixed[1:]),
                relative_anchor_fixed=comparison(anchor[1:], fixed[1:]),
                fullkv_draft_control=draft_control,
                prompt_component_bytes=sum(f.component_bytes for f in packets[policy]),
                greedy_ids=greedy.argmax(-1).tolist(),
                storage=storage["storage"],
                original_source_unchanged=before == causal.tensor_hash(source),
            )
            rows.append(row)
            saved[name + ":fixed"] = fixed
            saved[name + ":greedy"] = greedy
            del packets
        with (out / "raw.jsonl").open("x") as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        torch.save(saved, out / "logits.pt")
        result["status"] = "completed"
    except Exception as exc:
        result["failures"].append(f"{type(exc).__name__}: {exc}")
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
            complete=len(rows) == 6 and not failures,
            failures=failures,
            rows=rows,
            source_sha=plan["source_sha"],
            plan_sha256=expected,
            superiority_established=False,
            independent_quality_N=0,
            no_fresh_or_large_campaign_authorized=True,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "cpu", "worker", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha")
    args = parser.parse_args()
    if args.mode == "cpu":
        previous.cpu(args.root, args.plan_sha)
    elif args.mode == "worker":
        worker(args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    else:
        globals()[args.mode](args.root, args.plan_sha) if args.mode != "prepare" else prepare(
            args.root
        )
