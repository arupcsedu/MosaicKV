#!/usr/bin/env python3
"""Bounded common-codec precision study; six workers, same two consumed inputs."""

import argparse
import json
import os
import random
import subprocess
from dataclasses import asdict
from pathlib import Path

import evicastkv_lookm_selection_merge as causal
import evicastkv_request_validation as previous
from evicastkv_campaign import environment

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once
from mosaickv.evaluation.head_local_precision import POLICIES
from mosaickv.evaluation.lookm_merge_ablation import comparison

SOURCE = Path(__file__).resolve().parents[2]
GROUPS = (
    ("mosaickv_exact", "prompt_attention_topk"),
    ("lookm_reimpl",),
    ("vl_cache_reimpl", "prefixkv_reimpl"),
)
EXECUTED_POLICIES = ("union3", "head3k2v_refit", "head3k2v_headwise")


def prepare(root):
    if root.exists():
        raise ValueError("new immutable diagnostic root required")
    old = previous.read(causal.parent.PARENT / "plan.json")
    if sha(causal.parent.PARENT / "plan.json") != causal.parent.PARENT_SHA:
        raise ValueError("consumed reference changed")
    bound = dict(old["bound_files"])
    bound[str(causal.parent.PARENT / "plan.json")] = causal.parent.PARENT_SHA
    jobs = []
    for old_job in old["jobs"]:
        folder = causal.parent.PARENT / f"workers/{old_job['index']}"
        result = previous.read(folder / "result.json")
        if result["status"] != "completed" or result["failures"]:
            raise ValueError("consumed parent incomplete")
        bound[str(folder / "result.json")] = sha(folder / "result.json")
        for name, digest in result["outputs"].items():
            if sha(folder / name) != digest:
                raise ValueError("consumed artifact changed")
            bound[str(folder / name)] = digest
        for methods in GROUPS:
            jobs.append(
                dict(old_job, index=len(jobs), parent_index=old_job["index"], methods=methods)
            )
    protocol = SOURCE / "mosaickv/docs/head_local_precision_20261031.md"
    bound[str(protocol)] = sha(protocol)
    superseded = Path(
        "/scratch/djy8hg/runs/evicastkv/head-local-precision-consumed-20261031-v1/analysis/decision.json"
    )
    if superseded.exists():
        bound[str(superseded)] = sha(superseded)
    prior_study = Path(
        "/scratch/djy8hg/runs/evicastkv/head-local-precision-consumed-20261031-v2/analysis/decision.json"
    )
    bound[str(prior_study)] = sha(prior_study)
    plan = dict(
        old,
        jobs=jobs,
        source_sha=clean_source(SOURCE),
        bound_files=bound,
        parent=str(causal.parent.PARENT),
        stage="consumed_precision_specific_corrections_diagnostic",
        precision_policies={p: POLICIES[p] for p in EXECUTED_POLICIES},
        warmups=0,
        timed_repetitions=1,
        automatic_bounded_cohort_after_correctness=False,
        paper_campaign_authorized=False,
        fresh_cohort_authorized=False,
        maximum_a100s=6,
        maximum_new_workers=6,
        no_end_to_end_or_whole_request_peak_claim=True,
        same_source_capacity_not_encoded_byte_matching=True,
        common_codec_applied_to_every_method=True,
        unchanged_guard=dict(
            horizon=32, maximum_KL_FP32=0.02, maximum_logit_difference=1.0, token_agreement=1.0
        ),
        no_model_weight_updates_or_benchmark_label_access=True,
    )
    write_once(root / "plan.json", plan)
    print(sha(root / "plan.json"))


def worker(root, expected, index):
    import torch

    from mosaickv.adapters.huggingface.registry import load_hf_adapter
    from mosaickv.config import RunConfig, config_sha256
    from mosaickv.evaluation import storage_runs
    from mosaickv.evaluation.head_local_precision import bundle, gpu_controls
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
            raise ValueError("deterministic CUDA workspace required")
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
            raise ValueError("frozen model/backend control changed")
        adapter = load_hf_adapter(
            cfg.model.id,
            revision=cfg.model.revision,
            model_kwargs=_model_kwargs(cfg, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        payload = previous.read(job["inputs"][0])
        parent = Path(plan["parent"]) / f"workers/{job['parent_index']}"
        old = torch.load(parent / "logits.pt", weights_only=True, map_location="cpu")
        manifest = previous.read(parent / "manifest.json")
        manifest.update(
            source_sha=plan["source_sha"],
            plan_sha256=expected,
            base_config_sha256=config_sha256(cfg),
            environment=environment(),
            gpu=asdict(capture_gpu_environment(torch)),
            measurement_type="ablation_measured",
            result_category="unified_comparison",
            result_origin="locally_measured",
            vllm="not_used",
            sglang="not_used",
            diagnostic_plan=plan,
            no_independent_quality_claim=True,
        )
        write_once(out / "manifest.json", manifest)
        prepared = adapter.prepare_inputs(previous.load_messages(payload))
        with PromptQueryCapture(adapter, summaries=False, normalized=True, prior=False) as capture:
            prefill = adapter.prefill(prepared, capture_queries=False, capture_attentions=True)
        source, _ = adapter._legacy_layers(prefill.state.past_key_values)
        before = causal.tensor_hash(source)
        full = old["0-full_kv-ordinary"]
        if comparison(full[:1], prefill.logits.reshape(1, -1).float().cpu())[
            "maximum_logit_difference"
        ]:
            raise ValueError("common prefill changed from consumed FullKV")
        forced = tuple(full.argmax(-1)[:33].tolist())
        tensors, summaries = {}, []
        methods = list(job["methods"])
        random.Random(plan["seed"] + index).shuffle(methods)
        with (out / "raw.jsonl").open("x") as stream:
            for method in methods:
                formats, proof = bundle(
                    adapter,
                    prepared,
                    prefill,
                    capture,
                    cfg,
                    method,
                    plan["profile_path"],
                    plan["ratio"],
                    policies=tuple(plan["precision_policies"]),
                )
                write_once(out / f"{method}-construction.json", proof)
                fixed = {}
                greedy = {}
                records = {}
                policies = list(plan["precision_policies"])
                random.Random(plan["seed"] + index).shuffle(policies)
                for policy in policies:
                    row = dict(
                        method=method,
                        policy=policy,
                        sample_id=payload["sample_id"],
                        task=payload["task"],
                        status="failed",
                        source_sha=plan["source_sha"],
                        config_sha256=expected,
                        manifest_path=str(out / "manifest.json"),
                        construction_sha256=sha(out / f"{method}-construction.json"),
                        result_category="unified_comparison",
                        result_origin="locally_measured",
                        measurement_type="ablation_measured",
                    )
                    try:
                        fixed[policy], _ = storage_runs.run(
                            adapter,
                            prefill,
                            formats[policy],
                            tokens=33,
                            forced=forced,
                            sealed=False,
                            audit=False,
                        )
                        greedy[policy], records[policy] = storage_runs.run(
                            adapter,
                            prefill,
                            formats[policy],
                            tokens=128,
                            sealed=False,
                            audit=False,
                        )
                        tensors[f"{method}:{policy}:fixed"] = fixed[policy]
                        tensors[f"{method}:{policy}:greedy"] = greedy[policy]
                        row.update(
                            status="completed",
                            **records[policy],
                            fullkv_fixed_prefix=comparison(full[1:33], fixed[policy][1:]),
                            greedy_ids=greedy[policy].argmax(-1).tolist(),
                            prompt_component_bytes=sum(f.component_bytes for f in formats[policy]),
                        )
                    except Exception as error:
                        row["error"] = f"{type(error).__name__}: {error}"
                        result["failures"].append(
                            dict(method=method, policy=policy, error=row["error"])
                        )
                    summaries.append(row)
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
                if len(greedy) == len(plan["precision_policies"]):
                    control = comparison(old[f"0-{method}-query3"], greedy["union3"])
                    comparisons = []
                    for policy in plan["precision_policies"]:
                        c = comparison(fixed["union3"][1:], fixed[policy][1:])
                        own = comparison(greedy["union3"], greedy[policy])
                        comparisons.append(
                            dict(
                                policy=policy,
                                relative_anchor_fixed=c,
                                relative_anchor_greedy=own,
                                greedy_sequence_identical=own["maximum_logit_difference"] == 0,
                                greedy_tokens_identical=own["token_agreement"] == 1,
                                prompt_bytes=sum(f.component_bytes for f in formats[policy]),
                                reference_prompt_bytes=sum(
                                    f.component_bytes for f in formats["union3"]
                                ),
                                storage=records[policy]["storage"],
                            )
                        )
                    write_once(
                        out / f"{method}-comparison.json",
                        dict(
                            transport=control,
                            policies=comparisons,
                            control_pass=control["maximum_logit_difference"] <= 0.001
                            and control["token_agreement"] == 1,
                        ),
                    )
                del formats
        torch.save(tensors, out / "logits.pt")
        write_once(
            out / "isolation.json",
            dict(original_source_bitwise_unchanged=before == causal.tensor_hash(source)),
        )
        result["status"] = "completed" if not result["failures"] else "failed"
    except Exception as error:
        result["failures"].append(dict(error=f"{type(error).__name__}: {error}"))
    finally:
        result["outputs"] = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
        write_once(out / "result.json", result)


def analyze(root, expected):
    plan = previous.common.inputs(root, expected)
    summaries, failures, inputs = [], [], {}
    for job in plan["jobs"]:
        out = root / f"workers/{job['index']}"
        if not (out / "result.json").exists():
            failures.append(dict(index=job["index"], error="missing producer"))
            continue
        result = previous.read(out / "result.json")
        inputs[str(out / "result.json")] = sha(out / "result.json")
        if result["source_sha"] != plan["source_sha"] or result["plan_sha256"] != expected:
            raise ValueError("producer provenance mismatch")
        for name, digest in result["outputs"].items():
            if sha(out / name) != digest:
                raise ValueError("raw artifact changed")
        failures.extend(result["failures"])
        if result["status"] != "completed":
            continue
        isolation = previous.read(out / "isolation.json")["original_source_bitwise_unchanged"]
        for method in job["methods"]:
            comparison_ = previous.read(out / f"{method}-comparison.json")
            summaries.append(
                dict(index=job["index"], method=method, isolation=isolation, **comparison_)
            )
    eligible = []
    for policy in plan["precision_policies"]:
        if policy == "union3":
            continue
        rows = [next(p for p in s["policies"] if p["policy"] == policy) for s in summaries]
        if (
            len(summaries) == 10
            and not failures
            and all(s["isolation"] and s["control_pass"] for s in summaries)
            and all(
                r["relative_anchor_fixed"]["registered_guard_pass"]
                and r["greedy_tokens_identical"]
                and r["prompt_bytes"] < r["reference_prompt_bytes"]
                for r in rows
            )
        ):
            eligible.append(policy)
    decision = dict(
        source_sha=plan["source_sha"],
        plan_sha256=expected,
        input_hashes=inputs,
        complete=len(summaries) == 10 and not failures,
        failures=failures,
        summaries=summaries,
        common_policies_passing_consumed_guard_and_greedy_identity=eligible,
        no_fresh_or_large_campaign_authorized=True,
        superiority_established=False,
        scope="two consumed inputs; decode-owned backing only; no end-to-end timing claim",
    )
    write_once(root / "analysis/decision.json", decision)
    print(json.dumps(dict(complete=decision["complete"], eligible=eligible, failures=failures)))


def submit(root, expected):
    plan = previous.common.inputs(root, expected)
    proof = previous.read(root / "cpu.json")
    if (
        proof["source_sha"] != plan["source_sha"]
        or proof["returncode"]
        or proof["failed_or_skipped"]
        or proof["test_count"] < 2826
        or proof["xml_sha256"] != sha(root / "cpu.xml")
    ):
        raise ValueError("full clean CPU proof required")
    queue = subprocess.check_output(
        ["squeue", "-h", "-r", "-u", os.environ["USER"], "-o", "%i|%b"], text=True
    )
    if any("gpu" in line.lower() for line in queue.splitlines()):
        raise ValueError("six-worker study requires no other owned GPU work")
    (root / "slurm").mkdir()
    script = str(SOURCE / "mosaickv/slurm/evicastkv_head_local_precision.sbatch")
    args = [str(SOURCE), str(root), expected]
    gpu = subprocess.check_output(
        [
            "sbatch",
            "--parsable",
            "--partition=bii-gpu",
            "--account=bii_dsc_community",
            "--reservation=bi_fox_dgx",
            "--gres=gpu:a100:1",
            "--cpus-per-task=12",
            "--nodes=1",
            "--ntasks=1",
            "--mem=48G",
            "--time=00:30:00",
            "--array=0-5%6",
            "--job-name=mk-head-local",
            f"--output={root}/slurm/gpu-%A_%a.out",
            f"--error={root}/slurm/gpu-%A_%a.err",
            script,
            *args,
            "worker",
        ],
        text=True,
    ).strip()
    cpu = subprocess.check_output(
        [
            "sbatch",
            "--parsable",
            "--partition=bii",
            "--account=bii_dsc_community",
            "--cpus-per-task=2",
            "--mem=8G",
            "--time=00:10:00",
            "--job-name=mk-head-analysis",
            f"--dependency=afterany:{gpu}",
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
    write_once(root / "submission.json", dict(gpu_array=gpu, cpu_analysis=cpu))
    print(json.dumps(dict(gpu_array=gpu, cpu_analysis=cpu)))


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
        submit(args.root, args.plan_sha)
    elif args.mode == "submit":
        submit(args.root, args.plan_sha)
    elif args.mode == "worker":
        worker(args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    else:
        analyze(args.root, args.plan_sha)
