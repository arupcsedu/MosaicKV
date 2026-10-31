#!/usr/bin/env python3
"""Two consumed inputs only: separate LOOK-M selection, key merge, value merge."""

import argparse
import hashlib
import json
import os
import random
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import evicastkv_native_storage_control as parent
import evicastkv_request_validation as previous
from evicastkv_campaign import environment

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once
from mosaickv.evaluation.lookm_merge_ablation import ARMS, comparison

SOURCE = Path(__file__).resolve().parents[2]


def prepare(root):
    if root.exists() or sha(parent.PARENT / "plan.json") != parent.PARENT_SHA:
        raise ValueError("new root and unchanged consumed-parent plan required")
    old = previous.read(parent.PARENT / "plan.json")
    bound = dict(old["bound_files"])
    bound[str(parent.PARENT / "plan.json")] = parent.PARENT_SHA
    for job in old["jobs"]:
        folder = parent.PARENT / f"workers/{job['index']}"
        result = previous.read(folder / "result.json")
        if result["status"] != "completed" or result["failures"]:
            raise ValueError("consumed reference must be complete")
        bound[str(folder / "result.json")] = sha(folder / "result.json")
        for name, digest in result["outputs"].items():
            if sha(folder / name) != digest:
                raise ValueError("reference artifact changed")
            bound[str(folder / name)] = digest
    protocol = SOURCE / "mosaickv/docs/lookm_selection_merge_consumed_20261031.md"
    bound[str(protocol)] = sha(protocol)
    plan = dict(
        old,
        source_sha=clean_source(SOURCE),
        bound_files=bound,
        parent=str(parent.PARENT),
        stage="lookm_selection_versus_merging_consumed_causal_diagnostic",
        arms=ARMS,
        teacher_forced_cache_dependent_prefixes=32,
        greedy_predictions=128,
        warmups=0,
        timed_repetitions=1,
        no_performance_claim=True,
        strict_transport_logit_tolerance=0.001,
        maximum_new_workers=2,
        maximum_a100s=4,
        automatic_bounded_cohort_after_correctness=False,
        paper_campaign_authorized=False,
        fresh_cohort_authorized=False,
        no_benchmark_labels_or_references_loaded=True,
        full_source_retained_for_diagnostic=True,
        clone_repack_control=True,
        result_category="unified_comparison",
        result_origin="locally_measured",
    )
    write_once(root / "plan.json", plan)
    print(sha(root / "plan.json"))


def tensor_hash(pairs):
    digest = hashlib.sha256()
    for pair in pairs:
        for tensor in pair:
            digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
            digest.update(
                tensor.detach().contiguous().cpu().view(__import__("torch").uint8).numpy()
            )
    return digest.hexdigest()


def worker(root, expected, index):
    import torch

    from mosaickv.adapters.huggingface.registry import load_hf_adapter
    from mosaickv.config import RunConfig, config_sha256
    from mosaickv.evaluation import request_precision, storage_baselines
    from mosaickv.evaluation.lookm_merge_ablation import decode, make_payload
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
            raise ValueError("registered deterministic workspace required")
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
            raise ValueError("consumed model/precision/backend must not change")
        adapter = load_hf_adapter(
            cfg.model.id,
            revision=cfg.model.revision,
            model_kwargs=_model_kwargs(cfg, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        payload = previous.read(job["inputs"][0])
        manifest = previous.read(Path(plan["parent"]) / f"workers/{index}/manifest.json")
        manifest.update(
            source_sha=plan["source_sha"],
            plan_sha256=expected,
            environment=environment(),
            gpu=asdict(capture_gpu_environment(torch)),
            base_config_sha256=config_sha256(cfg),
            measurement_type="ablation_measured",
            diagnostic_plan=plan,
            result_category="unified_comparison",
            result_origin="locally_measured",
            no_independent_benchmark_quality_claim=True,
            vllm="not_used",
            sglang="not_used",
        )
        write_once(out / "manifest.json", manifest)
        old = torch.load(
            Path(plan["parent"]) / f"workers/{index}/logits.pt",
            weights_only=True,
            map_location="cpu",
        )
        prepared = adapter.prepare_inputs(previous.load_messages(payload))
        prefill = adapter.prefill(prepared, capture_queries=False, capture_attentions=True)
        source, _ = adapter._legacy_layers(prefill.state.past_key_values)
        before = tensor_hash(source)
        profile = tuple(previous.read(plan["profile_path"]))
        frozen, _ = request_precision.online_frozen(
            prefill, profile, plan["ratio"], method="lookm_reimpl"
        )
        torch.cuda.synchronize()
        started = time.monotonic()
        _, merged, proof = storage_baselines.baseline_payload(
            adapter,
            prepared,
            prefill,
            cfg,
            "lookm_reimpl",
            frozen,
            plan["profile_path"],
            borrow_source=True,
        )
        torch.cuda.synchronize()
        planning_seconds = time.monotonic() - started
        positions = proof["selected_source_positions_by_layer"]
        write_once(out / "selection.json", proof)
        full_positions = tuple(
            tuple(tuple(range(k.shape[2])) for _ in range(k.shape[1])) for k, _ in source
        )
        full, full_storage = decode(
            adapter,
            prefill,
            make_payload(adapter, prefill, source, full_positions, "selection_only"),
            predictions=128,
        )
        transport = comparison(old["0-full_kv-ordinary"], full)
        if transport["token_agreement"] != 1 or transport["maximum_logit_difference"] > 0.001:
            raise ValueError("new FullKV clone does not reproduce consumed ordinary reference")
        forced = tuple(full.argmax(-1)[:32].tolist())
        tensors, rows = {"full_kv_greedy": full}, []
        order = list(ARMS)
        random.Random(plan["seed"] + index).shuffle(order)
        with (out / "raw.jsonl").open("x") as stream:
            for arm in order:
                row = dict(
                    arm=arm,
                    method="lookm_reimpl"
                    if arm == "key_value_merge"
                    else f"lookm_{arm}_ablation_reimpl",
                    sample_id=payload["sample_id"],
                    task=payload["task"],
                    status="failed",
                    source_sha=plan["source_sha"],
                    config_sha256=expected,
                    manifest_path=str(out / "manifest.json"),
                    selection_sha256=sha(out / "selection.json"),
                    measurement_type="ablation_measured",
                    result_category="unified_comparison",
                    result_origin="locally_measured",
                    fixed_same_positions_and_raw_bytes=True,
                )
                try:
                    common, forced_storage = decode(
                        adapter,
                        prefill,
                        make_payload(adapter, prefill, merged, positions, arm),
                        predictions=33,
                        forced_prefix=forced,
                    )
                    greedy, greedy_storage = decode(
                        adapter,
                        prefill,
                        make_payload(adapter, prefill, merged, positions, arm),
                        predictions=128,
                    )
                    tensors[f"{arm}:forced"] = common
                    tensors[f"{arm}:greedy"] = greedy
                    for storage in (forced_storage, greedy_storage):
                        if (
                            storage["initial_active_kv_storage_bytes"]
                            != proof["source_slot_byte_cap"]
                        ):
                            raise ValueError("ablation did not retain identical physical bytes")
                    eos = adapter.model.generation_config.eos_token_id
                    eos_ids = (eos,) if type(eos) is int else tuple(eos or ())
                    ids = greedy.argmax(-1).tolist()
                    end = next((i for i, t in enumerate(ids) if t in eos_ids), len(ids))
                    row.update(
                        status="completed",
                        fixed_prefix_comparison=comparison(full[1:33], common[1:]),
                        greedy_comparison_contexts_may_differ=comparison(full, greedy),
                        own_greedy_ids=ids,
                        answer=adapter.processor.tokenizer.decode(
                            ids[:end], skip_special_tokens=True
                        ),
                        scored_token_count=end,
                        timed_or_generated_predictions=128,
                        forced_storage=forced_storage,
                        greedy_storage=greedy_storage,
                        prefill_logits_excluded_from_registered_guard=True,
                    )
                    if arm == "key_value_merge":
                        row["saved_merged_transport"] = comparison(
                            old["0-lookm_reimpl-lossless"], greedy
                        )
                except Exception as error:
                    row["error"] = f"{type(error).__name__}: {error}"
                    result["failures"].append(dict(arm=arm, error=row["error"]))
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
        repeat, _ = decode(
            adapter,
            prefill,
            make_payload(adapter, prefill, merged, positions, "key_value_merge"),
            predictions=33,
            forced_prefix=forced,
        )
        write_once(
            out / "controls.json",
            dict(
                FullKV_transport=transport,
                full_storage=full_storage,
                merged_clone_repack=comparison(tensors["key_value_merge:forced"], repeat),
                original_source_bitwise_unchanged=before == tensor_hash(source),
                original_source_sha256=before,
                shared_plan_seconds=planning_seconds,
                no_end_to_end_timing_or_whole_request_peak_comparison=True,
                one_prefill_reused=True,
                selection_sha256=sha(out / "selection.json"),
            ),
        )
        torch.save(tensors, out / "logits.pt")
        result["status"] = "completed" if not result["failures"] else "failed"
    except Exception as error:
        result["failures"].append(dict(error=f"{type(error).__name__}: {error}"))
    finally:
        result["outputs"] = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
        write_once(out / "result.json", result)


def analyze(root, expected):
    plan = previous.common.inputs(root, expected)
    failures, summaries, inputs = [], [], {}
    for job in plan["jobs"]:
        folder = root / f"workers/{job['index']}"
        if not (folder / "result.json").exists():
            failures.append(dict(index=job["index"], error="missing producer"))
            continue
        result = previous.read(folder / "result.json")
        if result["source_sha"] != plan["source_sha"] or result["plan_sha256"] != expected:
            raise ValueError("worker provenance changed")
        inputs[str(folder / "result.json")] = sha(folder / "result.json")
        for name, digest in result["outputs"].items():
            if sha(folder / name) != digest:
                raise ValueError("worker raw artifact changed")
        failures.extend(result["failures"])
        if result["status"] != "completed":
            continue
        controls = previous.read(folder / "controls.json")
        rows = [json.loads(line) for line in (folder / "raw.jsonl").read_text().splitlines()]
        if len(rows) != 4 or {r["arm"] for r in rows} != set(ARMS):
            failures.append(dict(index=job["index"], error="incomplete/duplicate factorial arms"))
            continue
        by_arm = {r["arm"]: r for r in rows}
        merged = by_arm["key_value_merge"]["saved_merged_transport"]
        cloned = controls["merged_clone_repack"]
        valid = (
            not any(r["status"] != "completed" for r in rows)
            and controls["original_source_bitwise_unchanged"]
            and cloned["maximum_logit_difference"] == 0
            and merged["token_agreement"] == 1
            and merged["maximum_logit_difference"] <= 0.001
            and len({r["selection_sha256"] for r in rows}) == 1
            and all(
                len({r[phase][field] for r in rows}) == 1
                for phase in ("forced_storage", "greedy_storage")
                for field in (
                    "initial_active_kv_storage_bytes",
                    "final_active_kv_storage_bytes",
                    "peak_active_kv_storage_bytes",
                    "physical_padding_bytes",
                )
            )
        )
        if not valid:
            failures.append(
                dict(index=job["index"], error="transport/storage/isolation control failed")
            )
        summaries.append(
            dict(
                index=job["index"],
                sample_id=rows[0]["sample_id"],
                controls_passed=valid,
                arms=[
                    dict(
                        arm=r["arm"],
                        fixed_prefix=r["fixed_prefix_comparison"],
                        answer=r["answer"],
                        greedy_storage=r["greedy_storage"],
                        greedy_token_agreement_to_FullKV=r["greedy_comparison_contexts_may_differ"][
                            "token_agreement"
                        ],
                    )
                    for r in rows
                ],
                merge_minus_selection_mean_KL=(
                    by_arm["key_value_merge"]["fixed_prefix_comparison"]["mean_KL_FP64"]
                    - by_arm["selection_only"]["fixed_prefix_comparison"]["mean_KL_FP64"]
                ),
            )
        )
    decision = dict(
        source_sha=plan["source_sha"],
        plan_sha256=expected,
        input_hashes=inputs,
        failures=failures,
        summaries=summaries,
        complete=len(summaries) == 2 and not failures,
        consumed_diagnostic_only=True,
        statistically_supported_superiority=False,
        fresh_cohort_and_larger_campaign_blocked=True,
        interpretation=(
            "Merge effects at identical membership, bytes, and FullKV conditioning; "
            "not benchmark quality."
        ),
    )
    write_once(root / "analysis/decision.json", decision)
    print(json.dumps(decision))


def submit(root, expected):
    plan = previous.common.inputs(root, expected)
    proof = previous.read(root / "cpu.json")
    if (
        proof["source_sha"] != plan["source_sha"]
        or proof["returncode"]
        or proof["failed_or_skipped"]
        or proof["test_count"] < 2808
        or proof["xml_sha256"] != sha(root / "cpu.xml")
    ):
        raise ValueError("clean full CPU proof required")
    queue = subprocess.check_output(
        ["squeue", "-h", "-r", "-u", os.environ["USER"], "-o", "%i|%b"], text=True
    )
    if any("gpu" in line.lower() for line in queue.splitlines()):
        raise ValueError("bounded admission requires no other owned GPU jobs")
    (root / "slurm").mkdir()
    script = str(SOURCE / "mosaickv/slurm/evicastkv_lookm_selection_merge.sbatch")
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
            "--time=00:20:00",
            "--array=0-1%2",
            "--job-name=mk-lookm-causal",
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
            "--mem=8G",
            "--time=00:10:00",
            "--job-name=mk-lookm-analysis",
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
        maximum_worker_minutes=20,
        maximum_a100s=4,
        arrays=[gpu],
        per_lane_static_throttle=2,
        other_lane_reserved_slots=2,
    )
    write_once(root / "scheduler/policy.json", policy)
    write_once(
        root / "scheduler/registration.json", tick(policy, {}, root / "scheduler/state.json", False)
    )
    write_once(root / "submission.json", dict(gpu_array=gpu, cpu_analysis=cpu_job))
    print(json.dumps(dict(gpu_array=gpu, cpu_analysis=cpu_job)))


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
