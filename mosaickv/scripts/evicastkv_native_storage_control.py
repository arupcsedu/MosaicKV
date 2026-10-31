#!/usr/bin/env python3
"""Finite consumed-input native-cache and raw-history controls; no new cohort."""

import argparse
import hashlib
import json
import os
import random
import subprocess
from dataclasses import asdict
from pathlib import Path

import evicastkv_request_validation as previous
from evicastkv_campaign import environment

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
PARENT = Path("/scratch/djy8hg/runs/evicastkv/request-validation-20261031-v3-single-pass-r1/canary")
PARENT_SHA = "0802bce676e2703cfdf8a9c867a91f0069569b38005029b0e63c4e07051af3eb"
PARENT_ARRAY = "22208958"
ARMS = [(m, "ordinary_selected_BF16") for m in previous.METHODS] + [
    ("mosaickv_exact", "query3_source_history"),
    ("mosaickv_exact", "query3_raw_history"),
]


def prepare(root):
    if root.exists() or sha(PARENT / "plan.json") != PARENT_SHA:
        raise ValueError("new output root and immutable consumed parent required")
    old = previous.read(PARENT / "plan.json")
    assert old["source_sha"] == "e7d7fcda7bca568fb28325fee4982d12088aa936"
    bound = dict(old["bound_files"])
    bound[str(PARENT / "plan.json")] = PARENT_SHA
    for job in old["jobs"]:
        folder = PARENT / f"workers/{job['index']}"
        result = previous.read(folder / "result.json")
        assert result["status"] == "completed" and not result["failures"]
        bound[str(folder / "result.json")] = sha(folder / "result.json")
        for name, digest in result["outputs"].items():
            assert sha(folder / name) == digest
            bound[str(folder / name)] = digest
    plan = dict(
        old,
        source_sha=clean_source(SOURCE),
        bound_files=bound,
        stage="consumed_native_transport_control",
        arms=ARMS,
        warmups=1,
        timed_repetitions=3,
        automatic_bounded_cohort_after_correctness=False,
        parent=str(PARENT),
        parent_plan_sha256=PARENT_SHA,
        maximum_a100s=6,
        maximum_new_workers=2,
        dependency_all_first_twelve_parent_shards=PARENT_ARRAY,
        strict_transport_logit_tolerance=0.001,
        no_new_benchmark_scoring=True,
        no_fresh_cohort_authorized=True,
    )
    write_once(root / "plan.json", plan)
    print(sha(root / "plan.json"))


def worker(root, expected, index):
    import torch
    from evicastkv_packed_attention_consumed import differences

    from mosaickv.adapters.huggingface.registry import load_hf_adapter
    from mosaickv.config import RunConfig, config_sha256
    from mosaickv.evaluation.native_storage_control import complete_control
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    plan = previous.common.inputs(root, expected)
    job = plan["jobs"][index]
    out = root / f"workers/{index}"
    out.mkdir(parents=True, exist_ok=False)
    result = dict(
        status="failed",
        failures=[],
        source_sha=plan["source_sha"],
        plan_sha256=expected,
        index=index,
    )
    try:
        if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name():
            raise ValueError("one A100 required")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ValueError("registered deterministic CUDA workspace required")
        torch.set_num_threads(12)
        torch.manual_seed(plan["seed"])
        torch.use_deterministic_algorithms(True)
        cfg = RunConfig.from_mapping(previous.resolved_mapping(plan, job))
        assert cfg.model.id == "llava-hf/llava-1.5-7b-hf"
        assert cfg.model.revision == "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
        assert cfg.model.precision.value == "bf16"
        assert cfg.execution.attention_implementation == "eager"
        adapter = load_hf_adapter(
            cfg.model.id,
            revision=cfg.model.revision,
            model_kwargs=_model_kwargs(cfg, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        parent = Path(plan["parent"]) / f"workers/{index}"
        reference = torch.load(parent / "logits.pt", map_location="cpu", weights_only=True)
        manifest = previous.read(parent / "manifest.json")
        manifest.update(
            source_sha=plan["source_sha"],
            plan_sha256=expected,
            core_reference_sha="e7d7fcda7bca568fb28325fee4982d12088aa936",
            environment=environment(),
            gpu=asdict(capture_gpu_environment(torch)),
            measurement_type="consumed_native_storage_transport_diagnostic",
            base_config_sha256=config_sha256(cfg),
            diagnostic_plan=plan,
            no_independent_benchmark_quality_claim=True,
        )
        write_once(out / "manifest.json", manifest)
        payload = previous.read(job["inputs"][0])
        rng = random.Random(plan["seed"] + index)
        hashes, tensors = {}, {}
        with (out / "raw.jsonl").open("x") as stream:
            for trial in range(-plan["warmups"], plan["timed_repetitions"]):
                arms = list(plan["arms"])
                rng.shuffle(arms)
                for method, mode in arms:
                    row = dict(
                        method=method,
                        mode=mode,
                        trial=trial,
                        warmup=trial < 0,
                        sample_id=payload["sample_id"],
                        task=payload["task"],
                        source_sha=plan["source_sha"],
                        plan_sha256=expected,
                        manifest_path=str(out / "manifest.json"),
                        status="failed",
                    )
                    try:
                        logits, record = complete_control(
                            adapter,
                            lambda: previous.load_messages(payload),
                            cfg,
                            method=method,
                            mode=mode,
                            profile_path=plan["profile_path"],
                            ratio=plan["ratio"],
                        )
                        key = (method, mode)
                        digest = hashlib.sha256(logits.contiguous().numpy().tobytes()).hexdigest()
                        if key in hashes and hashes[key] != digest:
                            raise ValueError("repeated request is not bitwise deterministic")
                        hashes[key] = digest
                        reference_precision = (
                            "lossless" if mode == "ordinary_selected_BF16" else "query3"
                        )
                        ref = reference[f"0-{method}-{reference_precision}"]
                        comparison = differences(ref, logits)
                        guard = differences(reference[f"0-{method}-lossless"][:32], logits[:32])
                        selection = record["online_variant"].get("selection", {})
                        catalog = selection.pop("selected_source_positions_by_layer", None)
                        if catalog is not None:
                            selection["source_position_catalog_sha256"] = hashlib.sha256(
                                json.dumps(catalog, separators=(",", ":")).encode()
                            ).hexdigest()
                        row.update(
                            record,
                            status="completed",
                            greedy_ids=logits.argmax(-1).tolist(),
                            logits_sha256=digest,
                            reference_comparison_128=comparison,
                            registered_guard_32=guard,
                            registered_guard_pass=(
                                guard["token_agreement"] == 1
                                and guard["maximum_logit_KL"] <= 0.02
                                and guard["maximum_logit_difference"] <= 1.0
                            ),
                            strict_transport_parity=(
                                comparison["token_agreement"] == 1
                                and comparison["maximum_logit_difference"]
                                <= plan["strict_transport_logit_tolerance"]
                            ),
                            config_sha256=hashlib.sha256(
                                json.dumps(
                                    dict(
                                        base=config_sha256(cfg),
                                        mode=mode,
                                        method=method,
                                        plan_sha256=expected,
                                    ),
                                    sort_keys=True,
                                ).encode()
                            ).hexdigest(),
                        )
                        if trial == 0:
                            tensors[f"{method}:{mode}"] = logits
                        del logits
                    except Exception as error:
                        row["error"] = f"{type(error).__name__}: {error}"
                        result["failures"].append(
                            dict(method=method, mode=mode, trial=trial, error=row["error"])
                        )
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
        torch.save(tensors, out / "logits.pt")
        result["status"] = "completed" if not result["failures"] else "failed"
    except Exception as error:
        result["failures"].append(dict(error=f"{type(error).__name__}: {error}"))
    finally:
        result["outputs"] = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
        write_once(out / "result.json", result)


def analyze(root, expected):
    import numpy as np

    plan = previous.common.inputs(root, expected)
    rows, failures, bound = [], [], {}
    for job in plan["jobs"]:
        folder = root / f"workers/{job['index']}"
        if not (folder / "result.json").exists():
            failures.append(dict(index=job["index"], error="missing worker result"))
            continue
        result = previous.read(folder / "result.json")
        assert result["source_sha"] == plan["source_sha"] and result["plan_sha256"] == expected
        failures.extend(result["failures"])
        bound[str(folder / "result.json")] = sha(folder / "result.json")
        for name, digest in result["outputs"].items():
            assert sha(folder / name) == digest
        rows.extend(json.loads(line) for line in (folder / "raw.jsonl").read_text().splitlines())
    scored = [r for r in rows if not r["warmup"]]
    summaries = []
    for method, mode in plan["arms"]:
        for sample in {r["sample_id"] for r in scored}:
            group = [
                r
                for r in scored
                if (r["method"], r["mode"], r["sample_id"]) == (method, mode, sample)
            ]
            ok = len(group) == 3 and all(r["status"] == "completed" for r in group)
            summaries.append(
                dict(
                    method=method,
                    mode=mode,
                    sample_id=sample,
                    complete=ok,
                    strict_transport_parity=ok and all(r["strict_transport_parity"] for r in group),
                    registered_guard_pass=ok and all(r["registered_guard_pass"] for r in group),
                    p50_e2e_seconds=float(np.median([r["end_to_end_seconds"] for r in group]))
                    if ok
                    else None,
                    initial_bytes=group[0]["storage"]["initial_active_kv_storage_bytes"]
                    if ok
                    else None,
                    final_bytes=group[0]["storage"]["final_active_kv_storage_bytes"]
                    if ok
                    else None,
                    whole_request_allocated_peak=group[0]["whole_request_peak_allocated_bytes"]
                    if ok
                    else None,
                )
            )
    write_once(
        root / "analysis/decision.json",
        dict(
            source_sha=plan["source_sha"],
            plan_sha256=expected,
            input_hashes=bound,
            failures=failures,
            scored_rows=len(scored),
            expected_rows=42,
            summaries=summaries,
            same_two_consumed_inputs=True,
            statistically_supported_superiority=False,
            no_fresh_cohort_or_paper_promotion=True,
            complete=len(scored) == 42 and not failures and all(s["complete"] for s in summaries),
            purpose=(
                "Distinguish storage reconstruction overhead from ordinary HF "
                "selected-cache latency."
            ),
        ),
    )
    print(json.dumps(dict(scored_rows=len(scored), failures=failures, summaries=summaries)))


def submit(root, expected):
    plan = previous.common.inputs(root, expected)
    proof = previous.read(root / "cpu.json")
    if (
        proof["returncode"]
        or proof["failed_or_skipped"]
        or proof["source_sha"] != plan["source_sha"]
    ):
        raise ValueError("clean full CPU proof required")
    assert proof["xml_sha256"] == sha(root / "cpu.xml") and proof["test_count"] >= 2798
    (root / "slurm").mkdir()
    script = str(SOURCE / "mosaickv/slurm/evicastkv_native_storage_control.sbatch")
    args = [str(SOURCE), str(root), expected]
    queue = subprocess.check_output(
        ["squeue", "-h", "-r", "-u", os.environ["USER"], "-o", "%i|%b"], text=True
    )
    gpu_rows = [line for line in queue.splitlines() if "a100" in line.lower()]
    if any(not line.split("|")[0].startswith(PARENT_ARRAY + "_") for line in gpu_rows):
        raise ValueError("unrelated owned A100 jobs prevent this bounded two-worker admission")
    remaining = [
        line.split("|")[0] for line in gpu_rows if int(line.split("|")[0].split("_")[-1]) < 12
    ]
    if not remaining and len(gpu_rows) > 4:
        raise ValueError("parent did not leave its promised four-worker maximum")
    # Completed array children may already have left slurmctld. Bind only live
    # prerequisites, never stale completed IDs that create invalid dependencies.
    dependencies = ["--dependency=afterany:" + ":".join(remaining)] if remaining else []
    write_once(
        root / "scheduler/eligibility.json",
        dict(
            owned_A100_rows=gpu_rows,
            remaining_first_twelve=remaining,
            maximum_new_workers=2,
            parent_remaining_at_most_four_after_dependencies=True,
        ),
    )
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
            "--array=0-1%2",
            "--job-name=mk-native-control",
            *dependencies,
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
            "--job-name=mk-native-analyze",
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
        per_lane_static_throttle=2,
        other_lane_reserved_slots=4,
    )
    write_once(root / "scheduler/policy.json", policy)
    write_once(
        root / "scheduler/registration.json", tick(policy, {}, root / "scheduler/state.json", False)
    )
    write_once(root / "submission.json", dict(gpu_array=gpu, cpu_analysis=cpu_job))
    print(json.dumps(dict(gpu_array=gpu, cpu_analysis=cpu_job)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "cpu", "worker", "submit", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.root)
    elif args.mode == "cpu":
        previous.cpu(args.root, args.plan_sha)
        submit(args.root, args.plan_sha)
    elif args.mode == "worker":
        worker(args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    elif args.mode == "submit":
        submit(args.root, args.plan_sha)
    else:
        analyze(args.root, args.plan_sha)
