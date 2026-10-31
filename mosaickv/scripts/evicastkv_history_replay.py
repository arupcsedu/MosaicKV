#!/usr/bin/env python3
"""Storage-only replay on existing inputs. Never increases independent quality N."""

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path

import evicastkv_request_validation as study
from evicastkv_campaign import environment

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
PARENT = Path("/scratch/djy8hg/runs/evicastkv/request-validation-20261031-v3-single-pass-r1/cohort")
CONTROL = Path("/scratch/djy8hg/runs/evicastkv/native-history-consumed-20261031-v1")
PARENT_ARRAY = "22208958"
PARENT_ANALYSIS = "22208959"


def identical_predictions(left, right):
    a, b = {r["sample_id"]: r for r in left}, {r["sample_id"]: r for r in right}
    if len(a) != len(left) or len(b) != len(right) or set(a) != set(b):
        raise ValueError("complete unique identical input identities required")
    return dict(
        sample_count=len(a),
        changed_greedy_sequences=sum(a[i]["greedy_ids"] != b[i]["greedy_ids"] for i in a),
        changed_scored_answers=sum(
            a[i]["scored_answer"]["answer"] != b[i]["scored_answer"]["answer"] for i in a
        ),
        changed_logit_hashes=sum(a[i]["logits_sha256"] != b[i]["logits_sha256"] for i in a),
    )


def prepare(root):
    if root.exists():
        raise ValueError("new write-once replay root required")
    old = study.read(PARENT / "plan.json")
    assert old["source_sha"] == "e7d7fcda7bca568fb28325fee4982d12088aa936"
    assert len(old["jobs"]) == 16 and sum(len(j["inputs"]) for j in old["jobs"]) == 128
    decision = study.read(CONTROL / "analysis/decision.json")
    assert decision["complete"] and not decision["failures"]
    raw = [r for r in decision["summaries"] if r["mode"] == "query3_raw_history"]
    refs = [r for r in decision["summaries"] if r["mode"] == "query3_source_history"]
    assert len(raw) == len(refs) == 2
    assert all(r["strict_transport_parity"] and r["registered_guard_pass"] for r in raw)
    assert all(
        r["p50_e2e_seconds"]
        < next(b["p50_e2e_seconds"] for b in refs if b["sample_id"] == r["sample_id"])
        for r in raw
    )
    # The candidate and selector code is unchanged; only the tested lossless
    # history-storage path differs. Do not derive anything from COCO answers.
    for file, commit in (
        ("src/mosaickv/evaluation/request_precision.py", old["source_sha"]),
        ("src/mosaickv/evaluation/native_storage_control.py", decision["source_sha"]),
    ):
        blob = subprocess.check_output(["git", "show", f"{commit}:mosaickv/{file}"], cwd=SOURCE)
        assert sha(SOURCE / "mosaickv" / file) == hashlib.sha256(blob).hexdigest()
    bound = dict(old["bound_files"])
    for p in (PARENT / "plan.json", CONTROL / "analysis/decision.json"):
        bound[str(p)] = sha(p)
    plan = dict(
        old,
        source_sha=clean_source(SOURCE),
        bound_files=bound,
        stage="same_128_inputs_raw_history_replay",
        arms=[["mosaickv_exact", "query3_raw_history"]],
        warmups=5,
        timed_repetitions=1,
        automatic_bounded_cohort_after_correctness=False,
        reference_source_sha=old["source_sha"],
        reference_root=str(PARENT),
        no_new_independent_quality_samples=True,
        no_answer_based_selection_change=True,
        parent_gpu_array=PARENT_ARRAY,
        parent_cpu_analysis=PARENT_ANALYSIS,
    )
    write_once(root / "plan.json", plan)
    print(sha(root / "plan.json"))


def worker(root, expected, index):
    import torch

    from mosaickv.adapters.huggingface.registry import load_hf_adapter
    from mosaickv.config import RunConfig, config_sha256
    from mosaickv.evaluation.native_storage_control import complete_control
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    plan = study.common.inputs(root, expected)
    job = plan["jobs"][index]
    out = root / f"workers/{index}"
    out.mkdir(parents=True, exist_ok=False)
    result = dict(source_sha=plan["source_sha"], plan_sha256=expected, status="failed", failures=[])
    try:
        assert torch.cuda.device_count() == 1 and "A100" in torch.cuda.get_device_name()
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
        torch.set_num_threads(12)
        torch.manual_seed(plan["seed"])
        torch.use_deterministic_algorithms(True)
        cfg = RunConfig.from_mapping(study.resolved_mapping(plan, job))
        assert cfg.model.id == "llava-hf/llava-1.5-7b-hf"
        assert cfg.model.revision == "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
        assert (
            cfg.model.precision.value == "bf16"
            and cfg.execution.attention_implementation == "eager"
        )
        adapter = load_hf_adapter(
            cfg.model.id,
            revision=cfg.model.revision,
            model_kwargs=_model_kwargs(cfg, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        parent = Path(plan["reference_root"]) / f"workers/{index}"
        manifest = study.read(parent / "manifest.json")
        manifest.update(
            source_sha=plan["source_sha"],
            plan_sha256=expected,
            environment=environment(),
            gpu=asdict(capture_gpu_environment(torch)),
            measurement_type="same_input_lossless_history_storage_replay",
            storage_mode="query3_raw_history",
            base_config_sha256=config_sha256(cfg),
            no_new_independent_quality_samples=True,
        )
        write_once(out / "manifest.json", manifest)
        with (out / "raw.jsonl").open("x") as stream:
            for local, path in enumerate(job["inputs"]):
                payload = study.read(path)
                digest_before = None
                for trial in range(-plan["warmups"] if local == 0 else 0, 1):
                    row = dict(
                        sample_id=payload["sample_id"],
                        task=payload["task"],
                        method="mosaickv_exact",
                        mode="query3_raw_history",
                        trial=trial,
                        warmup=trial < 0,
                        source_sha=plan["source_sha"],
                        plan_sha256=expected,
                        manifest_path=str(out / "manifest.json"),
                        status="failed",
                    )
                    try:
                        logits, record = complete_control(
                            adapter,
                            lambda payload=payload: study.load_messages(payload),
                            cfg,
                            method="mosaickv_exact",
                            profile_path=plan["profile_path"],
                            ratio=plan["ratio"],
                            mode="query3_raw_history",
                        )
                        digest = hashlib.sha256(logits.contiguous().numpy().tobytes()).hexdigest()
                        if digest_before is not None and digest_before != digest:
                            raise ValueError("warm and scored requests are not deterministic")
                        digest_before = digest
                        row.update(
                            record,
                            status="completed",
                            logits_sha256=digest,
                            greedy_ids=logits.argmax(-1).tolist(),
                            config_sha256=hashlib.sha256(
                                json.dumps(
                                    dict(
                                        base=config_sha256(cfg),
                                        mode="query3_raw_history",
                                        plan_sha256=expected,
                                    ),
                                    sort_keys=True,
                                ).encode()
                            ).hexdigest(),
                        )
                        del logits
                    except Exception as error:
                        row["error"] = f"{type(error).__name__}: {error}"
                        result["failures"].append(
                            dict(sample_id=payload["sample_id"], error=row["error"])
                        )
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
        result["status"] = "completed" if not result["failures"] else "failed"
    except Exception as error:
        result["failures"].append(dict(error=f"{type(error).__name__}: {error}"))
    finally:
        result["outputs"] = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
        write_once(out / "result.json", result)


def collect(root, plan, expected):
    rows, failures, bound = [], [], {}
    for job in plan["jobs"]:
        folder = root / f"workers/{job['index']}"
        path = folder / "result.json"
        if not path.exists():
            failures.append(dict(index=job["index"], error="missing result"))
            continue
        record = study.read(path)
        assert record["source_sha"] == plan["source_sha"] and record["plan_sha256"] == expected
        failures.extend(record["failures"])
        bound[str(path)] = sha(path)
        for name, digest in record["outputs"].items():
            assert sha(folder / name) == digest
        rows.extend(json.loads(line) for line in (folder / "raw.jsonl").read_text().splitlines())
    return [r for r in rows if not r["warmup"]], failures, bound


def analyze(root, expected):
    from mosaickv.evaluation.request_statistics import compare_requests

    plan = study.common.inputs(root, expected)
    parent = Path(plan["reference_root"])
    old_plan = study.read(parent / "plan.json")
    old, old_failures, old_bound = collect(parent, old_plan, sha(parent / "plan.json"))
    new, failures, bound = collect(root, plan, expected)
    failures.extend(old_failures)
    gate = dict(
        source_sha=plan["source_sha"],
        plan_sha256=expected,
        input_hashes=dict(old_bound, **bound),
        failures=failures,
        complete=False,
        independent_quality_N=128,
        additional_quality_N=0,
        superiority_established=False,
        no_paper_promotion=True,
        ordinary_BF16_baseline_population_latency_unverified=True,
    )
    anchor = [r for r in old if r["method"] == "mosaickv_exact" and r["precision"] == "query3"]
    if (
        len(new) == len(anchor) == 128
        and not failures
        and all(r["status"] == "completed" for r in new)
    ):
        identity = identical_predictions(anchor, new)
        gate["prediction_identity"] = identity
        equivalent = identity["changed_greedy_sequences"] == identity["changed_scored_answers"] == 0
        gate["complete"] = equivalent
        if equivalent:
            quality_path = parent / "analysis/quality_comparisons.json"
            if not quality_path.exists():
                gate.update(complete=False, error="parent statistics unavailable; no metric reuse")
                write_once(root / "analysis/decision.json", gate)
                return
            quality = study.read(quality_path)
            gate["quality_artifact_sha256"] = sha(quality_path)
            gate["corpus_CIDEr_and_bootstrap_unchanged_by_exact_prediction_identity"] = True
            gate["same_method_storage_comparison"] = compare_requests(
                new, anchor, resamples=10000, seed=plan["seed"]
            )
            comparisons = []
            for q in quality["comparisons"]:
                method, precision = q["baseline"].split(":")
                reference = [r for r in old if (r["method"], r["precision"]) == (method, precision)]
                comparisons.append(
                    dict(
                        baseline=q["baseline"],
                        cider_difference=q["cider_difference"],
                        simultaneous_ci95_bonferroni=q["simultaneous_ci95_bonferroni"],
                        strict_quality_pass=q["strict_quality_pass"],
                        paired_execution=compare_requests(
                            new, reference, resamples=10000, seed=plan["seed"]
                        ),
                    )
                )
            gate["encoded_point_comparisons"] = comparisons
    write_once(root / "analysis/decision.json", gate)
    print(
        json.dumps(
            {
                k: v
                for k, v in gate.items()
                if k
                not in (
                    "input_hashes",
                    "encoded_point_comparisons",
                    "same_method_storage_comparison",
                )
            }
        )
    )


def submit(root, expected):
    from control_a100_schedule import tick

    plan = study.common.inputs(root, expected)
    proof = study.read(root / "cpu.json")
    assert proof["returncode"] == proof["failed_or_skipped"] == 0
    assert proof["source_sha"] == plan["source_sha"] and proof["xml_sha256"] == sha(
        root / "cpu.xml"
    )
    queue = subprocess.check_output(
        ["squeue", "-h", "-r", "-u", os.environ["USER"], "-o", "%i|%b"], text=True
    )
    rows = [r for r in queue.splitlines() if "a100" in r.lower()]
    if len(rows) > 4 or any(not r.split("|")[0].startswith(PARENT_ARRAY + "_") for r in rows):
        raise ValueError("at most four parent GPU workers may coexist with initial replay")
    assert all((PARENT / f"workers/{i}/result.json").exists() for i in range(12))
    (root / "slurm").mkdir()
    script = str(SOURCE / "mosaickv/slurm/evicastkv_history_replay.sbatch")
    base = [
        "sbatch",
        "--parsable",
        "--account=bii_dsc_community",
        "--partition=bii-gpu",
        "--reservation=bi_fox_dgx",
        "--gres=gpu:a100:1",
        "--cpus-per-task=12",
        "--nodes=1",
        "--ntasks=1",
        "--mem=48G",
        "--time=00:30:00",
        "--job-name=mk-history-replay",
        f"--output={root}/slurm/gpu-%A_%a.out",
        f"--error={root}/slurm/gpu-%A_%a.err",
    ]
    args = [str(SOURCE), str(root), expected, "worker"]
    first = subprocess.check_output(
        [*base, "--array=0-11%2", script, *args, "0"], text=True
    ).strip()
    tail = subprocess.check_output(
        [
            *base,
            "--array=0-3%4",
            f"--dependency=afterany:{first}:{PARENT_ARRAY}",
            script,
            *args,
            "12",
        ],
        text=True,
    ).strip()
    cpu_job = subprocess.check_output(
        [
            "sbatch",
            "--parsable",
            "--account=bii_dsc_community",
            "--partition=bii",
            "--cpus-per-task=2",
            "--mem=8G",
            "--time=00:10:00",
            "--job-name=mk-history-analysis",
            f"--dependency=afterany:{first}:{tail}:{PARENT_ANALYSIS}",
            f"--output={root}/slurm/analysis-%j.out",
            f"--error={root}/slurm/analysis-%j.err",
            script,
            str(SOURCE),
            str(root),
            expected,
            "analyze",
        ],
        text=True,
    ).strip()
    for name, array, limit, other in [("first", first, 2, 4), ("tail", tail, 4, 2)]:
        policy = dict(
            timezone="America/New_York",
            maximum_worker_minutes=30,
            maximum_a100s=6,
            six_a100s_explicitly_authorized=True,
            arrays=[array],
            per_lane_static_throttle=limit,
            other_lane_reserved_slots=other,
        )
        write_once(root / f"scheduler/{name}/policy.json", policy)
        write_once(
            root / f"scheduler/{name}/registration.json",
            tick(policy, {}, root / f"scheduler/{name}/state.json", False),
        )
    write_once(
        root / "submission.json",
        dict(
            first_array=first,
            tail_array=tail,
            cpu_analysis=cpu_job,
            aggregate_gpu_bound=(
                "first(2)+parent(4), then tail(4)+parent-timing(2); arrays sequential"
            ),
        ),
    )
    print(json.dumps(study.read(root / "submission.json")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "cpu", "worker", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha")
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.root)
    elif args.mode == "cpu":
        study.cpu(args.root, args.plan_sha)
        submit(args.root, args.plan_sha)
    elif args.mode == "worker":
        worker(args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]) + args.offset)
    else:
        analyze(args.root, args.plan_sha)
