#!/usr/bin/env python3
"""Consumed-input actual-query fused attention guard and retrospective audit.

No inference latency/memory claim: parent eager oracle, full K/V cloning and
lossless packing remain explicit diagnostic work. No fresh data or quality run.
"""

import argparse
import json
import os
import subprocess
import traceback
from pathlib import Path

import evicastkv_full_prompt_native as driver
import evicastkv_hybrid_latency as latency

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = driver.SOURCE
PARENT = Path("/scratch/djy8hg/runs/evicastkv/hybrid-latency-consumed-20261103-v2/hybrid-study")
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_fused_model_guard.sbatch"


def submit(root, action, *, dependency=None, gpu=False):
    expected = sha(root / "plan.json")
    (root / "slurm").mkdir(exist_ok=True)
    args = ["sbatch", "--parsable", "--account=bii_dsc_community", "--nodes=1", "--ntasks=1"]
    if gpu:
        driver.cpu_proof(root, driver.checked(root, expected))
        args += [
            "--partition=bii-gpu",
            "--reservation=bi_fox_dgx",
            "--gres=gpu:a100:1",
            "--cpus-per-task=12",
            "--mem=48G",
            "--time=00:30:00",
            "--array=0-1%2",
        ]
    else:
        args += ["--partition=bii,standard", "--cpus-per-task=2", "--mem=16G", "--time=00:15:00"]
    if dependency:
        args.append(f"--dependency=afterany:{dependency}")
    args += [
        f"--job-name=mk-fused-guard-{action}",
        f"--output={root}/slurm/{action}-%A_%a.out",
        f"--error={root}/slurm/{action}-%A_%a.err",
        str(SCRIPT),
        str(SOURCE),
        str(root),
        expected,
        action,
    ]
    job = subprocess.check_output(args, text=True).strip()
    if gpu:
        from control_a100_schedule import tick

        policy = dict(
            timezone="America/New_York",
            maximum_worker_minutes=30,
            maximum_a100s=6,
            six_a100s_explicitly_authorized=True,
            arrays=[job],
        )
        write_once(root / "scheduler/policy.json", policy)
        write_once(
            root / "scheduler/registration.json",
            tick(policy, {}, root / "scheduler/state.json", False),
        )
    return job


def prepare(root, revision):
    if root.exists() or clean_source(SOURCE) != revision:
        raise ValueError("new immutable root and clean source required")
    old = driver.read(PARENT / "plan.json")
    decision = driver.read(PARENT / "analysis/decision.json")
    if (
        not decision["complete"]
        or decision["failures"]
        or len(decision["studies"]) != 2
        or not all(r["fusion"]["operator_gate"] for r in decision["studies"])
    ):
        raise ValueError("both actual-packet operator bitwise controls must pass first")
    bound = dict(old["bound_files"])
    for path in (
        PARENT / "plan.json",
        PARENT / "analysis/decision.json",
        SOURCE / "mosaickv/docs/fused_model_guard_20261103.md",
    ):
        bound[str(path)] = sha(path)
    for index in (0, 1):
        folder = PARENT / f"workers/{index}"
        record = driver.read(folder / "result.json")
        if record["status"] != "completed" or record["failures"]:
            raise ValueError("complete parent producer required")
        bound[str(folder / "result.json")] = sha(folder / "result.json")
        for name, expected in record["outputs"].items():
            if sha(folder / name) != expected:
                raise ValueError("parent operator/control artifact changed")
            bound[str(folder / name)] = expected
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=revision,
            bound_files=bound,
            stage="consumed_actual_query_fused_model_guard",
            chain_root=str(root),
            guard_layers=[0, 16, 31],
            guard_prefixes=32,
            guard_modes=["clone_repack", "fused"],
            all_word_bits_required=True,
            no_parent_fallback_for_failed_operator=True,
            complete_model_predictions=128,
            ABA_prefetch_request_indices=[0, 1, 0],
            ABA_only_worker=0,
            reverify_existing_arms_using_uint8_word_equality=True,
            no_serving_or_native_backend_promotion=True,
            no_latency_or_memory_efficiency_claim=True,
            automatic_full_request_submission=False,
            automatic_quality_submission=False,
            fresh_materialization_authorized=False,
            paper_campaign_authorized=False,
            retrospective_latency_root=str(PARENT / "paired-timing"),
            retrospective_family_maximum_comparisons=48,
        ),
    )
    write_once(root / "preparation_submission.json", dict(cpu=submit(root, "cpu")))


def cpu(root, expected):
    driver.pair.common.previous.cpu(root, expected)
    driver.cpu_proof(root, driver.checked(root, expected))
    gpu = submit(root, "guard", gpu=True)
    # The functional result is analyzed promptly; latency data may finish later.
    cpu_job = submit(root, "analyze", dependency=gpu)
    write_once(root / "submission.json", dict(gpu_array=gpu, analysis=cpu_job))


def worker(root, expected, index):
    import torch

    from mosaickv.evaluation.fused_model_guard import Guard, GuardFailure

    out = root / f"workers/{index}"
    out.mkdir(parents=True, exist_ok=False)
    result = dict(
        status="failed", failures=[], source_sha=clean_source(SOURCE), plan_sha256=expected
    )
    try:
        plan, adapter, refs, messages, profile, manifest = driver.setup(
            root,
            expected,
            index,
            manifest_overrides=dict(
                variant="consumed_actual_query_fused_numerical_substitution_guard",
                methods=["mosaickv_exact", "prefixkv_reimpl"],
                actual_timing_protocol=dict(
                    generation_predictions=128,
                    guarded_prefixes=32,
                    guarded_layers=[0, 16, 31],
                    no_latency_benchmark=True,
                    full_KV_parent_oracle_materialization_explicit=True,
                ),
            ),
        )
        word_controls = []
        for method in latency.METHODS:
            old_arms = driver.read(PARENT / f"workers/{index}/controls-{method}.json")["arms"]
            for name, record in latency.validated_arms(old_arms).items():
                measured = latency.execute(
                    adapter,
                    refs,
                    messages,
                    profile,
                    plan,
                    method,
                    policy=record["policy"],
                    reuse=record["reuse"],
                    prefetch=name == "prefetch",
                )
                if any(measured[k] != record["audited"][k] for k in driver.STORAGE[:2]):
                    raise RuntimeError("word revalidation changed an audited storage boundary")
                word_controls.append(
                    dict(method=method, arm=name, word_exact=True, controls=measured)
                )
        write_once(out / "word-controls.json", dict(records=word_controls))
        modes = []
        for mode in ("clone_repack", "fused"):
            diagnostic = Guard(adapter, mode=mode)
            record = dict(mode=mode, status="failed")
            try:
                with diagnostic.use():
                    measured = latency.execute(
                        adapter,
                        refs,
                        messages,
                        profile,
                        plan,
                        "mosaickv_exact",
                        collector=lambda _p: None,
                    )
                record.update(status="passed", controls=measured)
            except GuardFailure as error:
                record.update(status="rejected_numerical", error=str(error))
                torch.save(diagnostic.failed_tensors, out / f"failed-{mode}.pt")
            record["guard"] = diagnostic.report()
            if record["status"] == "passed" and not record["guard"]["complete_32_prefix_guard"]:
                raise RuntimeError(
                    "successful request did not execute all 96 guarded substitutions"
                )
            modes.append(record)
            write_once(out / f"guard-{mode}.json", record)
            with (out / "guard-raw.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        dict(
                            record,
                            task=manifest["task"],
                            sample_id=manifest["sample_id"],
                            source_sha=plan["source_sha"],
                            plan_sha256=expected,
                            manifest_path=str(out / "manifest.json"),
                            measurement_type="diagnostic_measured",
                        ),
                        allow_nan=False,
                    )
                    + "\n"
                )
            if mode == "clone_repack" and record["status"] != "passed":
                raise RuntimeError(
                    "clone/repack control failed; fused substitution is not interpretable"
                )
        aba = []
        if index == 0:
            for position, input_index in enumerate((0, 1, 0)):
                job = plan["jobs"][input_index]
                payload = driver.read(job["inputs"][0])
                other_refs = driver.pair.references(job["input_index"])
                other_messages = driver.pair.common.previous.load_messages(payload)
                measured = latency.execute(
                    adapter,
                    other_refs,
                    other_messages,
                    profile,
                    plan,
                    "mosaickv_exact",
                    reuse=True,
                    prefetch=True,
                    collector=lambda _p: None,
                )
                aba.append(
                    dict(
                        position=position,
                        input_index=input_index,
                        sample_id=payload["sample_id"],
                        task=payload["task"],
                        dataset=job["config"]["dataset"],
                        input_path=job["inputs"][0],
                        input_sha256=sha(Path(job["inputs"][0])),
                        actual_generated_token_ids=measured["generated_token_ids"],
                        word_exact_to_own_parent=True,
                        prefetch_hits=measured["prefetch_hits"],
                    )
                )
            if aba[0]["actual_generated_token_ids"] != aba[2]["actual_generated_token_ids"]:
                raise RuntimeError("A-B-A changed A generation")
        write_once(
            out / "guard.json",
            dict(
                modes=modes,
                word_revalidations=word_controls,
                ABA=aba,
                independent_quality_N=0,
                no_native_or_serving_backend_row=True,
                parent_oracle_and_full_clones_are_not_a_latency_or_memory_optimized_path=True,
                manifest_path=str(out / "manifest.json"),
                task=manifest["task"],
                sample_id=manifest["sample_id"],
            ),
        )
        result["status"] = "completed"
    except Exception as error:
        result["failures"].append(f"{type(error).__name__}: {error}")
        result["traceback"] = traceback.format_exc()
    driver.finish_worker(out, result)


def analyze(root, expected):
    plan = driver.checked(root, expected)
    rows, failures = driver.load_producers(root, plan, "guard.json")
    fused_pass = bool(
        not failures
        and len(rows) == 2
        and all(
            next(r for r in row["modes"] if r["mode"] == "fused")["status"] == "passed"
            for row in rows
        )
    )
    write_once(
        root / "analysis/decision.json",
        dict(
            complete=not failures,
            failures=failures,
            actual_query_32_prefix_fused_guard=fused_pass,
            summaries=rows,
            independent_quality_N=0,
            paper_superiority_established=False,
            no_reader_promoted=True,
            no_native_or_production_backend_claim=True,
            no_latency_or_memory_optimization_claim_from_oracle_diagnostic=True,
        ),
    )
    latency_root = Path(plan["retrospective_latency_root"])
    if (latency_root / "submission.json").exists():
        old_submission = driver.read(latency_root / "submission.json")
        job = submit(root, "latency_audit", dependency=old_submission["analysis"])
        write_once(root / "latency_audit_submission.json", dict(cpu=job))


def latency_audit(root, expected):
    from mosaickv.evaluation.latency_strongest_control import analyze as compute

    plan = driver.checked(root, expected)
    source = Path(plan["retrospective_latency_root"])
    old = driver.read(source / "plan.json")
    _rows, failures = driver.load_producers(source, old, "timing.json")
    reports, provenance = [], {}
    if not failures:
        for index in (0, 1):
            path = source / f"workers/{index}/trials.jsonl"
            data = [json.loads(line) for line in path.read_text().splitlines()]
            reports.append(compute(data, seed=plan["seed"]))
            provenance[str(path)] = sha(path)
    write_once(
        root / "analysis/strongest_latency_retrospective.json",
        dict(
            complete=not failures,
            failures=failures,
            retrospective=True,
            reports=reports,
            raw_input_hashes=provenance,
            producer_source_sha=old["source_sha"],
            evaluator_source_sha=plan["source_sha"],
            earlier_registered_gate_not_overwritten=True,
            independent_quality_N=0,
            paper_superiority_established=False,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "cpu", "guard", "analyze", "latency_audit"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-sha")
    parser.add_argument("--plan-sha")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root, args.source_sha)
    elif args.action == "guard":
        worker(args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    else:
        globals()[args.action](args.root, args.plan_sha)


if __name__ == "__main__":
    main()
