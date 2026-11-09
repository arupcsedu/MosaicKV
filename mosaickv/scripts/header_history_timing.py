#!/usr/bin/env python3
"""Consumed-input clean paired timing; never overrides the negative memory gate."""

import argparse
import json
import os
import subprocess
import traceback
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from time import monotonic

import exact_scale_replay_study as base
import numpy as np
import selective_history_study as history_study

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE, PYTHON = base.SOURCE, base.PYTHON
PARENT = Path("/scratch/djy8hg/runs/evicastkv/selective-history64-20261108-v1")
base.common.WORKER_WRAPPER = SOURCE / "mosaickv/slurm/header_history_timing.sbatch"
base.EXTRA_TESTS = (*base.EXTRA_TESTS, "test_header_history_timing.py")
ARMS = history_study.ARMS


def schedule(seed, warmups=5, repetitions=20):
    rng = np.random.default_rng(seed)
    return [
        dict(warmup=warm, repetition=rep, arm=str(arm))
        for warm, count in ((True, warmups), (False, repetitions))
        for rep in range(count)
        for arm in rng.permutation(ARMS)
    ]


def paired(values, *, seed, replicates=10000, family_comparisons=2):
    a, b = np.asarray(values[0], dtype=float), np.asarray(values[1], dtype=float)
    if a.shape != b.shape or a.size != 20 or not np.isfinite([a, b]).all() or (a <= 0).any():
        raise ValueError("exactly 20 complete finite paired positive trials required")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(a), size=(replicates, len(a)))
    ratios = b[indices].mean(1) / a[indices].mean(1)
    differences = (b[indices] - a[indices]).mean(1)
    if type(family_comparisons) is not int or family_comparisons < 1:
        raise ValueError("positive preregistered comparison family required")
    tail = 0.05 / (2 * family_comparisons)
    bounds = [tail, 1 - tail]
    return dict(
        mean_latency_ratio=float(b.mean() / a.mean()),
        ratio_adjusted_CI=np.quantile(ratios, bounds).tolist(),
        mean_difference_seconds=float((b - a).mean()),
        difference_adjusted_CI=np.quantile(differences, bounds).tolist(),
        control_p50=float(np.median(a)),
        candidate_p50=float(np.median(b)),
        control_p95=float(np.quantile(a, 0.95)),
        candidate_p95=float(np.quantile(b, 0.95)),
        equal_or_faster=bool(np.quantile(ratios, bounds[1]) <= 1.0),
        consumed_input_engineering_only=True,
    )


def prepare(root):
    if root.exists():
        raise FileExistsError("new immutable root required")
    p = base.common.read(PARENT / "plan.json")
    negative = base.common.read(PARENT / "analysis/history-component-decision.json")
    for i in (0, 1):
        path = PARENT / f"model/{i}/raw.jsonl"
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        if len(rows) != 2 or not all(
            r["status"] == "completed"
            and r["numerically_admissible"]
            and r["whole_request_certificate_valid"]
            for r in rows
        ):
            raise ValueError("complete consumed numerical/ownership evidence required")
        p["bound_files"][str(path)] = sha(path)
    for name in ("plan.json", "analysis/history-component-decision.json"):
        path = PARENT / name
        p["bound_files"][str(path)] = sha(path)
    p.update(
        source_sha=clean_source(SOURCE),
        study="common_headers_selective_history_paired_timing_v1",
        measurement_type="consumed_randomized_paired_uninstrumented_timing",
        timing_warmups=5,
        timing_repetitions=20,
        timing_bootstrap_replicates=10000,
        timing_worker_seconds=3300,
        prior_memory_gate_passed=negative["component_gate_passed"],
        prior_negative_memory_gate_NOT_overridden=True,
        fresh_or_paper_campaign_authorized=False,
        independent_quality_validation_authorized=False,
    )
    p["timing_schedules"] = {str(i): schedule(p["seed"] + i) for i in (0, 1)}
    root.mkdir()
    (root / "logs").mkdir()
    write_once(root / "plan.json", p)
    cpu = base.common.submit(root, "cpu")
    write_once(
        root / "submission.json", dict(cpu=cpu, follow=base.common.submit(root, "after_cpu", cpu))
    )
    print(json.dumps(base.common.read(root / "submission.json")), flush=True)


def worker(root, p, folder, i):
    import evicastkv_paper_precision as campaign
    import torch
    from compact_hybrid_study import manifest

    from mosaickv.adapters.huggingface import load_hf_adapter
    from mosaickv.config import RunConfig
    from mosaickv.evaluation.complete_transport_predicate import use as predicate
    from mosaickv.evaluation.early_scale_headers import use as early
    from mosaickv.evaluation.exact_scale_replay import use as late
    from mosaickv.evaluation.history_source_precision import use as history
    from mosaickv.evaluation.predictive_source_gpu import use as codec
    from mosaickv.evaluation.tiled_softmax import use as tile
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    manifest(p, sha(root / "plan.json"), folder, i, gpu=True)
    cfg = RunConfig.from_mapping(p["controls"]["parent"]["config"])
    adapter = load_hf_adapter(
        cfg.model.id,
        revision=cfg.model.revision,
        model_kwargs=_model_kwargs(cfg, local_files_only=True),
        processor_kwargs={"local_files_only": True},
    )
    transport = dict(
        p["controls"]["parent"],
        options={
            method: dict(options, native_history_graph=False)
            for method, options in p["controls"]["options"].items()
        },
    )
    sample = base.common.read(p["controls"]["inputs"])[i]
    reference = torch.load(
        p["controls"]["references"][str(i)]["storage"], weights_only=True, map_location="cpu"
    )
    same_precision_reference = (
        torch.load(p["same_precision_references"][str(i)], weights_only=True, map_location="cpu")
        if p.get("same_precision_references")
        else None
    )
    start = monotonic()
    errors = 0
    with (folder / "raw.jsonl").open("x") as output:
        for job in p["timing_schedules"][str(i)]:
            policy = p.get("arm_policies", {}).get(job["arm"], {})
            row = dict(
                job,
                sample_id=sample["sample_id"],
                manifest_path=str(folder / "manifest.json"),
                status="failed",
                whole_request_KV_peak_bytes=None,
                whole_request_KV_peak_NOT_measured_in_timing=True,
            )
            if monotonic() - start > p["timing_worker_seconds"]:
                row["error"] = "bounded timing worker budget exhausted; remaining trials absent"
                output.write(json.dumps(row) + "\n")
                errors += 1
                break
            try:
                with ExitStack() as stack:
                    stack.enter_context(
                        codec(
                            decoder="packed_words", encoding_policy="direct_additive_compact_rare"
                        )
                    )
                    stack.enter_context(tile(rows=p["tiled_softmax_rows"]))
                    stack.enter_context(predicate())
                    early_report = stack.enter_context(early())
                    late_report = stack.enter_context(late())
                    history_report = (
                        stack.enter_context(history(feature_group=32, exact_recent_tokens=64))
                        if policy.get(
                            "history", p.get("timing_history_on_both") or job["arm"] == ARMS[1]
                        )
                        else None
                    )
                    mixed_report = None
                    fused_unpack_report = None
                    paired_source_report = None
                    if policy.get(
                        "mixed", p.get("timing_mixed_candidate") and job["arm"] == ARMS[1]
                    ):
                        from mosaickv.evaluation.mixed_history_projection import use as mixed

                        profile = bool(
                            p.get("profile_first_warmup")
                            and job["warmup"]
                            and job["repetition"] == 0
                        )
                        mixed_report = stack.enter_context(mixed(profile=profile))
                    if policy.get("fused_unpack"):
                        from mosaickv.evaluation.fused_history_unpack import use as fused_unpack

                        fused_unpack_report = stack.enter_context(fused_unpack())
                    if policy.get("paired_source"):
                        from mosaickv.evaluation.paired_history_source import use as paired_source

                        paired_source_report = stack.enter_context(paired_source())
                    logits, details = campaign.execute(
                        adapter, sample, transport, "mosaickv_exact", audit=False
                    )
                if p.get("precision_equivalence_required"):
                    from compact_hybrid_study import fidelity

                    proof = fidelity(logits.cpu(), reference, strict=True, p=p)
                    proof["reference_is_frozen_unfused_same_precision_policy"] = True
                else:
                    mode = (
                        history_study.ARMS[1 if policy["history"] else 0] if policy else job["arm"]
                    )
                    proof = history_study.fidelity(logits.cpu(), reference, p, mode)
                if policy.get("history") and same_precision_reference is not None:
                    proof["frozen_same_precision_logits_bitwise"] = bool(
                        torch.equal(logits.cpu(), same_precision_reference)
                    )
                    proof["passed"] &= proof["frozen_same_precision_logits_bitwise"]
                if not proof["passed"]:
                    raise RuntimeError("unchanged complete-horizon numerical gate failed")
                if not job["warmup"] and job["repetition"] == 0:
                    torch.save(logits.cpu(), folder / (job["arm"] + ".logits.pt"))
                row.update(
                    status="completed",
                    fidelity=proof,
                    end_to_end_seconds=details["end_to_end_seconds"],
                    decode_seconds=details["decode_cuda_ms"] / 1000,
                    initial_GPU_KV_bytes=details["initial_encoded_bytes"],
                    final_GPU_KV_bytes=details["final_encoded_bytes"],
                    prefill_and_construction_seconds=details[
                        "prefill_construct_retire_wall_seconds"
                    ],
                    generated_token_ids=details["generated_token_ids"],
                    answer=details["answer"],
                    used_fullkv_fallback=details["used_fullkv_fallback"],
                    early_headers=early_report,
                    late_headers=[
                        {k: v for k, v in r.items() if k != "records"} for r in late_report
                    ],
                    selective_history=history_report,
                    mixed_projection=mixed_report,
                    fused_unpack=fused_unpack_report,
                    paired_source=paired_source_report,
                )
            except Exception:
                row["error"] = traceback.format_exc()
                errors += 1
            output.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
            output.flush()
    write_once(folder / "result.json", dict(status="completed", failures=errors))
    write_once(folder / "ending_gpu_environment.json", asdict(capture_gpu_environment(torch)))


def analyze(root, p):
    results, errors = [], []
    for i in (0, 1):
        path = root / f"model/{i}/raw.jsonl"
        rows = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
        if len(rows) != 50 or any(r["status"] != "completed" for r in rows):
            errors.append(dict(input=i, error="incomplete/failed warmups or timed trials"))
            continue
        timed = [r for r in rows if not r["warmup"]]
        if Counter(r["arm"] for r in timed) != Counter({arm: 20 for arm in ARMS}):
            errors.append(dict(input=i, error="missing registered method counts"))
            continue
        local = {}
        for metric in ("end_to_end_seconds", "decode_seconds"):
            values = [
                [r[metric] for r in sorted(timed, key=lambda r: r["repetition"]) if r["arm"] == arm]
                for arm in ARMS
            ]
            local[metric] = paired(
                values, seed=p["seed"] + i, replicates=p["timing_bootstrap_replicates"]
            )
        results.append(dict(input=i, **local))
    folder = root / "analysis"
    folder.mkdir(exist_ok=False)
    write_once(
        folder / "decision.json",
        dict(
            complete=len(results) == 2 and not errors,
            errors=errors,
            results=results,
            storage_superiority_NOT_established=True,
            six_component_superiority_established=False,
            fresh_cohort_authorized=False,
            prior_memory_gate_NOT_overridden=True,
        ),
    )
    lines = [
        "# Consumed-input uninstrumented paired latency",
        "",
        "Five warmups and 20 randomized paired repetitions per arm/input. "
        "Both arms receive the same common header/predicate optimizations.",
        "",
        "| Input | Metric | Mean ratio | Adjusted ratio CI | Equal/faster |",
        "|---|---|---:|---|---|",
    ]
    for r in results:
        for key in ("end_to_end_seconds", "decode_seconds"):
            v = r[key]
            lines.append(
                f"| {r['input']} | {key} | {v['mean_latency_ratio']:.6f} | "
                f"{v['ratio_adjusted_CI']} | {v['equal_or_faster']} |"
            )
    lines += [
        "",
        "The previous initial/peak memory gate remains negative. No new independent "
        "quality, complete superiority, or whole-request peak claim follows from timing.",
    ]
    (folder / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("prepare", "cpu", "after_cpu", "model", "analyze", "controller")
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha")
    parser.add_argument("--stage")
    args = parser.parse_args()
    root = args.root
    if args.action == "prepare":
        prepare(root)
        return
    p = base.common.checked(root, args.plan_sha)
    if args.action == "cpu":
        base.cpu(root, p)
    elif args.action == "after_cpu":
        if (root / "cpu.json").exists() and base.common.read(root / "cpu.json")[
            "status"
        ] == "completed":
            base.common.launch(root, "model")
        else:
            write_once(root / "CPU-negative.json", dict(no_GPU_submission=True))
    elif args.action == "model":
        import torch

        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(p["seed"])
        i = int(os.environ["SLURM_ARRAY_TASK_ID"])
        folder = root / f"model/{i}"
        folder.mkdir(parents=True, exist_ok=False)
        worker(root, p, folder, i)
    elif args.action == "analyze":
        analyze(root, p)
    else:
        subprocess.run(
            [
                PYTHON,
                str(SOURCE / "mosaickv/scripts/control_a100_schedule.py"),
                "--policy",
                str(root / f"admission-{args.stage}.json"),
                "--state",
                str(root / f"admission-{args.stage}-state.json"),
                "--apply",
                "--watch",
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
