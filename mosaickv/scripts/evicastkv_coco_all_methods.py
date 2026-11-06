#!/usr/bin/env python3
"""Finite user-authorized 256-image operating-point measurements, no claim promotion."""

import argparse
import gc
import json
import os
import random
import subprocess
import traceback
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import evicastkv_paper_boundaries as boundaries
import evicastkv_paper_precision as campaign

from mosaickv.evaluation.coco_all_methods import (
    caption_text,
    complete_grid,
    endpoints,
    validate_row,
)
from mosaickv.evaluation.fox_campaign import digest, sha, write_once

SOURCE = campaign.SOURCE
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_coco_all_methods.sbatch"
CONFIG = SOURCE / "mosaickv/configs/experiments/coco_256_all_methods_20261106.json"
METHODS = boundaries.METHODS
INDICES = tuple(range(2, 258))


def submit(root, action, **kwargs):
    return campaign.submit(root, action, script=SCRIPT, **kwargs)


def prepare(root):
    campaign.prepare(root, boundaries.PARENT, autostart=False)
    write_once(
        root / "scope.json",
        dict(
            protocol=campaign.read(CONFIG),
            protocol_sha256=sha(CONFIG),
            plan_sha256=sha(root / "plan.json"),
            previous_predictions_sha256=boundaries.QUALITY_ROWS_SHA256,
            original_mechanisms={name: False for name in boundaries.MECHANISMS},
            confirmation_or_superiority_authorized=False,
        ),
    )
    write_once(root / "submission.json", dict(cpu=submit(root, "cpu", minutes=45)))


def check(root, expected):
    plan = campaign.checked(root, expected)
    scope = campaign.read(root / "scope.json")
    if (
        scope["protocol_sha256"] != sha(CONFIG)
        or scope["protocol"] != campaign.read(CONFIG)
        or scope["plan_sha256"] != expected
        or sha(boundaries.PARENT / "quality/analysis/selected_quality_rows.json")
        != scope["previous_predictions_sha256"]
    ):
        raise ValueError("immutable study scope or parent predictions changed")
    return plan, scope["protocol"]


def launch(root, phase, jobs):
    campaign.launch_phase(root, phase, jobs, submitter=submit)


def cpu(root, expected):
    check(root, expected)
    with (root / "cpu_tests.txt").open("x") as output:
        result = subprocess.run(
            [
                "/scratch/djy8hg/env/mosaickv/bin/python",
                "-m",
                "pytest",
                "mosaickv/tests",
                "-q",
                "-m",
                "not gpu",
            ],
            cwd=SOURCE,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
    write_once(
        root / "cpu.json",
        dict(passed=result.returncode == 0, log_sha256=sha(root / "cpu_tests.txt")),
    )
    if result.returncode:
        return
    write_once(root / "ledger.json", campaign.load_ledger(boundaries.PARENT))
    write_once(
        root / "ledger_seal.json",
        dict(campaign.read(boundaries.PARENT / "ledger_seal.json"), plan_sha256=expected),
    )
    import evicastkv_uniform_all_baselines as old

    runtime = campaign.read(campaign.PARENT / "plan.json")
    record_path = old.screen.parent.PARENT / "workers/0/study.json"
    records = campaign.read(record_path)["records"]
    options = dict(campaign.read(root / "plan.json")["options"])
    with patch.object(old, "METHODS", METHODS[2:]):
        for method in METHODS[2:]:
            options[method] = old.configurations(records, method, runtime)[method, "graph"]
    write_once(
        root / "options.json",
        dict(
            options=options, parent_records=str(record_path), parent_records_sha256=sha(record_path)
        ),
    )
    launch(root, "canary", [dict(methods=[m], samples=[2]) for m in METHODS])


def load_plan(root, expected):
    plan, protocol = check(root, expected)
    sealed = campaign.read(root / "options.json")
    if sha(Path(sealed["parent_records"])) != sealed["parent_records_sha256"]:
        raise ValueError("frozen baseline constructor options changed")
    if not campaign.read(root / "cpu.json")["passed"]:
        raise ValueError("CPU proof required")
    return dict(plan, options=sealed["options"]), protocol


def worker(root, expected, phase, index):
    import torch
    from evicastkv_campaign import environment

    from mosaickv.adapters.huggingface import load_hf_adapter
    from mosaickv.config import RunConfig
    from mosaickv.evaluation.scheduler_identity import scheduler_identity
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    plan, protocol = load_plan(root, expected)
    if (
        torch.cuda.device_count() != 1
        or "A100" not in torch.cuda.get_device_name()
        or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
    ):
        raise ValueError("one deterministic A100 required")
    torch.set_num_threads(12)
    torch.manual_seed(protocol["seed"])
    torch.use_deterministic_algorithms(True)
    job = campaign.read(root / phase / "jobs.json")[index]
    folder = root / phase / f"workers/{index}"
    folder.mkdir(parents=True, exist_ok=False)
    cfg = RunConfig.from_mapping(plan["config"])
    ledger = campaign.load_ledger(root)
    failures = []
    try:
        adapter = load_hf_adapter(
            cfg.model.id,
            revision=cfg.model.revision,
            model_kwargs=_model_kwargs(cfg, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        manifest = dict(
            run_id=str(folder),
            git_sha=plan["source_sha"],
            source_dirty=False,
            config_sha=digest(
                dict(
                    plan=expected,
                    job=job,
                    phase=phase,
                    scope=sha(root / "scope.json"),
                    options=sha(root / "options.json"),
                )
            ),
            model=asdict(cfg.model),
            dataset=asdict(cfg.dataset),
            backend="huggingface",
            attention="eager",
            precision="bf16",
            seed=protocol["seed"],
            environment=environment(),
            cuda=torch.version.cuda,
            gpu=asdict(capture_gpu_environment(torch)),
            **scheduler_identity({}),
            processor_revision=cfg.model.revision,
            ledger_sha256=sha(root / "ledger.json"),
            reference_artifact_sha=sha(root / "ledger.json"),
            baseline_implementation_commit=plan["source_sha"],
            baseline_upstream_commits={
                m: plan["config"].get(key, {}).get("official_repository_sha", "not_used")
                for m, key in (("prefixkv_reimpl", "prefixkv"), ("lookm_reimpl", "lookm"))
            },
            measurement_type="consumed_operating_point_audit"
            if phase in ("canary", "memory")
            else "uninstrumented_complete_request_timing",
            evaluation_label="precision_storage_adaptations_not_official_not_byte_matched",
            original_six_mechanisms_active=False,
            phase=phase,
        )
        write_once(folder / "manifest.json", manifest)
        if phase in ("timing", "repeated"):
            # Uninstrumented, equal warmup policy. FullKV never enters latency ratios.
            for method in job["methods"]:
                for _ in range(protocol["warmups"]):
                    value, _details = campaign.execute(
                        adapter, ledger["samples"][job["samples"][0]], plan, method, audit=False
                    )
                    del value, _details
        with (folder / "raw.jsonl").open("x") as output:
            for sample_index in job["samples"]:
                repetitions = protocol["timing_repetitions"] if phase == "repeated" else 1
                for trial in range(repetitions):
                    order = list(job["methods"])
                    random.Random(f"{protocol['seed']}:{sample_index}:{trial}").shuffle(order)
                    for method in order:
                        sample = ledger["samples"][sample_index]
                        row = dict(
                            sample_id=sample["sample_id"],
                            sample_index=sample_index,
                            method=method,
                            phase=phase,
                            trial=trial,
                            status="failed",
                            source_sha=plan["source_sha"],
                            manifest_path=str(folder / "manifest.json"),
                            manifest_sha256=sha(folder / "manifest.json"),
                            same_physical_budget_claim=False,
                        )
                        try:
                            audited = phase in ("memory", "canary")
                            value, detail = campaign.execute(
                                adapter, sample, plan, method, audit=audited
                            )
                            row.update(detail, status="completed", memory_instrumentation=audited)
                            validate_row(row, audited=audited)
                            if phase == "canary":
                                fast, fast_detail = campaign.execute(
                                    adapter, sample, plan, method, audit=False
                                )
                                if (
                                    not torch.equal(value, fast)
                                    or endpoints(row)[:2] != endpoints(fast_detail)[:2]
                                ):
                                    raise RuntimeError(
                                        "audit changed model logits or storage endpoints"
                                    )
                                row["audit_fast_128_logits_bitwise_identical"] = True
                                if method in ("full_kv", "mosaickv_exact"):
                                    old = campaign.read(
                                        boundaries.PARENT
                                        / "quality/analysis/selected_quality_rows.json"
                                    )
                                    prior = next(
                                        r
                                        for r in old
                                        if r["method"] == method
                                        and r["sample_index"] == sample_index
                                    )
                                    if row["generated_token_ids"] != prior["generated_token_ids"]:
                                        raise RuntimeError("frozen parent token parity changed")
                                    row["previous_frozen_tokens_identical"] = True
                                del fast, fast_detail
                            del value, detail
                        except Exception:
                            row.update(status="failed", error=traceback.format_exc())
                            failures.append(row["error"])
                        output.write(json.dumps(row, allow_nan=False) + "\n")
                        output.flush()
                        gc.collect()
    except Exception:
        failures.append(traceback.format_exc())
    finally:
        write_once(
            folder / "result.json",
            dict(
                status="failed" if failures else "completed",
                failures=failures,
                source_sha=plan["source_sha"],
                plan_sha256=expected,
                job=job,
                raw_sha256=sha(folder / "raw.jsonl") if (folder / "raw.jsonl").exists() else None,
                manifest_sha256=sha(folder / "manifest.json")
                if (folder / "manifest.json").exists()
                else None,
            ),
        )


def collect(root, phase, expected):
    plan, protocol = check(root, expected)
    rows, errors = [], []
    for index, job in enumerate(campaign.read(root / phase / "jobs.json")):
        folder = root / phase / f"workers/{index}"
        try:
            result = campaign.read(folder / "result.json")
            if (
                result["source_sha"] != plan["source_sha"]
                or result["plan_sha256"] != expected
                or result["job"] != job
                or result["raw_sha256"] != sha(folder / "raw.jsonl")
                or result["manifest_sha256"] != sha(folder / "manifest.json")
            ):
                raise ValueError("producer source, manifest, job or raw hash mismatch")
            part = [json.loads(line) for line in (folder / "raw.jsonl").read_text().splitlines()]
            rows.extend(part)
            errors.extend(result["failures"])
            complete_grid(
                part,
                job["methods"],
                job["samples"],
                trials=protocol["timing_repetitions"] if phase == "repeated" else 1,
            )
            for row in part:
                validate_row(row, audited=phase in ("memory", "canary"))
        except Exception:
            errors.append(traceback.format_exc())
    return rows, errors


def analyze(root, expected, phase):
    import pandas as pd

    plan, protocol = check(root, expected)
    rows, errors = collect(root, phase, expected)
    folder = root / phase / "analysis"
    folder.mkdir(parents=True, exist_ok=False)
    write_once(folder / "all_attempts.json", rows)
    if phase != "canary":
        other_phase = "memory" if phase in ("timing", "repeated") else None
        if other_phase:
            memory = {
                (r["method"], r["sample_index"]): r
                for r in campaign.read(root / other_phase / "analysis/all_attempts.json")
            }
            for row in rows:
                prior = memory.get((row["method"], row["sample_index"]))
                if (
                    prior is None
                    or row.get("status") != "completed"
                    or prior.get("status") != "completed"
                    or any(row.get(k) != prior.get(k) for k in ("sample_id", "generated_token_ids"))
                    or endpoints(row)[:2] != endpoints(prior)[:2]
                ):
                    errors.append(
                        "audit/timing outputs or endpoints differ: "
                        f"{row['method']}/{row['sample_index']}"
                    )
    write_once(
        folder / "decision.json",
        dict(
            complete=not errors,
            errors=errors,
            observed_rows=len(rows),
            same_physical_budget_claim=False,
            quality_gate_cannot_hide_negative_results=True,
            source_sha=plan["source_sha"],
            raw_sha256=sha(folder / "all_attempts.json"),
        ),
    )
    pd.DataFrame(
        [
            dict(
                sample_id=r["sample_id"],
                sample_index=r["sample_index"],
                method=r["method"],
                trial=r["trial"],
                status=r["status"],
                initial_bytes=endpoints(r)[0] if r["status"] == "completed" else None,
                final_bytes=endpoints(r)[1] if r["status"] == "completed" else None,
                audited_decode_peak_bytes=r.get("decode_owned_peak_bytes"),
                end_to_end_seconds=r.get("end_to_end_seconds"),
                memory_instrumentation=r.get("memory_instrumentation"),
                manifest_path=r["manifest_path"],
            )
            for r in rows
        ]
    ).to_parquet(folder / "measurements.parquet", index=False)
    if errors:
        return
    if phase == "canary":
        if len(rows) != len(METHODS) or not all(
            r.get("audit_fast_128_logits_bitwise_identical") for r in rows
        ):
            raise RuntimeError("six complete audit/fast word-parity canaries required")
        launch(
            root,
            "memory",
            [
                dict(methods=[m], samples=list(INDICES[i : i + 16]))
                for m in METHODS
                for i in range(0, 256, 16)
            ],
        )
    elif phase == "memory":
        launch(
            root,
            "timing",
            [
                dict(methods=list(METHODS), samples=list(INDICES[i : i + 16]))
                for i in range(0, 256, 16)
            ],
        )
    elif phase == "timing":
        launch(
            root,
            "repeated",
            [dict(methods=list(METHODS[1:]), samples=[i]) for i in protocol["timing_subset"]],
        )
    else:
        final_analysis(root, expected)


def final_analysis(root, expected):
    import numpy as np
    import pandas as pd

    from mosaickv.evaluation.cider_resampling import ResampledCider, paired_draw_counts
    from mosaickv.evaluation.metrics.types import MetricSample

    plan, protocol = check(root, expected)
    memory, errors = collect(root, "memory", expected)
    timing, more = collect(root, "timing", expected)
    repeated, last = collect(root, "repeated", expected)
    if errors or more or last:
        raise RuntimeError("complete producers required before corpus scoring")
    complete_grid(memory, METHODS, INDICES)
    complete_grid(timing, METHODS, INDICES)
    complete_grid(repeated, METHODS[1:], protocol["timing_subset"], trials=20)
    ledger = campaign.load_ledger(root)
    groups = {
        (m, None): [
            MetricSample(
                r["sample_id"],
                caption_text(r),
                tuple(ledger["samples"][r["sample_index"]]["references"]),
            )
            for r in memory
            if r["method"] == m
        ]
        for m in METHODS
    }
    engine = ResampledCider(groups)
    counts = np.ones((1, 256), dtype=np.int64)
    scores = engine.score(counts)[0]
    engine.validate_official(counts, scores[None, :])
    draws = engine.score(
        paired_draw_counts(256, protocol["bootstrap_replicates"], protocol["seed"])
    )
    values = {m: scores[engine.points.index((m, None))] for m in METHODS}
    output = root / "summary"
    output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for method in METHODS:
        ms = [r for r in memory if r["method"] == method]
        ts = [r["end_to_end_seconds"] for r in timing if r["method"] == method]
        summaries.append(
            dict(
                method=method,
                completed=256,
                failed=0,
                initial_MiB=float(np.mean([endpoints(r)[0] for r in ms]) / 2**20),
                final_MiB=float(np.mean([endpoints(r)[1] for r in ms]) / 2**20),
                mean_audited_decode_peak_MiB=float(np.mean([endpoints(r)[2] for r in ms]) / 2**20),
                max_audited_decode_peak_MiB=float(max(endpoints(r)[2] for r in ms) / 2**20),
                cider=float(values[method]),
                latency_mean_s=float(np.mean(ts)),
                latency_p50_s=float(np.median(ts)),
                latency_p95_s=float(np.quantile(ts, 0.95)),
                timing_trials=256,
                same_realized_byte_budget=False,
                label="quality_reference"
                if method == "full_kv"
                else "precision_storage_adaptation",
            )
        )
    quality = []
    ci = engine.points.index(("mosaickv_exact", None))
    for method in METHODS:
        if method == "mosaickv_exact":
            continue
        j = engine.points.index((method, None))
        diff = draws[:, ci] - draws[:, j]
        quality.append(
            dict(
                baseline=method,
                cider_difference=float(scores[ci] - scores[j]),
                paired_95_ci=np.quantile(diff, [0.025, 0.975]).tolist(),
                familywise_95_ci=np.quantile(diff, [0.05 / 10, 1 - 0.05 / 10]).tolist(),
                byte_matched=False,
                claim="separate_frozen_operating_points_not_selection_policy_superiority",
            )
        )
    latency = campaign.timing_comparisons(repeated, protocol["seed"])
    pd.DataFrame(summaries).to_parquet(output / "summary.parquet", index=False)
    pd.DataFrame(quality).to_parquet(output / "quality_comparisons.parquet", index=False)
    write_once(
        output / "summary.json",
        dict(
            rows=summaries,
            quality=quality,
            latency=latency,
            source_sha=plan["source_sha"],
            same_realized_byte_budget=False,
            original_six_mechanisms_active=False,
            global_freshness_certified=False,
            whole_request_KV_peak_certified=False,
            audited_timing_excluded_from_latency=True,
            data_hashes={
                str(root / p / "analysis/all_attempts.json"): sha(
                    root / p / "analysis/all_attempts.json"
                )
                for p in ("memory", "timing", "repeated")
            },
            remaining=[
                "byte-matched strongest-baseline confirmation",
                "independent quality confirmation",
                "whole-request exact KV-only peak (decode-owned peak is not that boundary)",
                "six original mechanisms remain inactive; no full-design claim",
                "official baseline reproduction not established by these precision adaptations",
            ],
        ),
    )
    text = [
        "# 256-image COCO: frozen operating points",
        "",
        "Not byte-matched. Shared inputs/model/processor/generation; encoded sizes differ.",
        "Storage columns are cohort means. Peak is **decode-owned**, not whole-request HBM.",
        "Latency is uninstrumented request time. FullKV is not a latency competitor.",
        "",
        "| Method | N | Initial MiB | Decode peak MiB | Final MiB | CIDEr | p50 / p95 (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summaries:
        text.append(
            f"| {r['method']} | 256 | {r['initial_MiB']:.3f} | "
            f"{r['mean_audited_decode_peak_MiB']:.3f} | {r['final_MiB']:.3f} | "
            f"{r['cider']:.3f} | {r['latency_p50_s']:.3f} / {r['latency_p95_s']:.3f} |"
        )
    text.extend(
        [
            "",
            "Each compressed method also has 5 warmups and 20 trials on six fixed images.",
            "Paired adjusted timing ratios: summary.json. Audit times are excluded.",
            "Inspected IDs are development evidence. Negative results remain included.",
            "",
            "## What remains",
            "",
            *[f"- {x}" for x in campaign.read(output / "summary.json")["remaining"]],
        ]
    )
    (output / "report.md").write_text("\n".join(text) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root)
    elif args.action == "cpu":
        cpu(args.root, args.plan_sha)
    elif args.action.endswith("_controller"):
        phase = args.action.removesuffix("_controller")
        check(args.root, args.plan_sha)
        subprocess.run(
            [
                "/scratch/djy8hg/env/mosaickv/bin/python",
                str(SOURCE / "mosaickv/scripts/control_a100_schedule.py"),
                "--policy",
                str(args.root / phase / "scheduler/policy.json"),
                "--state",
                str(args.root / phase / "scheduler/state.json"),
                "--apply",
                "--watch",
            ],
            check=True,
        )
    elif args.action.endswith("_analysis"):
        analyze(args.root, args.plan_sha, args.action.removesuffix("_analysis"))
    elif args.action in ("canary", "memory", "timing", "repeated"):
        worker(args.root, args.plan_sha, args.action, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    else:
        raise ValueError("unknown finite action")


if __name__ == "__main__":
    main()
