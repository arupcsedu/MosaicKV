#!/usr/bin/env python3
"""Consumed-input budget feasibility and complete-request envelope audit.

No new cohort, no automatic large campaign, no nominal-ratio row promoted to a
byte-matched comparison. Preserve every attempt and write a bounded decision.
"""

import argparse
import gc
import json
import os
import subprocess
import traceback
from dataclasses import asdict
from pathlib import Path

import evicastkv_paper_precision as campaign

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once
from mosaickv.evaluation.paper_request_certification import (
    MECHANISMS,
    WholeRequestEnvelope,
    comparison_admission,
)

SOURCE = campaign.SOURCE
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_paper_boundaries.sbatch"
PARENT = Path("/scratch/djy8hg/runs/evicastkv/paper-quality-coco-20261106-guard-recovery1")
METHODS = (
    "full_kv",
    "mosaickv_exact",
    "prompt_attention_topk",
    "prefixkv_reimpl",
    "vl_cache_reimpl",
    "lookm_reimpl",
)


def submit(root, action, **kwargs):
    return campaign.submit(root, action, script=SCRIPT, **kwargs)


def prepare(root):
    if root.exists():
        raise FileExistsError("new immutable diagnostic root required")
    clean_source(SOURCE)
    # Reuse only consumed IDs, before inspecting any per-method answer.
    campaign.prepare(root, PARENT, autostart=False)
    write_once(
        root / "scope.json",
        dict(
            consumed_sample_indices=[2],
            methods=METHODS,
            classification="consumed_input_representation_budget_feasibility_not_paper_comparison",
            automatic_fresh_materialization=False,
            automatic_large_campaign=False,
            source_sha=clean_source(SOURCE),
            source_plan_sha256=sha(root / "plan.json"),
            original_mechanism_activity={name: False for name in MECHANISMS},
        ),
    )
    cpu = submit(root, "cpu", minutes=30)
    write_once(root / "submission.json", dict(cpu=cpu))


def check(root, expected):
    plan = campaign.checked(root, expected)
    scope = campaign.read(root / "scope.json")
    registered = dict(
        consumed_sample_indices=[2],
        methods=list(METHODS),
        classification="consumed_input_representation_budget_feasibility_not_paper_comparison",
        automatic_fresh_materialization=False,
        automatic_large_campaign=False,
        source_sha=plan["source_sha"],
        source_plan_sha256=expected,
        original_mechanism_activity={name: False for name in MECHANISMS},
    )
    if scope != registered:
        raise ValueError("scope/source binding changed")
    return plan, scope


def cpu(root, expected):
    plan, scope = check(root, expected)
    with (root / "cpu_tests.txt").open("x") as output:
        completed = subprocess.run(
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
        dict(passed=completed.returncode == 0, log_sha256=sha(root / "cpu_tests.txt")),
    )
    if completed.returncode:
        return
    write_once(root / "ledger.json", campaign.load_ledger(PARENT))
    write_once(
        root / "ledger_seal.json",
        dict(campaign.read(PARENT / "ledger_seal.json"), plan_sha256=expected),
    )
    from unittest.mock import patch

    import evicastkv_uniform_all_baselines as old

    runtime = campaign.read(campaign.PARENT / "plan.json")
    record_path = old.screen.parent.PARENT / "workers/0/study.json"
    records = campaign.read(record_path)["records"]
    options = dict(plan["options"])
    # Prefix is the previously frozen nonuniform path, not the uniform-only
    # comparison script's original three-method allowlist. No baseline math
    # is patched; extend only that options-construction allowlist locally.
    with patch.object(
        old, "METHODS", tuple(m for m in METHODS if m not in ("full_kv", "mosaickv_exact"))
    ):
        for method in METHODS[2:]:
            options[method] = old.configurations(records, method, runtime)[method, "graph"]
    write_once(
        root / "options.json",
        dict(
            options=options, parent_records=str(record_path), parent_records_sha256=sha(record_path)
        ),
    )
    jobs = [dict(method=m, samples=scope["consumed_sample_indices"]) for m in METHODS]
    campaign.launch_phase(root, "audit", jobs, submitter=submit)


def worker(root, expected, index):
    import torch
    from evicastkv_campaign import environment

    from mosaickv.adapters.huggingface import load_hf_adapter
    from mosaickv.config import RunConfig
    from mosaickv.evaluation.scheduler_identity import scheduler_identity
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    plan, scope = check(root, expected)
    if (
        not campaign.read(root / "cpu.json")["passed"]
        or torch.cuda.device_count() != 1
        or "A100" not in torch.cuda.get_device_name()
        or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
    ):
        raise ValueError("CPU suite and one deterministic A100 required")
    sealed_options = campaign.read(root / "options.json")
    if sha(Path(sealed_options["parent_records"])) != sealed_options["parent_records_sha256"]:
        raise ValueError("frozen baseline options changed")
    plan = dict(plan, options=sealed_options["options"])
    ledger = campaign.load_ledger(root)
    job = campaign.read(root / "audit/jobs.json")[index]
    folder = root / f"audit/workers/{index}"
    folder.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(12)
    torch.manual_seed(plan["protocol"]["seed"])
    torch.use_deterministic_algorithms(True)
    cfg = RunConfig.from_mapping(plan["config"])
    errors = []
    try:
        adapter = load_hf_adapter(
            cfg.model.id,
            revision=cfg.model.revision,
            model_kwargs=_model_kwargs(cfg, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        write_once(
            folder / "manifest.json",
            dict(
                run_id=str(folder),
                git_sha=plan["source_sha"],
                source_dirty=False,
                config_sha=digest(
                    dict(
                        plan=expected,
                        scope=scope,
                        job=job,
                        options_sha256=sha(root / "options.json"),
                    )
                ),
                model=asdict(cfg.model),
                dataset=asdict(cfg.dataset),
                backend="huggingface",
                attention="eager",
                seed=cfg.execution.seed,
                environment=environment(),
                cuda=torch.version.cuda,
                gpu=asdict(capture_gpu_environment(torch)),
                **scheduler_identity({}),
                processor_revision=cfg.model.revision,
                ledger_sha256=sha(root / "ledger.json"),
                reference_artifact_sha=sha(root / "ledger.json"),
                baseline_upstream_commits={
                    m: plan["config"].get(key, {}).get("official_repository_sha", "not_used")
                    for m, key in (("prefixkv_reimpl", "prefixkv"), ("lookm_reimpl", "lookm"))
                },
                baseline_implementation_commit=plan["source_sha"],
                evaluation_label="precision_storage_adaptation_not_official_baseline",
                measurement_type="consumed_diagnostic_measured",
                method=job["method"],
                scope_sha256=sha(root / "scope.json"),
            ),
        )
        with (folder / "raw.jsonl").open("x") as stream:
            for sample_index in job["samples"]:
                sample = ledger["samples"][sample_index]
                row = dict(
                    sample_id=sample["sample_id"],
                    sample_index=sample_index,
                    method=job["method"],
                    status="failed",
                    source_sha=plan["source_sha"],
                    manifest_path=str(folder / "manifest.json"),
                    comparison_controls_sha256=digest(
                        dict(
                            config=plan["config"],
                            ledger=sha(root / "ledger.json"),
                            sample=sample,
                            output_tokens=128,
                            precision="bf16",
                            attention="eager",
                            storage_boundary="unique_PyTorch_backing_not_nominal_ratio",
                        )
                    ),
                    whole_request_envelope=None,
                )
                audit = None
                try:
                    gc.collect()
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    with WholeRequestEnvelope(adapter.model) as audit:
                        values, details = campaign.execute(
                            adapter, sample, plan, job["method"], audit=True
                        )
                        # execute() has already released packets/detokenized. Its
                        # exported logits are CPU, not CUDA KV storage.
                        del values
                    row.update(
                        details, status="completed", whole_request_envelope=audit.certificate
                    )
                except Exception:
                    row["error"] = traceback.format_exc()
                    if audit is not None:
                        row["whole_request_envelope"] = audit.certificate
                    errors.append(row["error"])
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
    except Exception:
        errors.append(traceback.format_exc())
    finally:
        write_once(
            folder / "result.json",
            dict(
                status="completed" if not errors else "failed",
                errors=errors,
                source_sha=plan["source_sha"],
                plan_sha256=expected,
                job=job,
                raw_sha256=sha(folder / "raw.jsonl") if (folder / "raw.jsonl").exists() else None,
            ),
        )


def analyze(root, expected):
    import pandas as pd

    plan, scope = check(root, expected)
    rows, failures = [], []
    for index, job in enumerate(campaign.read(root / "audit/jobs.json")):
        folder = root / f"audit/workers/{index}"
        try:
            result = campaign.read(folder / "result.json")
            if (
                result["source_sha"] != plan["source_sha"]
                or result["plan_sha256"] != expected
                or result["job"] != job
                or result["raw_sha256"] != sha(folder / "raw.jsonl")
            ):
                raise ValueError("producer hash/source mismatch")
            records = [json.loads(line) for line in (folder / "raw.jsonl").read_text().splitlines()]
            rows.extend(records)
            failures.extend(result["errors"])
        except Exception:
            failures.append(traceback.format_exc())
    folder = root / "audit/analysis"
    folder.mkdir(parents=True, exist_ok=False)
    write_once(folder / "raw_attempts.json", rows)
    compact = [
        dict(
            sample_id=r["sample_id"],
            method=r["method"],
            status=r["status"],
            initial_bytes=r.get("initial_encoded_bytes"),
            final_bytes=r.get("final_encoded_bytes"),
            decode_peak_bytes=r.get("decode_owned_peak_bytes"),
            whole_request_KV_upper_bound_bytes=(r.get("whole_request_envelope") or {}).get(
                "kv_peak_upper_bound_bytes"
            ),
            end_to_end_seconds=r.get("end_to_end_seconds"),
        )
        for r in rows
    ]
    pd.DataFrame(compact).to_parquet(folder / "storage_feasibility.parquet", index=False)
    ledger = campaign.load_ledger(root)
    ids = [ledger["samples"][i]["sample_id"] for i in scope["consumed_sample_indices"]]
    old = campaign.read(PARENT / "quality/analysis/selected_quality_rows.json")
    caps = {
        r["sample_id"]: r["initial_encoded_bytes"]
        for r in old
        if r["method"] == "mosaickv_exact" and r["sample_id"] in ids
    }
    compressed = [r for r in rows if r["method"] != "full_kv"]
    decision = comparison_admission(
        compressed, sample_ids=ids, methods=list(METHODS[1:]), cap_bytes_by_sample=caps
    )
    decision.update(
        producer_failures=failures,
        consumed_diagnostic_only=True,
        source_sha=plan["source_sha"],
        parent_quality_preserved=True,
        new_large_campaign_submitted=False,
        physical_budget_enforced_in_this_feasibility_measurement=False,
    )
    write_once(folder / "decision.json", decision)
    lines = [
        "# Consumed-input comparison feasibility",
        "",
        "This is not a byte-matched quality comparison or a superiority test.",
        "",
        "All six original mechanisms remain inactive in the storage variant.",
        "",
        "| Method | Initial bytes | Final bytes | Audited decode peak | "
        "Whole-request CUDA KV upper bound |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in compact:
        lines.append(
            "| "
            + " | ".join(
                str(row[k])
                for k in (
                    "method",
                    "initial_bytes",
                    "final_bytes",
                    "decode_peak_bytes",
                    "whole_request_KV_upper_bound_bytes",
                )
            )
            + " |"
        )
    lines += [
        "",
        "The upper bound includes non-KV allocations. It is not an exact KV peak.",
        "No cache padding or cap relaxation was used to manufacture equality.",
        "",
        f"Admission blockers: {len(decision['blockers'])}. Larger execution remains blocked.",
    ]
    (folder / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=["prepare", "cpu", "audit", "audit_controller", "audit_analysis"]
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--plan-sha")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root)
    elif args.action == "cpu":
        cpu(args.root, args.plan_sha)
    elif args.action == "audit":
        worker(args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    elif args.action == "audit_analysis":
        analyze(args.root, args.plan_sha)
    else:
        check(args.root, args.plan_sha)
        subprocess.run(
            [
                "/scratch/djy8hg/env/mosaickv/bin/python",
                str(SOURCE / "mosaickv/scripts/control_a100_schedule.py"),
                "--policy",
                str(args.root / "audit/scheduler/policy.json"),
                "--state",
                str(args.root / "audit/scheduler/state.json"),
                "--watch",
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
