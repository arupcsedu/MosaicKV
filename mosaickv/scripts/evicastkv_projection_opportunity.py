#!/usr/bin/env python3
"""Bounded metadata-only feasibility audit; no inference sweep or kernel change."""

import argparse
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path

import evicastkv_paired_request_readers as pair

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = pair.SOURCE
PARENT = Path("/scratch/djy8hg/runs/evicastkv/source-hotpath-consumed-20261102-v1")
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_projection_opportunity.sbatch"


def read(path):
    # Immutable ledgers serialize paths as strings; pathlib callers are also
    # accepted. Do not require callers to mutate or rematerialize the ledger.
    return json.loads(Path(path).read_text())


def historical_report(root, decision):
    """Derived report only; never rewrite the unsuccessful parent closeout."""
    records = []
    lines = [
        "# Analysis of 22359260 and its bounded follow-up",
        "",
        "Two consumed inputs; independent quality N=0. Different encoded-byte",
        "operating points, NOT byte-matched algorithm comparisons. Peak below",
        "means decode-owned tensor cache storage, not whole-request GPU memory.",
        "",
        "| Input | Evi original s | Evi prepared s | Mean speedup % | "
        "Adjusted ratio upper | Prefix prepared s | Evi slower than Prefix % |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for i, s in enumerate(decision["summaries"]):
        rows = {(x["method"], x["reader"]): x for x in s["distributions"]}
        anchor = rows["mosaickv_exact", "anchor_hotpath"]
        candidate = rows["mosaickv_exact", "candidate_hotpath"]
        baseline = rows["prefixkv_reimpl", "candidate_hotpath"]
        original, prepared, prefix = (x["mean"] for x in (anchor, candidate, baseline))
        upper = s["within_method"]["mosaickv_exact"]["familywise_ratio_bounds"][1]
        speedup = 100 * (1 - prepared / original)
        slower = 100 * (prepared / prefix - 1)
        lines.append(
            f"| {i} | {original:.6f} | {prepared:.6f} | {speedup:.3f} | "
            f"{upper:.6f} | {prefix:.6f} | {slower:.3f} |"
        )
        path = PARENT / f"workers/{i}"
        result = read(path / "result.json")
        if result["status"] != "completed" or any(
            sha(path / name) != value for name, value in result["outputs"].items()
        ):
            raise ValueError("modified successful parent evidence")
        controls = read(path / "all_method_controls.json")["rows"]
        storage = {x["method"]: x["audited_storage_boundaries"] for x in controls}
        records.append(
            dict(
                input_index=i,
                original=anchor,
                prepared=candidate,
                prefix=baseline,
                adjusted_ratio_upper=upper,
                memory_bytes=storage,
                construction_gap_seconds=candidate["mean_prefill_construction_retirement_seconds"]
                - baseline["mean_prefill_construction_retirement_seconds"],
                decode_gap_seconds=candidate["mean_decode_seconds"]
                - baseline["mean_decode_seconds"],
            )
        )
    lines.extend(
        [
            "",
            "Both adjusted upper bounds exceed the frozen <=0.99 rule: not promoted.",
            "",
            "| Input | Method | Initial MiB | Final MiB | Decode-owned peak MiB |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for record in records:
        for method, storage in record["memory_bytes"].items():
            values = [
                storage[k] / 2**20
                for k in ("initial_encoded_bytes", "final_encoded_bytes", "decode_owned_peak_bytes")
            ]
            lines.append(
                f"| {record['input_index']} | {method} | "
                f"{values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} |"
            )
    lines.extend(
        [
            "",
            "The restore follow-up is incomplete: input 0 was slower; input 1",
            "stopped on the erroneous nonempty-patch activity requirement. Its saved",
            "failure record shows bitwise logits and equal audited storage, but this",
            "does NOT recover missing timed trials or establish a complete pass.",
            "No policy is promoted and no broad superiority is established.",
        ]
    )
    write_once(
        root / "historical_analysis.json",
        dict(
            parent_decision_sha256=sha(PARENT / "analysis/decision.json"),
            inputs=records,
            promotion_passed=False,
            independent_quality_N=0,
            paper_superiority_established=False,
        ),
    )
    with (root / "historical_report.md").open("x") as stream:
        stream.write("\n".join(lines) + "\n")


def prepare(root, revision):
    if root.exists() or clean_source(SOURCE) != revision:
        raise ValueError("new immutable output and clean source required")
    parent = read(PARENT / "plan.json")
    decision = read(PARENT / "analysis/decision.json")
    if not decision["complete"] or decision["hotpath_end_to_end_improvement_gate"]:
        raise ValueError("complete unpromoted source-launch study required")
    bound = dict(parent["bound_files"])
    for path in (
        PARENT / "plan.json",
        PARENT / "analysis/decision.json",
        PARENT / "engineering_closeout.json",
        PARENT / "restore/analysis/decision.json",
        PARENT / "restore/workers/1/validation_failure.json",
        SOURCE / "mosaickv/docs/projection_opportunity_20261103.md",
    ):
        bound[str(path)] = sha(path)
    write_once(
        root / "plan.json",
        dict(
            parent,
            source_sha=revision,
            bound_files=bound,
            stage="consumed_projection_overwrite_metadata_only",
            minimum_skip_fraction_on_each_input=0.10,
            automatic_kernel_rewrite_authorized=False,
            fresh_materialization_authorized=False,
            paper_campaign_authorized=False,
        ),
    )
    historical_report(root, decision)
    pair.common.previous.cpu(root, sha(root / "plan.json"))
    queue = subprocess.check_output(
        ["squeue", "-h", "-r", "-u", os.environ["USER"], "-o", "%i|%b"], text=True
    )
    if any("gpu" in line.lower() for line in queue.splitlines()):
        raise ValueError("finish preceding owned GPU work first")
    (root / "slurm").mkdir(exist_ok=True)
    args = [str(SOURCE), str(root), sha(root / "plan.json")]
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
            "--time=00:10:00",
            "--array=0-1%2",
            "--job-name=mk-proj-opportunity",
            f"--output={root}/slurm/gpu-%A_%a.out",
            f"--error={root}/slurm/gpu-%A_%a.err",
            str(SCRIPT),
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
            "--nodes=1",
            "--ntasks=1",
            "--cpus-per-task=2",
            "--mem=16G",
            "--time=00:10:00",
            f"--dependency=afterany:{gpu}",
            "--job-name=mk-proj-opportunity-analysis",
            f"--output={root}/slurm/analysis-%j.out",
            f"--error={root}/slurm/analysis-%j.err",
            str(SCRIPT),
            *args,
            "analyze",
        ],
        text=True,
    ).strip()
    from control_a100_schedule import tick

    policy = dict(
        timezone="America/New_York",
        maximum_worker_minutes=10,
        maximum_a100s=6,
        six_a100s_explicitly_authorized=True,
        arrays=[gpu],
    )
    write_once(root / "scheduler/policy.json", policy)
    write_once(
        root / "scheduler/registration.json", tick(policy, {}, root / "scheduler/state.json", False)
    )
    write_once(root / "submission.json", dict(gpu_array=gpu, cpu_analysis=cpu))


class MetadataCaptured(Exception):
    """Controlled exit after the original constructor, before any decode."""


def worker(root, expected, index):
    import torch

    from mosaickv.adapters.huggingface import load_hf_adapter
    from mosaickv.config import RunConfig
    from mosaickv.evaluation.common_streaming_timing import request
    from mosaickv.evaluation.projection_opportunity import packet_coverage
    from mosaickv.evaluation.scheduler_identity import scheduler_identity
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    plan = pair.common.previous.common.inputs(root, expected)
    out = root / f"workers/{index}"
    out.mkdir(parents=True, exist_ok=False)
    result = dict(status="failed", failures=[], source_sha=plan["source_sha"], plan_sha256=expected)
    try:
        if (
            index not in (0, 1)
            or torch.cuda.device_count() != 1
            or "A100" not in torch.cuda.get_device_name()
        ):
            raise ValueError("exactly one A100 and one of two consumed inputs required")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ValueError("deterministic launch environment required")
        proof = read(root / "cpu.json")
        if (
            proof["returncode"]
            or proof["failed_or_skipped"]
            or proof["source_sha"] != plan["source_sha"]
        ):
            raise ValueError("complete same-source CPU suite required")
        if proof["xml_sha256"] != sha(root / "cpu.xml"):
            raise ValueError("CPU proof changed")
        torch.set_num_threads(12)
        torch.manual_seed(plan["seed"])
        torch.use_deterministic_algorithms(True)
        job = plan["jobs"][index]
        cfg = RunConfig.from_mapping(pair.common.previous.resolved_mapping(plan, job))
        if (
            cfg.model.id,
            cfg.model.revision,
            cfg.model.precision.value,
            cfg.execution.attention_implementation,
        ) != (
            "llava-hf/llava-1.5-7b-hf",
            "b234b804b114d9e37bb655e11cbbb5f5e971b7a9",
            "bf16",
            "eager",
        ):
            raise ValueError("pinned source workload changed")
        adapter = load_hf_adapter(
            cfg.model.id,
            revision=cfg.model.revision,
            model_kwargs=_model_kwargs(cfg, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        manifest = read(pair.common.REFERENCE / f"workers/{job['reference_worker']}/manifest.json")
        manifest.update(scheduler_identity(manifest))
        manifest.update(
            source_sha=plan["source_sha"],
            source_dirty=False,
            plan_sha256=expected,
            run_id=str(out),
            measurement_type="diagnostic_measured",
            independent_quality_N=0,
            gpu=asdict(capture_gpu_environment(torch)),
            environment=pair.common.control.environment(),
            references_and_answers_accessed=False,
            no_decode_executed=True,
            representation_adaptation_not_complete_mosaickv=True,
        )
        write_once(out / "manifest.json", manifest)
        # Load ONLY the pinned packet digest. Do not load benchmark answers,
        # annotations, teacher-forcing targets or saved logits for this audit.
        parent_worker = 0 if index == 0 else 8
        payload = read(pair.common.AUDIT / f"workers/{parent_worker}/offline_payload.json")
        captured = []

        def capture(packets):
            digest = pair.common.control.digest_tensors(
                tuple(t for p in packets for t in p.tensors())
            )
            if digest != payload["payload_sha256"]:
                raise RuntimeError("metadata audit changed original constructed packet bits")
            captured.extend(dict(layer=i, **packet_coverage(p)) for i, p in enumerate(packets))
            raise MetadataCaptured

        messages = pair.common.previous.load_messages(read(job["inputs"][0]))
        profile = tuple(read(Path(plan["profile_path"])))
        try:
            request(
                adapter,
                messages,
                profile,
                "mosaickv_exact",
                minimum_only=plan["exact_minimum_only"],
                componentwise_fitting=True,
                input_packet_validator=capture,
                construction_geometry=plan["construction_geometry"],
                source_hotpath="original",
                reconstruction_readers="carry_literals",
                audit_decode_storage=False,
            )
        except MetadataCaptured:
            pass
        else:
            raise RuntimeError("metadata-only request unexpectedly executed decode")
        if len(captured) != len(profile):
            raise RuntimeError("incomplete source layer metadata")
        write_once(
            out / "opportunity.json",
            dict(
                input_index=index,
                layers=captured,
                packet_bits_identical=True,
                frozen_payload_sha256=payload["payload_sha256"],
                no_decode_executed=True,
                references_and_answers_accessed=False,
                independent_quality_N=0,
            ),
        )
        result.update(
            status="completed",
            outputs={name: sha(out / name) for name in ("manifest.json", "opportunity.json")},
        )
    except Exception as error:
        import traceback

        result["failures"].append(f"{type(error).__name__}: {error}")
        result["traceback"] = traceback.format_exc()
    write_once(out / "result.json", result)
    if result["status"] != "completed":
        raise SystemExit(1)


def analyze(root, expected):
    plan = pair.common.previous.common.inputs(root, expected)
    summaries, failures = [], []
    for index in (0, 1):
        try:
            out = root / f"workers/{index}"
            result = read(out / "result.json")
            if (
                result["status"] != "completed"
                or result["source_sha"] != plan["source_sha"]
                or result["plan_sha256"] != expected
            ):
                raise ValueError(f"incomplete producer: {result.get('failures')}")
            if any(sha(out / name) != value for name, value in result["outputs"].items()):
                raise ValueError("changed metadata producer")
            evidence = read(out / "opportunity.json")
            if not evidence["packet_bits_identical"] or not evidence["no_decode_executed"]:
                raise ValueError("packet or scope control failed")
            summary = dict(input_index=index, layer_count=len(evidence["layers"]))
            for kind in ("protected_only", "protected_plus_text"):
                total = sum(x[kind]["total_programs"] for x in evidence["layers"])
                covered = sum(x[kind]["wholly_overwritten_programs"] for x in evidence["layers"])
                summary[kind] = dict(
                    total_programs=total,
                    wholly_overwritten_programs=covered,
                    fraction=covered / total,
                )
            summary["nonempty_correction_components"] = sum(
                n > 0 for x in evidence["layers"] for n in x["correction_scalars_per_component"]
            )
            summaries.append(summary)
        except Exception as error:
            failures.append(dict(input_index=index, error=str(error)))
    feasible = (
        not failures
        and len(summaries) == 2
        and all(
            x["protected_only"]["fraction"] >= plan["minimum_skip_fraction_on_each_input"]
            for x in summaries
        )
    )
    write_once(
        root / "analysis/decision.json",
        dict(
            complete=not failures,
            failures=failures,
            source_sha=plan["source_sha"],
            plan_sha256=expected,
            summaries=summaries,
            protected_only_kernel_feasibility_gate=feasible,
            measured_latency_gain=False,
            paper_superiority_established=False,
            independent_quality_N=0,
            further_jobs_submitted=False,
            next_action="bounded skip-tile kernel with bitwise controls"
            if feasible
            else "stop protected-only tile skipping; no automatic kernel rewrite",
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "worker", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-sha")
    parser.add_argument("--plan-sha")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root, args.source_sha)
    elif args.action == "worker":
        worker(args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    else:
        analyze(args.root, args.plan_sha)
