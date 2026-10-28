#!/usr/bin/env python3
"""Bounded consumed-input arithmetic diagnosis, not a confirmation campaign."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import traceback
from dataclasses import asdict
from pathlib import Path

from evicastkv_campaign import SOURCE, environment, resolved_config
from evicastkv_evidence import exclusion_inventory
from evicastkv_finalize import confirmation_feasibility
from evicastkv_reduction_diagnostic import CASES

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

POLICIES = ("native_fp16", "fp32_attention")


def parity_passed(comparisons):
    return set(comparisons) == {"ordinary_generate", "reinjection"} and all(
        p["generated_tokens"] >= 16
        and p["token_agreement"] == 1
        and p["maximum_logit_difference"] <= 1e-4
        for p in comparisons.values()
    )


def one_case(adapter, prepared, workload):
    import torch

    from mosaickv.adapters.huggingface import compare_cache_reinjection, compare_with_generate
    from mosaickv.evaluation.fox_boundary import hold_back_last_text_token
    from mosaickv.evaluation.fox_numerics import unpruned_padding_control

    comparisons = {
        name: asdict(fn(adapter, prepared, max_new_tokens=16))
        for name, fn in (
            ("ordinary_generate", compare_with_generate),
            ("reinjection", compare_cache_reinjection),
        )
    }
    prefill = adapter.prefill(prepared, capture_queries=False, capture_attentions=False)
    cache_types = sorted(
        {
            str(t.dtype)
            for pair in adapter._legacy_layers(prefill.state.past_key_values)[0]
            for t in pair
        }
    )
    if cache_types != ["torch.float16"]:
        raise AssertionError("arithmetic experiment changed stored cache dtype")
    result = dict(
        parity=comparisons,
        parity_passed=parity_passed(comparisons),
        stored_cache_dtypes=cache_types,
    )
    if workload == "milebench":
        errors = unpruned_padding_control(
            adapter, prefill.state, int(prefill.next_token_id.item()), steps=16
        )
        maximum = max(e["normalized_linf"] for e in errors)
        result.update(
            padding_errors=errors,
            maximum_normalized_error=maximum,
            padding_control_passed=maximum <= 8 * 2**-10,
        )
    else:
        prefix, held = hold_back_last_text_token(prepared)
        a = prefill.logits.detach().float().cpu()
        prefix_prefill = adapter.prefill(prefix, capture_queries=False, capture_attentions=False)
        b = adapter.decode_one_token(held, prefix_prefill.state, capture_queries=False)
        b_logits = b.logits.detach().float().cpu()
        # Same chunked geometry, independently prefetched and reinjected. This
        # extra identity does NOT replace or waive the historical A/B criterion.
        other = adapter.prefill(prefix, capture_queries=False, capture_attentions=False)
        other.state.past_key_values = adapter.inject_past_key_values(
            adapter.extract_past_key_values(other.state.past_key_values, clone=True)
        )
        independent = adapter.decode_one_token(held, other.state, capture_queries=False)
        error = float((a - b_logits).abs().max().item())
        identity = float((b_logits - independent.logits.detach().float().cpu()).abs().max().item())
        result.update(
            A_B_maximum_logit_difference=error,
            A_B_original_control_passed=error <= 1e-4,
            A_B_greedy_equal=bool(torch.equal(a.argmax(-1), b_logits.argmax(-1))),
            B_reinjected_same_chunk_maximum_logit_difference=identity,
            B_same_chunk_identity_passed=identity <= 1e-4,
            same_chunk_is_additional_diagnostic_not_replacement_gate=True,
        )
    return result


def worker(root):
    import torch

    from mosaickv.adapters.huggingface import load_hf_adapter
    from mosaickv.evaluation.fox_attention_precision import attention_precision
    from mosaickv.evaluation.messages import MediaItem, build_multimodal_messages
    from mosaickv.evaluation.paper_native_lanes import _media_payload, load_paper_native_ledger
    from mosaickv.evaluation.tasks import default_task_registry
    from mosaickv.evaluation.worker_support import settle
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    protocol = json.loads((root / "protocol.json").read_text())
    clean_source(SOURCE, protocol["source_sha"])
    result = dict(
        cases=[],
        errors=[],
        source_sha=protocol["source_sha"],
        protocol_sha256=sha(root / "protocol.json"),
    )
    try:
        if not os.environ.get("SLURM_JOB_ID") or torch.cuda.device_count() != 1:
            raise RuntimeError("one allocated GPU required")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise RuntimeError("cuBLAS setting must precede Python")
        gres = subprocess.check_output(
            ["squeue", "-h", "-u", os.environ["USER"], "-t", "RUNNING,COMPLETING", "-o", "%b"],
            text=True,
        )
        allocated = sum(
            int(m[1])
            for line in gres.splitlines()
            if (m := re.search(r"gpu(?::[a-zA-Z0-9_-]+)?:(\d+)", line))
        )
        if allocated > 6:
            raise RuntimeError("aggregate GPU ceiling exceeded")
        torch.set_num_threads(12)
        torch.manual_seed(protocol["seed"])
        torch.use_deterministic_algorithms(True)
        result.update(
            environment=environment(),
            gpu=asdict(capture_gpu_environment(torch)),
            slurm_job_id=os.environ["SLURM_JOB_ID"],
            aggregate_user_gpus=allocated,
        )
        for workload, specification in protocol["workloads"].items():
            parent = Path(specification["parent"])
            if sha(parent / "protocol_lock.json") != specification["protocol_sha256"]:
                raise ValueError("parent protocol changed")
            lock = json.loads((parent / "protocol_lock.json").read_text())
            cfg = resolved_config(lock, workload, "full_kv", None)
            cohort = lock["cohorts"][workload]["validation"]
            ledger = load_paper_native_ledger(
                cohort["path"],
                expected_sha256=cohort["sha256"],
                task=default_task_registry().resolve(
                    "milebench" if workload == "milebench" else "coco_caption"
                ),
                dataset_revision=cfg.dataset.revision,
            )
            samples = {s.sample_id: s for s in ledger.samples}
            adapter = load_hf_adapter(
                cfg.model.id,
                revision=cfg.model.revision,
                model_kwargs=_model_kwargs(cfg, local_files_only=True),
                processor_kwargs={"local_files_only": True},
            )
            versions = [(p, p._version, p.dtype, p.device) for p in adapter.model.parameters()]
            for policy in POLICIES:
                with attention_precision(adapter, policy) as arithmetic:
                    for sid in specification["sample_ids"]:
                        sample = samples[sid]
                        prepared = adapter.prepare_inputs(
                            build_multimodal_messages(
                                sample.prompt,
                                tuple(MediaItem(m.kind, _media_payload(m)) for m in sample.media),
                            )
                        )
                        case = dict(
                            workload=workload,
                            sample_id=sid,
                            policy=policy,
                            model=asdict(cfg.model),
                            dataset=asdict(cfg.dataset),
                            arithmetic=arithmetic,
                            input_sha256=digest(
                                dict(prompt=sample.prompt, media=[m.sha256 for m in sample.media])
                            ),
                            **one_case(adapter, prepared, workload),
                        )
                        write_once(root / "cases" / f"{workload}-{sid}-{policy}.json", case)
                        result["cases"].append(case)
                        del prepared
                        settle(adapter)
            if any(
                p._version != version or p.dtype != dtype or p.device != device
                for p, version, dtype, device in versions
            ):
                raise AssertionError("model weights were modified or changed dtype")
            del versions, adapter
            import gc

            gc.collect()
            torch.cuda.empty_cache()
    except Exception as error:
        result["errors"].append(
            dict(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        )
    write_once(root / "results.json", result)
    if result["errors"]:
        raise SystemExit(1)


def analyze(root):
    protocol = json.loads((root / "protocol.json").read_text())
    clean_source(SOURCE, protocol["source_sha"])
    raw = root / "results.json"
    result = (
        json.loads(raw.read_text()) if raw.exists() else dict(cases=[], errors=["worker missing"])
    )
    expected = {
        (w, sid, p)
        for w, spec in protocol["workloads"].items()
        for sid in spec["sample_ids"]
        for p in POLICIES
    }
    cases = result["cases"]
    keys = [(c["workload"], c["sample_id"], c["policy"]) for c in cases]
    complete = not result["errors"] and set(keys) == expected and len(keys) == len(expected)
    reports = []
    for workload, field in (
        ("milebench", "padding_control_passed"),
        ("coco", "A_B_original_control_passed"),
    ):
        for policy in POLICIES:
            subset = [c for c in cases if c["workload"] == workload and c["policy"] == policy]
            passed = bool(subset) and all(c[field] and c["parity_passed"] for c in subset)
            reports.append(
                dict(
                    workload=workload,
                    policy=policy,
                    cases=len(subset),
                    status="passed_consumed_diagnostic_only"
                    if complete and passed
                    else "failed_or_incomplete",
                    registered_control=field,
                )
            )
    write_once(
        root / "analysis.json",
        dict(
            complete=complete,
            controls=reports,
            errors=result["errors"],
            source_sha=protocol["source_sha"],
            protocol_sha256=sha(root / "protocol.json"),
            result_sha256=sha(raw) if raw.exists() else None,
            old_failures_reclassified=False,
            thresholds_changed=False,
            arithmetic_variant_not_default=True,
            fresh_confirmation_authorized=False,
            quality_or_superiority_established=False,
        ),
    )
    with (root / "report.md").open("x") as stream:
        stream.write("# Consumed-input numerical recovery study\n\n")
        stream.write(
            "No threshold or stored weight/cache dtype changed. FP32 attention is an "
            "explicit arithmetic variant; old failures are preserved. No benchmark "
            "labels were used. No quality/confirmation successor is authorized.\n\n"
        )
        for report in reports:
            stream.write(f"- {report}\n")


def submit(root, mile_parent, coco_parent, dependency):
    revision = clean_source(SOURCE)
    workloads = {}
    for workload, parent in (("milebench", mile_parent), ("coco", coco_parent)):
        lock = json.loads((parent / "protocol_lock.json").read_text())
        ids = lock["cohorts"][workload]["validation"]["sample_ids"]
        selected = list(CASES) if workload == "milebench" else ids[:2]
        if not set(selected) <= set(ids):
            raise ValueError("diagnostic identities must already be consumed")
        workloads[workload] = dict(
            parent=str(parent),
            protocol_sha256=sha(parent / "protocol_lock.json"),
            sample_ids=selected,
        )
    write_once(
        root / "protocol.json",
        dict(
            source_sha=revision,
            seed=20271014,
            workloads=workloads,
            policies=list(POLICIES),
            numerical_tolerances=dict(
                retention_one=1e-4,
                padding_normwise=8 * 2**-10,
                A_B_original=1e-4,
                same_chunk_B=1e-4,
            ),
            weights_and_cache_dtype="fp16",
            backend="HF eager",
            steps=16,
            task_labels_used=False,
            fresh_inputs=False,
            automatic_promotion=False,
            resources=dict(GPUs=1, CPUs=12, memory_GiB=64, minutes=20, aggregate_GPU_limit=6),
        ),
    )
    exclusion_inventory(mile_parent, root / "history-inventory.json")
    confirmation_feasibility(root, root / "freshness_resolution.json")
    command = [
        "sbatch",
        "--parsable",
        "--account=bii_dsc_community",
        f"--output={root}/worker-%j.out",
        f"--error={root}/worker-%j.err",
    ]
    if dependency:
        command.append(f"--dependency=afterany:{dependency}")
    command += [
        str(SOURCE / "mosaickv/slurm/evicastkv_numerical_recovery.sbatch"),
        str(SOURCE),
        str(root),
        revision,
        "worker",
    ]
    gpu = subprocess.check_output(command, text=True).strip().split(";")[0]
    cpu_command = [
        "sbatch",
        "--parsable",
        "--account=bii_dsc_community",
        "--partition=bii",
        "--cpus-per-task=1",
        "--mem=4G",
        "--time=00:05:00",
        f"--dependency=afterany:{gpu}",
        f"--output={root}/analysis-%j.out",
        f"--error={root}/analysis-%j.err",
        str(SOURCE / "mosaickv/slurm/evicastkv_numerical_recovery_analysis.sbatch"),
        str(SOURCE),
        str(root),
        revision,
        "analyze",
    ]
    cpu = subprocess.check_output(cpu_command, text=True).strip().split(";")[0]
    write_once(
        root / "submission.json",
        dict(
            worker=gpu,
            analysis=cpu,
            GPU_command=command,
            CPU_command=cpu_command,
            source_sha=revision,
        ),
    )
    print(json.dumps(dict(worker=gpu, analysis=cpu)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("submit", "worker", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mile-parent", type=Path)
    parser.add_argument("--coco-parent", type=Path)
    parser.add_argument("--dependency")
    args = parser.parse_args()
    if args.mode == "submit":
        submit(args.root, args.mile_parent, args.coco_parent, args.dependency)
    else:
        (worker if args.mode == "worker" else analyze)(args.root)
