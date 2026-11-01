#!/usr/bin/env python3
"""Isolate packet differences on two failed, already consumed method/inputs."""

import argparse
import gc
import os
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import evicastkv_common_streaming as common

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE, previous = common.SOURCE, common.previous


def prepare(root):
    if root.exists():
        raise FileExistsError("new diagnostic root required")
    parent = common.AUDIT
    old = previous.read(parent / "plan.json")
    jobs, bound = [], dict(old["bound_files"])
    bound[str(parent / "plan.json")] = sha(parent / "plan.json")
    for reference in (3, 1):
        job = old["jobs"][reference]
        record = previous.read(parent / f"workers/{reference}/result.json")
        if record["status"] != "failed" or not record["failures"]:
            raise ValueError("only the observed failed Top-K/VL-Cache inputs permitted")
        for name in ("result.json", "offline_payload.json", "raw.jsonl"):
            path = parent / f"workers/{reference}/{name}"
            bound[str(path)] = sha(path)
        jobs.append(dict(job, reference_worker=reference, index=len(jobs)))
    write_once(
        root / "plan.json",
        dict(
            old,
            jobs=jobs,
            source_sha=clean_source(SOURCE),
            bound_files=bound,
            stage="consumed_streaming_packet_numerical_diagnostic",
            maximum_worker_minutes=15,
            maximum_simultaneous_workers=2,
            no_memory_or_timing_comparison=True,
            no_fresh_or_large_campaign_authorized=True,
            paper_campaign_authorized=False,
        ),
    )
    print(sha(root / "plan.json"))


def differences(old, new):
    import torch

    if len(old) != len(new):
        return [dict(tensor_count_old=len(old), tensor_count_new=len(new))]
    result = []
    for i, (a, b) in enumerate(zip(old, new, strict=True)):
        if a.shape != b.shape or a.dtype != b.dtype:
            result.append(
                dict(
                    tensor=i,
                    old_shape=list(a.shape),
                    new_shape=list(b.shape),
                    old_dtype=str(a.dtype),
                    new_dtype=str(b.dtype),
                )
            )
        elif not torch.equal(a.view(torch.uint8), b.view(torch.uint8)):
            mask = a != b
            ids = mask.flatten().nonzero().flatten()[:8]
            result.append(
                dict(
                    tensor=i,
                    shape=list(a.shape),
                    dtype=str(a.dtype),
                    bitwise=False,
                    numerically_equal=bool(torch.equal(a, b)),
                    changed_numeric_values=int(mask.sum()),
                    first_indices=ids.cpu().tolist(),
                    old_values=a.flatten()[ids].float().cpu().tolist(),
                    new_values=b.flatten()[ids].float().cpu().tolist(),
                )
            )
    return result


def _worker(root, expected, index):
    from dataclasses import asdict

    import torch

    from mosaickv.adapters.huggingface.registry import load_hf_adapter
    from mosaickv.config import RunConfig
    from mosaickv.evaluation.streaming_baselines import SelectedRetiringCache, StreamingBaseline
    from mosaickv.evaluation.streaming_prefill import SealAfterAttention
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    plan = previous.common.inputs(root, expected)
    job = plan["jobs"][index]
    method = common.method_of(job)
    folder = root / f"workers/{index}"
    folder.mkdir(parents=True, exist_ok=False)
    if (
        torch.cuda.device_count() != 1
        or "A100" not in torch.cuda.get_device_name()
        or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
    ):
        raise ValueError("one A100 and registered deterministic CUDA workspace required")
    torch.set_num_threads(12)
    torch.manual_seed(plan["seed"])
    torch.use_deterministic_algorithms(True)
    cfg = RunConfig.from_mapping(previous.resolved_mapping(plan, job))
    manifest = previous.read(common.REFERENCE / f"workers/{job['reference_worker']}/manifest.json")
    manifest.update(
        source_sha=plan["source_sha"],
        source_dirty=False,
        plan_sha256=expected,
        environment=common.control.environment(),
        gpu=asdict(capture_gpu_environment(torch)),
        no_memory_or_timing_comparison=True,
        measurement_type="diagnostic_measured",
    )
    write_once(folder / "manifest.json", manifest)
    adapter = load_hf_adapter(
        cfg.model.id,
        revision=cfg.model.revision,
        model_kwargs=_model_kwargs(cfg, local_files_only=True),
        processor_kwargs={"local_files_only": True},
    )
    payload = previous.read(job["inputs"][0])
    prepared = adapter.prepare_inputs(previous.load_messages(payload))
    profile = tuple(previous.read(Path(plan["profile_path"])))
    cap = previous.read(common.REFERENCE / f"workers/{job['reference_worker']}/construction.json")[
        "cap_bytes"
    ]
    packets, memberships, first = common.offline(
        adapter, prepared, cfg, job, profile, plan["profile_path"], cap
    )
    records = []
    for audit in (False, True):
        gc.collect()
        cache = SelectedRetiringCache(len(profile), audit=audit)
        encoder = StreamingBaseline(prepared, cache, profile, method)
        fresh = replace(prepared, model_inputs=dict(prepared.model_inputs, past_key_values=cache))
        with SealAfterAttention(adapter, cache, encoder):
            pref = adapter.prefill(fresh, capture_queries=False, capture_attentions=False)
        with torch.inference_mode():
            encoder.finish(adapter)
            new = cache.formats()
            layers = []
            for i, (old, updated, block) in enumerate(
                zip(packets, new, adapter._language_layers(), strict=True)
            ):
                difference = differences(old.tensors(), updated.tensors())
                if difference:
                    original = old.materialize(block.self_attn)
                    changed = updated.materialize(block.self_attn)
                    layers.append(
                        dict(
                            layer=i,
                            payload_differences=difference,
                            decoded_KV_differences=differences(original, changed),
                        )
                    )
            records.append(
                dict(
                    audit=audit,
                    prefill_logits_bitwise=torch.equal(first, pref.logits.cpu()),
                    memberships_identical=tuple(tuple(map(tuple, x)) for x in memberships)
                    == tuple(tuple(map(tuple, x)) for x in encoder.memberships),
                    layers_with_differences=layers,
                    no_memory_comparison=True,
                )
            )
        del pref, fresh, cache, encoder, new
    write_once(
        folder / "diagnostic.json",
        dict(
            source_sha=plan["source_sha"],
            plan_sha256=expected,
            method=method,
            sample_id=payload["sample_id"],
            rows=records,
            no_quality_or_memory_or_latency_claim=True,
        ),
    )


def worker(root, expected, index):
    try:
        _worker(root, expected, index)
    except Exception as exc:
        folder = root / f"workers/{index}"
        folder.mkdir(parents=True, exist_ok=True)
        write_once(
            folder / "failure.json",
            dict(
                status="failed",
                index=index,
                plan_sha256=expected,
                error=f"{type(exc).__name__}: {exc}",
            ),
        )
        raise


def analyze(root, expected):
    plan = previous.common.inputs(root, expected)
    rows, failures = [], []
    for index in range(2):
        path = root / f"workers/{index}/diagnostic.json"
        if path.exists():
            rows.append(previous.read(path))
        else:
            error = root / f"workers/{index}/failure.json"
            failures.append(
                previous.read(error) if error.exists() else f"missing numerical diagnostic {index}"
            )
    write_once(
        root / "analysis/decision.json",
        dict(
            source_sha=plan["source_sha"],
            plan_sha256=expected,
            rows=rows,
            failures=failures,
            complete=not failures,
            independent_quality_N=0,
            larger_campaign_blocked=True,
        ),
    )


def submit(root, expected):
    original = common.control.parent.subprocess.check_output

    def scoped(command, **kwargs):
        if command and command[0] == "sbatch":
            command = [
                str(SOURCE / "mosaickv/slurm/evicastkv_streaming_packet_diagnostic.sbatch")
                if x.endswith("/evicastkv_funded_delimiter_diagnostic.sbatch")
                else x
                for x in command
            ]
        return original(command, **kwargs)

    with patch.object(common.control.parent.subprocess, "check_output", scoped):
        common.control.parent.submit(root, expected)


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
    elif args.mode == "worker":
        worker(args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    else:
        globals()[args.mode](args.root, args.plan_sha)
