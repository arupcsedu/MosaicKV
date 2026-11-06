#!/usr/bin/env python3
"""Finite registered COCO evidence collection, not automatic claim promotion."""

import argparse
import gc
import inspect
import json
import math
import os
import random
import subprocess
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
CONFIG = SOURCE / "mosaickv/configs/experiments/paper_precision_coco_20261106.json"
PARENT = Path("/scratch/djy8hg/runs/evicastkv/uniform-core-paired-consumed-20261106-v1")
POOL = Path("/scratch/djy8hg/runs/mosaickv/coco-shared-pool-edb6edf-retry1-20261014/ledger.json")
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_paper_precision.sbatch"


def read(path):
    return json.loads(Path(path).read_text())


def checked(root, expected):
    if sha(root / "plan.json") != expected:
        raise ValueError("plan changed")
    plan = read(root / "plan.json")
    clean_source(SOURCE, plan["source_sha"])
    for path, value in plan["bound_files"].items():
        if sha(Path(path)) != value:
            raise ValueError(f"immutable input changed: {path}")
    return plan


def select_ids(rows, excluded, count, seed):
    if len({r[0] for r in rows}) != len(rows):
        raise ValueError("duplicate provider identities")
    ordered = sorted((r for r in rows if r[0] not in excluded), key=lambda r: digest([seed, r[0]]))
    if len(ordered) < count:
        raise ValueError("insufficient disjoint IDs")
    return ordered[:count]


def request_options(record, plan, method, audit):
    from mosaickv.evaluation.common_streaming_timing import request

    allowed = set(inspect.signature(request).parameters) | {"uniform_attention_core"}
    result = {k: v for k, v in record.items() if k in allowed}
    for key in (
        "defer_transport_validation",
        "coalesce_transport_nonzero",
        "batch_source_validation",
        "full_predicate_span",
    ):
        result.setdefault(key, plan["runtime_parent"].get(key, False))
    policy = record["direct_policy"]
    result["direct_text_policy"] = dict(
        modes=policy["modes"],
        maximum_added_bytes=policy["maximum_added_bytes"],
        accounting_mode="request_local_base_plus_allowance",
    )
    result.update(
        minimum_only=True,
        componentwise_fitting=True,
        code_delta=method in plan["runtime_parent"]["code_delta_methods"],
        construction_geometry=(16, 4),
        reconstruction_readers="carry_literals",
        reuse_kernel_dispatch=True,
        reuse_code_descriptors=True,
        source_hotpath="prepared_restore",
        audit_decode_storage=audit,
    )
    return result


def submit(root, action, *, array=None, dependency=None, minutes=120):
    args = ["sbatch", "--parsable", "--account=bii_dsc_community", "--nodes=1", "--ntasks=1"]
    if array is not None:
        args += [
            "--partition=bii-gpu",
            "--reservation=bi_fox_dgx",
            "--gres=gpu:a100:1",
            "--cpus-per-task=12",
            "--mem=48G",
            f"--array=0-{array - 1}%6",
            "--hold",
        ]
    else:
        args += ["--partition=bii", "--nodelist=udc-aj36-2c0", "--cpus-per-task=2", "--mem=16G"]
    args += [
        f"--time={minutes // 60:02}:{minutes % 60:02}:00",
        f"--job-name=mk-paper-{action}",
        f"--output={root}/slurm/{action}-%A_%a.out",
        f"--error={root}/slurm/{action}-%A_%a.err",
    ]
    if dependency:
        args += ["--dependency=afterany:" + str(dependency)]
    (root / "slurm").mkdir(parents=True, exist_ok=True)
    args += [str(SCRIPT), str(SOURCE), str(root), sha(root / "plan.json"), action]
    return subprocess.check_output(args, text=True).strip()


def launch_phase(root, phase, jobs):
    write_once(root / phase / "jobs.json", jobs)
    array = submit(root, phase, array=len(jobs))
    policy = dict(
        timezone="America/New_York",
        window="all_times",
        maximum_worker_minutes=120,
        maximum_a100s=6,
        six_a100s_explicitly_authorized=True,
        arrays=[array],
        # submit() created this exact array with --hold. Never adopt another
        # user's/manual hold; ownership is limited to our new sbatch receipt.
        adopt_existing_holds=[array],
    )
    write_once(root / phase / "scheduler/policy.json", policy)
    controller = submit(root, phase + "_controller", minutes=360)
    analysis = submit(root, phase + "_analysis", dependency=array, minutes=60)
    write_once(
        root / phase / "submission.json",
        dict(array=array, controller=controller, analysis=analysis),
    )


def prepare(root, reuse_ledger=None):
    if root.exists():
        raise FileExistsError("new immutable root required")
    source_sha = clean_source(SOURCE)
    protocol = read(CONFIG)
    runtime = read(PARENT / "plan.json")
    engineering = PARENT / "analysis/decision.json"
    if not read(engineering).get("formal_latency_acceptance_against_both_Prefix_controls"):
        raise ValueError("passed frozen engineering evidence required before new inputs")
    # Recover exactly the passed options, never historical module defaults.
    import evicastkv_uniform_all_baselines as old

    records_path = old.screen.parent.PARENT / "workers/0/study.json"
    records = read(records_path)["records"]
    groups = old.configurations(records, "prompt_attention_topk", runtime)
    options = {"mosaickv_exact": groups[old.CANDIDATE]}
    for value in options.values():
        if not value.get("direct_policy"):
            raise ValueError("frozen per-layer text modes required")
    cfg = dict(
        runtime["config"],
        execution=dict(runtime["config"]["execution"], seed=protocol["seed"]),
        dataset=dict(
            id="lmms-lab/COCO-Caption",
            revision="c3213b54e173ae86517d481de17f31439d6f8c72",
            split="val",
        ),
    )
    bound = {
        str(p): sha(p)
        for p in (
            CONFIG,
            PARENT / "plan.json",
            engineering,
            records_path,
            POOL,
            Path(runtime["profile_path"]),
            SOURCE / "mosaickv/env/common/requirements.lock",
        )
    }
    if reuse_ledger is not None:
        parent = read(reuse_ledger / "plan.json")
        if (
            parent["protocol"] != protocol
            or parent["config"] != cfg
            or parent["options"] != options
        ):
            raise ValueError("ledger reuse requires identical registered execution policies")
        load_ledger(reuse_ledger)
        for name in ("plan.json", "ledger.json", "ledger_seal.json"):
            bound[str(reuse_ledger / name)] = sha(reuse_ledger / name)
    write_once(
        root / "plan.json",
        dict(
            protocol=protocol,
            source_sha=source_sha,
            bound_files=bound,
            config=cfg,
            options=options,
            runtime_parent={
                k: runtime[k]
                for k in (
                    "code_delta_methods",
                    "defer_transport_validation",
                    "coalesce_transport_nonzero",
                    "batch_source_validation",
                    "full_predicate_span",
                )
            },
            profile_path=runtime["profile_path"],
            profile_sha256=sha(Path(runtime["profile_path"])),
            reused_ledger_parent=str(reuse_ledger) if reuse_ledger is not None else None,
        ),
    )
    write_once(root / "submission.json", dict(cpu=submit(root, "cpu", minutes=60)))


def cpu(root, expected):
    checked(root, expected)
    log = root / "cpu_tests.txt"
    with log.open("x") as stream:
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
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    write_once(root / "cpu.json", dict(passed=result.returncode == 0, log_sha256=sha(log)))
    if result.returncode:
        return
    plan = read(root / "plan.json")
    if plan.get("reused_ledger_parent"):
        parent = Path(plan["reused_ledger_parent"])
        ledger = load_ledger(parent)
        if (
            len(ledger["samples"])
            != plan["protocol"]["canary_samples"] + plan["protocol"]["evaluation_samples"]
        ):
            raise ValueError("reused cohort size differs")
        write_once(root / "ledger.json", ledger)
        write_once(
            root / "ledger_seal.json", dict(read(parent / "ledger_seal.json"), plan_sha256=expected)
        )
        if sha(root / "ledger.json") != sha(parent / "ledger.json"):
            raise ValueError("reused ledger content changed")
    else:
        materialize(root, expected)
    p = plan["protocol"]
    launch_phase(
        root,
        "canary",
        [dict(method=m, samples=[i]) for i in range(p["canary_samples"]) for m in p["methods"]],
    )


def materialize(root, expected):
    from datasets import Dataset
    from materialize_coco_shared_bytes import freeze_sample

    plan = checked(root, expected)
    protocol = plan["protocol"]
    pool = read(POOL)
    excluded = {s["sample_id"].split(":")[-1] for s in pool["samples"]}
    inventories = {}
    # Explicit inventories, not a claim to have scanned all worldwide exposure.
    inventory_root = Path("/scratch/djy8hg/runs/evicastkv")
    paths = set(inventory_root.glob("*/references.json"))
    paths.update(inventory_root.glob("*/cohort/references.json"))
    for path in sorted(paths):
        value = read(path)
        if not isinstance(value, dict):
            continue
        inventories[str(path)] = sha(path)
        for row in value.get("samples", []):
            sid = str(row.get("sample_id", ""))
            if "COCO_val2014_" in sid:
                excluded.add(sid.split(":")[-1])
    arrow = Path(
        "/scratch/djy8hg/cache/mosaickv/datasets/lmms-lab___coco-caption/default/0.0.0/c3213b54e173ae86517d481de17f31439d6f8c72"
    )
    rows, files = [], {}
    for i in range(14):
        path = arrow / f"coco-caption-val-{i:05d}-of-00014.arrow"
        files[str(path)] = sha(path)
        for offset, doc in enumerate(
            Dataset.from_file(str(path)).select_columns(["question_id", "file_name"])
        ):
            if doc["question_id"] != doc["file_name"]:
                raise ValueError("provider ID mismatch")
            rows.append((doc["file_name"], str(path), offset))
    selected = select_ids(
        rows,
        excluded,
        protocol["canary_samples"] + protocol["evaluation_samples"],
        protocol["seed"],
    )
    samples, pixel_ids, consumed_pixels = [], set(), set()
    import hashlib

    from PIL import Image

    for job in read(PARENT / "plan.json")["jobs"]:
        for filename in job["inputs"]:
            for medium in read(filename)["media"]:
                with Image.open(medium["path"]) as image:
                    rgb = image.convert("RGB")
                    consumed_pixels.add(
                        hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                    )

    for sid, path, offset in selected:
        sample = freeze_sample(Dataset.from_file(path)[offset], "coco_caption:" + sid, root)
        with Image.open(sample["media"][0]["path"]) as image:
            rgb = image.convert("RGB")
            pixels = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        if pixels in pixel_ids or pixels in consumed_pixels:
            raise ValueError(
                "duplicate or candidate-tuning RGB image; do not replace after outcomes"
            )
        pixel_ids.add(pixels)
        samples.append(sample)
    write_once(
        root / "ledger.json",
        dict(
            dataset=plan["config"]["dataset"],
            samples=samples,
            parent_arrow_files=files,
            excluded_inventories=inventories,
            excluded_identity_count=len(excluded),
            global_untouched_claim=False,
            prior_public_pool_excluded_sha256=sha(POOL),
            selected_rgb_hashes=sorted(pixel_ids),
            explicit_candidate_tuning_rgb_hashes=sorted(consumed_pixels),
            historical_cross_dataset_media_disjointness_certified=False,
        ),
    )
    write_once(
        root / "ledger_seal.json",
        dict(
            ledger_sha256=sha(root / "ledger.json"),
            plan_sha256=expected,
            selected_ids=[s[0] for s in selected],
        ),
    )


def load_ledger(root):
    seal = read(root / "ledger_seal.json")
    if sha(root / "ledger.json") != seal["ledger_sha256"]:
        raise ValueError("cohort ledger changed")
    return read(root / "ledger.json")


def execute(adapter, sample, plan, method, *, audit=False, original=False, retention_one=False):
    import torch
    from evicastkv_request_validation import load_messages

    from mosaickv.config import RunConfig
    from mosaickv.evaluation.common_streaming_timing import request
    from mosaickv.evaluation.paper_geometry import registered_geometry
    from mosaickv.evaluation.precision_quality import answer_prefix
    from mosaickv.evaluation.request_precision import complete_request

    payload = {k: sample[k] for k in ("sample_id", "prompt", "media")}
    payload["task"] = "coco_caption"
    for medium in payload["media"]:
        if sha(Path(medium["path"])) != medium["sha256"]:
            raise ValueError("media hash changed")
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    if method != "full_kv" and not retention_one:
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    if method == "full_kv" or retention_one:
        values, details = complete_request(
            adapter,
            lambda: load_messages(payload),
            RunConfig.from_mapping(plan["config"]),
            method=method,
            precision="ordinary" if method == "full_kv" else "query3",
            profile_path=plan["profile_path"],
            ratio=1.0,
            audit=False,
        )
        packets = ()
    else:
        messages = load_messages(payload)
        options = request_options(plan["options"][method], plan, method, audit)
        if original:
            options["uniform_attention_core"] = False
        with registered_geometry(plan["protocol"]["prompt_length"]):
            values, packets, details = request(
                adapter, messages, tuple(read(plan["profile_path"])), method, **options
            )
    tokens = values.argmax(-1).tolist()
    if not bool(torch.isfinite(values).all()):
        raise RuntimeError("nonfinite model logits")
    if method != "full_kv" and not retention_one:
        if (
            details["prefill_storage"]["logical_lengths"]
            != [plan["protocol"]["prompt_length"]] * 32
        ):
            raise RuntimeError("actual original prompt differs from registered geometry")
        core = details["uniform_attention_core"]
        if core["requested"] and (
            not core["graphs_retired"] or core["aborted"] or core["retained_tensor_owners"]
        ):
            raise RuntimeError("request-private graph owners were not retired")
    eos = adapter.model.generation_config.eos_token_id
    eos = (eos,) if type(eos) is int else tuple(eos)
    answer = answer_prefix(
        tokens,
        eos=eos,
        cap=128,
        decode=lambda ids: adapter.processor.tokenizer.decode(ids, skip_special_tokens=True),
    )
    del packets
    torch.cuda.synchronize()
    details.update(
        end_to_end_seconds=time.monotonic() - started,
        includes_media_processor_prefill_construction_graph_lifetime_decode_export_detokenization=True,
        whole_request_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        whole_request_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        baseline_model_allocated_bytes=baseline,
        whole_request_allocator_peak_certified=True,
        whole_request_KV_only_peak_certified=False,
        answer=answer,
        generated_token_ids=tokens,
        used_fullkv_fallback=False,
        original_six_mechanisms_disabled=True,
        same_physical_budget_claim=False,
    )
    if len(tokens) != 128 or not math.isfinite(details["end_to_end_seconds"]):
        raise RuntimeError("invalid complete output/timing")
    return values, details


def worker(root, expected, phase, index):
    import torch
    from evicastkv_campaign import environment

    from mosaickv.adapters.huggingface import load_hf_adapter
    from mosaickv.adapters.huggingface.parity import (
        compare_cache_reinjection,
        compare_with_generate,
    )
    from mosaickv.config import RunConfig
    from mosaickv.evaluation.fused_lossless_attention import bits_equal
    from mosaickv.evaluation.scheduler_identity import scheduler_identity
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    plan = checked(root, expected)
    if (
        not read(root / "cpu.json")["passed"]
        or torch.cuda.device_count() != 1
        or "A100" not in torch.cuda.get_device_name()
        or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
    ):
        raise ValueError("CPU proof and one deterministic A100 required")
    ledger = load_ledger(root)
    job = read(root / phase / "jobs.json")[index]
    output = root / phase / f"workers/{index}"
    output.mkdir(parents=True, exist_ok=False)
    failures, rows = [], []
    cfg = RunConfig.from_mapping(plan["config"])
    result = dict(
        source_sha=plan["source_sha"],
        plan_sha256=expected,
        job=job,
        status="failed",
        failures=failures,
    )
    try:
        torch.set_num_threads(12)
        torch.manual_seed(plan["protocol"]["seed"])
        torch.use_deterministic_algorithms(True)
        adapter = load_hf_adapter(
            cfg.model.id,
            revision=cfg.model.revision,
            model_kwargs=_model_kwargs(cfg, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        manifest = dict(
            run_id=str(output),
            source_sha=plan["source_sha"],
            source_dirty=False,
            config_sha256=digest(dict(plan=expected, job=job)),
            plan_sha256=expected,
            ledger_sha256=sha(root / "ledger.json"),
            model=asdict(cfg.model),
            processor_revision=cfg.model.revision,
            dataset=asdict(cfg.dataset),
            execution=asdict(cfg.execution),
            seed=plan["protocol"]["seed"],
            environment=environment(),
            environment_lock_sha256=sha(SOURCE / "mosaickv/env/common/requirements.lock"),
            cuda_runtime=torch.version.cuda,
            gpu=asdict(capture_gpu_environment(torch)),
            evaluator="lmms_eval==0.7.2/pycocoevalcap==1.2/PTBTokenizer+Cider_x100",
            evaluator_commit=plan["source_sha"],
            reference_artifact_sha=sha(root / "ledger.json"),
            baseline_commit=plan["source_sha"],
            baseline_upstream_commits={
                m: cfg_raw.get("official_repository_sha", "not_used")
                for m, cfg_raw in (
                    ("prefixkv_reimpl", plan["config"]["prefixkv"]),
                    ("lookm_reimpl", plan["config"]["lookm"]),
                )
            },
            baseline_label=plan["protocol"]["baseline_labels"],
            method=job["method"],
            result_category="unified_comparison",
            result_origin="locally_measured",
            measurement_type="reference_measured"
            if job["method"] == "full_kv"
            else "precision_variant_measured",
            **scheduler_identity({}),
        )
        write_once(output / "manifest.json", manifest)
        if phase == "canary" and job["method"] == "full_kv":
            from evicastkv_request_validation import load_messages

            sample = ledger["samples"][job["samples"][0]]
            payload = dict(
                task="coco_caption", **{k: sample[k] for k in ("sample_id", "prompt", "media")}
            )
            parity = {}
            for name, function in (
                ("generate", compare_with_generate),
                ("reinjection", compare_cache_reinjection),
            ):
                report = function(
                    adapter, adapter.prepare_inputs(load_messages(payload)), max_new_tokens=16
                )
                parity[name] = asdict(report)
                if report.token_agreement != 1 or report.maximum_logit_difference > 0.001:
                    raise RuntimeError("FullKV parity failed")
            write_once(output / "parity.json", parity)
        rng = random.Random(plan["protocol"]["seed"] + index)
        deterministic_tokens = {}
        sample_indices = list(job["samples"])
        rng.shuffle(sample_indices)
        with (output / "raw.jsonl").open("x") as stream:
            for ordinal, sample_index in enumerate(sample_indices):
                sample = ledger["samples"][sample_index]
                repetitions = 20 if phase == "timing" else 1
                warmups = 0 if phase == "canary" or ordinal else 5
                for trial in range(-warmups, repetitions):
                    methods = list(job.get("methods", [job["method"]]))
                    rng.shuffle(methods)
                    for method in methods:
                        row = dict(
                            task="coco_caption",
                            sample_id=sample["sample_id"],
                            method=method,
                            sample_index=sample_index,
                            trial=trial,
                            warmup=trial < 0,
                            phase=phase,
                            manifest_path=str(output / "manifest.json"),
                            source_sha=plan["source_sha"],
                            plan_sha256=expected,
                            status="failed",
                        )
                        try:
                            values, details = execute(
                                adapter, sample, plan, method, audit=phase == "canary"
                            )
                            if phase == "canary" and method != "full_kv":
                                control, _ = execute(adapter, sample, plan, method, original=True)
                                capture = output / f"logits-{sample_index}-{method}.pt"
                                torch.save(dict(optimized=values, parent=control), capture)
                                row.update(
                                    logit_capture=str(capture), logit_capture_sha256=sha(capture)
                                )
                                if not bits_equal(values, control):
                                    raise RuntimeError(
                                        "new-geometry optimized/parent 128-logit word parity failed"
                                    )
                                details["optimized_parent_128_logit_words_bitwise"] = True
                                del control
                            if phase == "canary" and method == "full_kv":
                                identity, identity_details = execute(
                                    adapter, sample, plan, "mosaickv_exact", retention_one=True
                                )
                                if (
                                    not bits_equal(values, identity)
                                    or not identity_details.get("retention_one_identity")
                                    or not identity_details.get("quantization_disabled")
                                ):
                                    raise RuntimeError(
                                        "retention-1 did not bypass all lossy transformations"
                                    )
                                capture = output / f"logits-{sample_index}-identity.pt"
                                torch.save(dict(fullkv=values, retention_one=identity), capture)
                                row.update(
                                    logit_capture=str(capture), logit_capture_sha256=sha(capture)
                                )
                                details["retention_one_candidate_128_logit_words_bitwise"] = True
                                del identity
                            row.update(
                                details,
                                status="completed",
                                token_ids_sha256=digest(values.argmax(-1).tolist()),
                            )
                            token_key = (sample_index, method)
                            token_hash = row["token_ids_sha256"]
                            if deterministic_tokens.setdefault(token_key, token_hash) != token_hash:
                                raise RuntimeError(
                                    "greedy output changed across identical repetitions"
                                )
                            del values
                        except Exception:
                            row["status"] = "failed"
                            row["error"] = traceback.format_exc()
                            failures.append(
                                dict(
                                    sample_id=sample["sample_id"],
                                    method=method,
                                    trial=trial,
                                    error=row["error"],
                                )
                            )
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                        stream.flush()
                        rows.append(row)
        result["status"] = "completed" if not failures else "failed"
    except Exception:
        failures.append(traceback.format_exc())
    finally:
        result.update(
            parity_sha256=sha(output / "parity.json")
            if (output / "parity.json").exists()
            else None,
            raw_sha256=sha(output / "raw.jsonl") if (output / "raw.jsonl").exists() else None,
            manifest_sha256=sha(output / "manifest.json")
            if (output / "manifest.json").exists()
            else None,
        )
        write_once(output / "result.json", result)


def analyze(root, expected, phase):
    import numpy as np
    import pandas as pd

    plan = checked(root, expected)
    jobs = read(root / phase / "jobs.json")
    rows, failures = [], []
    for index, job in enumerate(jobs):
        folder = root / phase / f"workers/{index}"
        try:
            result = read(folder / "result.json")
            if (
                result["source_sha"] != plan["source_sha"]
                or result["plan_sha256"] != expected
                or result["job"] != job
            ):
                raise ValueError(f"invalid producer {index}: {result['failures']}")
            # Keep verified failed rows in attempted denominators too.
            if (folder / "raw.jsonl").exists():
                if result["raw_sha256"] != sha(folder / "raw.jsonl"):
                    raise ValueError("raw producer hash mismatch")
                rows.extend(
                    json.loads(line) for line in (folder / "raw.jsonl").read_text().splitlines()
                )
            if result["status"] != "completed" or result["failures"]:
                raise ValueError(f"failed producer {index}: {result['failures']}")
            if result["raw_sha256"] != sha(folder / "raw.jsonl") or result[
                "manifest_sha256"
            ] != sha(folder / "manifest.json"):
                raise ValueError("producer hash mismatch")
            if (
                job["method"] == "full_kv"
                and phase == "canary"
                and result["parity_sha256"] != sha(folder / "parity.json")
            ):
                raise ValueError("FullKV parity artifact changed")
        except Exception as error:
            failures.append(str(error))
    scored = [r for r in rows if not r["warmup"]]
    repetitions = 20 if phase == "timing" else 1
    expected_keys = {
        (i, m, t)
        for j in jobs
        for i in j["samples"]
        for m in j.get("methods", [j["method"]])
        for t in range(repetitions)
    }
    observed_keys = [(r["sample_index"], r["method"], r["trial"]) for r in scored]
    if set(observed_keys) != expected_keys or len(observed_keys) != len(expected_keys):
        failures.append("missing or duplicate sample-method-trial rows")
    if any(
        r["status"] != "completed"
        or len(r.get("generated_token_ids", [])) != 128
        or r.get("used_fullkv_fallback") is not False
        for r in scored
    ):
        failures.append("failed sample, incomplete output or fallback")
    out = root / phase / "analysis"
    out.mkdir(parents=True, exist_ok=False)
    compact = []
    for row in scored:
        simple = {k: v for k, v in row.items() if v is None or type(v) in (str, int, float, bool)}
        storage = row.get("storage", {})
        simple.update(
            initial_cache_bytes=row.get(
                "initial_encoded_bytes", storage.get("initial_active_kv_storage_bytes")
            ),
            final_cache_bytes=row.get(
                "final_encoded_bytes", storage.get("final_active_kv_storage_bytes")
            ),
            decode_owned_peak_bytes=row.get("decode_owned_peak_bytes"),
            answer=row.get("answer", {}).get("answer"),
        )
        compact.append(simple)
    pd.DataFrame(compact).to_parquet(out / "observations.parquet", index=False)
    decision = dict(
        complete=not failures,
        failures=failures,
        phase=phase,
        source_sha=plan["source_sha"],
        plan_sha256=expected,
        attempted=len(scored),
        successful=sum(r["status"] == "completed" for r in scored),
        failed_or_missing=len(expected_keys) - sum(r["status"] == "completed" for r in scored),
        expected=len(expected_keys),
        global_untouched_claim=False,
        byte_matched_claim=False,
        paper_superiority_established=False,
    )
    if not failures and phase == "canary":
        for row in scored:
            if sha(Path(row["logit_capture"])) != row["logit_capture_sha256"]:
                failures.append("canary model-logit tensor capture changed")
        if any(
            r["method"] == "full_kv"
            and r.get("retention_one_candidate_128_logit_words_bitwise") is not True
            for r in scored
        ):
            failures.append("retention-1 identity control missing")
        if any(
            r["method"] != "full_kv"
            and r.get("optimized_parent_128_logit_words_bitwise") is not True
            for r in scored
        ):
            failures.append("optimized parent word control missing")
        for index, job in enumerate(jobs):
            if job["method"] == "full_kv":
                proof = root / phase / f"workers/{index}/parity.json"
                if not proof.exists():
                    failures.append("FullKV generate/reinjection parity artifact missing")
        decision.update(complete=not failures, failures=failures)
    if not failures and phase == "quality":
        from mosaickv.evaluation.cider_resampling import ResampledCider, paired_draw_counts
        from mosaickv.evaluation.metrics.types import MetricSample

        ledger = load_ledger(root)
        references = {s["sample_id"]: tuple(s["references"]) for s in ledger["samples"]}
        methods = plan["protocol"]["methods"]
        groups = {
            m: [
                MetricSample(r["sample_id"], r["answer"]["answer"], references[r["sample_id"]])
                for r in scored
                if r["method"] == m
            ]
            for m in methods
        }
        reducer = ResampledCider(groups)
        counts = paired_draw_counts(
            len(next(iter(groups.values()))), 10000, plan["protocol"]["seed"]
        )
        values = reducer.score(counts, batch_size=16)
        reducer.validate_official(counts[:3], values[:3])
        point_counts = np.ones((1, len(next(iter(groups.values())))), dtype=int)
        points = reducer.score(point_counts)[0]
        reducer.validate_official(point_counts, points[None])
        candidate = reducer.points.index("mosaickv_exact")
        comparisons = []
        for m in methods:
            if m == "mosaickv_exact":
                continue
            col = reducer.points.index(m)
            interval = np.quantile(values[:, candidate] - values[:, col], [0.025, 0.975]).tolist()
            comparisons.append(
                dict(
                    baseline=m,
                    candidate_cider=float(points[candidate]),
                    baseline_cider=float(points[col]),
                    difference=float(points[candidate] - points[col]),
                    adjusted_95_interval=interval,
                    zero_margin_quality_pass=interval[0] >= 0,
                )
            )
        pd.DataFrame(comparisons).to_parquet(out / "quality_comparisons.parquet", index=False)
        decision.update(
            quality_comparisons=comparisons,
            all_quality_gates_pass=all(c["zero_margin_quality_pass"] for c in comparisons),
            corpus_recomputed_inside_each_bootstrap=True,
        )
        grouped = {(r["sample_index"], r["method"]): r for r in compact}
        memory = []
        for method in methods:
            if method == "mosaickv_exact":
                continue
            for field in (
                "initial_cache_bytes",
                "final_cache_bytes",
                "whole_request_peak_allocated_bytes",
            ):
                pairs = [
                    (r[field], grouped[i, method][field])
                    for (i, m), r in grouped.items()
                    if m == "mosaickv_exact"
                ]
                valid = all(type(a) is int and type(b) is int and 0 < a < b for a, b in pairs)
                memory.append(
                    dict(
                        baseline=method,
                        boundary=field,
                        every_sample_smaller=valid,
                        mean_candidate_bytes=float(np.mean([a for a, _ in pairs]))
                        if all(a is not None for a, _ in pairs)
                        else None,
                        mean_baseline_bytes=float(np.mean([b for _, b in pairs]))
                        if all(b is not None for _, b in pairs)
                        else None,
                    )
                )
        pd.DataFrame(memory).to_parquet(out / "storage_comparisons.parquet", index=False)
        decision.update(
            memory_comparisons=memory,
            all_declared_memory_gates_pass=all(r["every_sample_smaller"] for r in memory),
        )
    if not failures and phase == "timing":
        values = [r["end_to_end_seconds"] for r in scored]
        decision.update(
            candidate_only_descriptive_timing=True,
            mean_seconds=float(np.mean(values)),
            p50_seconds=float(np.median(values)),
            p95_seconds=float(np.quantile(values, 0.95)),
            latency_competitor="not_measured_FullKV_excluded",
            all_latency_gates_pass=False,
        )
    write_once(out / "decision.json", decision)
    if not failures and phase == "canary":
        methods = plan["protocol"]["methods"]
        start = plan["protocol"]["canary_samples"]
        stop = start + plan["protocol"]["evaluation_samples"]
        jobs = [
            dict(method=m, samples=list(range(i, min(i + 8, stop))))
            for i in range(start, stop, 8)
            for m in methods
        ]
        random.Random(plan["protocol"]["seed"]).shuffle(jobs)
        launch_phase(root, "quality", jobs)
    elif (
        phase == "quality"
        and decision.get("all_quality_gates_pass")
        and decision.get("all_declared_memory_gates_pass")
    ):
        # Fixed before outcomes. FullKV is not a compressed latency competitor.
        indices = [2 + i * 43 for i in range(6)]
        launch_phase(
            root,
            "timing",
            [
                dict(
                    method="paired_methods",
                    methods=[m for m in plan["protocol"]["methods"] if m != "full_kv"],
                    samples=[i],
                )
                for i in indices
            ],
        )


def timing_comparisons(rows, seed):
    """Paired examples and repetitions, same GPU per example; never FullKV."""
    import numpy as np

    samples = sorted({r["sample_index"] for r in rows})
    methods = sorted({r["method"] for r in rows})
    if "full_kv" in methods or len(samples) != 6:
        raise ValueError("six preregistered paired timing examples; exclude FullKV")
    lookup = {(r["sample_index"], r["method"], r["trial"]): r["end_to_end_seconds"] for r in rows}
    if len(lookup) != len(rows) or len(rows) != 6 * len(methods) * 20:
        raise ValueError("twenty complete paired trials per timing example required")
    arrays = {m: np.array([[lookup[s, m, t] for t in range(20)] for s in samples]) for m in methods}
    if any(not np.isfinite(a).all() or (a <= 0).any() for a in arrays.values()):
        raise ValueError("positive finite synchronized request timings required")
    rng = np.random.default_rng(seed)
    sample_draws = rng.integers(0, 6, (10000, 6))
    trial_draws = rng.integers(0, 20, (10000, 6, 20))
    draws = {m: a[sample_draws[:, :, None], trial_draws].mean((1, 2)) for m, a in arrays.items()}
    return [
        dict(
            baseline=m,
            mean_ratio=float(arrays["mosaickv_exact"].mean() / a.mean()),
            adjusted_ratio_interval=np.quantile(
                draws["mosaickv_exact"] / draws[m], [0.05 / 8, 1 - 0.05 / 8]
            ).tolist(),
            timing_scope="six_fixed_examples_same_GPU_randomized_method_order",
        )
        for m, a in arrays.items()
        if m != "mosaickv_exact"
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha")
    parser.add_argument("--reuse-ledger", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root, args.reuse_ledger)
    elif args.action == "cpu":
        cpu(args.root, args.plan_sha)
    elif args.action.endswith("_controller"):
        phase = args.action.removesuffix("_controller")
        checked(args.root, args.plan_sha)
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
    else:
        worker(args.root, args.plan_sha, args.action, int(os.environ["SLURM_ARRAY_TASK_ID"]))


if __name__ == "__main__":
    main()
