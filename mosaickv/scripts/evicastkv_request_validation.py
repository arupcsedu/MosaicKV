#!/usr/bin/env python3
"""Finite complete-request canary -> disjoint-ID COCO study -> paired timing.

The online selector is an explicitly new precision variant, not the frozen
graph-based exact-only paper method. No automatic paper/candidate promotion.
"""

import argparse
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import evicastkv_lifetime_history as common
import evicastkv_lifetime_v2 as launch
from evicastkv_campaign import environment

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once
from mosaickv.evaluation.request_precision import METHODS, REQUEST_POLICY

SOURCE = Path(__file__).resolve().parents[2]
BASE = Path("/scratch/djy8hg/runs/evicastkv/request-validation-20261030-v2-unprofiled")
PARENT = Path(
    "/scratch/djy8hg/runs/evicastkv/precision-corrections-consumed-20261030-v1/diagnostic"
)
INVENTORY = Path(
    "/scratch/djy8hg/runs/evicastkv/fox-review-20261028-opportunity-history-refresh-v1"
)
REVISION = "c3213b54e173ae86517d481de17f31439d6f8c72"
ARROW = (
    Path("/scratch/djy8hg/cache/mosaickv/datasets/lmms-lab___coco-caption/default/0.0.0") / REVISION
)
COCO_POOL = Path(
    "/scratch/djy8hg/runs/mosaickv/coco-shared-pool-edb6edf-retry1-20261014/ledger.json"
)
ARMS = [("full_kv", "ordinary")] + [(m, p) for m in METHODS for p in ("lossless", "query3")]
SEED = 20261031


def read(path):
    return json.loads(Path(path).read_text())


def rgb_identity(path):
    from PIL import Image

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return hashlib.sha256(str(rgb.size).encode() + b"/RGB/" + rgb.tobytes()).hexdigest()


def prepare_canary(root):
    if root.exists() or root.name != "canary":
        raise ValueError("new bounded canary root required")
    parent = read(PARENT / "plan.json")
    bound = {str(PARENT / "plan.json"): sha(PARENT / "plan.json")}
    cfg = parent["jobs"][0]["arms"][0]["config"]
    profile = cfg["bounded_decode"]["incremental_history"]["layer_profile_path"]
    bound[profile] = sha(Path(profile))
    jobs = []
    seen = set()
    for old in parent["jobs"]:
        if old["sample_id"] in seen:
            continue
        seen.add(old["sample_id"])
        payload = read(old["input"])
        payload.pop("fullkv_prefix_tokens")
        path = root / f"inputs/{len(jobs)}.json"
        write_once(path, payload)
        bound[str(path)] = sha(path)
        bound.update({m["path"]: m["sha256"] for m in payload["media"]})
        jobs.append(dict(index=len(jobs), inputs=[str(path)], config=old["arms"][0]["config"]))
    protocol = SOURCE / "mosaickv/docs/request_precision_validation_20261030.md"
    bound[str(protocol)] = sha(protocol)
    plan = dict(
        source_sha=clean_source(SOURCE),
        bound_files=bound,
        jobs=jobs,
        config=cfg,
        profile_path=profile,
        arms=ARMS,
        stage="canary",
        protocol="complete_request_profiled_query_precision_v2_unprofiled",
        ratio=0.5,
        output_tokens=128,
        seed=SEED,
        warmups=0,
        maximum_a100s=6,
        requested_workers=2,
        cpus_per_worker=12,
        automatic_bounded_cohort_after_correctness=True,
        cohort_size=128,
        shard_size=8,
        cohort_warmups=5,
        timing_inputs=2,
        timing_warmups=5,
        timing_repetitions=20,
        bootstrap_replicates=10000,
        bootstrap_seed=SEED,
        shared_encoded_byte_budget_claim=False,
        paper_campaign_authorized=False,
        no_label_tuning=True,
        request_policy=REQUEST_POLICY,
        numerical_guard=dict(
            horizon=32, maximum_logit_KL=0.02, maximum_logit_difference=1.0, token_agreement=1.0
        ),
        request_tensor_auditing_enabled=False,
        allocator_peak_counters_remain_enabled=True,
    )
    write_once(root / "plan.json", plan)
    return sha(root / "plan.json")


def load_messages(payload):
    from PIL import Image

    from mosaickv.evaluation.messages import MediaItem, MediaKind, build_multimodal_messages

    if set(payload) != {"task", "sample_id", "prompt", "media"}:
        raise ValueError("worker input may contain neither references nor draft tokens")
    media = []
    for item in payload["media"]:
        if item["kind"] != "image":
            raise ValueError("image route only")
        with Image.open(item["path"]) as img:
            media.append(MediaItem(MediaKind.IMAGE, img.convert("RGB").copy()))
    return build_multimodal_messages(payload["prompt"], tuple(media))


def resolved_mapping(plan, job):
    mapping = dict(job.get("config", plan["config"]))
    mapping["execution"] = dict(mapping["execution"], seed=plan["seed"])
    return mapping


def balanced_pair_orders(arms, repetitions, seed):
    if len(arms) != 2 or repetitions < 2 or repetitions % 2:
        raise ValueError("two arms and an even paired repetition count required")
    flips = [i % 2 for i in range(repetitions)]
    random.Random(seed).shuffle(flips)
    return [list(reversed(arms)) if flip else list(arms) for flip in flips]


def worker(root, expected, index):
    import torch

    from mosaickv.adapters.huggingface import load_hf_adapter
    from mosaickv.adapters.huggingface.parity import (
        compare_cache_reinjection,
        compare_with_generate,
    )
    from mosaickv.config import RunConfig, canonical_config, config_sha256
    from mosaickv.evaluation.request_precision import complete_request
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    plan = common.inputs(root, expected)
    if plan.get("request_policy") != REQUEST_POLICY:
        raise ValueError("resolved request policy drifted from immutable source")
    job = plan["jobs"][index]
    if "method_arms" in job:
        plan = dict(plan, arms=job["method_arms"])
    out = root / f"workers/{index}"
    out.mkdir(parents=True, exist_ok=False)
    result = dict(
        status="failed",
        failures=[],
        source_sha=plan["source_sha"],
        plan_sha256=expected,
        index=index,
    )
    started = time.monotonic()
    try:
        if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name():
            raise ValueError("one allocated A100 required")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ValueError("deterministic CUDA workspace missing")
        torch.set_num_threads(12)
        torch.manual_seed(plan["seed"])
        torch.use_deterministic_algorithms(True)
        cfg = RunConfig.from_mapping(resolved_mapping(plan, job))
        if (
            cfg.model.id != "llava-hf/llava-1.5-7b-hf"
            or cfg.model.revision != "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
            or cfg.model.precision.value != "bf16"
            or cfg.execution.attention_implementation != "eager"
        ):
            raise ValueError(
                "request variant supports only its pinned LLaVA/BF16/HF-eager combination"
            )
        adapter = load_hf_adapter(
            cfg.model.id,
            revision=cfg.model.revision,
            model_kwargs=_model_kwargs(cfg, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        first = read(job["inputs"][0])
        parity = {}
        for name, fn in (
            ("generate", compare_with_generate),
            ("reinjection", compare_cache_reinjection),
        ):
            report = fn(adapter, adapter.prepare_inputs(load_messages(first)), max_new_tokens=16)
            parity[name] = asdict(report)
            if report.token_agreement != 1 or report.maximum_logit_difference > 0.001:
                raise RuntimeError("ordinary FullKV generate/reinjection parity failed")
        write_once(out / "ordinary_parity.json", parity)
        from mosaickv.adapters.huggingface.registry import _resolve_model_source

        snapshot = Path(
            _resolve_model_source(cfg.model.id, cfg.model.revision, local_files_only=True)
        )
        processor_files = {
            str(p): sha(p)
            for p in snapshot.iterdir()
            if p.suffix in (".json", ".model", ".jinja", ".txt")
        }
        write_once(
            out / "manifest.json",
            dict(
                source_sha=plan["source_sha"],
                source_dirty=False,
                config_sha256=config_sha256(cfg),
                plan_sha256=expected,
                base_loader_and_legacy_baseline_config=canonical_config(cfg),
                request_policy=plan["request_policy"],
                source_slot_ratio=plan["ratio"],
                model=asdict(cfg.model),
                execution=asdict(cfg.execution),
                dataset=asdict(cfg.dataset)
                if plan["stage"] == "canary"
                else dict(id="lmms-lab/COCO-Caption", revision=REVISION, split="val"),
                environment=environment(),
                cuda_runtime=torch.version.cuda,
                gpu=asdict(capture_gpu_environment(torch)),
                seed=plan["seed"],
                input_hashes={p: sha(Path(p)) for p in job["inputs"]},
                environment_lock_sha256=sha(SOURCE / "mosaickv/env/common/requirements.lock"),
                baseline_commit=plan["source_sha"],
                evaluator_commit=plan["source_sha"],
                processor_assets=processor_files,
                processor_revision=cfg.model.revision,
                vllm="not_used",
                sglang="not_used",
                evaluator="lmms-eval==0.7.2/pycocoevalcap==1.2/CIDEr_corpus",
                baseline_origin="local_append_only_generalized_reimpl_not_official",
                prototype_or_repair=False,
                precision_policy_offered_identically_to_all_methods=True,
                measurement_type="complete_request_online_precision_variant_validation",
                result_category="unified_comparison",
                result_origin="locally_measured",
                no_shared_byte_matching_claim=True,
                slurm_job_id=os.environ.get("SLURM_JOB_ID"),
            ),
        )
        rng = random.Random(plan["seed"] + index)
        paired_orders = (
            balanced_pair_orders(plan["arms"], plan["timed_repetitions"], plan["seed"] + index)
            if plan["stage"] == "timing"
            else None
        )
        logits_saved = {}
        deterministic_hashes = {}
        with (out / "raw.jsonl").open("x") as stream:
            for local_index, path in enumerate(job["inputs"]):
                payload = read(path)
                arms = list(plan["arms"])
                for rep in range(
                    -plan["warmups"] if local_index == 0 else 0, plan.get("timed_repetitions", 1)
                ):
                    if paired_orders is not None and rep >= 0:
                        arms = paired_orders[rep]
                    else:
                        rng.shuffle(arms)
                    for order, (method, precision) in enumerate(arms):
                        row = dict(
                            task=payload["task"],
                            sample_id=payload["sample_id"],
                            method=method,
                            precision=precision,
                            trial=rep,
                            warmup=rep < 0,
                            order=order,
                            source_sha=plan["source_sha"],
                            plan_sha256=expected,
                            manifest_path=str(out / "manifest.json"),
                            status="failed",
                        )
                        try:
                            logits, measured = complete_request(
                                adapter,
                                lambda payload=payload: load_messages(payload),
                                cfg,
                                method=method,
                                precision=precision,
                                profile_path=plan["profile_path"],
                                ratio=plan["ratio"],
                                audit=False,
                            )
                            row.update(measured, status="completed")
                            # Catalog serialization is artifact work, outside
                            # the complete-request inference timer for everyone.
                            selection = row["online_variant"].get("selection", {})
                            catalog = selection.pop("selected_source_positions_by_layer", None)
                            if catalog is not None:
                                selection["source_position_catalog_sha256"] = hashlib.sha256(
                                    json.dumps(catalog, separators=(",", ":")).encode()
                                ).hexdigest()
                                selection["source_position_count"] = sum(
                                    len(h) for layer in catalog for h in layer
                                )
                            row["method_label"] = (
                                "EviCastKV_online_profiled_precision_v1_" + precision
                                if method == "mosaickv_exact"
                                else method + "_common_" + precision
                            )
                            row["config_sha256"] = hashlib.sha256(
                                json.dumps(
                                    dict(
                                        base_config_sha256=config_sha256(cfg),
                                        request_policy=plan["request_policy"],
                                        actual_method=method,
                                        actual_precision=precision,
                                        resolved_online_policy=plan.get(
                                            "explicit_per_policy_overrides", {}
                                        ).get(precision, {}),
                                        source_slot_ratio=plan["ratio"],
                                        resolved_cache_geometry=selection.get("cache_config", {}),
                                        baseline_parameters=selection.get(
                                            "baseline_parameters", {}
                                        ),
                                        profile_sha256=plan["bound_files"][plan["profile_path"]],
                                    ),
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ).encode()
                            ).hexdigest()
                            row["resolved_online_policy"] = plan.get(
                                "explicit_per_policy_overrides", {}
                            ).get(precision, {})
                            row["greedy_ids"] = logits.argmax(-1).tolist()
                            digest = hashlib.sha256(logits.numpy().tobytes()).hexdigest()
                            identity = (payload["sample_id"], method, precision)
                            if (
                                identity in deterministic_hashes
                                and deterministic_hashes[identity] != digest
                            ):
                                raise RuntimeError("repeated full request changed its own logits")
                            deterministic_hashes[identity] = digest
                            row["logits_sha256"] = digest
                            if rep == 0 and plan["stage"] == "canary":
                                logits_saved[f"{local_index}-{method}-{precision}"] = logits
                            del logits
                        except Exception as error:
                            row["error"] = f"{type(error).__name__}: {error}"
                            result["failures"].append(
                                dict(
                                    sample_id=payload["sample_id"],
                                    method=method,
                                    precision=precision,
                                    error=row["error"],
                                )
                            )
                            gc.collect()
                        stream.write(json.dumps(row) + "\n")
                        stream.flush()
                # All methods ran on this GPU in counterbalanced order. A-B-A
                # control recomputes the first source input, never reuses cache.
                if local_index == 0 and not result["failures"] and plan["stage"] != "timing":
                    control_method, control_precision = (
                        plan["arms"][0] if "method_arms" in job else ("mosaickv_exact", "query3")
                    )
                    logits, _ = complete_request(
                        adapter,
                        lambda payload=payload: load_messages(payload),
                        cfg,
                        method=control_method,
                        precision=control_precision,
                        profile_path=plan["profile_path"],
                        ratio=plan["ratio"],
                        audit=False,
                    )
                    digest = hashlib.sha256(logits.numpy().tobytes()).hexdigest()
                    if (
                        digest
                        != deterministic_hashes[
                            (payload["sample_id"], control_method, control_precision)
                        ]
                    ):
                        raise RuntimeError("A-B-A logits isolation failed")
                    write_once(
                        out / "isolation.json",
                        dict(
                            recomputed_request=True,
                            source_retired=True,
                            unrelated_methods_between_requests=len({a[0] for a in plan["arms"]})
                            > 1,
                            distinct_policies_between_requests=len({a[1] for a in plan["arms"]})
                            > 1,
                            checked_bitwise_logits=True,
                        ),
                    )
                    del logits
        if logits_saved:
            identity, control = complete_request(
                adapter,
                lambda: load_messages(first),
                cfg,
                method="mosaickv_exact",
                precision="query3",
                profile_path=plan["profile_path"],
                ratio=1.0,
                audit=False,
            )
            if not torch.equal(identity, logits_saved["0-full_kv-ordinary"]):
                raise RuntimeError("128-token retention-one identity failed")
            write_once(out / "retention_one.json", dict(bitwise_logits=True, record=control))
            torch.save(logits_saved, out / "logits.pt")
        result["status"] = "completed" if not result["failures"] else "completed_with_errors"
    except Exception as error:
        result["failures"].append(f"{type(error).__name__}: {error}")
        raise
    finally:
        result["wall_seconds"] = time.monotonic() - started
        result["outputs"] = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
        write_once(out / "result.json", result)


def materialize_cohort(parent, root):
    from datasets import Dataset
    from materialize_coco_shared_bytes import freeze_sample

    if root.exists():
        raise ValueError("new cohort directory required")
    old = read(parent / "plan.json")
    bound = dict(old["bound_files"])
    bound[str(parent / "analysis/GATE.json")] = sha(parent / "analysis/GATE.json")
    excluded = {"COCO_val2014_000000532947.jpg"}
    for name in ("mosaickv-inventory.json", "evicastkv-inventory.json"):
        path = INVENTORY / name
        bound[str(path)] = sha(path)
        excluded.update(str(s).split(":")[-1] for s in read(path)["COCO_consumed_ids"])
    pool = read(COCO_POOL)
    bound[str(COCO_POOL)] = sha(COCO_POOL)
    excluded.update(str(s["sample_id"]).split(":")[-1] for s in pool["samples"])
    historical_media = {m["path"]: m["sha256"] for s in pool["samples"] for m in s["media"]}
    for job in old["jobs"]:
        for target in job["inputs"]:
            historical_media.update({m["path"]: m["sha256"] for m in read(target)["media"]})
    from concurrent.futures import ThreadPoolExecutor

    def checked_pixels(item):
        path, digest = item
        if sha(Path(path)) != digest:
            raise ValueError("historical media changed during disjointness audit")
        return rgb_identity(path)

    with ThreadPoolExecutor(max_workers=2) as executor:
        historical_pixels = set(executor.map(checked_pixels, historical_media.items()))
    candidates = []
    for shard in range(14):
        path = ARROW / f"coco-caption-val-{shard:05d}-of-00014.arrow"
        bound[str(path)] = sha(path)
        table = Dataset.from_file(str(path)).select_columns(["question_id", "file_name"])
        for offset, item in enumerate(table):
            sid = str(item["question_id"])
            filename = str(item["file_name"])
            if sid != filename or not filename.startswith("COCO_val2014_"):
                raise ValueError("authoritative COCO filename identity mismatch")
            if sid not in excluded:
                candidates.append((sid, path, offset))
    ordered = sorted(
        candidates, key=lambda row: hashlib.sha256(f"{SEED}/{row[0]}".encode()).hexdigest()
    )
    if len({row[0] for row in ordered}) != len(ordered) or len(ordered) < old["cohort_size"]:
        raise ValueError("insufficient unique disjoint-ID COCO candidates")
    selected = ordered[: old["cohort_size"]]
    media_hashes, samples, inputs = set(), [], []
    for sid, path, offset in selected:
        sample = freeze_sample(Dataset.from_file(str(path))[offset], "coco_caption:" + sid, root)
        digest = sample["media"][0]["sha256"]
        pixels = rgb_identity(sample["media"][0]["path"])
        if pixels in historical_pixels:
            raise ValueError(
                "selected image repeats historical RGB media; no replacement authorized"
            )
        if digest in media_hashes:
            raise ValueError("duplicate cohort media content")
        media_hashes.add(digest)
        samples.append(sample)
        payload = {k: sample[k] for k in ("sample_id", "prompt", "media")}
        payload["task"] = "coco_caption"
        target = root / f"inputs/{len(inputs)}.json"
        write_once(target, payload)
        inputs.append(str(target))
        bound[str(target)] = sha(target)
        bound.update({m["path"]: m["sha256"] for m in sample["media"]})
    ledger = root / "references.json"
    write_once(
        ledger,
        dict(
            dataset_id="lmms-lab/COCO-Caption",
            dataset_revision=REVISION,
            split="val",
            samples=samples,
            global_untouched_claim=False,
            disjoint_from_explicit_registered_COCO_inventories=True,
        ),
    )
    bound[str(ledger)] = sha(ledger)
    write_once(
        root / "disjointness.json",
        dict(
            excluded_coco_ids=len(excluded),
            selected_ids=[s[0] for s in selected],
            overlap=sorted(excluded & {s[0] for s in selected}),
            independent_of_candidate_tuning_ids=True,
            audited_historical_exact_RGB_media=len(historical_media),
            selected_media_overlap_with_explicit_historical_pool=False,
            historical_RGB_identity_set_sha256=hashlib.sha256(
                json.dumps(sorted(historical_pixels)).encode()
            ).hexdigest(),
            global_cross_dataset_media_history_certified=False,
            limitation=(
                "new IDs relative to bound inventories; no globally untouched confirmation claim"
            ),
        ),
    )
    jobs = [
        dict(index=i // old["shard_size"], inputs=inputs[i : i + old["shard_size"]])
        for i in range(0, len(inputs), old["shard_size"])
    ]
    provenance = root / "dataset_provenance.json"
    arrow_files = {p: bound.pop(p) for p in tuple(bound) if p.endswith(".arrow")}
    write_once(
        provenance,
        dict(
            pinned_raw_split_files=arrow_files,
            verified_before_materialization=True,
            no_other_splits_prepared=True,
        ),
    )
    bound[str(provenance)] = sha(provenance)
    resolved_config = dict(
        old["config"], dataset=dict(id="lmms-lab/COCO-Caption", revision=REVISION, split="val")
    )
    plan = dict(
        old,
        config=resolved_config,
        source_sha=clean_source(SOURCE),
        stage="cohort",
        jobs=jobs,
        bound_files=bound,
        references=str(ledger),
        warmups=old["cohort_warmups"],
        automatic_bounded_cohort_after_correctness=False,
        cohort_manifest=str(root / "disjointness.json"),
        requested_workers=len(jobs),
    )
    write_once(root / "plan.json", plan)
    return sha(root / "plan.json")


def analyze(root, expected):
    import numpy as np
    import pandas as pd

    plan = common.inputs(root, expected)
    output = root / "analysis"
    if output.exists():
        raise ValueError("new write-once analysis required")
    output.mkdir()
    rows, failures = [], []
    for job in plan["jobs"]:
        folder = root / f"workers/{job['index']}"
        result_path = folder / "result.json"
        if not result_path.exists():
            failures.append(dict(index=job["index"], error="missing worker result"))
            continue
        result = read(result_path)
        failures.extend(result["failures"])
        if result["source_sha"] != plan["source_sha"] or result["plan_sha256"] != expected:
            failures.append(dict(index=job["index"], error="source/plan binding changed"))
        for name, digest in result["outputs"].items():
            if sha(folder / name) != digest:
                raise ValueError("worker evidence changed")
        path = folder / "raw.jsonl"
        if path.exists():
            rows.extend(json.loads(line) for line in path.read_text().splitlines())
    scored = [r for r in rows if not r["warmup"]]
    expected_rows = (
        sum(len(j["inputs"]) for j in plan["jobs"])
        * len(plan["arms"])
        * plan.get("timed_repetitions", 1)
    )
    keys = {(r["sample_id"], r["method"], r["precision"], r["trial"]) for r in scored}
    complete = len(keys) == expected_rows and len(scored) == expected_rows and not failures
    quality_guard = []
    if any(r.get("audit_enabled") is True for r in scored):
        complete = False
        failures.append("tensor-workspace auditing contaminated an unprofiled timing request")
    if complete and any(
        r["status"] != "completed"
        or len(r["greedy_ids"]) != 128
        or not math.isfinite(r["end_to_end_seconds"])
        or r["end_to_end_seconds"] <= 0
        or not r["whole_request_allocator_peak_certified"]
        or (r["method"] != "full_kv" and r.get("audit_enabled") is not False)
        or (r["method"] != "full_kv" and not r["source_KV_owner_retired_before_decode"])
        for r in scored
    ):
        complete = False
        failures.append("request accounting, output or timing incomplete")
    if plan["stage"] == "canary" and complete:
        import torch
        from evicastkv_packed_attention_consumed import differences

        for job in plan["jobs"]:
            tensors = torch.load(
                root / f"workers/{job['index']}/logits.pt", map_location="cpu", weights_only=True
            )
            comparison = differences(
                tensors["0-mosaickv_exact-lossless"][:32], tensors["0-mosaickv_exact-query3"][:32]
            )
            quality_guard.append(dict(index=job["index"], **comparison))
        complete = all(
            r["token_agreement"] == 1
            and r["maximum_logit_KL"] <= 0.02
            and r["maximum_logit_difference"] <= 1
            for r in quality_guard
        )
        if not complete:
            failures.append("new online variant failed unchanged precision guard; no cohort")
    pd.DataFrame(scored).to_parquet(output / "observations.parquet", index=False)
    gate = dict(
        complete=complete,
        failures=failures,
        expected_rows=expected_rows,
        attempted_rows=len(scored),
        successful_rows=sum(r["status"] == "completed" for r in scored),
        numeric_guard=quality_guard,
        no_byte_matched_claim=True,
        whole_request_allocator_peak_measured=True,
        whole_request_KV_only_peak_certified=False,
        superiority_established=False,
    )
    write_once(output / "GATE.json", gate)
    if plan["stage"] == "canary" and complete:
        child = root.parent / "cohort"
        digest = materialize_cohort(root, child)
        launch.submit(
            child, digest, script_name="evicastkv_request_validation.sbatch", memory_gb=48
        )
        write_once(root / "next_stage.json", dict(cohort=str(child), plan_sha256=digest))
    elif plan["stage"] == "cohort" and complete:
        # Metric observations and uncertainty stay separate from raw executions.
        from mosaickv.evaluation.cider_resampling import ResampledCider, paired_draw_counts
        from mosaickv.evaluation.metrics.types import MetricSample

        reference = {
            s["sample_id"]: tuple(s["references"]) for s in read(plan["references"])["samples"]
        }
        groups = {}
        for r in scored:
            point = (r["method"] + ":" + r["precision"], 0.5)
            groups.setdefault(point, []).append(
                MetricSample(
                    r["sample_id"], r["scored_answer"]["answer"], reference[r["sample_id"]]
                )
            )
        scorer = ResampledCider(groups)
        draw = paired_draw_counts(
            len(reference), plan["bootstrap_replicates"], plan["bootstrap_seed"]
        )
        counts = np.vstack((np.ones((1, len(reference)), dtype=np.int64), draw))
        values = scorer.score(counts, batch_size=16)
        error = scorer.validate_official(counts[:4], values[:4])
        candidate = scorer.points.index(("mosaickv_exact:query3", 0.5))
        metrics = [
            dict(method=p[0], cider=float(values[0, i])) for i, p in enumerate(scorer.points)
        ]
        comparisons = []
        eligible = [
            i for i, p in enumerate(scorer.points) if p[0] != "full_kv:ordinary" and i != candidate
        ]
        alpha = 0.05 / len(eligible)
        for i in eligible:
            delta = values[1:, candidate] - values[1:, i]
            lo, hi = np.quantile(delta, [alpha / 2, 1 - alpha / 2])
            from mosaickv.evaluation.request_statistics import compare_requests

            paired = compare_requests(
                [
                    r
                    for r in scored
                    if r["method"] == "mosaickv_exact" and r["precision"] == "query3"
                ],
                [r for r in scored if r["method"] + ":" + r["precision"] == scorer.points[i][0]],
                resamples=10000,
                seed=SEED,
            )
            comparisons.append(
                dict(
                    candidate="mosaickv_exact:query3",
                    baseline=scorer.points[i][0],
                    cider_difference=float(values[0, candidate] - values[0, i]),
                    simultaneous_ci95_bonferroni=[float(lo), float(hi)],
                    strict_quality_pass=bool(lo >= 0),
                    paired_execution=paired,
                    joint_development_gate=bool(
                        lo >= 0
                        and paired["latency_no_slower"]
                        and all(m["all_inputs_smaller"] for m in paired["active_storage"].values())
                    ),
                )
            )
        pd.DataFrame(metrics).to_parquet(output / "cider.parquet", index=False)
        write_once(
            output / "quality_comparisons.json",
            dict(
                comparisons=comparisons,
                bootstrap_replicates=len(draw),
                reducer_recomputed_per_draw=True,
                official_reducer_max_error=error,
                unique_quality_N=len(reference),
                global_untouched_confirmation_claim=False,
                superiority_established=False,
            ),
        )
        # Preregistered rule: timing's comparator is the highest-CIDEr measured
        # compressed baseline point, including the common precision variant;
        # ties use smaller mean initial backing, then method ID. Never choose
        # an easier baseline after seeing a latency result.
        baseline_points = [
            i for i in eligible if not scorer.points[i][0].startswith("mosaickv_exact:")
        ]
        strongest = max(
            baseline_points,
            key=lambda i: (
                values[0, i],
                -np.mean(
                    [
                        r["storage"]["initial_active_kv_storage_bytes"]
                        for r in scored
                        if r["method"] + ":" + r["precision"] == scorer.points[i][0]
                    ]
                ),
                scorer.points[i][0],
            ),
        )
        target = scorer.points[strongest][0].split(":")
        timing = root.parent / "timing"
        paths = [p for j in plan["jobs"] for p in j["inputs"]][: plan["timing_inputs"]]
        child = dict(
            plan,
            stage="timing",
            source_sha=clean_source(SOURCE),
            jobs=[dict(index=i, inputs=[p]) for i, p in enumerate(paths)],
            arms=[["mosaickv_exact", "query3"], target],
            selected_strongest_baseline=scorer.points[strongest][0],
            warmups=plan["timing_warmups"],
            timed_repetitions=plan["timing_repetitions"],
            requested_workers=len(paths),
        )
        child["bound_files"] = dict(
            plan["bound_files"],
            **{
                str(output / "quality_comparisons.json"): sha(output / "quality_comparisons.json"),
                str(output / "cider.parquet"): sha(output / "cider.parquet"),
            },
        )
        write_once(timing / "plan.json", child)
        launch.submit(
            timing,
            sha(timing / "plan.json"),
            script_name="evicastkv_request_validation.sbatch",
            memory_gb=48,
        )
    elif plan["stage"] == "timing" and complete:
        pairs = []
        for sample_id in sorted({r["sample_id"] for r in scored}):
            by_trial = {}
            for r in scored:
                if r["sample_id"] == sample_id:
                    by_trial.setdefault(r["trial"], {})[r["method"] + ":" + r["precision"]] = r
            ratios = []
            for values_ in by_trial.values():
                a, b = (
                    values_["mosaickv_exact:query3"],
                    values_[plan["selected_strongest_baseline"]],
                )
                ratios.append(a["end_to_end_seconds"] / b["end_to_end_seconds"])
            if len(ratios) != 20:
                raise ValueError("all twenty paired complete-request repetitions required")
            ratios = np.asarray(ratios)
            boot = np.median(
                ratios[np.random.default_rng(SEED).integers(0, 20, (10000, 20))], axis=1
            )
            pairs.append(
                dict(
                    sample_id=sample_id,
                    median_ratio=float(np.median(ratios)),
                    technical_ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
                    repeat_count=20,
                    no_extra_quality_samples=True,
                )
            )
        write_once(
            output / "paired_latency.json",
            dict(
                strongest_baseline=plan["selected_strongest_baseline"],
                comparisons=pairs,
                independent_timing_input_count=len(pairs),
                latency_win=all(p["technical_ci95"][1] <= 1 for p in pairs),
                no_general_superiority_claim_from_two_timing_inputs=True,
            ),
        )
    write_once(
        output / "manifest.json",
        dict(
            source_sha=plan["source_sha"],
            plan_sha256=expected,
            inputs={
                str(root / f"workers/{j['index']}/result.json"): sha(
                    root / f"workers/{j['index']}/result.json"
                )
                for j in plan["jobs"]
                if (root / f"workers/{j['index']}/result.json").exists()
            },
            outputs={p.name: sha(p) for p in output.iterdir()},
            no_paper_promotion=True,
        ),
    )


def cpu(root, expected):
    import xml.etree.ElementTree as ET

    common.inputs(root, expected)
    xml = root / "cpu.xml"
    command = [
        "/scratch/djy8hg/env/mosaickv/bin/python",
        "-m",
        "pytest",
        "-q",
        "mosaickv/tests/unit",
        "mosaickv/tests/integration",
        f"--junitxml={xml}",
    ]
    result = subprocess.run(command, cwd=SOURCE, check=False)
    tests = ET.parse(xml).getroot().findall(".//testcase") if xml.exists() else []
    write_once(
        root / "cpu.json",
        dict(
            source_sha=clean_source(SOURCE),
            returncode=result.returncode,
            command=command,
            test_count=len(tests),
            xml_sha256=sha(xml) if xml.exists() else None,
            failed_or_skipped=sum(
                any(t.find(k) is not None for k in ("failure", "error", "skipped")) for t in tests
            ),
        ),
    )
    if result.returncode:
        raise SystemExit(result.returncode)


def admit(root, expected):
    plan = common.inputs(root, expected)
    record = read(root / "cpu.json")
    if (
        record["source_sha"] != plan["source_sha"]
        or record["returncode"] != 0
        or record["failed_or_skipped"]
        or record["test_count"] < 2744
        or record["xml_sha256"] != sha(root / "cpu.xml")
    ):
        raise ValueError("complete clean CPU regression proof required")
    queue = subprocess.check_output(
        ["squeue", "-u", os.environ["USER"], "-h", "-o", "%T|%b"], text=True
    )
    if any("a100" in line.lower() for line in queue.splitlines()):
        raise ValueError("admit no new A100 array while another user-owned array exists")
    launch.submit(root, expected, script_name="evicastkv_request_validation.sbatch", memory_gb=48)


def seal_profiled(root, expected):
    if sha(root / "plan.json") != expected:
        raise ValueError("profiled parent plan changed")
    parent = read(root / "plan.json")
    if parent["protocol"] != "complete_request_profiled_query_precision_v1":
        raise ValueError("only the superseded tensor-audited parent may be sealed")
    records, hashes = [], {str(root / "plan.json"): expected}
    for job in parent["jobs"]:
        folder = root / f"workers/{job['index']}"
        path = folder / "result.json"
        if not path.exists():
            records.append(
                dict(worker=job["index"], status="missing_result", error="incomplete producer")
            )
            continue
        result = read(path)
        if result["source_sha"] != parent["source_sha"] or result["plan_sha256"] != expected:
            raise ValueError("parent source binding changed")
        hashes[str(path)] = sha(path)
        for name, digest in result["outputs"].items():
            if sha(folder / name) != digest:
                raise ValueError("parent evidence changed")
            hashes[str(folder / name)] = digest
        rows = [json.loads(line) for line in (folder / "raw.jsonl").read_text().splitlines()]
        records.append(
            dict(
                worker=job["index"],
                status=result["status"],
                failures=result["failures"],
                attempted_rows=len(rows),
                scored_rows=sum(r["status"] == "completed" for r in rows),
                tensor_audited_rows=sum(r.get("audit_enabled") is True for r in rows),
            )
        )
    write_once(
        root / "derived_measurement_seal/decision.json",
        dict(
            source_sha=clean_source(SOURCE),
            producer_sha=parent["source_sha"],
            inputs=hashes,
            records=records,
            classification="preserved_profiled_correctness_diagnostic_not_unprofiled_performance",
            comparative_latency_admissible=False,
            automatic_cohort_launcher_cancelled="22207828",
            reason="TorchDispatch workspace observation overhead was inside the measured requests",
            replacement_protocol="complete_request_profiled_query_precision_v2_unprofiled",
            superiority_established=False,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("prepare", "submit", "worker", "analyze", "cpu", "admit", "seal-profiled")
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha")
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.mode == "prepare":
        print(prepare_canary(args.root))
    elif args.mode in ("submit", "admit"):
        admit(args.root, args.plan_sha)
    elif args.mode == "cpu":
        cpu(args.root, args.plan_sha)
    elif args.mode == "seal-profiled":
        seal_profiled(args.root, args.plan_sha)
    elif args.mode == "worker":
        worker(
            args.root,
            args.plan_sha,
            args.index if args.index is not None else int(os.environ["SLURM_ARRAY_TASK_ID"]),
        )
    else:
        analyze(args.root, args.plan_sha)
