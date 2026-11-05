#!/usr/bin/env python3
"""Verify identical direct construction before bounded consumed-input timing."""

import argparse
import gc
import json
import os
import random
import traceback
from pathlib import Path
from unittest.mock import patch

import evicastkv_cached_text_latency as common

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

driver, SOURCE = common.driver, common.SOURCE
PARENT = Path("/scratch/djy8hg/runs/evicastkv/cached-text-consumed-20261103-v1")
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_direct_text_latency.sbatch"


def packet_signature(packets):
    digest = driver.pair.common.control.digest_tensors
    return dict(
        digest=digest(tuple(t for p in packets for t in p.tensors())),
        layers=[digest(p.tensors()) for p in packets],
    )


def submit(root, action, **kwargs):
    with patch.object(common, "SCRIPT", SCRIPT):
        return common.submit(root, action, **kwargs)


def stage(root, action, analyzer):
    gpu = submit(root, action, gpu=True)
    analysis = submit(root, analyzer, dependency=gpu)
    write_once(root / "submission.json", dict(gpu_array=gpu, analysis=analysis))


def prepare(root, revision):
    if root.exists() or clean_source(SOURCE) != revision:
        raise ValueError("new root and immutable clean source required")
    parent = driver.read(PARENT / "analysis/decision.json")
    if not parent["complete"] or "text-6291456" not in parent["meaningful_common_text_arms"]:
        raise ValueError("both validated frozen text-row policies required")
    plan = driver.read(PARENT / "plan.json")
    bound = dict(plan["bound_files"])
    files = [
        PARENT / "analysis/decision.json",
        SOURCE / "mosaickv/docs/direct_text_latency_20261103.md",
    ]
    files.extend(PARENT / f"workers/{i}/study.json" for i in (0, 1))
    negative = Path("/scratch/djy8hg/runs/evicastkv/direct-text-consumed-20261103-v1")
    if (negative / "analysis/decision.json").exists():
        files.append(negative / "analysis/decision.json")
        files.extend(negative / f"workers/{i}/result.json" for i in (0, 1))
    for path in files:
        bound[str(path)] = sha(path)
    write_once(
        root / "plan.json",
        dict(
            plan,
            source_sha=revision,
            bound_files=bound,
            direct_text_root=str(root),
            stage="direct_identical_text_construction",
            no_fresh_identities=True,
            no_online_timing_search=True,
            no_benchmark_label_access=True,
            immutable_parent_policies=True,
            complete_packet_tensor_digest_required=True,
            paper_campaign_authorized=False,
            automatic_quality_submission=False,
        ),
    )
    write_once(root / "preparation_submission.json", dict(cpu=submit(root, "cpu")))


def cpu(root, expected):
    driver.pair.common.previous.cpu(root, expected)
    driver.cpu_proof(root, driver.checked(root, expected))
    stage(root, "study", "study_analysis")


def execute(
    adapter, refs, messages, profile, plan, method, record, *, audit=False, final_validator=None
):
    import torch

    from mosaickv.evaluation.common_streaming_timing import request
    from mosaickv.evaluation.fused_lossless_attention import bits_equal

    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    values, packets, measured = request(
        adapter,
        messages,
        profile,
        method,
        minimum_only=plan["exact_minimum_only"],
        componentwise_fitting=True,
        code_delta=method in plan.get("code_delta_methods", []),
        construction_geometry=plan.get("construction_geometry"),
        reconstruction_readers="carry_literals",
        audit_decode_storage=audit,
        reuse_kernel_dispatch=True,
        reuse_code_descriptors=True,
        source_hotpath="prepared_restore",
        hybrid_reader_policy=record.get("whole_policy"),
        text_overlay_policy=record.get("text_policy"),
        direct_text_policy=record.get("direct_policy"),
        final_packet_validator=final_validator,
        defer_transport_validation=record.get(
            "defer_transport_validation", plan.get("defer_transport_validation", False)
        ),
        coalesce_transport_nonzero=record.get(
            "coalesce_transport_nonzero", plan.get("coalesce_transport_nonzero", False)
        ),
        batch_source_validation=record.get(
            "batch_source_validation", plan.get("batch_source_validation", False)
        ),
        full_predicate_span=record.get(
            "full_predicate_span", plan.get("full_predicate_span", False)
        ),
        prompt_graph_reader=record.get("prompt_graph_reader", False),
        prompt_graph_policy=record.get("prompt_graph_policy"),
        lt_source_reader=record.get("lt_source_reader", False),
        lt_fit_policy=record.get("lt_fit_policy", "full"),
        selective_text_rows=record.get("selective_text_rows", False),
    )
    if not bits_equal(values, refs[method]["logits"]):
        raise RuntimeError("direct text construction changed parent vocabulary words")
    if (
        record.get("direct_policy") is None
        and measured["original_precompression_packet_bytes"]
        != refs[method]["payload"]["physical_bytes"]
    ):
        raise RuntimeError("unchanged preconversion representation bytes differ")
    measured.update(
        logits_bitwise=True,
        no_FullKV_fallback=True,
        torch_peak_allocated=torch.cuda.max_memory_allocated(),
        torch_peak_reserved=torch.cuda.max_memory_reserved(),
        generated_token_ids=values.argmax(-1).tolist(),
        measurement_type="diagnostic_measured",
        independent_quality_N=0,
    )
    del values, packets
    return measured


def study(root, expected, index):
    out = root / f"workers/{index}"
    out.mkdir(parents=True, exist_ok=False)
    result = dict(
        status="failed", failures=[], source_sha=clean_source(SOURCE), plan_sha256=expected
    )
    try:
        plan, adapter, refs, messages, profile, _ = driver.setup(
            root,
            expected,
            index,
            manifest_overrides=dict(
                variant="consumed_direct_text_identical_packet_controls",
                methods=list(common.previous.ALL_METHODS),
                actual_timing_protocol=dict(
                    output_predictions=128, audited_controls_not_latency_evidence=True
                ),
            ),
        )
        parent = driver.read(PARENT / f"workers/{index}/study.json")
        records = {}
        for method in common.previous.ALL_METHODS:
            old = parent["records"][method]["arms"]["text-6291456"]
            policy = old["text_policy"]
            captured = {}

            def capture(packets, captured=captured):
                captured.update(packet_signature(packets))

            cached_record = dict(text_policy=policy)
            cached = execute(
                adapter,
                refs,
                messages,
                profile,
                plan,
                method,
                cached_record,
                audit=True,
                final_validator=capture,
            )
            if any(cached[k] != old["audited"][k] for k in driver.STORAGE):
                raise RuntimeError("same-source cached-text storage changed")
            direct_policy = dict(
                policy,
                expected_private_bytes=cached["cached_text_conversion"][
                    "constructed_private_bytes"
                ],
            )
            direct_record = dict(direct_policy=direct_policy)

            expected_layers = tuple(captured["layers"])

            def require_same(
                packets,
                expected_digest=captured["digest"],
                expected_layers=expected_layers,
                method=method,
            ):
                expected_signature = dict(digest=expected_digest, layers=list(expected_layers))
                actual = packet_signature(packets)
                if actual != expected_signature:
                    changed = [
                        i
                        for i, (a, b) in enumerate(
                            zip(actual["layers"], expected_signature["layers"], strict=True)
                        )
                        if a != b
                    ]
                    write_once(
                        out / f"packet-mismatch-{method}.json",
                        dict(expected=expected_signature, actual=actual, changed_layers=changed),
                    )
                    raise RuntimeError(
                        "direct packet tensor words/shapes/dtypes differ from frozen reader: "
                        f"changed_layers={changed}"
                    )

            direct = execute(
                adapter,
                refs,
                messages,
                profile,
                plan,
                method,
                direct_record,
                audit=True,
                final_validator=require_same,
            )
            if any(direct[k] != cached[k] for k in driver.STORAGE):
                raise RuntimeError("direct initial/final/decode-owned-peak storage changed")
            records[method] = dict(
                cached=dict(cached_record, audited=cached),
                direct=dict(direct_record, audited=direct),
                packet_sha256=captured["digest"],
                all_tensor_words_identical=True,
            )
            if method == "prefixkv_reimpl":
                whole = parent["records"][method]["arms"]["whole-6291456"]
                r = dict(whole_policy=whole["whole_policy"])
                measured = execute(adapter, refs, messages, profile, plan, method, r, audit=True)
                if any(measured[k] != whole["audited"][k] for k in driver.STORAGE):
                    raise RuntimeError("strongest whole-layer control changed storage")
                records[method]["whole-6291456"] = dict(r, audited=measured)
            write_once(out / f"controls-{method}.json", records[method])
        write_once(out / "study.json", dict(records=records, independent_quality_N=0))
        result["status"] = "completed"
    except Exception as error:
        result["failures"].append(f"{type(error).__name__}: {error}")
        result["traceback"] = traceback.format_exc()
    driver.finish_worker(out, result)


def study_analysis(root, expected):
    plan = driver.checked(root, expected)
    rows, failures = driver.load_producers(root, plan, "study.json")
    common.previous.closeout(
        root,
        not failures,
        failures,
        studies=rows,
        no_reader_promoted=True,
        packet_and_storage_identity=not failures,
    )
    if not failures:
        stage(common.previous.child(root, "paired-timing"), "timing", "timing_analysis")


def contrasts(rows, seed):
    import numpy as np

    keys = [(r["method"], r["arm"], r["trial"]) for r in rows]
    arms = {(m, a) for m, a, _ in keys}
    if not {
        ("mosaickv_exact", "cached"),
        ("mosaickv_exact", "direct"),
        ("prefixkv_reimpl", "direct"),
        ("prefixkv_reimpl", "cached"),
        ("prefixkv_reimpl", "whole-6291456"),
    }.issubset(arms):
        raise ValueError("same-reader anchor and all registered PrefixKV controls required")
    if len(keys) != len(set(keys)) or set(keys) != {
        (m, a, t) for m, a in arms for t in range(-5, 20)
    }:
        raise ValueError("complete unique five-warmup/twenty-trial arms required")
    arrays, summaries = {}, []
    for m, a in sorted(arms):
        ordered = sorted(
            (r for r in rows if (r["method"], r["arm"]) == (m, a) and r["trial"] >= 0),
            key=lambda r: r["trial"],
        )
        v = np.asarray([r["measurement"]["total_wall_seconds"] for r in ordered])
        if not np.isfinite(v).all() or not (v > 0).all():
            raise ValueError("finite positive complete-request timings required")
        arrays[m, a] = v
        summaries.append(
            dict(
                method=m,
                arm=a,
                mean=float(v.mean()),
                p50=float(np.median(v)),
                p95=float(np.quantile(v, 0.95)),
            )
        )
    draws = np.random.default_rng(seed).integers(0, 20, (10000, 20))
    v, tail = arrays["mosaickv_exact", "direct"], 0.05 / (2 * 48)

    def comparison(m, a, other):
        ci = np.quantile(v[draws].mean(1) / other[draws].mean(1), [tail, 1 - tail]).tolist()
        return dict(
            method=m, arm=a, mean_ratio=float(v.mean() / other.mean()), adjusted_ratio_CI=ci
        )

    own = comparison("mosaickv_exact", "cached", arrays["mosaickv_exact", "cached"])
    baselines = [comparison(m, a, x) for (m, a), x in arrays.items() if m != "mosaickv_exact"]
    return dict(
        summaries=summaries,
        own=own,
        baselines=baselines,
        own_speedup_gate=own["adjusted_ratio_CI"][1] <= 0.99,
        strongest_prefix_similar_gate=all(
            r["adjusted_ratio_CI"][1] <= 1.05 for r in baselines if r["method"] == "prefixkv_reimpl"
        ),
        all_baselines_no_slower_gate=all(r["adjusted_ratio_CI"][1] <= 1 for r in baselines),
        family_maximum_engineering_contrasts=48,
        intervals_over_repetitions_not_independent_examples=True,
    )


def timing(root, expected, index, *, all_methods=False):
    out = root / f"workers/{index}"
    out.mkdir(parents=True, exist_ok=False)
    result = dict(
        status="failed", failures=[], source_sha=clean_source(SOURCE), plan_sha256=expected
    )
    try:
        methods = common.previous.ALL_METHODS if all_methods else common.previous.METHODS
        plan, adapter, refs, messages, profile, _ = driver.setup(
            root,
            expected,
            index,
            manifest_overrides=dict(
                variant="consumed_direct_text_full_request_paired_latency",
                methods=list(methods),
                actual_timing_protocol=dict(
                    output_predictions=128,
                    warmups=5,
                    repetitions=20,
                    randomized_method_order=True,
                    construction_and_conversion_included=True,
                ),
            ),
        )
        original = driver.read(Path(plan["direct_text_root"]) / f"workers/{index}/study.json")
        arms = [
            (m, a, r)
            for m in methods
            for a, r in original["records"][m].items()
            if a in ("cached", "direct", "whole-6291456")
        ]
        rng, raw = random.Random(plan["seed"]), []
        for trial in range(-5, 20):
            placement = list(arms)
            rng.shuffle(placement)
            for method, arm, record in placement:
                measured = execute(adapter, refs, messages, profile, plan, method, record)
                if any(measured[k] != record["audited"][k] for k in driver.STORAGE[:2]):
                    raise RuntimeError("timed cache differs from independent audited companion")
                row = dict(
                    method=method,
                    arm=arm,
                    trial=trial,
                    warmup=trial < 0,
                    measurement=measured,
                    source_sha=plan["source_sha"],
                    plan_sha256=expected,
                )
                raw.append(row)
                with (out / "trials.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
        write_once(out / "timing.json", contrasts(raw, plan["seed"]))
        result["status"] = "completed"
    except Exception as error:
        result["failures"].append(f"{type(error).__name__}: {error}")
        result["traceback"] = traceback.format_exc()
    driver.finish_worker(out, result)


def timing_analysis(root, expected, *, final=False):
    plan = driver.checked(root, expected)
    rows, failures = driver.load_producers(root, plan, "timing.json")
    passed = (
        not failures
        and len(rows) == 2
        and all(r["own_speedup_gate"] and r["strongest_prefix_similar_gate"] for r in rows)
    )
    common.previous.closeout(
        root,
        not failures,
        failures,
        timings=rows,
        all_input_latency_gate=passed,
        no_reader_promoted=True,
        all_baselines_no_slower_gate=passed
        and all(r["all_baselines_no_slower_gate"] for r in rows),
        packet_storage_controls_required=True,
        no_independent_quality_inference=True,
    )
    if passed and not final:
        stage(common.previous.child(root, "all-baselines"), "all_baselines", "all_analysis")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "prepare",
            "cpu",
            "study",
            "study_analysis",
            "timing",
            "timing_analysis",
            "all_baselines",
            "all_analysis",
        ),
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--source-sha")
    parser.add_argument("--plan-sha")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root, args.source_sha)
    elif args.action == "all_baselines":
        timing(args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]), all_methods=True)
    elif args.action == "all_analysis":
        timing_analysis(args.root, args.plan_sha, final=True)
    elif args.action in ("study", "timing"):
        globals()[args.action](args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    else:
        globals()[args.action](args.root, args.plan_sha)


if __name__ == "__main__":
    main()
