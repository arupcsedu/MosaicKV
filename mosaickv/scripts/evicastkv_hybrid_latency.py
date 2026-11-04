#!/usr/bin/env python3
"""Finite consumed-input latency DAG. No fresh identities or paper campaign.

CPU -> repaired real-model profiling -> operator/storage/fusion diagnostics ->
paired complete requests -> conditional all-baseline verification. Every edge
uses afterany, validates producers, and writes an immutable decision.
"""

import argparse
import gc
import json
import os
import random
import subprocess
import traceback
from pathlib import Path

import evicastkv_full_prompt_native as driver
import evicastkv_native_model_profile as profiler

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = driver.SOURCE
PARENT = Path("/scratch/djy8hg/runs/evicastkv/native-model-context-consumed-20261103-v1")
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_hybrid_latency.sbatch"
METHODS = ("mosaickv_exact", "prefixkv_reimpl")
ALL_METHODS = (*METHODS, "prompt_attention_topk", "vl_cache_reimpl", "lookm_reimpl")


def submit(root, action, *, gpu=False, dependency=None):
    (root / "slurm").mkdir(exist_ok=True)
    expected = sha(root / "plan.json")
    args = ["sbatch", "--parsable", "--account=bii_dsc_community", "--nodes=1", "--ntasks=1"]
    if gpu:
        driver.cpu_proof(root, driver.checked(root, expected))
        limit = {
            "profile": "00:15:00",
            "study": "00:45:00",
            "timing": "01:30:00",
            "all_baselines": "01:45:00",
        }[action]
        args += [
            "--partition=bii-gpu",
            "--reservation=bi_fox_dgx",
            "--gres=gpu:a100:1",
            "--cpus-per-task=12",
            "--mem=48G",
            f"--time={limit}",
            "--array=0-1%2",
        ]
    else:
        args += [
            "--partition=bii,parallel,standard",
            "--cpus-per-task=2",
            "--mem=16G",
            "--time=00:15:00",
        ]
    if dependency:
        args.append(f"--dependency=afterany:{dependency}")
    args += [
        f"--job-name=mk-hybrid-{action}",
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
            maximum_worker_minutes=120,
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


def submit_stage(root, action, analyzer):
    gpu = submit(root, action, gpu=True)
    cpu = submit(root, analyzer, dependency=gpu)
    write_once(root / "submission.json", dict(gpu_array=gpu, analysis=cpu))


def prepare(root, revision):
    if root.exists() or clean_source(SOURCE) != revision:
        raise ValueError("new immutable root and clean source required")
    previous = driver.read(PARENT / "plan.json")
    bound = dict(previous["bound_files"])
    for p in (
        PARENT / "plan.json",
        PARENT / "analysis/decision.json",
        SOURCE / "mosaickv/docs/hybrid_latency_20261103.md",
    ):
        bound[str(p)] = sha(p)
    write_once(
        root / "plan.json",
        dict(
            previous,
            source_sha=revision,
            bound_files=bound,
            chain_root=str(root),
            stage="repaired_profile_then_lossless_hybrid_latency",
            maximum_simultaneous_workers=2,
            maximum_worker_minutes=120,
            hybrid_maximum_added_bytes=[0, 6 * 2**20],
            hybrid_decode_steps=127,
            hybrid_warmups=5,
            hybrid_repetitions=20,
            automatic_bounded_followup=True,
            automatic_quality_submission=False,
            automatic_full_request_submission=True,
            fresh_materialization_authorized=False,
            paper_campaign_authorized=False,
            independent_quality_N=0,
            numerical_gate="all_128_parent_logits_bitwise; no threshold relaxation",
            fusion_model_integration_requires_bitwise_operator_pass=True,
            no_byte_matched_algorithm_claim=True,
        ),
    )
    write_once(root / "preparation_submission.json", dict(cpu=submit(root, "cpu")))


def child(root, name):
    target = root / name
    old = driver.read(root / "plan.json")
    bound = dict(old["bound_files"])
    for p in (root / "plan.json", root / "analysis/decision.json"):
        bound[str(p)] = sha(p)
    for index in (0, 1):
        folder = root / f"workers/{index}"
        record = folder / "result.json"
        if not record.exists():
            raise ValueError("both immutable producer records required before child creation")
        producer = driver.read(record)
        if producer["status"] != "completed" or producer["failures"]:
            raise ValueError("failed producer cannot authorize follow-up")
        bound[str(record)] = sha(record)
        for name, expected in producer["outputs"].items():
            artifact = folder / name
            if sha(artifact) != expected:
                raise ValueError("parent producer artifact changed")
            bound[str(artifact)] = expected
    write_once(target / "plan.json", dict(old, bound_files=bound, stage=name))
    driver.copy_cpu(root, target)
    return target


def closeout(root, complete, failures, **extra):
    write_once(
        root / "analysis/decision.json",
        dict(
            complete=complete,
            failures=failures,
            independent_quality_N=0,
            paper_superiority_established=False,
            whole_request_GPU_peak_certified=False,
            no_fresh_or_large_campaign_submitted=True,
            **extra,
        ),
    )


def cpu(root, expected):
    driver.pair.common.previous.cpu(root, expected)
    driver.cpu_proof(root, driver.checked(root, expected))
    submit_stage(root, "profile", "profile_analysis")


def profile_analysis(root, expected):
    profiler.analyze(root, expected)
    report = driver.read(root / "analysis/decision.json")
    if report["complete"]:
        target = child(root, "hybrid-study")
        submit_stage(target, "study", "study_analysis")


def execute(
    adapter,
    refs,
    messages,
    profile,
    plan,
    method,
    *,
    policy=None,
    reuse=False,
    audit=False,
    collector=None,
    prefetch=False,
):
    import torch

    from mosaickv.evaluation.common_streaming_timing import request

    def validate(packets):
        value = driver.pair.common.control.digest_tensors(
            tuple(t for p in packets for t in p.tensors())
        )
        if value != refs[method]["payload"]["payload_sha256"]:
            raise RuntimeError("original packet bits changed")
        if collector is not None:
            collector(packets)

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
        input_packet_validator=validate if collector is not None else None,
        construction_geometry=plan.get("construction_geometry"),
        reconstruction_readers="carry_literals",
        audit_decode_storage=audit,
        hybrid_reader_policy=policy,
        reuse_kernel_dispatch=reuse,
        reuse_code_descriptors=reuse,
        source_hotpath="prepared_restore" if reuse else "original",
        prompt_prefetch=prefetch,
    )
    if not torch.equal(values, refs[method]["logits"]):
        difference = float((values - refs[method]["logits"]).abs().max())
        raise RuntimeError(f"hybrid request changed parent logits: max_abs={difference}")
    if (
        measured["original_precompression_packet_bytes"]
        != refs[method]["payload"]["physical_bytes"]
    ):
        raise RuntimeError("original construction storage changed")
    measured.update(
        logits_bitwise=True,
        no_FullKV_fallback=True,
        torch_peak_allocated=torch.cuda.max_memory_allocated(),
        torch_peak_reserved=torch.cuda.max_memory_reserved(),
    )
    if prefetch and (
        measured["prefetch_misses"] != 1
        or measured["prefetch_hits"] != 127 * len(profile) - 1
        or measured["prefetch_launches"] != measured["prefetch_hits"]
        or not 0 < measured["prefetch_buffer_bytes"] <= 6 * 2**20
    ):
        raise RuntimeError("prefetch did not engage at every registered decode layer")
    del values, packets
    return measured


def fusion_diagnostic(packet, attention, seed):
    import torch

    from mosaickv.evaluation.exact_storage import encode
    from mosaickv.evaluation.fused_lossless_attention import reference, run
    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
    from mosaickv.evaluation.sequential_kv_lease import fill_component

    generator = torch.Generator(device=packet.private()[0].device).manual_seed(seed)
    query = torch.randn(
        (1, packet.shape[1], 1, 128),
        device=packet.private()[0].device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    pair = []
    for c in (0, 1):
        original = torch.empty(packet.shape, device=query.device, dtype=torch.bfloat16)
        fill_component(packet, attention, c, original)
        pair.append(encode(original))
        del original
    expected, expected_weights = reference(query, *pair, scale=128**-0.5)
    actual, weights = run(query, *pair, scale=128**-0.5)
    bitwise = torch.equal(expected.view(torch.int16), actual.view(torch.int16))
    weights_bitwise = torch.equal(expected_weights.view(torch.int16), weights.view(torch.int16))
    return dict(
        attention_output_bitwise=bitwise,
        weights_bitwise=weights_bitwise,
        maximum_attention_error=float((actual.float() - expected.float()).abs().max()),
        maximum_weight_error=float((weights.float() - expected_weights.float()).abs().max()),
        operator_gate=bitwise and weights_bitwise,
        packed_storage_bytes=backing_bytes(tuple(t for b in pair for t in b.buffers)),
        score_weight_output_buffers_bytes=backing_bytes((actual, weights)),
        opaque_library_scratch_not_certified=True,
        no_full_KV_materialization_in_candidate_kernel=True,
        query_is_seeded_operator_probe_not_model_future_evidence=True,
        model_integration_authorized=False,
        independent_quality_N=0,
    )


def validated_arms(records):
    return {name: row for name, row in records.items() if row.get("status") == "passed"}


def study(root, expected, index):
    from mosaickv.evaluation.hybrid_reader_study import plans
    from mosaickv.evaluation.hybrid_reader_study import profile as cost_profile

    out = root / f"workers/{index}"
    out.mkdir(parents=True, exist_ok=False)
    result = dict(
        status="failed", failures=[], source_sha=clean_source(SOURCE), plan_sha256=expected
    )
    try:
        plan, adapter, refs, messages, layer_profile, _ = driver.setup(
            root,
            expected,
            index,
            manifest_overrides=dict(
                variant="consumed_lossless_hybrid_operator_and_storage_study",
                methods=list(METHODS),
                actual_timing_protocol=dict(
                    operator_warmups=5,
                    operator_repetitions=20,
                    output_predictions=128,
                    full_request_times_are_audited_controls_not_latency_evidence=True,
                ),
            ),
        )
        records, fusion = {}, None
        attentions = tuple(layer.self_attn for layer in adapter._language_layers())
        for method in METHODS:
            collected = {}

            def collect(packets, method=method, collected=collected):
                nonlocal fusion
                collected.update(cost_profile(packets, attentions, seed=plan["seed"]))
                if method == "mosaickv_exact":
                    try:
                        fusion = fusion_diagnostic(packets[0], attentions[0], plan["seed"])
                    except Exception as error:
                        fusion = dict(
                            operator_gate=False,
                            error=f"{type(error).__name__}: {error}",
                            traceback=traceback.format_exc(),
                            model_integration_authorized=False,
                        )

            base = execute(
                adapter, refs, messages, layer_profile, plan, method, audit=True, collector=collect
            )
            write_once(out / f"cost-{method}.json", collected)
            policies = plans(collected)
            arms = dict(original=dict(status="passed", policy=None, reuse=False, audited=base))
            proposals = [("reuse", None, True), ("prefetch", None, True)] + [
                (f"hybrid-{p['policy']['maximum_added_bytes']}", p["policy"], True)
                for p in policies
                if p["predicted_improvement_ns"] > 0
                and any(m != "source" for row in p["policy"]["modes"] for m in row)
            ]
            for name, policy, reuse in proposals:
                try:
                    measured = execute(
                        adapter,
                        refs,
                        messages,
                        layer_profile,
                        plan,
                        method,
                        policy=policy,
                        reuse=reuse,
                        audit=True,
                        prefetch=name == "prefetch",
                    )
                    allowance = (
                        6 * 2**20
                        if name == "prefetch"
                        else 0
                        if policy is None
                        else policy["maximum_added_bytes"]
                    )
                    if any(measured[k] > base[k] + allowance for k in driver.STORAGE):
                        raise RuntimeError(
                            "constructed initial/final/decode-peak exceeds allowance"
                        )
                    if not measured["peak_measurement_admissible"]:
                        raise RuntimeError("independent audited storage companion required")
                    arms[name] = dict(status="passed", policy=policy, reuse=reuse, audited=measured)
                except Exception as error:
                    arms[name] = dict(
                        status="rejected",
                        policy=policy,
                        reuse=reuse,
                        error=f"{type(error).__name__}: {error}",
                        traceback=traceback.format_exc(),
                    )
            records[method] = dict(plans=policies, arms=arms)
            write_once(out / f"controls-{method}.json", records[method])
        write_once(
            out / "study.json",
            dict(records=records, fusion=fusion, no_new_quality_or_speedup_claim=True),
        )
        result["status"] = "completed"
    except Exception as error:
        result["failures"].append(f"{type(error).__name__}: {error}")
        result["traceback"] = traceback.format_exc()
    driver.finish_worker(out, result)


def study_analysis(root, expected):
    plan = driver.checked(root, expected)
    rows, failures = driver.load_producers(root, plan, "study.json")
    if not failures:
        for index, row in enumerate(rows):
            for method in METHODS:
                if "reuse" not in validated_arms(row["records"][method]["arms"]):
                    failures.append(
                        dict(input_index=index, method=method, error="reuse controls failed")
                    )
    closeout(
        root,
        not failures,
        failures,
        studies=rows,
        fusion_is_separate_and_not_promoted=True,
        new_quality_evidence=False,
    )
    if not failures:
        target = child(root, "paired-timing")
        submit_stage(target, "timing", "timing_analysis")


def contrasts(rows, *, seed):
    import numpy as np

    trial_ids = np.random.default_rng(seed).integers(0, 20, (10000, 20))
    arms = sorted({(r["method"], r["arm"]) for r in rows})
    expected = {(m, a, t) for m, a in arms for t in range(-5, 20)}
    keys = [(r["method"], r["arm"], r["trial"]) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("complete unique five-warmup/twenty-trial paired arms required")
    arrays, summaries = {}, []
    for method, arm in arms:
        group = sorted(
            (r for r in rows if r["method"] == method and r["arm"] == arm and r["trial"] >= 0),
            key=lambda r: r["trial"],
        )
        values = np.asarray([r["measurement"]["total_wall_seconds"] for r in group])
        if not np.isfinite(values).all() or not (values > 0).all():
            raise ValueError("finite positive synchronized request timings required")
        arrays[method, arm] = values
        summaries.append(
            dict(
                method=method,
                arm=arm,
                mean=float(values.mean()),
                p50=float(np.median(values)),
                p95=float(np.quantile(values, 0.95)),
            )
        )
    # Fixed maximum 16 engineering contrasts across both consumed inputs.
    tail = 0.05 / (2 * 16)
    comparisons = []
    for (method, arm), values in arrays.items():
        if arm == "original":
            continue
        anchor = arrays[method, "original"]
        reference = arrays["prefixkv_reimpl", "reuse"]

        def ratio_bounds(other, values=values):
            return np.quantile(
                values[trial_ids].mean(1) / other[trial_ids].mean(1), [tail, 1 - tail]
            ).tolist()

        own, prefix = ratio_bounds(anchor), ratio_bounds(reference)
        comparisons.append(
            dict(
                method=method,
                arm=arm,
                own_ratio=float(values.mean() / anchor.mean()),
                own_adjusted_ratio_CI=own,
                prefix_adjusted_ratio_CI=prefix,
                own_speedup_gate=own[1] <= 0.99,
                prefix_similar_gate=prefix[1] <= 1.05,
                prefix_no_slower_gate=prefix[1] <= 1.0,
            )
        )
    return dict(
        summaries=summaries,
        comparisons=comparisons,
        intervals_over_engineering_repetitions_not_independent_examples=True,
    )


def timing(root, expected, index):
    out = root / f"workers/{index}"
    out.mkdir(parents=True, exist_ok=False)
    result = dict(
        status="failed", failures=[], source_sha=clean_source(SOURCE), plan_sha256=expected
    )
    try:
        plan, adapter, refs, messages, layer_profile, _ = driver.setup(
            root,
            expected,
            index,
            manifest_overrides=dict(
                variant="consumed_lossless_hybrid_paired_request_timing",
                methods=list(METHODS),
                actual_timing_protocol=dict(
                    warmups=5,
                    repetitions=20,
                    output_predictions=128,
                    conversion_included=True,
                    randomized_arm_order=True,
                ),
            ),
        )
        study_path = root.parent / f"workers/{index}/study.json"
        study = driver.read(study_path)
        arms = [
            (m, name, record)
            for m in METHODS
            for name, record in validated_arms(study["records"][m]["arms"]).items()
        ]
        rng, raw = random.Random(plan["seed"]), []
        for trial in range(-5, 20):
            order = list(arms)
            rng.shuffle(order)
            for method, name, record in order:
                measured = execute(
                    adapter,
                    refs,
                    messages,
                    layer_profile,
                    plan,
                    method,
                    policy=record["policy"],
                    reuse=record["reuse"],
                    audit=False,
                    prefetch=name == "prefetch",
                )
                audited = record["audited"]
                if (
                    any(measured[k] != audited[k] for k in driver.STORAGE[:2])
                    or measured["decode_owned_peak_bytes"] is not None
                    or measured["peak_measurement_admissible"]
                ):
                    raise RuntimeError("timed storage changed or counters were used as peak proof")
                row = dict(
                    method=method,
                    arm=name,
                    trial=trial,
                    warmup=trial < 0,
                    measurement=measured,
                    source_sha=plan["source_sha"],
                    plan_sha256=expected,
                )
                raw.append(row)
                with (out / "trials.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
        write_once(out / "timing.json", contrasts(raw, seed=plan["seed"]))
        result["status"] = "completed"
    except Exception as error:
        result["failures"].append(f"{type(error).__name__}: {error}")
        result["traceback"] = traceback.format_exc()
    driver.finish_worker(out, result)


def timing_analysis(root, expected):
    plan = driver.checked(root, expected)
    rows, failures = driver.load_producers(root, plan, "timing.json")
    eligible = []
    if not failures:
        for row in rows:
            eligible.append(
                [
                    r
                    for r in row["comparisons"]
                    if r["method"] == "mosaickv_exact"
                    and r["own_speedup_gate"]
                    and r["prefix_similar_gate"]
                ]
            )
    shared = (
        set.intersection(*(set(r["arm"] for r in group) for group in eligible))
        if eligible and all(eligible)
        else set()
    )
    selected = min(shared) if shared else None
    # No opportunistic new reader/kernel or fresh-cohort submission. A later
    # independent protocol is required even if an engineering contrast passes.
    closeout(
        root,
        not failures,
        failures,
        timings=rows,
        eligible_engineering_variants=eligible,
        all_input_latency_gate=bool(shared),
        selected_engineering_arm=selected,
        no_reader_promoted=True,
        no_paper_quality_inference=True,
        next_action="retain controls; freeze a passing variant before independent quality",
    )
    if selected is not None:
        target = child(root, "all-baselines")
        submit_stage(target, "all_baselines", "all_analysis")


def all_baselines(root, expected, index):
    from mosaickv.evaluation.common_streaming_timing import order, summarize
    from mosaickv.evaluation.hybrid_reader_study import plans
    from mosaickv.evaluation.hybrid_reader_study import profile as cost_profile

    out = root / f"workers/{index}"
    out.mkdir(parents=True, exist_ok=False)
    result = dict(
        status="failed", failures=[], source_sha=clean_source(SOURCE), plan_sha256=expected
    )
    try:
        plan, adapter, refs, messages, layer_profile, _ = driver.setup(
            root,
            expected,
            index,
            manifest_overrides=dict(
                variant="consumed_lossless_hybrid_all_baseline_timing",
                methods=list(ALL_METHODS),
                actual_timing_protocol=dict(
                    warmups=5,
                    repetitions=20,
                    output_predictions=128,
                    conversion_included=True,
                    randomized_method_order=True,
                ),
            ),
        )
        selected = driver.read(root.parent / "analysis/decision.json")["selected_engineering_arm"]
        allowance = (
            int(selected.removeprefix("hybrid-")) if selected.startswith("hybrid-") else None
        )
        attentions = tuple(layer.self_attn for layer in adapter._language_layers())
        policies, controls = {}, {}
        for method in ALL_METHODS:
            if allowance is not None:
                collected = {}

                def collect(packets, collected=collected):
                    collected.update(cost_profile(packets, attentions, seed=plan["seed"]))

                execute(adapter, refs, messages, layer_profile, plan, method, collector=collect)
                write_once(out / f"cost-{method}.json", collected)
                policies[method] = plans(collected, allowances=(allowance,))[0]["policy"]
            else:
                policies[method] = None
            controls[method] = execute(
                adapter,
                refs,
                messages,
                layer_profile,
                plan,
                method,
                policy=policies[method],
                reuse=True,
                audit=True,
                prefetch=selected == "prefetch",
            )
        write_once(
            out / "controls.json", dict(selected_arm=selected, policies=policies, audited=controls)
        )
        raw = []
        for placement in order(plan["seed"]):
            method = placement["method"]
            measured = execute(
                adapter,
                refs,
                messages,
                layer_profile,
                plan,
                method,
                policy=policies[method],
                reuse=True,
                audit=False,
                prefetch=selected == "prefetch",
            )
            if any(measured[k] != controls[method][k] for k in driver.STORAGE[:2]):
                raise RuntimeError("all-method storage differs from audited companion")
            row = dict(placement, **measured, source_sha=plan["source_sha"], plan_sha256=expected)
            raw.append(row)
            with (out / "trials.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        write_once(out / "all.json", summarize(raw, plan["seed"]))
        result["status"] = "completed"
    except Exception as error:
        result["failures"].append(f"{type(error).__name__}: {error}")
        result["traceback"] = traceback.format_exc()
    driver.finish_worker(out, result)


def all_analysis(root, expected):
    plan = driver.checked(root, expected)
    rows, failures = driver.load_producers(root, plan, "all.json")
    closeout(
        root,
        not failures,
        failures,
        all_baseline_engineering_summaries=rows,
        no_reader_promoted=True,
        no_independent_quality_claim=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "prepare",
            "cpu",
            "profile",
            "profile_analysis",
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
    elif args.action == "profile":
        profiler.worker(args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    elif args.action in ("study", "timing", "all_baselines"):
        globals()[args.action](args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    else:
        globals()[args.action](args.root, args.plan_sha)


if __name__ == "__main__":
    main()
