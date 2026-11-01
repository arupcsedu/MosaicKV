#!/usr/bin/env python3
"""Same-consumed-input GPU construction/model gate for full-prompt source precision."""

import argparse
import gc
import hashlib
import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import evicastkv_funded_delimiter_diagnostic as parent
import evicastkv_request_validation as previous
from evicastkv_campaign import environment

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
OLD = Path("/scratch/djy8hg/runs/evicastkv/funded-delimiter-consumed-20261031-v2")
STUDY = Path("/scratch/djy8hg/runs/evicastkv/protected-source-cpu-20261101-v2")
ARMS = ("shadow", "source3_full_prompt", "source3_exclude_global_exact")


def precision_header_label(arm):
    if arm.startswith("minimum__") or arm.endswith(("exact_exceptions", "coherent_columns")):
        return "lossless_bf16_minimum_fp32_scale"
    return "fp16" if "fp16" in arm else "fp32"


def history_method_jobs(jobs, variants, *, seed=0):
    """One method/input per bounded worker; immutable parent input remains explicit."""
    if len(jobs) != 2 or len(set(variants)) != len(variants) or not variants:
        raise ValueError("two consumed inputs and unique registered variants required")
    import random

    result = []
    for i, job in enumerate(jobs):
        order = list(variants)
        random.Random(seed + i).shuffle(order)
        start = len(result)
        result.extend(
            dict(job, index=start + j, input_index=i, evaluation_variant=variant)
            for j, variant in enumerate(order)
        )
    return result


def expected_row_count(plan):
    return sum(
        1 if "evaluation_variant" in job else len(plan.get("evaluation_variants", plan["variants"]))
        for job in plan["jobs"]
    )


def prepare(
    root,
    *,
    sequential_shadow=False,
    corrected_source=False,
    optimized_source=False,
    fp32_tiling=False,
    teacher_metrics=False,
    minimum_headers=False,
    query_affine=False,
    hybrid_forecast=False,
    sparse_exceptions=False,
    coherent_channels=False,
    sequential_history=False,
    common_history=False,
):
    if root.exists():
        raise ValueError("new immutable source-runtime root required")
    old = previous.read(OLD / "plan.json")
    study = previous.read(STUDY / "decision.json")
    if not study["complete"] or study["failures"] or [3, 2] not in study["passing_both_inputs"]:
        raise ValueError("frozen protected three-bit per-channel CPU feasibility required")
    bound = dict(old["bound_files"])
    for name in ("decision.json", "plan.json", "cpu.json", "cpu.xml"):
        bound[str(STUDY / name)] = sha(STUDY / name)
    bound.update(previous.read(STUDY / "plan.json")["bound_files"])
    for i in range(2):
        folder = OLD / f"workers/{i}"
        record = previous.read(folder / "result.json")
        if record["status"] != "completed" or record["failures"]:
            raise ValueError("complete consumed source-shadow controls required")
        bound[str(folder / "result.json")] = sha(folder / "result.json")
        for name, digest in record["outputs"].items():
            if sha(folder / name) != digest:
                raise ValueError("source shadow artifact changed")
            bound[str(folder / name)] = digest
    protocol = SOURCE / "mosaickv/docs/shared_source_runtime_20261101.md"
    bound[str(protocol)] = sha(protocol)
    if sequential_shadow:
        protocol = SOURCE / "mosaickv/docs/sequential_kv_lease_20261101.md"
        bound[str(protocol)] = sha(protocol)
    if corrected_source:
        transport = Path(
            "/scratch/djy8hg/runs/evicastkv/sequential-lease-consumed-20261101-v1/analysis/decision.json"
        )
        proof = previous.read(transport)
        if (
            not proof["complete"]
            or proof["failures"]
            or not proof["common_sequential_transport_gate"]
        ):
            raise ValueError("positive common one-buffer transport gate required")
        operator = Path(
            "/scratch/djy8hg/runs/evicastkv/shared-source-operator-consumed-20261101-v2/analysis/decision.json"
        )
        diagnostic = previous.read(operator)
        if (
            not diagnostic["complete"]
            or diagnostic["failures"]
            or diagnostic["mismatched_element_totals"]["bf16_default"] != 2
        ):
            raise ValueError("complete operator diagnosis required before correcting construction")
        for path in (
            transport,
            operator,
            SOURCE / "mosaickv/docs/source_transport_corrected_20261101.md",
        ):
            bound[str(path)] = sha(path)
    if optimized_source:
        micro = Path(
            "/scratch/djy8hg/runs/evicastkv/source-operator-micro-consumed-20261101-v1/analysis/decision.json"
        )
        data = previous.read(micro)
        if (
            not data["complete"]
            or data["failures"]
            or data["operator_selection"]["chosen_grouped_operator"] != "fused_h2"
        ):
            raise ValueError("complete bitwise two-head operator gate required")
        for path in (micro, SOURCE / "mosaickv/docs/source_input_fusion_20261101.md"):
            bound[str(path)] = sha(path)
    selected_head_group = None
    if fp32_tiling:
        micro = Path(
            "/scratch/djy8hg/runs/evicastkv/fp32-source-operator-consumed-20261101-v1/analysis/decision.json"
        )
        data = previous.read(micro)
        selected = data["operator_selection"]["chosen_grouped_operator"]
        if not data["complete"] or data["failures"] or selected not in ("fused_h2", "fused_h4"):
            raise ValueError("positive bitwise FP32 grouped operator gate required")
        selected_head_group = int(selected.removeprefix("fused_h"))
        for path in (micro, SOURCE / "mosaickv/docs/fp32_source_tiled_model_20261101.md"):
            bound[str(path)] = sha(path)
    teacher_ledgers = {}
    minimum_controls = {}
    if teacher_metrics:
        from mosaickv.evaluation.request_precision import METHODS

        validation = Path(
            "/scratch/djy8hg/runs/evicastkv/fp32-tiled-model-consumed-20261101-v1/analysis/decision.json"
        )
        checked = previous.read(validation)
        gates = [g for g in checked["gates"] if g["arm"] == "source3_fp32_tiled_text"]
        if (
            not checked["complete"]
            or checked["failures"]
            or len(gates) != 1
            or not gates[0]["model_gate"]
        ):
            raise ValueError("positive FP32 tiled model diagnostic required")
        bound[str(validation)] = sha(validation)
        for task in ("llava_description", "mmvet_prefixkv"):
            path = (
                Path(
                    "/scratch/djy8hg/runs/mosaickv/paper-native-full-v1-61e5a68-setup/ledgers/prefixkv"
                )
                / task
                / "full-ledger.json"
            )
            teacher_ledgers[task] = str(path)
            bound[str(path)] = sha(path)
        path = SOURCE / "mosaickv/docs/source_native_metrics_20261101.md"
        bound[str(path)] = sha(path)
    if minimum_headers:
        control_root = Path(
            "/scratch/djy8hg/runs/evicastkv/source-native-metrics-consumed-20261101-v1"
        )
        control = previous.read(control_root / "analysis/decision.json")
        if (
            not control["complete"]
            or control["failures"]
            or len(control["teacher_metric_summary"]) != 2
        ):
            raise ValueError(
                "complete full-target controls required before lossless header experiment"
            )
        for i in range(2):
            folder = control_root / f"workers/{i}"
            record = previous.read(folder / "result.json")
            for name, digest in record["outputs"].items():
                if sha(folder / name) != digest:
                    raise ValueError("minimum control changed")
                bound[str(folder / name)] = digest
        bound[str(control_root / "analysis/decision.json")] = sha(
            control_root / "analysis/decision.json"
        )
        path = SOURCE / "mosaickv/docs/lossless_affine_minimum_20261101.md"
        bound[str(path)] = sha(path)
        minimum_controls = {
            "minimum__source3_fp32_tiled_text": "source3_fp32_tiled_text",
            **{
                f"minimum__baseline_{m}__fp32": f"baseline_{m}__fp32"
                for m in METHODS
                if m != "mosaickv_exact"
            },
        }
    if query_affine:
        numeric = Path(
            "/scratch/djy8hg/runs/evicastkv/query-weighted-affine-cpu-consumed-20261101-v1/decision.json"
        )
        status = previous.read(numeric)
        if not status["complete"] or status["failures"] or not status["numerical_feasibility_gate"]:
            raise ValueError("positive frozen query-weighted CPU feasibility required")
        parent_metrics = Path(
            "/scratch/djy8hg/runs/evicastkv/source-native-metrics-consumed-20261101-v1"
        )
        for path in (
            numeric,
            numeric.parent / "result.json",
            SOURCE / "mosaickv/docs/query_affine_gpu_20261101.md",
            parent_metrics / "analysis/decision.json",
        ):
            bound[str(path)] = sha(path)
        for i in range(2):
            folder = parent_metrics / f"workers/{i}"
            record = previous.read(folder / "result.json")
            for name, digest in record["outputs"].items():
                if sha(folder / name) != digest:
                    raise ValueError("previous metric controls changed")
                bound[str(folder / name)] = digest
    if hybrid_forecast:
        if not query_affine:
            raise ValueError("hybrid requires common query-aware fitting")
        negative_root = Path("/scratch/djy8hg/runs/evicastkv/query-affine-gpu-consumed-20261101-v1")
        negative = previous.read(negative_root / "analysis/decision.json")
        if not negative["complete"] or negative["failures"]:
            raise ValueError("complete previous prompt-only diagnostic required")
        for path in (
            negative_root / "analysis/decision.json",
            SOURCE / "mosaickv/docs/hybrid_source_forecast_20261101.md",
        ):
            bound[str(path)] = sha(path)
        for i in range(2):
            folder = negative_root / f"workers/{i}"
            record = previous.read(folder / "result.json")
            for name, digest in record["outputs"].items():
                if sha(folder / name) != digest:
                    raise ValueError("negative prompt-only evidence changed")
                bound[str(folder / name)] = digest
    if sparse_exceptions:
        if not hybrid_forecast:
            raise ValueError("common online evidence required for exact exceptions")
        path = Path(
            "/scratch/djy8hg/runs/evicastkv/hybrid-query-affine-gpu-consumed-20261101-v1/analysis/decision.json"
        )
        result = previous.read(path)
        if not result["complete"] or result["failures"]:
            raise ValueError(
                "complete hybrid diagnostic required before separately versioned exceptions"
            )
        for evidence in (path, SOURCE / "mosaickv/docs/sparse_original_exceptions_20261101.md"):
            bound[str(evidence)] = sha(evidence)
    if coherent_channels:
        if not hybrid_forecast or sparse_exceptions:
            raise ValueError(
                "coherent columns require common forecast and no output-scalar exceptions"
            )
        negative = Path(
            "/scratch/djy8hg/runs/evicastkv/sparse-original-exceptions-consumed-20261101-v2"
        )
        decision = previous.read(negative / "analysis/decision.json")
        if not decision["complete"] or decision["failures"]:
            raise ValueError("complete negative independent-output-correction diagnostic required")
        for path in (
            negative / "analysis/decision.json",
            SOURCE / "mosaickv/docs/coherent_source_columns_20261101.md",
        ):
            bound[str(path)] = sha(path)
        for i in range(2):
            folder = negative / f"workers/{i}"
            record = previous.read(folder / "result.json")
            for name, digest in record["outputs"].items():
                if sha(folder / name) != digest:
                    raise ValueError("negative correction evidence changed")
                bound[str(folder / name)] = digest
    if sequential_history:
        if coherent_channels or sparse_exceptions or hybrid_forecast:
            raise ValueError(
                "history engineering freezes the prompt codec and disables corrections/forecast"
            )
        proof_root = Path(
            "/scratch/djy8hg/runs/evicastkv/coherent-source-columns-consumed-20261101-v1"
        )
        check = previous.read(proof_root / "analysis/decision.json")
        direct = [g for g in check["gates"] if g["arm"] == "source3_direct_fp32"]
        if (
            not check["complete"]
            or check["failures"]
            or len(direct) != 1
            or not direct[0]["model_gate"]
        ):
            raise ValueError("positive direct-constructor identity required")
        for path in (
            proof_root / "analysis/decision.json",
            SOURCE / "mosaickv/docs/sequential_source_history_20261101.md",
        ):
            bound[str(path)] = sha(path)
        for i in range(2):
            folder = proof_root / f"workers/{i}"
            record = previous.read(folder / "result.json")
            for name, digest in record["outputs"].items():
                if sha(folder / name) != digest:
                    raise ValueError("frozen prompt controls changed")
                bound[str(folder / name)] = digest
    if common_history:
        if not sequential_history:
            raise ValueError("common history timing requires the validated exact history mode")
        canary = Path(
            "/scratch/djy8hg/runs/evicastkv/sequential-source-history-consumed-20261101-v1"
        )
        decision = previous.read(canary / "analysis/decision.json")
        if (
            not decision["complete"]
            or decision["failures"]
            or not decision["common_exact_history_transport_gate"]
            or not decision["audited_history_memory_gate"]
        ):
            raise ValueError("complete generation/teacher bit identity and memory gates required")
        for path in (canary / "plan.json", canary / "analysis/decision.json"):
            bound[str(path)] = sha(path)
        for i in range(2):
            folder = canary / f"workers/{i}"
            record = previous.read(folder / "result.json")
            if record["status"] != "completed" or record["failures"]:
                raise ValueError("positive history producer evidence required")
            bound[str(folder / "result.json")] = sha(folder / "result.json")
            for name, digest in record["outputs"].items():
                if sha(folder / name) != digest:
                    raise ValueError("history canary artifact changed")
                bound[str(folder / name)] = digest
        path = SOURCE / "mosaickv/docs/common_exact_history_timing_20261101.md"
        bound[str(path)] = sha(path)
    variants = ("shadow",) if sequential_shadow else ARMS
    if corrected_source:
        variants = ("shadow", "source3_exclude_global_exact", "source3_exact_text_source")
    if optimized_source:
        from mosaickv.evaluation.request_precision import METHODS

        variants = (
            *variants,
            "shadow_fp16_metadata",
            "source3_fp16_group2_text",
            "source3_fp16_native_text",
            "source3_fp16_native_fused_inputs",
            *(
                f"baseline_{method}__{header}"
                for method in METHODS
                if method != "mosaickv_exact"
                for header in ("fp32", "fp16")
            ),
        )
    if fp32_tiling:
        variants = (*variants, "source3_fp32_tiled_text")
    if teacher_metrics:
        variants = (
            *variants,
            *(
                f"baseline_{method}__{header}"
                for method in METHODS
                if method != "mosaickv_exact"
                for header in ("fp32", "fp16")
            ),
        )
    if minimum_headers:
        variants = (*variants, *minimum_controls)
    if query_affine:
        affine_suffix = "query_affine_hybrid" if hybrid_forecast else "query_affine"
        if sparse_exceptions:
            affine_suffix = "exact_exceptions"
        variants = (
            *variants,
            f"source3_{affine_suffix}",
            *(
                f"baseline_{method}__{affine_suffix}"
                for method in METHODS
                if method != "mosaickv_exact"
            ),
        )
    evaluation_variants = tuple(minimum_controls) if minimum_headers else variants
    if query_affine:
        evaluation_variants = (
            "source3_fp32_tiled_text",
            f"source3_{affine_suffix}",
            *(
                f"baseline_{method}__{kind}"
                for method in METHODS
                if method != "mosaickv_exact"
                for kind in ("fp32", affine_suffix)
            ),
        )
    if coherent_channels or sequential_history:
        variants = tuple(a for a in variants if "query_affine" not in a)
        variants = (
            *variants,
            "source3_direct_fp32",
            *(("source3_coherent_columns",) if coherent_channels else ()),
        )
        evaluation_variants = (
            "source3_fp32_tiled_text",
            "source3_direct_fp32",
            *(("source3_coherent_columns",) if coherent_channels else ()),
            *(f"baseline_{method}__fp32" for method in METHODS if method != "mosaickv_exact"),
        )
    if sequential_history:
        evaluation_variants = (
            "source3_direct_fp32",
            *(f"baseline_{method}__fp32" for method in METHODS if method != "mosaickv_exact"),
        ) if common_history else ("source3_direct_fp32", "baseline_prompt_attention_topk__fp32")
    write_once(
        root / "plan.json",
        dict(
            old,
            jobs=history_method_jobs(old["jobs"], evaluation_variants, seed=old["seed"])
            if common_history else old["jobs"],
            source_sha=clean_source(SOURCE),
            bound_files=bound,
            stage="common_sequential_lease_consumed_control"
            if sequential_shadow
            else "consumed_full_prompt_shared_projection_source_precision",
            variants=variants,
            sequential_shadow=sequential_shadow,
            corrected_source=corrected_source,
            optimized_source=optimized_source,
            fp32_tiling=fp32_tiling,
            selected_head_group=selected_head_group,
            teacher_metrics=teacher_metrics,
            teacher_ledgers=teacher_ledgers,
            teacher_maximum_tokens=1024 if teacher_metrics else None,
            minimum_headers=minimum_headers,
            minimum_controls=minimum_controls,
            minimum_control_root=str(control_root) if minimum_headers else None,
            evaluation_variants=evaluation_variants,
            query_affine=query_affine,
            hybrid_forecast=hybrid_forecast,
            sparse_exceptions=sparse_exceptions,
            coherent_channels=coherent_channels,
            sequential_source_history=sequential_history,
            history_audit=True if sequential_history else None,
            maximum_worker_minutes=90 if common_history else 60 if sequential_history else 30,
            maximum_simultaneous_workers=6 if common_history else 2,
            one_method_per_worker=common_history,
            common_history=common_history,
            paired_history_timing=common_history,
            history_timing_warmups=5 if common_history else 0,
            history_timing_repetitions=20 if common_history else 0,
            randomized_method_and_history_order=common_history,
            history_reference_root=str(proof_root) if sequential_history else None,
            no_prompt_codec_or_selection_change=sequential_history,
            same_history_optimization_for_every_baseline=sequential_history,
            coherent_column_allowance_bytes=2**20 if coherent_channels else 0,
            baseline_comparison_admissible=not (coherent_channels or sequential_history),
            common_storage_controls_not_byte_matched_algorithm_evidence=sequential_history,
            coherent_column_phase="source_transport_and_own_parent_diagnostic_not_comparative_claim"
            if coherent_channels
            else None,
            maximum_exact_exception_bytes=2**20 if sparse_exceptions else 0,
            draft_tokens=4 if hybrid_forecast else 0,
            forecast_family_weights=[0.5, 0.5] if hybrid_forecast else None,
            query_reference_root=str(parent_metrics) if query_affine else None,
            affine_steps=0
            if sparse_exceptions or coherent_channels or sequential_history
            else 2
            if query_affine
            else None,
            sequential_transport=sequential_shadow or corrected_source,
            original_caps_unchanged=True,
            one_prefill=True,
            retire_original_source_before_decode=True,
            original_retention_one_bypasses_all_lossy_formats=True,
            source_position_fraction=0.5 if sequential_shadow else 1.0,
            source_codec_bits=3,
            source_codec_axis=2,
            source_scalar_KV_patches_disabled=not sequential_shadow and not sparse_exceptions,
            independent_quality_N=0,
            no_fresh_or_large_campaign_authorized=True,
            paper_campaign_authorized=False,
        ),
    )
    print(sha(root / "plan.json"))


def digest_tensors(tensors):
    import torch

    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
        digest.update(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def build(
    adapter,
    prepared,
    prefill,
    capture,
    text_capture,
    source_capture,
    phase_capture,
    cfg,
    plan,
    full,
    out,
    execute=None,
):
    import torch

    from mosaickv.evaluation import storage_runs
    from mosaickv.evaluation.exact_storage import ExactFormat
    from mosaickv.evaluation.head_local_precision import bundle
    from mosaickv.evaluation.lookm_merge_ablation import comparison
    from mosaickv.evaluation.lossless_bf16 import PackedBF16
    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
    from mosaickv.evaluation.precision_tier import unpack_cpu
    from mosaickv.evaluation.shared_source_format import construct
    from mosaickv.evaluation.text_source_storage import optimize

    execute = storage_runs.run if execute is None else execute
    diagnostic_build_started = time.monotonic()
    forecast = None
    forecast_proof = None
    if plan.get("hybrid_forecast", False):
        from mosaickv.evaluation.source_draft_capture import collect

        forecast, forecast_proof, draft_logits = collect(adapter, prefill)
        draft_control = comparison(full[:5], draft_logits)
        if (
            draft_control["maximum_logit_difference"] > 0.001
            or draft_control["token_agreement"] != 1
        ):
            raise RuntimeError("private FullKV draft changed ordinary trajectory")
        forecast_proof["ordinary_fullkv_validation_only"] = draft_control
        write_once(out / "forecast.json", forecast_proof)
        torch.save(draft_logits, out / "forecast_logits.pt")
        del draft_logits

    source, _ = adapter._legacy_layers(prefill.state.past_key_values)
    before = parent.causal.tensor_hash(source)
    source_bytes = backing_bytes(tuple(t for pair in source for t in pair))
    raw = tuple(
        ExactFormat(
            "kv",
            tuple(k.shape),
            pair=(PackedBF16(tuple(k.shape), True, (k,)), PackedBF16(tuple(v.shape), True, (v,))),
        )
        for k, v in source
    )
    ordinary, ordinary_info = execute(adapter, prefill, raw, tokens=128, sealed=False, audit=False)
    control = comparison(full, ordinary)
    if control["maximum_logit_difference"] > 0.001 or control["token_agreement"] != 1:
        raise RuntimeError("retention-one raw FullKV 128-token parity failed")
    write_once(
        out / "fullcache_control.json",
        dict(
            control,
            storage=ordinary_info["storage"],
            lossy_codec_bypassed=True,
            lossless_history_identity_control_not_FullKV_systems_row=plan.get(
                "sequential_source_history", False
            ),
        ),
    )
    started = time.monotonic()
    packets, proof = bundle(
        adapter,
        prepared,
        prefill,
        capture,
        cfg,
        "mosaickv_exact",
        plan["profile_path"],
        0.5,
        policies=("head3_boundary8",),
        allocation_mode="balanced",
    )
    packets = packets["head3_boundary8"]
    cap = backing_bytes(tuple(t for f in packets for t in f.tensors()))
    shadow = tuple(
        optimize(
            f,
            text_capture.records[i],
            proof["source_memberships"][i],
            text_capture.positions,
            b.self_attn,
        )[0]
        for i, (f, b) in enumerate(zip(packets, adapter._language_layers(), strict=True))
    )
    torch.cuda.synchronize()
    times = {"shadow": time.monotonic() - started}
    h, n = source[0][0].shape[1:3]
    identity = (
        torch.arange(n, device=adapter.device, dtype=torch.int32),
        torch.arange(n, device=adapter.device, dtype=torch.int16)[None].expand(h, n),
    )
    formats = {"shadow": shadow}
    construction_checks = {}
    caps_by_arm = {"shadow": cap}
    setup_wall = time.monotonic()
    observed_pairs = {}
    observed_memberships = {}
    for arm in plan["variants"][1:]:
        torch.cuda.synchronize()
        begun = time.monotonic()
        if arm == "source3_direct_fp32":
            from mosaickv.evaluation.direct_source_constructor import (
                DirectSourcePolicy,
            )
            from mosaickv.evaluation.direct_source_constructor import (
                construct as direct_construct,
            )

            trial, checks = direct_construct(
                adapter,
                prepared,
                prefill,
                source_capture,
                text_capture,
                phase_capture,
                tuple(previous.read(Path(plan["profile_path"]))),
                cap,
                policy=DirectSourcePolicy(
                    head_group=plan["selected_head_group"], lossless_minimum=False
                ),
            )
            torch.cuda.synchronize()
            times[arm] = time.monotonic() - begun
            if checks["source_memberships"] != proof["source_memberships"]:
                raise RuntimeError("direct construction changed online memberships")
            if digest_tensors(tuple(t for f in trial for t in f.tensors())) != digest_tensors(
                tuple(t for f in formats["source3_fp32_tiled_text"] for t in f.tensors())
            ):
                raise RuntimeError("direct construction changed an encoded payload bit")
            formats[arm] = trial
            caps_by_arm[arm] = cap
            construction_checks[arm] = dict(
                status="constructed",
                direct=checks,
                encoded_payload_bitwise=True,
                includes_own_online_selection=True,
                no_discarded_shadow_codec_constructed=True,
            )
            continue
        if arm == "source3_coherent_columns":
            from mosaickv.evaluation.source_channel_exceptions import formats as channel_formats

            trial, checks = channel_formats(
                adapter,
                prefill,
                capture,
                source_capture,
                text_capture,
                phase_capture,
                formats["source3_exclude_global_exact"],
                forecast,
                head_group=plan["selected_head_group"],
                cap_bytes=cap,
            )
            torch.cuda.synchronize()
            times[arm] = (
                time.monotonic()
                - begun
                + times["source3_exclude_global_exact"]
                + forecast_proof["wall_seconds_excludes_integrity_hashes"]
            )
            formats[arm] = trial
            caps_by_arm[arm] = cap
            construction_checks[arm] = dict(
                status="constructed",
                fit=checks,
                original_quantizer_codes_unchanged=True,
                coupled_K_and_V_input=True,
                no_benchmark_label_or_reference_logit_access=True,
            )
            continue
        if arm.endswith("exact_exceptions"):
            from mosaickv.evaluation.sparse_original_exceptions import formats as add_exceptions

            if arm.startswith("source3_"):
                prior_arm = "source3_fp32_tiled_text"
                originals = source
                positions = [(tuple(range(n)),) * h] * len(source)
                own_cap = cap
            else:
                method = arm.removeprefix("baseline_").split("__")[0]
                prior_arm = f"baseline_{method}__fp32"
                originals = observed_pairs[method]
                positions = observed_memberships[method]
                own_cap = caps_by_arm[prior_arm]
            trial, checks = add_exceptions(
                adapter,
                prefill,
                capture,
                formats[prior_arm],
                originals,
                positions,
                forecast,
                cap_bytes=own_cap,
                maximum_extra_bytes=plan["maximum_exact_exception_bytes"],
            )
            torch.cuda.synchronize()
            times[arm] = (
                time.monotonic()
                - begun
                + times[prior_arm]
                + forecast_proof["wall_seconds_excludes_integrity_hashes"]
            )
            formats[arm] = trial
            caps_by_arm[arm] = own_cap
            construction_checks[arm] = dict(
                status="constructed",
                initial_bytes=backing_bytes(tuple(t for f in trial for t in f.tensors())),
                cap_bytes=own_cap,
                fit=checks,
                original_encoded_body_unchanged=True,
                actual_CUDA_verification=True,
                no_benchmark_label_or_reference_logit_access=True,
            )
            continue
        if "query_affine" in arm:
            from mosaickv.evaluation.query_affine_formats import baseline_formats, source_formats

            source_arm = arm.startswith("source3_")
            use_forecast = forecast if arm.endswith("_hybrid") else None
            if source_arm:
                trial, checks = source_formats(
                    adapter,
                    prefill,
                    capture,
                    source_capture,
                    text_capture,
                    phase_capture,
                    formats["source3_exclude_global_exact"],
                    head_group=plan["selected_head_group"],
                    forecast=use_forecast,
                )
                own_cap = cap
            else:
                method = arm.removeprefix("baseline_").split("__")[0]
                prior_arm = f"baseline_{method}__fp32"
                trial, checks = baseline_formats(
                    adapter,
                    prefill,
                    capture,
                    formats[prior_arm],
                    observed_pairs[method],
                    observed_memberships[method],
                    forecast=use_forecast,
                )
                own_cap = caps_by_arm[prior_arm]
            torch.cuda.synchronize()
            times[arm] = time.monotonic() - begun
            times[arm] += times["source3_exclude_global_exact"] if source_arm else times[prior_arm]
            if use_forecast is not None:
                times[arm] += forecast_proof["wall_seconds_excludes_integrity_hashes"]
            size = backing_bytes(tuple(t for f in trial for t in f.tensors()))
            caps_by_arm[arm] = own_cap
            construction_checks[arm] = dict(
                status="constructed" if size <= own_cap else "unsupported",
                initial_bytes=size,
                cap_bytes=own_cap,
                fit=checks,
                actual_CUDA_refinement=True,
                no_benchmark_label_or_reference_logit_access=True,
            )
            if size <= own_cap:
                formats[arm] = trial
            continue
        if arm.startswith("minimum__"):
            from mosaickv.evaluation.lossless_affine_minimum import encode_format

            parent_arm = plan["minimum_controls"][arm]
            trial = tuple(encode_format(f) for f in formats[parent_arm])
            formats[arm] = trial
            caps_by_arm[arm] = caps_by_arm.get(parent_arm, cap)
            torch.cuda.synchronize()
            times[arm] = times[parent_arm] + time.monotonic() - begun
            verification_started = time.monotonic()
            for parent_packet, packet, block in zip(
                formats[parent_arm], trial, adapter._language_layers(), strict=True
            ):
                original_values = parent_packet.materialize(block.self_attn)
                new_values = packet.materialize(block.self_attn)
                if not all(
                    torch.equal(x.view(torch.int16), y.view(torch.int16))
                    for x, y in zip(original_values, new_values, strict=True)
                ):
                    raise RuntimeError("lossless minimum changed constructed K/V bits")
            del original_values, new_values
            torch.cuda.synchronize()
            construction_checks[arm] = dict(
                status="constructed",
                source_dependency=parent_arm,
                lossless_minimum_encoding=True,
                original_FP32_scale_unchanged=True,
                all_layer_KV_bitwise=True,
                verification_seconds=time.monotonic() - verification_started,
                encoding_time_excludes_verification=True,
            )
            continue
        if arm == "shadow_fp16_metadata" or arm.startswith("baseline_"):
            from mosaickv.evaluation.compact_quantization_metadata import compact_format

            method = (
                "mosaickv_exact"
                if arm == "shadow_fp16_metadata"
                else arm.removeprefix("baseline_").split("__")[0]
            )
            if arm == "shadow_fp16_metadata":
                trial = tuple(compact_format(f) for f in shadow)
                own_cap = cap
            elif arm.endswith("__fp16"):
                prior_arm = arm.removesuffix("fp16") + "fp32"
                trial = tuple(compact_format(f) for f in formats[prior_arm])
                own_cap = caps_by_arm[prior_arm]
            else:
                baseline, baseline_proof = bundle(
                    adapter,
                    prepared,
                    prefill,
                    capture,
                    cfg,
                    method,
                    plan["profile_path"],
                    0.5,
                    policies=("head3_boundary8",),
                    allocation_mode="balanced" if method == "prompt_attention_topk" else "profile",
                    pair_observer=(
                        lambda layer, pair, method=method: observed_pairs.setdefault(
                            method, []
                        ).append(pair)
                    )
                    if plan.get("query_affine", False)
                    else None,
                )
                observed_memberships[method] = baseline_proof["source_memberships"]
                packets2 = baseline["head3_boundary8"]
                own_cap = backing_bytes(tuple(t for f in packets2 for t in f.tensors()))
                trial = tuple(
                    optimize(
                        f,
                        text_capture.records[i],
                        baseline_proof["source_memberships"][i],
                        text_capture.positions,
                        block.self_attn,
                    )[0]
                    for i, (f, block) in enumerate(
                        zip(packets2, adapter._language_layers(), strict=True)
                    )
                )
            torch.cuda.synchronize()
            times[arm] = time.monotonic() - begun
            formats[arm] = trial
            caps_by_arm[arm] = own_cap
            membership_hash = (
                construction_checks[prior_arm].get("source_memberships_sha256")
                if arm.endswith("__fp16")
                else hashlib.sha256(
                    json.dumps(
                        proof["source_memberships"]
                        if method == "mosaickv_exact"
                        else baseline_proof["source_memberships"],
                        sort_keys=True,
                    ).encode()
                ).hexdigest()
            )
            construction_checks[arm] = dict(
                status="constructed",
                method_override=method,
                source_memberships_sha256=membership_hash,
                header_dtype="fp16"
                if arm.endswith("fp16") or arm == "shadow_fp16_metadata"
                else "fp32",
                source_selection_ratio=0.5,
                cap_bytes=own_cap,
            )
            continue
        if arm in ("source3_fp16_group2_text", "source3_fp16_native_text"):
            from mosaickv.evaluation.compact_quantization_metadata import compact_format
            from mosaickv.evaluation.shared_source_format import fit_transport

            native = arm == "source3_fp16_native_text"
            results = []
            native_checks = []
            identity_positions = (tuple(range(n)),) * h
            for i, (f, block) in enumerate(
                zip(
                    formats["source3_exclude_global_exact"], adapter._language_layers(), strict=True
                )
            ):
                packet = replace(
                    compact_format(f),
                    projection_backend="native_bmm_256" if native else "fused",
                    source_head_group=1 if native else 2,
                )
                packet, numerics = fit_transport(packet, block.self_attn)
                packet, text_proof = optimize(
                    packet,
                    text_capture.records[i],
                    identity_positions,
                    text_capture.positions,
                    block.self_attn,
                )
                results.append(packet)
                native_checks.append(dict(numerical=numerics, text=text_proof))
            trial = tuple(results)
            torch.cuda.synchronize()
            times[arm] = times["source3_exclude_global_exact"] + time.monotonic() - begun
            own_cap = cap
            size = backing_bytes(tuple(t for f in trial for t in f.tensors()))
            caps_by_arm[arm] = own_cap
            construction_checks[arm] = dict(
                status="constructed" if size <= cap else "unsupported",
                initial_bytes=size,
                cap_bytes=cap,
                checks=native_checks,
            )
            if size <= cap:
                formats[arm] = trial
            continue
        if arm == "source3_fp32_tiled_text":
            from mosaickv.evaluation.source_retile import retile

            parent_arm = "source3_exact_text_source"
            trial = tuple(retile(f, plan["selected_head_group"]) for f in formats[parent_arm])
            formats[arm] = trial
            caps_by_arm[arm] = cap
            times[arm] = times[parent_arm] + time.monotonic() - begun
            construction_checks[arm] = dict(
                status="constructed",
                source_dependency=parent_arm,
                hardware_tiling_only=True,
                metadata_precision="fp32",
                head_group=plan["selected_head_group"],
            )
            continue
        if arm == "source3_fp16_native_fused_inputs":
            from mosaickv.evaluation.integrated_source_input import fuse

            parent_arm = "source3_fp16_native_text"
            if parent_arm not in formats:
                construction_checks[arm] = dict(
                    status="unsupported", reason="native source parent unsupported"
                )
                continue
            results = tuple(
                fuse(f, block.self_attn)
                for f, block in zip(formats[parent_arm], adapter._language_layers(), strict=True)
            )
            trial = tuple(r[0] for r in results)
            torch.cuda.synchronize()
            times[arm] = times[parent_arm] + time.monotonic() - begun
            formats[arm] = trial
            caps_by_arm[arm] = cap
            construction_checks[arm] = dict(
                status="constructed",
                fusion_checks=[r[1] for r in results],
                source_dependency=parent_arm,
            )
            continue
        if arm == "source3_exact_text_source":
            previous_arm = "source3_exclude_global_exact"
            if previous_arm not in formats:
                construction_checks[arm] = dict(
                    status="unsupported", reason="source parent unsupported"
                )
                continue
            h, n = source[0][0].shape[1:3]
            identity_positions = (tuple(range(n)),) * h
            results = tuple(
                optimize(
                    f,
                    text_capture.records[i],
                    identity_positions,
                    text_capture.positions,
                    block.self_attn,
                )
                for i, (f, block) in enumerate(
                    zip(formats[previous_arm], adapter._language_layers(), strict=True)
                )
            )
            trial = tuple(r[0] for r in results)
            torch.cuda.synchronize()
            times[arm] = times[previous_arm] + time.monotonic() - begun
            size = backing_bytes(tuple(t for f in trial for t in f.tensors()))
            if size > cap:
                raise RuntimeError("lossless source-text construction increased the hard cap")
            formats[arm] = trial
            construction_checks[arm] = dict(
                status="constructed",
                initial_bytes=size,
                cap_bytes=cap,
                text_storage_checks=[r[1] for r in results],
                source_dependency=previous_arm,
            )
            continue
        trial = tuple(
            construct(
                source_capture.records[i],
                pair,
                proof["source_memberships"][i],
                text_capture.positions,
                phase_capture.phases,
                identity,
                exclude_global_exact=arm.endswith("exclude_global_exact"),
            )
            for i, pair in enumerate(source)
        )
        numerical_checks = []
        if plan.get("corrected_source", False):
            from mosaickv.evaluation.shared_source_format import fit_transport

            results = tuple(
                fit_transport(f, block.self_attn)
                for f, block in zip(trial, adapter._language_layers(), strict=True)
            )
            trial = tuple(r[0] for r in results)
            numerical_checks = [r[1] for r in results]
        torch.cuda.synchronize()
        times[arm] = time.monotonic() - begun
        size = backing_bytes(tuple(t for f in trial for t in f.tensors()))
        if size > cap:
            construction_checks[arm] = dict(
                status="unsupported", attempted_initial_bytes=size, cap_bytes=cap
            )
            continue
        checks = []
        for i, (packet, block, original) in enumerate(
            zip(trial, adapter._language_layers(), source, strict=True)
        ):
            # Numerical verification only. Production reads codes in registers,
            # not this explicitly temporary CPU-decoded projection input.
            body = replace(
                packet.source,
                codes=packet.source.codes.cpu(),
                minimum=packet.source.minimum.cpu(),
                scale=packet.source.scale.cpu(),
            )
            x = unpack_cpu(body).squeeze(1).to(adapter.device)
            attn = block.self_attn
            k = attn.k_proj(x).reshape(1, n, h, 128).transpose(1, 2)
            v = attn.v_proj(x).reshape(1, n, h, 128).transpose(1, 2)
            k = (
                k * packet.cos[:, None]
                + torch.cat((-k[..., 64:], k[..., :64]), -1) * packet.sin[:, None]
            )
            reference = k.contiguous(), v.contiguous()
            ids = packet.row_positions.long()
            head, position = ids // n, ids % n
            actual = packet.materialize(attn)
            for expected, value, src in zip(reference, actual, original, strict=True):
                expected[0, head, position] = src[0, head, position]
                torch.testing.assert_close(value, expected, rtol=0.01, atol=2**-6)
                if not torch.equal(
                    value[0, head, position].view(torch.int16),
                    src[0, head, position].view(torch.int16),
                ):
                    raise RuntimeError("protected original GPU K/V changed bits")
            tail = tuple(
                torch.full((1, h, n + 5, 128), -71, device=adapter.device, dtype=torch.bfloat16)
                for _ in range(2)
            )
            packet.fill(attn, tail, direct=True)
            if not all(
                torch.equal(a, b[..., :n, :]) and bool((b[..., n:, :] == -71).all())
                for a, b in zip(actual, tail, strict=True)
            ):
                raise RuntimeError("source projection changed strided prompt or history sentinel")
            checks.append(
                dict(
                    layer=i,
                    protected_scalars_bitwise=True,
                    strided_prompt_and_history=True,
                    maximum_native_projection_difference=max(
                        float((a.float() - b.float()).abs().max())
                        for a, b in zip(actual, reference, strict=True)
                    ),
                )
            )
        formats[arm] = trial
        construction_checks[arm] = dict(
            status="constructed",
            initial_bytes=size,
            cap_bytes=cap,
            checks=checks,
            numerical_transport_checks=numerical_checks,
        )
    if before != parent.causal.tensor_hash(source):
        raise RuntimeError("source construction mutated original prefill K/V")
    owners = {(str(t.device), t.untyped_storage().data_ptr()) for pair in source for t in pair}
    if any(
        t.numel() and (str(t.device), t.untyped_storage().data_ptr()) in owners
        for packets in formats.values()
        for f in packets
        for t in f.tensors()
    ):
        raise RuntimeError("encoded format retains original FullKV storage")
    return formats, dict(
        original_source_bytes=source_bytes,
        cap_bytes=cap,
        caps_by_arm=caps_by_arm,
        additional_formats_setup_seconds=time.monotonic() - setup_wall,
        diagnostic_build_total_seconds=time.monotonic() - diagnostic_build_started,
        encoded_formats_disjoint_from_original_KV=True,
        shared_protection_setup_seconds=times["shadow"],
        construction_seconds=times,
        construction_checks=construction_checks,
        forecast=forecast_proof,
    )


def worker(root, expected, index):
    import torch

    from mosaickv.adapters.huggingface.registry import load_hf_adapter
    from mosaickv.adapters.huggingface.types import QueryVectors
    from mosaickv.config import RunConfig, config_sha256
    from mosaickv.evaluation import storage_runs
    from mosaickv.evaluation.lookm_merge_ablation import comparison
    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
    from mosaickv.evaluation.metrics.scorers import rouge_l_f1
    from mosaickv.evaluation.phase_capacity import PhaseCapture
    from mosaickv.evaluation.request_precision import PromptQueryCapture
    from mosaickv.evaluation.shared_source_format import SourceCapture
    from mosaickv.evaluation.text_source_storage import TextSourceCapture
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    plan = previous.common.inputs(root, expected)
    job = plan["jobs"][index]
    input_index = job.get("input_index", index)
    arms = (job["evaluation_variant"],) if "evaluation_variant" in job else tuple(
        plan.get("evaluation_variants", plan["variants"])
    )
    if plan.get("common_history", False):
        import random

        arms = list(arms)
        random.Random(plan["seed"] + input_index).shuffle(arms)
        arms = tuple(arms)
    execute = storage_runs.run
    if plan.get("sequential_transport", plan.get("sequential_shadow", False)):
        from mosaickv.evaluation.sequential_kv_lease import paired_run

        execute = paired_run
    if plan.get("sequential_source_history", False):
        from functools import partial

        from mosaickv.evaluation.sequential_source_history import paired_run

        execute = partial(paired_run, audit_history=plan["history_audit"])
    job = plan["jobs"][index]
    out = root / f"workers/{index}"
    out.mkdir(parents=True, exist_ok=False)
    result = dict(status="failed", failures=[], source_sha=plan["source_sha"], plan_sha256=expected)
    (out / "raw.jsonl").touch(exist_ok=False)
    arm = "setup"
    payload = None
    try:
        payload = previous.read(job["inputs"][0])
        if (
            torch.cuda.device_count() != 1
            or "A100" not in torch.cuda.get_device_name()
            or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
        ):
            raise ValueError("one A100 and deterministic CUDA workspace required")
        torch.set_num_threads(12)
        torch.manual_seed(plan["seed"])
        torch.use_deterministic_algorithms(True)
        if plan.get("query_affine", False):
            from mosaickv.evaluation.query_affine_formats import gpu_controls

            write_once(out / "affine_cuda_controls.json", gpu_controls())
            torch.manual_seed(plan["seed"])
        if plan.get("coherent_channels", False):
            from mosaickv.evaluation.source_channel_exceptions import gpu_controls

            write_once(out / "coherent_cuda_controls.json", gpu_controls())
            torch.manual_seed(plan["seed"])
        if plan.get("optimized_source", False):
            from mosaickv.evaluation.integrated_source_input_kernel import gpu_controls

            write_once(out / "input_scatter_controls.json", gpu_controls())
            torch.manual_seed(plan["seed"])
        cfg = RunConfig.from_mapping(previous.resolved_mapping(plan, job))
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
            raise ValueError("pinned full-MHA LLaVA BF16/eager only")
        adapter = load_hf_adapter(
            cfg.model.id,
            revision=cfg.model.revision,
            model_kwargs=_model_kwargs(cfg, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        folder = Path(plan["parent"]) / f"workers/{input_index}"
        full = torch.load(folder / "logits.pt", map_location="cpu", weights_only=True)[
            "0-full_kv-ordinary"
        ]
        prepared = adapter.prepare_inputs(previous.load_messages(payload))
        if not prepared.model_inputs["attention_mask"].bool().all():
            raise ValueError("one unpadded consumed image prompt required")
        with (
            PromptQueryCapture(adapter, summaries=False, normalized=True, prior=False) as capture,
            TextSourceCapture(adapter, prepared) as text_capture,
            SourceCapture(adapter) as source_capture,
            PhaseCapture(adapter) as phase_capture,
        ):
            prefill = adapter.prefill(prepared, capture_queries=False, capture_attentions=True)
        with torch.inference_mode():
            import weakref

            original_refs = tuple(
                weakref.ref(t)
                for pair in adapter._legacy_layers(prefill.state.past_key_values)[0]
                for t in pair
            )
            formats, proof = build(
                adapter,
                prepared,
                prefill,
                capture,
                text_capture,
                source_capture,
                phase_capture,
                cfg,
                plan,
                full,
                out,
                execute=execute,
            )
        write_once(out / "construction.json", proof)
        teacher_targets = None
        if plan.get("teacher_metrics", False):
            from mosaickv.evaluation.exact_storage import ExactFormat
            from mosaickv.evaluation.lossless_bf16 import PackedBF16
            from mosaickv.evaluation.source_teacher_protocol import load_targets
            from mosaickv.evaluation.storage_teacher_forcing import score_formats, score_ordinary

            teacher_targets, teacher_provenance = load_targets(
                plan, payload, adapter.processor.tokenizer
            )
            ordinary_teacher, expected_teacher = score_ordinary(
                adapter,
                prefill,
                teacher_targets,
                maximum=plan["teacher_maximum_tokens"],
                return_logits=True,
            )
            layers, _ = adapter._legacy_layers(prefill.state.past_key_values)
            teacher_raw = tuple(
                ExactFormat(
                    "kv",
                    tuple(k.shape),
                    pair=(
                        PackedBF16(tuple(k.shape), True, (k,)),
                        PackedBF16(tuple(v.shape), True, (v,)),
                    ),
                )
                for k, v in layers
            )
            packed_teacher, actual_teacher = score_formats(
                adapter,
                prefill,
                teacher_raw,
                teacher_targets,
                maximum=plan["teacher_maximum_tokens"],
                return_logits=True,
            )
            difference = float((expected_teacher - actual_teacher).abs().max())
            if (
                difference > 0.001
                or ordinary_teacher["conditional_greedy_tokens"]
                != packed_teacher["conditional_greedy_tokens"]
            ):
                raise RuntimeError("complete teacher-forcing FullKV reinjection parity failed")
            write_once(
                out / "teacher_control.json",
                dict(
                    ordinary=ordinary_teacher,
                    packed=packed_teacher,
                    maximum_logit_difference=difference,
                    provenance=teacher_provenance,
                    retention_one_raw_bypass=True,
                ),
            )
            torch.save(expected_teacher, out / "teacher_fullkv_logits.pt")
            del layers, teacher_raw, expected_teacher, actual_teacher
        runtime = replace(
            prefill,
            state=replace(
                prefill.state, past_key_values=None, model_state=dict(prefill.state.model_state)
            ),
            query_vectors=QueryVectors(()),
            attention_weights=(),
        )
        torch.cuda.synchronize()
        before_retire = torch.cuda.memory_allocated()
        del prefill
        capture.records.clear()
        text_capture.records.clear()
        source_capture.records.clear()
        gc.collect()
        torch.cuda.synchronize()
        after_retire = torch.cuda.memory_allocated()
        live_originals = sum(ref() is not None for ref in original_refs)
        if live_originals:
            raise RuntimeError(
                f"{live_originals} original KV tensors remain alive after retirement"
            )
        if before_retire - after_retire < proof["original_source_bytes"]:
            raise RuntimeError(
                "original prefill/source retirement not certified by allocation drop"
            )
        retirement = dict(
            original_KV_bytes=proof["original_source_bytes"],
            allocated_before=before_retire,
            allocated_after=after_retire,
            allocation_drop=before_retire - after_retire,
            original_cache_refs_removed=True,
            original_KV_tensor_objects_alive=live_originals,
            encoded_formats_disjoint_from_original_KV=proof[
                "encoded_formats_disjoint_from_original_KV"
            ],
            whole_request_peak_certified=False,
        )
        write_once(out / "retirement.json", retirement)
        manifest = previous.read(folder / "manifest.json")
        manifest.update(
            source_sha=plan["source_sha"],
            source_dirty=False,
            plan_sha256=expected,
            config_sha256=hashlib.sha256(
                json.dumps(dict(config=config_sha256(cfg), plan=expected), sort_keys=True).encode()
            ).hexdigest(),
            environment=environment(),
            gpu=asdict(capture_gpu_environment(torch)),
            cuda_runtime=torch.version.cuda,
            evaluator_commit=plan["source_sha"],
            baseline_commit=plan["source_sha"],
            measurement_type="ablation_measured",
            diagnostic_only=True,
            independent_quality_N=0,
            variant="common_sequential_lease_unchanged_source_shadow"
            if plan.get("sequential_shadow", False)
            else "full_prompt_shared_source_precision_not_mosaickv_exact",
            disabled_components=[
                "future_query_forecast",
                "evidence_graph",
                "value_aware_block_utility",
                "submodular_selector",
                "prototype",
                "residual_repair",
            ],
            old_scalar_KV_patch_policy_replaced_by_source_codec=not plan.get(
                "sequential_shadow", False
            ),
            retention_one_uses_raw_no_quantization=True,
            source_position_fraction=plan["source_position_fraction"],
            retention_ratio="not_applicable_for_lossy_byte_compressed_variant",
            cpu_residual_bytes=0,
            no_superiority_or_byte_matched_claim=True,
            whole_request_peak_certified=False,
            actual_timing_protocol=dict(
                warmups=5 if plan.get("paired_history_timing", False) else 0,
                preceding_fixed_prefix_control=True,
                repetitions=20 if plan.get("paired_history_timing", False) else 1,
                order="randomized_method_blocks_and_within_pair_history_order"
                if plan.get("paired_history_timing", False)
                else "fixed",
                not_paper_latency=True,
            ),
        )
        write_once(out / "manifest.json", manifest)
        forced = tuple(full.argmax(-1)[:33].tolist())
        original = torch.load(
            OLD / f"workers/{input_index}/mosaickv_exact--source-logits.pt",
            map_location="cpu",
            weights_only=True,
        )
        hashes = {
            name: digest_tensors(tuple(t for p in f for t in p.tensors()))
            for name, f in formats.items()
        }
        all_format_bytes = backing_bytes(
            tuple(t for f in formats.values() for p in f for t in p.tensors())
        )
        for arm in arms:
            common = dict(
                variant=arm,
                task=payload["task"],
                sample_id=payload["sample_id"],
                source_sha=plan["source_sha"],
                plan_sha256=expected,
                config_sha256=hashlib.sha256(
                    (manifest["config_sha256"] + arm).encode()
                ).hexdigest(),
                manifest_path=str(out / "manifest.json"),
                measurement_type="ablation_measured",
            )
            if arm not in formats:
                row = dict(
                    common, status="unsupported", construction=proof["construction_checks"][arm]
                )
            else:
                packet = formats[arm]
                gc.collect()
                torch.cuda.synchronize()
                fixed, _ = execute(
                    adapter, runtime, packet, tokens=33, forced=forced, sealed=False, audit=False
                )
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                greedy, details = execute(
                    adapter, runtime, packet, tokens=128, sealed=False, audit=False
                )
                allocated, reserved = (
                    torch.cuda.max_memory_allocated(),
                    torch.cuda.max_memory_reserved(),
                )
                torch.save(dict(fixed=fixed, greedy=greedy), out / (arm + "-logits.pt"))
                if arm in plan.get("minimum_controls", {}):
                    previous_arm = plan["minimum_controls"][arm]
                    values = torch.load(
                        Path(plan["minimum_control_root"])
                        / f"workers/{input_index}"
                        / (previous_arm + "-logits.pt"),
                        map_location="cpu",
                        weights_only=True,
                    )
                    if not torch.equal(fixed, values["fixed"]) or not torch.equal(
                        greedy, values["greedy"]
                    ):
                        raise RuntimeError("lossless minimum encoding changed model logits")
                if arm == "shadow" and (
                    not torch.equal(fixed, original["fixed"])
                    or not torch.equal(greedy, original["greedy"])
                ):
                    raise RuntimeError("source shadow changed historical 128-token trajectory")
                if arm.startswith("baseline_") and arm.endswith("__fp32"):
                    method = arm.removeprefix("baseline_").split("__")[0]
                    values = torch.load(
                        OLD / f"workers/{input_index}/{method}--source-logits.pt",
                        map_location="cpu",
                        weights_only=True,
                    )
                    if not torch.equal(fixed, values["fixed"]) or not torch.equal(
                        greedy, values["greedy"]
                    ):
                        raise RuntimeError("historical baseline source control changed logits")
                if arm == "source3_fp16_native_fused_inputs":
                    values = torch.load(
                        out / "source3_fp16_native_text-logits.pt",
                        map_location="cpu",
                        weights_only=True,
                    )
                    if not torch.equal(fixed, values["fixed"]) or not torch.equal(
                        greedy, values["greedy"]
                    ):
                        raise RuntimeError(
                            "integrated exact input fusion changed native parent logits"
                        )
                if arm == "source3_fp32_tiled_text":
                    reference_path = out / "source3_exact_text_source-logits.pt"
                    if plan.get("query_affine", False):
                        reference_path = (
                            Path(plan["query_reference_root"])
                            / f"workers/{input_index}/source3_fp32_tiled_text-logits.pt"
                        )
                    values = torch.load(
                        reference_path,
                        map_location="cpu",
                        weights_only=True,
                    )
                    if not torch.equal(fixed, values["fixed"]) or not torch.equal(
                        greedy, values["greedy"]
                    ):
                        raise RuntimeError("FP32 hardware tiling changed model logits")
                if arm == "source3_direct_fp32":
                    reference_path = (
                        (
                            Path(plan["history_reference_root"])
                            / f"workers/{input_index}/source3_direct_fp32-logits.pt"
                        )
                        if plan.get("sequential_source_history", False)
                        else out / "source3_fp32_tiled_text-logits.pt"
                    )
                    values = torch.load(
                        reference_path,
                        map_location="cpu",
                        weights_only=True,
                    )
                    if not torch.equal(fixed, values["fixed"]) or not torch.equal(
                        greedy, values["greedy"]
                    ):
                        raise RuntimeError("direct construction changed model logits")
                if arm == "source3_exact_text_source":
                    previous_values = torch.load(
                        out / "source3_exclude_global_exact-logits.pt",
                        map_location="cpu",
                        weights_only=True,
                    )
                    if not torch.equal(fixed, previous_values["fixed"]) or not torch.equal(
                        greedy, previous_values["greedy"]
                    ):
                        raise RuntimeError(
                            "common lossless text storage changed source-variant logits"
                        )
                if details["storage"]["initial_active_kv_storage_bytes"] > proof["caps_by_arm"].get(
                    arm, proof["cap_bytes"]
                ):
                    raise RuntimeError("actual source cache exceeds unchanged original cap")
                if digest_tensors(tuple(t for p in packet for t in p.tensors())) != hashes[arm]:
                    raise RuntimeError("decode mutated immutable encoded formats")
                eos = adapter.model.generation_config.eos_token_id
                eos_ids = {eos} if type(eos) is int else set(eos or ())
                scored = parent.scored_text(
                    adapter.processor.tokenizer, greedy.argmax(-1).tolist(), eos_ids
                )
                reference = parent.scored_text(
                    adapter.processor.tokenizer, full.argmax(-1).tolist(), eos_ids
                )
                label_arm = plan.get("minimum_controls", {}).get(arm, arm)
                baseline_arm = label_arm in (
                    "shadow",
                    "shadow_fp16_metadata",
                ) or label_arm.startswith("baseline_")
                details["storage"].update(
                    unchanged_original_selected_memberships=baseline_arm,
                    all_original_prompt_positions_represented=not baseline_arm,
                )
                row = dict(
                    common,
                    status="completed",
                    method=label_arm.removeprefix("baseline_").split("__")[0]
                    if label_arm.startswith("baseline_")
                    else "mosaickv_exact"
                    if baseline_arm
                    else "evicastkv_shared_source_precision_variant",
                    fullkv_fixed_prefix=comparison(full[1:33], fixed[1:]),
                    fullkv_reference_rouge_l=rouge_l_f1(scored["answer"], (reference["answer"],)),
                    scored_answer=scored,
                    storage=details["storage"],
                    decode_wall_seconds=details["decode_wall_seconds"],
                    decode_cuda_ms=details["decode_cuda_ms"],
                    construction_seconds=proof["construction_seconds"][arm],
                    original_own_cap_bytes=proof["caps_by_arm"].get(arm, proof["cap_bytes"]),
                    end_to_end_measured=False,
                    instrumented_history_timing_not_performance_evidence=plan.get(
                        "history_audit", False
                    ),
                    source_component_timing_excludes_shared_protection_setup=not baseline_arm,
                    nominal_selection_ratio=0.5 if baseline_arm else None,
                    prompt_precision_policy="head3_boundary8_KV"
                    if baseline_arm
                    else "three_bit_shared_normalized_source",
                    local_precision_variant_not_unmodified_published_baseline=baseline_arm,
                    query_weighted_affine_refit="query_affine" in arm,
                    sparse_original_precision_exceptions=arm.endswith("exact_exceptions"),
                    coherent_source_columns=arm == "source3_coherent_columns",
                    coherent_column_count=proof["construction_checks"][arm]["fit"]["channels"]
                    if arm == "source3_coherent_columns"
                    else 0,
                    comparison_with_baselines_admissible=plan.get(
                        "baseline_comparison_admissible", True
                    ),
                    draft_forecast_used=arm.endswith(
                        ("query_affine_hybrid", "exact_exceptions", "coherent_columns")
                    ),
                    forecast_overhead_seconds=proof["forecast"][
                        "wall_seconds_excludes_integrity_hashes"
                    ]
                    if arm.endswith(("query_affine_hybrid", "exact_exceptions", "coherent_columns"))
                    else 0.0,
                    quantization_header_dtype=precision_header_label(arm),
                    torch_peak_allocated=allocated,
                    torch_peak_reserved=reserved,
                    inactive_diagnostic_format_bytes=all_format_bytes
                    - backing_bytes(tuple(t for p in packet for t in p.tensors())),
                    whole_request_peak_certified=False,
                    cpu_residual_bytes=0,
                    source_retired_before_decode=True,
                    no_FullKV_fallback_or_forced_draft_emission=True,
                    independent_quality_N=0,
                    different_memory_points_not_byte_matched=True,
                    not_official_MM_Vet_or_PPL=True,
                )
                if teacher_targets is not None:
                    from mosaickv.evaluation.storage_teacher_forcing import score_formats

                    teacher_result = score_formats(
                        adapter,
                        runtime,
                        packet,
                        teacher_targets,
                        maximum=plan["teacher_maximum_tokens"],
                        return_logits=plan.get("sequential_source_history", False),
                        audit_history=plan.get("sequential_source_history", False)
                        and plan["history_audit"],
                    )
                    if plan.get("sequential_source_history", False):
                        original_teacher, original_logits = teacher_result
                        gc.collect()
                        torch.cuda.synchronize()
                        teacher, actual_logits = score_formats(
                            adapter,
                            runtime,
                            packet,
                            teacher_targets,
                            maximum=plan["teacher_maximum_tokens"],
                            sequential_source_history=True,
                            audit_history=plan["history_audit"],
                            return_logits=True,
                        )
                        if not torch.equal(original_logits, actual_logits):
                            raise RuntimeError(
                                "source history changed complete teacher-forcing logits"
                            )
                        for field in (
                            "per_token_nll",
                            "nll_sum",
                            "conditional_greedy_tokens",
                            "evaluated_token_count",
                        ):
                            if teacher[field] != original_teacher[field]:
                                raise RuntimeError(
                                    "source history changed complete teacher-forced likelihood"
                                )
                        teacher["storage"]["raw_history_control"] = dict(
                            storage=original_teacher["storage"],
                            complete_teacher_logits_bitwise=True,
                            original_logit_sha256=digest_tensors((original_logits,)),
                            source_history_logit_sha256=digest_tensors((actual_logits,)),
                            maximum_logit_difference=float(
                                (original_logits - actual_logits).abs().max()
                            ),
                        )
                        del original_logits, actual_logits, teacher_result
                    else:
                        teacher = teacher_result
                    if digest_tensors(tuple(t for p in packet for t in p.tensors())) != hashes[arm]:
                        raise RuntimeError("teacher forcing mutated immutable encoded formats")
                    teacher["encoded_formats_immutable_after_complete_target"] = True
                    if arm == "source3_direct_fp32":
                        reference_rows = (
                            (
                                Path(plan["history_reference_root"])
                                / f"workers/{input_index}/raw.jsonl"
                            )
                            if plan.get("sequential_source_history", False)
                            else out / "raw.jsonl"
                        )
                        reference_arm = (
                            "source3_direct_fp32"
                            if plan.get("sequential_source_history", False)
                            else "source3_fp32_tiled_text"
                        )
                        parent_rows = [
                            json.loads(line) for line in reference_rows.read_text().splitlines()
                        ]
                        original_teacher = next(
                            r["teacher_forced"]
                            for r in parent_rows
                            if r["variant"] == reference_arm
                        )
                        for field in (
                            "per_token_nll",
                            "nll_sum",
                            "conditional_greedy_tokens",
                            "evaluated_token_count",
                        ):
                            if teacher[field] != original_teacher[field]:
                                raise RuntimeError(
                                    "direct construction changed complete teacher forcing"
                                )
                        row["direct_complete_teacher_forcing_bitwise"] = True
                    if arm in plan.get("minimum_controls", {}):
                        prior_rows = [
                            json.loads(line)
                            for line in (
                                Path(plan["minimum_control_root"])
                                / f"workers/{input_index}/raw.jsonl"
                            )
                            .read_text()
                            .splitlines()
                        ]
                        prior = next(
                            r for r in prior_rows if r["variant"] == plan["minimum_controls"][arm]
                        )
                        old_teacher = prior["teacher_forced"]
                        if (
                            teacher["per_token_nll"] != old_teacher["per_token_nll"]
                            or teacher["conditional_greedy_tokens"]
                            != old_teacher["conditional_greedy_tokens"]
                        ):
                            raise RuntimeError(
                                "lossless minimum encoding changed full-target scoring"
                            )
                        row["lossless_controls"] = dict(
                            generation_logits_bitwise=True,
                            complete_teacher_NLL_bitwise=True,
                            original_variant=prior["variant"],
                            original_generation_storage=prior["storage"],
                            original_teacher_storage=old_teacher["storage"],
                        )
                    row.update(
                        teacher_forced=teacher,
                        teacher_target_provenance=teacher_provenance,
                        complete_teacher_forced_PPL=True,
                        not_official_MM_Vet_or_PPL=True,
                        local_two_input_PPL_not_paper_reproduction=True,
                    )
                if plan.get("paired_history_timing", False):
                    from mosaickv.evaluation.exact_history_timing import measure, summarize

                    trial_path = out / (arm + "-timing.jsonl")
                    trial_path.touch(exist_ok=False)

                    def save_trial(value, path=trial_path, provenance=common):
                        with path.open("a") as stream:
                            stream.write(
                                json.dumps(
                                    dict(
                                        provenance,
                                        **value,
                                        method_variant=provenance["variant"],
                                        seed=plan["seed"] + input_index,
                                    ),
                                    allow_nan=False,
                                ) + "\n"
                            )

                    trials = measure(
                        adapter,
                        runtime,
                        packet,
                        greedy,
                        seed=plan["seed"] + input_index,
                        record=save_trial,
                    )
                    row["paired_history_timing"] = summarize(
                        trials, seed=plan["seed"] + input_index
                    )
                    row["paired_history_trials"] = str(trial_path)
                    row["paired_history_trials_sha256"] = sha(trial_path)
                    if digest_tensors(tuple(t for p in packet for t in p.tensors())) != hashes[arm]:
                        raise RuntimeError("timed history mutated encoded prompt formats")
            with (out / "raw.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        isolation_arm = (
            "shadow" if plan.get("sequential_shadow", False) else arms[min(1, len(arms) - 1)]
        )
        if isolation_arm in formats:
            gc.collect()
            again, _ = execute(
                adapter,
                runtime,
                formats[isolation_arm],
                tokens=33,
                forced=forced,
                sealed=False,
                audit=False,
            )
            prior = torch.load(
                out / (isolation_arm + "-logits.pt"), map_location="cpu", weights_only=True
            )["fixed"]
            if not torch.equal(again, prior):
                raise RuntimeError("A-B-A source request isolation failed")
            write_once(
                out / "isolation.json", dict(A_B_A_fixed_prefix_bitwise=True, private_history=True)
            )
        result["status"] = "completed"
    except Exception as exc:
        result["failures"].append(f"{type(exc).__name__}: {exc}")
        with (out / "raw.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    dict(
                        status="failed",
                        variant=arm,
                        task=None if payload is None else payload.get("task"),
                        sample_id=None if payload is None else payload.get("sample_id"),
                        error=result["failures"][-1],
                        source_sha=plan["source_sha"],
                        plan_sha256=expected,
                    )
                )
                + "\n"
            )
    finally:
        result["outputs"] = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
        write_once(out / "result.json", result)
    if result["status"] != "completed":
        raise SystemExit(1)


def analyze(root, expected):
    import math

    plan = previous.common.inputs(root, expected)
    rows, failures = [], []
    for job in plan["jobs"]:
        i = job["index"]
        folder = root / f"workers/{i}"
        try:
            record = previous.read(folder / "result.json")
            if record["source_sha"] != plan["source_sha"] or record["plan_sha256"] != expected:
                raise ValueError("producer provenance changed")
            for name, digest in record["outputs"].items():
                if sha(folder / name) != digest:
                    raise ValueError("producer changed")
            failures.extend(record["failures"])
            rows.extend(
                json.loads(line) for line in (folder / "raw.jsonl").read_text().splitlines()
            )
        except (ValueError, OSError) as exc:
            failures.append(str(exc))
    gates = []
    for arm in (a for a in plan["variants"][1:] if a.startswith("source3")):
        per_input = []
        for job in plan["jobs"]:
            sample = previous.read(job["inputs"][0])["sample_id"]
            group = {
                r["variant"]: r
                for r in rows
                if r.get("sample_id") == sample and r["status"] == "completed"
            }
            a, b = (
                group.get(
                    "source3_fp32_tiled_text" if plan.get("query_affine", False) else "shadow"
                ),
                group.get(arm),
            )
            if a and b:
                per_input.append(
                    dict(
                        sample_id=sample,
                        meaningful_coherent_change=b.get("coherent_column_count", 0) > 0
                        if b.get("coherent_source_columns", False)
                        else True,
                        guard=b["fullkv_fixed_prefix"]["registered_guard_pass"],
                        quality_not_worse=b["fullkv_reference_rouge_l"]
                        >= a["fullkv_reference_rouge_l"],
                        teacher_PPL_not_worse=(
                            b["teacher_forced"]["nll_sum"]
                            / b["teacher_forced"]["evaluated_token_count"]
                            <= a["teacher_forced"]["nll_sum"]
                            / a["teacher_forced"]["evaluated_token_count"]
                        )
                        if "teacher_forced" in b and "teacher_forced" in a
                        else None,
                        storage_reductions={
                            field: 1 - b["storage"][field] / a["storage"][field]
                            for field in (
                                "initial_active_kv_storage_bytes",
                                "final_active_kv_storage_bytes",
                                "peak_active_kv_storage_bytes",
                            )
                        },
                    )
                )
        gates.append(
            dict(
                arm=arm,
                per_input=per_input,
                model_gate=len(per_input) == 2
                and all(
                    p["guard"]
                    and p["meaningful_coherent_change"]
                    and p["quality_not_worse"]
                    and p["teacher_PPL_not_worse"] is not False
                    for p in per_input
                ),
                all_storage_boundaries_smaller=len(per_input) == 2
                and all(all(v > 0 for v in p["storage_reductions"].values()) for p in per_input),
            )
        )
    transport_gate = plan.get("sequential_shadow", False) and len(rows) == 2 and not failures
    if transport_gate:
        transport_gate = all(
            r["status"] == "completed"
            and r["storage"]["sequential_lease_control"]["logits_bitwise_equal"]
            and r["storage"]["sequential_lease_control"]["persistent_storage_identical"]
            for r in rows
        )
    teacher_summary = []
    if plan.get("teacher_metrics", False):
        representatives = {}
        for job in plan["jobs"]:
            representatives.setdefault(job.get("input_index", job["index"]), job["index"])
        for i in representatives.values():
            try:
                control = previous.read(root / f"workers/{i}/teacher_control.json")
                full_teacher = control["ordinary"]
                count = full_teacher["evaluated_token_count"]
                per_input = []
                selected = [
                    r for r in rows if r.get("sample_id") == control["provenance"]["sample_id"]
                ]
                if len(selected) != len(plan.get("evaluation_variants", plan["variants"])):
                    raise ValueError("incomplete teacher-forcing arms")
                for row in selected:
                    teacher = row["teacher_forced"]
                    if (
                        row["status"] != "completed"
                        or teacher["evaluated_token_count"] != count
                        or len(teacher["per_token_nll"]) != count
                        or teacher["target_ids"] != full_teacher["target_ids"]
                        or not math.isclose(
                            math.fsum(teacher["per_token_nll"]), teacher["nll_sum"], abs_tol=1e-9
                        )
                    ):
                        raise ValueError("incomplete or mismatched teacher targets/losses")
                    per_input.append(
                        dict(
                            variant=row["variant"],
                            nll_sum=teacher["nll_sum"],
                            evaluated_token_count=count,
                            PPL=math.exp(teacher["nll_sum"] / count),
                            delta_mean_NLL_from_FullKV=(
                                teacher["nll_sum"] - full_teacher["nll_sum"]
                            )
                            / count,
                            fullkv_reference_ROUGE_L=row["fullkv_reference_rouge_l"],
                            teacher_storage=teacher["storage"],
                            generation_storage=row["storage"],
                        )
                    )
                teacher_summary.append(
                    dict(
                        provenance=control["provenance"],
                        fullkv_PPL=math.exp(full_teacher["nll_sum"] / count),
                        fullkv_reinjection_max_logit_difference=control["maximum_logit_difference"],
                        metrics_kept_separate=True,
                        independent_quality_N=0,
                        rows=per_input,
                    )
                )
            except (KeyError, ValueError, OSError, OverflowError) as exc:
                failures.append("teacher metric analysis: " + str(exc))
    history_gate = (
        plan.get("sequential_source_history", False)
        and len(rows) == expected_row_count(plan)
        and not failures
    )
    history_memory = []
    if history_gate:
        for row in rows:
            if row["status"] != "completed":
                history_gate = False
                continue
            try:
                storage = row["storage"]
                prior = storage["raw_history_control"]
                teacher = row["teacher_forced"]["storage"]
                old_teacher = teacher["raw_history_control"]
                history_gate = (
                    history_gate
                    and prior["logits_bitwise"]
                    and old_teacher["complete_teacher_logits_bitwise"]
                )
                history_memory.append(
                    dict(
                        sample_id=row["sample_id"],
                        variant=row["variant"],
                        generation={
                            k: 1 - storage[k] / prior["storage"][k]
                            for k in (
                                "initial_active_kv_storage_bytes",
                                "final_active_kv_storage_bytes",
                                "peak_active_kv_storage_bytes",
                            )
                        },
                        teacher_forcing={
                            k: 1 - teacher[k] / old_teacher["storage"][k]
                            for k in (
                                "initial_active_kv_storage_bytes",
                                "final_active_kv_storage_bytes",
                                "peak_active_kv_storage_bytes",
                            )
                        },
                    )
                )
            except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
                history_gate = False
                failures.append("history accounting analysis: " + str(exc))
    write_once(
        root / "analysis/decision.json",
        dict(
            complete=len(rows) == expected_row_count(plan)
            and not failures,
            failures=failures,
            rows=rows,
            gates=gates,
            common_sequential_transport_gate=transport_gate,
            teacher_metric_summary=teacher_summary,
            common_exact_history_transport_gate=history_gate,
            history_memory_reductions=history_memory,
            audited_history_memory_gate=history_gate
            and all(
                all(
                    r[phase][key] > 0
                    for phase in ("generation", "teacher_forcing")
                    for key in ("final_active_kv_storage_bytes", "peak_active_kv_storage_bytes")
                )
                for r in history_memory
            ),
            paired_history_timing_complete=plan.get("paired_history_timing", False)
            and not failures
            and len(rows) == expected_row_count(plan)
            and all(
                r.get("paired_history_timing", {}).get("timed_pairs") == 20
                and r.get("paired_history_timing", {}).get("warmup_pairs") == 5
                and sha(Path(r["paired_history_trials"])) == r["paired_history_trials_sha256"]
                for r in rows
            ),
            lossless_minimum_gate=plan.get("minimum_headers", False)
            and len(rows) == 10
            and not failures
            and all(
                r.get("lossless_controls", {}).get("generation_logits_bitwise")
                and r.get("lossless_controls", {}).get("complete_teacher_NLL_bitwise")
                for r in rows
            ),
            source_sha=plan["source_sha"],
            plan_sha256=expected,
            independent_quality_N=0,
            superiority_established=False,
            no_fresh_or_large_campaign_authorized=True,
        ),
    )


def submit(root, expected):
    original = parent.subprocess.check_output

    def scoped(command, **kwargs):
        if command and command[0] == "sbatch":
            command = [
                str(SOURCE / "mosaickv/slurm/evicastkv_shared_source_runtime.sbatch")
                if x.endswith("/evicastkv_funded_delimiter_diagnostic.sbatch")
                else x
                for x in command
            ]
        return original(command, **kwargs)

    with patch.object(parent.subprocess, "check_output", scoped):
        parent.submit(root, expected)


def auto_minimum(root, expected):
    """One bounded engineering successor, never a fresh research campaign."""
    import subprocess

    from mosaickv.evaluation.lossless_affine_minimum import eligible_parent

    prior = Path("/scratch/djy8hg/runs/evicastkv/source-native-metrics-consumed-20261101-v1")
    if sha(prior / "plan.json") != expected:
        raise ValueError("registered parent plan changed")
    control = previous.read(prior / "analysis/decision.json")
    source_quality = eligible_parent(control)
    if not source_quality:
        write_once(
            root / "blocked.json",
            dict(
                source_sha=clean_source(SOURCE),
                parent_decision_sha256=sha(prior / "analysis/decision.json"),
                complete=control["complete"],
                failures=control["failures"],
                source_not_worse_than_old_shadow_both_metrics=source_quality,
                no_GPU_submitted=True,
                no_fresh_or_large_campaign_authorized=True,
            ),
        )
        return
    prepare(
        root, corrected_source=True, fp32_tiling=True, teacher_metrics=True, minimum_headers=True
    )
    digest = sha(root / "plan.json")
    wrapper = str(SOURCE / "mosaickv/slurm/evicastkv_shared_source_runtime.sbatch")
    common = [
        "sbatch",
        "--parsable",
        "--partition=bii",
        "--account=bii_dsc_community",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=2",
        "--mem=16G",
    ]
    cpu = subprocess.check_output(
        [
            *common,
            "--time=00:20:00",
            "--job-name=mk-min-cpu",
            f"--output={root}/cpu-%j.out",
            f"--error={root}/cpu-%j.err",
            wrapper,
            str(SOURCE),
            str(root),
            digest,
            "cpu",
        ],
        text=True,
    ).strip()
    admit = subprocess.check_output(
        [
            *common,
            "--time=00:10:00",
            f"--dependency=afterok:{cpu}",
            "--job-name=mk-min-admit",
            f"--output={root}/admit-%j.out",
            f"--error={root}/admit-%j.err",
            wrapper,
            str(SOURCE),
            str(root),
            digest,
            "submit",
        ],
        text=True,
    ).strip()
    write_once(
        root / "automatic_cpu_chain.json",
        dict(
            cpu=cpu,
            admit=admit,
            parent_plan_sha256=expected,
            source_sha=clean_source(SOURCE),
            no_fresh_or_large_campaign_authorized=True,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "prepare",
            "prepare_sequential",
            "prepare_corrected",
            "prepare_optimized",
            "prepare_fp32_tiling",
            "prepare_teacher_metrics",
            "prepare_lossless_minimum",
            "auto_minimum",
            "prepare_query_affine",
            "prepare_hybrid_query_affine",
            "prepare_sparse_exceptions",
            "prepare_coherent_columns",
            "prepare_sequential_history",
            "prepare_common_history",
            "cpu",
            "submit",
            "worker",
            "analyze",
        ),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.root)
    elif args.mode == "prepare_sequential":
        prepare(args.root, sequential_shadow=True)
    elif args.mode == "prepare_corrected":
        prepare(args.root, corrected_source=True)
    elif args.mode == "prepare_optimized":
        prepare(args.root, corrected_source=True, optimized_source=True)
    elif args.mode == "prepare_fp32_tiling":
        prepare(args.root, corrected_source=True, fp32_tiling=True)
    elif args.mode == "prepare_teacher_metrics":
        prepare(args.root, corrected_source=True, fp32_tiling=True, teacher_metrics=True)
    elif args.mode == "prepare_lossless_minimum":
        prepare(
            args.root,
            corrected_source=True,
            fp32_tiling=True,
            teacher_metrics=True,
            minimum_headers=True,
        )
    elif args.mode == "prepare_query_affine":
        prepare(
            args.root,
            corrected_source=True,
            fp32_tiling=True,
            teacher_metrics=True,
            query_affine=True,
        )
    elif args.mode == "prepare_hybrid_query_affine":
        prepare(
            args.root,
            corrected_source=True,
            fp32_tiling=True,
            teacher_metrics=True,
            query_affine=True,
            hybrid_forecast=True,
        )
    elif args.mode == "prepare_sparse_exceptions":
        prepare(
            args.root,
            corrected_source=True,
            fp32_tiling=True,
            teacher_metrics=True,
            query_affine=True,
            hybrid_forecast=True,
            sparse_exceptions=True,
        )
    elif args.mode == "prepare_coherent_columns":
        prepare(
            args.root,
            corrected_source=True,
            fp32_tiling=True,
            teacher_metrics=True,
            query_affine=True,
            hybrid_forecast=True,
            coherent_channels=True,
        )
    elif args.mode in ("prepare_sequential_history", "prepare_common_history"):
        prepare(
            args.root,
            corrected_source=True,
            fp32_tiling=True,
            teacher_metrics=True,
            query_affine=True,
            sequential_history=True,
            common_history=args.mode == "prepare_common_history",
        )
    elif args.mode == "cpu":
        previous.cpu(args.root, args.plan_sha)
    elif args.mode == "worker":
        worker(args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    else:
        globals()[args.mode](args.root, args.plan_sha)
