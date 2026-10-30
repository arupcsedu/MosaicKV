#!/usr/bin/env python3
"""Finite, consumed-only lossy precision ablation; no automatic paper promotion."""

import argparse
import copy
import gc
import json
import random
import sys
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import evicastkv_storage_successor as base
from evicastkv_packed_attention_consumed import differences, gathered

from mosaickv.config import RunConfig, canonical_config, config_sha256
from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

ROOT = Path("/scratch/djy8hg/runs/evicastkv/precision-tier-consumed-20261030-v2")
PARENT = Path("/scratch/djy8hg/runs/evicastkv/consolidated-history-consumed-20261030-v1/canary")
METHODS = (
    "full_kv",
    "prompt_attention_topk",
    "prefixkv_reimpl",
    "vl_cache_reimpl",
    "lookm_reimpl",
    "mosaickv_exact",
)


def prepare(root, phase):
    if phase != "baselines" or root.exists() or root.name not in ("canary", "comparison"):
        raise ValueError("new consumed canary/comparison only")
    original = base.common.read(PARENT / "plan.json")
    bound = dict(original["bound_files"])
    for name in ("plan.json", "analysis/source_history_decision.json", "analysis/manifest.json"):
        path = PARENT / name
        bound[str(path)] = sha(path)
    if not base.common.read(PARENT / "analysis/source_history_decision.json")[
        "runtime_canary_passed"
    ]:
        raise ValueError("original source-history transport must pass first")
    if root.name == "comparison":
        decision = root.parent / "canary/analysis/precision_decision.json"
        if not base.common.read(decision)["four_bit_canary_passed"]:
            raise ValueError("both 4-bit numerical/memory canaries must pass")
        bound[str(decision)] = sha(decision)
    jobs = []
    for old in original["jobs"]:
        for method in ("mosaickv_exact",) if root.name == "canary" else METHODS:
            job = copy.deepcopy(old)
            job.update(index=len(jobs), method=method)
            for arm in job["arms"]:
                cfg = RunConfig.from_mapping(arm["config"])
                cfg = replace(cfg, generation=replace(cfg.generation, max_new_tokens=128))
                arm.update(config=canonical_config(cfg), config_sha=config_sha256(cfg))
            jobs.append(job)
    plan = dict(
        original,
        source_sha=clean_source(base.SOURCE),
        bound_files=bound,
        jobs=jobs,
        protocol="lossy_prompt_precision_consumed_v2_dynamic_decode_geometry",
        original_KV_bits_required=False,
        strict_bitwise_original_logits_required=False,
        strict_bitwise_required_for_identity_and_repack_only=True,
        no_selection_or_answer_change=False,
        common_format_choice="explicit_lossy_middle_layers_with_exact_text_and_boundary_layers",
        source_history_comparison_kind="lossy_prompt_precision_with_exact_generated_history",
        consolidated_history=False,
        stage=root.name,
        phase="baselines",
        output_tokens=128,
        requested_workers=len(jobs),
        precision_bits=[3, 4] if root.name == "canary" else [4],
        key_group_axis="time",
        value_group_axis="features",
        group_size=128,
        exact_first_selected_positions=4,
        exact_last_selected_positions=8,
        all_selected_text_slots_exact=True,
        first_and_last_exact_layers=2,
        no_group_padding_in_code_payload=True,
        generated_history_lossless=True,
        lossy_prompt_explicit=True,
        method_suffix="_prompt_precision_variant",
        retention_one_disables_quantization=True,
        absolute_kl_limit=0.02,
        absolute_logit_limit=1.0,
        selected_token_agreement_required=1.0,
        warmups=5,
        timed_repetitions=20,
        timing_seed=20261030,
        finite_next_phase_authorized=False,
        maximum_a100s=6,
        no_benchmark_label_access=True,
        common_precision_offered_to_all_methods=True,
        exact_encoded_byte_matching_claim=False,
        whole_request_peak_certified=False,
        fresh_evaluation_authorized=False,
        calibration_authorized=False,
        paper_campaign_authorized=False,
    )
    write_once(root / "plan.json", plan)
    return sha(root / "plan.json")


def gpu_controls():
    import torch

    from mosaickv.evaluation.precision_tier import QuantizedBody, construct, unpack_cpu

    g = torch.Generator().manual_seed(20261030)
    records = []
    for bits in (3, 4):
        for n in (1, 129, 300):
            pair = tuple(
                torch.randn((1, 2, n, 128), generator=g).bfloat16().cuda() for _ in range(2)
            )
            fmt = construct(pair, bits=bits)
            actual = tuple(
                torch.full((1, 2, n + 7, 128), -5, device="cuda", dtype=torch.bfloat16)
                for _ in range(2)
            )
            fmt.fill(None, actual, direct=True)
            for body, original, output in zip(fmt.pair, pair, actual, strict=True):
                cpu = QuantizedBody(
                    body.shape, body.bits, body.axis, body.group, *(t.cpu() for t in body.tensors())
                )
                expected = unpack_cpu(cpu)
                expected[..., fmt.exact_indices.cpu().long(), :] = original.cpu()[
                    ..., fmt.exact_indices.cpu().long(), :
                ]
                if not torch.equal(expected, output[..., :n, :].cpu()) or not bool(
                    (output[..., n:, :] == -5).all()
                ):
                    raise RuntimeError(
                        "GPU precision transport differs from CPU oracle or overwrites history"
                    )
            records.append(
                dict(bits=bits, n=n, cpu_gpu_bits_equal=True, strided_history_preserved=True)
            )
            encoded = construct(pair, bits=bits, encoded_exact=True)
            alternate = tuple(torch.full_like(t, -5) for t in actual)
            encoded.fill(None, alternate, direct=True)
            if not all(
                torch.equal(a.view(torch.int16), b.view(torch.int16))
                for a, b in zip(actual, alternate, strict=True)
            ):
                raise RuntimeError("lossless exact-overlay coding changed precision output bits")
    from mosaickv.evaluation.protected_exact import gpu_controls as exact_controls

    records.extend(exact_controls())
    return records


def study(adapter, prefill, capture, frozen, job, cfg, full_reference, prepared, out, plan):
    import time

    import torch
    from evicastkv_encoded_baselines import fused_choices

    from mosaickv.evaluation import storage_runs
    from mosaickv.evaluation.exact_storage import ExactFormat, encode
    from mosaickv.evaluation.precision_tier import construct
    from mosaickv.evaluation.storage_baselines import baseline_payload

    write_once(out / "precision_gpu_controls.json", gpu_controls())
    source, _ = adapter._legacy_layers(prefill.state.past_key_values)
    if plan["stage"] == "canary":
        exact = tuple(
            ExactFormat("kv", tuple(k.shape), pair=tuple(encode(t) for t in (k, v)))
            for k, v in source
        )
        full, identity = storage_runs.run(adapter, prefill, exact, tokens=32, audit=True)
        if not torch.equal(full, full_reference):
            raise RuntimeError("unquantized retention-one identity failed")
        write_once(
            out / "retention_one.json",
            dict(bitwise_logits=True, quantization_disabled=True, record=identity),
        )
        del exact, full
    positions, pairs, selection = baseline_payload(
        adapter,
        prepared,
        prefill,
        cfg,
        job["method"],
        frozen,
        cfg.bounded_decode.incremental_history.layer_profile_path,
    )
    if pairs is None:
        pairs = gathered(adapter, prefill, positions)
    if job["method"] == "lookm_reimpl":
        reference_formats = tuple(
            ExactFormat("kv", tuple(k.shape), pair=tuple(encode(t) for t in (k, v)))
            for k, v in pairs
        )
    else:
        variants, _ = fused_choices(adapter, prefill, capture, positions, compact=True)
        reference_formats = next(iter(variants.values()))
    common = dict(tokens=128, source_history=True, source_history_constructor="direct")
    if plan.get("common_history_consolidated"):
        common["consolidated_history"] = True
    reference, reference_record = storage_runs.run(
        adapter, prefill, reference_formats, audit=True, **common
    )
    controls, outputs = {}, {"reference": reference}
    trials = []
    write_once(out / "selection.json", selection)
    for bits in plan["precision_bits"]:
        torch.cuda.synchronize()
        start = time.monotonic()
        memberships = (
            positions if positions is not None else selection["selected_source_positions_by_layer"]
        )
        text = {
            i
            for span in prefill.state.modality_map
            if span.modality == "text"
            for i in range(span.start, span.end)
        }
        protected = [
            tuple(
                sorted(
                    {
                        j
                        for head in heads
                        for j, source_position in enumerate(head)
                        if source_position in text
                    }
                )
            )
            for heads in memberships
        ]
        formats = tuple(
            reference_formats[i]
            if i < 2 or i >= len(pairs) - 2
            else construct(
                pair,
                bits=bits,
                protected=slots,
                encoded_exact=plan.get("encode_protected_exact", False),
            )
            for i, (pair, slots) in enumerate(zip(pairs, protected, strict=True))
        )
        torch.cuda.synchronize()
        construction = time.monotonic() - start
        write_once(
            out / f"precision_{bits}_construction.json",
            dict(
                construction_seconds=construction,
                format_count=len(formats),
                no_per_token_stride_specialization=True,
            ),
        )
        actual, record = storage_runs.run(adapter, prefill, formats, audit=True, **common)
        write_once(
            out / f"precision_{bits}_decode.json",
            dict(record=record, comparison32=differences(reference[:32], actual[:32])),
        )
        record["storage"].update(
            includes="lossy_prompt_codes_FP32_scales_minima,exact_protected_text_rows,lossless_generated_source_history,temporary_BF16_layer_leases,observed_workspace",
            prompt_precision_lossy=True,
            prompt_bits=bits,
            all_selected_text_slots_exact=True,
            protected_text_lossless_codec=plan.get("encode_protected_exact", False),
        )
        exact_codec_pass = True
        raw_exact_record = None
        if plan.get("encode_protected_exact"):
            raw_formats = tuple(
                reference_formats[i]
                if i < 2 or i >= len(pairs) - 2
                else construct(pair, bits=bits, protected=slots)
                for i, (pair, slots) in enumerate(zip(pairs, protected, strict=True))
            )
            raw_logits, raw_exact_record = storage_runs.run(
                adapter, prefill, raw_formats, audit=True, **common
            )
            exact_codec_pass = torch.equal(raw_logits, actual)
            torch.save(raw_logits, out / f"precision_{bits}_raw_exact_control.pt")
            del raw_formats, raw_logits
        # Repacking the dequantized values must preserve the exact model arithmetic.
        dequantized = tuple(
            ExactFormat(
                "kv", f.shape, pair=tuple(encode(t) for t in f.materialize(layer.self_attn))
            )
            for f, layer in zip(formats, adapter._language_layers(), strict=True)
        )
        clone, _ = storage_runs.run(adapter, prefill, dequantized, tokens=32, audit=True)
        clone_pass = torch.equal(clone, actual[:32])
        del clone, dequantized
        comparison = differences(reference[:32], actual[:32])
        numerical = (
            clone_pass
            and exact_codec_pass
            and comparison["token_agreement"] == 1
            and comparison["maximum_logit_KL"] <= 0.02
            and comparison["maximum_logit_difference"] <= 1.0
        )
        fields = [
            "initial_active_kv_storage_bytes",
            "final_active_kv_storage_bytes",
            "peak_active_kv_storage_bytes",
        ]
        memory = all(record["storage"][k] < reference_record["storage"][k] for k in fields)
        controls[str(bits)] = dict(
            comparison=comparison,
            exploratory_128_token_comparison=differences(reference, actual),
            clone_repack_bitwise=clone_pass,
            lossless_exact_codec_128_bitwise=exact_codec_pass,
            raw_exact_control=raw_exact_record,
            numerical_gate=numerical,
            smaller_initial_final_peak=memory,
            constructor_seconds=construction,
            candidate=record,
        )
        outputs[str(bits)] = actual
        timing_eligible = (numerical and memory) or (
            plan.get("time_all_transport_valid_methods", False) and clone_pass and exact_codec_pass
        )
        if plan["stage"] == "comparison" and timing_eligible and job["method"] != "full_kv":
            rng = random.Random(plan["timing_seed"])
            for trial in range(-5, 20):
                order = [("reference", reference_formats), (str(bits), formats)]
                rng.shuffle(order)
                for order_index, (variant, body) in enumerate(order):
                    gc.collect()
                    logits, timed = storage_runs.run(adapter, prefill, body, audit=False, **common)
                    if not torch.equal(logits, outputs[variant]):
                        raise RuntimeError("repeated precision run changed its own logits")
                    timed.pop("storage")
                    trials.append(
                        dict(
                            trial=trial,
                            warmup=trial < 0,
                            variant=variant,
                            order_index=order_index,
                            **timed,
                        )
                    )
    row = dict(
        method=job["method"],
        result_label=job["method"] + "_prompt_precision_variant",
        reference=reference_record,
        controls=controls,
        trials=trials,
        annotation_targets_used=False,
        source_memberships_unchanged=True,
        precision_changes_are_lossy=True,
        no_independent_quality_claim=True,
    )
    torch.save(outputs, out / "logits.pt")
    with (out / "predictions.jsonl").open("x") as stream:
        for name, logits in outputs.items():
            stream.write(
                json.dumps(
                    dict(
                        sample_id=job["sample_id"],
                        task=job["task"],
                        method=job["method"],
                        variant=name,
                        method_label=(
                            "full_position_lossless_control"
                            if job["method"] == "full_kv" and name == "reference"
                            else job["method"]
                            + (
                                "_common_lossless_transport"
                                if name == "reference"
                                else "_prompt_precision_variant"
                            )
                        ),
                        cache_precision_bits=None if name == "reference" else int(name),
                        fixed_output_length_for_timing=128,
                        requires_eos_aware_quality_reduction=True,
                        generated_token_ids=logits.argmax(-1).tolist(),
                        answer=adapter.processor.tokenizer.decode(
                            logits.argmax(-1).tolist(), skip_special_tokens=True
                        ),
                        manifest_path=str(out / "manifest.json"),
                        annotation_targets_used=False,
                    )
                )
                + "\n"
            )
    write_once(out / "precision_controls.json", row)
    return [row]


original_analyze = base.analyze


def analyze(root, expected):
    import numpy as np

    original_analyze(root, expected)
    gate = base.common.read(root / "analysis/GATE.json")
    plan = base.common.read(root / "plan.json")
    selected = str(plan.get("comparison_precision_bits", 4))
    complete = not gate["failures"] and len(gate["controls"]) == len(
        base.common.read(root / "plan.json")["jobs"]
    )
    passed = complete and all(
        r["controls"][selected]["numerical_gate"]
        and r["controls"][selected]["smaller_initial_final_peak"]
        for r in gate["controls"]
    )
    timing = []
    for row in gate["controls"]:
        timed = [t for t in row["trials"] if not t["warmup"]]
        a = {t["trial"]: t["decode_wall_seconds"] for t in timed if t["variant"] == "reference"}
        b = {t["trial"]: t["decode_wall_seconds"] for t in timed if t["variant"] == selected}
        if a or b:
            if set(a) != set(range(20)) or set(a) != set(b):
                raise ValueError("missing paired timing repetitions")
            ratios = np.array([b[i] / a[i] for i in sorted(a)])
            boot = np.median(
                ratios[np.random.default_rng(20261030).integers(0, 20, (10000, 20))], axis=1
            )
            timing.append(
                dict(
                    task=row["task"],
                    method=row["method"],
                    ratio_p50=float(np.median(ratios)),
                    ratio_ci95=np.percentile(boot, [2.5, 97.5]).tolist(),
                )
            )
    write_once(
        root / "analysis/precision_decision.json",
        dict(
            four_bit_canary_passed=passed and selected == "4",
            registered_precision_bits=int(selected),
            registered_precision_passed=passed,
            complete=complete,
            failures=gate["failures"],
            timing=timing,
            superiority_established=False,
            whole_request_peak_certified=False,
            fresh_evaluation_authorized=False,
            paper_campaign_authorized=False,
        ),
    )
    if root.name == "canary" and passed:
        child = root.parent / "comparison"
        digest = prepare(child, "baselines")
        base.launch.submit(
            child,
            digest,
            script_name=base.SCRIPT_NAME,
            worker_minutes=30,
            memory_gb=24,
        )


def admit(root, expected):
    plan = base.common.inputs(root, expected)
    script = plan.get("worker_script", "evicastkv_precision_tier.sbatch")
    if script not in ("evicastkv_precision_tier.sbatch", "evicastkv_precision_threebit.sbatch"):
        raise ValueError("unregistered precision worker script")
    proof = base.common.read(root / "cpu_record.json")
    xml = Path(proof["cpu_xml"])
    if (
        proof["source_sha"] != plan["source_sha"]
        or proof["returncode"] != 0
        or proof["cpu_xml_sha256"] != sha(xml)
    ):
        raise ValueError("CPU proof changed or does not match immutable source")
    document = ET.parse(xml).getroot()
    suites = list(document.iter("testsuite"))
    count = sum(int(s.get("tests", "0")) for s in suites)
    if (
        count < 2699
        or count != len(list(document.iter("testcase")))
        or any(int(s.get(k, "0")) for s in suites for k in ("failures", "errors", "skipped"))
    ):
        raise ValueError("full CPU suite must pass before GPU admission")
    from control_a100_schedule import ACTIVE, gpu_count, read_queue

    needed = min(6, len(plan["jobs"]))
    queue = read_queue()
    if sum(gpu_count(r["gres"]) for r in queue if r["state"] in ACTIVE) + needed > 6 or any(
        gpu_count(r["gres"]) and r["state"] == "PENDING" for r in queue
    ):
        raise ValueError("precision array would exceed or race six aggregate A100s")
    write_once(
        root / "cpu_admission.json",
        dict(source_sha=plan["source_sha"], junit_sha256=sha(xml), tests=count),
    )
    base.launch.submit(
        root,
        expected,
        script_name=script,
        worker_minutes=30,
        memory_gb=24,
    )


if __name__ == "__main__":
    base.BASE = ROOT
    base.__dict__["PHASES"] = ("baselines",)
    base.SCRIPT_NAME = "evicastkv_precision_tier.sbatch"
    base.prepare, base.baseline_study, base.analyze = prepare, study, analyze
    if len(sys.argv) > 1 and sys.argv[1] in ("cpu", "admit"):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("mode", choices=("cpu", "admit"))
        parser.add_argument("--root", type=Path, required=True)
        parser.add_argument("--plan-sha", required=True)
        args = parser.parse_args()
        if args.mode == "cpu":
            from evicastkv_consolidated_history import cpu

            cpu(args.root, args.plan_sha)
        else:
            admit(args.root, args.plan_sha)
    else:
        base.main()
