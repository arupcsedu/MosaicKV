#!/usr/bin/env python3
"""Bounded, consumed-input transport and realized-storage repair controls.

Three separate questions: original retained-bit fidelity, same-selected native
transport parity, and FullKV predictive fidelity. Passing the first two NEVER
changes the third. No new samples, paper claims, or automatic large campaign.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
import traceback
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import evicastkv_construct_shared_byte as constructed
import evicastkv_review_source_controls as previous
import evicastkv_review_source_prefix as fixed

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
TARGETS = (47710208, 51380224)
SELECTORS = ("prompt_topk_original_exact_control", "shared_prior_original_exact_control")
POLICY = dict(
    id="fox_V20_fidelity_and_final_peak_byte_repair_controls_v1",
    classification="two_consumed_inputs_engineering_diagnostic_not_paper_evidence",
    model=fixed.MODEL,
    precision="bf16",
    attention="eager",
    predictions=128,
    updates=127,
    seed=13,
    initial_prompt_caps_bytes=TARGETS,
    cap_is_lifetime=False,
    mandatory="all_text_first4_last8_original_positions",
    raw_selection="original_post_RoPE_BF16_KV_no_requantization",
    shared_prior_weight=0.5,
    shared_prior_is_frozen_graph_method=False,
    disabled_mechanisms=[
        "forecast_rollout",
        "evidence_graph",
        "value_utility",
        "submodular_selector",
        "prototypes",
        "repair",
    ],
    common_history="raw_BF16_append_only_16_slot_growth_allocated128_valid127",
    common_reader="ExactCache_unmodified_HF_eager_one_layer_lease_at_a_time",
    selected_native_reference="SAME_selected_original_KV_not_FullKV",
    retained_original_bits_required=True,
    transport_all128_logits_bitwise_required=True,
    FullKV_guard=fixed.POLICY["guard"],
    FullKV_guard_reference_never_replaced=True,
    source_control="same_three_bit_packet_fused_vs_native_quantized_source_projection",
    source_control_not_lossless_FullKV=True,
    workspace_padding_to_force_matching_forbidden=True,
    whole_request_KV_peak_certified=False,
    baseline_or_quality_superiority_claim=False,
    labels_or_references_used=False,
    maximum_workers=2,
    maximum_A100s=6,
    worker_soft_seconds=1500,
    fresh_or_large_campaign_authorized=False,
)


def read(path):
    return json.loads(Path(path).read_text())


def prepare(root):
    root = Path(root)
    revision = clean_source(SOURCE)
    old, jobs, bindings = fixed._bind_parent(fixed.DEFAULT_PARENT)
    costs = constructed.V10 / "byte-feasibility.json"
    table = read(costs)
    if table["common_conditional_initial_targets_bytes"][fixed.ARMS[5]] != list(TARGETS):
        raise ValueError("frozen V10 constructed targets changed")
    bindings[str(costs)] = sha(costs)
    prepared = []
    for i, job in enumerate(jobs):
        witnesses = table["inputs"][i]["arms"][fixed.ARMS[5]]["conditional_layer_mix_witnesses"]
        prepared.append(dict(job, witnesses={str(t): witnesses[str(t)] for t in TARGETS}))
    value = dict(
        source_sha=revision,
        policy=POLICY,
        policy_sha256=digest(POLICY),
        config=old["config"],
        jobs=prepared,
        bindings=bindings,
        expected_workers=2,
        expected_rows_per_worker=9,
        no_new_identity_access=True,
    )
    write_once(root / "plan.json", value)
    return dict(status="prepared", source_sha=revision, workers=2, expected_rows=18)


def load(root):
    plan = read(Path(root) / "plan.json")
    clean_source(SOURCE, plan["source_sha"])
    if digest(plan["policy"]) != digest(POLICY) or plan["policy_sha256"] != digest(POLICY):
        raise ValueError("frozen V20 policy changed")
    for name, expected in plan["bindings"].items():
        if sha(Path(name)) != expected:
            raise ValueError(f"bound consumed artifact changed: {name}")
    return plan


def parity(reference, candidate):
    import torch

    if (
        reference.shape != candidate.shape
        or reference.shape[0] != 128
        or reference.dtype != candidate.dtype
    ):
        raise ValueError("complete identical 128-prediction shapes required")
    if not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():
        raise ValueError("nonfinite transport outputs")
    diff = (reference.double() - candidate.double()).abs()
    tokens = reference.argmax(-1) == candidate.argmax(-1)
    bitwise = bool(
        torch.equal(
            reference.contiguous().view(torch.uint8), candidate.contiguous().view(torch.uint8)
        )
    )
    return dict(
        predictions=128,
        identical_greedy_predictions=int(tokens.sum()),
        maximum_logit_absolute_difference=float(diff.max()),
        bitwise_logits_equal=bitwise,
        pass_all128_bitwise=bool(bitwise and tokens.all()),
        reference="same_selected_native_or_retention1_as_explicitly_labeled",
    )


def storage_pair(left, right):
    keys = (
        "initial_active_kv_storage_bytes",
        "final_active_kv_storage_bytes",
        "peak_active_kv_storage_bytes",
    )
    a, b = left["storage"], right["storage"]
    values = {}
    for key in keys:
        if type(a.get(key)) is not int or type(b.get(key)) is not int or min(a[key], b[key]) < 1:
            raise ValueError("positive measured backing bytes required at all three boundaries")
        values[key] = dict(left=a[key], right=b[key], difference=b[key] - a[key])
    history = all(
        record["history_updates_per_layer"] == [127] * len(record["history_updates_per_layer"])
        and bool(record["history_updates_per_layer"])
        and record["outstanding_layer_leases"] == 0
        for record in (a, b)
    )
    matched = all(item["difference"] == 0 for item in values.values())
    return dict(
        measured=values,
        all_three_measured_boundaries_equal=matched,
        generated_history_complete_and_no_lease_leak=history,
        pass_constructed_decode_boundary=matched and history,
        boundary="constructed_active_cache_backing_including_history_growth_and_layer_leases",
        metadata_separately_reported=True,
        whole_request_KV_peak_certified=False,
    )


def _metadata(state):
    """Actual GPU mask/position storage outside the existing cache ledger."""
    import torch

    found = {}

    def visit(value, path):
        if isinstance(value, torch.Tensor) and value.device.type == "cuda":
            s = value.untyped_storage()
            key = (str(value.device), s.data_ptr(), s.nbytes())
            found.setdefault(key, dict(bytes=s.nbytes(), paths=[]))["paths"].append(path)
        elif isinstance(value, dict):
            for name, item in value.items():
                visit(item, f"{path}.{name}")
        elif isinstance(value, (tuple, list)):
            for i, item in enumerate(value):
                visit(item, f"{path}[{i}]")

    visit(state.attention_mask, "attention_mask")
    visit(state.model_state, "model_state")
    return dict(backing_bytes=sum(v["bytes"] for v in found.values()), owners=list(found.values()))


def _raw_run(adapter, prefill, formats, prefix, cap=None, native_source=None):
    import torch

    from mosaickv.evaluation.storage_runs import run

    records = []
    ordinary = adapter.decode_one_token

    def observed(*args, **kwargs):
        if not records:
            records.append(dict(boundary="before_first_update", **_metadata(args[1])))
        output = ordinary(*args, **kwargs)
        records.append(dict(boundary=f"completed_update_{len(records)}", **_metadata(output.state)))
        return output

    scope = native_source if native_source is not None else nullcontext()
    with patch.object(adapter, "decode_one_token", observed), scope:
        logits, measured = run(
            adapter,
            prefill,
            formats,
            tokens=128,
            forced=prefix,
            audit=True,
            sealed=False,
            source_history=False,
            shared_workspace=False,
            batched_history=False,
            sequential_lease=False,
            initial_physical_byte_cap=cap,
        )
    measured["state_GPU_metadata_outside_cache_ledger"] = records
    measured["metadata_final_backing_bytes"] = records[-1]["backing_bytes"]
    measured["metadata_maximum_observed_backing_bytes"] = max(r["backing_bytes"] for r in records)
    measured["whole_request_KV_peak_certified"] = False
    measured["not_latency_benchmark"] = True
    if native_source is not None:
        measured["native_quantized_source_control"] = dict(native_source.report)
    torch.cuda.synchronize()
    return logits, measured


def worker(root, index):
    import torch
    from evicastkv_campaign import environment
    from evicastkv_request_validation import load_messages

    from mosaickv.adapters.huggingface.registry import load_hf_adapter
    from mosaickv.config import RunConfig, config_sha256
    from mosaickv.evaluation.exact_selected_control import (
        build_raw_formats,
        mandatory_positions,
        raw_format_report,
        run_native_same_selected,
        select_uniform_positions,
    )
    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
    from mosaickv.evaluation.native_quantized_source_control import NativeQuantizedSourceControl
    from mosaickv.evaluation.phase_capacity import PhaseCapture
    from mosaickv.evaluation.request_precision import PromptQueryCapture
    from mosaickv.evaluation.shared_source_format import SourceCapture
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    root = Path(root)
    plan = load(root)
    if index not in range(2) or not os.environ.get("SLURM_JOB_ID", "").isdigit():
        raise ValueError("two registered Slurm workers only")
    out = root / f"workers/{index}"
    out.mkdir(parents=True, exist_ok=False)
    rows, failures, tensors = [], [], {}
    started = time.monotonic()
    try:
        if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name():
            raise ValueError("one visible A100 required")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ValueError("CUBLAS_WORKSPACE_CONFIG must precede Python")
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(12)
        torch.manual_seed(13)
        fixed.analysis_cpu_controls()
        job = plan["jobs"][index]
        resolved = json.loads(json.dumps(job["config"]))
        resolved["execution"]["seed"] = POLICY["seed"]
        cfg = RunConfig.from_mapping(resolved)
        if (cfg.model.id, cfg.model.revision) != fixed.MODEL or cfg.model.precision.value != "bf16":
            raise ValueError("same pinned BF16 checkpoint required")
        payload = fixed.label_blind_payload(job["inputs"][0])
        write_once(
            out / "manifest.json",
            dict(
                source_sha=plan["source_sha"],
                source_dirty=False,
                config_sha256=digest(dict(model_loader_config=resolved, execution_policy=POLICY)),
                model_loader_config_sha256=config_sha256(cfg),
                plan_sha256=sha(root / "plan.json"),
                model=plan["config"]["model"],
                dataset=resolved["dataset"],
                sample_id=payload["sample_id"],
                input_sha256=sha(Path(job["inputs"][0])),
                environment=environment(),
                GPU=asdict(capture_gpu_environment(torch)),
                backend="hf",
                attention="eager",
                seed=13,
                measurement_type=POLICY["classification"],
                vllm="not_used",
                sglang="not_used",
                evaluator_source_sha=plan["source_sha"],
                labels_or_answers_read=False,
            ),
        )
        adapter = load_hf_adapter(
            cfg.model.id,
            revision=cfg.model.revision,
            model_kwargs=_model_kwargs(cfg, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        prepared = adapter.prepare_inputs(load_messages(payload))
        with (
            PromptQueryCapture(adapter, summaries=True, normalized=False, prior=True) as captured,
            SourceCapture(adapter) as normalized,
            PhaseCapture(adapter) as rotary,
        ):
            prefill = adapter.prefill(prepared, capture_queries=False, capture_attentions=False)
        pairs, _ = adapter._legacy_layers(prefill.state.past_key_values)
        n, heads, width = pairs[0][0].shape[2], pairs[0][0].shape[1], pairs[0][0].shape[-1]
        mandatory = mandatory_positions(prepared, n)
        verifier_tensors = (
            tuple(t for pair in pairs for t in pair)
            + tuple(normalized.records.values())
            + tuple(captured.windows.values())
            + tuple(captured.priors.values())
            + tuple(rotary.phases)
        )
        verifier_bytes = backing_bytes(verifier_tensors)
        write_once(
            out / "verification_scope.json",
            dict(
                retained_verifier_source_and_capture_backing_bytes=verifier_bytes,
                verifier_owners_excluded_from_active_decode_cache_boundary=True,
                verifier_capture_and_prefill_not_online_persistent_cache=True,
                verifier_inventory_is_partial=True,
                additional_verifier_owners=(
                    "prepared media tensors, prefill logits, "
                    "input/output IDs and metadata are not in this subtotal"
                ),
                whole_request_KV_peak_certified=False,
                source_positions=n,
                mandatory_positions=list(mandatory),
                original_logical_length=prefill.state.logical_sequence_length,
                initial_next_decode_position=prefill.state.next_decode_position,
                new_model_or_dataset_access=False,
            ),
        )
        full_positions = tuple((tuple(range(n)),) * heads for _ in pairs)
        full_formats = build_raw_formats(adapter, pairs, full_positions, mandatory=mandatory)
        reference, native_report = run_native_same_selected(
            adapter, prefill, full_formats, tokens=128
        )
        tensors["FullKV_native_greedy"] = reference
        prefix = fixed.forced_prefix(reference)

        def save(arm, cap, logits, measured, extra=None):
            name = f"{arm}__{cap if cap is not None else 'full'}"
            if name in tensors:
                raise ValueError("duplicate immutable row identity")
            tensors[name] = logits
            row = dict(
                row_id=name,
                arm=arm,
                initial_cap_bytes=cap,
                sample_id=payload["sample_id"],
                task=payload["task"],
                status="completed",
                measured=measured,
                FullKV_quality=fixed.comparisons(reference, logits),
                source_sha=plan["source_sha"],
                plan_sha256=sha(root / "plan.json"),
                conditioning_prefix_sha256=digest(prefix),
                benchmark_scoring_performed=False,
                whole_request_KV_peak_certified=False,
                disabled_mechanisms=POLICY["disabled_mechanisms"],
            )
            if extra:
                row.update(extra)
            rows.append(row)
            write_once(out / "traces" / f"{name}.json", row)
            return row

        logits, measured = _raw_run(adapter, prefill, full_formats, prefix)
        save(
            "retention1_original_exact_transport",
            None,
            logits,
            measured,
            dict(
                original_bits=raw_format_report(full_formats),
                native_transport=parity(reference, logits),
                native_reference=native_report,
            ),
        )
        del logits, measured, full_formats
        gc.collect()

        for cap in TARGETS:
            if time.monotonic() - started > POLICY["worker_soft_seconds"]:
                raise TimeoutError("bounded diagnostic soft deadline")
            capacity = fixed.raw_topk_capacity(
                cap,
                layers=len(pairs),
                heads=heads,
                length=n,
                width=width,
                mandatory_count=len(mandatory),
            )
            if not capacity["feasible"] or capacity["actual_bytes"] != cap:
                raise ValueError("registered common raw capacity is not exactly attainable")
            count = capacity["count"]
            for selector in SELECTORS:
                scores = []
                for layer in range(len(pairs)):
                    score = captured.windows[layer][0].float().mean(-2)
                    if selector == SELECTORS[1]:
                        score = 0.5 * score + 0.5 * captured.priors[layer]
                    scores.append(score)
                positions = tuple(select_uniform_positions(s, count, mandatory) for s in scores)
                formats = build_raw_formats(adapter, pairs, positions, mandatory=mandatory)
                original = raw_format_report(formats)
                native, report = run_native_same_selected(
                    adapter, prefill, formats, tokens=128, forced=prefix
                )
                tensors[f"native_same_selected__{selector}__{cap}"] = native
                logits, measured = _raw_run(adapter, prefill, formats, prefix, cap)
                save(
                    selector,
                    cap,
                    logits,
                    measured,
                    dict(
                        original_bits=original,
                        native_transport=parity(native, logits),
                        native_reference=report,
                        selector_uses_only_prompt_evidence=True,
                        six_component_or_frozen_paper_method=False,
                    ),
                )
                del formats, native, logits, measured, scores, positions
                gc.collect()
            # This is deliberately separate from the exact-selected repair.
            # Old low-byte FullKV failures are retained, never relabeled.
            proof = {}
            args = (
                adapter,
                prepared,
                prefill,
                captured,
                normalized,
                rotary,
                "source3_attention_optional",
            )
            formats, _ = constructed.mixed_formats(args, job["witnesses"][str(cap)], cap, proof)
            before = previous._packet_hash(formats)
            fused, measured = _raw_run(adapter, prefill, formats, prefix, cap)
            save(
                "source3_fused_same_packet_raw_history",
                cap,
                fused,
                measured,
                dict(construction_proof=proof),
            )
            control = NativeQuantizedSourceControl(
                formats, [layer.self_attn for layer in adapter._language_layers()]
            )
            native, measured = _raw_run(adapter, prefill, formats, prefix, cap, control)
            if before != previous._packet_hash(formats):
                raise RuntimeError("source packet mutated across reader controls")
            save(
                "source3_native_quantized_reader_raw_history",
                cap,
                native,
                measured,
                dict(
                    fused_vs_native_quantized_reader=parity(fused, native),
                    source_packet_unchanged=True,
                    original_FullKV_losslessness_claim=False,
                    extra_native_projection_workspace_charged_separately=control.report[
                        "tensor_workspace_upper_bound"
                    ],
                ),
            )
            del formats, control, fused, native, measured
            gc.collect()
    except BaseException as exc:
        failures.append(
            dict(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        )
    finally:
        if tensors:
            torch.save(tensors, out / "logits.pt")
        with (out / "raw.jsonl").open("x") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            for failure in failures:
                stream.write(json.dumps(dict(status="failed", **failure), sort_keys=True) + "\n")
        outputs = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()}
        result = dict(
            status="completed" if not failures and len(rows) == 9 else "failed",
            source_sha=plan["source_sha"],
            plan_sha256=sha(root / "plan.json"),
            failures=failures,
            completed_rows=len(rows),
            expected_rows=9,
            wall_seconds=time.monotonic() - started,
            outputs=outputs,
            SLURM_JOB_ID=os.environ.get("SLURM_JOB_ID"),
            SLURM_ARRAY_JOB_ID=os.environ.get("SLURM_ARRAY_JOB_ID"),
            SLURM_ARRAY_TASK_ID=os.environ.get("SLURM_ARRAY_TASK_ID"),
        )
        write_once(out / "result.json", result)
    return result


def analyze(root):
    root = Path(root)
    if not (root / "plan.json").is_file():
        gate = dict(
            complete=False,
            failures=[
                dict(error="preparation did not produce an immutable plan; no GPU promotion")
            ],
            calibration_or_paper_campaign_authorized=False,
        )
        write_once(root / "analysis/GATE_fidelity_byte_v20.json", gate)
        return gate
    plan = load(root)
    fixed.analysis_cpu_controls()
    import torch

    observations, failures = [], []
    for index in range(2):
        out = root / f"workers/{index}"
        try:
            record = read(out / "result.json")
            if (
                record["status"] != "completed"
                or record["completed_rows"] != 9
                or record["failures"]
            ):
                raise ValueError(f"worker incomplete: {record}")
            if record["source_sha"] != plan["source_sha"] or record["plan_sha256"] != sha(
                root / "plan.json"
            ):
                raise ValueError("worker immutable provenance mismatch")
            for name, expected in record["outputs"].items():
                if sha(out / name) != expected:
                    raise ValueError("worker output hash mismatch")
            tensors = torch.load(out / "logits.pt", weights_only=True, map_location="cpu")
            rows = [json.loads(line) for line in (out / "raw.jsonl").read_text().splitlines()]
            by = {(r["arm"], r["initial_cap_bytes"]): r for r in rows}
            if len(by) != 9:
                raise ValueError("duplicate/missing rows")
            reference = tensors["FullKV_native_greedy"]
            for row in rows:
                fixed.verify_recomputed_diagnostics(
                    fixed.comparisons(reference, tensors[row["row_id"]]), row["FullKV_quality"]
                )
                if "native_transport" in row:
                    baseline = (
                        reference
                        if row["initial_cap_bytes"] is None
                        else tensors[
                            f"native_same_selected__{row['arm']}__{row['initial_cap_bytes']}"
                        ]
                    )
                    if parity(baseline, tensors[row["row_id"]]) != row["native_transport"]:
                        raise ValueError("native transport diagnostic recomputation changed")
                if "fused_vs_native_quantized_reader" in row:
                    baseline = tensors[
                        f"source3_fused_same_packet_raw_history__{row['initial_cap_bytes']}"
                    ]
                    if (
                        parity(baseline, tensors[row["row_id"]])
                        != row["fused_vs_native_quantized_reader"]
                    ):
                        raise ValueError("source reader contrast recomputation changed")
            comparisons = []
            for cap in TARGETS:
                left, right = (by[(selector, cap)] for selector in SELECTORS)
                match = storage_pair(left["measured"], right["measured"])
                metadata_equal = all(
                    left["measured"][key] == right["measured"][key]
                    for key in (
                        "metadata_final_backing_bytes",
                        "metadata_maximum_observed_backing_bytes",
                    )
                )
                comparisons.append(
                    dict(
                        initial_cap_bytes=cap,
                        byte_matching=match,
                        GPU_mask_position_metadata_backing_equal=metadata_equal,
                        exact_transport_pass=all(
                            r["native_transport"]["pass_all128_bitwise"] for r in (left, right)
                        ),
                        retained_original_bits_verified=all(
                            r["original_bits"]["original_KV_bitwise_verified"]
                            for r in (left, right)
                        ),
                        source_reader_control=by[
                            ("source3_native_quantized_reader_raw_history", cap)
                        ],
                        exact_selected_FullKV_quality={
                            r["arm"]: r["FullKV_quality"] for r in (left, right)
                        },
                    )
                )
            observations.append(
                dict(
                    worker=index,
                    sample_id=rows[0]["sample_id"],
                    task=rows[0]["task"],
                    retention1_native_transport=by[("retention1_original_exact_transport", None)][
                        "native_transport"
                    ],
                    caps=comparisons,
                )
            )
        except BaseException as exc:
            failures.append(dict(worker=index, error_type=type(exc).__name__, error=str(exc)))
    engineering = (
        not failures
        and len(observations) == 2
        and all(
            o["retention1_native_transport"]["pass_all128_bitwise"]
            and all(
                c["exact_transport_pass"]
                and c["retained_original_bits_verified"]
                and c["byte_matching"]["pass_constructed_decode_boundary"]
                and c["GPU_mask_position_metadata_backing_equal"]
                for c in o["caps"]
            )
            for o in observations
        )
    )
    gate = dict(
        source_sha=plan["source_sha"],
        plan_sha256=sha(root / "plan.json"),
        complete=not failures and len(observations) == 2,
        failures=failures,
        observations=observations,
        original_exact_selected_transport_and_common_byte_engineering_gate_pass=engineering,
        original_three_bit_FullKV_fidelity_failure_cleared=False,
        old_negative_results_preserved=True,
        whole_request_KV_peak_certified=False,
        fresh_confirmation_or_superiority_established=False,
        calibration_or_paper_campaign_authorized=False,
        next_action=(
            "inspect original-FullKV guards and source-reader contrast; "
            "no automatic cohort expansion"
        ),
    )
    write_once(root / "analysis/GATE_fidelity_byte_v20.json", gate)
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "worker", "analyze"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.action == "worker":
        result = worker(args.root, args.index)
    else:
        result = globals()[args.action](args.root)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    if result.get("status") == "failed" or result.get("failures"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
