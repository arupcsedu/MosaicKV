#!/usr/bin/env python3
"""One fixed, consumed-input numerical repair; never a quality/budget sweep."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import evicastkv_fidelity_byte_v20 as parent

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
V20 = Path("/scratch/djy8hg/workdir/fox-review-v20/run-20261111-1")
V21 = Path("/scratch/djy8hg/workdir/fox-review-v21/run-20261111-2")
CAP = 51380224
ARMS = (
    "native_quantized_parent",
    "native_exact_text_replacement",
    "native_exact_text_plus_draft_logit_input_rows",
)
POLICY = dict(
    id="V22_four_draft_logit_gradient_input_repair_v1",
    source="V20_same_two_consumed_inputs_and_frozen49MiB_packet",
    maximum_initial_bytes=CAP,
    initial_byte_equality_claim=False,
    source_codes="unchanged_three_bit_codes_including_overlaid_rows_still_charged",
    text="lossless_native_parent_replacement_only_after_all_KV_bits_verified",
    additional_rows="positive_signed_frozen_model_draft_KL_margin_gain",
    allocation="actual_freed_owned_bytes_only_no_new_allowance",
    optional_grid_or_tuning=False,
    constructor_access="prefill_KV_inputs_and_signed_four_draft_scores_no_raw_logits_or_labels",
    prospective_scoring="four_FullKV_draft_steps_then_one_private_source_KV_backward",
    horizon=4,
    gradient_loss="sum_KL_FullKV_source_divided_by_0.02_plus_relu_reference_margin_minus_source_margin",
    predictive_gain="minus_source_KV_gradient_dot_original_minus_native_source_KV_per_row",
    training_or_weight_updates=False,
    oracle_128_prefix_access_during_selection=False,
    output_predictions=128,
    continuation_updates=127,
    history="common_raw_BF16_append_only128_allocated_slots",
    guard=parent.fixed.POLICY["guard"],
    fullkv_reference="V20_actual_native_FreeGreedy128_evaluation_only",
    parent_reference="V20_native_quantized_reader49MiB_not_fused_reader",
    whole_request_KV_peak_certified=False,
    paper_or_fresh_campaign_authorized=False,
    superiority_claim=False,
    source_family_peak_matching_unsupported=True,
    seed=13,
    workers=2,
    soft_deadline_seconds=1500,
    disabled_mechanisms=parent.POLICY["disabled_mechanisms"],
)


def read(path):
    return json.loads(Path(path).read_text())


def prepare(root):
    root = Path(root)
    revision = clean_source(SOURCE)
    old = read(V20 / "plan.json")
    gate = read(V20 / "analysis/GATE_fidelity_byte_v20.json")
    if (
        not gate["complete"]
        or gate["failures"]
        or not gate["original_exact_selected_transport_and_common_byte_engineering_gate_pass"]
        or gate["plan_sha256"] != sha(V20 / "plan.json")
    ):
        raise ValueError("complete V20 numerical/storage engineering gate required")
    bindings = dict(old["bindings"])
    negative = read(V21 / "analysis/GATE_exact_input_v21.json")
    if (
        not negative["complete"]
        or negative["failures"]
        or negative["bounded_numerical_repair_all_inputs_pass"]
    ):
        raise ValueError("complete negative V21 diagnostic is required, not ignored")
    bindings[str(V21 / "analysis/GATE_exact_input_v21.json")] = sha(
        V21 / "analysis/GATE_exact_input_v21.json"
    )
    for name in ("plan.json", "analysis/GATE_fidelity_byte_v20.json"):
        bindings[str(V20 / name)] = sha(V20 / name)
    for i in range(2):
        directory = V20 / f"workers/{i}"
        result = read(directory / "result.json")
        if result["status"] != "completed" or result["failures"]:
            raise ValueError("V20 parent worker incomplete")
        bindings[str(directory / "result.json")] = sha(directory / "result.json")
        for name, expected in result["outputs"].items():
            if sha(directory / name) != expected:
                raise ValueError("V20 bound worker output changed")
            bindings[str(directory / name)] = expected
    write_once(
        root / "plan.json",
        dict(
            source_sha=revision,
            policy=POLICY,
            policy_sha256=digest(POLICY),
            jobs=old["jobs"],
            bindings=bindings,
            expected_workers=2,
            expected_rows_per_worker=3,
            new_identities_accessed=False,
        ),
    )
    return dict(status="prepared", source_sha=revision, expected_workers=2)


def load(root):
    plan = read(Path(root) / "plan.json")
    clean_source(SOURCE, plan["source_sha"])
    if plan["policy_sha256"] != digest(POLICY) or digest(plan["policy"]) != digest(POLICY):
        raise ValueError("frozen V22 policy changed")
    for path, expected in plan["bindings"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"immutable diagnostic binding changed: {path}")
    return plan


def verify_draft(report, reference, native_parent):
    def digest_logits(value):
        return hashlib.sha256(value[1:5].contiguous().numpy().tobytes()).hexdigest()

    if report["horizon"] != 4:
        raise ValueError("only four prospective draft steps were registered")
    for flag in (
        "no_second_prefill",
        "original_cache_unchanged",
        "model_weights_unchanged",
        "model_parameter_gradients_unchanged",
        "private_cache_only",
        "returns_cpu_scores_only",
    ):
        if report[flag] is not True:
            raise ValueError(f"prospective draft isolation failed: {flag}")
    if report["reference_logits_sha256"] != digest_logits(reference):
        raise ValueError("prospective FullKV draft differs from saved native inference")
    if report["parent_logits_sha256"] != digest_logits(native_parent):
        raise ValueError("gradient-mode native parent differs from ordinary native inference")
    return True


def outcome(rows):
    if len(rows) != 3 or tuple(r["arm"] for r in rows) != ARMS:
        raise ValueError("exactly the three registered diagnostic arms required")

    def valid_storage(row):
        storage = row["measured"]["storage"]
        initial = storage["initial_active_kv_storage_bytes"]
        final = storage["final_active_kv_storage_bytes"]
        peak = storage["peak_active_kv_storage_bytes"]
        return (
            all(type(v) is int and v > 0 for v in (initial, final, peak))
            and initial <= CAP
            and peak >= max(initial, final)
            and bool(storage["history_updates_per_layer"])
            and all(v == 127 for v in storage["history_updates_per_layer"])
            and storage["outstanding_layer_leases"] == 0
            and row["measured"]["used_fullkv_fallback"] is False
        )

    cap = all(valid_storage(row) for row in rows)
    text_exact = rows[1]["parent_transport_parity"]["pass_all128_bitwise"]
    original_guard = rows[2]["FullKV_quality"]["registered_guard_32"]["registered_guard_pass"]
    later = rows[2]["FullKV_quality"]["prefix_128"]["registered_guard_pass"]
    meaningful = rows[2]["added_original_input_rows"] > 0
    return dict(
        construction_cap_pass=cap,
        lossless_text_replacement_all128_logits_bitwise=text_exact,
        meaningful_original_row_restoration=meaningful,
        repaired_FullKV32_guard_pass=original_guard,
        repaired_FullKV128_guard_pass=later,
        pass_bounded_numerical_repair=bool(
            cap and text_exact and original_guard and later and meaningful
        ),
        byte_matched_comparison_claim=False,
        whole_request_KV_peak_certified=False,
        benchmark_quality_or_superiority_claim=False,
        independent_confirmation_or_campaign_authorized=False,
    )


def worker(root, index):
    import torch
    from evicastkv_campaign import environment
    from evicastkv_request_validation import load_messages

    from mosaickv.adapters.huggingface.registry import load_hf_adapter
    from mosaickv.config import RunConfig
    from mosaickv.evaluation.draft_logit_source_scores import draft_row_scores
    from mosaickv.evaluation.exact_input_replacement import build_replacement_formats
    from mosaickv.evaluation.native_quantized_source_control import NativeQuantizedSourceControl
    from mosaickv.evaluation.phase_capacity import PhaseCapture
    from mosaickv.evaluation.request_precision import PromptQueryCapture
    from mosaickv.evaluation.shared_source_format import SourceCapture
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    root = Path(root)
    plan = load(root)
    if index not in (0, 1) or not os.environ.get("SLURM_JOB_ID", "").isdigit():
        raise ValueError("two registered Slurm workers required")
    out = root / f"workers/{index}"
    out.mkdir(parents=True, exist_ok=False)
    rows, errors, tensors = [], [], {}
    start = time.monotonic()
    try:
        if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name():
            raise ValueError("one deterministic A100 required")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ValueError("CUBLAS config must precede Python")
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(12)
        torch.manual_seed(13)
        parent.fixed.analysis_cpu_controls()
        job = plan["jobs"][index]
        config = json.loads(json.dumps(job["config"]))
        config["execution"]["seed"] = 13
        cfg = RunConfig.from_mapping(config)
        if (
            cfg.model.id,
            cfg.model.revision,
        ) != parent.fixed.MODEL or cfg.model.precision.value != "bf16":
            raise ValueError("same pinned BF16 LLaVA required")
        payload = parent.fixed.label_blind_payload(job["inputs"][0])
        write_once(
            out / "manifest.json",
            dict(
                source_sha=plan["source_sha"],
                source_dirty=False,
                plan_sha256=sha(root / "plan.json"),
                config_sha256=digest(dict(config=config, policy=POLICY)),
                dataset=config["dataset"],
                model=config["model"],
                environment=environment(),
                GPU=asdict(capture_gpu_environment(torch)),
                seed=13,
                backend="hf",
                attention="eager",
                measurement_type="same_consumed_input_single_preregistered_numerical_repair",
                vllm="not_used",
                sglang="not_used",
                benchmark_answers_read=False,
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
            PromptQueryCapture(adapter, summaries=True, normalized=False, prior=False) as captured,
            SourceCapture(adapter) as normalized,
            PhaseCapture(adapter) as rotary,
        ):
            prefill = adapter.prefill(prepared, capture_queries=False, capture_attentions=False)
        pairs, _ = adapter._legacy_layers(prefill.state.past_key_values)
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
        original, _ = parent.constructed.mixed_formats(args, job["witnesses"][str(CAP)], CAP, proof)
        text = tuple(
            p
            for span in prepared.modality_map
            if getattr(span.modality, "value", span.modality) == "text"
            for p in range(span.start, span.end)
        )
        # Generate only four forecast steps from this prefill. Saved 128-prefix
        # tensors remain inaccessible until after construction below.
        original_packet_hash = parent.previous._packet_hash(original)
        row_scores, forecast = draft_row_scores(adapter, prefill, original, pairs, horizon=4)
        # Retain the tiny diagnostic horizon even if a later hash/constructor
        # gate fails; numerical post-processing must not require a GPU rerun.
        tensors["prospective_FullKV4"] = torch.tensor(
            forecast.pop("reference_logits_fp32"), dtype=torch.float32
        )
        tensors["prospective_native_parent4"] = torch.tensor(
            forecast.pop("parent_logits_fp32"), dtype=torch.float32
        )
        if parent.previous._packet_hash(original) != original_packet_hash:
            raise RuntimeError("prospective derivative changed the immutable source packet")
        write_once(out / "draft.json", forecast)
        constructor_start = time.monotonic()
        text_formats, corrected_formats, construction = build_replacement_formats(
            adapter,
            original,
            normalized.records,
            pairs,
            captured.windows,
            text,
            cap_bytes=CAP,
            prospective_row_scores=row_scores,
            ranking_mode="draft_kl_margin_gradient",
        )
        torch.cuda.synchronize()
        construction["wall_seconds"] = time.monotonic() - constructor_start
        write_once(out / "construction.json", construction)
        references = torch.load(
            V20 / f"workers/{index}/logits.pt", weights_only=True, map_location="cpu"
        )
        reference = references["FullKV_native_greedy"]
        baseline = references[f"source3_native_quantized_reader_raw_history__{CAP}"]
        tensors.update(FullKV_native=reference, V20_native_quantized_parent=baseline)
        verify_draft(forecast, reference, baseline)
        if not torch.equal(prefill.logits.reshape(-1).float().cpu(), reference[0]):
            raise ValueError("fresh diagnostic prefill changed from immutable V20 FullKV")
        prefix = parent.fixed.forced_prefix(reference)
        for arm, formats in zip(ARMS, (original, text_formats, corrected_formats), strict=True):
            if time.monotonic() - start > POLICY["soft_deadline_seconds"]:
                raise TimeoutError("bounded V22 soft deadline")
            scope = (
                NativeQuantizedSourceControl(
                    original, [b.self_attn for b in adapter._language_layers()]
                )
                if arm == ARMS[0]
                else None
            )
            packet_hash = parent.previous._packet_hash(formats)
            logits, measured = parent._raw_run(adapter, prefill, formats, prefix, CAP, scope)
            if packet_hash != parent.previous._packet_hash(formats):
                raise RuntimeError("immutable source/row packet changed during decoding")
            row = dict(
                arm=arm,
                status="completed",
                sample_id=payload["sample_id"],
                task=payload["task"],
                source_sha=plan["source_sha"],
                plan_sha256=sha(root / "plan.json"),
                measured=measured,
                parent_transport_parity=parent.parity(baseline, logits),
                FullKV_quality=parent.fixed.comparisons(reference, logits),
                added_original_input_rows=(
                    construction["additional_input_rows"] if arm == ARMS[2] else 0
                ),
                cap_is_initial_not_lifetime=True,
                byte_matched_comparison_claim=False,
                verifier_original_prefill_and_all_candidate_packet_owners_retained=True,
                verifier_storage_excluded_from_decode_cache_boundary=True,
                whole_request_KV_peak_certified=False,
                disabled_mechanisms=POLICY["disabled_mechanisms"],
                representation_sha256=packet_hash,
                immutable_representation_unchanged=True,
                draft_forecast=forecast,
                forecast_overhead_excluded_from_decode_timer_but_reported=True,
            )
            if arm == ARMS[0] and not row["parent_transport_parity"]["pass_all128_bitwise"]:
                raise RuntimeError("V20 native quantized parent replay changed")
            tensors[arm] = logits
            rows.append(row)
            write_once(out / "traces" / f"{arm}.json", row)
            gc.collect()
    except BaseException as exc:
        errors.append(
            dict(
                error_type=type(exc).__name__,
                error=str(exc),
                traceback=traceback.format_exc(),
                structured_failure=getattr(exc, "report", None),
                worker_index=index,
                sample_id=plan["jobs"][index]["sample_id"],
                stage_or_arm=locals().get("arm", "construction_or_setup"),
            )
        )
    finally:
        if tensors:
            torch.save(tensors, out / "logits.pt")
        with (out / "raw.jsonl").open("x") as stream:
            for item in rows + [dict(status="failed", **e) for e in errors]:
                stream.write(json.dumps(item, sort_keys=True, allow_nan=False) + "\n")
        outputs = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()}
        result = dict(
            status="completed" if not errors and len(rows) == 3 else "failed",
            source_sha=plan["source_sha"],
            plan_sha256=sha(root / "plan.json"),
            failures=errors,
            actual_rows=len(rows),
            outputs=outputs,
            wall_seconds=time.monotonic() - start,
            SLURM_JOB_ID=os.environ.get("SLURM_JOB_ID"),
            SLURM_ARRAY_JOB_ID=os.environ.get("SLURM_ARRAY_JOB_ID"),
            SLURM_ARRAY_TASK_ID=os.environ.get("SLURM_ARRAY_TASK_ID"),
        )
        write_once(out / "result.json", result)
    return result


def analyze(root):
    root = Path(root)
    observations, failures = [], []
    if (root / "plan.json").is_file():
        import torch

        plan = load(root)
        parent.fixed.analysis_cpu_controls()
        for index in range(2):
            try:
                out = root / f"workers/{index}"
                result = read(out / "result.json")
                if (
                    result["status"] != "completed"
                    or result["failures"]
                    or result["actual_rows"] != 3
                ):
                    raise ValueError(f"incomplete V22 worker: {result}")
                if result["source_sha"] != plan["source_sha"] or result["plan_sha256"] != sha(
                    root / "plan.json"
                ):
                    raise ValueError("worker immutable provenance changed")
                for path, expected in result["outputs"].items():
                    if sha(out / path) != expected:
                        raise ValueError("saved output hash changed")
                tensors = torch.load(out / "logits.pt", weights_only=True, map_location="cpu")
                verify_draft(
                    read(out / "draft.json"),
                    tensors["FullKV_native"],
                    tensors["V20_native_quantized_parent"],
                )
                rows = [json.loads(line) for line in (out / "raw.jsonl").read_text().splitlines()]
                for row in rows:
                    parent.fixed.verify_recomputed_diagnostics(
                        parent.fixed.comparisons(tensors["FullKV_native"], tensors[row["arm"]]),
                        row["FullKV_quality"],
                    )
                    if (
                        parent.parity(tensors["V20_native_quantized_parent"], tensors[row["arm"]])
                        != row["parent_transport_parity"]
                    ):
                        raise ValueError("bitwise parent replay diagnostic changed")
                observations.append(
                    dict(
                        worker=index,
                        gates=outcome(rows),
                        rows=rows,
                        construction=read(out / "construction.json"),
                    )
                )
            except BaseException as exc:
                failures.append(dict(worker=index, error_type=type(exc).__name__, error=str(exc)))
    else:
        failures.append(dict(error="preparation failed; no immutable plan"))
    gate = dict(
        complete=not failures and len(observations) == 2,
        observations=observations,
        failures=failures,
        bounded_numerical_repair_all_inputs_pass=not failures
        and len(observations) == 2
        and all(o["gates"]["pass_bounded_numerical_repair"] for o in observations),
        calibration_or_paper_campaign_authorized=False,
        whole_request_KV_peak_certified=False,
        negative_V20_results_unchanged=True,
        no_fresh_identities=True,
    )
    write_once(root / "analysis/GATE_exact_input_v21.json", gate)
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "worker", "analyze"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    result = (
        worker(args.root, args.index)
        if args.action == "worker"
        else globals()[args.action](args.root)
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    if result.get("status") == "failed" or result.get("failures"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
