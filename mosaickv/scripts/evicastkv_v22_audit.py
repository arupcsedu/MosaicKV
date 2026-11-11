#!/usr/bin/env python3
"""Paired old/new four-prefix census audit; reuse, never relabel, V22 failures."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import evicastkv_draft_input_v22 as previous

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
PRODUCER_SOURCE = Path("/scratch/djy8hg/workdir/fox-review-v22-code")
PRODUCER_SHA = "6ea20f889d3e9abf0b736157827bffbd4b0c3d93"
OLDROOT = Path("/scratch/djy8hg/workdir/fox-review-v22/run-20261111-1")
SCORER = Path("mosaickv/src/mosaickv/evaluation/draft_logit_source_scores.py")
UNCHANGED = (
    "mosaickv/src/mosaickv/evaluation/exact_input_replacement.py",
    "mosaickv/src/mosaickv/evaluation/exact_storage.py",
    "mosaickv/src/mosaickv/evaluation/storage_runs.py",
    "mosaickv/src/mosaickv/evaluation/lossless_prompt_cache.py",
    "mosaickv/src/mosaickv/evaluation/projection_cache.py",
    "mosaickv/src/mosaickv/evaluation/lossless_bf16.py",
    "mosaickv/src/mosaickv/evaluation/precision_tier.py",
    "mosaickv/src/mosaickv/evaluation/native_quantized_source_control.py",
    "mosaickv/src/mosaickv/evaluation/lossless_source_control.py",
    "mosaickv/src/mosaickv/evaluation/shared_source_format.py",
    "mosaickv/src/mosaickv/evaluation/shared_source_kernel.py",
    "mosaickv/src/mosaickv/evaluation/dual_projection_kernel.py",
    "mosaickv/src/mosaickv/adapters/huggingface/base.py",
    "mosaickv/src/mosaickv/adapters/huggingface/llava.py",
)
POLICY = dict(
    id="V22_constant_time_saved_tensor_census_audit_v1",
    classification="two_consumed_inputs_counter_instrumentation_only",
    horizon=4,
    model_layers=32,
    initial_cap_bytes=previous.CAP,
    order_by_worker=[["old", "new"], ["new", "old"]],
    score_vectors_and_four_prefix_logits_bitwise_required=True,
    saved_tensor_peak_and_pack_count_exact_required=True,
    corrected_representation_SHA_must_match_original_V22=True,
    new_128_token_decode=False,
    old_quality_producer_SHA=PRODUCER_SHA,
    historical_128_negative_preserved=True,
    counter_timing_not_paper_latency_evidence=True,
    source_family_peak_matching_unsupported=True,
    whole_request_KV_peak_certified=False,
    fresh_identities_or_campaign_authorized=False,
)


def read(path):
    return json.loads(Path(path).read_text())


def _historical(root=OLDROOT):
    root = Path(root)
    old = read(root / "plan.json")
    if old["source_sha"] != PRODUCER_SHA or old["policy_sha256"] != digest(previous.POLICY):
        raise ValueError("original V22 immutable producer/protocol mismatch")
    negative = root / "analysis/GATE_exact_input_v21.json"
    gate = read(negative)
    if not gate["complete"] or gate["failures"] or gate["bounded_numerical_repair_all_inputs_pass"]:
        raise ValueError("complete negative V22 evidence required; never promote it")
    bindings = dict(old["bindings"])
    for path in (root / "plan.json", negative):
        bindings[str(path)] = sha(path)
    for index in range(2):
        directory = root / f"workers/{index}"
        result_path = directory / "result.json"
        result = read(result_path)
        if (
            result["status"] != "completed"
            or result["failures"]
            or result["actual_rows"] != 3
            or result["source_sha"] != PRODUCER_SHA
            or result["plan_sha256"] != sha(root / "plan.json")
        ):
            raise ValueError("original V22 worker incomplete or misattributed")
        bindings[str(result_path)] = sha(result_path)
        for name, expected in result["outputs"].items():
            path = directory / name
            if sha(path) != expected:
                raise ValueError("original V22 worker artifact changed")
            bindings[str(path)] = expected
    return old, bindings


def prepare(root):
    revision = clean_source(SOURCE)
    clean_source(PRODUCER_SOURCE, PRODUCER_SHA)
    old, bindings = _historical()
    for name in UNCHANGED:
        left, right = PRODUCER_SOURCE / name, SOURCE / name
        if sha(left) != sha(right):
            raise ValueError(f"decoder/constructor changed: {name}")
        bindings[str(left)], bindings[str(right)] = sha(left), sha(right)
    old_scorer = PRODUCER_SOURCE / SCORER
    bindings[str(old_scorer)] = sha(old_scorer)
    bindings[str(SOURCE / SCORER)] = sha(SOURCE / SCORER)
    write_once(
        Path(root) / "plan.json",
        dict(
            source_sha=revision,
            producer_source=str(PRODUCER_SOURCE),
            producer_sha=PRODUCER_SHA,
            old_root=str(OLDROOT),
            policy=POLICY,
            policy_sha256=digest(POLICY),
            old_scorer_path=str(old_scorer),
            old_scorer_sha256=sha(old_scorer),
            jobs=old["jobs"],
            bindings=bindings,
            expected_workers=2,
            historical_128_outputs_remain_at_original_producer_SHA=True,
        ),
    )
    return dict(status="prepared", source_sha=revision, expected_workers=2)


def load(root):
    plan = read(Path(root) / "plan.json")
    clean_source(SOURCE, plan["source_sha"])
    clean_source(PRODUCER_SOURCE, PRODUCER_SHA)
    if (
        plan["producer_sha"] != PRODUCER_SHA
        or plan["producer_source"] != str(PRODUCER_SOURCE)
        or plan["old_root"] != str(OLDROOT)
        or plan["policy_sha256"] != digest(POLICY)
        or digest(plan["policy"]) != digest(POLICY)
    ):
        raise ValueError("fixed census audit scope/producer changed")
    for name, expected in plan["bindings"].items():
        if sha(Path(name)) != expected:
            raise ValueError(f"immutable census audit binding changed: {name}")
    return plan


def old_scorer(plan):
    path = Path(plan["old_scorer_path"])
    if path != PRODUCER_SOURCE / SCORER or sha(path) != plan["old_scorer_sha256"]:
        raise ValueError("hash-bound original scoring module required")
    spec = importlib.util.spec_from_file_location("_evicastkv_original_v22_scorer", path)
    if spec is None or spec.loader is None:
        raise ValueError("original scoring module cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.draft_row_scores


def tensor_sha(value):
    import torch

    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def same_bits(left, right):
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and tensor_sha(left) == tensor_sha(right)
    )


def verify_four_logits(saved, reports):
    import torch

    for name in ("old", "new"):
        for kind, field in (("FullKV4", "reference"), ("parent4", "parent")):
            tensor = saved[f"{name}_{kind}"]
            report = reports[name]
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.device.type != "cpu"
                or tensor.dtype != torch.float32
                or tensor.ndim != 2
                or tensor.shape[0] != 4
                or not torch.isfinite(tensor).all().item()
                or list(tensor.shape) != report[f"{field}_logits_shape"]
                or tensor_sha(tensor) != report[f"{field}_logits_sha256"]
            ):
                raise ValueError("saved four-prefix tensor/report digest mismatch")
    if not all(
        same_bits(saved[f"old_{kind}"], saved[f"new_{kind}"]) for kind in ("FullKV4", "parent4")
    ):
        raise ValueError("paired four-prefix logits are not bitwise identical")
    return True


def compare_forecasts(old_scores, new_scores, old_report, new_report, *, layers, length):
    """Compare all signed scores and semantic counters, never timing metadata."""
    import torch

    for scores in (old_scores, new_scores):
        if (
            type(scores) is not dict
            or any(type(k) is not int for k in scores)
            or set(scores) != set(range(layers))
            or any(
                not isinstance(v, torch.Tensor)
                or v.device.type != "cpu"
                or v.dtype != torch.float32
                or v.shape != (length,)
                or not torch.isfinite(v).all().item()
                for v in scores.values()
            )
        ):
            raise ValueError("complete finite signed CPU FP32 vectors required")
    score_equal = all(same_bits(old_scores[i], new_scores[i]) for i in range(layers))
    semantic = (
        "kind",
        "horizon",
        "KL_normalization",
        "decision_margin_normalization",
        "source_shape",
        "reference_logits_sha256",
        "parent_logits_sha256",
        "objective_value",
        "draft_input_token_ids",
        "draft_predicted_token_ids",
        "reference_logits_shape",
        "parent_logits_shape",
        "reference_KL_per_prefix",
        "reference_greedy_margins",
        "parent_reference_token_margins",
        "margin_shortfalls",
        "positive_row_gains",
        "negative_row_gains",
        "zero_row_gains",
        "nonzero_gradient_scalars",
        "prompt_leaf_KV_backing_bytes",
        "autograd_saved_nonweight_peak_backing_bytes",
        "autograd_saved_tensor_observations",
        "no_second_prefill",
        "original_cache_unchanged",
        "model_weights_unchanged",
        "model_parameter_gradients_unchanged",
        "private_cache_only",
        "requires_grad_flags_unchanged",
        "returns_cpu_scores_only",
    )
    mismatches = [name for name in semantic if old_report[name] != new_report[name]]
    required = (
        "no_second_prefill",
        "original_cache_unchanged",
        "model_weights_unchanged",
        "model_parameter_gradients_unchanged",
        "private_cache_only",
        "requires_grad_flags_unchanged",
        "returns_cpu_scores_only",
    )
    valid = (
        old_report["horizon"] == 4
        and old_report["source_shape"][2] == length
        and all(old_report[k] is True for k in required)
    )
    return dict(
        all_signed_score_vectors_bitwise_equal=score_equal,
        old_score_sha256={str(i): tensor_sha(old_scores[i]) for i in range(layers)},
        new_score_sha256={str(i): tensor_sha(new_scores[i]) for i in range(layers)},
        semantic_report_mismatches=mismatches,
        pass_exact_counter_and_forecast_control=bool(score_equal and not mismatches and valid),
        old_total_forecast_seconds=old_report["total_wall_seconds"],
        new_total_forecast_seconds=new_report["total_wall_seconds"],
        diagnostic_only_not_full_request_latency=True,
    )


def worker(root, index):
    import torch
    from evicastkv_campaign import environment
    from evicastkv_request_validation import load_messages

    from mosaickv.adapters.huggingface.registry import load_hf_adapter
    from mosaickv.config import RunConfig
    from mosaickv.evaluation.draft_logit_source_scores import draft_row_scores
    from mosaickv.evaluation.exact_input_replacement import build_replacement_formats
    from mosaickv.evaluation.phase_capacity import PhaseCapture
    from mosaickv.evaluation.request_precision import PromptQueryCapture
    from mosaickv.evaluation.shared_source_format import SourceCapture
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    root, plan = Path(root), load(root)
    if index not in (0, 1) or not os.environ.get("SLURM_JOB_ID", "").isdigit():
        raise ValueError("two registered Slurm workers required")
    out = root / f"workers/{index}"
    out.mkdir(parents=True, exist_ok=False)
    errors, reports, saved, summary = [], {}, {}, None
    stage, started = "setup", time.monotonic()
    job = plan["jobs"][index]
    try:
        if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name():
            raise ValueError("one visible A100 required")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ValueError("deterministic CUBLAS configuration must precede Python")
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(12)
        torch.manual_seed(13)
        previous.parent.fixed.analysis_cpu_controls()
        config = json.loads(json.dumps(job["config"]))
        config["execution"]["seed"] = 13
        cfg = RunConfig.from_mapping(config)
        if (
            cfg.model.id,
            cfg.model.revision,
        ) != previous.parent.fixed.MODEL or cfg.model.precision.value != "bf16":
            raise ValueError("same pinned original BF16 checkpoint required")
        payload = previous.parent.fixed.label_blind_payload(job["inputs"][0])
        write_once(
            out / "manifest.json",
            dict(
                source_sha=plan["source_sha"],
                producer_source_sha=PRODUCER_SHA,
                plan_sha256=sha(root / "plan.json"),
                config_sha256=digest(dict(config=config, policy=POLICY)),
                model=config["model"],
                dataset=config["dataset"],
                sample_id=payload["sample_id"],
                environment=environment(),
                GPU=asdict(capture_gpu_environment(torch)),
                backend="hf",
                attention="eager",
                vllm="not_used",
                sglang="not_used",
                seed=13,
                measurement_type=POLICY["classification"],
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
        if len(pairs) != POLICY["model_layers"]:
            raise ValueError("the pinned original 32-layer model is required")
        args = (
            adapter,
            prepared,
            prefill,
            captured,
            normalized,
            rotary,
            "source3_attention_optional",
        )
        packets, _ = previous.parent.constructed.mixed_formats(
            args, job["witnesses"][str(previous.CAP)], previous.CAP, {}
        )
        original_packet_sha = previous.parent.previous._packet_hash(packets)
        scorers = {"old": old_scorer(plan), "new": draft_row_scores}
        for name in POLICY["order_by_worker"][index]:
            stage = f"four_prefix_{name}"
            scores, report = scorers[name](adapter, prefill, packets, pairs, horizon=4)
            saved[f"{name}_scores"] = scores
            saved[f"{name}_FullKV4"] = torch.tensor(
                report.pop("reference_logits_fp32"), dtype=torch.float32
            )
            saved[f"{name}_parent4"] = torch.tensor(
                report.pop("parent_logits_fp32"), dtype=torch.float32
            )
            reports[name] = report
            write_once(out / f"{name}_draft.json", report)
            if previous.parent.previous._packet_hash(packets) != original_packet_sha:
                raise RuntimeError("counter audit mutated the original packet")
        stage = "exact_comparison_and_construction"
        verify_four_logits(saved, reports)
        comparison = compare_forecasts(
            saved["old_scores"],
            saved["new_scores"],
            reports["old"],
            reports["new"],
            layers=len(pairs),
            length=pairs[0][0].shape[2],
        )
        if not comparison["pass_exact_counter_and_forecast_control"]:
            raise RuntimeError("paired counter/forecast semantics changed")
        text = tuple(
            p
            for span in prepared.modality_map
            if getattr(span.modality, "value", span.modality) == "text"
            for p in range(span.start, span.end)
        )
        _, corrected, construction = build_replacement_formats(
            adapter,
            packets,
            normalized.records,
            pairs,
            captured.windows,
            text,
            cap_bytes=previous.CAP,
            prospective_row_scores=saved["new_scores"],
            ranking_mode="draft_kl_margin_gradient",
        )
        packet_sha = previous.parent.previous._packet_hash(corrected)
        old_dir = OLDROOT / f"workers/{index}"
        old_rows = [json.loads(line) for line in (old_dir / "raw.jsonl").read_text().splitlines()]
        old_row = next(r for r in old_rows if r["arm"] == previous.ARMS[2])
        if old_row["source_sha"] != PRODUCER_SHA or packet_sha != old_row["representation_sha256"]:
            raise RuntimeError("new corrected packet differs from original V22 representation")
        # Historical long-horizon tensors enter only AFTER score construction.
        historical = torch.load(old_dir / "logits.pt", weights_only=True, map_location="cpu")
        for name in ("old", "new"):
            previous.verify_draft(
                reports[name],
                historical["FullKV_native"],
                historical["V20_native_quantized_parent"],
            )
            for current, old in (
                (f"{name}_FullKV4", "prospective_FullKV4"),
                (f"{name}_parent4", "prospective_native_parent4"),
            ):
                if not same_bits(saved[current], historical[old]):
                    raise RuntimeError("paired four-prefix logits differ from historical V22")
        summary = dict(
            status="completed",
            worker=index,
            sample_id=payload["sample_id"],
            task=payload["task"],
            source_sha=plan["source_sha"],
            quality_producer_sha=PRODUCER_SHA,
            comparison=comparison,
            representation_sha256=packet_sha,
            original_representation_sha256=old_row["representation_sha256"],
            historical_128_predictions_reusable=True,
            new_128_decode_performed=False,
            old_quality_guard32=old_row["FullKV_quality"]["registered_guard_32"][
                "registered_guard_pass"
            ],
            old_quality_guard128=old_row["FullKV_quality"]["prefix_128"]["registered_guard_pass"],
            source_family_peak_matching_unsupported=True,
            whole_request_KV_peak_certified=False,
            paper_or_superiority_claim=False,
        )
        write_once(out / "construction.json", construction)
    except BaseException as exc:
        errors.append(
            dict(
                status="failed",
                stage=stage,
                worker=index,
                sample_id=job["sample_id"],
                error_type=type(exc).__name__,
                error=str(exc),
                traceback=traceback.format_exc(),
            )
        )
    finally:
        if saved:
            torch.save(saved, out / "forecast_tensors.pt")
        with (out / "raw.jsonl").open("x") as stream:
            for item in ([summary] if summary else []) + errors:
                stream.write(json.dumps(item, sort_keys=True, allow_nan=False) + "\n")
        outputs = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()}
        result = dict(
            status="completed" if summary and not errors else "failed",
            failures=errors,
            source_sha=plan["source_sha"],
            plan_sha256=sha(root / "plan.json"),
            worker=index,
            outputs=outputs,
            order=POLICY["order_by_worker"][index],
            wall_seconds=time.monotonic() - started,
            SLURM_JOB_ID=os.environ.get("SLURM_JOB_ID"),
            SLURM_ARRAY_JOB_ID=os.environ.get("SLURM_ARRAY_JOB_ID"),
            SLURM_ARRAY_TASK_ID=os.environ.get("SLURM_ARRAY_TASK_ID"),
        )
        write_once(out / "result.json", result)
    return result


def regenerate_historical_analysis():
    """Separate CPU derivation; original producer SHA and wrong-name artifact survive."""
    return previous.analyze(
        OLDROOT, producer_source=PRODUCER_SOURCE, output_dir=OLDROOT / "analysis-v22-corrected"
    )


def analyze(root):
    import torch

    root = Path(root)
    previous.parent.fixed.analysis_cpu_controls()
    observations, failures = [], []
    if (root / "plan.json").is_file():
        plan = load(root)
        for index in range(2):
            try:
                out = root / f"workers/{index}"
                result = read(out / "result.json")
                if (
                    result["status"] != "completed"
                    or result["failures"]
                    or result["worker"] != index
                ):
                    raise ValueError("paired audit worker incomplete")
                if result["source_sha"] != plan["source_sha"] or result["plan_sha256"] != sha(
                    root / "plan.json"
                ):
                    raise ValueError("paired audit producer/plan mismatch")
                for name, expected in result["outputs"].items():
                    if sha(out / name) != expected:
                        raise ValueError("paired audit output hash changed")
                saved = torch.load(
                    out / "forecast_tensors.pt", weights_only=True, map_location="cpu"
                )
                a, b = read(out / "old_draft.json"), read(out / "new_draft.json")
                verify_four_logits(saved, {"old": a, "new": b})
                compared = compare_forecasts(
                    saved["old_scores"],
                    saved["new_scores"],
                    a,
                    b,
                    layers=POLICY["model_layers"],
                    length=a["source_shape"][2],
                )
                (summary,) = [
                    json.loads(line) for line in (out / "raw.jsonl").read_text().splitlines()
                ]
                if (
                    compared != summary["comparison"]
                    or not compared["pass_exact_counter_and_forecast_control"]
                ):
                    raise ValueError("saved paired score/census recomputation failed")
                old_rows = [
                    json.loads(line)
                    for line in (OLDROOT / f"workers/{index}/raw.jsonl").read_text().splitlines()
                ]
                old_row = next(r for r in old_rows if r["arm"] == previous.ARMS[2])
                if (
                    summary["representation_sha256"] != old_row["representation_sha256"]
                    or summary["quality_producer_sha"] != PRODUCER_SHA
                ):
                    raise ValueError(
                        "historical quality reuse is not representation/producer-bound"
                    )
                observations.append(summary)
            except BaseException as exc:
                failures.append(dict(worker=index, error_type=type(exc).__name__, error=str(exc)))
    else:
        failures.append(dict(error="preparation failed; no immutable audit plan"))
    corrected = None
    if all((root / f"workers/{index}/result.json").is_file() for index in range(2)):
        try:
            corrected = regenerate_historical_analysis()
            if (
                not corrected.get("complete")
                or corrected.get("failures")
                or corrected.get("producer_source_sha") != PRODUCER_SHA
                or corrected.get("analysis_source_sha") != plan["source_sha"]
                or corrected.get("bounded_numerical_repair_all_inputs_pass") is not False
            ):
                raise ValueError(
                    "derived historical analysis must preserve the original negative producer"
                )
        except BaseException as exc:
            failures.append(
                dict(stage="CPU_historical_analysis", error_type=type(exc).__name__, error=str(exc))
            )
    gate = dict(
        complete=not failures and len(observations) == 2,
        failures=failures,
        observations=observations,
        paired_counter_optimization_verified=not failures and len(observations) == 2,
        historical_quality_analysis=corrected,
        historical_quality_producer_sha=PRODUCER_SHA,
        historical_negative_128_result_preserved=True,
        new_128_decode_performed=False,
        source_family_peak_matching_unsupported=True,
        whole_request_KV_peak_certified=False,
        paper_or_fresh_campaign_authorized=False,
    )
    write_once(root / "analysis/GATE_v22_audit.json", gate)
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
