#!/usr/bin/env python3
"""Analyze the frozen 128-sample output-preserving development validation."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from dataclasses import asdict, replace
from pathlib import Path

from analyze_output_preserving_capacity_sweep import worker
from run_prefixkv_decode_preflight import digest, write_once

from mosaickv.evaluation.coco_shared_bytes import TokenizedCider, bootstrap_cider_curve
from mosaickv.evaluation.metrics.registry import default_metric_registry
from mosaickv.evaluation.metrics.types import MetricSample

STORAGE_FIELDS = (
    "initial_active_kv_storage_bytes",
    "final_active_kv_storage_bytes",
    "peak_active_kv_storage_bytes",
)
TIMING_FIELDS = ("prefill_time", "compression_time", "decode_time", "end_to_end_time")
MANIFEST_PATHS = (
    ("dataset", "id"),
    ("dataset", "revision"),
    ("model", "id"),
    ("model", "revision"),
    ("model", "precision"),
    ("execution", "backend"),
    ("execution", "attention_implementation"),
    ("execution", "seed"),
    ("environment", "detected_packages", "torch"),
    ("environment", "detected_packages", "transformers"),
    ("environment", "detected_packages", "vllm"),
    ("environment", "detected_packages", "sglang"),
    ("hardware", "gpu_type"),
    ("hardware", "gpu_count"),
    ("evaluation", "reference_output_sha256"),
    ("measurement_type",),
)


def _nested(payload: dict[str, object], path: tuple[str, ...]) -> object:
    value: object = payload
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"manifest is missing {'.'.join(path)}")
        value = value[key]
    if value is None or value == "":
        raise ValueError(f"manifest has empty {'.'.join(path)}")
    return value


def analyze(args: argparse.Namespace) -> int:
    import pandas as pd

    root = args.root.resolve()
    if digest(root / "plan.json") != args.plan_sha256:
        raise ValueError("plan hash mismatch")
    plan = json.loads((root / "plan.json").read_text())
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "status": "failed",
        "classification": plan["classification"],
        "heldout_claim": False,
        "paper_claim_authorized": False,
        "paper_campaign_submission_authorized": False,
        "plan_sha256": args.plan_sha256,
        "missing_or_failed": [],
        "gates": {},
    }
    try:
        ledger = json.loads(Path(plan["ledger"]).read_text())
        refs = {sample["sample_id"]: tuple(sample["references"]) for sample in ledger["samples"]}
        ids = tuple(plan["sample_ids"])
        rows: dict[tuple[str, str], dict[str, object]] = {}
        traces: dict[tuple[str, str], dict[str, object]] = {}
        manifests: dict[str, dict[str, object]] = {}
        missing = report["missing_or_failed"]
        assert isinstance(missing, list)
        for job in plan["jobs"]:
            try:
                result = worker(Path(job["output"]))
            except ValueError as error:
                missing.append({"index": job["index"], "error": str(error)})
                continue
            if result.get("classification") != plan["classification"]:
                raise ValueError("worker classification differs from plan")
            label = job["analysis_label"]
            measured = result["record"]["rows"]
            if len(measured) != len(ids):
                raise ValueError(f"{label} has {len(measured)} rows instead of {len(ids)}")
            for row in measured:
                sid = row["sample_id"]
                key = label, sid
                if key in rows:
                    raise ValueError(f"duplicate result row: {key}")
                if sid not in refs or row.get("status") != "completed" or row.get("error"):
                    raise ValueError(f"failed or unregistered row: {key}")
                if not 0 < row.get("generated_tokens", 0) <= plan["output_tokens"]:
                    raise ValueError(f"invalid generated-token count: {key}")
                for field in TIMING_FIELDS:
                    value = row.get(field)
                    if not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                        raise ValueError(f"invalid {field}: {key}")
                manifest_path = Path(row["manifest_path"])
                if not manifest_path.is_file():
                    raise ValueError(f"missing manifest: {manifest_path}")
                manifests.setdefault(label, json.loads(manifest_path.read_text()))
                rows[key] = dict(row, label=label, cap_mib=job["cap_mib"])
                trace_path = Path(result["traces"][sid])
                if not trace_path.is_file():
                    raise ValueError(f"missing trace: {key}")
                traces[key] = json.loads(trace_path.read_text())
        if missing:
            raise ValueError("incomplete validation; metrics withheld")
        if set(rows) != {(label, sid) for label in manifests for sid in ids}:
            raise ValueError("methods do not contain identical sample identities")
        for manifest in manifests.values():
            for path in MANIFEST_PATHS:
                _nested(manifest, path)
            if manifest["dataset"]["revision"] != plan["dataset_revision"]:
                raise ValueError("dataset revision differs from plan")
            if manifest["model"]["revision"] != plan["model_revision"]:
                raise ValueError("model revision differs from plan")

        candidate = str(plan.get("candidate_analysis_label", "output_preserving_draft_gate"))
        prompt25 = "prompt_attention_topk_25p5"
        prompt40 = "prompt_attention_topk_40p0"
        expected_labels = {"full_kv", prompt25, prompt40, candidate}
        if set(manifests) != expected_labels:
            raise ValueError(
                f"analysis labels differ from the registered four arms: {set(manifests)}"
            )

        cap25 = round(25.5 * 2**20)
        cap40 = round(40.0 * 2**20)
        byte_records = []
        memory_pass = True
        same_budget_pass = True
        logit_trace_pass = True
        fail_closed_count = 0
        for sid in ids:
            if rows[candidate, sid]["initial_active_kv_storage_bytes"] != cap25:
                raise ValueError(f"candidate missed the exact initial cap: {sid}")
            if rows[prompt25, sid]["initial_active_kv_storage_bytes"] != cap25:
                raise ValueError(f"25.5 MiB Prompt Top-K missed the exact initial cap: {sid}")
            if rows[prompt40, sid]["initial_active_kv_storage_bytes"] != cap40:
                raise ValueError(f"40 MiB Prompt Top-K missed the exact initial cap: {sid}")
            for field in STORAGE_FIELDS:
                c = int(rows[candidate, sid][field])
                p25 = int(rows[prompt25, sid][field])
                p40 = int(rows[prompt40, sid][field])
                same_budget_pass &= c == p25
                reduction = 1 - c / p40
                memory_pass &= reduction >= 0.30
                byte_records.append(
                    {
                        "sample_id": sid,
                        "boundary": field.removesuffix("_active_kv_storage_bytes"),
                        "candidate_bytes": c,
                        "same_budget_prompt_bytes": p25,
                        "reference_prompt_bytes": p40,
                        "reduction_vs_40_mib": reduction,
                    }
                )
            trace = traces[candidate, sid]
            reconstruction = trace.get("output_reconstruction", {})
            expected_method = plan["fixed_candidate"]["effective_method"]
            expected_ranking = plan["fixed_candidate"].get("draft_ranking_mode", "veto")
            if (
                trace.get("effective_method") != expected_method
                or trace.get("forecast_statistics", {}).get("draft_cache_isolated") is not True
                or reconstruction.get("draft_validation") is not True
                or reconstruction.get("draft_ranking_mode") != expected_ranking
                or trace.get("residual_kv_bytes") != 0
            ):
                raise ValueError(f"candidate trace contract failed: {sid}")
            if expected_ranking == "joint_normalized":
                swaps = [
                    swap
                    for route in reconstruction.get("routes", ())
                    for swap in route.get("swaps", ())
                ]
                if any(
                    not isinstance(swap.get("joint_normalized_gain"), (float, int))
                    or swap["joint_normalized_gain"] <= 0
                    for swap in swaps
                ):
                    raise ValueError(f"nonpositive joint swap in candidate trace: {sid}")
            if plan["fixed_candidate"].get("short_horizon_logit_reranking") is True:
                stability = trace.get("short_horizon_logit_stability", {})
                evaluations = stability.get("evaluated_candidates", ())
                strategy = plan["fixed_candidate"].get("logit_candidate_strategy", "weight_grid")
                if strategy in {
                    "route_exchange_beam",
                    "layer_capacity_beam",
                    "layer_capacity_surrogate_beam",
                    "layer_capacity_vectorized_surrogate_beam",
                    "layer_capacity_directional_single_beam",
                    "within_route_position_beam",
                }:
                    candidate_count_valid = (
                        len(evaluations) == (4 if strategy == "within_route_position_beam" else 2)
                        if strategy
                        in {
                            "layer_capacity_surrogate_beam",
                            "layer_capacity_vectorized_surrogate_beam",
                            "layer_capacity_directional_single_beam",
                            "within_route_position_beam",
                        }
                        else (
                            (8 if strategy == "layer_capacity_beam" else 1)
                            <= len(evaluations)
                            <= plan["fixed_candidate"]["logit_route_candidate_limit"]
                        )
                    )
                    strategy_trace_valid = bool(
                        stability.get("candidate_strategy") == strategy
                        and stability.get("batched_probe_execution") is True
                        and isinstance(stability.get("probe_batch_count"), int)
                        and stability["probe_batch_count"] >= 1
                    )
                    if strategy == "layer_capacity_beam":
                        strategy_trace_valid &= bool(
                            stability.get("extended_horizon")
                            == plan["fixed_candidate"]["logit_extended_horizon"]
                            and 2 <= len(stability.get("extended_candidate_ids", ())) <= 4
                            and stability.get("layer_capacity_candidate_count")
                            == plan["fixed_candidate"]["logit_layer_transfer_limit"]
                        )
                    elif strategy in {
                        "layer_capacity_surrogate_beam",
                        "layer_capacity_vectorized_surrogate_beam",
                        "layer_capacity_directional_single_beam",
                        "within_route_position_beam",
                    }:
                        timings = stability.get("probe_phase_timings", ())
                        timing_stages = {
                            item.get("stage") for item in timings if isinstance(item, dict)
                        }
                        strategy_trace_valid &= bool(
                            stability.get("extended_horizon")
                            == plan["fixed_candidate"]["logit_extended_horizon"]
                            and len(stability.get("extended_candidate_ids", ()))
                            <= (2 if strategy == "within_route_position_beam" else 1)
                            and (
                                stability.get("single_route_candidate_count")
                                == plan["fixed_candidate"]["logit_route_candidate_limit"]
                                if strategy == "within_route_position_beam"
                                else stability.get("layer_capacity_candidate_count")
                                == plan["fixed_candidate"]["logit_layer_transfer_limit"]
                            )
                            and stability.get("surrogate_candidate_count")
                            == (
                                plan["fixed_candidate"]["logit_route_candidate_limit"]
                                if strategy == "within_route_position_beam"
                                else plan["fixed_candidate"]["logit_layer_transfer_limit"]
                            )
                            and stability.get("model_probed_candidate_count")
                            == (4 if strategy == "within_route_position_beam" else 2)
                            and "surrogate_generation" in timing_stages
                            and (
                                "short_four_candidate_probe"
                                if strategy == "within_route_position_beam"
                                else "short_two_candidate_probe"
                            )
                            in timing_stages
                        )
                else:
                    candidate_count_valid = 2 <= len(evaluations) <= 4
                    strategy_trace_valid = True
                fail_closed_count += int(stability.get("used_fail_closed_scalar_anchor") is True)
                logit_trace_pass &= bool(
                    stability.get("status") == "completed_label_blind"
                    and stability.get("benchmark_labels_accessed") is False
                    and stability.get("fullkv_prefill_reused") is True
                    and stability.get("second_prefill_performed") is False
                    and stability.get("probe_states_discarded_before_measured_decode") is True
                    and stability.get("horizon")
                    == plan["fixed_candidate"]["logit_stability_horizon"]
                    and candidate_count_valid
                    and strategy_trace_valid
                    and reconstruction.get("selected_candidate_id")
                    == stability.get("selected_candidate_id")
                )
        if not same_budget_pass:
            raise ValueError("candidate and Prompt Top-K do not have identical 25.5 MiB storage")

        groups = {
            label: [MetricSample(sid, rows[label, sid]["answer"], refs[sid]) for sid in ids]
            for label in ("full_kv", prompt25, prompt40, candidate)
        }
        definition = replace(
            default_metric_registry().resolve("cider"),
            reducer=TokenizedCider([sample for group in groups.values() for sample in group]),
        )
        aggregates = []
        for label, group in groups.items():
            measured = [rows[label, sid] for sid in ids]
            aggregates.append(
                {
                    "label": label,
                    "sample_count": len(ids),
                    "cider": definition.aggregate(group),
                    **{
                        field: statistics.mean(float(row[field]) for row in measured)
                        for field in STORAGE_FIELDS
                    },
                    **{
                        field: statistics.mean(float(row[field]) for row in measured)
                        for field in TIMING_FIELDS
                    },
                }
            )
        comparisons = (
            (prompt25, candidate),
            (prompt40, candidate),
            ("full_kv", candidate),
            ("full_kv", prompt25),
            ("full_kv", prompt40),
        )
        paired = bootstrap_cider_curve(
            groups,
            comparisons,
            resamples=plan["bootstrap_resamples"],
            seed=plan["bootstrap_seed"],
            workers=min(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")), plan["analysis_cpus"]),
        )
        comparison_rows = [
            {"reference": reference, "candidate": compared, **asdict(paired[reference, compared])}
            for reference, compared in comparisons
        ]
        cross = paired[prompt40, candidate]
        quality_pass = cross.ci_low >= 0
        gates = {
            "complete_rows_and_provenance": True,
            "exact_25p5_and_40_mib_initial_caps": True,
            "identical_25p5_storage_at_all_boundaries": same_budget_pass,
            "minimum_30_percent_reduction_at_every_sample_and_boundary": memory_pass,
            "paired_cider_zero_margin_noninferiority_vs_prompt40": quality_pass,
            "retention_one_parity_artifact_required": True,
            "complete_label_blind_logit_probe_trace": logit_trace_pass,
        }
        fallback_limit = plan["fixed_candidate"].get("maximum_fail_closed_fraction")
        if isinstance(fallback_limit, (float, int)):
            gates["maximum_fail_closed_fraction"] = fail_closed_count / len(ids) <= float(
                fallback_limit
            )
        parity = list(root.glob("outputs/*/attempt-*/retention-one-parity.json"))
        parity_pass = len(parity) == 1
        if parity_pass:
            payload = json.loads(parity[0].read_text())
            parity_pass = (
                payload.get("token_agreement") == 1.0
                and payload.get("maximum_logit_difference") <= 1e-5
            )
        gates["retention_one_parity_artifact_required"] = parity_pass
        passed = all(gates.values())

        pd.DataFrame(rows.values()).to_parquet(output / "paired_samples.parquet", index=False)
        pd.DataFrame(aggregates).to_parquet(output / "aggregates.parquet", index=False)
        # Unlike the capacity-sweep analyzer, this validation uses scalar
        # analysis labels rather than ``(method, cap)`` curve-point tuples.
        # Keeping those strings directly also makes the Parquet schema match
        # the JSON report without an invalid two-value unpack.
        pd.DataFrame(comparison_rows).to_parquet(
            output / "quality_comparisons.parquet", index=False
        )
        pd.DataFrame(byte_records).to_parquet(output / "byte_gates.parquet", index=False)
        report.update(
            status="passed",
            validation_gate_passed=passed,
            paper_campaign_eligible_for_separate_authorization=passed,
            interpretation=(
                "fresh_development_validation_passed"
                if passed
                else "fresh_development_validation_failed_no_paper_campaign"
            ),
            candidate_analysis_label=candidate,
            gates=gates,
            aggregates=aggregates,
            quality_comparisons=comparison_rows,
            cross_budget_comparison={
                "reference": prompt40,
                "candidate": candidate,
                **asdict(cross),
            },
            minimum_observed_memory_reduction=min(
                record["reduction_vs_40_mib"] for record in byte_records
            ),
            fail_closed_count=fail_closed_count,
            fail_closed_fraction=fail_closed_count / len(ids),
        )
    except Exception as error:
        report.update(
            status="failed",
            validation_gate_passed=False,
            paper_campaign_eligible_for_separate_authorization=False,
            interpretation="analysis_failed_metrics_must_not_be_used",
            error=f"{type(error).__name__}: {error}",
        )
    write_once(output / "report.json", report)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    raise SystemExit(analyze(parser.parse_args()))
