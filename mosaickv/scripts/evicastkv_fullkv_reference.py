"""Recover completed FullKV predictions after a compressed-only postcheck failed.

No worker/gate is relabeled, no inference is repeated, and no latency result is
recovered. Only this exact postprocessing error is admissible. The complete
registered cohort and immutable FullKV parity/trace evidence are mandatory.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from evicastkv_analyze import collect
from evicastkv_transport_scores import ARITHMETIC, comparison_controls, gated_rows, validate_rows

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once
from mosaickv.evaluation.milebench import route_milebench_dataset, score_milebench_sample

SOURCE = Path(__file__).resolve().parents[2]
POSTCHECK_ERROR = "AssertionError: missing compressed representation or forced continuation"


def recover_trace(row, trace):
    """Pure admission check; benchmark scores never affect trace acceptance."""
    if (
        row["status"] != "failed"
        or row["error"] != POSTCHECK_ERROR
        or row["method"] != "full_kv"
        or row["budget_bytes"] is not None
        or trace["status"] != "completed"
        or trace["run_id"] != row["run_id"]
        or trace["sample_id"] != row["sample_id"]
        or trace["method"] != "full_kv"
        or trace["requested_method"] != "full_kv"
        or trace["effective_method"] != "full_kv"
        or trace["retention_ratio"] != 1.0
        or trace["tier_mode"] != "full_cache"
        or len(trace["generated_token_ids"]) != 64
        or trace["decode_runtime_policy"]["planned_decode_updates"] != 63
        or len(trace["timing_breakdown"]["per_token_decode"]) != 63
    ):
        raise ValueError("only complete, identity FullKV postcheck failures may be recovered")
    if (
        trace["selected_blocks"]
        or trace["prototypes"]
        or trace["repair_events"]
        or trace["prefixkv_decode_events"]
        or trace["elastic_cache_decode_events"]
        or trace["residual_kv_bytes"] != 0
        or trace["packed_padding_kv_bytes"] != 0
        or trace["prefill_attention_collection"]["model_output_collector"]
        or trace["prefill_attention_collection"]["private_module_hooks"]
    ):
        raise ValueError("FullKV must not compress, repair, pad, or collect policy evidence")
    layout = trace["source_kv_layout"]
    initial, per_position = 0, 0
    for layer in layout:
        for name in ("key", "value"):
            shape, size = layer[f"{name}_shape"], layer[f"{name}_element_size"]
            if len(shape) != 4 or any(type(x) is not int or x <= 0 for x in shape):
                raise ValueError("invalid B,H,N,D FullKV source shape")
            initial += math.prod(shape) * size
            per_position += math.prod(shape[:2] + shape[3:]) * size
    life = trace["cache_lifecycle"]
    if (
        not layout
        or initial != trace["source_prefill_kv_bytes"]
        or initial != life["initial_active_kv_bytes"]
        or initial != life["initial_active_kv_storage_bytes"]
        or initial + 63 * per_position != life["final_active_kv_bytes"]
        or life["final_active_kv_bytes"] != life["final_active_kv_storage_bytes"]
        or life["peak_active_kv_storage_bytes"] != life["final_active_kv_storage_bytes"]
    ):
        raise ValueError("FullKV tensor/storage append accounting is inconsistent")
    quality = trace["quality_decoding"]
    if quality["timed_token_count"] != 64 or not 0 <= quality["scored_token_count"] <= 64:
        raise ValueError("incomplete quality/timing token boundary")
    return dict(
        answer=trace["answer"],
        initial_kv_bytes=life["initial_active_kv_storage_bytes"],
        final_kv_bytes=life["final_active_kv_storage_bytes"],
        peak_kv_bytes=life["peak_active_kv_storage_bytes"],
        evaluated_token_count=quality["scored_token_count"],
    )


def parity_passed(parity):
    if set(parity) != {"ordinary_generate", "retention_one", "mosaickv_runtime_retention_one"}:
        raise ValueError("all three original FullKV parity controls are required")
    for result in parity.values():
        if (
            result["generated_tokens"] < 16
            or result["token_agreement"] != 1.0
            or result["reference_token_ids"] != result["candidate_token_ids"]
            or result["maximum_logit_difference"] != 0.0
        ):
            raise ValueError("FullKV parity is not exactly satisfied")


def recover(root, scoring, output):
    import pandas as pd

    revision = clean_source(SOURCE)
    if output.exists():
        raise FileExistsError("derived reference output must be new")
    lock_path = root / "protocol_lock.json"
    lock = json.loads(lock_path.read_text())
    clean_source(Path(lock["source_root"]), lock["source_sha"])
    if lock["arithmetic"] != ARITHMETIC or lock["methods"] != {"milebench": ["full_kv"]}:
        raise ValueError("unexpected FullKV-only/common-transport registration")
    registration = json.loads((scoring / "registration.json").read_text())
    for path, expected in registration["bindings"].items():
        if sha(Path(path)) != expected:
            raise ValueError("compressed scoring evidence changed")
    gate_path = scoring / "GATE_scoring.json"
    gate = json.loads(gate_path.read_text())
    if gate["completed"] != 128 or gate["failed"] or not gate["official_scorer_fixtures_passed"]:
        raise ValueError("complete, metric-validated compressed scoring is required")
    samples = json.loads(Path(registration["input_ledger"]).read_text())["samples"]
    rows, bindings = gated_rows([Path(p) for p in registration["prediction_roots"]])
    validate_rows(rows, samples)
    by_id = {s["sample_id"]: s for s in samples}
    controls = {}
    for row in rows:
        cfg = json.loads(Path(row["resolved_config_path"]).read_text())
        control = comparison_controls(cfg)
        control.pop("cache")  # FullKV is not a compressed-method budget point.
        value = (row["input_sha"], row["processor_sha"], row["scorer_sha"], control)
        if controls.setdefault(row["sample_id"], value) != value:
            raise ValueError("compressed controls are inconsistent")
    spec, workers, failed_rows, _errors = collect(root, "correctness")
    if len(workers) != len(spec["jobs"]) or len(failed_rows) != 32:
        raise ValueError("complete original worker denominator is required")
    ids = [r["sample_id"] for r in failed_rows]
    if len(set(ids)) != 32 or set(ids) != set(by_id):
        raise ValueError("cannot recover a surviving intersection or duplicate identities")
    bindings.update({str(lock_path): sha(lock_path), str(gate_path): sha(gate_path)})
    observations = []
    for row in failed_rows:
        attempt = Path(row["manifest_path"]).parent
        bindings[str(attempt / "worker.json")] = sha(attempt / "worker.json")
        parity_path = attempt / "retention_one.json"
        parity_passed(json.loads(parity_path.read_text()))
        config_path = attempt / "resolved-configs" / f"{row['run_id']}.json"
        cfg = json.loads(config_path.read_text())
        if row["config_sha"] != digest(dict(runtime_config=cfg, arithmetic=ARITHMETIC)):
            raise ValueError("FullKV resolved configuration differs from its raw binding")
        control = comparison_controls(cfg)
        control.pop("cache")
        if (row["input_sha"], row["processor_sha"], row["scorer_sha"], control) != controls[
            row["sample_id"]
        ]:
            raise ValueError("FullKV and compressed model/input/generation controls differ")
        manifest = json.loads(Path(row["manifest_path"]).read_text())
        if manifest["arithmetic_policy"] != ARITHMETIC:
            raise ValueError("FullKV used a different transport policy")
        traces = list((attempt / "traces" / row["run_id"]).rglob("*.json"))
        if len(traces) != 1:
            raise ValueError("exactly one completed trace per failed postcheck is required")
        values = recover_trace(row, json.loads(traces[0].read_text()))
        sample = by_id[row["sample_id"]]
        route = route_milebench_dataset(
            sample["milebench_dataset"], sample["milebench_question_type"]
        )
        if len(sample["references"]) != 1:
            raise ValueError("this route requires exactly one reference")
        score = score_milebench_sample(
            route,
            values["answer"],
            sample["references"][0],
            choices=tuple(sample.get("choices", [])),
        )
        observations.append(
            dict(
                **values,
                run_id=output.name,
                source_run_id=row["run_id"],
                sample_id=row["sample_id"],
                task=sample["milebench_dataset"],
                method="full_kv",
                metric_name=route.metric.value,
                metric_value=score,
                metric_direction="higher",
                metric_scope="sample",
                metric_implementation="pinned_LOOK-M_compatible_MileBench",
                metric_version=sha(SOURCE / "mosaickv/src/mosaickv/evaluation/milebench.py"),
                evaluator_commit=revision,
                reference_artifact_sha=registration["bindings"][registration["input_ledger"]],
                manifest_path=row["manifest_path"],
                source_sha=row["source_sha"],
                config_sha=row["config_sha"],
                original_worker_status="failed_postprocessing",
                original_worker_error=POSTCHECK_ERROR,
                measurement_type="derived_recovered_fullkv_prediction_not_latency",
            )
        )
        for path in (parity_path, config_path, traces[0], Path(row["manifest_path"])):
            bindings[str(path)] = sha(path)
    output.mkdir(parents=True, exist_ok=False)
    frame = pd.DataFrame(observations)
    frame.to_parquet(output / "fullkv_metric_observations.parquet", index=False)
    aggregates = frame.groupby(["task", "metric_name"], as_index=False).agg(
        metric_value=("metric_value", "mean"),
        scored=("sample_id", "count"),
        initial_kv_bytes=("initial_kv_bytes", "mean"),
        final_kv_bytes=("final_kv_bytes", "mean"),
        peak_kv_bytes=("peak_kv_bytes", "mean"),
    )
    aggregates.to_parquet(output / "fullkv_subtasks.parquet", index=False)
    aggregates.to_csv(output / "fullkv_subtasks.csv", index=False)
    write_once(
        output / "GATE_reference.json",
        dict(
            status="passed_trace_recovery_consumed_development_only",
            source_sha=revision,
            recovered_predictions=32,
            attempted=32,
            unrecovered=0,
            new_GPU_inference=0,
            original_worker_gate="failed_preserved_not_reclassified",
            recovery_reason=POSTCHECK_ERROR,
            family_scores=aggregates.groupby("metric_name").metric_value.mean().to_dict(),
            metric_families_kept_separate=True,
            latency_claim_allowed=False,
            fresh_confirmation=False,
            superiority_established=False,
            bindings=bindings,
            recovery_script_sha256=sha(Path(__file__)),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scoring", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    recover(args.root, args.scoring, args.output)
