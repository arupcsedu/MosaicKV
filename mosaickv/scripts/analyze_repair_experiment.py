#!/usr/bin/env python3
"""Aggregate Experiment-4 repair rows, events, metrics, and qualitative cases."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, cast

import pandas as pd  # type: ignore[import-untyped]

PRIMARY_VARIANTS = ("N", "E", "P", "C", "O")
QUALITATIVE_VARIANTS = (*PRIMARY_VARIANTS, "TC")
MODALITY_LABEL = {
    "mmstar": "ordinary image",
    "docvqa": "OCR/document",
    "chartqa": "chart",
    "videomme": "video",
}


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _trace(trace_root: Path, run_id: str, sample_id: str) -> tuple[Path, dict[str, Any]]:
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted((trace_root / run_id).glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("sample_id") == sample_id and value.get("status") == "completed":
            matches.append((path, cast("dict[str, Any]", value)))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one completed trace for {run_id}/{sample_id}, found {len(matches)}"
        )
    return matches[0]


def _score(value: object) -> float:
    result = float(cast("float | int", value))
    if not math.isfinite(result):
        raise ValueError("task scores must be finite")
    return result


def _event_latency_ms(event: dict[str, Any]) -> float:
    return float(event["transfer_time_ms"]) + float(event["re_decode_time_ms"])


def _answer_flags(
    *,
    triggered: bool,
    full_answer: str,
    full_score: float,
    no_answer: str,
    no_score: float,
    answer: str,
    score: float,
) -> dict[str, bool]:
    answer_changed = answer != no_answer
    recovered = triggered and no_answer != full_answer and answer == full_answer
    improved = triggered and score > no_score
    incorrect = triggered and answer_changed and answer != full_answer and score <= no_score
    no_effect = (not triggered) or (not answer_changed and score == no_score)
    return {
        "answer_changed": answer_changed,
        "recovered_fullkv_answer": recovered,
        "benchmark_score_improved": improved,
        "answer_changed_incorrectly": incorrect,
        "no_effect": no_effect,
        "beneficial": recovered or improved,
        "fullkv_score_changed": score != full_score,
    }


def _event_row(
    sample: dict[str, Any],
    event: dict[str, Any],
    event_index: int,
) -> dict[str, Any]:
    clusters = cast("list[dict[str, Any]]", event.get("triggered_prototype_clusters", []))
    blocks = cast("list[dict[str, Any]]", event.get("restored_blocks", []))
    return {
        "task": sample["task"],
        "sample_id": sample["sample_id"],
        "variant_id": sample["variant_id"],
        "variant": sample["variant"],
        "policy": sample["policy"],
        "event_index": event_index,
        "trigger_step": int(event["trigger_step"]),
        "triggered": bool(event["triggered"]),
        "trigger_reason": str(event["trigger_reason"]),
        "normalized_entropy": float(event["normalized_entropy"]),
        "maximum_prototype_risk": float(event["maximum_prototype_risk"]),
        "triggered_cluster_ids_json": json.dumps(
            [cluster["prototype_id"] for cluster in clusters], separators=(",", ":")
        ),
        "triggered_clusters_json": json.dumps(clusters, sort_keys=True, separators=(",", ":")),
        "restored_block_ids_json": json.dumps(event["restored_block_ids"], separators=(",", ":")),
        "restored_modalities_json": json.dumps(
            [block["modality"] for block in blocks], separators=(",", ":")
        ),
        "restored_positions_json": json.dumps(
            [block["original_logical_positions"] for block in blocks],
            separators=(",", ":"),
        ),
        "restored_blocks_json": json.dumps(blocks, sort_keys=True, separators=(",", ":")),
        "restored_bytes": int(event["restored_bytes"]),
        "transfer_time_ms": float(event["transfer_time_ms"]),
        "re_decode_time_ms": float(event["re_decode_time_ms"]),
        "repair_latency_ms": _event_latency_ms(event),
        "provisional_token_id": int(event["provisional_token_id"]),
        "final_token_id": int(event["final_token_id"]),
        "token_changed": bool(event["token_changed"]),
        "maximum_logit_change": float(event["maximum_logit_change"]),
        "active_cost_before": int(event["active_cost_before"]),
        "active_cost_after": int(event["active_cost_after"]),
        "active_budget_value": int(event["active_budget_value"]),
        "fullkv_answer": sample["fullkv_answer"],
        "no_repair_answer": sample["no_repair_answer"],
        "final_answer": sample["answer"],
        "beneficial": sample["beneficial"],
    }


def aggregate(index: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    jobs = _rows(index)
    matrix_manifest = json.loads((index.parent / "manifest.json").read_text(encoding="utf-8"))
    output_root = Path(matrix_manifest["output_root"])
    trace_root = output_root / "traces"
    samples: list[dict[str, Any]] = []
    event_records: list[dict[str, Any]] = []
    failures: list[str] = []
    source_provenance: set[tuple[str, bool, str]] = set()
    model_provenance: set[tuple[str, str, str]] = set()
    software_provenance: set[str] = set()
    manifest_paths: set[Path] = set()
    for job in jobs:
        task_rows: dict[str, dict[str, dict[str, Any]]] = {}
        task_traces: dict[tuple[str, str], tuple[Path, dict[str, Any]]] = {}
        task_manifests: dict[str, tuple[Path, dict[str, Any]]] = {}
        cells = cast("list[dict[str, Any]]", job["cells"])
        for cell in cells:
            run_directory = output_root / str(cell["run_id"])
            result_path = run_directory / "results.jsonl"
            manifest_path = run_directory / "manifest.json"
            if not result_path.exists():
                failures.append(f"missing results: {result_path}")
                continue
            if not manifest_path.exists():
                failures.append(f"missing manifest: {manifest_path}")
                continue
            manifest = cast("dict[str, Any]", json.loads(manifest_path.read_text(encoding="utf-8")))
            manifest_paths.add(manifest_path.resolve())
            source = cast("dict[str, Any]", manifest["source"])
            model = cast("dict[str, Any]", manifest["model"])
            software = cast("dict[str, Any]", manifest["software"])
            dataset = cast("dict[str, Any]", manifest["dataset"])
            if source["config_sha"] != cell["config_sha"]:
                failures.append(f"config SHA mismatch: {cell['run_id']}")
            if dataset["revision"] != job["dataset_revision"]:
                failures.append(f"dataset revision mismatch: {cell['run_id']}")
            source_provenance.add(
                (
                    str(source["git_sha"]),
                    bool(source["git_dirty"]),
                    str(source["patch_sha"]),
                )
            )
            model_provenance.add(
                (str(model["id"]), str(model["revision"]), str(model["precision"]))
            )
            software_provenance.add(json.dumps(software, sort_keys=True, separators=(",", ":")))
            task_manifests[str(cell["variant_id"])] = (manifest_path, manifest)
            rows = _rows(result_path)
            by_sample: dict[str, dict[str, Any]] = {}
            for row in rows:
                sample_id = str(row["sample_id"])
                if sample_id in by_sample:
                    failures.append(f"duplicate result: {cell['run_id']}/{sample_id}")
                by_sample[sample_id] = row
                if row.get("status") != "completed" or row.get("error") is not None:
                    failures.append(f"failed result: {cell['run_id']}/{sample_id}")
                task_traces[(str(cell["variant_id"]), sample_id)] = _trace(
                    trace_root, str(cell["run_id"]), sample_id
                )
            task_rows[str(cell["variant_id"])] = by_sample
        expected_ids = tuple(str(item) for item in job["sample_ids"])
        if set(task_rows) != {str(cell["variant_id"]) for cell in cells}:
            continue
        reference = task_rows["F"]
        no_repair = task_rows["N"]
        for sample_id in expected_ids:
            if any(sample_id not in variant_rows for variant_rows in task_rows.values()):
                failures.append(f"missing paired sample: {job['task_key']}/{sample_id}")
                continue
            full_row = reference[sample_id]
            no_row = no_repair[sample_id]
            full_answer = str(full_row["answer"])
            full_score = _score(full_row["task_score"])
            no_answer = str(no_row["answer"])
            no_score = _score(no_row["task_score"])
            compressed_budgets: set[int] = set()
            for cell in cells:
                variant_id = str(cell["variant_id"])
                row = task_rows[variant_id][sample_id]
                trace_path, trace = task_traces[(variant_id, sample_id)]
                manifest_path, manifest = task_manifests[variant_id]
                source = cast("dict[str, Any]", manifest["source"])
                model = cast("dict[str, Any]", manifest["model"])
                hardware = cast("dict[str, Any]", manifest["hardware"])
                events = cast("list[dict[str, Any]]", trace.get("repair_events", []))
                triggered_events = [event for event in events if bool(event["triggered"])]
                if variant_id != "F":
                    compressed_budgets.add(int(row["active_kv_bytes"]))
                    residual = cast("dict[str, Any] | None", trace.get("residual_configuration"))
                    if residual is None:
                        failures.append(
                            f"missing residual configuration: {cell['run_id']}/{sample_id}"
                        )
                    else:
                        actual = cast("list[str]", residual["actual_placements"])
                        if actual != [cell["placement"]]:
                            failures.append(
                                f"placement mismatch: {cell['run_id']}/{sample_id}: {actual}"
                            )
                        if residual["storage_dtype"] != cell["storage_dtype"]:
                            failures.append(f"dtype mismatch: {cell['run_id']}/{sample_id}")
                answer = str(row["answer"])
                score = _score(row["task_score"])
                flags = _answer_flags(
                    triggered=bool(triggered_events),
                    full_answer=full_answer,
                    full_score=full_score,
                    no_answer=no_answer,
                    no_score=no_score,
                    answer=answer,
                    score=score,
                )
                repair_latency_ms = sum(_event_latency_ms(event) for event in triggered_events)
                sample = {
                    "task": str(job["task_key"]),
                    "task_name": str(job["task"]),
                    "modality_group": MODALITY_LABEL[str(job["task_key"])],
                    "sample_id": sample_id,
                    "variant_id": variant_id,
                    "variant": str(cell["variant"]),
                    "policy": str(cell["policy"]),
                    "entropy_threshold": float(cell["entropy_threshold"]),
                    "prototype_risk_threshold": float(cell["prototype_risk_threshold"]),
                    "prototype_risk_min_entropy": float(
                        cell.get("prototype_risk_min_entropy", 0.0)
                    ),
                    "max_blocks_per_step": int(cell["max_blocks_per_step"]),
                    "candidate_ranking": str(cell.get("candidate_ranking", "prototype_weight")),
                    "preserve_partial_prototype": bool(
                        cell.get("preserve_partial_prototype", False)
                    ),
                    "storage_dtype": str(cell["storage_dtype"]),
                    "placement": str(cell["placement"]),
                    "dataset_revision": str(job["dataset_revision"]),
                    "config_sha": str(cell["config_sha"]),
                    "run_id": str(cell["run_id"]),
                    "manifest_path": str(manifest_path.resolve()),
                    "trace_path": str(trace_path.resolve()),
                    "git_sha": str(source["git_sha"]),
                    "git_dirty": bool(source["git_dirty"]),
                    "git_patch_sha": str(source["patch_sha"]),
                    "model_id": str(model["id"]),
                    "model_revision": str(model["revision"]),
                    "model_precision": str(model["precision"]),
                    "gpu_type": str(hardware["gpu_type"]),
                    "gpu_count": int(hardware["gpu_count"]),
                    "status": str(row["status"]),
                    "answer": answer,
                    "reference": str(row["reference"]),
                    "task_score": score,
                    "fullkv_answer": full_answer,
                    "fullkv_score": full_score,
                    "no_repair_answer": no_answer,
                    "no_repair_score": no_score,
                    "compressed_differs_from_fullkv": no_answer != full_answer,
                    "score_delta_from_no_repair": score - no_score,
                    "score_delta_from_fullkv": score - full_score,
                    "repair_triggered": bool(triggered_events),
                    "triggered_event_count": len(triggered_events),
                    "restored_bytes": sum(
                        int(event["restored_bytes"]) for event in triggered_events
                    ),
                    "repair_latency_ms": repair_latency_ms,
                    "active_kv_bytes": int(row["active_kv_bytes"]),
                    "residual_kv_bytes": int(row["residual_kv_bytes"]),
                    "end_to_end_time": float(row["end_to_end_time"]),
                    "latency_delta_from_no_repair": (
                        float(row["end_to_end_time"]) - float(no_row["end_to_end_time"])
                    ),
                    **flags,
                }
                samples.append(sample)
                event_records.extend(
                    _event_row(sample, event, event_index)
                    for event_index, event in enumerate(events)
                )
            if len(compressed_budgets) != 1:
                failures.append(
                    f"active-byte budgets differ for {job['task_key']}/{sample_id}: "
                    f"{sorted(compressed_budgets)}"
                )
    if len(source_provenance) != 1:
        failures.append(f"source provenance differs across cells: {sorted(source_provenance)}")
    if len(model_provenance) != 1:
        failures.append(f"model provenance differs across cells: {sorted(model_provenance)}")
    if len(software_provenance) != 1:
        failures.append("software provenance differs across cells")
    if failures:
        raise RuntimeError("repair aggregation failed:\n" + "\n".join(failures))
    sample_frame = pd.DataFrame(samples)
    event_frame = pd.DataFrame(event_records)
    oracle = sample_frame[sample_frame["variant_id"] == "O"].set_index(["task", "sample_id"])
    no_rows = sample_frame[sample_frame["variant_id"] == "N"].set_index(["task", "sample_id"])
    repairable: set[tuple[str, str]] = set()
    for identity, no_row in no_rows.iterrows():
        oracle_row = oracle.loc[identity]
        if bool(no_row["compressed_differs_from_fullkv"]) and (
            oracle_row["answer"] == no_row["fullkv_answer"]
            or float(oracle_row["task_score"]) > float(no_row["task_score"])
        ):
            repairable.add(cast("tuple[str, str]", identity))
    summaries: list[dict[str, Any]] = []
    for variant_id, group in sample_frame.groupby("variant_id", sort=False):
        triggered = group[group["repair_triggered"]]
        triggered_events = event_frame[
            (event_frame["variant_id"] == variant_id) & event_frame["triggered"]
        ]
        repaired_failures = 0
        for row in group.itertuples(index=False):
            identity = (str(row.task), str(row.sample_id))
            if identity in repairable and (
                row.answer == row.fullkv_answer
                or float(row.task_score) > float(row.no_repair_score)
            ):
                repaired_failures += 1
        beneficial_repairs = int(triggered["beneficial"].sum())
        summaries.append(
            {
                "variant_id": variant_id,
                "variant": str(group.iloc[0]["variant"]),
                "policy": str(group.iloc[0]["policy"]),
                "sample_count": len(group),
                "compressed_failure_count": int(group["compressed_differs_from_fullkv"].sum()),
                "triggered_repair_count": len(triggered),
                "triggered_event_count": len(triggered_events),
                "beneficial_repair_count": beneficial_repairs,
                "repair_precision": (
                    beneficial_repairs / len(triggered) if len(triggered) else math.nan
                ),
                "repairable_failure_count": len(repairable),
                "repaired_failure_count": repaired_failures,
                "repair_recall": repaired_failures / len(repairable) if repairable else math.nan,
                "average_repair_latency_ms": (
                    float(triggered_events["repair_latency_ms"].mean())
                    if len(triggered_events)
                    else math.nan
                ),
                "p95_repair_latency_ms": (
                    float(triggered_events["repair_latency_ms"].quantile(0.95))
                    if len(triggered_events)
                    else math.nan
                ),
                "average_restored_bytes": (
                    float(triggered_events["restored_bytes"].mean())
                    if len(triggered_events)
                    else math.nan
                ),
                "average_residual_storage_bytes": float(group["residual_kv_bytes"].mean()),
                "net_quality_gain": float(group["score_delta_from_no_repair"].mean()),
                "net_end_to_end_latency_cost_seconds": float(
                    group["latency_delta_from_no_repair"].mean()
                ),
                "recovered_fullkv_answer_count": int(group["recovered_fullkv_answer"].sum()),
                "benchmark_score_improved_count": int(group["benchmark_score_improved"].sum()),
                "incorrect_answer_change_count": int(group["answer_changed_incorrectly"].sum()),
                "no_effect_count": int(group["no_effect"].sum()),
            }
        )
    metadata = {
        "classification": matrix_manifest["classification"],
        "index_sha": matrix_manifest["index_sha"],
        "source_provenance": {
            "git_sha": next(iter(source_provenance))[0],
            "git_dirty": next(iter(source_provenance))[1],
            "patch_sha": next(iter(source_provenance))[2],
        },
        "model_provenance": {
            "model_id": next(iter(model_provenance))[0],
            "model_revision": next(iter(model_provenance))[1],
            "precision": next(iter(model_provenance))[2],
        },
        "software": json.loads(next(iter(software_provenance))),
        "manifest_count": len(manifest_paths),
        "repairable_failures": sorted([list(item) for item in repairable]),
        "output_root": str(output_root.resolve()),
    }
    return sample_frame, event_frame, pd.DataFrame(summaries), metadata


def _fmt(value: object, digits: int = 4) -> str:
    number = float(cast("float | int", value))
    return "N/A" if not math.isfinite(number) else f"{number:.{digits}f}"


def _markdown_report(
    samples: pd.DataFrame,
    events: pd.DataFrame,
    summary: pd.DataFrame,
    metadata: dict[str, Any],
) -> str:
    primary = summary[summary["variant_id"].isin(PRIMARY_VARIANTS)]
    lines = [
        "# Experiment 4: residual repair",
        "",
        "> Classification: exploratory small-sample validation. The run used a dirty worktree "
        "with preserved config/trace provenance and is not a canonical paper result.",
        "",
        "## Design and completeness",
        "",
        f"The study contains {len(samples)} paired sample-policy rows and "
        f"{len(events)} decode-step "
        "records over seven fixed examples (MMStar 2, DocVQA 2, ChartQA 2, Video-MME 1). "
        "Qwen2.5-VL-3B, BF16, HF eager, deterministic greedy decoding, eight output tokens, "
        "retention 0.2, block size 16, and the physical packed-prefill KV-byte budget are fixed.",
        "",
        f"Immutable matrix index SHA: `{metadata['index_sha']}`. All compressed cells for a paired "
        "sample were required to have exactly the same active KV byte count. Requested residual "
        "dtype and pinned/pageable placement were checked against the runtime trace.",
        "",
        "Thresholds are an exploratory log-scale sensitivity sweep, not fitted benchmark "
        "hyperparameters. A canonical study must choose thresholds on a disjoint calibration set.",
        "",
        f"All {metadata['manifest_count']} run manifests agree on Git "
        f"`{metadata['source_provenance']['git_sha']}` with "
        f"dirty patch SHA `{metadata['source_provenance']['patch_sha']}` and model revision "
        f"`{metadata['model_provenance']['model_revision']}`. The exploratory classification "
        "is retained explicitly; these rows must not be presented as clean-tree paper evidence.",
        "",
        "## Primary policy comparison",
        "",
        (
            "| Variant | Triggered samples | Precision | Recall | Repair ms avg / p95 | "
            "Restored bytes avg | Quality gain | E2E cost (s) | Recovered / improved / "
            "incorrect / no effect |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in primary.itertuples(index=False):
        lines.append(
            f"| {row.variant} | {row.triggered_repair_count}/{row.sample_count} | "
            f"{_fmt(row.repair_precision)} | {_fmt(row.repair_recall)} | "
            f"{_fmt(row.average_repair_latency_ms, 3)} / {_fmt(row.p95_repair_latency_ms, 3)} | "
            f"{_fmt(row.average_restored_bytes, 1)} | {_fmt(row.net_quality_gain)} | "
            f"{_fmt(row.net_end_to_end_latency_cost_seconds, 3)} | "
            f"{row.recovered_fullkv_answer_count} / {row.benchmark_score_improved_count} / "
            f"{row.incorrect_answer_change_count} / {row.no_effect_count} |"
        )
    lines.extend(
        [
            "",
            "Precision uses beneficial triggered sample-policy cells divided by triggered "
            "sample-policy cells. A benefit is either recovery of the paired FullKV answer or a "
            "strict benchmark-score improvement. Recall uses failures that the oracle repaired as "
            "the repairable denominator.",
            "",
            "## Sensitivity sweep",
            "",
            (
                "| Variant | Triggered | Precision | Recall | Avg repair ms | Restored bytes | "
                "Residual MiB | Quality gain | E2E cost (s) | Improved / incorrect / no effect |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary[~summary["variant_id"].isin({"F", "N", "E", "P", "O"})].itertuples(
        index=False
    ):
        lines.append(
            f"| {row.variant} | {row.triggered_repair_count} | {_fmt(row.repair_precision)} | "
            f"{_fmt(row.repair_recall)} | {_fmt(row.average_repair_latency_ms, 3)} | "
            f"{_fmt(row.average_restored_bytes, 1)} | "
            f"{_fmt(row.average_residual_storage_bytes / (1024 * 1024), 2)} | "
            f"{_fmt(row.net_quality_gain)} | "
            f"{_fmt(row.net_end_to_end_latency_cost_seconds, 3)} | "
            f"{row.benchmark_score_improved_count} / {row.incorrect_answer_change_count} / "
            f"{row.no_effect_count} |"
        )
    failure_count = int(
        samples[samples["variant_id"] == "N"]["compressed_differs_from_fullkv"].sum()
    )
    repairable_count = len(cast("list[list[str]]", metadata["repairable_failures"]))
    lines.extend(
        [
            "",
            "## Interpretation boundaries",
            "",
            f"The no-repair cache differed from FullKV on {failure_count}/7 answer strings; "
            f"{repairable_count} were repairable by the evaluation-only oracle under this "
            "one-redecode, fixed-budget mechanism. Results are reported whether favorable, "
            "neutral, or harmful. With seven examples, percentages are diagnostics rather than "
            "population estimates.",
            "",
            "Repair latency is measured as synchronized residual transfer plus synchronized "
            "re-decode time. Net end-to-end cost is paired against the no-repair row and includes "
            "all online work; the oracle's independent FullKV diagnostic trajectory is excluded "
            "from online latency by construction.",
            "",
            "Restored bytes are the promoted source-format KV bytes that re-enter the active "
            "GPU cache. Residual storage bytes, which vary for FP8/INT8 versus lossless storage, "
            "remain separately recorded in the sample table and runtime trace.",
            "",
        ]
    )
    return "\n".join(lines)


def _qualitative(samples: pd.DataFrame, events: pd.DataFrame) -> str:
    lines = [
        "# Residual-repair qualitative appendix",
        "",
        "This appendix includes every fixed sample in the small study, not a success-only "
        "selection. Thus it contains ordinary-image, OCR/document, chart, and video cases, plus "
        "both repairable and non-repairable failures. The conservative-threshold sensitivity "
        "row is included because it contains both the observed benchmark improvement and the "
        "observed harmful answer change.",
        "",
    ]
    primary = samples[samples["variant_id"].isin(QUALITATIVE_VARIANTS)]
    for (task, sample_id), group in primary.groupby(["task", "sample_id"], sort=False):
        no_row = group[group["variant_id"] == "N"].iloc[0]
        lines.extend(
            [
                f"## {MODALITY_LABEL[str(task)].title()}: `{sample_id}`",
                "",
                f"Reference: `{no_row['reference']}`. FullKV: `{no_row['fullkv_answer']}` "
                f"(score {_fmt(no_row['fullkv_score'])}). No repair: "
                f"`{no_row['no_repair_answer']}` (score {_fmt(no_row['no_repair_score'])}). "
                f"Compressed differs from FullKV: "
                f"`{str(bool(no_row['compressed_differs_from_fullkv'])).lower()}`.",
                "",
                (
                    "| Policy | Triggered | Final answer | Score | Outcome | Restored clusters / "
                    "modality / positions / bytes | Token/logit change |"
                ),
                "|---|---:|---|---:|---|---|---|",
            ]
        )
        for row in group.itertuples(index=False):
            policy_events = events[
                (events["task"] == task)
                & (events["sample_id"] == sample_id)
                & (events["variant_id"] == row.variant_id)
                & events["triggered"]
            ]
            if row.recovered_fullkv_answer:
                outcome = "recovered FullKV answer"
            elif row.benchmark_score_improved:
                outcome = "improved benchmark score"
            elif row.answer_changed_incorrectly:
                outcome = "changed answer incorrectly"
            elif row.no_effect:
                outcome = "no effect"
            else:
                outcome = "changed answer, score unchanged"
            details: list[str] = []
            changes: list[str] = []
            for event in policy_events.itertuples(index=False):
                clusters = json.loads(event.triggered_cluster_ids_json)
                modalities = json.loads(event.restored_modalities_json)
                positions = json.loads(event.restored_positions_json)
                details.append(
                    f"step {event.trigger_step}: clusters {clusters}, {modalities}, "
                    f"positions {positions}, {event.restored_bytes} B"
                )
                changes.append(
                    f"{event.provisional_token_id}->{event.final_token_id}; "
                    f"max |Δlogit|={_fmt(event.maximum_logit_change)}"
                )
            lines.append(
                f"| {row.variant} | {str(bool(row.repair_triggered)).lower()} | "
                f"`{row.answer}` | {_fmt(row.task_score)} | {outcome} | "
                f"{'<br>'.join(details) if details else 'none'} | "
                f"{'<br>'.join(changes) if changes else 'none'} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument(
        "--result-directory", type=Path, default=project_root / "results" / "repair"
    )
    parser.add_argument(
        "--analysis", type=Path, default=project_root / "analysis" / "residual_repair.md"
    )
    parser.add_argument(
        "--qualitative",
        type=Path,
        default=project_root / "analysis" / "residual_repair_qualitative.md",
    )
    args = parser.parse_args()
    samples, events, summary, metadata = aggregate(args.index.resolve())
    result_directory = args.result_directory.resolve()
    result_directory.mkdir(parents=True, exist_ok=True)
    samples.to_json(result_directory / "raw.jsonl", orient="records", lines=True)
    samples.to_parquet(result_directory / "sample_outcomes.parquet", index=False)
    events.to_parquet(result_directory / "repair_events.parquet", index=False)
    summary.to_parquet(result_directory / "summary.parquet", index=False)
    (result_directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.analysis.parent.mkdir(parents=True, exist_ok=True)
    args.analysis.write_text(_markdown_report(samples, events, summary, metadata), encoding="utf-8")
    args.qualitative.write_text(_qualitative(samples, events), encoding="utf-8")
    print(
        json.dumps(
            {
                "events": len(events),
                "repairable_failures": len(metadata["repairable_failures"]),
                "sample_rows": len(samples),
                "summary_rows": len(summary),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
