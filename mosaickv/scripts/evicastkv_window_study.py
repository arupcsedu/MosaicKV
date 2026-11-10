#!/usr/bin/env python3
"""Preregistered, consumed-development prompt-window sensitivity; no submission.

Each sample executes FullKV once plus Top-K/EviCastKV at windows 8/16/32/64.
This is ordinary policy sensitivity, NOT an isolated fixed-candidate ablation.
No winning configuration is frozen or promoted by the analysis.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from evicastkv_campaign import SOURCE, load_lock
from evicastkv_development import runtime_files

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

STAGE = "window_sensitivity"
WINDOWS = (8, 16, 32, 64)
METHODS = ("prompt_attention_topk", "mosaickv_exact")
CAP = 40 * 2**20
FAMILY_SIZE = 7


def validate_window(window):
    if type(window) is not int or window not in WINDOWS:
        raise ValueError("registered integer prompt window must be one of 8/16/32/64")
    return window


def arms():
    return [["full_kv", None, "complete"]] + [
        [method, 40, f"window-{window}"] for window in WINDOWS for method in METHODS
    ]


def jobs():
    return [
        dict(
            workload="coco",
            subset="development",
            stage=STAGE,
            indices=list(range(start, start + 4)),
            arms=arms(),
        )
        for start in range(0, 64, 4)
    ]


def window_config(config, method, window):
    """Only prompt window and frozen Evi route fraction differ from parent."""
    validate_window(window)
    if method not in METHODS or config.cache.budget_value != CAP:
        raise ValueError("registered compressed method and exact 40 MiB cap required")
    updated = replace(config, forecasting=replace(config.forecasting, prompt_window=window))
    if method == "mosaickv_exact":
        updated = replace(
            updated,
            graph=replace(updated.graph, uncertain_head_fraction=1 / 16),
            prototypes=replace(updated.prototypes, uncertain_head_fraction=1 / 16),
        )
    return updated


@contextlib.contextmanager
def config_overlay():
    """Patch only the evaluation worker's resolved-config call, not model math."""
    import evicastkv_worker as worker

    original = worker.resolved_config

    def resolved(lock, workload, method, cap, *, variant="complete"):
        if variant.startswith("window-"):
            suffix = variant.removeprefix("window-")
            if suffix not in {str(window) for window in WINDOWS} or workload != "coco" or cap != 40:
                raise ValueError("unregistered window/cap/workload requested")
            config = original(lock, workload, method, cap, variant="complete")
            return window_config(config, method, int(suffix))
        return original(lock, workload, method, cap, variant=variant)

    with patch.object(worker, "resolved_config", resolved):
        yield


def comparisons():
    paired = [
        (f"prompt_attention_topk:w{window}", f"mosaickv_exact:w{window}") for window in WINDOWS
    ]
    paired += [("mosaickv_exact:w16", f"mosaickv_exact:w{w}") for w in WINDOWS if w != 16]
    return paired


def prepare(root, parent):
    revision = clean_source(SOURCE)
    if root.exists():
        raise FileExistsError("window sensitivity root must be new")
    parent_path = parent / "protocol_lock.json"
    lock = json.loads(parent_path.read_text())
    prior = Path(lock["source_root"])
    clean_source(prior, lock["source_sha"])
    if runtime_files(SOURCE) != runtime_files(prior):
        raise ValueError("production runtime differs from measured controls")
    for name in (
        "mosaickv/scripts/evicastkv_worker.py",
        "mosaickv/scripts/evicastkv_campaign.py",
        "mosaickv/configs/experiments/evicastkv_fox/coco/full_kv.json",
        "mosaickv/configs/experiments/evicastkv_fox/coco/prompt_attention_topk.json",
        "mosaickv/configs/experiments/evicastkv_fox/coco/mosaickv_exact.json",
    ):
        if sha(SOURCE / name) != sha(prior / name):
            raise ValueError("shared worker/configuration resolver changed")
    gate_bindings = {}
    for stage in ("correctness", "scoring", "profiling", "measurement"):
        path = parent / "analysis" / stage / f"GATE_{stage}.json"
        gate = json.loads(path.read_text())
        if (
            gate.get("status") != "passed"
            or gate.get("blockers")
            or gate.get("source_sha") != lock["source_sha"]
            or gate.get("protocol_sha256") != sha(parent_path)
        ):
            raise ValueError(f"parent gate not passed: {stage}")
        gate_bindings[str(path)] = sha(path)
    cohort = lock["cohorts"]["coco"]["development"]
    ledger_path = Path(cohort["path"])
    if sha(ledger_path) != cohort["sha256"]:
        raise ValueError("consumed development ledger changed")
    samples = json.loads(ledger_path.read_text())["samples"]
    ids = [s["sample_id"] for s in samples]
    if len(ids) != 64 or len(set(ids)) != 64 or ids != cohort["sample_ids"]:
        raise ValueError("complete ordered 64-input consumed cohort required")
    lock = copy.deepcopy(lock)
    lock.update(
        name="evicastkv_prompt_window_development_v1", source_root=str(SOURCE), source_sha=revision
    )
    lock["resources"].update(
        maximum_a100s=6,
        array_concurrency=6,
        six_a100s_explicitly_authorized=True,
        cpus_per_worker=12,
        wall_minutes=120,
    )
    lock["authorization"].update(automatic_stage_progression=False, confirmation_authorized=False)
    lock.setdefault("inherited_gates", {}).update(gate_bindings)
    lock["historical_bindings"][str(parent_path)] = sha(parent_path)
    lock["window_study"] = dict(
        windows=list(WINDOWS),
        methods=list(METHODS),
        sample_ids=ids,
        fixed_cap_bytes=CAP,
        fixed_evicast_routing_fraction=1 / 16,
        expected_rows=576,
        exploratory_family_size=FAMILY_SIZE,
        bootstrap_resamples=10000,
        contrasts=comparisons(),
        scope="ordinary policy sensitivity; candidates may change with prompt window",
        fresh_inputs=False,
        label_based_configuration_selection=False,
        winner_freezing_authorized=False,
        headline_latency_eligible=False,
        admission="after mechanisms/full-graph chains and graph CPU diagnostic finish",
    )
    write_once(root / "protocol_lock.json", lock)
    write_once(
        root / "jobs" / f"{STAGE}.json",
        dict(
            protocol_sha256=sha(root / "protocol_lock.json"),
            jobs=jobs(),
        ),
    )
    load_lock(root)
    return dict(root=str(root), workers=16, spec_sha256=sha(root / "jobs" / f"{STAGE}.json"))


def worker(root, index, spec_sha):
    from evicastkv_worker import run_worker

    load_lock(root)
    path = root / "jobs" / f"{STAGE}.json"
    spec = json.loads(path.read_text())
    if sha(path) != spec_sha or spec.get("jobs") != jobs() or index not in range(16):
        raise ValueError("registered complete window job grid/hash required")
    with config_overlay():
        run_worker(root, STAGE, index, spec_sha)


def validate_rows(rows, ids):
    """All windows and FullKV, no surviving intersection or duplicate reference."""
    expected = {(sid, method, variant) for sid in ids for method, _, variant in arms()}
    actual = [(r["sample_id"], r["method"], r["variant"]) for r in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("incomplete or duplicate registered window grid")
    controls = {}
    model_controls = {}
    cache_controls = {}
    references = {}
    source_lengths = {}
    storage = {}
    windows = []
    for row in rows:
        if row["status"] != "completed":
            raise ValueError("failed sensitivity sample retained; statistics blocked")
        path = Path(row["resolved_config_path"])
        if sha(path) != row["resolved_config_file_sha256"]:
            raise ValueError("resolved sensitivity config changed")
        config = json.loads(path.read_text())
        sid = row["sample_id"]
        annotations = tuple(row["references"])
        if not annotations or any(not isinstance(text, str) for text in annotations):
            raise ValueError("complete COCO reference annotation list required")
        if sid in references and references[sid] != annotations:
            raise ValueError("reference annotations differ across methods/windows")
        references[sid] = annotations
        if row.get("effective_method") != row["method"] or (
            row.get("emitted_tokens") != 64 or row.get("forward_steps") != 63
        ):
            raise ValueError("fallback or generation schedule changed")
        common_config = json.dumps(
            {key: config[key] for key in ("model", "dataset", "generation", "execution")},
            sort_keys=True,
        )
        if sid in model_controls and model_controls[sid] != common_config:
            raise ValueError("model/dataset/precision/backend/generation controls changed")
        model_controls[sid] = common_config
        identity = tuple(
            row[k] for k in ("input_sha", "processor_sha", "scorer_sha", "forward_steps")
        )
        if sid in controls and controls[sid] != identity:
            raise ValueError("matched input/processor/scorer/generation controls changed")
        controls[sid] = identity
        if row["method"] == "full_kv":
            continue
        cache_fields = (
            "block_size",
            "physical_budget_block_size",
            "physical_budget_mandatory_policy",
            "budget_unit",
            "budget_value",
            "accounting_spec_sha",
            "packing_mode",
            "exact_gather_mode",
            "total_storage_budget_bytes",
        )
        cache_policy = tuple(config["cache"][field] for field in cache_fields)
        if sid in cache_controls and cache_controls[sid] != cache_policy:
            raise ValueError("matched cache/block/mandatory policy controls changed")
        cache_controls[sid] = cache_policy
        window = int(row["variant"].removeprefix("window-"))
        validate_window(window)
        if row["budget_bytes"] != CAP or row["initial_kv_bytes"] != CAP:
            raise ValueError("exact shared 40 MiB byte matching failed")
        if config["forecasting"]["prompt_window"] != window:
            raise ValueError("requested and actual prompt window differ")
        trace_path = Path(row["trace_path"])
        if sha(trace_path) != row["trace_sha256"]:
            raise ValueError("source-length trace changed")
        trace = json.loads(trace_path.read_text())
        layout = trace["source_kv_layout"]
        if not layout or any(
            len(layer["key_shape"]) != 4 or len(layer["value_shape"]) != 4 for layer in layout
        ):
            raise ValueError("pinned B,H,N,D source cache layout required")
        lengths = {layer[k][2] for layer in layout for k in ("key_shape", "value_shape")}
        if len(lengths) != 1:
            raise ValueError("uniform original source length required")
        source_length = next(iter(lengths))
        if type(source_length) is not int or source_length < 1:
            raise ValueError("positive original source length required")
        if sid in source_lengths and source_lengths[sid] != source_length:
            raise ValueError("original source length differs across windows/methods")
        source_lengths[sid] = source_length
        expected_effective = min(window, source_length)
        actual_window = None
        if row["method"] == "mosaickv_exact":
            provenance = trace.get("forecast_statistics", {}).get("provenance", {})
            expected_provenance = {
                "actual_prompt_window": expected_effective,
                "requested_prompt_window": window,
                "original_logical_sequence_length": source_length,
            }
            if any(
                type(provenance.get(key)) is not int or provenance[key] != value
                for key, value in expected_provenance.items()
            ):
                raise ValueError("trace forecast window/request/source length mismatch")
            actual_window = provenance["actual_prompt_window"]
            window_provenance = "measured forecast_statistics.provenance, checked against source N"
        else:
            window_provenance = (
                "inferred from immutable attention collector/config and source N; not measured"
            )
        if row["method"] == "mosaickv_exact" and (
            config["graph"]["uncertain_head_fraction"] != 1 / 16
            or config["prototypes"]["uncertain_head_fraction"] != 1 / 16
        ):
            raise ValueError("registered routing fraction changed")
        measured = tuple(
            row[key] for key in ("initial_kv_bytes", "final_kv_bytes", "peak_kv_bytes")
        )
        if sid in storage and storage[sid] != measured:
            raise ValueError("realized initial/final/peak storage differs across windows/methods")
        storage[sid] = measured
        windows.append(
            dict(
                sample_id=sid,
                method=row["method"],
                actual_prompt_window=actual_window,
                resolved_prompt_window=window,
                requested_prompt_window=window,
                source_sequence_length=source_length,
                effective_prompt_window=expected_effective,
                effective_window_provenance=window_provenance,
                actual_window_measured=actual_window is not None,
                source_trace_sha256=row["trace_sha256"],
            )
        )
    return windows


def analyze(root):
    from evicastkv_analyze import collect, export_descriptive

    from mosaickv.evaluation.fox_statistics import corpus_cider_statistics
    from mosaickv.evaluation.metrics.types import MetricSample

    lock = load_lock(root)
    _, _, rows, errors = collect(root, STAGE)
    output = root / "analysis" / STAGE
    try:
        actual_windows = validate_rows(rows, lock["window_study"]["sample_ids"])
    except (ValueError, KeyError, FileNotFoundError) as error:
        errors.append(dict(error=f"{type(error).__name__}: {error}"))
        actual_windows = []
    write_once(output / "failures.json", errors)
    write_once(output / "actual_windows.json", actual_windows)
    if errors:
        write_once(
            output / "GATE_window_sensitivity.json",
            dict(
                status="blocked",
                actual_rows=len(rows),
                expected_rows=576,
                blockers=errors,
                fresh_confirmation_authorized=False,
                winner_freezing_authorized=False,
            ),
        )
        return dict(status="blocked", failed_or_missing=True)
    export_descriptive(root, STAGE, rows)
    points, groups = {}, {}
    for row in rows:
        key = (
            "full_kv"
            if row["method"] == "full_kv"
            else (row["method"] + ":w" + row["variant"].removeprefix("window-"))
        )
        points.setdefault(key, []).append(
            MetricSample(row["sample_id"], row["answer"], tuple(row["references"]))
        )
        groups[row["sample_id"]] = tuple(row["media_group_ids"])
    stats = corpus_cider_statistics(
        {k: tuple(v) for k, v in points.items()},
        comparisons(),
        groups,
        resamples=10000,
        seed=lock["seed"],
        workers=1,
        family_size=FAMILY_SIZE,
    )
    write_once(output / "corpus_cider_statistics.json", stats)
    write_once(
        output / "GATE_window_sensitivity.json",
        dict(
            status="passed_exploratory_sensitivity_not_confirmation",
            actual_rows=len(rows),
            expected_rows=576,
            failed_rows=0,
            contrasts=7,
            statistical_family_size=7,
            bootstrap_resamples=10000,
            protocol_sha256=sha(root / "protocol_lock.json"),
            source_sha=lock["source_sha"],
            all_points_reported=True,
            fullkv_once_per_input=True,
            headline_latency_eligible=False,
            winner_freezing_authorized=False,
            fresh_confirmation_authorized=False,
            superiority_claim_allowed=False,
        ),
    )
    return dict(status="complete_exploratory", rows=len(rows), contrasts=7)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "worker", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--spec-sha")
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.parent is None:
            parser.error("prepare requires --parent")
        result = prepare(args.root, args.parent)
    elif args.mode == "worker":
        if args.spec_sha is None:
            parser.error("worker requires --spec-sha")
        result = worker(args.root, args.index, args.spec_sha)
    else:
        result = analyze(args.root)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
